#This is the arguement format: python script.py -i path/to/file.yaml
import os
import subprocess
from pathlib import Path
from argparse import ArgumentParser
import io_utils as io
from candidate import CandidateListGenerator
from launcher import launch_TCcalculation, launch_TMcalculation
from molecule import Molecule

def read_single_arguments():
    description_string = "This script will launch AutoPilot on a geometry"
    parser = ArgumentParser(description=description_string)
    parser.add_argument("-i", "--input_yaml", type=Path, required=True, help="Path of yaml input file")
    return parser.parse_args()

def wait_for_completion_wrap(log_files, timeout, interval):
    #I am using this retroactively, it was defined in the gradient.py first, but will serve autopilot.py as well
    from gradient import wait_for_completion  # Import inside function to avoid circular calling
    return wait_for_completion(log_files, timeout, interval)

def subprocess_casci(folder_path, geom_file, settings):
    process = subprocess.Popen(
        ["python3", "-c",
         f"import sys; sys.path.append('{Path(__file__).parent.resolve()}'); "
         f"from launcher import launch_TCcalculation; "
         f"launch_TCcalculation('{folder_path}', '{geom_file}', {settings})"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return process

def casci_c0(fol_name, candidate, geom_file):
    geom_name = geom_file.stem
    energy_calc_path = fol_name / candidate.full_method
    if 'casci' in candidate.full_method.lower():
        scr_location = energy_calc_path / f"scr.{geom_name}" / 'c0'
        return scr_location if scr_location.exists() else None
    return None

def main():
    args = read_single_arguments()
    fn = args.input_yaml
    fol_name = fn.absolute().parents[0]
    settings = io.yload(fn)
    charge = settings['general']['charge']
    n_singlets = settings['reference']['singlets']
    geometry = settings['general']['coordinates']
    geom_file = fol_name / geometry
    mol_name = Path(geometry).stem
    print(f"I'm reading coordinates from {geometry} and will use {settings['general']['basis']} for all the calculations.")

    mol = Molecule.from_xyz(geometry)
    nelec = mol.nelectrons - charge
    print(f"This molecule has {nelec} electrons.")
    assert ((nelec % 2) == 0), "Wait a second.. is this a radical? I don't like that."

    if 'optimization' in settings:
        print(f"\nI will optimize the geometry with {settings['optimization']['method']}. It might take a while..")
        opt_settings = {
                'run': 'minimize',
                'threall': '1.1e-14',
                'convthre': '1.0e-6',
                'precision': 'mixed',
                'maxit': '10000',
                'gpus': '1'
        }

        if 'run' in settings['optimization'] and settings['optimization']['run'].lower() in ['ciopt', 'conical']:
            print(f"Optimization explicitly set to {settings['optimization']['run']}.")
            del opt_settings['run']

        calc_settings = settings['general'] | settings['optimization'] | opt_settings
        opt_path = fol_name / 'opt'
        launch_TCcalculation(opt_path, geom_file, calc_settings)
        file_out = opt_path / 'tc.out'
        file_opt = opt_path / f'scr.{mol_name}' / 'optim.xyz'
        status = io.dynamic_check_opt_status(file_out, file_opt)
        if status:
            os.rename(f'{mol_name}.xyz', f'{mol_name}_initial.xyz')
            optim = io.read_trajectory(file_opt)
            io.write_last_frame_to_file(optim, fol_name / f'{mol_name}.xyz')
            print(f'I wrote the optimized geometry as {mol_name}.xyz and moved the initial geometry to {mol_name}_initial.xyz.')
            
            geom_file = fol_name / f'{mol_name}.xyz'

    assert (settings['reference']['method'] == 'eom'), "I only know how to use EOM-CC2 in TURBOMOLE as a reference."
    print("\nI will use the EOM-CC2 as a reference. Launching calculation now. This one might take a while to finish!")
    launch_TMcalculation(fol_name / 'eom', geom_file, n_singlets-2)

    casci_groups = {}

    if settings['candidates']:
        for calc_type in settings['candidates']:
            print(f"I will launch {calc_type} calculations now.")
            vee_settings = {
                'charge': charge,
                'nelec': nelec,
                'n_singlets': n_singlets,
                'calc_type': calc_type,
            }
            case_settings = vee_settings | settings['candidates'][calc_type]
            candidates_list = CandidateListGenerator(**case_settings).create_candidate_list()

            for candidate in candidates_list:
                folder_path = fol_name / f'{calc_type}{candidate.folder_name}'

                if 'casci' in candidate.full_method.lower():
                    method_search = candidate.full_method.split('_')
                    AS = next((part for part in method_search if part.startswith('AS')), None)
                    if AS:
                        casci_groups.setdefault((calc_type, AS), []).append((candidate, folder_path))
                    continue

    casci_firstjobs = {}
    casci_firstlogs = []
    batch_casci = {}
    other_jobs = {}
    all_calcs_logs = []
    
    #run CASCI first so that we can read in orbitals if different fon temperatures. 
    if casci_groups:
        for (calc_type, AS), candidates in casci_groups.items():
            print(f"\nProcessing CASCI {calc_type} calculations for {AS}")

            multi_fon = any('_T0.' in candidate.full_method for candidate, _ in candidates)

            if multi_fon:
                candidates.sort(key=lambda x: float(x[0].full_method.split('_T0.')[1].split('_')[0]))
                first_candidate, first_fol = candidates[0]
                print(f"\nLaunching first CASCI calculation: {first_candidate.full_method}")

                os.makedirs(first_fol, exist_ok=True)

                candidate_settings = {**settings['general'], **first_candidate.calc_settings}
                launch_TCcalculation(first_fol, geom_file, candidate_settings)
                casci_firstjobs[first_fol] = subprocess_casci(first_fol, geom_file, candidate_settings)
                casci_firstlogs.append((first_fol / 'tc.out', first_fol))

            else:
                print(f"\nOnly one FON temperature detected for {AS}. Submitting all CASCI jobs normally.")
                for candidate, folder_path in candidates:
                    os.makedirs(folder_path, exist_ok=True)
                    candidate_settings = {**settings['general'], **candidate.calc_settings}

                    print(f"\nSubmitting CASCI calculation: {candidate.full_method}")
                    launch_TCcalculation(folder_path, geom_file, candidate_settings)
                    batch_casci[folder_path] = subprocess_casci(folder_path, geom_file, candidate_settings)

        if casci_firstjobs:
            for folder_path, process in casci_firstjobs.items():
                process.wait()

        if casci_firstlogs:
            wait_for_completion_wrap(casci_firstlogs, timeout=3600, interval=45)

    #Submit the rest of the CASCI calcs with the c0 from the frist one, per AS
    for (calc_type, AS), candidates in casci_groups.items():
        first_candidate, first_fol = candidates[0]
        first_c0_path = casci_c0(fol_name, first_candidate, geom_file)
            
        if first_c0_path:
            print(f"\nUsing {first_c0_path} as a guess for subsequent calculations.")

        if len(candidates) == 1:
            print(f"\nOnly one CASCI calculation exists for {AS}: {first_candidate.full_method}. No batch submission needed.")
            continue

        for candidate, folder_path in candidates[1:]:
            os.makedirs(folder_path, exist_ok=True)
            candidate_settings = {**settings['general'], **candidate.calc_settings}
            if first_c0_path:
                candidate_settings['guess'] = str(first_c0_path)

            print(f"\nSubmitting CASCI calculation with guess: {candidate.full_method}")
            launch_TCcalculation(folder_path, geom_file, candidate_settings)
            batch_casci[folder_path] = subprocess_casci(folder_path, geom_file, candidate_settings)
            print("\nFirst CASCI calculations completed.")

    #Now run CASSCF and hhTDA
    for calc_type in settings['candidates']:
        print(f"\nNow launching {calc_type} calculations.")

        vee_settings = {
            'charge': charge,
            'nelec': nelec,
            'n_singlets': n_singlets,
            'calc_type': calc_type,
        }
        case_settings = {**vee_settings, **settings['candidates'][calc_type]}
        candidates_list = CandidateListGenerator(**case_settings).create_candidate_list()

        for candidate in candidates_list:
            folder_path = fol_name / f'{calc_type}{candidate.folder_name}'
            os.makedirs(folder_path, exist_ok=True)

            candidate_settings = {**settings['general'], **candidate.calc_settings}

            if 'casci' not in candidate.full_method.lower():
                launch_TCcalculation(folder_path, geom_file, {**settings['general'], **candidate.calc_settings})
                other_jobs[folder_path] = subprocess_casci(folder_path, geom_file, candidate_settings)
                all_calcs_logs.append((folder_path / 'tc.out', folder_path))

    all_calcs = list(batch_casci.values()) + list(other_jobs.values())

    for process in all_calcs:
        process.wait()
    
    print("\nAll Calculations Submitted\n")

    if all_calcs_logs:
        print("\nWaiting for all calculation to complete...")
        wait_for_completion_wrap(all_calcs_logs, timeout=3600, interval=45)
        print("\nAll calculations logs processed. Ready to launch orbitals.py.")

    print(f"\nLaunching orbitals.py with {fn} as input...")
    subprocess.run(["python3", str(Path(__file__).parent / "orbitals.py"), "-i", str(fn)])
    print("\norbitals.py execution finished.")

if __name__ == "__main__":
    main()
