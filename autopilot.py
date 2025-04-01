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

def casci_c0(fol_path, mol_name):
    scr_location = fol_path / f"scr.{mol_name}" / "c0"
    print(f"Checking for c0 file at: {scr_location}")

    if scr_location.exists():
        print(f"Found c0 file at {scr_location}")
        return scr_location
    else:
        print(f"No c0 file found at {scr_location}")
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

    #Group all FOMO-CASCI and CASSCF methods
    cas_group = {}

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
                print(f"DEBUG: Creating directory for {candidate.full_method}")
                folder_path = fol_name / candidate.full_method
                
                if 'casci_fomo' in candidate.full_method.lower() or 'casscf' in candidate.full_method.lower():
                    method_parts = candidate.full_method.split('_')
                    AS = next((part for part in method_parts if part.startswith('AS')), None)
                    if AS:
                        cas_group.setdefault(AS, []).append((candidate, folder_path))

    print("\nFiltered CASCI and CASSCF groups:", cas_group.keys())

    casci_firstjobs = {}
    casci_firstlogs = []
    batch_cas = {}
    other_jobs = {}
    all_calcs_logs = []
    first_c0 = {}
    
    #run CASCI first so that we can read in orbitals. 
    if cas_group:
        for AS, candidates in cas_group.items():
            print(f"\nProcessing CASCI/CASSCF calculations for {AS}")

            casci_candidates = [c for c, _ in candidates if 'casci_fomo' in c.full_method.lower() and 'casci' in c.full_method.lower()]

            if casci_candidates:
                casci_candidates = sorted(
                    [c for c, _ in candidates if 'casci_fomo' in c.full_method.lower()],
                    key=lambda c: float(c.full_method.split('_T0.')[1].split('_')[0]) if '_T0.' in c.full_method else float('inf')
    )
                first_candidate = casci_candidates[0]
                first_fol = fol_name / first_candidate.full_method

                print(f"\nLaunching first CASCI calculation: {first_candidate.full_method}")

                os.makedirs(first_fol, exist_ok=True)

                candidate_settings = {**settings['general'], **first_candidate.calc_settings}
                launch_TCcalculation(first_fol, geom_file, candidate_settings)
                casci_firstjobs[first_fol] = subprocess_casci(first_fol, geom_file, candidate_settings)
                casci_firstlogs.append((first_fol / 'tc.out', first_fol))

            else:
                print(f"\nOnly CASSCF found for {AS}. Submitting all calculations normally.")
                for candidate, folder_path in candidates:
                    os.makedirs(folder_path, exist_ok=True)
                    candidate_settings = {**settings['general'], **candidate.calc_settings}
                    print(f"\nSubmitting CASSCF calculation: {candidate.full_method}")
                    launch_TCcalculation(folder_path, geom_file, candidate_settings)
                    batch_cas[folder_path] = subprocess_casci(folder_path, geom_file, candidate_settings)

        if casci_firstjobs:
            for folder_path, process in casci_firstjobs.items():
                process.wait()

        if casci_firstlogs:
            wait_for_completion_wrap(casci_firstlogs, timeout=3600, interval=60)

    #Submit the rest of the CASCI and CASSCF calcs with the c0 from the first CASCI calculation, per AS
    for AS, candidates in cas_group.items():
        print(f"\nProcessing calculations for {AS}")
        casci_candidates = [c for c, _ in candidates if 'casci_fomo' in c.full_method.lower()]
    
        if not casci_candidates:
            print(f"No CASCI-FOMO calculations found for {AS}. Skipping AS.")
            continue

        casci_candidates.sort(
            key=lambda c: float(c.full_method.split('_T0.')[1].split('_')[0]) if '_T0.' in c.full_method else float('inf')
        )

        first_candidate, first_fol = next(((c, f) for c, f in candidates if c == casci_candidates[0]), (None, None))
        
        if first_candidate and first_fol:
            first_c0_path = casci_c0(first_fol, mol_name)

            if first_c0_path:
                first_c0[AS] = first_c0_path
                print(f"\nUsing {first_c0_path} as a guess for subsequent calculations in {AS}.")
            else:
                print(f"\nNo CASCI c0 found for {AS}. Skipping guess assignment.")

        # Now apply the c0 to all other calculations in this AS
        for candidate, folder_path in candidates:
            os.makedirs(folder_path, exist_ok=True)
            candidate_settings = {**settings['general'], **candidate.calc_settings}

            # Only apply the c0 guess if it's not the first CASCI calculation
            if candidate == casci_candidates[0]:
                print(f"\n Avoding Resubmission of {candidate.full_method}")
                continue
        
            if AS in first_c0:
                candidate_settings['guess'] = str(first_c0[AS])
                print(f"\nAssigned c0 guess from {first_c0[AS]} to {candidate.full_method} in {AS}.")
            else:
                print(f"\nSkipping guess for {candidate.full_method} in {AS}.")

            print(f"\nSubmitting calculation for: {candidate.full_method}")
            launch_TCcalculation(folder_path, geom_file, candidate_settings)
            batch_cas[folder_path] = subprocess_casci(folder_path, geom_file, candidate_settings)

            all_calcs_logs.append((folder_path / 'tc.out', folder_path))

    print("\nFirst CASCI calculations completed.")

    #Now runhhTDA and any CASCF that didnt have a CASCI of the same AS
    for calc_type in settings['candidates']:
        print(f"\n***Now launching remaining calculations.***")

        vee_settings = {
            'charge': charge,
            'nelec': nelec,
            'n_singlets': n_singlets,
            'calc_type': calc_type,
        }
        case_settings = {**vee_settings, **settings['candidates'][calc_type]}
        candidates_list = CandidateListGenerator(**case_settings).create_candidate_list()

        for candidate in candidates_list:
            folder_path = fol_name / f"{candidate.full_method}"
            os.makedirs(folder_path, exist_ok=True)

            candidate_settings = {**settings['general'], **candidate.calc_settings}

            if 'casci_fomo' not in candidate.full_method.lower() and 'casscf' not in candidate.full_method.lower():
                launch_TCcalculation(folder_path, geom_file, {**settings['general'], **candidate.calc_settings})
                other_jobs[folder_path] = subprocess_casci(folder_path, geom_file, candidate_settings)
                all_calcs_logs.append((folder_path / 'tc.out', folder_path))

    all_calcs = list(batch_cas.values()) + list(other_jobs.values())

    for process in all_calcs:
        process.wait()
    
    print("\nAll Calculations Submitted\n")

    if all_calcs_logs:
        print("\nWaiting for all calculation to complete.")
        wait_for_completion_wrap(all_calcs_logs, timeout=3600000, interval=90)
        print("\nAll calculations logs processed. Ready to launch orbitals.py.")

    print(f"\nLaunching orbitals.py with {fn} as input.")
    subprocess.run(["python3", str(Path(__file__).parent / "orbitals.py"), "-i", str(fn)])
    print("\norbitals.py execution finished.")

if __name__ == "__main__":
    main()
