import os
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

def main():
    args = read_single_arguments()
    fn = args.input_yaml
    fol_name = fn.absolute().parents[0]
    settings = io.yload(fn)
    charge = settings['general']['charge']
    n_singlets = settings['reference']['singlets']
    n_triplets = settings['reference']['triplets']
    geometry = settings['general']['coordinates']
    geom_file = fol_name / geometry
    mol_name = Path(geometry).stem
    print(f"I'm reading coordinates from {geometry} and will use {settings['general']['basis']} for all the calculations.")

    mol = Molecule.from_xyz(geometry)
    nelec = mol.nelectrons - charge
    print(f"This molecule has {nelec} electrons.")
    assert ((nelec % 2) == 0), "Wait a second.. is this a radical? I don't like that."

    if 'optimization' in settings:
        print(f"I will optimize the geometry with {settings['optimization']['method']}. It might take a while..")
        opt_settings = {
                'run': 'minimize',
                'threall': '1.1e-14',
                'convthre': '1.0e-6',
                'precision': 'mixed',
                'gpus': '1'
        }
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

    assert (settings['reference']['method'] == 'eom'), "I only know how to use EOM-CC2 in TURBOMOLE as a reference."
    
    singlet_eom_path = fol_name / 'eom_singlet'
    print("I will use the EOM-CC2 as a reference. This calculation is for singlets!")
    launch_TMcalculation(singlet_eom_path, geom_file, n_singlets-2, 0)  # Triplets set to 0 for singlet calculation

    triplet_eom_path = fol_name / 'eom_triplet'
    print("I will use the EOM-CC2 as a reference. This calculation is for triplets!")
    launch_TMcalculation(triplet_eom_path, geom_file, 0, n_triplets)  # Singlets set to 0 for triplet calculation

    if settings['candidates']:
        for calc_type in settings['candidates']:
            print(f"I will launch {calc_type} calculations now.")
            # Generate and launch singlet candidates
            singlet_vee_settings = {
                'charge': charge,
                'nelec': nelec,
                'n_singlets': n_singlets,
                'n_triplets': 0,
                'calc_type': calc_type,
            }
            singlet_case_settings = singlet_vee_settings | settings['candidates'][calc_type]
            singlet_candidates_list = CandidateListGenerator(**singlet_case_settings)
            for candidate in singlet_candidates_list.create_candidate_list(n_singlets=n_singlets):
                launch_TCcalculation(fol_name / f'{calc_type}{candidate.folder_name}', geom_file, settings['general'] | candidate.calc_settings)

            # Generate and launch triplet candidates
            triplet_vee_settings = {
                'charge': charge,
                'nelec': nelec,
                'n_singlets': 0,
                'n_triplets': n_triplets,
                'calc_type': calc_type,
            }
            triplet_case_settings = triplet_vee_settings | settings['candidates'][calc_type]
            triplet_candidates_list = CandidateListGenerator(**triplet_case_settings)
            for candidate in triplet_candidates_list.create_candidate_list(n_triplets=n_triplets):
                launch_TCcalculation(fol_name / f'{calc_type}{candidate.folder_name}', geom_file, settings['general'] | candidate.calc_settings)

if __name__ == "__main__":
    main()
