import os
import time
import re
import shutil
from argparse import ArgumentParser
from pathlib import Path
from molecule import Molecule  
from candidate import CandidateListGenerator
from launcher import launch_TCcalculation
from results_collector import exclude_TC_unsuccessful_calculations
import io_utils as io  

def read_single_arguments():
    """Parse input YAML file argument."""
    parser = ArgumentParser(description="Launch AutoPilot on a geometry.")
    parser.add_argument("-i", "--input_yaml", type=Path, required=True, help="Path of yaml input file")
    return parser.parse_args()

def run_gradient_calculations(yaml_file, timeout=12000, interval=10):
    """Run gradient calculations for candidates in the YAML file and wait for all to complete."""
    settings = io.yload(yaml_file)
    n_singlets = settings['reference']['singlets']
    n_triplets = settings['reference'].get('triplets', 0)
    fol_name = Path(yaml_file).absolute().parent
    charge = settings['general']['charge']
    geometry = settings['general']['coordinates']
    geom_file = fol_name / geometry
    mol = Molecule.from_xyz(geometry)
    nelec = mol.nelectrons - charge
    
    log_files = []
    if 'candidates' in settings:
        for calc_type in settings['candidates']:
            vee_settings = settings['candidates'][calc_type]
            vee_settings.update({
                'calc_type': calc_type,
                'nelec': nelec,
                'charge': charge,
                'n_singlets': n_singlets,
                'n_triplets': n_triplets,
            })

            case_settings = vee_settings | settings['candidates'][calc_type]
            candidates_list = CandidateListGenerator(**case_settings).create_candidate_list()
            completed_candidates = [
                candidate for candidate in candidates_list
                if exclude_TC_unsuccessful_calculations(fol_name, fol_name / f'{candidate.full_method}' / 'tc.out')
            ]

            for candidate in completed_candidates:
                gradient_folder_path = fol_name / f'gradient_{candidate.full_method}'
                if not gradient_folder_path.exists():
                    os.makedirs(gradient_folder_path, exist_ok=True)
                    launch_TCcalculation(gradient_folder_path, geom_file, candidate.calc_settings | settings['general'])
                log_files.append((gradient_folder_path / 'tc.out', gradient_folder_path))

    failed_jobs, gradient_errors = wait_for_completion(log_files, timeout, interval)
    move_failed_jobs(failed_jobs, gradient_errors, fol_name)

def wait_for_completion(log_files, timeout=12000, interval=10):
    """Wait for all gradient calculations to complete or timeout, returning lists of incomplete and error jobs."""
    start_time = time.time()
    extract_time_pattern = re.compile(r'Total processing time:\s*([\d.]+)\s*sec')
    error_pattern = re.compile(r'terminated|error')
    failed_jobs = []
    gradient_errors = []

    while time.time() - start_time < timeout:
        all_completed = True
        for log_file, gradient_folder_path in log_files:
            try:
                # Only process logs for folders named with "gradient_" to avoid non-gradient jobs
                if "gradient_" not in gradient_folder_path.name:
                    continue

                with open(log_file, 'r') as file:
                    contents = file.read()
                    found_processing_time = extract_time_pattern.search(contents)
                    found_error = error_pattern.search(contents)

                    if found_processing_time:
                        continue  # Successful completion, skip to next log
                    elif found_error:
                        gradient_errors.append((log_file, gradient_folder_path))  # Log as gradient error
                    else:
                        all_completed = False  # Incomplete, continue monitoring

            except FileNotFoundError:
                all_completed = False  # Log file not found yet, keep waiting

        if all_completed:
            print("All gradient calculations have completed or failed.")
            break

        print("Still waiting for gradient calculations to complete...")
        time.sleep(interval)

    return failed_jobs, gradient_errors

def move_failed_jobs(failed_jobs, gradient_errors, fol_name):
    """Move failed jobs to SPEs_of_failed_gradients or failed_gradient_jobs after all jobs are confirmed complete."""
    SPEs_of_failed_gradients = fol_name / "SPEs_of_failed_gradients"
    failed_gradient_jobs = fol_name / "failed_gradient_jobs"
    SPEs_of_failed_gradients.mkdir(exist_ok=True)
    failed_gradient_jobs.mkdir(exist_ok=True)

    extract_time_pattern = re.compile(r'Total processing time:\s*([\d.]+)\s*sec')  # Re-check for success
    for log_file, gradient_folder_path in failed_jobs:
        # Double-check the pattern to avoid incorrect moves
        with open(log_file, 'r') as file:
            if extract_time_pattern.search(file.read()):
                print(f"Skipping move: {gradient_folder_path} completed successfully.")
                continue  # Skip moving successful candidates

        candidate_directory = gradient_folder_path.parent / gradient_folder_path.name.replace("gradient_", "")
        dest_path = SPEs_of_failed_gradients / candidate_directory.name
        print(f"Moving failed gradient job {candidate_directory} to {dest_path}")
        shutil.move(str(candidate_directory), str(dest_path))

    # Move specific gradient errors
    for log_file, gradient_folder_path in gradient_errors:
        candidate_directory = gradient_folder_path.parent / gradient_folder_path.name.replace("gradient_", "")
        dest_path = failed_gradient_jobs / candidate_directory.name
        print(f"Moving job with gradient error {candidate_directory} to {dest_path}")
        shutil.move(str(candidate_directory), str(dest_path))

if __name__ == "__main__":
    args = read_single_arguments()
    yaml_file = args.input_yaml
    run_gradient_calculations(yaml_file)
