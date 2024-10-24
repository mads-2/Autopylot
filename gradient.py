import os
import time
import re
import sys
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
    description_string = "This script will launch AutoPilot on a geometry"
    parser = ArgumentParser(description=description_string)
    parser.add_argument("-i", "--input_yaml", type=Path, required=True, help="Path of yaml input file")
    return parser.parse_args()

def run_gradient_calculations(yaml_file, timeout=12000, interval=10):
    """Run gradient calculations for candidates as specified in the YAML file and wait for them to complete."""
    # Load settings from the YAML file
    settings = io.yload(yaml_file)
    n_singlets = settings['reference']['singlets']
    n_triplets = settings['reference'].get('triplets', 0)  # Default to 0 if not defined
    fol_name = Path(yaml_file).absolute().parents[0]
    charge = settings['general']['charge']
    geometry = settings['general']['coordinates']
    geom_file = fol_name / geometry
    mol_name = Path(geometry).stem

    mol = Molecule.from_xyz(geometry)
    nelec = mol.nelectrons - charge
    
    log_files = []

    if 'candidates' in settings:
        for calc_type in settings['candidates']:
            print(f"I will launch gradient calculations now for calc_type: {calc_type}")

            # Create the generator with the correct settings
            vee_settings = settings['candidates'][calc_type]
            vee_settings.update({
                'calc_type': calc_type,
                'nelec': nelec,
                'charge': charge,
                'n_singlets': n_singlets,
                'n_triplets': n_triplets,
            })

            # Merge settings specific to the calculation type from YAML
            case_settings = vee_settings | settings['candidates'][calc_type]

            # Generate a list of candidates using the updated settings
            candidates_list = CandidateListGenerator(**case_settings).create_candidate_list()
            print(f"Generated {len(candidates_list)} candidates for calc_type: {calc_type}")

            completed_candidates = []
            for candidate in candidates_list:
                log_name = fol_name / f'{candidate.full_method}' / 'tc.out'
                if exclude_TC_unsuccessful_calculations(fol_name, log_name):
                    completed_candidates.append(candidate)
                else:  
                    print(f"Skipping {candidate.folder_name}, failed SPE calculation.")

                print(f"Filtered to {len(completed_candidates)} successful candidates for calc_type: {calc_type}")

            for candidate in completed_candidates:
                candidate.validate_as()

                # Build the folder path and check if it exists
                gradient_folder_path = fol_name / f'gradient_{candidate.full_method}'
                print(f"Folder path: {gradient_folder_path}")

                calc_settings = {**settings['general'], **candidate.calc_settings}
                calc_settings['run'] = 'gradient'

                if gradient_folder_path.exists():
                    print(f"Directory {gradient_folder_path} already exists. Using existing data.")
                else:
                    os.makedirs(gradient_folder_path, exist_ok=True)
                    launch_TCcalculation(gradient_folder_path, geom_file, calc_settings)
                    print(f"Launched gradient calculation for {candidate.folder_name} in {gradient_folder_path}")

                log_file = gradient_folder_path / 'tc.out'
                log_files.append((log_file, gradient_folder_path, geom_file, calc_settings))

    failed_jobs = wait_for_completion(log_files, timeout, interval, fol_name)

    if failed_jobs:
        log_failed_jobs(failed_jobs)

def wait_for_completion(log_files, timeout=12000, interval=10, fol_name=Path()):
    """Wait for all calculations to complete or timeout."""
    start_time = time.time()
    extract_time_pattern = re.compile(r'Total processing time:\s*([\d.]+)\s*sec')
    error_pattern = re.compile(r'terminated|error')
    failed_jobs = []

    SPEs_of_failed_gradients = fol_name / "SPEs_of_failed_gradients" # Directory for candidates SPEs when their Gradient failed
    if not SPEs_of_failed_gradients.exists():
        SPEs_of_failed_gradients.mkdir()

    while time.time() - start_time < timeout:
        all_completed = True
        for log_file, gradient_folder_path, geom_file, calc_settings in log_files:
            try:
                # Check the gradient output file for completion
                with open(log_file, 'r') as file:
                    contents = file.read()
                    found_processing_time = extract_time_pattern.search(contents)
                    found_error = error_pattern.search(contents)

                    if found_processing_time:
                        print(f"Job completed successfully: {log_file}")
                        continue  # Job completed successfully, skip to next job

                    elif found_error:
                        print(f"Error detected in {log_file}. Adding {gradient_folder_path} to failed jobs.")
                        print(f"Moving {candidate_directory} from CWD. With no gradeint calculation, candidate can not be graded.")
                        shutil.move(str(candidate_directory), str(SPEs_of_failed_gradients / candidate_directory.name))
                        failed_jobs.append((log_file, gradient_folder_path, geom_file, calc_settings))
                        continue  # Move to next job after adding to failed jobs

                    else:
                        # If neither error nor success is found, consider it incomplete
                        all_completed = False
             
            except FileNotFoundError:
                print(f"Log file {log_file} not found. Adding {gradient_folder_path} to failed jobs.")
                failed_jobs.append((log_file, gradient_folder_path, geom_file, calc_settings))
        
        if all_completed:
            print("All gradient calculations have completed (successfully or with errors).")
            return failed_jobs

        time.sleep(interval)  # Sleep before checking again
        print("Still waiting for gradient calculations to complete...")

def log_failed_jobs(failed_jobs, log_file="failed_gradient_jobs.log", failed_dir="failed_gradient_jobs"):
    """Log failed jobs for manual inspection and move them to a new directory."""

    # Create a directory for failed jobs if it doesn't exist
    failed_dir_path = Path(failed_dir)
    failed_dir_path.mkdir(exist_ok=True)

    with open(log_file, 'w') as file:
        for log_file, folder_path, _, _ in failed_jobs:
            # Log the failed job directory
            file.write(f"Failed: {folder_path}\n")

            # Move the failed job to the new directory
            dest_path = failed_dir_path / folder_path.name
            print(f"Moving {folder_path} to {dest_path}")
            shutil.move(str(folder_path), str(dest_path))  # Move the entire directory

    print(f"Logged {len(failed_jobs)} failed jobs in {log_file}")

def print_failed_job_counts(failed_spe_dir, failed_gradient_dir):
    # Get the number of directories in failed_SPE_jobs
    spe_path = Path(failed_spe_dir)
    gradient_path = Path(failed_gradient_dir)
    
    # Count directories in failed_SPE_jobs
    spe_dirs = [d for d in spe_path.iterdir() if d.is_dir()]
    num_spe_dirs = len(spe_dirs)
    
    # Count directories in failed_gradient_jobs
    gradient_dirs = [d for d in gradient_path.iterdir() if d.is_dir()]
    num_gradient_dirs = len(gradient_dirs)    

    # Print the results
    print(f"Number of unsuccessful SPE calcuations: {num_spe_dirs}. Please see failed_SPE_jobs Directory")
    print(f"Number of unsuccessful Gradient calcuations {failed_gradient_dir}: {num_gradient_dirs}. Please see failed_SPE_jobs Directory")

if __name__ == "__main__":
    args = read_single_arguments()
    yaml_file = args.input_yaml

    run_gradient_calculations(yaml_file)
    #print_failed_job_counts('failed_SPE_jobs', 'failed_gradient_jobs')
