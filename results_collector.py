import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

import io_utils as io
from autopilot import read_single_arguments
from energies_parser import FileParser
import csv

import shutil
from pathlib import Path

def exclude_TC_unsuccessful_calculations(fol_name, fn):
    successful_calculation = True
    successful_string = "Job finished:"
    unsuccessful_path = fol_name / "failed_SPE_jobs"
    
    # Check if the log file exists
    if not fn.exists():
        print(f"File {fn} not found. Checking failed_SPE_jobs directory...")

        # Check if the directory has already been moved to failed_SPE_jobs
        moved_path = unsuccessful_path / fn.parent.name
        if moved_path.exists():
            print(f"Directory {moved_path} already exists in failed_SPE_jobs. Skipping.")
            return False
        else:
            print(f"Directory {fn.parent} does not exist in failed_SPE_jobs. Continuing.")
            return False  # If no log file and directory is not already moved, skip

    try:
        # Open the log file and check for the successful string
        with open(fn) as out_file:
            if successful_string in out_file.read():
                print(f'{fn} ended correctly')
            else:
                print(f'{fn} HAS PROBLEMS')
                successful_calculation = False

                # Move to failed_SPE_jobs if unsuccessful
                if not unsuccessful_path.is_dir():
                    unsuccessful_path.mkdir()

                # Move the directory to failed_SPE_jobs
                shutil.move(str(fn.parent), str(unsuccessful_path / fn.parent.name))
                print(f"Moved {fn.parent} to {unsuccessful_path}")
    except FileNotFoundError:
        print(f"Log file {fn} not found, but directory not yet moved. Skipping calculation.")
        successful_calculation = False

    return successful_calculation

@dataclass
class SinglePointResults:
    pointname: str
    n_singlets: int
    #n_triplets: int
    pwd: Path

    @property
    def results_dict(self):
        results_dict = {}
        self.runtime_only = {}
        output_files = list(self.pwd.glob('*_*/tc.out'))
        #print("Files found for processing:")
        #print([str(fn) for fn in output_files])  # Debug: List all output files

        for fn in output_files:
            # Check for gradient methods
            if 'gradient_' in str(fn.parent.name):
                parent_method = fn.parent.name
                print(f"Processing gradient calculation for runtime only: {parent_method}")
                self.runtime_only[parent_method] = fn
                continue

            # Process non-gradient methods
            successful_calculation = exclude_TC_unsuccessful_calculations(self.pwd, fn)
            if successful_calculation:
                entry_key = str(fn.parent.name)
                print(f"Processing method: {entry_key}")

                if 'hhtda' in str(fn):
                    parser = FileParser(self.n_singlets, True, fn)
                else:
                    parser = FileParser(self.n_singlets, False, fn)

                parser.parse_TC()
                new_entry = parser.create_dict_entry()
                results_dict[entry_key] = new_entry

        # Process EOM-CC2 reference
        try:
            parse_eom = FileParser(self.n_singlets, False, 'eom/ricc2.out')
            parse_eom.parse_TM()
            new_entry = parse_eom.create_dict_entry()
            results_dict['EOM-CC2'] = new_entry
        except FileNotFoundError:
            print("EOM-CC2 reference file not found. Skipping.")

        # Write results to CSV for debugging
        with open("TCresult.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Method", "Data"])
            for key, val in results_dict.items():
                writer.writerow([key, val])

        # Debug: Print runtime_only
        #print("Runtime-only gradient methods:")
        #print(self.runtime_only)

        return results_dict

    @property
    def state_list(self):
        state_list = []
        for i in range(self.n_singlets):
            state_list.append(f'S{i}')
        #for i in range(self.n_triplets):
            #state_list.append(f'T{i+1}')
        return state_list

    @property
    def col_header_list(self):
        col_header_list = []
        col_header_list.append('Method')
        for label in self.state_list:
            col_header_list.append(f'{label} energy')
            col_header_list.append(f'{label} osc.')
        return col_header_list

    @property
    def data_list(self):
        data_list = []
        for k, v in self.results_dict.items():
            sub = []
            for label in self.state_list:
                if label == 'S0':
                    sub.append(0.00)
                    sub.append(None)
                else:
                    excitation = v.get(label)  # Safely get the Excitation object if it exists
                    if excitation:
                        energy = excitation.energy  # Directly access attributes from Excitation object
                        osc = excitation.osc
                    else:
                        energy = None
                        osc = None
                    sub.append(energy)
                    sub.append(osc)
            data_list.append([k] + sub)
        return data_list

    @property
    def df(self):
        return pd.DataFrame(self.data_list, columns=self.col_header_list)

    def save_csv(self):
        self.df.to_csv(f'{self.pointname}_RAW_results.csv')


def main():
    args = read_single_arguments()
    fn = args.input_yaml
    fol_name = fn.absolute().parents[0]
    settings = io.yload(fn)
    n_singlets = settings['reference']['singlets']
    #n_triplets = settings['reference']['triplets']
    results = SinglePointResults('S0min', n_singlets, fol_name)
    results.save_csv()
    #results.plot_results()


if __name__ == "__main__":
    main()
