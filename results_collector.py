import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

import io_utils as io
from autopilot import read_single_arguments
from energies_parser import FileParser


def exclude_TC_unsuccessful_calculations(fol_name, fn):
    successful_calculation = True
    successful_string = "Job finished:"
    unsuccessful_path = fol_name / "unsuccessful-calculations"
    with open(fn) as out_file:
        if successful_string in out_file.read():
            print(f'{fn} ended correctly')
        else:
            print(f'{fn} HAS PROBLEMS')
            successful_calculation = False
            if not unsuccessful_path.is_dir():
                unsuccessful_path.mkdir()
            shutil.move(fn.parent, unsuccessful_path)
    return successful_calculation


@dataclass
class SinglePointResults:
    pointname: str
    n_singlets: int
    n_triplets: int
    pwd: Path

    @property
    def results_dict(self):
        results_dict = {}
        output_files = list(self.pwd.glob('*_*/tc.out'))
        for fn in output_files:
            successful_calculation = exclude_TC_unsuccessful_calculations(self.pwd, fn)
            if successful_calculation:
                entry_key = str(fn.parent.name)
                hhtda_based = True if 'hhtda' in str(fn) else False
                parser = FileParser(self.n_singlets, self.n_triplets, hhtda_based, fn)
                parser.parse_TC()
                new_entry = parser.create_dict_entry()
                results_dict[entry_key] = new_entry
        parse_eom = FileParser(self.n_singlets, self.n_triplets, False, 'eom/ricc2.out')
        parse_eom.parse_TM()
        new_entry = parse_eom.create_dict_entry()
        results_dict['EOM-CC2'] = new_entry
        return results_dict

    @property
    def state_list(self):
        state_list = []
        for i in range(self.n_singlets):
            state_list.append(f'S{i}')
        for i in range(self.n_triplets):
            state_list.append(f'T{i+1}')
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
                    sub.append(v[label].energy)
                    sub.append(v[label].osc)
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
    n_triplets = settings['reference']['triplets']
    results = SinglePointResults('S0min', n_singlets, n_triplets, fol_name)
    results.save_csv()
    #results.plot_results()


if __name__ == "__main__":
    main()
