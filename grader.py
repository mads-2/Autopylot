from dataclasses import dataclass
import pandas as pd
import scoring_functions as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score
from autopilot import read_single_arguments
import io_utils as io
import os 
import re
from results_collector import SinglePointResults

pd.set_option('display.max_columns', None)

@dataclass
class Grader:
    pointname: str
    data: pd.DataFrame
    n_singlets: int
    n_triplets: int
    ref_string: str

    @property
    def state_list(self):
        state_list = []
        for i in range(self.n_singlets):
            state_list.append(f'S{i}')
        for i in range(self.n_triplets):
            state_list.append(f'T{i+1}')
        return state_list

    @property
    def ref_ind(self):
        ref_ind = sf.get_ref_df_index(self.data, self.ref_string)
        return ref_ind

    def osc_score(self):
        osc_array = np.array([self.data[x] for x in [f'S{x+1} osc.' for x in range(self.n_singlets-1)]]).transpose()
        brightness_array = sf.sigmoid(osc_array)
        tot_brightness_array = np.sum(brightness_array, axis=1)
        delta_brightness_array = abs(tot_brightness_array - tot_brightness_array[self.ref_ind])
        osc_score_list = []
        for i, x in enumerate(delta_brightness_array):
            if x > 0.5:
                osc_score_list.append(float(self.n_singlets - x))
            else:
                ref = brightness_array[self.ref_ind]
                vec = brightness_array[i]
                ind_0, ind_1 = sf.get_diff_indexes(ref, vec)
                combinations = sf.calculate_combinations(ind_0, ind_1)
                min_swaps = sf.calculate_min_swaps(combinations)
                osc_score_list.append(self.n_singlets+(1/(min_swaps+1)))
        return np.array(osc_score_list)

    def energy_score(self):
        ene_array = np.array([self.data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
        norm_ene_array = sf.normalize_energy_array(ene_array)
        #sum_score_list = []
        rmsd_score_list = []
        for i in range(norm_ene_array.shape[0]):
            rmsd = np.sqrt(np.mean((norm_ene_array[self.ref_ind] - norm_ene_array[i]) ** 2))
            rmsd_score_list.append(1 - rmsd)
        return np.array(rmsd_score_list)
    
    def extract_run_times(self):
        run_times = {}
        cwd = os.getcwd()
        for method in self.data['Method']:
            folder_path = os.path.join(cwd, method)
            output_path = os.path.join(folder_path, 'tc.out')
            try:    
                with open(output_path, 'r', encoding='utf-8') as file:
                    found_time = False
                    for line in file:
                        #print(f"Processing line: |{line}|")
                        match = re.search(r'Total processing time:\s*([\d.]+)\s*sec', line)
                        if match:
                            time = float(match.group(1))
                            run_times[method] = time
                            found_time = True
                            print(f"Found time: {time} sec for method: {method}")
                            break
                        if not found_time:
                            #print(f"Time not found in file: {output_path}")
                            run_times[method] = np.nan
            except FileNotFoundError:
                if method not in ['EOM-CC2', 'EOM-CCSD']:
                    print(f"File not found: {output_path}, and if its EOM-CC2/CCSD thats ok!")
                    run_times[method] = np.nan
        return run_times

    def time_pen(self):
        time_penalties = {}
        run_times = self.extract_run_times()
        valid_times = [time for time in run_times.values() if not np.isnan(time)]
        average_time = sum(valid_times) / len(valid_times)
        print(f"Average time: {average_time} sec")
        for method, time in run_times.items():
            time_penalties[method] = ((time - average_time)*0.01) if not np.isnan(time) else np.nan
            time_penalties['EOM-CC2'] = -1  # Explicitly set time penalty/bonus for EOM-CC2 to -1, giving it a boost to n+3 total grade
        return self.data['Method'].map(time_penalties)

    def suggested_alpha(self):
        ene_array = np.array([self.data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
        ref_max = np.max(ene_array[self.ref_ind])
        alpha_list = []
        for exc in ene_array:
            alpha_list.append(ref_max/np.max(exc))
        return np.array(alpha_list)

    def append_score_columns_to_df(self):
        self.data['Osc. score'] = self.osc_score()
        self.data['Energies score'] = self.energy_score()
        self.data['Time penalty/bonus'] = self.time_pen()
        self.data['Total score'] = (self.data['Osc. score'] + self.data['Energies score'] - self.data['Time penalty/bonus'])
        self.data['Suggested alpha'] = self.suggested_alpha()
        self.data.sort_values(by=['Total score'], ascending=False, inplace=True, ignore_index=True)
        print("Method Scores:")
        print(self.data[['Method', 'Osc. score', 'Energies score', 'Time penalty/bonus', 'Total score']])

    def plot_results(self):
        ene_array = np.array([self.data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
        norm_ene_array = sf.normalize_energy_array(ene_array)
        cand_number = len(self.data)
        xs = range(cand_number)
        methods = self.data['Method']
        plt.rcParams.update({'font.size': 18})
        fig, [ax0, ax1, ax2] = plt.subplots(3, 1, figsize=[cand_number/2, 18], sharex=True, sharey=False, gridspec_kw={'wspace': 0, 'hspace': 0})
        colors = cm.get_cmap('Dark2', len(self.state_list))
        my_cmap = cm.get_cmap('Dark2_r')
        for i, state in enumerate(self.state_list):
            for j in range(cand_number):
                dark = state != 'S0' and state[0] == 'S' and self.data[f'{state} osc.'][j] < 0.1
                if dark:
                    style = (0, (1, 1))
                else:
                    style = 'solid'
                ax0.hlines(y=self.data[f'{state} energy'][j], xmin=xs[j]-0.3, xmax=xs[j]+0.3, linewidth=5, color=colors(i), ls=style, label=self.state_list[i] if j == 0 else "")
                ax1.hlines(y=norm_ene_array[j][i], xmin=xs[j]-0.3, xmax=xs[j]+0.3, linewidth=5, color=colors(i), ls=style)
        print(self.data['Total score'].dtype)
        ax2.bar(xs, self.data['Total score'], color=my_cmap(sf.rescale(self.data['Total score'])))
        ax0.set_ylabel('Energy difference [eV]')
        ax1.set_ylabel('Norm. E. diff.')
        ax2.set_ylabel('Score')
        ax2.set_xticks(xs)
        ax2.set_xlim(-1.0, cand_number)
        ax2.set_ylim(0.0, max(self.data['Total score']))
        ax2.set_xticklabels(methods, rotation=90)
        ax0.legend(fontsize=20, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        fig.savefig(f'{self.pointname}_results.png', dpi=300)

    def plot_time_histogram(self):
        run_times = self.extract_run_times()
        valid_times = [time for time in run_times.values() if not np.isnan(time)]

        plt.figure(dpi=300)
        plt.hist(valid_times, bins=20, color='blue', alpha=0.7)
        plt.xlabel('Run Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Run Times')
        plt.grid(True)
        plt.savefig(f'histogram.png')
        plt.close()
    
    def save_csv(self):
        self.data.dropna(axis='columns', how='all').loc[:, (self.data != 0).any(axis=0)].to_csv(f'{self.pointname}_results.csv', index=False)


def main():
    args = read_single_arguments()
    fn = args.input_yaml
    settings = io.yload(fn)
    n_singlets = settings['reference']['singlets']
    n_triplets = settings['reference']['triplets']
    fol_name = fn.absolute().parents[0]
    results = SinglePointResults('S0min', n_singlets, n_triplets, fol_name)
    results.save_csv()
    data = pd.read_csv('S0min_RAW_results.csv', index_col=0)
    scores = Grader('S0min', data, n_singlets, n_triplets, 'EOM-CC2')
    scores.append_score_columns_to_df()
    scores.plot_results()
    scores.save_csv()
    print(scores.data.dropna(axis='columns', how='all').loc[:, (scores.data != 0).any(axis=0)])

    # Plot RMSD reference and candidates with y=x trace
    ene_array = np.array([data[x] for x in [f'{x} energy' for x in scores.state_list]]).transpose()
    norm_ene_array = sf.normalize_energy_array(ene_array)
    ref_norm_ene = norm_ene_array[scores.ref_ind]
    for i, method in enumerate(data['Method']):
        if i != scores.ref_ind:
            candidate_norm_ene = norm_ene_array[i]
            rmsd = np.sqrt(np.mean((ref_norm_ene - candidate_norm_ene)**2))
            plt.figure(dpi=300)
            plt.scatter(ref_norm_ene, candidate_norm_ene, label='Data')
            plt.plot([0, 1], [0, 1], color='red', label='y=x')
            plt.xlabel('EOM-CC2 Energies')
            plt.ylabel(f'{method} Energies')
            #plt.title(f'Linear Regression: Reference vs {method}')
            plt.text(0.05, 0.95, f'RMSD = {rmsd:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
            plt.savefig(f'{scores.pointname}_{method}_rmsd.png')
            plt.close()

    scores.plot_time_histogram()

if __name__ == "__main__":
    main()
