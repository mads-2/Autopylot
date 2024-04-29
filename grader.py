from dataclasses import dataclass
import pandas as pd
import scoring_functions as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from autopilot import read_single_arguments
import io_utils as io
import shutil
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

    def singlet_osc_order_filter(self):
        osc_array = np.array([self.data[x] for x in [f'S{x+1} osc.' for x in range(self.n_singlets-1)]]).transpose()
        brightness_array = sf.sigmoid(osc_array)
        tot_brightness_array = np.sum(brightness_array, axis=1)
        delta_brightness_array = abs(tot_brightness_array - tot_brightness_array[self.ref_ind])
        osc_score_list = []
        print("Detailing Oscillator Order Scores Calculation:")
        for i, x in enumerate(delta_brightness_array):
            print(f"Candidate {i}: Total Brightness = {tot_brightness_array[i]}, Delta Brightness = {x}")
            if x > 0.1:
                osc_score = float(self.n_singlets - x)
            else:
                ref = brightness_array[self.ref_ind]
                vec = brightness_array[i]
                ind_0, ind_1 = sf.get_diff_indexes(ref, vec)
                combinations = sf.calculate_combinations(ind_0, ind_1)
                min_swaps = sf.calculate_min_swaps(combinations)
                osc_score = self.n_singlets + (1 / (min_swaps + 1))
        
            osc_score_list.append(osc_score)
            print(f"Candidate {i} Score: {osc_score}")

        return np.array(osc_score_list)

    def s_t_order_filter(self):
        singlet_ene_array = np.array([self.data[x] for x in [f'S{x} energy' for x in range(self.n_singlets)]]).transpose()
        triplet_ene_array = np.array([self.data[x] for x in [f'T{x} energy' for x in range(1, self.n_triplets + 1)]]).transpose()
        combined_ene_array = np.concatenate((singlet_ene_array, triplet_ene_array), axis=1)
        ref_ene = combined_ene_array[self.ref_ind]
        energy_order_scores = []
        for i in range(combined_ene_array.shape[0]):
            if i == self.ref_ind:
                energy_order_scores.append(1)
                continue
            candidate_ene = combined_ene_array[i]
            correct_order = all(candidate_ene[j] <= candidate_ene[j + 1] for j in range(len(candidate_ene) - 1))
            score = 1 if correct_order else 0
            energy_order_scores.append(score)
        return np.array(energy_order_scores)

    def abs_score(self, cutoff=0.1):
        abs_score_list = []
        ref_osc_strengths = [self.data.loc[self.ref_ind, f'{state} osc.'] for state in self.state_list[1:]]
        ref_characters = ['Bright' if osc >= cutoff else 'Dark' for osc in ref_osc_strengths]
        print(f"Reference character: {ref_characters}")

        for i, row in self.data.iterrows():
            method_name = row['Method']
            score = 0
            deviations = []
            for j, state in enumerate(self.state_list[1:]):
                f_osc = row[f'{state} osc.']
                cand_character = 'Bright' if f_osc >= cutoff else 'Dark'
                print(f"{method_name}, State {state}, Oscillator strength: {f_osc}, Character: {cand_character}")

                # If ref and cand characters match, 1 point awarded. If they don't match, 1-y points awarded
                if cand_character == ref_characters[j]:
                    score += 1
                else:
                    deviation = abs(f_osc - cutoff)
                    deviations.append(deviation)
                    penalty = sf.sigmoid2(deviation, steepness=50, inflection_point=cutoff/2)
                    print(f"Deviation: {deviation}, Penalty: {penalty: 0.10f}")
                    score += (1 - penalty)

            # Average/Normalized score, S0 excluded
            score /= (len(self.state_list) - 1)
            print(f"{method_name}, Abs score: {score}")
            abs_score_list.append(score)

            plt.figure()
            x_values = np.linspace(0, max(ref_osc_strengths) * 1.5, 100)  # Adjust the range as needed
            y_values = 1 / (1 + np.power(10, 50 * (cutoff / 2 - x_values)))
            plt.plot(x_values, y_values, label='Sigmoid')
            plt.scatter(deviations, [1 / (1 + np.power(10, 50 * (cutoff / 2 - d))) for d in deviations], label=f'{method_name} Deviations')
            plt.xlabel('Deviation')
            plt.ylabel('Penalty')
            plt.title(f'Sigmoid and Deviations for {method_name}')
            plt.legend()
            plt.savefig(f'{method_name}_sigmoid_deviation.png')
            plt.close()

        return np.array(abs_score_list)

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
        time_score = {}
        run_times = self.extract_run_times()
        
        if 'EOM-CC2' in run_times:
            del run_times['EOM-CC2']  # Ensure EOM-CC2 does not affect the calculations
       
        valid_times = [time for time in run_times.values() if not np.isnan(time)]
        min_time = min(valid_times)
        max_time = max(valid_times)
    
        for method, time in run_times.items():
            if not np.isnan(time):
                # Min-max scaling
                scaled_penalty = (time - min_time) / (max_time - min_time) if max_time > min_time else 0
                time_score[method] = 1 - scaled_penalty
            else:
                 time_score[method] = np.nan

        time_score_mapped = self.data['Method'].map(time_score)
        time_score_mapped.loc[self.data['Method'] == 'EOM-CC2'] = 1

        return time_score_mapped

    def suggested_alpha(self):
        ene_array = np.array([self.data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
        ref_max = np.max(ene_array[self.ref_ind])
        alpha_list = []
        for exc in ene_array:
            alpha_list.append(ref_max/np.max(exc))
        return np.array(alpha_list)

    def append_score_columns_to_df(self):
        self.data['abs score'] = self.abs_score()
        self.data['Energies score'] = self.energy_score()
        self.data['Time score'] = self.time_pen()
        self.data['Total score'] = (self.data['abs score'] + self.data['Energies score'] + self.data['Time score'])
        self.data['Suggested alpha'] = self.suggested_alpha()
        self.data.sort_values(by=['Total score'], ascending=False, inplace=True, ignore_index=True)
        print("Method Scores:")
        print(self.data[['Method', 'abs score', 'Energies score', 'Time score', 'Total score']])

    def plot_results(self):
        ene_array = np.array([self.data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
        norm_ene_array = sf.normalize_energy_array(ene_array)
        cand_number = len(self.data)
        xs = list(self.data.index)
        methods = self.data['Method']
        plt.rcParams.update({'font.size': 18})
        fig, [ax0, ax1, ax2] = plt.subplots(3, 1, figsize=[cand_number*2, 18], sharex=True)
        colors = [(0, 0, 1), (1, 0, 0)]
        cmap_name = 'grade_colormap'
        grade_colormap = LinearSegmentedColormap.from_list(cmap_name, colors)
        colors = cm.get_cmap('Dark2', len(self.state_list))
        #my_cmap = cm.get_cmap('Dark2_r')
        
        for i, state in enumerate(self.state_list):
            if 'T' in state:
                linestyle = 'solid' 
                marker='D'
                marker_size = 20
                markerfacecolor = colors(i)
                markeredgewidth = 2
            else:
                linestyle = 'solid'
                marker = None
                marker_size = 0
                markerfacecolor = None
                markeredgewidth = 0
            
            for j in self.data.index:
                dark = state != 'S0' and self.data.loc[j, f'{state} osc.'] < 0.1 and 'S' in state
                style = 'dotted' if dark else linestyle
                line_color = colors(i)
                x_positions = [j - 0.3, j + 0.3]
                y_value = self.data.loc[j, f'{state} energy']
                ax0.plot(x_positions, [y_value, y_value], linestyle=style, marker=marker, color=line_color, linewidth=5, label=self.state_list[i] if j == xs[0] else "")
                ax1.plot(x_positions, [norm_ene_array[j][i], norm_ene_array[j][i]], linestyle=style, marker=marker, color=line_color, linewidth=5)


        rescaled_scores = sf.rescale(self.data['Total score'])
        ax2.bar(xs, self.data['Total score'], color=grade_colormap(rescaled_scores))
        ax0.set_ylabel('Energy difference [eV]')
        ax1.set_ylabel('Norm. E. diff.')
        ax2.set_ylabel('Score')
        ax2.set_xticks(xs)
        ax2.set_xlim(-1.0, cand_number)
        ax2.set_ylim(0.0, max(self.data['Total score']))
        ax2.set_xticklabels(methods, rotation=90)
        ax0.legend(fontsize=20, loc='upper left', bbox_to_anchor=(1.01, 1.0))        
        fig.tight_layout(pad=3.0)
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0, wspace=0.5)
        fig.savefig(f'{self.pointname}_results.png', dpi=300, bbox_inches='tight')

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
   
    def energy_RMSD_plot(self, grader, output_dir):
        # Plot RMSD reference and candidates with y=x trace
        ene_array = np.array([grader.data[x] for x in [f'{x} energy' for x in grader.state_list]]).transpose()
        norm_ene_array = sf.normalize_energy_array(ene_array)
        ref_norm_ene = norm_ene_array[grader.ref_ind]
        for i, method in enumerate(grader.data['Method']):
            if i != grader.ref_ind:
                candidate_norm_ene = norm_ene_array[i]
                rmsd = np.sqrt(np.mean((ref_norm_ene - candidate_norm_ene)**2))
                plt.figure(dpi=300)
                plt.scatter(ref_norm_ene, candidate_norm_ene, label='Data')
                plt.plot([0, 1], [0, 1], color='red', label='y=x')
                plt.xlabel('EOM-CC2 Energies')
                plt.ylabel(f'{method} Energies')
                #plt.title(f'Linear Regression: Reference vs {method}')
                plt.text(0.05, 0.95, f'RMSD = {rmsd:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
                plt.savefig(f'{grader.pointname}_{method}_rmsd.png')
                plt.close()
    
def main():
    args = read_single_arguments()
    fn = args.input_yaml
    settings = io.yload(fn)
    n_singlets = settings['reference']['singlets']
    n_triplets = settings['reference']['triplets']
    singlet_triplet = settings ['general']['singlet_triplet']
    fol_name = fn.absolute().parents[0]
    results = SinglePointResults('S0min', n_singlets, n_triplets, fol_name)
    results.save_csv()
    data = pd.read_csv('S0min_RAW_results.csv', index_col=0)
    
    # Move any "failed" methods into new directory. Failed if reorganization of singlet and triplet states has occurred 
    grader = Grader('S0min', data, n_singlets, n_triplets, 'EOM-CC2')
    grader.plot_time_histogram()
    grader.append_score_columns_to_df()
    singlet_osc_order = grader.singlet_osc_order_filter()
    energy_order_scores = grader.s_t_order_filter()
    grader.data['Osc. order score'] = singlet_osc_order
    grader.data['Energy order score'] = 1  # Default value 
    thresh = n_singlets

    if singlet_triplet:
        grader.data['Energy order score'] = grader.energy_order_filter()
        failed_cand = grader.data[((grader.data['Osc. order score'] <= thresh) & (grader.data['Method'] != 'EOM-CC2')) | (grader.data['Energy order score'] <= 0)]

        print("Failed candidates and their scores:")
        for index, row in failed_cand.iterrows():
            print(f"{row['Method']}: Osc. Order Score = {row['Osc. order score']}, Energy Order Score = {row['Energy order score']}")
    else:
        failed_cand = grader.data[(grader.data['Osc. order score'] <= thresh) & (grader.data['Method'] != 'EOM-CC2')]

        print("Failed candidates and their scores:")
        for index, row in failed_cand.iterrows():
            print(f"{row['Method']}: Osc. Order Score = {row['Osc. order score']}")

    failed_cand_dir = os.path.join(fol_name, 'failed_candidates')

    if not os.path.exists(failed_cand_dir):
        os.makedirs(failed_cand_dir)
    else:
        for filename in os.listdir(failed_cand_dir):
            file_path = os.path.join(failed_cand_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    for method in failed_cand['Method'].tolist():
        src_dir = os.path.join(fol_name, method)
        dst_dir = os.path.join(failed_cand_dir, method)
        if os.path.exists(src_dir):
            shutil.move(src_dir, dst_dir)
            print(f"Moved {method} to {failed_cand_dir}")

    # Only methods who passed the filters
    passed_candidates = grader.data[grader.data['Osc. order score'] > thresh]
    if singlet_triplet:
        passed_candidates = passed_candidates[passed_candidates['Energy order score'] > 0]

    grader.data = passed_candidates
    grader.data.reset_index(drop=True, inplace=True)
    grader.plot_results()
    #grader.save_csv()
    print(grader.data.dropna(axis='columns', how='all').loc[:, (grader.data != 0).any(axis=0)])

    output_dir_ene = os.path.join(fol_name, 'energy_score_plots')
    if not os.path.exists(output_dir_ene):
        os.makedirs(output_dir_ene)

if __name__ == "__main__":
    main()
