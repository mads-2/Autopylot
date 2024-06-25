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
import time
import os 
import re
from pathlib import Path
from results_collector import SinglePointResults
from candidate import CandidateListGenerator
from launcher import launch_TCcalculation
from molecule import Molecule

pd.set_option('display.max_columns', None) 

@dataclass
class Grader:
    pointname: str
    data: pd.DataFrame
    n_singlets: int
    n_triplets: int
    ref_string: str
    settings: dict
    dft_methods_avail = ['wpbe', 'wB97x', 'wpbe', 'wpbeh', 'camb3lyp', 'rhf', 'b3lyp', 'pbe0', 'pbe', 'bhandhlyp', 'blyp', 'pw91', 'b3pw91', 'wb97'] #This is what the code will recognize for the time being

    @property
    def state_list(self):
        return [f'S{i}' for i in range(self.n_singlets)] + [f'T{i+1}' for i in range(self.n_triplets)]

    @property
    def ref_ind(self):
        return sf.get_ref_df_index(self.data, self.ref_string)

    def singlet_osc_order_filter(self):
        char_thresh = self.settings['grader']['character_filter']['char_thresh']
        inflection_point = self.settings['grader']['character_filter']['inflection_point']
        osc_array = np.array([self.data[x] for x in [f'S{x+1} osc.' for x in range(self.n_singlets-1)]]).transpose()
        brightness_array = sf.sigmoid(osc_array, steepness=50, inflection_point=inflection_point)
        tot_brightness_array = np.sum(brightness_array, axis=1)
        delta_brightness_array = abs(tot_brightness_array - tot_brightness_array[self.ref_ind])
        osc_score_list = []
        print("Detailing Oscillator Order Scores Calculation:")
        for i, x in enumerate(delta_brightness_array):
            print(f"Candidate {i}: Total Brightness = {tot_brightness_array[i]}, Delta Brightness = {x}")
            if x > char_thresh:
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
        energy_threshold = self.settings['grader']['order_filter']['Ediff']
        singlet_ene_array = np.array([self.data[x] for x in [f'S{x} energy' for x in range(self.n_singlets)]]).transpose()
        triplet_ene_array = np.array([self.data[x] for x in [f'T{x} energy' for x in range(1, self.n_triplets + 1)]]).transpose()
        combined_ene_array = np.concatenate((singlet_ene_array, triplet_ene_array), axis=1)
        ref_ene = combined_ene_array[self.ref_ind]
        ref_sorted_indices = np.argsort(ref_ene)
        #print("Combined Energy Array:")
        #print(combined_ene_array)
        energy_order_scores = []
        for i in range(combined_ene_array.shape[0]):
            candidate_ene = combined_ene_array[i]
            candidate_sorted_indices = np.argsort(candidate_ene)
            correct_order = np.array_equal(ref_sorted_indices, candidate_sorted_indices)
            
            if not correct_order:
                correct_order = True #Assume correct unless proven otherwise (aka leniency to small mistakes in ordering)
                for j in range(len(candidate_ene)):
                    if candidate_sorted_indices[j] != ref_sorted_indices[j]:
                        actual_index = candidate_sorted_indices[j]
                        ref_index = ref_sorted_indices[j]
                        if abs(candidate_ene[actual_index] - ref_ene[ref_index]) > energy_threshold:
                            correct_order = False
                            break
            
            print(f"Candidate {i} energies: {candidate_ene}, Order Correct: {correct_order}")
            score = 1 if correct_order else 0
            energy_order_scores.append(score)
        return np.array(energy_order_scores)

    def abs_score(self, cutoff=0.1):
        multiplier = self.settings['grader']['abs_grader']['pen_mult_a']
        cutoff = self.settings['grader']['abs_grader']['a_cutoff']
        a_steepness = self.settings['grader']['abs_grader']['a_steepness']
        a_inflection_point = self.settings['grader']['abs_grader']['a_inflection_point']
        target_dir = 'sigmoid_plots'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

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
                    penalty = sf.sigmoid2(deviation, steepness=a_steepness, inflection_point=a_inflection_point)
                    print(f"Deviation: {deviation}, Penalty: {penalty: 0.10f}")
                    score_increment = max(0,(1 - (penalty * multiplier)))
                    score += score_increment

            # Average/Normalized score, S0 excluded
            score /= (len(self.state_list) - 1)
            print(f"{method_name}, Abs score: {score}")
            abs_score_list.append(score)

            plt.figure()
            x_values = np.linspace(0, max(ref_osc_strengths) * 1.5, 100)  # Adjust the range as needed
            y_values = 1 / (1 + np.power(10, a_steepness * (a_inflection_point - x_values)))
            plt.plot(x_values, y_values, label='Sigmoid')
            plt.scatter(deviations, [1 / (1 + np.power(10, a_steepness * (a_inflection_point - d))) for d in deviations], label=f'{method_name} Deviations')
            plt.xlabel('Deviation')
            plt.ylabel('Penalty')
            plt.title(f'Sigmoid and Deviations for {method_name}')
            plt.legend()
            file_path = os.path.join(target_dir, f'{method_name}_sigmoid_deviation.svg')
            plt.savefig(file_path)
            plt.close()
            print(f"Saved plot to {file_path}")

        return np.array(abs_score_list)

    def energy_score(self):
        multiplier = self.settings['grader']['energy_grader']['pen_mult_b'] 
        ene_array = np.array([self.data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
        norm_ene_array = sf.normalize_energy_array(ene_array)
        #sum_score_list = []
        rmsd_score_list = []
        for i in range(norm_ene_array.shape[0]):
            rmsd = np.sqrt(np.mean((norm_ene_array[self.ref_ind] - norm_ene_array[i]) ** 2))
            rmsd_score_list.append(1 - (multiplier*rmsd))
        return np.array(rmsd_score_list)
    
    def extract_run_times(self, methods):
        run_times = {}
        cwd = os.getcwd()
        extract_time_pattern = re.compile(r'Total processing time:\s*([\d.]+)\s*sec')
        for method in methods:
            folder_path = os.path.join(cwd, f"gradient_{method}")  # Assuming gradient calculations are stored in directories prefixed with 'gradient_'
            output_path = os.path.join(folder_path, 'tc.out')  # Adjust filename as per your gradient calculation output file
            try:
                with open(output_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        match = extract_time_pattern.search(line)
                        if match:
                            run_times[method] = float(match.group(1))
                            break
                    else:
                        run_times[method] = np.nan  # Set NaN if no time found
            except FileNotFoundError:
                run_times[method] = np.nan  # Set NaN if file not found
        return run_times

    def time_pen(self):
        multiplier = self.settings['grader']['time_grader']['pen_mult_c']
        methods = self.data['Method'].unique()  # Only consider methods for the passed candidates
        run_times = self.extract_run_times(methods)
    
        valid_times = [time for time in run_times.values() if not np.isnan(time)]
        if valid_times:
            min_time = min(valid_times)
            max_time = max(valid_times)
            time_score = {method: (1 - ((time - min_time) / (max_time - min_time) * multiplier))
                          for method, time in run_times.items() if not np.isnan(time)}
        else:
            time_score = {method: np.nan for method in methods}

        # Map the scores back to the DataFrame
        self.data['Time score'] = self.data['Method'].map(time_score)
        self.data.loc[self.data['Method'] == 'EOM-CC2', 'Time score'] = 1  # Optionally assign full score to EOM-CC2 if needed

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

    def prepare_data_for_grouping(self):
        def extract_components(method_name):
            parts = method_name.split('_')
            method_base = parts[0]
            active_space = None
            dft_method = None

            for part in parts[1:]:
                if part.startswith('AS'):
                    if not active_space:
                        active_space = part
                elif any(dft in part for dft in self.dft_methods_avail):
                    if not dft_method:
                        dft_method = part

            return pd.Series([method_base, active_space, dft_method], index=['Method Base', 'Active Space', 'DFT Method'])

        # Apply the extraction to create a new DataFrame slice
        components_df = self.data['Method'].apply(extract_components)
        self.data = self.data.join(components_df)

        # Debug: Print to verify columns
        print("Extracted components preview:")
        print(components_df.head())
        print("Data preview with new columns:")
        print(self.data.head())

    def rep_cand_select(self):
        if not self.data.empty:
            print("Preparing data for grouping. Initial data shape:", self.data.shape)

            # Ensure the necessary columns exist
            if 'Method Base' in self.data.columns and 'Active Space' in self.data.columns and 'DFT Method' in self.data.columns:
                # Determine which column to use for each method dynamically
                self.data['Group Criterion'] = self.data.apply(
                    lambda x: 'Active Space' if pd.notna(x['Active Space']) else 'DFT Method',
                    axis=1
                )

                # Handle None properly by replacing it with a placeholder
                self.data[['Method Base', 'Active Space', 'DFT Method']] = self.data[['Method Base', 'Active Space', 'DFT Method']].fillna('None')

                # Create a new grouping column based on the determined criterion for each row
                self.data['Grouping Key'] = self.data.apply(
                    lambda x: f"{x['Method Base']}_{x[x['Group Criterion']]}", axis=1
                )

                # Group by 'Grouping Key'
                grouped = self.data.groupby('Grouping Key', as_index=False)
                # Apply selection to find the maximum 'Total score' in each group
                self.data = grouped.apply(lambda x: x.nlargest(1, 'Total score')).reset_index(drop=True)

                print("Data after selecting representative candidates:")
                print(self.data[['Method', 'Total score']])  # Debug print

                if self.data.empty:
                    print("Warning: No data available after grouping and selecting representatives. Check grouping logic.")
            else:
                print("Necessary grouping columns are missing from DataFrame.")
        else:
            print("Data is empty before grouping. Skipping representative selection.")

    def wait_for_completion(self, log_files, timeout=5000, interval=10):
        """
        Poll log files for the completion message.

        :param log_files: List of paths to log files.
        :param timeout: Maximum time to wait in seconds.
        :param interval: Time between checks in seconds.
        :return: True if all files have the completion message, False if timeout.
        """
        start_time = time.time()
        time_pattern = re.compile(r'Total processing time:\s*([\d.]+)\s*sec')
        while time.time() - start_time < timeout:
            all_completed = True
            for log_file in log_files:
                try:
                    with open(log_file, 'r') as file:
                        contents = file.read()
                        if not time_pattern.search(contents):
                            all_completed = False
                            break
                except FileNotFoundError:
                    all_completed = False
                    break
            if all_completed:
                print("All gradient calculations have completed.")
                return True
            time.sleep(interval)  # Sleep before checking again
            print("Still waiting for gradient calculations to complete...")
        print("Timeout waiting for gradient calculations to complete.")
        return False

    def plot_results(self):
        filtered_data = self.data[~self.data['Method'].str.contains('gradient_', na=False)]
        if not filtered_data.empty and 'Total score' in filtered_data.columns and not filtered_data['Total score'].isna().all():
            ene_array = np.array([filtered_data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
            norm_ene_array = sf.normalize_energy_array(ene_array)
            cand_number = len(filtered_data)
            xs = list(filtered_data.index)
            methods = filtered_data['Method']

            max_width = 150
            fig_width = min(cand_number * 2, max_width)
            fig_height = 24

            plt.rcParams.update({'font.size': 18})
            fig, [ax0, ax1, ax2, ax3] = plt.subplots(4, 1, figsize=[fig_width, fig_height], sharex=True)
            colors = [(0, 0, 1), (1, 0, 0)]
            cmap_name = 'grade_colormap'
            grade_colormap = LinearSegmentedColormap.from_list(cmap_name, colors)
            colors = cm.get_cmap('tab10', len(self.state_list))
        
            for i, state in enumerate(self.state_list):
                if 'T' in state:
                    linestyle = 'dotted' 
                    marker='D'
                    marker_size = 20
                    markerfacecolor = colors(i)
                    markeredgewidth = 2
                    line_width = 2
                else:
                    linestyle = 'solid'
                    marker = None
                    marker_size = 0
                    markerfacecolor = None
                    markeredgewidth = i
                    line_width = 5
            
                for j in filtered_data.index:
                    dark = state != 'S0' and filtered_data.loc[j, f'{state} osc.'] < 0.1 and 'S' in state
                    style = 'dotted' if dark else linestyle
                    line_color = colors(i)
                    x_positions = [j - 0.3, j + 0.3]
                    y_value = filtered_data.loc[j, f'{state} energy']
                    ax0.plot(x_positions, [y_value, y_value], linestyle=style, marker=marker, color=line_color, linewidth=line_width, label=f'${self.state_list[i].replace("T", "T_").replace("S", "S_")}$' if j == xs[0] else "")
                    ax1.plot(x_positions, [norm_ene_array[j][i], norm_ene_array[j][i]], linestyle=style, marker=marker, color=line_color, linewidth=line_width)

            # Plot total scores as a bar graph on ax3
            rescaled_scores = sf.rescale(filtered_data['Total score'])
            bars = ax3.bar(xs, filtered_data['Total score'], color=grade_colormap(rescaled_scores))

            for bar, score in zip(bars, filtered_data['Total score']):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{score:.2f}', ha='center', va='center', color='black', fontsize=32, rotation=90)

            ax2.plot(xs, filtered_data['abs score'], color='r', label='Abs Score', linewidth=5, marker='8', markersize=20)
            ax2.plot(xs, filtered_data['Energies score'], color='g', label='Energies Score', linewidth=5, marker='8', markersize=20)
            ax2.plot(xs, filtered_data['Time score'], color='b', label='Time Score', linewidth=5, marker='8', markersize=20)
            ax2.plot(xs, filtered_data['Total score'], color='black', label='Total Score', linewidth=5, marker='8', markersize=20)

            ax0.set_ylabel('Energy difference [eV]')
            ax1.set_ylabel('Norm. E. diff.')
            ax3.set_ylabel('Score')
            ax2.set_ylabel('Component Scores and Total')
            ax2.legend()
            ax3.set_xticks(xs)
            ax3.set_xlim(-1.0, cand_number)
            ax3.set_ylim(0.0, max(filtered_data['Total score']))
            ax3.set_xticklabels(methods, rotation=90)
            ax0.legend(fontsize=20, loc='upper left', bbox_to_anchor=(1.01, 1.0))

            fig.tight_layout(pad=3.0)
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0, wspace=0.5)
            fig.savefig(f'{self.pointname}_results.svg', dpi=300, bbox_inches='tight')
        else:
            print("No valid data available to plot.")

    def plot_time_histogram(self):
        run_times = self.extract_run_times()
        valid_times = [time for time in run_times.values() if not np.isnan(time)]

        plt.figure(dpi=300)
        plt.hist(valid_times, bins=20, color='blue', alpha=0.7)
        plt.xlabel('Run Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Run Times')
        plt.grid(True)
        plt.savefig(f'histogram.svg')
        plt.close()
    
    def export_scores_to_txt(self, filename='passing_scores.txt'):
        self.data = self.data[~self.data['Method'].str.contains('gradient_', na=False)]
        if not self.data.empty:
            #columns to export
            columns_to_export = ['Method', 'abs score', 'Energies score', 'Time score', 'Total score', 'Osc. order score', 'Energy order score']
            # Filter the columns to ensure they exist in DataFrame to avoid KeyError
            existing_columns = [col for col in columns_to_export if col in self.data.columns]
            # Create a string representation of the DataFrame with the selected columns
            scores_str = self.data[existing_columns].to_string(index=False, header=True)
        
            with open(filename, 'w') as file:
                file.write(scores_str)
        
            print(f"Scores exported to {filename}.")
        else:
            print("No data available to export.")

    def save_csv(self):
        self.data.dropna(axis='columns', how='all').loc[:, (self.data != 0).any(axis=0)].to_csv(f'{self.pointname}_results.csv', index=False)
   
    def energy_RMSD_plot(self, output_dir='energy_score_plots'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot RMSD reference and candidates with y=x trace
        ene_array = np.array([self.data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
        norm_ene_array = sf.normalize_energy_array(ene_array)
        ref_norm_ene = norm_ene_array[self.ref_ind]
        
        if norm_ene_array.size == 0:
            print("Normalized energy array is empty. Check your data inputs.")
            return
        
        for i, method in enumerate(self.data['Method']):
            if i == self.ref_ind:
                continue # avoids plotting reference to itself
            
            candidate_norm_ene = norm_ene_array[i]
            rmsd = np.sqrt(np.mean((ref_norm_ene - candidate_norm_ene)**2))
            try:    
                plt.figure(dpi=300)
                plt.scatter(ref_norm_ene, candidate_norm_ene, label='Data')
                plt.plot([0, 1], [0, 1], color='red', label='y=x')
                plt.xlabel('EOM-CC2 Energies')
                plt.ylabel(f'{method} Energies')
                #plt.title(f'Linear Regression: Reference vs {method}')
                plt.text(0.05, 0.95, f'RMSD = {rmsd:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
                plot_filename = f'{self.pointname}_{method}_RMSD.svg'
                plt.savefig(plot_filename)
                plt.close()
                print(f"Saved RMSD plot for {method} in {output_dir}")

            except Exception as e:
                print(f"Failed to generate or save RMSD plot for {method}. Error: {e}")
                plt.close()

def main():
    args = read_single_arguments()
    fn = args.input_yaml
    settings = io.yload(fn)
    n_singlets = settings['reference']['singlets']
    n_triplets = settings['reference']['triplets']
    fol_name = fn.absolute().parents[0]
    charge = settings['general']['charge']
    geometry = settings['general']['coordinates']
    geom_file = fol_name / geometry
    mol_name = Path(geometry).stem
    print(f"I'm reading coordinates from {geometry} and will use {settings['general']['basis']} for all the calculations.")

    mol = Molecule.from_xyz(geometry)
    nelec = mol.nelectrons - charge
    results = SinglePointResults('S0min', n_singlets, n_triplets, fol_name)
    results.save_csv()
    data = pd.read_csv('S0min_RAW_results.csv', index_col=0)

    grader = Grader('S0min', data, n_singlets, n_triplets, 'EOM-CC2', settings)
    grader.append_score_columns_to_df()
    grader.data['Osc. order score'] = grader.singlet_osc_order_filter()
    grader.data['Energy order score'] = grader.s_t_order_filter()

    # Identify passed candidates
    if settings['grader']['order_filter']['singlet_triplet']:
        passed_candidates = grader.data[(grader.data['Osc. order score'] > n_singlets) & (grader.data['Energy order score'] > 0)]
    else:
        passed_candidates = grader.data[grader.data['Osc. order score'] > n_singlets]

    print(f"Passed candidates:\n{passed_candidates[['Method', 'Osc. order score', 'Energy order score']]}")

    failed_candidates = grader.data[~grader.data.index.isin(passed_candidates.index)]

    # Move failed candidates
    failed_cand_dir = fol_name / 'failed_candidates'
    failed_cand_dir.mkdir(exist_ok=True)
    for method in failed_candidates['Method'].unique():
        src_dir = fol_name / method
        dst_dir = failed_cand_dir / method
        if src_dir.exists():
            shutil.move(str(src_dir), str(dst_dir))
            print(f"Moved {method} to {failed_cand_dir}")

    log_files = []

    # Run gradient calculations first
    if not passed_candidates.empty:
        passed_methods = set(passed_candidates['Method'].unique())
        print(f"Passed methods for gradient calculation: {passed_methods}")

        if settings['candidates']:
            for calc_type in settings['candidates']:
                print(f"I will launch gradient calculations now for calc_type: {calc_type}")

                # Base settings for the calculation type
                vee_settings = {
                    'charge': charge,
                    'nelec': nelec,
                    'n_singlets': n_singlets,
                    'n_triplets': n_triplets,
                    'calc_type': calc_type,
                }

                # Merge settings specific to the calculation type from YAML
                case_settings = vee_settings | settings['candidates'][calc_type]

                # Generate a list of candidates using the updated settings
                candidates_list = CandidateListGenerator(**case_settings).create_candidate_list()
                print(f"Generated {len(candidates_list)} candidates for calc_type: {calc_type}")

                for candidate in candidates_list:
                    candidate_full_method = candidate.full_method.replace('__', '_')  # Fix double underscores

                    print(f"Checking candidate: {candidate.folder_name}, calc_type: {calc_type}, methods: {candidate_full_method}")

                    if candidate_full_method in passed_methods:
                        print(f"calc_type {calc_type} is in passed_methods.")
                        calc_settings = settings['general'] | candidate.calc_settings

                        # Construct the folder path with 'gradient' instead of 'energy'
                        folder_path = fol_name / f'gradient_{calc_type}{candidate.folder_name}'

                        if folder_path.exists():
                            print(f"Directory {folder_path} already exists. Using existing data.")
                        else:
                            calc_settings = settings['general'] | candidate.calc_settings
                            calc_settings['run'] = 'gradient'  # Adjust run setting for gradient calculations

                            # Launch the calculation
                            print(f"Launching gradient calculation for {candidate.folder_name} with settings: {calc_settings}")
                            launch_TCcalculation(folder_path, geom_file, calc_settings)

                        log_file = folder_path / 'tc.out'  # Assuming the log file is named 'tc.out'
                        log_files.append(str(log_file))

                        print(f"Launched gradient calculation for {candidate.folder_name} in {folder_path}")
                    else:
                        print(f"Skipping candidate: {candidate.folder_name}, calc_type: {calc_type}, methods: {candidate_full_method}")

        if log_files and not grader.wait_for_completion(log_files):
            print("Failed to complete all gradient calculations within the timeout.")
            return

        # Update times from gradient calculations
        grader.time_pen()

        # Recalculate total scores with updated time scores
        grader.data['Total score'] = grader.data['abs score'] + grader.data['Energies score'] + grader.data['Time score']
        grader.data.sort_values(by=['Total score'], ascending=False, inplace=True, ignore_index=True)
        grader.rep_cand_select()

    else:
        print("No valid data available to plot. Check filter conditions.")

    grader.export_scores_to_txt()
    grader.plot_results()
    grader.energy_RMSD_plot()
    grader.save_csv()

    print("Final DataFrame:", grader.data)

if __name__ == "__main__":
    main()

