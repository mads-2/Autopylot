from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from autopilot import read_single_arguments
import io_utils as io
from pathlib import Path
from scipy.integrate import simpson
import scoring_functions as sf
from results_collector import SinglePointResults
from candidate import CandidateListGenerator
from launcher import launch_TCcalculation
from molecule import Molecule
import os
import re
import shutil
import time

pd.set_option('display.max_columns', None)

@dataclass
class Grader:
    pointname: str
    data: pd.DataFrame
    n_singlets: int
    n_triplets: int
    ref_string: str
    settings: dict
    dft_methods_avail = ['wpbe', 'wB97x', 'wpbe', 'wpbeh', 'camb3lyp', 'rhf', 'b3lyp', 'pbe0', 'pbe', 'bhandhlyp', 'blyp', 'pw91', 'b3pw91', 'wb97']

    @property
    def state_list(self):
        return [f'S{i}' for i in range(self.n_singlets)] + [f'T{i+1}' for i in range(self.n_triplets)]

    @property
    def ref_ind(self):
        return sf.get_ref_df_index(self.data, self.ref_string)

    def gauss(self, x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def save_csv(self):
        self.data.dropna(axis='columns', how='all').loc[:, (self.data != 0).any(axis=0)].to_csv(f'{self.pointname}_results.csv', index=False)

    def extract_run_times(self, methods):
        run_times = {}
        cwd = os.getcwd()
        extract_time_pattern = re.compile(r'Total processing time:\s*([\d.]+)\s*sec')
        error_pattern = re.compile(r'terminated')
        
        for method in methods:
            folder_path = os.path.join(cwd, f"gradient_{method}")
            output_path = os.path.join(folder_path, 'tc.out')
            
            try:
                with open(output_path, 'r', encoding='utf-8') as file:
                    contents = file.read()
                    if error_pattern.search(contents):
                        print(f"Error detected in {output_path}. Skipping method {method}.")
                        run_times[method] = np.nan
                        continue

                    match = extract_time_pattern.search(contents)
                    if match:
                        run_times[method] = float(match.group(1))
                    else:
                        run_times[method] = np.nan
            except FileNotFoundError:
                run_times[method] = np.nan
        return run_times

    def time_pen(self):
        multiplier = self.settings['grader']['time_grader']['pen_mult_c']
        methods = self.data['Method'].unique()
        run_times = self.extract_run_times(methods)

        self.data['Run Time'] = self.data['Method'].map(run_times)

        valid_times = [time for time in run_times.values() if not np.isnan(time)]
        if valid_times:
            min_time = min(valid_times)
            max_time = max(valid_times)
            if min_time == max_time:
                time_score = {method: 1.0 for method in run_times.keys()}
            else:
                time_score = {method: (1 - ((time - min_time) / (max_time - min_time) * multiplier))
                              for method, time in run_times.items() if not np.isnan(time)}
        else:
            time_score = {method: np.nan for method in methods}

        self.data['Time score'] = self.data['Method'].map(time_score)
        self.data.loc[self.data['Method'] == 'EOM-CC2', 'Time score'] = 1
 
    def suggested_alpha(self):
        ene_array = np.array([self.data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
        ref_energies = ene_array[self.ref_ind]
        
        alpha_list = []

        for i, exc in enumerate(ene_array):
            alpha_per_state = [ref_energies[j] / exc[j] if exc[j] != 0 else 1 for j in range(len(exc))]
            alpha_list.append(alpha_per_state)

            print(f"Candidate {i}: Suggested Alphas = {alpha_per_state}")

        return np.array(alpha_list)

    def append_score_columns_to_df(self):
        methods = self.data['Method'].unique()
        self.data['AUC score'] = self.interval_auc_overlap()
        self.data['Average AUC Overlap'] = self.auc_overlap
        self.data['Total score'] = self.data['AUC score'] + self.data['Time score']
        self.data.sort_values(by=['Total score'], ascending=False, inplace=True, ignore_index=True)
        
        print("Method Scores:")
        print(self.data[['Method', 'AUC score','Average AUC Overlap', 'Time score', 'Run Time', 'Total score']])

    def interval_auc_overlap(self):
        min_energy = 0
        max_energy = 20 # defaults for when things go wrong

        for state in self.state_list:
            # Get min and max for each state's energy column
            min_energy = self.data[f'{state} energy'].min()
            max_energy = self.data[f'{state} energy'].max()

        #Extend the min and max range by ±2 eV
        energy_min = min_energy - 2  # Extend 2 eV below the minimum energy
        energy_max = max_energy + 2  # Extend 2 eV above the maximum energy

        E = np.linspace(energy_min, energy_max, 5000)  # Raw energy range
        sigma = 0.15  # HWHM

        # Generate intervals using np.arange with a step size of 0.1
        intervals = np.arange(energy_min, energy_max, 0.1)  # Raw energy intervals
        print(f"Intervals: {intervals}")

        # Apply suggested alpha for each candidate, including the reference
        suggested_alphas = self.suggested_alpha()

        # Build the reference spectrum (raw energy)
        ref_row = self.data.iloc[self.ref_ind]
        ref_spectrum = np.zeros_like(E)
        for state in range(1, self.n_singlets):
            energy_col = f'S{state} energy'.strip()
            osc_col = f'{energy_col.replace("energy", "osc.")}'.strip()
            osc_value = ref_row.get(osc_col)
            if pd.notna(ref_row[energy_col]) and pd.notna(osc_value):
                x0 = ref_row[energy_col]
                ref_spectrum += self.gauss(E, osc_value, x0, sigma)

        auc_overlap = []

        # Iterate through all candidates, including the reference
        for idx, row in self.data.iterrows():
            # Build the candidate spectrum
            cand_spectrum = np.zeros_like(E)

            for state in range(1, self.n_singlets):
                energy_col = f'S{state} energy'.strip()
                osc_col = f'{energy_col.replace("energy", "osc.")}'.strip()
                osc_value = row.get(osc_col)
                if pd.notna(row[energy_col]) and pd.notna(osc_value):
                    cand_spectrum += self.gauss(E, osc_value, row[energy_col], sigma)

            # Loop through alphas and apply them to shift the states
            alphas_for_candidates = suggested_alphas[idx]
            overlap_per_candidate = []

            # Apply each alpha, and shift all states
            for i, alpha in enumerate(alphas_for_candidates):
                spectrum = np.copy(cand_spectrum)  # Start with the original spectrum

                # Adjust energy of each state for this alpha
                for state in range(1, self.n_singlets):
                    energy_col = f'S{state} energy'.strip()
                    osc_col = f'{energy_col.replace("energy", "osc.")}'.strip()
                    osc_value = row.get(osc_col)
                    if pd.notna(row[energy_col]) and pd.notna(osc_value):
                        scaled_energy = row[energy_col] * alpha  # Apply the current alpha
                        spectrum += self.gauss(E, osc_value, scaled_energy, sigma) - self.gauss(E, osc_value, row[energy_col], sigma)

                # Calculate AUC difference for each interval
                interval_overlaps = []
                for j in range(len(intervals) - 1):
                    mask = (E >= intervals[j]) & (E < intervals[j + 1])
                    if np.sum(mask) > 1:  # Ensure there's more than one point
                        ref_spectrum_segment = ref_spectrum[mask]
                        spectrum_segment = spectrum[mask]
                        energy_segment = E[mask]

                        # Calculate overlap area for each interval
                        overlap_area = simpson(y=np.minimum(ref_spectrum_segment, spectrum_segment), x=energy_segment)
                        interval_overlaps.append(overlap_area)

                # Store the total overlap for this alpha
                overlap_total = sum(interval_overlaps)
                overlap_per_candidate.append(overlap_total)
                print(f"Candidate {idx}, Alpha {i + 1}: Total AUC overlap = {overlap_total}")

            if not hasattr(self, 'auc_overlap'):
                self.auc_overlap = []

            # Calculate the average AUC overlap across all alphas for the candidate
            average_overlap = np.mean(overlap_per_candidate)
            auc_overlap.append(average_overlap)
            print(f"Candidate {idx}: Average AUC overlap: {average_overlap}")

            self.auc_overlap.append(average_overlap)

        # Filter out the EOM-CC2 candidate for normalization
        non_eom_overlaps = [x for i, x in enumerate(auc_overlap) if self.data.iloc[i]['Method'] != 'EOM-CC2']
    
        if non_eom_overlaps:  # Ensure non-eom overlaps are not empty
            max_overlap = max(non_eom_overlaps)
            min_overlap = min(non_eom_overlaps)
        else:
            max_overlap = 1
            min_overlap = 0

        auc_scores_normalized = []
        for i, overlap in enumerate(auc_overlap):
            if self.data.iloc[i]['Method'] == 'EOM-CC2':
                auc_scores_normalized.append(1.0)  # Full score for EOM-CC2
            else:
                if max_overlap == min_overlap:
                    auc_scores_normalized.append(1.0)  # Avoid division by zero, assign full score
                else:
                    auc_scores_normalized.append((overlap - min_overlap) / (max_overlap - min_overlap))

        return pd.Series(auc_scores_normalized, index=self.data.index)

    def make_spectra(self):
        min_energy = 0 #Yes, this section is repeated instead of just globalized, sorry. 
        max_energy = 20 # defaults for when things go wrong 

        def gauss(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        sigma = 0.15  # HWHM

        # Create a directory to store the plots
        output_dir = os.path.join(os.getcwd(), f"{self.pointname}_UVVis_plots")
        os.makedirs(output_dir, exist_ok=True)

        # Apply suggested alpha for each candidate
        suggested_alphas = self.suggested_alpha()

        # Extract energies from reference
        ref_row = self.data[self.data['Method'] == 'EOM-CC2'].iloc[0]
        ref_spectrum = np.zeros(1000)
        ref_energies= []
        for state in range(1, self.n_singlets):
            energy_col = f'S{state} energy'.strip()
            osc_col = f'{energy_col.replace("energy", "osc.")}'.strip()
            osc_value = ref_row.get(osc_col)
            if pd.notna(ref_row[energy_col]) and pd.notna(osc_value):
                x0 = ref_row[energy_col]
                ref_energies.append(x0)

        if ref_energies:
            ref_min_energy = min(ref_energies)
            ref_max_energy = max(ref_energies)
        else: 
            ref_min_energy = min_energy
            ref_max_energy = max_energy
  
        # Process each candidate spectrum
        for idx, row in self.data.iterrows():
            # Skip the EOM-CC2 reference itself
            if row['Method'] == 'EOM-CC2' or 'gradient_' in row['Method']:
                continue

            # Get the suggested alphas for the current candidate
            alphas_for_candidates = suggested_alphas[idx]

            formatted_method_name = self.format_method_name(row['Method'])

            # Define a color cycle for the different alphas
            color_cycle = plt.cm.tab10(np.linspace(0, 1, len(alphas_for_candidates)))

            candidate_energies = []
            for state in range(1, self.n_singlets):
                state_energy = row[f'S{state} energy']
                if pd.notna(state_energy):
                    candidate_energies.append(state_energy)

            if candidate_energies:
                candidate_min_energy = min(candidate_energies)
                candidate_max_energy = max(candidate_energies)
            else:
                candidate_min_energy = min_energy
                candidate_max_energy = max_energy

            min_energy = min(ref_min_energy, candidate_min_energy) - 0.5 # +/- to leave room for the gaussians
            max_energy = max(ref_max_energy, candidate_max_energy) + 0.5 # +/- to leave room for the gaussians
                    
            E = np.linspace(min_energy, max_energy, 1000)

            #Now Y-axis Max
            max_y_val = 1 # default, check after spectrum for the calculation

            num_alphas = len(alphas_for_candidates)
            num_cols = int(np.ceil(np.sqrt(num_alphas)))  # Number of columns
            num_rows = int(np.ceil(num_alphas / num_cols))  # Number of rows
            fig, axs = plt.subplots(num_rows, num_cols, figsize=((num_cols * 4), (num_rows * 4)), dpi=300)
                    
            axs = axs.flatten() if num_alphas > 1 else [axs]
            
            # Apply each alpha and plot the spectrum
            for i, alpha in enumerate(alphas_for_candidates):
           
                ref_spectrum = np.zeros_like(E)
                for state in range(1, self.n_singlets):
                    energy_col = f'S{state} energy'.strip()
                    osc_col = f'{energy_col.replace("energy", "osc.")}'.strip()
                    osc_value = ref_row.get(osc_col)
                    if pd.notna(ref_row[energy_col]) and pd.notna(osc_value):
                        x0 = ref_row[energy_col]
                        ref_spectrum += gauss(E, osc_value, x0, sigma)

                axs[i].plot(E, ref_spectrum, label="EOM-CC2", linestyle='--', color='black')

                spectrum = np.zeros_like(E)
                for state in range(1, self.n_singlets):
                    energy_col = f'S{state} energy'.strip()
                    osc_col = f'{energy_col.replace("energy", "osc.")}'.strip()
                    if pd.notna(row[energy_col]) and pd.notna(row[osc_col]):
                        # Apply the suggested alpha to the candidate's energy
                        scaled_energy = row[energy_col] * alpha  # Shift energy by the current alpha
                        spectrum += gauss(E, row[osc_col], scaled_energy, sigma)
 
                aligned_energy = row[f'S{i} energy'] * alpha
                
                if i > 0:
                    axs[i].axvline(x=aligned_energy, color="black", label=f"Aligned to $S_{i}$ Energy")

                #Set new Y-axis max
                if alpha == 1:
                    max_y_value = np.max(spectrum)

                axs[i].plot(E, spectrum, label=f"{formatted_method_name} [$S_{i}$] [α = {alpha:.2f}]", color=color_cycle[i])
                axs[i].set_ylim(0, max_y_value)

                # Add labels, legend, and title
                axs[i].set_xlabel("Energy (eV)")
                axs[i].set_ylabel("Absorbance")
                axs[i].legend(loc='best', fontsize=6)
                axs[i].set_title(f"Reference vs {formatted_method_name} [$S_{i}$] [α = {alpha:.2f}]", fontsize=8)
                axs[i].grid(False)
        
            for ax in axs[num_alphas:]:
                fig.delaxes(ax)

            plt.tight_layout()

            # Save the plot
            plot_filename = os.path.join(output_dir, f'{self.pointname}_{row["Method"]}_UVVis_spectra.png')
            plt.savefig(plot_filename)
            plt.close()

        print(f"All plots saved in {output_dir}")

    def format_method_name(self, method_name):
        formatted_name = method_name

        # Handle FOMO formatting (e.g., fomo_T0.95 -> FOMO(t0=0.95))
        fomo_match = re.search(r'fomo_T(\d+\.\d+)', method_name, re.IGNORECASE)
        if fomo_match:
            t0_value = fomo_match.group(1)
            formatted_name = re.sub(r'fomo_T\d+\.\d+', f'FOMO(t₀ = {t0_value})', formatted_name)

        # Handle CASSCF active space formatting (e.g., AS86 -> (8,6))
        active_space_match = re.search(r'AS(\d+)', method_name)
        if active_space_match:
            active_space = active_space_match.group(1)
            if len(active_space) == 4:  # Handle two-digit active spaces like AS1210 -> (12,10)
                formatted_active_space = f'({active_space[:2]},{active_space[2:]})'
            elif len(active_space) == 2:  # Handle standard active spaces like AS86 -> (8,6)
                formatted_active_space = f'({active_space[0]},{active_space[1]})'
            else:
                formatted_active_space = f'[{active_space}]'
            # Replace "AS" part
            formatted_name = re.sub(r'AS\d+', f'CASSCF{formatted_active_space}', formatted_name)

        if "FOMO" in formatted_name and "CASSCF" in formatted_name:
            formatted_name = formatted_name.replace("CASSCF", "CASCI")

        # Remove redundant "casscf_" and "CASSCF_" prefixes
        formatted_name = re.sub(r'casscf_', '', formatted_name, flags=re.IGNORECASE)
        formatted_name = re.sub(r'CASSCF_CASSCF', 'CASSCF', formatted_name)

        formatted_name = formatted_name.replace('_', '-')

        # Handle `rc_w` formatting (e.g., rc_w = 0.15 -> ω = 0.15)
        rc_w_match = re.search(r'w(\d+.\d+)', method_name)
        if rc_w_match:
            rc_w_value = rc_w_match.group(1)
            formatted_name += f' (ω = {rc_w_value})'

        # Handle hhtda_fomo formatting (e.g., hhtda_fomo_wpbe_T0.15_w0.2 -> FOMO(t0=0.15)hhTDA_wPBE(ω = 0.2))
        hhtda_match = re.search(r'hhtda_fomo_(\w+)_T(\d+\.\d+)_w(\d+\.\d+)', method_name, re.IGNORECASE)
        if hhtda_match:
            functional = hhtda_match.group(1)
            t0_value = hhtda_match.group(2)
            omega_value = hhtda_match.group(3)
            formatted_name = f'FOMO(t₀ = {t0_value})hhTDA_{functional}(ω = {omega_value})'

        # Handle hhtda formatting (e.g., hhtda_wpbe_w0.2 -> hhTDA_wPBE(ω = 0.2))
        hhtda_match = re.search(r'hhtda_(\w+)_w(\d+\.\d+)', method_name, re.IGNORECASE)
        if hhtda_match:
            functional = hhtda_match.group(1)
            omega_value = hhtda_match.group(2)
            formatted_name = f'hhTDA_{functional}(ω = {omega_value})'

        return formatted_name

    def plot_results(self):
        # Filter out gradient-related calculations
        filtered_data = self.data[~self.data['Method'].str.contains('gradient_', na=False)].copy()

        # Ensure there's valid data to plot
        if not filtered_data.empty and 'Total score' in filtered_data.columns and not filtered_data['Total score'].isna().all():
            ene_array = np.array([filtered_data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
            norm_ene_array = sf.normalize_energy_array(ene_array)

            # Store the normalized energies in the DataFrame
            for i, state in enumerate(self.state_list):
                filtered_data[f'{state} energy'] = ene_array[:, i]
                filtered_data[f'{state} norm_energy'] = norm_ene_array[:, i]

            self.data.loc[filtered_data.index, :] = filtered_data

            cand_number = len(filtered_data)
            xs = list(filtered_data.index)

            filtered_data['Method'] = filtered_data['Method'].apply(self.format_method_name)
            methods = filtered_data['Method'].copy()

            # Find the reference index
            ref_idx = filtered_data[filtered_data['Method'] == 'EOM-CC2'].index[0]

            max_width = 150
            fig_width = min(cand_number * 2, max_width)
            fig_height = 24

            plt.rcParams.update({'font.size': 18})
            fig, [ax0, ax1, ax2, ax3] = plt.subplots(4, 1, figsize=[fig_width, fig_height], sharex=True)

            # Gold color for the reference in the grader axis (ax3)
            colors = [(0, 0, 1), (1, 0, 0)]  # For candidates
            cmap_name = 'grade_colormap'
            grade_colormap = LinearSegmentedColormap.from_list(cmap_name, colors)
        
            ref_color = 'gold'  # Reference gets a gold color

            color_map = cm.get_cmap('tab10', len(self.state_list))  # For ax0 and ax1
            colors = [color_map(i) for i in range(len(self.state_list))]

            # Plot energy and normalized energy for each state
            for i, state in enumerate(self.state_list):
                if 'T' in state:
                    linestyle = 'dotted' 
                    marker='D'
                    marker_size = 20
                    markerfacecolor = colors[i]
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
                    line_color = colors[i]
                    x_positions = [j - 0.3, j + 0.3]
                    y_value = filtered_data.loc[j, f'{state} energy']
                    ax0.plot(x_positions, [y_value, y_value], linestyle=style, marker=marker, color=line_color, linewidth=line_width, label=f'${self.state_list[i].replace("T", "T_").replace("S", "S_")}$' if j == xs[0] else "")
                    
                    y_value_norm = filtered_data.loc[j, f'{state} norm_energy']
                    ax1.plot(x_positions, [y_value_norm, y_value_norm], linestyle=style, color=line_color, linewidth=line_width)

            # Plot total scores as a bar graph on ax3
            rescaled_scores = sf.rescale(filtered_data['Total score'])
            colors = [ref_color if i == ref_idx else grade_colormap(rescaled_scores[i]) for i in range(len(xs))]
            bars = ax3.bar(xs, filtered_data['Total score'], color=colors)

            for bar, score in zip(bars, filtered_data['Total score']):
                bar_height = bar.get_height()

                # Check if the bar is too small for the font size
                if bar_height < 0.5:  # Threshold for small bars (you can adjust this)
                    text_y_position = bar.get_y() + bar_height + 0.1  # Place text above the bar
                    ax3.text(bar.get_x() + bar.get_width() / 2, text_y_position, f'{score:.2f}', 
                    ha='center', va='bottom', color='black', fontsize=32)
                else:
                    # Default behavior for larger bars
                    ax3.text(bar.get_x() + bar.get_width() / 2, bar_height / 2, f'{score:.2f}', 
                    ha='center', va='center', color='black', fontsize=32,  rotation=90)

            # Plotting component scores and total score on ax2
            ax2.plot(xs, filtered_data['AUC score'], color='r', label='AUC Score', linewidth=5, marker='8', markersize=20)
            ax2.plot(xs, filtered_data['Time score'], color='b', label='Time Score', linewidth=5, marker='8', markersize=20)
            ax2.plot(xs, filtered_data['Total score'], color='black', label='Total Score', linewidth=5, marker='8', markersize=20)

            # Add a black vertical line to separate reference from candidates
            ax0.axvline(x=ref_idx + 0.5, color='black', linewidth=4, linestyle='--')
            ax1.axvline(x=ref_idx + 0.5, color='black', linewidth=4, linestyle='--')
            ax2.axvline(x=ref_idx + 0.5, color='black', linewidth=4, linestyle='--')
            ax3.axvline(x=ref_idx + 0.5, color='black', linewidth=4, linestyle='--')

            ax0.set_ylabel('Energy [eV]')
            ax1.set_ylabel('Normalized Energy [eV]')
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
            fig.savefig(f'{self.pointname}_results.png', dpi=300, bbox_inches='tight')
        else:
            print("No valid data available to plot.")

    def AUC_histogram(self):
        # Get the AUC scores and filter out NaN values
        auc_scores = self.data['AUC score'].dropna()

        # Filter the data again to match any relevant criteria
        filtered_data = self.data[~self.data['Method'].str.contains('gradient_', na=False)]
        matching_indices = filtered_data.index.intersection(auc_scores.index)
        filtered_auc_scores = auc_scores[matching_indices].dropna()

        if filtered_auc_scores.empty:
            print("No valid AUC scores available to plot.")
            return

        plt.figure(dpi=300)
        plt.hist(filtered_auc_scores, bins=20, color='green', alpha=0.7)
        plt.xlabel('AUC Score')
        plt.ylabel('Frequency')
        plt.title('Histogram of AUC Scores')
        plt.grid(False)
        plt.savefig(f'{self.pointname}_AUC_histogram.png', dpi=300)
        plt.close()

    def export_scores_to_txt(self, filename='Final_Scores.txt'):
        self.data = self.data[~self.data['Method'].str.contains('gradient_', na=False)]
        if not self.data.empty:
            self.data['Method'] = self.data['Method'].apply(self.format_method_name)

            # Columns to export
            columns_to_export = ['Method', 'AUC score', 'Average AUC Overlap', 'Time score', 'Run Time', 'Total score']
            # Filter the columns to ensure they exist in DataFrame to avoid KeyError
            existing_columns = [col for col in columns_to_export if col in self.data.columns]
            # Create a string representation of the DataFrame with the selected columns
            scores_str = self.data[existing_columns].to_string(index=False, header=True)

            with open(filename, 'w') as file:
                file.write(scores_str)

            print(f"Scores exported to {filename}.")
        else:
            print("No data available to export.")

def main():
    args = read_single_arguments()
    fn = args.input_yaml
    settings = io.yload(fn)
    n_singlets = settings['reference']['singlets']
    n_triplets = settings['reference'].get('triplets', 0)  # Default to 0 if not defined
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

    grader = Grader(
        pointname='S0min',
        data=data,
        n_singlets=n_singlets,
        n_triplets=n_triplets,
        ref_string='EOM-CC2',
        settings=settings
    )

    log_files = []

    grader.time_pen()  # Calculate the time scores
    grader.append_score_columns_to_df()  # Compute AUC score, Total score, etc.
    grader.make_spectra()
    grader.AUC_histogram()
    grader.export_scores_to_txt()
    grader.save_csv()
    grader.plot_results()

    print("Final DataFrame:", grader.data)

if __name__ == "__main__":
    main()
