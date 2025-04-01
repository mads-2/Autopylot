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
import yaml
import sys
import subprocess

pd.set_option('display.max_columns', None)

@dataclass
class Grader:
    pointname: str
    data: pd.DataFrame
    n_singlets: int
    ref_string: str
    settings: dict
    #dft_methods_avail = ['wB97x', 'wpbe', 'wpbeh', 'camb3lyp', 'rhf', 'b3lyp', 'pbe0', 'pbe', 'bhandhlyp', 'blyp', 'pw91', 'b3pw91', 'wb97']
    
    @property
    def state_list(self):
        return [f'S{i}' for i in range(self.n_singlets)]

    @property
    def ref_ind(self):
        return sf.get_ref_df_index(self.data, self.ref_string)

    def gauss(self, x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def save_csv(self):
        try:
            # Check if `self.data` is empty
            if self.data is None or self.data.empty:
                print("Warning: `self.data` is empty. No CSV will be saved.")
                return
        
            # Drop empty columns and ensure only non-zero columns are kept
            filtered_data = self.data.dropna(axis='columns', how='all').loc[:, (self.data != 0).any(axis=0)]
        
            # Ensure there's data to save after filtering
            if filtered_data.empty:
                print("Warning: Filtered data is empty. No CSV will be saved.")
                return
        
            # Save the CSV file
            filtered_data.to_csv(f'{self.pointname}_results.csv', index=False)
            print(f"CSV saved successfully as {self.pointname}_results.csv \n")
    
        except Exception as e:
            print(f"Error in `save_csv`: {e}")
            raise

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
        multiplier = self.settings['grader']['time_grader']['time_weight']
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
                time_score = {method: (1 - ((time - min_time) / (max_time - min_time))) * multiplier
                              for method, time in run_times.items() if not np.isnan(time)}
        else:
            time_score = {method: np.nan for method in methods}

        self.data['Time Component'] = self.data['Method'].map(time_score)
        self.data.loc[self.data['Method'] == 'EOM-CC2', 'Time Component'] = multiplier

    def suggested_alpha(self):
        ene_array = np.array([self.data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
        ref_energies = ene_array[self.ref_ind]
        
        alpha_list = []

        for i, exc in enumerate(ene_array):
            alpha_per_state = [ref_energies[j] / exc[j] if exc[j] != 0 else 1 for j in range(len(exc))]
            alpha_list.append(alpha_per_state)

            print(f"Candidate {i}: Suggested Alphas = {alpha_per_state}")
            print()

        print()
        return np.array(alpha_list)

    def append_score_columns_to_df(self):
        methods = self.data[~self.data['Method'].str.contains('gradient_', na=False)]
        self.data['Normalized Autopylot Component'] = np.nan
        self.data['Overlap'] = np.nan
        self.data['Spectra Penalty'] = np.nan
        self.data['RMSD Penalty'] = np.nan

        self.data.loc[methods.index, 'Normalized Autopylot Component'] = self.interval_auc_overlap()
        self.data.loc[methods.index,'Raw Autopylot Score'] = self.auc_overlap
        self.data.loc[methods.index, 'Overlap'] = self.best_overlaps

        print(f"Length of methods.index: {len(methods.index)}")
        print(f"Length of self.abs_penalties: {len(self.abs_penalties)}")
        print(f"Length of self.e_penalties: {len(self.e_penalties)}")

        self.data.loc[methods.index, 'Spectra Penalty'] = self.abs_penalties
        self.data.loc[methods.index, 'Rel. Energy Penalty'] = self.e_penalties
        
        self.data['Parent'] = self.data['Method'].str.replace('gradient','', regex=False)
        combine_scores = self.data.groupby('Parent').agg({
            'Normalized Autopylot Component': 'max',
            'Time Component': 'max',
        }).reset_index()

        combine_scores['Final Score'] = combine_scores['Normalized Autopylot Component'].fillna(0) + combine_scores['Time Component'].fillna(0)
        self.data = self.data.merge(combine_scores, on='Parent', how='left', suffixes=('',' '))
        self.data.sort_values(by=['Final Score'], ascending=False, inplace=True, ignore_index=True)
        
        print("Method Scores:")
        print(self.data[['Method', 'Normalized Autopylot Component', 'Raw Autopylot Score','Overlap', 'Spectra Penalty', 'Rel. Energy Penalty', 'Time Component', 'Run Time', 'Final Score']])
        print()

    def interval_auc_overlap(self):
        # Initialize auc_overlap as a class attribute
        self.auc_overlap = []

        methods= self.data[~self.data['Method'].str.contains('gradient_', na=False)]

        if methods.empty:
            print("Missing energies, probalby still counting gradient calculations when it shouldn't")
            return pd.Series([np.nan] * len(self.data),index=self.data.index)

        excited_states = sorted([s for s in self.state_list if s != 'S0'], key=lambda x: int(x[1:]))

        # Select the first excited state for min_energy
        if excited_states:
            first_excited = excited_states[0]
            min_energy = methods[f'{first_excited} energy'].min()

        # Select the highest energy state for max_energy
        if excited_states:
            highest_excited = excited_states[-1]
            max_energy = methods[f'{highest_excited} energy'].max()

        print()
        print(f"Min Energy: {min_energy}")
        print(f"Max Energy: {max_energy}\n")

        # Extend the min and max range by ±2 eV
        energy_min = min_energy - 2
        energy_max = max_energy + 2

        E = np.linspace(energy_min, energy_max, 5000)  # Raw energy range
        sigma = 0.15  # HWHM

        intervals = np.arange(energy_min, energy_max, 0.1)  # Raw energy intervals

        # Apply suggested alpha for each candidate, including the reference
        suggested_alphas = self.suggested_alpha()

        # Build the reference spectrum
        ref_row = methods.iloc[self.ref_ind]
        ref_spectrum = np.zeros_like(E)
        for state in range(1, self.n_singlets):
            energy_col = f'S{state} energy'
            osc_col = f'{energy_col.replace("energy", "osc.")}'
            if pd.notna(ref_row[energy_col]) and pd.notna(ref_row[osc_col]):
                x0 = ref_row[energy_col]
                osc_value = ref_row[osc_col]
                ref_spectrum += self.gauss(E, osc_value, x0, sigma)

        rmsd_scores, _ = self.RMSD_pen()
        
        self.best_overlaps = []
        self.abs_penalties = []
        self.e_penalties = []

        # Iterate through all candidates, including the reference
        for idx, (i, row) in enumerate(self.data.iterrows()):
            overlap_per_candidate = []
            all_interval_differences = []

            subint_log = False

            # Build the candidate spectrum
            cand_spectrum = np.zeros_like(E)
            for state in range(1, self.n_singlets):
                energy_col = f'S{state} energy'
                osc_col = f'{energy_col.replace("energy", "osc.")}'
                if pd.notna(row[energy_col]) and pd.notna(row[osc_col]):
                    x0 = row[energy_col]
                    osc_value = row[osc_col]
                    cand_spectrum += self.gauss(E, osc_value, x0, sigma)

            alphas_for_candidates = suggested_alphas[i]
            
            # Calculate total overlaps and AUC differences for each alpha
            for alpha in alphas_for_candidates:
                shifted_spectrum = np.copy(cand_spectrum)

                # Shift all states by the current alpha
                for state in range(1, self.n_singlets):
                    energy_col = f'S{state} energy'
                    osc_col = f'{energy_col.replace("energy", "osc.")}'
                    if pd.notna(row[energy_col]) and pd.notna(row[osc_col]):
                        scaled_energy = row[energy_col] * alpha
                        shifted_spectrum += (
                            self.gauss(E, row[osc_col], scaled_energy, sigma)
                            - self.gauss(E, row[osc_col], row[energy_col], sigma)
                        )

                interval_overlaps = []
                interval_differences = []

                for j in range(len(intervals) - 1):
                    mask = (E >= intervals[j]) & (E < intervals[j + 1])
                    if np.sum(mask) > 1:  # Ensure there are valid points
                        ref_segment = ref_spectrum[mask]
                        cand_segment = shifted_spectrum[mask]
                        energy_segment = E[mask]

                        subint_E_size = np.diff(energy_segment)[0]
                        num_int = len(np.diff(energy_segment))

                        if not subint_log: 
                            print("\nSimpson's Rule Subinterval Calculator:")
                            print(f"Candidate: {row['Method']} Interval: 0.1")
                            print(f"Number of subintervals:{num_int}")
                            print(f"Subinterval size: {subint_E_size:.6f}\n")
                            subint_log = True

                        # Calculate AUC for the reference and candidate within this interval
                        auc_ref = simpson(y=ref_segment, x=energy_segment)
                        auc_cand = simpson(y=cand_segment, x=energy_segment)

                        # Compute the raw overlap
                        raw_overlap = simpson(y=np.minimum(ref_segment, cand_segment), x=energy_segment)
                        interval_overlaps.append(raw_overlap)

                        # Compute the intensity difference for the penalty
                        intensity_difference = abs(auc_ref - auc_cand)
                        interval_differences.append(intensity_difference)

                # Store total overlap and intensity differences for this alpha
                overlap_total = sum(interval_overlaps)
                overlap_per_candidate.append(overlap_total)
                all_interval_differences.append(interval_differences)

            max_overlap_idx = np.argmax(overlap_per_candidate[1:]) + 1
            best_overlap = overlap_per_candidate[max_overlap_idx]
            best_alpha = alphas_for_candidates[max_overlap_idx]
            print(f"Best alpha for {row['Method']}: S{max_overlap_idx}, Alpha: {best_alpha}, Overlap: {best_overlap}\n")

            best_intensity_differences = all_interval_differences[max_overlap_idx]

            # Calculate the raw penalty at the best alpha
            abs_penalty = sum(best_intensity_differences)
            print(f"\nTotal Intensity Difference for {row['Method']} (Best Alpha: {best_alpha}): {abs_penalty:.6f}")

            e_penalty = rmsd_scores[i]

            self.best_overlaps.append(best_overlap)
            self.abs_penalties.append(abs_penalty)
            self.e_penalties.append(e_penalty)

            # Adjust the average overlap by subtracting the penalty
            adjusted_score = (best_overlap - abs_penalty) - e_penalty
            self.auc_overlap.append(adjusted_score)  # Append adjusted score to the class attribute
        
        # Normalize the scores
        all_overlaps = [
            x for i, x in enumerate(self.auc_overlap) if methods.iloc[i]["Method"]]
        max_overlap = max(all_overlaps, default=1)
        min_overlap = min(all_overlaps, default=0)

        auc_scores_normalized = []
        for i, overlap in enumerate(self.auc_overlap):
            if self.data.iloc[i]["Method"] == "EOM-CC2":
                auc_scores_normalized.append(1.0)
            else:
                if max_overlap == min_overlap:
                    auc_scores_normalized.append(1.0)  # Avoid division by zero
                else:
                    auc_scores_normalized.append((overlap - min_overlap) / (max_overlap - min_overlap))

        return pd.Series(auc_scores_normalized, index=self.data.index)

    def RMSD_pen(self):
        ene_array = np.array([self.data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
        norm_ene_array = sf.normalize_energy_array(ene_array)
        
        ref_norm_ene = norm_ene_array[self.ref_ind]
        rmsd_scores = np.sqrt(np.mean((ref_norm_ene - norm_ene_array) ** 2, axis=1))

        print("\nRMSD Scores:")
        for i, method in enumerate(self.data['Method']):
            print(f"{method}: {rmsd_scores[i]:.4f}\n")

        return rmsd_scores, norm_ene_array

    def energy_RMSD_plot(self, image_type, output_dir='energy_score_plots'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        rmsd_scores, norm_ene_array = self.RMSD_pen()
        ref_norm_ene = norm_ene_array[self.ref_ind]

        if norm_ene_array.size == 0:
            print("Normalized energy array is empty. Check your data inputs.")
            return

        for i, method in enumerate(self.data['Method']):
            if i == self.ref_ind:
                continue # avoids plotting reference to itself

            try:
                formatted_method_name = self.format_method_name(method)
                plt.figure(dpi=300)
                plt.scatter(ref_norm_ene, norm_ene_array[i], label='Data')
                plt.plot([0, 1], [0, 1], color='red', label='y=x')
                plt.xlabel('EOM-CC2 Energies')
                plt.ylabel(f'{formatted_method_name} Energies')
                #plt.title(f'Linear Regression: Reference vs {method}')
                plt.text(0.05, 0.95, f'RMSD = {rmsd_scores[i]:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

                plot_filename = f'{self.pointname}_{method}_RMSD.{image_type}'
                plt.savefig(os.path.join(output_dir, plot_filename))
                plt.close()
                print(f"Saved RMSD plot for {method} in {output_dir} \n")

            except Exception as e:
                print(f"Failed to generate or save RMSD plot for {method}. Error: {e}")
                plt.close()

    def make_spectra(self, image_type):
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
            # Skip the over gradients, no spectra from them
            if row['Method'] == 'gradient_' in row['Method']:
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

                if alpha == 1:
                    max_y_value = max(np.max(spectrum), np.max(ref_spectrum))

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
            plot_filename = os.path.join(output_dir, f'{self.pointname}_{row["Method"]}_UVVis_spectra.{image_type}')
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

        formatted_name = re.sub(r'fomo', 'FOMO', formatted_name, flags=re.IGNORECASE)

        # Handle CAS active space formatting (e.g., AS86 -> (8,6))
        active_space_match = re.search(r'AS(\d+)', method_name)
        if active_space_match:
            active_space = active_space_match.group(1)
            
            if len(active_space) == 4:  # Handle two-digit active spaces like AS1210 -> (12,10)
                formatted_active_space = f'({active_space[:2]},{active_space[2:]})'
            
            elif len(active_space) == 3:
                if active_space[0] in ['1', '2']:
                    first_part = int(active_space[:2])  #first two digits
                    second_part = int(active_space[2:])  # last digit
                else:
                    first_part = int(active_space[:1])  # first digit
                    second_part = int(active_space[1:])  # last two digits
                formatted_active_space = f'({first_part},{second_part})'
            
            elif len(active_space) == 2:  # Handle standard active spaces like AS86 -> (8,6)
                formatted_active_space = f'({active_space[0]},{active_space[1]})'
            
            else:
                formatted_active_space = f'[{active_space}]'
            
            if "casci" in method_name.lower():
                formatted_name = re.sub(r'AS\d+', f'CASCI{formatted_active_space}', formatted_name)
            else:
                formatted_name = re.sub(r'AS\d+', f'CASSCF{formatted_active_space}', formatted_name)

        if "FOMO" in formatted_name and "CASSCF" in formatted_name:
            formatted_name = formatted_name.replace("CASSCF", "CASCI")

        formatted_name = re.sub(r'casscf_', '', formatted_name, flags=re.IGNORECASE)
        formatted_name = re.sub(r'casci_', '', formatted_name, flags=re.IGNORECASE)
        formatted_name = re.sub(r'CASSCF_CASSCF', 'CASSCF', formatted_name)

        formatted_name = formatted_name.replace('_', '-')

        rc_w_match = re.search(r'-w(\d+\.\d+|\d+)', formatted_name)
        rc_w_value = rc_w_match.group(1) if rc_w_match else None

        # Handle hhTDA functional formatting
        hhtda_match = re.search(r'hhtda-(\w+)(?:-T(\d+\.\d+))?(?:-w(\d+\.\d+))?', formatted_name, re.IGNORECASE)
        if hhtda_match:
            functional = hhtda_match.group(1)
            t0_value = hhtda_match.group(2)
            omega_value = hhtda_match.group(3)

            formatted_name = f"hhTDA-{functional}"

            if t0_value:
                formatted_name = f'(t₀ = {t0_value})' + formatted_name

            if omega_value:
                formatted_name += f'(ω = {omega_value})'

        # Handle `rc_w` formatting separately if not already processed
        if rc_w_value and '(ω =' not in formatted_name:
            formatted_name = re.sub(r'-w' + re.escape(rc_w_value), f'(ω = {rc_w_value})', formatted_name)

        formatted_name = re.sub(r'\bhhtda\b', 'hhTDA', formatted_name, flags=re.IGNORECASE)

        return formatted_name

    def reverse_format_method_name(self, formatted_name):
        reversed_name = formatted_name

        t0_value = None
        omega_value = None

        # Handle FOMO formatting (e.g., FOMO(t₀ = 0.55) -> fomo_T0.55)
        fomo_match = re.search(r'FOMO\(t₀\s*=\s*([\d.]+)\)', formatted_name, re.IGNORECASE)
        if fomo_match:
            t0_value = fomo_match.group(1)
            reversed_name = re.sub(r'FOMO\(t₀\s*=\s*[\d.]+\)', '', reversed_name).strip()

        # Handle CAS active space formatting (e.g., CASCI(8,7) -> casci_AS87)
        casci_match = re.search(r'CASCI\((\d+),(\d+)\)', reversed_name, re.IGNORECASE)
        if casci_match:
            active_space = f"AS{casci_match.group(1)}{casci_match.group(2)}"
            if t0_value:
                reversed_name = re.sub(r'CASCI\(\d+,\d+\)', f'casci_fomo_T{t0_value}_{active_space}', reversed_name)
            else:
                reversed_name = re.sub(r'CASCI\(\d+,\d+\)', f'casci_{active_space}', reversed_name)

        casscf_match = re.search(r'CASSCF\((\d+),(\d+)\)', reversed_name, re.IGNORECASE)
        if casscf_match:
            active_space = f"AS{casscf_match.group(1)}{casscf_match.group(2)}"
            reversed_name = re.sub(r'CASSCF\(\d+,\d+\)', f'casscf_{active_space}', reversed_name)

        # Restore original underscores for consistency with the input naming
        reversed_name = reversed_name.replace('-', '_')

        # Handle `ω` (omega) value extraction
        rc_w_match = re.search(r'\(ω\s*=\s*([\d.]+)\)', reversed_name, re.IGNORECASE)
        if rc_w_match:
            omega_value = rc_w_match.group(1)
            reversed_name = re.sub(r'\(ω\s*=\s*[\d.]+\)', '', reversed_name).strip()

        # Handle hhTDA functional formatting (e.g., hhTDA-wPBE -> hhtda_wpbe)
        hhtda_match = re.search(r'hhTDA-(\w+)', reversed_name, re.IGNORECASE)
        if hhtda_match:
            functional = hhtda_match.group(1).lower()  # Convert back to lowercase
            if t0_value and omega_value:
                reversed_name = re.sub(r'hhTDA-\w+', f'hhtda_{functional}_T{t0_value}_w{omega_value}', reversed_name)
            elif t0_value:
                reversed_name = re.sub(r'hhTDA-\w+', f'hhtda_{functional}_T{t0_value}', reversed_name)
            elif omega_value:
                reversed_name = re.sub(r'hhTDA-\w+', f'hhtda_{functional}_w{omega_value}', reversed_name)
            else:
                reversed_name = re.sub(r'hhTDA-\w+', f'hhtda_{functional}', reversed_name)

        return reversed_name

    def plot_results(self, image_type):
        top_scores = self.settings['visuals'].get('top_scores', len(self.data)) + 1
        # Filter out gradient-related calculations
        filtered_data = self.data[~self.data['Method'].str.contains('gradient_', na=False)].copy()

        # Ensure there's valid data to plot
        if not filtered_data.empty and 'Final Score' in filtered_data.columns and not filtered_data['Final Score'].isna().all():
            filtered_data = filtered_data.sort_values(by='Final Score', ascending=False).head(top_scores)

            reference_method = 'EOM-CC2'
            if reference_method in filtered_data['Method'].values:
                # Separate reference method and other methods
                ref_data = filtered_data[filtered_data['Method'] == reference_method]
                other_data = filtered_data[filtered_data['Method'] != reference_method]

                # Concatenate with reference first
                filtered_data = pd.concat([ref_data, other_data])
                filtered_data.reset_index(drop=True, inplace=True)  # Reset indices

            # Update ref_idx to 0
            ref_idx = 0
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

            plt.rcParams.update({'font.size': 18})
        
            istime_zero= (filtered_data['Time Component'].fillna(0) == 0).all()

            num_plots = 3 if istime_zero else 4
            fig, axes = plt.subplots(num_plots, 1, figsize=[min(len(xs) * 2, 150), 24], sharex=True)

            ax0 = axes[0]
            ax1 = axes[1]
            ax2 = None
            ax3 = None

            if istime_zero:
                ax3 = axes[2]  # Only 3 subplots
            else:
                ax2 = axes[2]
                ax3 = axes[3]  # 4th plot when time isn't zero

            # Gold color for the reference in the grader axis (ax3)
            colors = [(0, 0, 1), (1, 0, 0)]  # For candidates
            cmap_name = 'grade_colormap'
            grade_colormap = LinearSegmentedColormap.from_list(cmap_name, colors)
        
            ref_color = 'gold'  # Reference gets a gold color
            
            nstates = len(self.state_list) - 1 #exclude S0
            tab20 = cm.get_cmap('tab20', nstates)
            tab20b = cm.get_cmap('tab20b_r', nstates)
            alt = []
            for i in range(nstates):
                cmap = tab20 if i % 2 == 0 else tab20b
                alt.append(cmap(i))

            colors = ['black'] + ['red'] + ['blue'] + ['magenta'] + ['orange'] + ['purple'] + ['green'] + ['brown'] + ['dimgrey'] + ['cyan'] + ['navy'] + alt

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
                    bright = self.settings['visuals']['countas_bright']
                    dark = state != 'S0' and filtered_data.loc[j, f'{state} osc.'] < bright and 'S' in state
                    style = 'dotted' if dark else linestyle
                    line_color = colors[i]
                    x_positions = [j - 0.3, j + 0.3]
                    y_value = filtered_data.loc[j, f'{state} energy']
                    ax0.plot(x_positions, [y_value, y_value], linestyle=style, marker=marker, color=line_color, linewidth=line_width, label=f'${self.state_list[i].replace("T", "T_{{").replace("S", "S_{{")}}}$'.replace("_{{", "_{").replace("}}", "}") if j == xs[0] else "")
                    y_value_norm = filtered_data.loc[j, f'{state} norm_energy']
                    ax1.plot(x_positions, [y_value_norm, y_value_norm], linestyle=style, color=line_color, linewidth=line_width)

            # Plot total scores as a bar graph on ax3
            rescaled_scores = sf.rescale(filtered_data['Final Score'])
            colors = [ref_color if i == ref_idx else grade_colormap(rescaled_scores[i]) for i in range(len(xs))]
            bars = ax3.bar(xs, filtered_data['Final Score'], color=colors)

            for bar, score in zip(bars, filtered_data['Final Score']):
                bar_height = bar.get_height()

                # Check if the bar is too small for the font size
                if bar_height < 0.3:  # Threshold for small bars (you can adjust this)
                    text_y_position = bar.get_y() + bar_height + 0.1  # Place text above the bar
                    ax3.text(bar.get_x() + bar.get_width() / 2, text_y_position, f'{score:.2f}', 
                    ha='center', va='bottom', color='black', fontsize=32)
                else:
                    # Default behavior for larger bars
                    ax3.text(bar.get_x() + bar.get_width() / 2, bar_height / 2, f'{score:.2f}', 
                    ha='center', va='center', color='black', fontsize=32,  rotation=90)

            # Plotting component scores and total score on ax2i
            if not istime_zero: 
                ax2.plot(xs, filtered_data['Normalized Autopylot Component'], color='r', label='Normalized Autopylot Component', linewidth=5, marker='8', markersize=20)
                ax2.plot(xs, filtered_data['Time Component'], color='b', label='Time Component', linewidth=5, marker='8', markersize=20)
                #ax2.plot(xs, filtered_data['Total score'], color='black', label='Total Score', linewidth=5, marker='8', markersize=20)
                ax2.set_ylabel('Component Scores')
                ax2.legend()


            # Add a black vertical line to separate reference from candidates
            for ax in [ax0, ax1, ax2, ax3] if not istime_zero else [ax0, ax1, ax3]:
                ax.axvline(x=ref_idx + 0.5, color='black', linewidth=4, linestyle='--')

            ax0.set_ylabel('Energy [eV]')
            ax1.set_ylabel('Normalized Energy [eV]')
            ax3.set_ylabel('Score')
            ax3.set_xticks(xs)
            ax3.set_xlim(-1.0, cand_number)
            ax3.set_ylim(0.0, max(filtered_data['Final Score']))
            ax3.set_xticklabels(methods, rotation=90)
            ax0.legend(fontsize=20, loc='upper left', bbox_to_anchor=(1.01, 1.0))

            fig.tight_layout(pad=3.0)
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0, wspace=0.5)
            fig.savefig(f'{self.pointname}_results.{image_type}', dpi=300, bbox_inches='tight')
        else:
            print("No valid data available to plot.")

    def AUC_histogram(self, image_type):
        # Get the AUC scores and filter out NaN values
        auc_scores = self.data['Normalized Autopylot Component'].dropna()

        # Filter the data again to match any relevant criteria
        filtered_data = self.data[~self.data['Method'].str.contains('gradient_', na=False)]
        matching_indices = filtered_data.index.intersection(auc_scores.index)
        filtered_auc_scores = auc_scores[matching_indices].dropna()

        if filtered_auc_scores.empty:
            print("No valid AUC scores available to plot.")
            return

        plt.figure(dpi=300)
        plt.hist(filtered_auc_scores, bins=20, color='green', alpha=0.7)
        plt.xlabel('Normalized Autopylot Component')
        plt.ylabel('Frequency')
        plt.title('Histogram of Noramlized Autopylot Component Values')
        plt.grid(False)
        plt.savefig(f'{self.pointname}_overlap_histogram.{image_type}', dpi=300)
        plt.close()

    def export_scores_to_txt(self, filename='Final_Scores.txt'):
        self.data = self.data[~self.data['Method'].str.contains('gradient_', na=False)]
        if not self.data.empty:
            formatted_data = self.data.copy()
            formatted_data['Method'] = formatted_data['Method'].apply(self.format_method_name)
            print(formatted_data[['Method']])

            # Columns to export
            columns_to_export = ['Method', 'Normalized Autopylot Component', 'Raw Autopylot Score','Overlap', 'Spectra Penalty', 'Rel. Energy Penalty', 'Time Component', 'Run Time', 'Final Score']
            # Filter the columns to ensure they exist in DataFrame to avoid KeyError
            #existing_columns = [col for col in columns_to_export if col in formatted_data.columns]
            formatted_data[columns_to_export] = formatted_data[columns_to_export].round(4)

            # Create a string representation of the DataFrame with the selected columns
            scores_str = formatted_data[columns_to_export].astype(str).to_string(index=False, header=True)

            with open(filename, 'w') as file:
                file.write(scores_str)

            print(f"\n Scores exported to {filename}.")
        else:
            print("No data available to export.")

    def bright_state_optimization_autopilot(self, new_dir, input_yaml, geom_file):
        settings = self.settings

        if settings['grader']['excited_state'].get('bright_opt', 'no').lower() != 'yes':
            print("\n Skipping bright state optimization")
            return

        os.makedirs(new_dir, exist_ok=True)

        candidates_only = self.data[self.data['Method'] != 'EOM-CC2']
        ref_row = self.data[self.data['Method'] == 'EOM-CC2'].iloc[0]

        filtered_data = self.data[self.data['Method'].isin(self.data['Method'])]
        if filtered_data.empty or 'Final Score' not in filtered_data.columns:
            print("No scores available to determine the highest-scoring method. Skipping bright state optimization.")
            return

        # Identify the highest-scoring method
        top_Cand_row = candidates_only.sort_values(by='Final Score', ascending=False).iloc[0]
        top_Cand = self.reverse_format_method_name(top_Cand_row['Method'])
        print(f"Highest-scoring method selected for 1st bright state optimization: {top_Cand}")
        
        # Find the bright state target, options lowest E bright state or all bright states
        what_opt = self.settings['grader']['excited_state'].get('opt_state_target', 'lowest').lower()
        if what_opt not in ["lowest","all"]:
            what_opt = "all"

        #print(f"DEBUG: Using what_opt = {what_opt}")

        bright_thresh = self.settings['visuals']['countas_bright']
        bright_targets = []

        print(f"States List = {self.state_list}")

        for i, state in enumerate(self.state_list[1:], start=1):
            osc_strength = ref_row.get(f'{state} osc.', 0)  
            print(f"Checking {state}: Oscillator Strength = {osc_strength}")  

            if osc_strength >= bright_thresh:
                bright_targets.append(i)  
                print(f"Added {state} to bright_targets")  

                if what_opt == "lowest":
                    print("Stopping at the first bright state (lowest energy over bright state cutoff, countas_bright).")
                    break

        print(f"Final bright_targets: {bright_targets}")

        if not bright_targets:
            print("No bright state found, skipping bright state optimization")
            return

        print(f"Bright states selected for optimization: {', '.join([f'S{bt}' for bt in bright_targets])}")

        processes = []

        for bright_target in bright_targets:
            opt_dir = Path(new_dir) / f"S{bright_target}"
            os.makedirs(opt_dir, exist_ok=True)

            with open(input_yaml, 'r') as file:
                yaml_settings = yaml.safe_load(file)
                yaml_settings_copy = yaml_settings.copy()
                yaml_settings_copy['optimization'] = yaml_settings_copy.get('optimization', {})
                
            if settings['grader']['excited_state'].get('TDDFT', 'no').lower() != 'yes':
                print("Using top scoring Candidate method for optimization")

                cwd = os.getcwd()

                Cand_dir = Path(cwd) / f"{top_Cand}"
                if not Cand_dir.exists():
                    print(f"Error: Directory for {top_Cand} not found, Skipping optimization")
                    continue

                settings_path = Cand_dir / 'tc.in'
                if not settings_path.exists():
                    print(f"Error: Settings for {top_Cand} not found, Skipping optimization. Check to see if tc.in is in directory.")
                    continue

                settings = {}
                with open(settings_path, 'r') as file:
                    for line in file:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if ' ' in line:
                                key,value = map(str.strip, line.split(maxsplit=1))
                                settings[key] = value

                    yaml_settings_copy['optimization'].update({
                        'method': settings.get('method', 'unknown'),
                        'basis': settings.get('basis', 'unknown'),
                        'charge': int(settings.get('charge', 0)),
                        'casscf': 'no',
                        'casci': 'no',
                        'fon': settings.get('fon', 'no'),
                        'fon_temperature': float(settings.get('fon_temperature', 0.0)),
                        'closed': int(settings.get('closed', 0)),
                        'active': int(settings.get('active', 0)),
                        'cassinglets': int(settings.get('cassinglets', 0)),
                        'gpus': int(settings.get('gpus', 1)),
                        'cphfiter': 10000
                    })

                    if settings.get('method', '').lower() == 'rhf':
                        if 'casscf' in top_Cand.lower():
                            yaml_settings_copy['optimization']['casscf'] = 'yes'
                        elif 'casci' in top_Cand.lower():
                            yaml_settings_copy['optimization']['casci'] = 'yes'

                    if settings.get('method', '').lower() in ['casscf', 'casci', 'rhf']:
                        yaml_settings_copy['optimization']['castarget'] = bright_target
                        print(f"Optimization using CASSCF/CASCI method, castarget keyword: {yaml_settings['optimization']['castarget']}")
                    elif 'hhtda' in settings.get('method', '').lower():
                        yaml_settings_copy['optimization']['cistarget'] = bright_target
                        print(f"HHTDA method, cistarget keyword: {yaml_settings['optimization']['cistarget']}")
                    else:
                        print(f"Unknown method type for optimization: {settings.get('method', '').lower()}. Skipping.")
                        continue
            else:
                print(f"TD-DFT for S{bright_target} optimization")
                method_set = settings['optimization'].get('method', 'unknown')

                yaml_settings_copy['optimization'] = ({
                    'method': method_set,
                    'gpus': int(settings.get('gpus', 1)),
                    'maxit': 1000,
                    'cis': 'yes',
                    'cistarget': bright_target,
                    'cisnumstates': bright_target
                })

            # Save new YAML and run autopilot
            new_yaml_path = opt_dir / Path(input_yaml).name
            with open(new_yaml_path, 'w') as file:
                yaml.dump(yaml_settings_copy, file, default_flow_style=True)

            # Copy the coordinates file to the new directory
            coords_file = Path(geom_file)
            if coords_file.exists():
                new_coords_path = opt_dir / coords_file.name
                shutil.copy(coords_file, new_coords_path)
                print(f"Coordinates file copied to: {new_coords_path}\n")
            else:
                print(f"Warning: Coordinates file {coords_file} not found. Skipping copy.\n")

            log_file = opt_dir / f"autopilot_S{bright_target}.log"
            print(f"Running autopilot.py for S{bright_target} optimization.")
            autopilot_path = Path(__file__).parent / "autopilot.py"

            print(f"Submitting optimization for S{bright_target} using: {sys.executable} {autopilot_path} -i {new_yaml_path}")
            with open(log_file, "w") as log:
                process = subprocess.Popen(
                    [sys.executable, str(autopilot_path), "-i", str(new_yaml_path)],
                    cwd=opt_dir,
                    stdout=log,
                    stderr=log
                )
                processes.append(process)

        for process in processes:
            process.wait()
        
        print(f"autopilot.py run in {opt_dir} with input {new_yaml_path}")

    def grouping(self):
        if self.settings['visuals'].get('grouping', 'no').lower() != 'yes':
            print("Grouping hhTDA and FOMO-CASCI methods with same DFT Method and AS respectively.")
            return

        self.data.columns = self.data.columns.str.strip()
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]
        data_copy = self.data.copy()

        ungrouped = data_copy[~data_copy['Method'].str.contains('casci_fomo|hhtda', case=False, na=False)].copy()

        #Separate CASCI-FOMO and hhTDA candidates
        casci_cand = data_copy[data_copy['Method'].str.contains('casci_fomo', case=False, na=False)].copy()
        hhtda_cand = data_copy[data_copy['Method'].str.contains('hhtda', case=False, na=False)].copy()

        #Group CASCI-FOMO by Active Space (AS)
        casci_cand['AS_Group'] = casci_cand['Method'].str.extract(r'AS(\d+)')
        casci_cand.dropna(subset=['AS_Group'], inplace=True)
        casci_cand['AS_Group'] = casci_cand['AS_Group'].astype(str)

        if not casci_cand.empty:
            casci_cand = casci_cand.loc[casci_cand.groupby('AS_Group')['Final Score'].idxmax().dropna()].reset_index(drop=True)

        #Group hhTDA by DFT method**
        hhtda_cand['DFT_Group'] = hhtda_cand['Method'].str.extract(r'hhtda_([\w]+)')
        hhtda_cand.dropna(subset=['DFT_Group'], inplace=True)
        hhtda_cand['DFT_Group'] = hhtda_cand['DFT_Group'].astype(str)

        if not hhtda_cand.empty:
            hhtda_cand = hhtda_cand.loc[hhtda_cand.groupby('DFT_Group')['Final Score'].idxmax().dropna()].reset_index(drop=True)

        #Column alignment
        all_columns = self.data.columns
        casci_cand = casci_cand.reindex(columns=all_columns, fill_value=np.nan)
        hhtda_cand = hhtda_cand.reindex(columns=all_columns, fill_value=np.nan)
        ungrouped = ungrouped.reindex(columns=all_columns, fill_value=np.nan)

        #Add the grouped candidates with ungrouped ones
        self.data = pd.concat([casci_cand, hhtda_cand, ungrouped], ignore_index=True, sort=False).reset_index(drop=True)
        self.data.sort_values(by='Final Score', ascending=False, inplace=True, ignore_index=True)

        print("Grouping complete: Only the highest-scoring candidates retained per group. See output for all Cand Scores.")

def main():
    print("Welcome to the Absorption Spectra Grader!\n")

    args = read_single_arguments()
    fn = args.input_yaml
    settings = io.yload(fn)
    n_singlets = settings['reference']['singlets']
    fol_name = fn.absolute().parents[0]
    charge = settings['general']['charge']
    geometry = settings['general']['coordinates']
    geom_file = fol_name / geometry
    mol_name = Path(geometry).stem
    bright = settings['visuals']['countas_bright']
    top_scores = settings['visuals']['top_scores']
    image_type = settings['visuals'].get('image_type', 'png').lower()

    mol = Molecule.from_xyz(geometry)
    nelec = mol.nelectrons - charge
    pointname=settings['visuals']['title']
    
    results = SinglePointResults(pointname, n_singlets, fol_name)
    results.save_csv()

    data = pd.read_csv(f'{pointname}_RAW_results.csv', index_col=0)

    grader = Grader(
        pointname=pointname,
        data=data,
        n_singlets=n_singlets,
        ref_string='EOM-CC2',
        settings=settings
    )

    log_files = []

    grader.time_pen()
    grader.append_score_columns_to_df()  # Compute AUC score, Final score, etc.
    #print("Columns in DataFrame:", grader.data.columns.tolist())
    grader.grouping()
    grader.make_spectra(image_type)
    grader.AUC_histogram(image_type)
    grader.energy_RMSD_plot(image_type)
    grader.export_scores_to_txt()
    grader.save_csv()
    grader.plot_results(image_type)

    print("Final DataFrame:", grader.data)
    
    new_dir = fol_name / "bright_state_opt"
    grader.bright_state_optimization_autopilot(new_dir, fn, geom_file)

if __name__ == "__main__":
    main()
