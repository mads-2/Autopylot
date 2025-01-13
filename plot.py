from dataclasses import dataclass
import pandas as pd
import scoring_functions as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from autopilot import read_single_arguments
import io_utils as io
from results_collector import SinglePointResults


@dataclass
class Plot:
    pointname: str
    data: pd.DataFrame
    n_singlets: int
    #n_triplets: int
    #ref_string: str

    @property
    def state_list(self):
        state_list = []
        for i in range(self.n_singlets):
            state_list.append(f'S{i}')
        #for i in range(self.n_triplets):
            #state_list.append(f'T{i+1}')
        return state_list

    #@property
    #def ref_ind(self):
    #    ref_ind = sf.get_ref_df_index(self.data, self.ref_string)
    #    return ref_ind

    #def osc_score(self):
        #osc_array = np.array([self.data[x] for x in [f'S{x+1} osc.' for x in range(self.n_singlets-1)]]).transpose()
        #brightness_array = sf.sigmoid(osc_array)
        #tot_brightness_array = np.sum(brightness_array, axis=1)
        #delta_brightness_array = abs(tot_brightness_array - tot_brightness_array[self.ref_ind])
        #osc_score_list = []
        #for i, x in enumerate(delta_brightness_array):
        #   if x > 0.5:
        #       osc_score_list.append(float(self.n_singlets - x))
        #   else:
        #        ref = brightness_array[self.ref_ind]
        #        vec = brightness_array[i]
        #        ind_0, ind_1 = sf.get_diff_indexes(ref, vec)
        #        combinations = sf.calculate_combinations(ind_0, ind_1)
        #        min_swaps = sf.calculate_min_swaps(combinations)
        #        osc_score_list.append(self.n_singlets+(1/(min_swaps+1)))
        #return np.array(osc_score_list)

    #def energy_score(self):
        #ene_array = np.array([self.data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
        #norm_ene_array = sf.normalize_energy_array(ene_array)
        #sum_score_list = []
        #for i in range(norm_ene_array.shape[0]):
        #   diff_vec = abs(norm_ene_array[self.ref_ind]-norm_ene_array[i])
        #   sum_cand = np.sum(diff_vec)
        #   sum_score_list.append(sum_cand)
        #return np.array(sum_score_list)

    #def suggested_alpha(self):
        #ene_array = np.array([self.data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
        #ref_max = np.max(ene_array[self.ref_ind])
        #alpha_list = []
        #for exc in ene_array:
        #    alpha_list.append(ref_max/np.max(exc))
        #return np.array(alpha_list)

    #def append_score_columns_to_df(self):
        #self.data['Osc. score'] = self.osc_score()
        #self.data['Energies score'] = self.energy_score()
        #self.data['Total score'] = (self.osc_score() - self.energy_score())
        #self.data['Suggested alpha'] = self.suggested_alpha()
        #self.data.sort_values(by=['Total score'], ascending=False, inplace=True, ignore_index=True)

    def plot_results(self):
        ene_array = np.array([self.data[x] for x in [f'{x} energy' for x in self.state_list]]).transpose()
        norm_ene_array = sf.normalize_energy_array(ene_array)
        cand_number = len(self.data)
        xs = range(cand_number)
        methods = self.data['Method']
        plt.rcParams.update({'font.size': 18})
        #fig, [ax0, ax1, ax2] = plt.subplots(3, 1, figsize=[cand_number/2, 18], sharex=True, sharey=False, gridspec_kw={'wspace': 0, 'hspace': 0})
        fig, [ax0, ax1] = plt.subplots(2, 1, figsize=[cand_number/2, 18], sharex=True, sharey=False, gridspec_kw={'wspace': 0, 'hspace': 0})
        colors = cm.get_cmap('gist_rainbow', len(self.state_list))
        my_cmap = cm.get_cmap('rainbow_r')
        for i, state in enumerate(self.state_list):
            for j in range(cand_number):
                dark = state != 'S0' and state[0] == 'S' and self.data[f'{state} osc.'][j] < 0.1
                if dark:
                    style = (0, (1, 1))
                else:
                    style = 'solid'
                ax0.hlines(y=self.data[f'{state} energy'][j], xmin=xs[j]-0.3, xmax=xs[j]+0.3, linewidth=5, color=colors(i), ls=style, label=self.state_list[i] if j == 0 else "")
                ax1.hlines(y=norm_ene_array[j][i], xmin=xs[j]-0.3, xmax=xs[j]+0.3, linewidth=5, color=colors(i), ls=style)
        #ax2.bar(xs, self.data['Total score'], color=my_cmap(sf.rescale(self.data['Total score'])))
        ax0.set_ylabel('Energy difference [eV]')
        ax1.set_ylabel('Norm. E. diff.')
        #ax2.set_ylabel('Score')
        ax1.set_xticks(xs)
        #ax2.set_xlim(-1.0, cand_number)
        #ax2.set_ylim(0.0, max(self.data['Total score']))
        ax1.set_xticklabels(methods, rotation=90)
        ax0.legend(fontsize=20, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        fig.savefig(f'{self.pointname}_results.png', dpi=300)

    def save_csv(self):
        self.data.dropna(axis='columns', how='all').loc[:, (self.data != 0).any(axis=0)].to_csv(f'{self.pointname}_results.csv', index=False)


def main():
    args = read_single_arguments()
    fn = args.input_yaml
    settings = io.yload(fn)
    n_singlets = settings['reference']['singlets']
    #n_triplets = settings['reference']['triplets']
    fol_name = fn.absolute().parents[0]
    results = SinglePointResults('S0min', n_singlets, fol_name)
    results.save_csv()
    data = pd.read_csv('S0min_RAW_results.csv', index_col=0)
    plot = Plot('S0min', data, n_singlets)
    #scores.append_score_columns_to_df()
    plot.plot_results()
    #scores.save_csv()
    #print(scores.data.dropna(axis='columns', how='all').loc[:, (scores.data != 0).any(axis=0)])


if __name__ == "__main__":
    main()
