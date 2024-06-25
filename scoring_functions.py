import pandas as pd
import numpy as np
import itertools as it


def sigmoid(x, steepness, inflection_point):
    y = 1/(1+np.power(10, steepness*(inflection_point-x)))
    return y.astype(int)

def sigmoid2(x, steepness, inflection_point):
    y = 1/(1+np.power(10, steepness*(inflection_point-x)))
    return y

def get_ref_df_index(data, string):
    ref_ind = data[data['Method'] == string].index[0]
    return ref_ind


def get_diff_indexes(a, b):
    diff = abs(a-b)
    indexes = np.where(diff == 1)[0]
    ind_0 = []
    ind_1 = []
    for ind in indexes:
        if b[ind] == 0:
            ind_0.append(ind)
        else:
            ind_1.append(ind)
    return ind_0, ind_1


def normalize_energy_array(arr):
    norm_list = []
    for line in arr:
        norm_line = (line - np.min(line)) / (np.max(line) - np.min(line))
        norm_list.append(norm_line)
    return np.array(norm_list)


def calculate_combinations(a, b):
    combinations = []
    for a, b in zip(it.repeat(a), it.permutations(b)):
        combinations.append(list(zip(a, b)))
    return combinations


def rescale(arr):
    return arr / np.max(arr)


def calculate_min_swaps(combinations):
    swaps = []
    for comb in combinations:
        x = 0
        for pair in comb:
            x += abs(pair[0]-pair[1])
        swaps.append(x)
    return min(swaps)


def main():
    data = pd.read_csv('S0min_results.csv')
    ref = data['EOM-CC2']
    print(ref)


if __name__ == "__main__":
    main()
