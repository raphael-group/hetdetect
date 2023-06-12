#!/bin/python

import sys
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd
import pickle
from hmmlearn import hmm
from collections import defaultdict
from itertools import repeat

def filter_false_snps(decoded_states, AD, DP):
    """
    Filters and creates a new state for falsely-labeled het SNPs.
    """
    # find indices where decoded_states is "normal"
    normal_indices = np.where(decoded_states == 1)[0]
    AD_norm = np.array(AD)[normal_indices]
    DP_norm = np.array(DP)[normal_indices]
    norm_count = normal_indices.size

    # initialize counts (for percentages)
    count_00 = 0
    count_11 = 0

    # for each "normal" het SNP, change decoded state to 2 if p_val < 0.05
    # these states are likely falsely-labeled het SNPs
    for i in range(norm_count):
        p_value = scipy.stats.binomtest(AD_norm[i], n=DP_norm[i], p=0.5, alternative='two-sided').pvalue
        curr_BAF = AD_norm[i]/DP_norm[i]
        if p_value < 0.05:
            decoded_states[normal_indices[i]] = 2
            if curr_BAF < 0.5:
                count_00 += 1
            else:
                count_11 += 1
    
    # report percentage of Normal hetSNPs that are marked as homo ref vs. homo alt
    percentage_00 = (count_00 / norm_count) * 100
    percentage_11 = (count_11 / norm_count) * 100   
    print(f"Normal -> 0/0: {percentage_00}%")
    print(f"Normal -> 1/1: {percentage_11}%")

    return decoded_states

def plot_snps(df):
    """
    Creates a scatterplot of het-SNP BAF values.
    """
    groups = df.groupby('z')
    for name, group in groups:
        plt.plot(group.x, group.y, marker='o', linestyle='', markersize=0.5, label=name)
    
    plt.ylim([0, 1])
    plt.xlabel("SNP Position")
    plt.ylabel("BAF")
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

    plt.show()

def hmm_decode(infile, outfile):
    """
    Uses Gaussian HMM to decode and return a sequence of hidden states (LOH/normal) of each heterozygous SNP.
    """
    # unload input df
    with open(infile, 'rb') as file:
        het_df_snp = pickle.load(file)
        file.close()

    AD = het_df_snp['AD'].to_numpy()
    DP = het_df_snp['DP'].to_numpy()

    # create 1-D matrix of observed states (BAF) with all values <1 (LAF)
    baf_full = np.divide(AD, DP)
    with np.nditer(baf_full, op_flags=['readwrite']) as it:
        for x in it:
            if x > 0.5:
                x[...] = 1 - x
    BAF = np.array(baf_full).reshape(-1, 1)

    # set probability matrices, mean, covariance
    start_prob = np.array([0.5, 0.5])
    trans_prob = np.array([[1 - 1/300000000, 1/300000000],
                           [1/300000000, 1 - 1/300000000]])
    covariance = np.array([[0.2], [0.2]])
    mean = np.array([[(1 - 0.8)/(2 - 0.8)], [0.5]]) # random values for init

    # create and initialize Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=2,
        init_params='m',
        params='m'
        )
    model.startprob_ = start_prob
    model.transmat_ = trans_prob
    model.covars_ = covariance
    model.means_ = mean

    # decode using BAF matrix
    model.fit(BAF)
    logprob, decoded_states = model.decode(BAF, algorithm="viterbi")
    print(f"means: {model.means_}")
    print(f"covar: {model.covars_}")

    state_labels = {0: "LOH", 1: "Normal"}
    sequence = []

    ### write out decoded sequence
    # for state in decoded_states:
    #     label = state_labels[state]
    #     sequence.append(label)

    # pickle decoded sequence
    with open(outfile, 'wb') as file:
        pickle.dump(decoded_states, file)
        file.close()
    
    # filter out falsely-labeled het SNPs
    decoded_states = filter_false_snps(decoded_states, AD, DP)
    
    # plots het SNP BAF values
    het_df = pd.DataFrame({'x': het_df_snp["POS"].to_numpy(),
                           'y': np.divide(AD, DP),
                           'z': decoded_states})
    plot_snps(het_df)

def main(infile, outfile):
    hmm_decode(infile, outfile)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
