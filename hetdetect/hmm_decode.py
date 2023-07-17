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

def filter_false_snps(sequence, AD, DP):
    """
    Filters and creates a new state for falsely-labeled het SNPs.
    """

    # for each het SNP, change decoded state to 5 if p_val < 0.05
    # these states are likely falsely-labeled het SNPs
    for i in range(len(sequence)):
        p_value = scipy.stats.binomtest(min(AD[i], DP[i] - AD[i]), 
                                        n=DP[i], 
                                        p=sequence[i], 
                                        alternative='less').pvalue
        if p_value < 0.05:
            sequence[i] = 5
    return sequence

def plot_snps(df, plot_file):
    """
    Creates a scatterplot of het-SNP BAF values.
    """
    #import pdb; pdb.set_trace;
    groups = df.groupby('z')

    for name, group in groups:
        print(name,len(group))
        plt.plot(group.x, group.y, marker='o', linestyle='', markersize=0.5, label=name)
    
    plt.ylim([0, 1])
    plt.xlabel("SNP Position")
    plt.ylabel("BAF")
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

    plt.savefig(plot_file)

def run_HMM_pomegranade(AD,DP,numstates):
    pass

def run_HMM(AD,DP,numstates):
        # create 1-D matrix of observed states (BAF) with all values <1 (LAF)
    baf_full = np.divide(AD, DP)
    with np.nditer(baf_full, op_flags=['readwrite']) as it:
        for x in it:
            if x > 0.5:
                x[...] = 1 - x
    BAF = np.array(baf_full).reshape(-1, 1)

    # set probability matrices, mean, covariance
    start_prob = np.array([1]*numstates)/numstates
    tau = 1/30000
    trans_prob = np.full((numstates, numstates), tau)
    np.fill_diagonal(trans_prob, 1 - (numstates-1) * tau)
    covariance = np.array([[0.2]]*numstates)
    #mean = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])

    # create and initialize Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=numstates,
        init_params='m',
        params='m',
        n_iter=100,)
    model.startprob_ = start_prob
    model.transmat_ = trans_prob
    model.covars_ = covariance
    #model.means_ = mean

    # decode using BAF matrix
    model.fit(BAF)
    logprob, decoded_states = model.decode(BAF, algorithm="viterbi")
    print(f"means: {model.means_}")
    print(f"covar: {model.covars_}")
    
    return logprob, decoded_states, model

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

    logprob, decoded_states, model = run_HMM(AD,DP,5)

    means_labels = {0: model.means_[0][0], 
                    1: model.means_[1][0], 
                    2: model.means_[2][0], 
                    3: model.means_[3][0], 
                    4: model.means_[4][0]}
    sequence = []

    # file1 = open(sequence_file, 'w')
    for state in decoded_states:
        label = means_labels[state]
    #     file1.write(f"{state}, ")
        sequence.append(label)
    # file1.close()
    
    # pickle decoded sequence
    with open(outfile, 'wb') as file:
        pickle.dump(decoded_states, file)
        file.close()
        
    # filter out falsely-labeled het SNPs
    decoded_states = filter_false_snps(sequence, AD, DP)
    
    # plots het SNP BAF values
    het_df = pd.DataFrame({'x': het_df_snp["POS"].to_numpy(),
                           'y': np.divide(AD, DP),
                           'z': decoded_states})
    plot_snps(het_df)

def main(infile, outfile):
    hmm_decode(infile, outfile)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])