#!/bin/python

import sys
import scipy
import pickle
import torch

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def filter_false_snps(decoded_states, AD, DP):
    """
    Filters and creates a new state for falsely-labeled het SNPs.
    """

    # for each het SNP, change decoded state to 100 if p_val < 0.05
    # these states are likely falsely-labeled het SNPs
    for i in range(len(decoded_states)):
        p_value = scipy.stats.binomtest(min(AD[i], DP[i] - AD[i]), 
                                        n=DP[i], 
                                        p=decoded_states[i], 
                                        alternative='less').pvalue
        if p_value < 0.05:
            decoded_states[i] = 100
    return decoded_states

def plot_snps(df, plot_file):
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

    plt.savefig(plot_file)

def hmm_decode(infile, outfile, plot_file):
    """
    Decodes and returns a sequence of hidden states (LOH/normal) of each heterozygous SNP using an HMM.
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
    BAF = torch.tensor(np.array(baf_full).reshape(-1, 39846, 1)).float().to(device)

    # BAF_tensor = tf.convert_to_tensor(BAF)
    # print(BAF_tensor)

    # set probability matrices, mean, covariance
    start_prob = torch.tensor(np.array([0.2, 0.2, 0.2, 0.2, 0.2])).float().to(device)
    trans_prob = torch.tensor(np.array([[1 - 4/30000, 1/30000, 1/30000, 1/30000, 1/30000],
                  [1/30000, 1 - 4/30000, 1/30000, 1/30000, 1/30000],
                  [1/30000, 1/30000, 1 - 4/30000, 1/30000, 1/30000],
                  [1/30000, 1/30000, 1/30000, 1 - 4/30000, 1/30000],
                  [1/30000, 1/30000, 1/30000, 1/30000, 1 - 4/30000]])).float().to(device)
    dists = [Normal([0.1], [[0.2]], covariance_type='diag').to(device), 
             Normal([0.2], [[0.2]], covariance_type='diag').to(device),
             Normal([0.3], [[0.2]], covariance_type='diag').to(device),
             Normal([0.4], [[0.2]], covariance_type='diag').to(device),
             Normal([0.5], [[0.2]], covariance_type='diag').to(device)]
    
    print(f"{BAF.dtype}, {start_prob.dtype}, {trans_prob.dtype}")

    # create and initialize Gaussian HMM
    model = DenseHMM(distributions=dists, 
                     edges=trans_prob, 
                     starts=start_prob,
                     random_state=0).to(device)

    # decode using BAF matrix
    # model.fit(BAF) 
    decoded_states = model.predict(BAF).to('cpu').numpy().ravel()
    model = model.to('cpu')

    # re-label decoded_states with mean values
    means_arr = []
    for d in model.distributions:
        means_arr.append(list(d.means.numpy().ravel()))

    means_labels = {0: means_arr[0][0], 
                    1: means_arr[1][0], 
                    2: means_arr[2][0], 
                    3: means_arr[3][0], 
                    4: means_arr[4][0]}
    sequence = []

    for state in decoded_states:
        label = means_labels[state]
        sequence.append(label)

    decoded_states = sequence

    # pickle decoded sequence to outfile
    with open(outfile, 'wb') as file:
        pickle.dump(decoded_states, file)
        file.close()

    # filter out falsely-labeled het SNPs
    decoded_states = filter_false_snps(decoded_states, AD, DP)

    # plot het SNP BAF values
    het_df = pd.DataFrame({'x': het_df_snp["POS"].to_numpy(),
                           'y': np.divide(AD, DP),
                           'z': decoded_states})
    plot_snps(het_df, plot_file)

def main(infile, outfile, plot_file):
    # infile    =  pickle obj, parsed snps (from filter_snps.py)
    # outfile   =  pickle obj, decoded sequence
    # plot_file =  png, plot file name
    hmm_decode(infile, outfile, plot_file)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])