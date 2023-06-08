#!/bin/python

import sys
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd
from hmmlearn import hmm
from collections import defaultdict
import gzip
from multiprocessing import Pool, cpu_count
from itertools import repeat
import os

count = 0

def get_ALT_alleles_from_db(db, chrom):
    """
    Parses a standard VCF file, of format:
    CHROM POS ID REF ALT
    where REF is a string of 1 or more nucleotides (more than 1 if deletion)
    and ALT is list of ALT alleles

    This function excludes sites in which the REF allele is longer than 1 nucleotide,
    and it excludes ALT alleles longer than 1 nucleotide
    """
    db_ALT_alleles = defaultdict(list)
    with gzip.open(db, 'rt') as vcf:
        for line in vcf:
            if not line.startswith('#'):
                line = line.split()
                if line[0] == chrom:
                    position = int(line[1])
                    ALTs = line[4].split(',')
                    REF = line[3]
                    # filter out indels, where REF or ALT may be more than 1 nucleotide
                    if len(REF)==1:
                        for a in ALTs:
                            if len(a)==1:
                                db_ALT_alleles[position].append(a)
    return(db_ALT_alleles)

def parse_FORMAT(df_snp):
    """
    get AD from FORMAT field, which includes only informative reads; INFO field also has AD and DP info but here DP also contains uninformative reads
    """
    depths = defaultdict(list)
    
    for i,row in df_snp.iterrows():
        form = np.array(row.FORMAT.split(':'))
        sample = row.SAMPLE.split(':')
        # find index in sample column that correspond to 'AD', where the position of information in sample column is specified in FORMAT column
        AD_index = np.where(form == 'AD')[0][0]
        AD = sample[AD_index].split(',')
        # assume we've previously filtered for bi-allelic sites
        assert len(AD) <=2
        if len(AD)==2:
            # ALT allele present, AD is list
            depths['AD'].append( int(AD[1]) ) 
            depths['DP'].append( int(AD[0])+int(AD[1]) ) 
        else:
            # no ALT allele, AD is scalar
            depths['AD'].append( 0 ) 
            depths['DP'].append( int(AD[0]) ) 

    assert len(depths['AD']) == len(depths['DP'])
    return depths

def ALT_in_db(db, df):
    indices = []
    for i,row in df.iterrows():
        # because we removed indels, some position at which we called SNPs may not be in filtered db from function get_ALT_alleles_from_db
        if row['POS'] in db:
            # row['ALT'] should be '.' or a single nucleotide since we filtered for bi-allelic sites
            if row['ALT'] == '.' or row['ALT'] in db[row['POS']]:
                indices.append(i)
    return indices

def regenotype(format, sample):
    """
    Re-assigns genotypes to each SNP.
    """
    x = sample.split(":")

    # parse dp, ad, ref values from format
    if format == "GT:DP:AD":
        dp, ad = int(x[1]), 0
        ref = dp
    else:
        pl, dp, ad, ref = x[1], int(x[2]), int(x[3].split(",")[1]), int(x[3].split(",")[0])
    
    # reassign genotypes
    if ref == 0 and ad != 0:
        gt = "1/1"
    elif ref != 0 and ad == 0:
        gt = "0/0"
    else:
        gt = "0/1"
    
    # reconstruct + return new sample string
    if format == "GT:DP:AD":
        new_sample = f"{gt}:{dp}:{ref}"
    else:
        new_sample = f"{gt}:{pl}:{dp}:{ref},{ad}"
    return new_sample

# --------------- FOR CUSTOM EMISSION PROBABILITIES --------------- ###
#
# def emission_loh(DP, AD, lost, retained):
#     global count
#     emission = math.comb(DP[count], AD[count]) * pow(lost, AD[count]) * pow(retained, DP[count] - AD[count])
#     count += 1
#     print("loh computed")
#     return emission
#
# def emission_norm(DP, AD):
#     global count
#     emission = math.comb(DP[count], AD[count]) * pow(0.5, AD[count]) * pow(0.5, DP[count] - AD[count])
#     count += 1
#     print("norm computed")
#     return emission
#
# ----------------------------------------------------------------- ###

def hmm_sequence(AD, DP, purity):
    """
    Uses Gaussian HMM to decode and return a sequence of hidden states (LOH/normal) of each heterozygous SNP.
    """
    # create 1-D matrix of observed states (BAF) with all values <1 (LAF)
    baf_full = np.divide(AD, DP)
    with np.nditer(baf_full, op_flags=['readwrite']) as it:
        for x in it:
            if x > 0.5:
                x[...] = 1 - x
    BAF = np.array(baf_full).reshape(-1, 1)
    p = float(purity)

    # set probability matrices, mean, covariance
    lost = (1 - p)/(2 - p)
    retained = (p)/(2 - p)
    start_prob = np.array([0.5, 0.5])
    trans_prob = np.array([[1 - 1/30000, 1/30000],
                           [1/30000, 1 - 1/30000]])
    mean = np.array([[(1 - p)/(2 - p)], [0.5]])
    covariance = np.array([[0.2], [0.2]])

    # create and initialize Gaussian HMM
    model = hmm.GaussianHMM(n_components=2)
    model.startprob_ = start_prob
    model.transmat_ = trans_prob
    model.means_ = mean
    model.covars_ = covariance
      
    # decode using BAF matrix
    logprob, decoded_states = model.decode(BAF, algorithm="viterbi")
    state_labels = {0: "LOH", 1: "Normal"}
    sequence = []

    ### FOR WRITING OUT HMM SEQUENCE ONTO A FILE
    # file1 = open(sequence_file, 'w')
    for state in decoded_states:
        label = state_labels[state]
        # file1.write(f"{label}, ")
        sequence.append(label)
    # file1.close()        

    return decoded_states

def plot_snps(df):
    groups = df.groupby('z')
    for name, group in groups:
        plt.plot(group.x, group.y, marker='o', linestyle='', markersize=0.5, label=name)
    
    plt.ylim([0, 1])
    plt.xlabel("SNP Position")
    plt.ylabel("BAF")
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

    plt.show()

def filter_snps(vcf_file, outfile, purity):
    """
    in order to preserve original VCF formatting as much as possible to avoid downstream errors with other programs
    """
    db = "/Users/melodychoi/Desktop/research/task1/00-common_all.vcf.gz"

    vcf_file_name = vcf_file.split("/")[-1]
    chromosome = vcf_file_name.split('.')[0]
    db_ALT_alleles = get_ALT_alleles_from_db(db, chromosome)

    df_snp = pd.read_csv(vcf_file, comment="#", sep="\t", names=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "SAMPLE"])

    # only consider bi-allelic sites
    df_snp = df_snp[df_snp['ALT'].isin(['A','G','C','T','.'])]

    # get ALT allele depth (AD) and total depth (DP) information
    depths = parse_FORMAT(df_snp)

    df_snp["AD"] = depths["AD"]
    df_snp["DP"] = depths["DP"]

    # remove records with DP < 8
    hi_depth = df_snp[df_snp.DP >= 8]

    # select only sites in which ALT allele agrees with ALT in database
    hi_depth.reset_index(inplace=True, drop=True)
    indices_to_select = ALT_in_db(db_ALT_alleles, hi_depth)
    hi_depth = hi_depth.iloc[indices_to_select]
    hi_depth.reset_index(inplace=True, drop=True)

    # create a df containing the computed BAF values for SNPs marked as het by a previous algorithm.
    het_df_snp = df_snp[df_snp['SAMPLE'].str.contains('0/1')]
    AD = het_df_snp['AD'].to_numpy()
    DP = het_df_snp['DP'].to_numpy()

    # pass in AD, DP, purity, start/trans probabilities into hmm function
    decoded_states = hmm_sequence(AD, DP, purity)

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

    # plots het SNP BAF values
    het_df = pd.DataFrame({'x': het_df_snp["POS"].to_numpy(),
                           'y': np.divide(AD, DP),
                           'z': decoded_states})
    plot_snps(het_df)

    hi_depth_positions = set(zip(hi_depth.CHROM.astype('str'), hi_depth.POS.astype('str')))
    to_print = []
    with gzip.open(vcf_file, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                to_print.append(line)
            else:
                split = line.split()
                chrom, pos = split[0], split[1]
                if (chrom, pos) in hi_depth_positions:
                    format, sample = split[8], split[9]
                    split[9] = regenotype(format, sample)
                    line = '\t'.join(split)
                    to_print.append(line)

    with gzip.open(outfile, 'wt') as o:
        o.write("\n".join(to_print))

def main(infile, outfile, purity):
    filter_snps(infile, outfile, purity)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
