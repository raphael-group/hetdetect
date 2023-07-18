#!/usr/bin/env python3
import logging
import argparse
import os 
from cyvcf2 import VCF, Writer
import shutil
from hetdetect.hmm_decode import plot_snps, run_HMM
from hetdetect.version import __version__
from os.path import join
import numpy as np
import scipy
import pandas as pd

def compress_output_decision():
    if options.compress:
        logging.info(f"Compressing output VCF files")
        os.system(f"bgzip -f {join(options.output_fp, 'hetdetect.vcf')}")
        for k in bychrom_writers.keys():
            os.system(f"bgzip -f {join(options.output_fp, 'bychrom', f'{k}.vcf')}")

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Input arguments parser
    parser = argparse.ArgumentParser(description='VCF file genotyper with increased robustness to allelic imbalance due to mutogenesis',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='store_true', help='print the current version')
    parser.add_argument("-i", "--input", dest="input_fp",
                                  help="path to the input VCF file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output_fp",
                                  help="path for the output directory where files will be placed",
                                  metavar="DIRECTORY")
    parser.add_argument("--nohmm", dest="nohmm", action='store_true', default=False,
                                  help="turn off hmm genotyping")
    parser.add_argument("--g", dest="gpu", action='store_true', default=False,
                                  help="turn on gpu usage")
    parser.add_argument("-c", "--compress", dest="compress", action='store_true', default=False,
                                  help="output compressed VCF files")
    parser.add_argument("-f", "--dpfilter", type=int, dest="dp_filter", default=0,
                                  help="minimum sequencing depth (DP) needed to include in the output VCF file(s)"
                                       , metavar="NUMBER")
    parser.add_argument("-n", "--numstates", type=int, dest="numstates", default=5,
                                  help="the number of HMM states to fit the data"
                                       , metavar="NUMBER")
    parser.add_argument("-T", "--threads", type=int, dest="num_thread", default=0,
                                  help="number of cores. " )       

    options = parser.parse_args()

    if options.version:
            print("hetdetect version " + __version__, flush=True)
            exit(0)                           


    vcf_reader = VCF(options.input_fp)
    if not os.path.exists(options.output_fp):
        os.mkdir(options.output_fp)

    nosampleinput = False
    if len(vcf_reader.samples) <= 0:
        logging.info("Input VCF does not have existing genotyping. We are taking DP and AD values from INFO field and start calling")

        header = """##fileformat=VCFv4.1
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Raw read depth">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths (high-quality bases)">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Raw read depth">
##INFO=<ID=AD,Number=1,Type=Integer,Description="Allelic depths (high-quality bases)">
##INFO=<ID=OTH,Number=1,Type=Integer,Description="Other bases">
##contig=<ID=chr1,length=248956422>
##contig=<ID=chr2,length=242193529>
##contig=<ID=chr3,length=198295559>
##contig=<ID=chr4,length=190214555>
##contig=<ID=chr5,length=181538259>
##contig=<ID=chr6,length=170805979>
##contig=<ID=chr7,length=159345973>
##contig=<ID=chr8,length=145138636>
##contig=<ID=chr9,length=138394717>
##contig=<ID=chr10,length=133797422>
##contig=<ID=chr11,length=135086622>
##contig=<ID=chr12,length=133275309>
##contig=<ID=chr13,length=114364328>
##contig=<ID=chr14,length=107043718>
##contig=<ID=chr15,length=101991189>
##contig=<ID=chr16,length=90338345>
##contig=<ID=chr17,length=83257441>
##contig=<ID=chr18,length=80373285>
##contig=<ID=chr19,length=58617616>
##contig=<ID=chr20,length=64444167>
##contig=<ID=chr21,length=46709983>
##contig=<ID=chr22,length=50818468>
##contig=<ID=chrX,length=156040895>
##contig=<ID=chrY,length=57227415>
##contig=<ID=chrM,length=16569>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	sample
"""
        vcf_writer = Writer.from_string(join(options.output_fp, "hetdetect.nohmm.vcf"), header)
        nosampleinput = True
    elif len(vcf_reader.samples) > 1:
        logging.ERROR("We currently only support single sample VCF files")
        exit(1)
    else:
        vcf_writer = Writer(join(options.output_fp, "hetdetect.nohmm.vcf"), vcf_reader)
    
    if not os.path.exists(join(options.output_fp, "bychrom")):
        os.mkdir(join(options.output_fp, "bychrom"))
    bychrom_writers = dict()


    def write_record(record):
        vcf_writer.write_record(record)
        # In addition to the main vcf output file, we write each chromosome separately to individual files
        if record.CHROM not in bychrom_writers:
                if nosampleinput:
                    bychrom_writers[record.CHROM] = Writer.from_string(join(options.output_fp, "bychrom",f"{record.CHROM}.nohmm.vcf"), header)
                else:
                    bychrom_writers[record.CHROM] = Writer(join(options.output_fp, "bychrom",f"{record.CHROM}.nohmm.vcf"), vcf_reader)
        bychrom_writers[record.CHROM].write_record(record)

    logging.info("Re-genotyping without HMM")
    #re-genotyping AD > 0 & DP-AD > 0 as 0/1
    for record in vcf_reader:
        if record.num_unknown > 0: # 0 REF and 0 ALT
            continue
        if nosampleinput:
            #save all record fields into separate variables
            CHROM=record.CHROM
            POS=record.POS
            ID="."
            REF=record.REF
            ALT=record.ALT[0]
            QUAL="."
            FILTER=record.FILTERS[0]
            DP=record.INFO["DP"]
            AD=record.INFO["AD"]
            ADs = [int(DP) - int(AD)]
            if int(AD) > 0:
                ADs.append(int(AD))
            ADstring = ",".join([str(x) for x in ADs])
            OTH = record.INFO["OTH"]
            FORMAT="GT:DP:AD"
            sample=f"0/0:{DP}:{ADstring}"
            # converting from cellsnplite format to bcftools format temporarily
            record = vcf_writer.variant_from_string(f"{CHROM}\t{POS}\t{ID}\t{REF}\t{ALT}\t{QUAL}\t{FILTER}\tDP={DP};AD={ADstring};OTH={OTH}\t{FORMAT}\t{sample}")

        if type(record.INFO.get("AD")) == int:
            write_record(record)           
            continue
        REF, ALT = record.INFO.get("AD")[0:2]
        if REF > 0 and ALT > 0:
            record.genotypes = np.array([[0,1,False]])
        elif REF == 0 and ALT > 0:
            record.genotypes = np.array([[1,1,False]])
        else:
            record.genotypes = np.array([[0,0,False]])
        write_record(record)

    vcf_writer.close()
    for k,v in bychrom_writers.items():
        v.close()

    if options.nohmm:
        # move temporary files to output file names since we won't run HMM
        logging.info("Writing noHMM genotypes into the output file.")
        shutil.move(join(options.output_fp, "hetdetect.nohmm.vcf"),
                    join(options.output_fp, "hetdetect.vcf"))
        for k, _ in bychrom_writers.items():
            shutil.move(join(options.output_fp, "bychrom",f"{k}.nohmm.vcf"),
                        join(options.output_fp, "bychrom",f"{k}.vcf"))
        compress_output_decision()
        exit(0)

    def hmm_genotyper(k):
        logging.info(f"HMM genotyping chromosome {k}.")
        hets = []
        ADs = []
        DPs = []
        with open(join(options.output_fp, "bychrom",f"{k}.nohmm.vcf"),"r") as f:
            vcf_reader = VCF(f)
            for record in vcf_reader:
                #sample = record.samples[0]
                # only process het SNPs in HMM
                if record.genotypes[0][0] != record.genotypes[0][1]:
                    hets += [(record.CHROM, record.POS)]
                    # get DP and AD. These can be lists or integers, so being careful here
                    REF, ALT = record.INFO.get("AD")[0:2]
                    DP = REF + ALT
                    AD = ALT
                    ADs.append(AD)
                    DPs.append(DP)
        logging.info(f"Running Baum-Welch for chromosome {k}.")
        ADs = np.array(ADs)
        DPs = np.array(DPs)
        logprob, decoded_states, model= run_HMM(ADs, DPs, options.numstates)
        model_means = model.means_
        logging.info(f"Log probability for {k}: {logprob}")
        logging.info(f"Decoded states for {k}:  {decoded_states}")

        # run binomial test and regenotype het SNPs as homozygous if pvalue is below the threshold
        # we use the inferred hidden state's mean as binomial test p parameter
        newGTs = []

        testSave = dict()
        for j, (AD, DP, state) in enumerate(zip(ADs, DPs, decoded_states)):
            Bk = min(AD, DP - AD)
            Bn = DP
            Bp = float(model_means[state])
            if (Bk,Bn,Bp) in testSave:
                p_value = testSave[(Bk,Bn,Bp)] 
            else:
                p_value = scipy.stats.binomtest(Bk, 
                                n=Bn, 
                                p=Bp, 
                                alternative='less').pvalue
                testSave[(Bk,Bn,Bp)] = p_value
            if p_value < 0.025:
                if AD < DP - AD:
                    newGTs.append([0,0,False])
                    decoded_states[j] = -1 # mark as false het
                else:
                    newGTs.append([1,1,False])
                    decoded_states[j] = -1 # mark as false het  
            else:
                newGTs.append([0,1,False]) 
        
        newGTdict = dict(zip(hets, newGTs))
        logging.info("Done with the binomial test")
        # plot het SNPs
        het_df = pd.DataFrame({'x': np.array([het[1] for het in hets]),
                    'y': np.divide(ADs, DPs),
                    'z': decoded_states})
        # make plots directory
        if not os.path.exists(join(options.output_fp, "plots")):
            os.mkdir(join(options.output_fp, "plots"))
        plot_snps(het_df, join(options.output_fp, 'plots', f'{k}.png'))
        logging.info(f"Ploted chromosome {k}")
        
        # write output files by reading the temporrary "nohmm" file and overwrite the genotypes
        with open(join(options.output_fp, "bychrom",f"{k}.nohmm.vcf"),"r") as f:
            vcf_reader = VCF(f)
            vcf_writer = Writer(join(options.output_fp, "bychrom",f"{k}.vcf"), vcf_reader)
            for record in vcf_reader:
                index = (record.CHROM, record.POS)
                if index in newGTdict:
                    record.genotypes = np.array([newGTdict[index]])
                vcf_writer.write_record(record)
            vcf_writer.close()
        logging.info(f"New HMM genotypes are written for chromosome {k}")
        os.remove(join(options.output_fp, "bychrom",f"{k}.nohmm.vcf"))
                

    # can be made multithreaded using pool
    for k, _ in bychrom_writers.items():
        hmm_genotyper(k)
    
    logging.info(f"Combining individual chromosome VCFs into one")
    # combine individual chromosome VCFs and write to the main output VCF
    with open(join(options.output_fp, "hetdetect.nohmm.vcf"),"r") as output_main_nohmm_fp:
        vcf_reader = VCF(output_main_nohmm_fp)
        # using output_main_nohmm_fp header as template
        vcf_writer = Writer(join(options.output_fp, "hetdetect.vcf"), vcf_reader)
        for k in bychrom_writers.keys():
            with open(join(options.output_fp, "bychrom",f"{k}.vcf"),"r") as f:
                vcf_reader = VCF(f)
                for record in vcf_reader:
                    vcf_writer.write_record(record)
        vcf_writer.close()
    logging.info(f"Done writing the output VCF file")
    compress_output_decision()
    os.remove(join(options.output_fp, "hetdetect.nohmm.vcf"))
    
    
    
            
            


