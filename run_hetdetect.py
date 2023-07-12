#!/usr/bin/env python3
import logging
import argparse
import os 
from cyvcf2 import VCF, Writer
import shutil
from hetdetect.hmm_decode import run_HMM
from hetdetect.version import __version__
from os.path import join
import numpy as np
import scipy

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

    if len(vcf_reader.samples) <= 0:
        logging.info("Input VCF does not have existing genotyping. We are taking DP and AD values from INFO field and start calling")
        vcf_reader._column_headers += ["FORMAT", "sample"] 
    elif len(vcf_reader.samples) > 1:
        logging.ERROR("We currently only support single sample VCF files")
        exit(1)
    else:
        pass # proceed with execution
    
    vcf_writer = Writer(join(options.output_fp, "hetdetect.nohmm.vcf"), vcf_reader)
    if not os.path.exists(join(options.output_fp, "bychrom")):
        os.mkdir(join(options.output_fp, "bychrom"))
    bychrom_writers = dict()

    i = 0

    def write_record(record):
        vcf_writer.write_record(record)
        # In addition to the main vcf output file, we write each chromosome separately to individual files
        if record.CHROM not in bychrom_writers:
                bychrom_writers[record.CHROM] = Writer(join(options.output_fp, "bychrom",f"{record.CHROM}.nohmm.vcf"), vcf_reader)
        bychrom_writers[record.CHROM].write_record(record)

    logging.info("Re-genotyping without HMM")
    #re-genotyping AD > 0 & DP-AD > 0 as 0/1
    for record in vcf_reader:
        if record.num_unknown > 0: # 0 REF and 0 ALT
            continue
        if record.num_called == 0:
            # converting from cellsnplite format to bcftools format temporarily
            CallData = vcf.model.make_calldata_tuple(["GT","DP","AD"])
            DP = int(record.INFO["DP"][0])
            AD = int(record.INFO["AD"][0])
            REF = DP - AD
            record.samples = [vcf.model._Call(record,
                                                sample="sample",
                                                data=CallData("0/0", DP=DP, AD=[REF, AD]))] # 0/0 is a placeholder
            record.FORMAT = "GT:DP:AD"

        if type(record.INFO.get("AD")) == int:
            vcf_writer.write_record(record)
            write_record(record)           
            continue
        REF, ALT = record.INFO.get("AD")[0:2]
        if REF > 0 and ALT > 0:
            record.genotypes = np.array([[0,1,False]])
        write_record(record)
        # sample = record.samples[0]
        # REF = sample.data.AD[0]
        # if sample.data.DP < options.dp_filter:
        #         continue
        # if len(sample.data.AD) > 1:
        #         ALT = sample.data.AD[1]
        # else:
        #         ALT = 0

        # if ALT > 0 and REF > 0:
        #     GT = "0/1"
        # elif ALT > 0 and REF == 0:
        #     GT = "1/1"
        # else:
        #     GT = "0/0"
        
        # # overwrite the Calldata with the new genotype
        # CallData = vcf.model.make_calldata_tuple(["GT","DP","AD"])
        # # using cellsnplite format where AD is the alternate allele count only
        # record.samples[0].data = CallData(GT,DP=sample.data.DP, AD=ALT)
        # record.FORMAT = "GT:DP:AD"
        # vcf_writer.write_record(record)

        # bychrom_writers[record.CHROM].write_record(record)

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
        for AD, DP, state in zip(ADs, DPs, decoded_states):
            p_value = scipy.stats.binomtest(min(AD, DP - AD), 
                            n=DP, 
                            p=model_means[state], 
                            alternative='two-sided').pvalue
            if p_value < 0.05:
                if AD < DP - AD:
                    newGTs.append([0,0,False])
                else:
                    newGTs.append([1,1,False])
            else:
                newGTs.append([0,1,False]) 
        
        newGTdict = dict(zip(hets, newGTs))
        
        # write output files by reading the temporrary "nohmm" file and overwrite the genotypes
        with open(join(options.output_fp, "bychrom",f"{k}.nohmm.vcf"),"r") as f:
            vcf_reader = VCF(f)
            vcf_writer = Writer(join(options.output_fp, "bychrom",f"{k}.vcf"), vcf_reader)
            for record in vcf_reader:
                index = (record.CHROM, record.POS)
                if index in newGTdict:
                    record.genotypes = np.array([newGTdict[index]])
                    # sample = record.samples[0]
                    # CallData = vcf.model.make_calldata_tuple(["GT","DP","AD"])
                    # record.samples[0].data = CallData(newGTdict[index],DP=sample.data.DP, AD=sample.data.AD)
                    # record.FORMAT = "GT:DP:AD"
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
    
    
    
            
            


