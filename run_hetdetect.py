import logging
import argparse 
import vcf
from hetdetect.version import __version__
from os.path import join


if __name__ == "__main__":

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
    parser.add_argument("-f", "--dpfilter", type=int, dest="dp_filter", default=0,
                                  help="minimum sequencing depth (DP) needed to include in the output VCF file(s)"
                                       , metavar="NUMBER")
    parser.add_argument("-T", "--threads", type=int, dest="num_thread", default=0,
                                  help="number of cores. " )       

    options = parser.parse_args()

    if options.version:
            print("hetdetect version " + __version__, flush=True)
            exit(0)                           

    with open(options.input_fp, "r") as input_fp:
        vcf_reader = vcf.Reader(input_fp)

        if len(vcf_reader.samples) <= 0:
             logging.INFO("Input VCF does not have existing genotyping. We are taking DP and AD values from INFO field and start calling") 
        elif len(vcf_reader.samples) > 1:
             logging.ERROR("We currently only support single sample VCF files")
             exit(1)
        else:
             pass # proceed with execution
        
        output_fp = open(join(options.output_fp, "hetdetect.vcf"),"w")
        vcf_writer = vcf.Writer(output_fp, vcf_reader)

        i = 0
        #re-genotyping AD > 0 & DP-AD > 0 as 0/1
        for record in vcf_reader:
            sample = record.samples[0]
            REF = sample.data.AD[0]
            if len(sample.data.AD) > 1:
                 ALT = sample.data.AD[1]
            else:
                 ALT = 0

            if ALT > 0 and REF > 0:
                GT = "0/1"
            elif ALT > 0 and REF == 0:
                GT = "1/1"
            else:
                GT = "0/0"
            
            CallData = vcf.model.make_calldata_tuple(["GT","DP","AD"])
            record.samples[0].data = CallData(GT,DP=sample.data.DP, AD=sample.data.AD)
            record.FORMAT = "GT:DP:AD"
            if options.nohmm:
                vcf_writer.write_record(record)


