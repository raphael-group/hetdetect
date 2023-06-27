import logging
from argparse 

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
                                  