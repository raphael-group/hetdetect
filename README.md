# :dna:	HetDetect
**_Inferring heterozygous SNP positions from tumor samples without a matched normal using Hidden Markov Models_**

Authors: Melody Choi, Metin Balaban, Ben Raphael 



## :microscope:	Overview
HetDetect allows for the inference of hetSNPs. It takes as input
- any VCF file (bcftools, cellSNPlite),
- a specified output path,
- any additional arguments,

uses _cyvcf2_ to parse the input VCF files, and finally outputs a re-genotyped VCF with the inferred hetSNPs.

It uses a Hidden Markov Model consisting of:
- a user defined number of hidden states,
- a fixed, low transition probability matrix (tau = 3*10^-4),
- a 1D Gaussian emission probability matrix, and
- LAF as observed states.

It also produces a colored BAF scatterplot, as such:
![](/assets/images/chr8.png)

These plots can be found in the specific outdirectory provided as input to `run_hetdetect.py`.


## Installation

- Clone the repository and change directory
- Run `pip3 install -e .`


## :pushpin:	Features
- **_GPU usage_**. For accelerated performance using pomegranate, PyTorch, and CUDA GPU, users can specify if they would like to run the model with tensors on GPU with the option `--g`. 
- **_User defined number of hidden states_**. Users can customize the HMM to define any number of hidden states corresponding to the number of mean/covariance pairs that the model will infer.
- **_Binomial test._** To filter out SNPs that may have been falsely labeled as heterozygous by the HMM, a binomial statistical test is performed to re-label false het-SNPs as homozygous (either 0/0 for homozygous REF or 1/1 for homozygous ALT). 



## :computer:	Using HetDetect
To run HetDetect, call `python run_hetdetect.py -i [input file path] -o [output file path]` along with any other arguments as necessary.

Run `python run_detect.py -h` to see all the options.


