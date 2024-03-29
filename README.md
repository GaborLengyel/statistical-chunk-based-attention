# statistical-chunk-based-attention
This repository contains the statistical analysis for the following article:

title: Statistically defined visual chunks engage object-based attention

Authors: Gábor Lengyel, Marton Nagy & József Fiser

DOI: https://doi.org/10.1038/s41467-020-20589-z

The scripts were written by Gabor Lengyel (early & busy phd years = ugly inefficient coding)


Instructions:

0. download the data files from the following osf project https://osf.io/paqhd/ and put all csv files in the /data folder (the data is already in the data folder so you can skip step 0)

1. Run *Experiment1_FrequentistStat.ipynb* with jupyter notebook (python 2 or 3) to see the frequentist statistical analysis and plots for experiment 1

2. Run *Experiment1_BayesianStat.ipynb* with jupyter notebook (R with BayesFactor 0.9.12-4.2. package) to see the Bayesian statistical analysis for experiment 1 (this script uses csv files that are written out in *Experiment1_FrequentistStat.ipynb* to /data/results folder)

1. Run *Experiment2_FrequentistStat.ipynb* with jupyter notebook (python 2 or 3) to see the frequentist statistical analysis and plots for experiment 2

2. Run *Experiment2_BayesianStat.ipynb* with jupyter notebook (R with BayesFactor package) to see the Bayesian statistical analysis for experiment 2 (this script uses csv files that are written out in *Experiment2_FrequentistStat.ipynb* to /data/results folder)



The scripts were tested with
- python 2.7.15
- R 3.5.1 with BayesFactor package 0.9.12-4.2.


Correspondence to: lengyel.gaabor@gmail.com
