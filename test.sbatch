#!/bin/bash -l
#SBATCH --partition=kipac
#SBATCH -t 2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mydo@stanford.edu
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=submit.23p.out


source ~/setup.sh
conda activate wfsim

python simtest.py --atmSeed 23 --outdir heightPsfws --outfile corr23.pkl --usePsfws
