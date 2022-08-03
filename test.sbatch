#!/bin/bash -l
#SBATCH --partition=kipac
#SBATCH -t 2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mydo@stanford.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=submit.out


source ~/setup.sh
conda activate wfsim

python fit_2pcfs.py --simdir heightPsfws --simfile outh_psfws_6.pkl --seed 6 --plotfile test2cf.pdf --dictfile dict.pkl
