#!/bin/bash -l
#SBATCH --partition=kipac
#SBATCH -t 2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mydo@stanford.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=submit.%a.out
#SBATCH --array=8

source ~/setup.sh
conda activate wfsim

python bleed.py --simdir heightRand --simfile outh_rand_$SLURM_ARRAY_TASK_ID.pkl --seed $SLURM_ARRAY_TASK_ID --dictfile bleed_rand_$SLURM_ARRAY_TASK_ID.pkl --plotfile bleed_rand_$SLURM_ARRAY_TASK_ID.pdf --plotdir .

python bleed.py --simdir heightPsfws --simfile outh_psfws_$SLURM_ARRAY_TASK_ID.pkl --seed $SLURM_ARRAY_TASK_ID --dictfile bleed_psfws_$SLURM_ARRAY_TASK_ID.pkl --plotfile bleed_psfws_$SLURM_ARRAY_TASK_ID.pdf --plotdir .
