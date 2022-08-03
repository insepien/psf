#!/bin/bash -l
#SBATCH --partition=kipac
#SBATCH -t 2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mydo@stanford.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=submit.%a.out
#SBATCH --array=6-10

source ~/setup.sh
conda activate wfsim

python fit_2pcfs.py --simdir heightRand --simfile outh_rand_$SLURM_ARRAY_TASK_ID.pkl --seed $SLURM_ARRAY_TASK_ID --dictfile dict_rand_$SLURM_ARRAY_TASK_ID.pkl --plotfile plt2c_rand_$SLURM_ARRAY_TASK_ID.pdf --plotdir plt2c_rand
