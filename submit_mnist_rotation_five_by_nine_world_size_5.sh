#!/bin/bash
#SBATCH --array=0-14
#SBATCH --partition=shared
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_outputs/mnist_rotation_five_by_nine_world_size_5.log   # Standard output and error log
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=spandan_madan@g.harvard.edu

eval "$(conda shell.bash hook)"
conda activate domain_adaptation
bash mnist_rotation_five_by_nine_world_size_5.sh
