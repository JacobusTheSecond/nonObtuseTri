#!/bin/bash
#SBATCH --account=ag_ifi_driemel
#SBATCH --job-name=seededRunsOutside
#SBATCH --output=out/logDepth1
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=lm_long
#SBATCH --mem=900G
#SBATCH --time=5-00:00:00

module load Anaconda3

source /home/${USER}/.bashrc
source activate nonobtuse

srun --unbuffered python src/main.py --parallel=True
