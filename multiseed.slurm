#!/bin/bash
#SBATCH --account=ag_ifi_driemel
#SBATCH --job-name=seededRunsOutside
#SBATCH --output=out/log
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=intelsr_medium
#SBATCH --mem=900G
#SBATCH --time=1-00:00:00

module load Anaconda3

source /home/${USER}/.bashrc
source activate nonobtuse

srun --unbuffered python src/main.py --parallel=True
