#!/bin/bash
#SBATCH --account=ag_ifi_driemel
#SBATCH --job-name=gigaCombiner
#SBATCH --output=out/log3
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=lm_long
#SBATCH --mem=1000G
#SBATCH --time=7-00:00:00

module load Anaconda3

source /home/${USER}/.bashrc
source activate nonobtuse

srun --unbuffered python src/main.py --parallel=True
