#!/bin/bash
#SBATCH --account=ag_ifi_driemel
#SBATCH --job-name=gigaCombiner
#SBATCH --output=out/log_big_True_5
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=intelsr_long
#SBATCH --mem=800G
#SBATCH --time=7-00:00:00

module load Anaconda3

source /home/${USER}/.bashrc
source activate nonobtuse

srun --unbuffered python src/main.py --parallel --queue --num=5
