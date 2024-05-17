#!/bin/sh
#
#SBATCH --job-name="job_name"
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load python
module load openmpi
module load py-torch
module load py-pip
python -m pip install --user accelerate
python -m pip install --user transformers
python -m pip install --user datasets
python -m pip install --user wandb

srun python ~/CSE3000/models/gpt-neo.py /scratch/eugenewu/CSE3000/data/TokenizedTinyStories ~/CSE3000/results ~/CSE3000/10k-tok ~/CSE3000/wandb_key.txt