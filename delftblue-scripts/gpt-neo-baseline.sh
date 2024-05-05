#!/bin/sh
#
#SBATCH --job-name="job_name"
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load cuda/11.6
module load python
module load py-transformers
module load py-datasets
module load py-wandb

srun python ~/CSE3000/models/gpt-neo.py /scratch/eugenewu/CSE3000/data/TokenizedTinyStories ~/CSE3000/results ~/CSE3000/10k-tok ~/CSE3000/wandb_key.txt