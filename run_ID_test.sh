#!/bin/bash
#SBATCH --job-name=ID_TEST
#SBATCH --account=Project_2002243
#SBATCH --partition=gpusmall
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
## if local fast disk on a node is also needed, replace above line with:
## Please remember to load the environment your application may need.
## And use the variable $LOCAL_SCRATCH in your batch job script 
## to access the local fast storage on each node.

module load pytorch/2.0
srun python ID_test.py --model_path filter_True_lr_pro_0.5_lr_model_0.01_batch_size_32/model --save_excel ./results/ID_acc.xlsx