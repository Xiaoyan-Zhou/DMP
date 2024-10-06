#!/bin/bash
#SBATCH --job-name=DMPL_K40
#SBATCH --account=Project_2002243
#SBATCH --partition=gpusmall
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
## if local fast disk on a node is also needed, replace above line with:
## Please remember to load the environment your application may need.
## And use the variable $LOCAL_SCRATCH in your batch job script 
## to access the local fast storage on each node. #

module load pytorch/2.0
srun python 0-main.py --filter --uncertainty_th 0.9 --loss_init HierarchicalClustering --K 40 --batch_size 32 --lr_model 0.01 --lr_pro 0.5 --centroids_path ./init_model_npy/resnet18_CE_10_K40_HierarchicalClustering_centroids.npy --label_path ./init_model_npy/resnet18_CE_10_K40_HierarchicalClustering_labels.npy