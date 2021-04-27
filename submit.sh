#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=5-12:00:00
singularity exec --nv --overlay /scratch/wz727/chestXR/data/mimic-cxr.sqsh:ro --overlay /scratch/wz727/chestXR/data/chestxray8.sqsh:ro --overlay /scratch/lhz209/data/padchest.sqf:ro --overlay /scratch/lhz209/data/chexpert.sqf:ro --overlay /scratch/lhz209/pytorch1.7.0-cuda11.0.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif bash -c "source /ext3/env.sh; "
# python vqvae.py --train --n_embeddings 64 --n_epochs 500 --ema --cuda 0 --dataset joint_chest --output_dir joint_dh --hosp true --restore_dir joint_dh && python vqvae_prior.py --vqvae_dir joint_dh --train --n_epochs 20 --batch_size 256 --lr 0.00005 --which_prior top  --n_cond_classes 4 --n_channels 128 --n_res_layers 5 --n_out_stack_layers 10 --n_cond_stack_layers 0 --drop_rate 0.1 --cuda 0 --output_dir joint_dh_prior --hosp true
# python vqvae.py --train --n_embeddings 64 --n_epochs 500 --ema --cuda 0 --dataset joint_chest --output_dir joint_d  --hosp false  --restore_dir joint_d && python vqvae_prior.py --vqvae_dir joint_d --train --n_epochs 20 --batch_size 256 --lr 0.00005 --which_prior top  --n_cond_classes 2 --n_channels 128 --n_res_layers 5 --n_out_stack_layers 10 --n_cond_stack_layers 0 --drop_rate 0.1 --cuda 0 --output_dir joint_d_prior --hosp false
# python vqvae.py --train --n_embeddings 64 --n_epochs 500 --ema --cuda 0 --dataset joint_chest --output_dir joint_dz  --hosp false --cond_x_top true  --restore_dir joint_dz && python vqvae_prior.py --vqvae_dir joint_dz --train --n_epochs 20 --batch_size 256 --lr 0.00005 --which_prior top  --n_cond_classes 2 --n_channels 128 --n_res_layers 5 --n_out_stack_layers 10 --n_cond_stack_layers 0 --drop_rate 0.1 --cuda 0 --output_dir joint_dz_prior --hosp false --cond_x_top true
