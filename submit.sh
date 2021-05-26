#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=5-12:00:00
singularity exec --nv --overlay /scratch/wz727/chestXR/data/mimic-cxr.sqsh:ro --overlay /scratch/wz727/chestXR/data/chestxray8.sqsh:ro --overlay /scratch/lhz209/data/padchest.sqf:ro --overlay /scratch/lhz209/data/chexpert.sqf:ro --overlay /scratch/lhz209/pytorch1.7.0-cuda11.0.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif bash -c "source /ext3/env.sh; "
# python vqvae.py --train --n_embeddings 64 --n_epochs 600 --ema --cuda 0 --dataset joint_chest --output_dir joint_dh --hosp true --restore_dir joint_dh && python vqvae_prior.py --vqvae_dir joint_dh --train --n_epochs 200 --batch_size 256 --lr 0.00005 --which_prior top  --n_cond_classes 4 --n_channels 128 --n_res_layers 5 --n_out_stack_layers 10 --n_cond_stack_layers 0 --drop_rate 0.1 --cuda 0 --output_dir joint_dh_prior --hosp true && python vqvae_prior.py --vqvae_dir joint_dh --train --n_epochs 200 --batch_size 256 --lr 0.00005 --which_prior bottom --n_cond_classes 4 --n_channels 128 --n_res_layers 20 --n_out_stack_layers 0 --n_cond_stack_layers 10 --drop_rate 0.1 --output_dir joint_dh_prior_bottom --cuda 0
# python vqvae.py --train --n_embeddings 64 --n_epochs 600 --ema --cuda 0 --dataset joint_chest --output_dir joint_d  --restore_dir joint_d && python vqvae_prior.py --vqvae_dir joint_d --train --n_epochs 200 --batch_size 256 --lr 0.00005 --which_prior top  --n_cond_classes 2 --n_channels 128 --n_res_layers 5 --n_out_stack_layers 10 --n_cond_stack_layers 0 --drop_rate 0.1 --cuda 0 --output_dir joint_d_prior && python vqvae_prior.py --vqvae_dir joint_d --train --n_epochs 200 --batch_size 256 --lr 0.00005 --which_prior bottom --n_cond_classes 2 --n_channels 128 --n_res_layers 20 --n_out_stack_layers 0 --n_cond_stack_layers 10 --drop_rate 0.1 --output_dir joint_d_prior_bottom --cuda 0
# python vqvae.py --train --n_embeddings 64 --n_epochs 600 --ema --cuda 0 --dataset joint_chest --output_dir joint_dz --cond_x_top true  --restore_dir joint_dz && python vqvae_prior.py --vqvae_dir joint_dz --train --n_epochs 200 --batch_size 256 --lr 0.00005 --which_prior top  --n_cond_classes 2 --n_channels 128 --n_res_layers 5 --n_out_stack_layers 10 --n_cond_stack_layers 0 --drop_rate 0.1 --cuda 0 --output_dir joint_dz_prior --cond_x_top true && python vqvae_prior.py --vqvae_dir joint_dz --train --n_epochs 200 --batch_size 256 --lr 0.00005 --which_prior bottom --n_cond_classes 2 --n_channels 128 --n_res_layers 20 --n_out_stack_layers 0 --n_cond_stack_layers 10 --drop_rate 0.1 --output_dir joint_dz_prior_bottom  --cuda 0 --cond_x_top true

python vqvae.py --train --n_embeddings 64 --n_epochs 60 --ema --cuda 0 --dataset joint_chest --output_dir joint_dz_quick --cond_x_top true  --restore_dir joint_dz_quick && \
python vqvae_prior.py --vqvae_dir joint_dz_quick --train --n_epochs 50 --batch_size 256 --lr 0.00005 --which_prior top  --n_cond_classes 2 --n_channels 128 --n_res_layers 5 --n_out_stack_layers 10 --n_cond_stack_layers 0 --drop_rate 0.1 --cuda 0 --output_dir joint_dz_quick_prio --cond_x_top true && \
python vqvae_prior.py --vqvae_dir joint_dz_quick --train --n_epochs 50 --batch_size 256 --lr 0.00005 --which_prior bottom --n_cond_classes 2 --n_channels 128 --n_res_layers 20 --n_out_stack_layers 0 --n_cond_stack_layers 10 --drop_rate 0.1 --output_dir joint_dz_quick_prior_bottom  --cuda 0 --cond_x_top true


python vqvae.py --train --n_embeddings 64 --n_epochs 60 --ema --cuda 0 --dataset joint_chest --output_dir joint_dz_quick_rho.8 --cond_x_top true --rho .8 && \
python vqvae_prior.py --vqvae_dir joint_dz_quick_rho.8 --train --n_epochs 50 --batch_size 256 --lr 0.00005 --which_prior top  --n_cond_classes 2 --n_channels 128 --n_res_layers 5 --n_out_stack_layers 10 --n_cond_stack_layers 0 --drop_rate 0.1 --cuda 0 --output_dir joint_dz_quick_rho.8_prior --cond_x_top true --rho .8 && \
python vqvae_prior.py --vqvae_dir joint_dz_quick_rho.8 --train --n_epochs 50 --batch_size 256 --lr 0.00005 --which_prior bottom --n_cond_classes 2 --n_channels 128 --n_res_layers 20 --n_out_stack_layers 0 --n_cond_stack_layers 10 --drop_rate 0.1 --output_dir joint_dz_quick_rho.8_prior_bottom  --cuda 0 --cond_x_top true --rho .8

python vqvae.py --train --n_embeddings 64 --n_epochs 60 --ema --cuda 0 --dataset joint_chest --output_dir joint_dz_quick_rho.8_same --cond_x_top true --rho .8 --rho_same true && \
python vqvae_prior.py --vqvae_dir joint_dz_quick_rho.8_same --train --n_epochs 50 --batch_size 256 --lr 0.00005 --which_prior top  --n_cond_classes 2 --n_channels 128 --n_res_layers 5 --n_out_stack_layers 10 --n_cond_stack_layers 0 --drop_rate 0.1 --cuda 0 --output_dir joint_dz_quick_rho.8_same_prior --cond_x_top true --rho .8 --rho_same true && \
python vqvae_prior.py --vqvae_dir joint_dz_quick_rho.8_same --train --n_epochs 50 --batch_size 256 --lr 0.00005 --which_prior bottom --n_cond_classes 2 --n_channels 128 --n_res_layers 20 --n_out_stack_layers 0 --n_cond_stack_layers 10 --drop_rate 0.1 --output_dir joint_dz_quick_rho.8_same_prior_bottom  --cuda 0 --cond_x_top true --rho .8 --rho_same true

# padchest
python vqvae.py --train --n_embeddings 64 --n_epochs 60 --ema --cuda 0 --dataset joint_chest --output_dir joint_dz_quick_padchest --cond_x top --second_dataset padchest && \
python vqvae_prior.py --vqvae_dir joint_dz_quick_padchest --train --n_epochs 50 --batch_size 256 --lr 0.00005 --which_prior top  --n_cond_classes 2 --n_channels 128 --n_res_layers 5 --n_out_stack_layers 10 --n_cond_stack_layers 0 --drop_rate 0.1 --cuda 0 --output_dir joint_dz_quick_padchest_prior --cond_x top --second_dataset padchest && \
python vqvae_prior.py --vqvae_dir joint_dz_quick_padchest --train --n_epochs 50 --batch_size 256 --lr 0.00005 --which_prior bottom --n_cond_classes 2 --n_channels 128 --n_res_layers 20 --n_out_stack_layers 0 --n_cond_stack_layers 10 --drop_rate 0.1 --output_dir joint_dz_quick_padchest_prior_bottom  --cuda 0 --cond_x top --second_dataset padchest && \
python  vqvae_prior.py --vqvae_dir joint_dz_quick_padchest --restore_dir joint_dz_quick_prior_bottom joint_dz_quick_prior --generate --n_samples 12800 --cuda 0  --n_cond_classes 2  --cond_x top --batch_size 128 --hosp true --data_output_dir generated/nurd_padchest_top --second_dataset padchest

# mimic outer
python vqvae.py --train --n_embeddings 64 --n_epochs 60 --ema --cuda 0 --dataset joint_chest --output_dir joint_dz_quick_mimic_outer --cond_x outer && \
python vqvae_prior.py --vqvae_dir joint_dz_quick_mimic_outer --train --n_epochs 50 --batch_size 256 --lr 0.00005 --which_prior top  --n_cond_classes 2 --n_channels 128 --n_res_layers 5 --n_out_stack_layers 10 --n_cond_stack_layers 0 --drop_rate 0.1 --cuda 0 --output_dir joint_dz_quick_mimic_outer_prior --cond_x outer && \
python vqvae_prior.py --vqvae_dir joint_dz_quick_mimic_outer --train --n_epochs 50 --batch_size 256 --lr 0.00005 --which_prior bottom --n_cond_classes 2 --n_channels 128 --n_res_layers 20 --n_out_stack_layers 0 --n_cond_stack_layers 10 --drop_rate 0.1 --output_dir joint_dz_quick_mimic_outer_prior_bottom  --cuda 0 --cond_x outer && \
python  vqvae_prior.py --vqvae_dir joint_dz_quick_mimic_outer --restore_dir joint_dz_quick_mimic_outer_prior_bottom joint_dz_quick_mimic_outer_prior --generate --n_samples 12800 --cuda 0  --n_cond_classes 2  --cond_x outer --batch_size 128 --hosp true --data_output_dir generated/nurd_mimic_outer

# padchest outer
python vqvae.py --train --n_embeddings 64 --n_epochs 60 --ema --cuda 0 --dataset joint_chest --output_dir joint_dz_quick_padchest_outer --cond_x outer --second_dataset padchest && \
python vqvae_prior.py --vqvae_dir joint_dz_quick_padchest_outer --train --n_epochs 50 --batch_size 256 --lr 0.00005 --which_prior top  --n_cond_classes 2 --n_channels 128 --n_res_layers 5 --n_out_stack_layers 10 --n_cond_stack_layers 0 --drop_rate 0.1 --cuda 0 --output_dir joint_dz_quick_padchest_outer_prior --cond_x outer --second_dataset padchest && \
python vqvae_prior.py --vqvae_dir joint_dz_quick_padchest_outer --train --n_epochs 50 --batch_size 256 --lr 0.00005 --which_prior bottom --n_cond_classes 2 --n_channels 128 --n_res_layers 20 --n_out_stack_layers 0 --n_cond_stack_layers 10 --drop_rate 0.1 --output_dir joint_dz_quick_padchest_outer_prior_bottom  --cuda 0 --cond_x outer --second_dataset padchest && \
python  vqvae_prior.py --vqvae_dir joint_dz_quick_padchest_outer --restore_dir joint_dz_quick_padchest_outer_prior_bottom joint_dz_quick_padchest_outer_prior --generate --n_samples 12800 --cuda 0  --n_cond_classes 2  --cond_x outer --batch_size 128 --hosp true --data_output_dir generated/nurd_padchest_outer --second_dataset padchest

# python -m torch.distributed.launch --nproc_per_node 4 --use_env \
# ERM dh
python \
  vqvae_prior.py --vqvae_dir joint_dh \
                 --restore_dir  joint_dh_prior_bottom joint_dh_prior\
                 --generate \
                 --n_samples 6250 \
                 --cuda 0 \
                 --n_cond_classes 4 \
                 --hosp true
------

# ERM d
python \
  vqvae_prior.py --vqvae_dir joint_d \
                 --restore_dir  joint_d_prior_bottom joint_d_prior\
                 --generate \
                 --n_samples 12500 \
                 --cuda 0 \
                 --n_cond_classes 2


# NURD d write cond_x_top_prior, code for prior to use x_top (doesn't make sense)
python \
  vqvae_prior.py --vqvae_dir joint_d \
                 --restore_dir  joint_d_prior_bottom joint_d_prior\
                 --generate \
                 --n_samples 12500 \
                 --cuda 0 \
                 --n_cond_classes 2 \
                 --cond_x_top_prior true


# ERM dz TODO write sampling code for y_from_data
python \
  vqvae_prior.py --vqvae_dir joint_dz_quick \
                 --restore_dir  joint_dz_quick_prior_bottom joint_dz_quick_prior\
                 --generate \
                 --n_samples 12800 \
                 --cuda 0 \
                 --n_cond_classes 2 \
                 --cond_x_top true \
                 --y_from_data true \
                 --batch_size 128 \
                 --hosp true \
                 --data_output_dir generated/vqvae_erm_quick_hosp


# NURD
python \
  vqvae_prior.py --vqvae_dir joint_dz_quick \
                 --restore_dir  joint_dz_quick_prior_bottom joint_dz_quick_prior\
                 --generate \
                 --n_samples 12800 \
                 --cuda 0 \
                 --n_cond_classes 2 \
                 --cond_x_top true \
                 --batch_size 128 \
                 --hosp true \
                 --data_output_dir generated/vqvae_nurd_quick_hosp



python -m ipdb \
  vqvae_prior.py --vqvae_dir joint_dz_quick \
                 --restore_dir  joint_dz_quick_prior_bottom joint_dz_quick_prior\
                 --generate \
                 --n_samples 128 \
                 --cuda 0 \
                 --n_cond_classes 2 \
                 --cond_x_top true \
                 --batch_size 128 \
                 --hosp true \
                 --data_output_dir generated/vqvae_nurd_quick_hosp_correct \
                 --rho .9 \
                 --rho_same
