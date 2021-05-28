import os
import sys
import shutil
import json
from datetime import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader

from tensorboardX import SummaryWriter

from subprocess import check_call


def set_writer(log_path, comment='', restore=False):
    """ setup a tensorboardx summarywriter """
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if restore:
        log_path = os.path.dirname(log_path)
    else:
        log_path = os.path.join(log_path, current_time + comment)
    writer = SummaryWriter(log_dir=log_path)
    return writer


def save_checkpoint(state, checkpoint, is_best=None, quiet=False):
    """ saves model and training params at checkpoint + 'last.pt'; if is_best also saves checkpoint + 'best.pt'

    args
        state -- dict; with keys model_state_dict, optimizer_state_dict, epoch, scheduler_state_dict, etc
        is_best -- bool; true if best model seen so far
        checkpoint -- str; folder where params are to be saved
    """

    filepath = os.path.join(checkpoint, 'state_checkpoint.pt')
    if not os.path.exists(checkpoint):
        if not quiet:
            print('Checkpoint directory does not exist Making directory {}'.format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)

#    if is_best:
#        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_state_checkpoint.pt'))

    if not quiet:
        print('Checkpoint saved.')


def load_checkpoint(checkpoint, models, optimizers=None, scheduler=None, best_metric=None, map_location='cpu'):
    """ loads model state_dict from filepath; if optimizer and lr_scheduler provided also loads them

    args
        checkpoint -- string of filename
        model -- torch nn.Module model
        optimizer -- torch.optim instance to resume from checkpoint
        lr_scheduler -- torch.optim.lr_scheduler instance to resume from checkpoint
    """

    if not os.path.exists(checkpoint):
        raise('File does not exist {}'.format(checkpoint))

    checkpoint = torch.load(checkpoint, map_location=map_location)
    models = [m.load_state_dict(checkpoint['model_state_dicts'][i]) for i, m in enumerate(models)]

    if optimizers:
        try:
            optimizers = [o.load_state_dict(checkpoint['optimizer_state_dicts'][i]) for i, o in enumerate(optimizers)]
        except KeyError:
            print('No optimizer state dict in checkpoint file')

    if best_metric:
        try:
            best_metric = checkpoint['best_val_acc']
        except KeyError:
            print('No best validation accuracy recorded in checkpoint file.')

    if scheduler:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except KeyError:
            print('No lr scheduler state dict in checkpoint file')

    return checkpoint['epoch']

def holemask(x, h=24, w=24, SIDE=32):
    """ Returns x with a hole of zeros """
    N_CHANNELS = 1
    margin_h = (SIDE-h)//2
    margin_w = (SIDE-w)//2

    mask = torch.ones(1, N_CHANNELS, SIDE, SIDE)
    mask[:,:,margin_h : h + margin_h, margin_w : w + margin_w] = 0
    mask_gpu = mask.to(x.device)
    # x with hole of zeros
    return x * mask_gpu

def onlycenter(x, h=24, w=24, SIDE=32):
    N_CHANNELS = 1
    margin_h = (SIDE-h)//2
    margin_w = (SIDE-w)//2

    mask = torch.zeros(1, 1, SIDE, SIDE)
    mask[:,:,margin_h : h + margin_h, margin_w : w + margin_w] = 1
    mask_gpu = mask.to(x.device)
    return x*mask_gpu

def load_precreated_data(args, mode="train", include="x"):
    """
    include: "x", "xy", "xyh"
    """
    tensordata = torch.load(f'/scratch/apm470/nuisance-orthogonal-prediction/code/nrd-xray/erm-on-generated/joint_chexpert_{args.second_dataset}_dataset_rho09_saved_{mode}.pt')
    if mode == "val":
        # 0 is test, 1 is val
        tensordata = tensordata[1]
    x = tensordata.tensors[0]
    y = tensordata.tensors[1]
    h = tensordata.tensors[2]
    y_oh = torch.zeros(y.shape[0], 2)
    y_oh[range(y.shape[0]), y.squeeze(1).long()] = 1
    if include == "xyh":
        x_dataset = TensorDataset(x, y_oh, h)
    elif include == "xy":
        x_dataset = TensorDataset(x, y_oh)
    elif include == "x":
        x_dataset = x
    else:
        raise ValueError(f"include not supported: {include}")
    x_dataloader = DataLoader(x_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=('cuda' in args.device))
    return x_dataloader