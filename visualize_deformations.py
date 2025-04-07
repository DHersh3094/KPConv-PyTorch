#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#

# Usage example:
# 1. --path is the path to the data
# 2. --chosen_log is the model folder (goings to /checkpoints)
# python visualize_deformations.py \
# --path /media/davidhersh/T7/Data/DataApr3_v2/data_2 \
# --chosen_log /media/davidhersh/T7/Data/DataApr3_v2/results_minsubsample_0.25_deformable_kp_15_2025_04_05_10_22/data_2

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
import numpy as np
import sys
import torch
import argparse

# Dataset
from datasets.ModelNet40 import *
from datasets.S3DIS import *
from datasets.NeuesPalaisTrees import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.visualizer import ModelVisualizer
from models.architectures import KPCNN, KPFCNN
from models.blocks import KPConv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KP_CLOUDS_DIR = os.path.join(BASE_DIR, 'KP_clouds')

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def model_choice(chosen_log):

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


def print_points_for_each_deform_idx(net, loader, config):
    """
    Run a single forward pass on the first batch and print
    how many points go into each deformable KPConv layer.
    """
    net.eval()  # Set the network to eval mode (no dropout, etc.)

    # We only need one batch to see the shape of the data
    for batch_idx, batch in enumerate(loader):
        # Forward pass
        _ = net(batch, config)

        # Loop through all modules (layers) of the network
        deform_i = 0
        for module in net.modules():
            # Check if it's a KPConv and if it's deformable
            if isinstance(module, KPConv) and module.deformable:
                # The shape of `module.min_d2` is typically [B, num_kernel_points]
                # But each row corresponds to the number of input points used here
                # So the batch dimension => #points in that layer
                n_points = module.min_d2.shape[0] if module.min_d2 is not None else 0

                print(f"Deform idx={deform_i}, #points={n_points}")
                deform_i += 1

        # Break so we only look at one batch
        break

def print_layer_point_counts(loader):
    """
    Print how many points are in each layer for the first batch.
    This shows how the dataset/dataloader progressively subsamples points.
    """
    for batch_idx, batch in enumerate(loader):
        for l, pts_l in enumerate(batch.points):
            print(f"Layer {l}: {pts_l.shape[0]} points")
        break


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, help='Path to data (eg. data_2')
    parser.add_argument('--chosen_log', type=str, help='To results.../data_2')
    args = parser.parse_args()

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > 'results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    # chosen_log = '/media/davidhersh/T7 Shield/DataJan5/results_minsubsample_0.4/data_2_subsample_0.4'
    # chosen_log = '/media/davidhersh/T7\ Shield/JanuaryResults/Jan17/results_minsubsample_0.2_deformable_kp_15_drad_3.0_2025_01_20_13_26/data_1'
    chosen_log = args.chosen_log
    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    # Eventually you can choose which feature is visualized (index of the deform convolution in the network)
    deform_idx = 0

    # Deal with 'last_XXX' choices
    chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.path = '/media/davidhersh/T7/Data/DataApr3_v2/data_2'
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    # config.augment_noise = 1
    # config.batch_num = 1
    # config.in_radius = 2.0
    # config.input_threads = 0

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initiate dataset
    if config.dataset.startswith('ModelNet40'):
        test_dataset = ModelNet40Dataset(config, train=False)
        test_sampler = ModelNet40Sampler(test_dataset)
        collate_fn = ModelNet40Collate
    elif config.dataset.startswith('NeuesPalaisTrees'):
        test_dataset = NeuesPalaisTreesDataset(config, mode='train')
        test_sampler = NeuesPalaisTreesSampler(test_dataset)
        collate_fn = NeuesPalaisTreesCollate
    
    elif config.dataset == 'S3DIS':
        test_dataset = S3DISDataset(config, set='validation', use_potentials=True)
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)
    print_layer_point_counts(test_loader)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    if config.dataset_task == 'classification':
        net = KPCNN(config)
        print_points_for_each_deform_idx(net, test_loader, config)
    elif config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError('Unsupported dataset_task for deformation visu: ' + config.dataset_task)

    # Define a visualizer class
    visualizer = ModelVisualizer(net, config, chkp_path=chosen_chkp, on_gpu=False)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart visualization')
    print('*******************')

    # Training
    visualizer.show_deformable_kernels(net, test_loader, config, deform_idx)




