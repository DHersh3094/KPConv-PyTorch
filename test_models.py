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

# Dataset
from datasets.ModelNet40 import *
from datasets.S3DIS import *
from datasets.SensatUrban import *
from datasets.SemanticKitti import *
from datasets.Toronto3D import *
from datasets.Apple import *
from datasets.NeuesPalais import *
from datasets.NeuesPalaisTrees import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS', 'last_sensaturban']:

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

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS', 'last_SensatUrban']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':
    import argparse
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = argparse.ArgumentParser()

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    # chosen_log = 'results/Log_2024-09-15_18-18-11'
    # chosen_log = "results/Log_2024-09-17_13-07-55"
    # chosen_log = "Dec11_v33_0.4subsample_kpsubsample_0.4"
    # chosen_log = '/media/davidhersh/T7 Shield/results_2024_12_12_17_44/data_1_subsample_0.4'
    parser.add_argument('--results_path', type=str, default=None, help='Results path')
    parser.add_argument('--data_path', type=str, default=None, help='Data path')
    parser.add_argument('--do_grid_subsample', help='Enable subsampling or not')
    parser.add_argument('--first_subsampling_dl', type=float, default=None, help='The KPConv subsampling value')
    parser.add_argument('--num_test_models', type=int,  help='# of trees')

    # parser.add_argument('--test_saving_path', type=str, default=None, help='Test saving path')

    args = parser.parse_args()
    #
    # if args.results_path is not None:
    #     chosen_log = args.results_path    

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    # Choose to test on validation or test split
    on_val = True

    # Deal with 'last_XXXXXX' choices
    # chosen_log = model_choice(chosen_log)
    chosen_log = args.results_path

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
    config.load(chosen_log)

    if args.data_path is not None:
        config.path = args.data_path
        
    # Handle do_grid_subsample
    if args.do_grid_subsample is not None:
        config.do_grid_subsample = True if args.do_grid_subsample.lower() == 'true' else False
    else:
        # Default disabled
        config.do_grid_subsample = True

    if args.first_subsampling_dl is not None:
        config.first_subsampling_dl = args.first_subsampling_dl

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    #config.batch_num = 3
    #config.in_radius = 4
    # config.max_epoch = 10 # I set this to make sure things run
    config.validation_size = 512 # Starting point: 200
    config.input_threads = 32 # Starting point: 10

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'ModelNet40':
        test_dataset = ModelNet40Dataset(config, train=False)
        test_sampler = ModelNet40Sampler(test_dataset)
        collate_fn = ModelNet40Collate
    if config.dataset == 'NeuesPalaisTrees':
        test_dataset = NeuesPalaisTreesDataset(config, mode='test',  num_test_models = int(args.num_test_models))
        test_sampler = NeuesPalaisTreesSampler(test_dataset)
        collate_fn = NeuesPalaisTreesCollate
    elif config.dataset == 'S3DIS':
        test_dataset = S3DISDataset(config, set='validation', use_potentials=True)
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    elif config.dataset == 'SensatUrban':
        test_dataset = SensatUrbanDataset(config, set='validation', use_potentials=True)
        test_sampler = SensatUrbanSampler(test_dataset)
        collate_fn = SensatUrbanCollate
    elif config.dataset == 'Toronto3D':
        test_dataset = Toronto3DDataset(config, set='test', use_potentials=True)
        test_sampler = Toronto3DSampler(test_dataset)
        collate_fn = Toronto3DCollate

    elif config.dataset == 'Apple':
        test_dataset = AppleDataset(config, set='validation', use_potentials=True)
        test_sampler = AppleSampler(test_dataset)
        collate_fn = AppleCollate

    elif config.dataset == 'NeuesPalais':
        test_dataset = NeuesPalaisDataset(config, set='validation', use_potentials=True)
        test_sampler = NeuesPalaisSampler(test_dataset)
        collate_fn = NeuesPalaisCollate

    elif config.dataset == 'SemanticKitti':
        test_dataset = SemanticKittiDataset(config, set=set, balance_classes=False)
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = SemanticKittiCollate

    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    if config.dataset_task == 'classification':
        net = KPCNN(config)
    elif config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp, output_folder=chosen_log)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Training
    if config.dataset_task == 'classification':
        tester.classification_test(net, test_loader, config)
    elif config.dataset_task == 'cloud_segmentation':
        tester.cloud_segmentation_test(net, test_loader, config)
    elif config.dataset_task == 'slam_segmentation':
        tester.slam_segmentation_test(net, test_loader, config)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)
