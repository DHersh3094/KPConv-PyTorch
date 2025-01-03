#!/usr/bin/env python
# coding: utf-8

# In[84]:


import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import laspy as lp
import subprocess
import glob
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
import shutil
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from jakteristics import las_utils, compute_features, FEATURE_NAMES
from tqdm import tqdm
import re
import tempfile

class PipelineConfig:
    def __init__(self, input_folder, copied_folder, dataset_dir, point_threshold, n_splits,
                 min_subsample_distance, rotations, normals_search_radius,
                 max_epochs, first_kpconv_subsampling_dl, saving_path):
        self.input_folder = input_folder
        self.copied_folder = copied_folder
        self.dataset_dir = dataset_dir
        self.point_threshold = point_threshold
        self.n_splits = n_splits
        self.min_subsample_distance = min_subsample_distance
        self.normals_search_radius = normals_search_radius
        self.rotations = rotations
        self.normals_search_radius = normals_search_radius
        self.max_epochs = max_epochs
        self.first_kpconv_subsampling_dl = first_kpconv_subsampling_dl

        self.train_folders = None
        self.test_folders = None

        self.num_train_files = None
        self.num_test_files = None

        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.saving_path = os.path.join(saving_path, f'results_{self.timestamp}_minsubsample_{min_subsample_distance}')
        os.makedirs(self.saving_path, exist_ok=True)


# In[88]:


def copy_folder(config):
    input_folder = config.input_folder
    copied_folder=config.copied_folder
    point_threshold=config.point_threshold
    plot = True
    redo = True
    """
    Copy single tree ALS files (ending with _g meaning ground classified) for a specific list of trees if they have a minimum point count
    """
    if redo:

        try:
            if not os.path.exists(copied_folder):
                os.makedirs(copied_folder)
        except OSError:
            print(f'Unable to make folder')
        # Save figs
        figdir = copied_folder.replace('/tmp', '/processing_figures')
        if not os.path.exists(figdir):
            os.makedirs(figdir)
        
        # Only copy certain species
        species_to_copy = [
            "PseMen",
            "FagSyl",
            "PinSyl",
            "QueRub",
            "PicAbi",
            "QuePet"
        ]
    
        files = []
        species_counter = Counter()
        
        for root, dirs, filenames in os.walk(input_folder):
            for filename in filenames:
                try:
                    species_name = filename.split('_')[0]
        
                    # Study area BR06 has an issue with ground. Do not copy for further processing
                    # File format: FagSyl_BR02_04_2019-07-05_q2_ALS-on_g.laz
                    study_area = filename.split('_')[1] # BR02
        
                    if filename.endswith('ALS-on_g.laz') and species_name in species_to_copy and study_area != "BR06":
                            las = lp.read(os.path.join(root, filename))
                            number_of_nonground_points = len(las.points[las.classification !=2])
                            if number_of_nonground_points <= point_threshold:

                                try:
                                    species_counter[species_name] += 1
                                    files.append(os.path.join(root, filename))  # Store full file paths
                                    shutil.copyfile(os.path.join(root, filename), os.path.join(copied_folder, filename))
                                except:
                                    print(f'Error copying.')
                except: 
                    pass
        print(f'Copying {len(files)} files')
        print(f'Unique species: {species_counter}')
        
        if plot:
            fig, ax = plt.subplots(figsize=(12,8))
            plt.bar(species_counter.keys(), species_counter.values())
            plt.ylabel('Number of trees')
            plt.xlabel('Species')
            plt.title(f"Number of trees with more than {point_threshold} non-ground points")
            plt.savefig(os.path.join(figdir, f'species_count_greaterthan_{point_threshold}_points.png'), bbox_inches='tight')
            # plt.show()
        
        return copied_folder
    
    else:
        print(f'Skipping copying...')


# In[89]:


# copied_als_folder, species_counter = copy_folder(input_ALS_folder, output_folder='/tmp', point_threshold=0.5, plot=False, redo=True)


# In[90]:


def calculate_hag(las_file):
    las = lp.read(las_file)
    x = las.x
    y = las.y
    z = las.z
    
    classification = las.classification
    
    ground_indices = classification == 2
    x_ground = x[ground_indices]
    y_ground = y[ground_indices]
    z_ground = z[ground_indices]
    
    X, Y = np.meshgrid(x_ground, y_ground)
    ground_interpolated = LinearNDInterpolator(list(zip(x_ground, y_ground)), z_ground)
    Z = ground_interpolated(X, Y)
    z_ground_at_points = ground_interpolated(x, y)
    min_ground = np.nanmin(z_ground)
    z_ground_at_points = np.where(np.isnan(z_ground_at_points), min_ground, z_ground_at_points)
    hag = z - z_ground_at_points
    las.z = hag
    las = las[las.classification != 2]
    
    las.write(las_file)


# In[91]:


# Can do HAG calculation before split

def convert_hag_to_z(config, redo=True):
    input_folder = config.copied_folder # Run on copied files

    if redo:
        files = os.listdir(input_folder)
        for file in tqdm(files, desc='Processing files', unit='file'):
            full_path = os.path.join(input_folder, file)
            calculate_hag(full_path)
    else:
        print("Skipping HAG step...")
        
# convert_hag_to_z(input_folder=copied_als_folder)

def calculate_normals(config, las_file):
    normals_search_radius = config.normals_search_radius
    las = lp.read(las_file)
    FEATURE_NAMES = ['nx', 'ny', 'nz']

    # Remove nx, ny, nz if existing
    for feature_name in FEATURE_NAMES:
        if feature_name in las.point_format.dimension_names:
            las.point_format.remove_extra_dimension(feature_name)

    xyz = las_utils.read_las_xyz(las_file)

    features = compute_features(xyz, search_radius=normals_search_radius, feature_names=FEATURE_NAMES)

    output_file = las_file.replace('.laz', '_normals.laz')

    if not os.path.exists(output_file):
        las_utils.write_with_extra_dims(las_file, output_file, features, FEATURE_NAMES)
        os.remove(las_file)


# In[93]:


def stratified_k_fold_split(config):
    input_folder = config.copied_folder
    output_folder = input_folder.replace('/tmp', '/kfolders')
    n_splits = config.n_splits

    files = []
    labels = []
    
    for file in os.listdir(input_folder):
        class_name = file.split('_')[0] #Fagsyl etc...
        files.append(os.path.join(input_folder, file))
        labels.append(class_name)
    class_counts = Counter(labels)
    print(f'Class counts: {class_counts}')
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=32)
    
    train_folders = []
    test_folders = []
    
    for fold, (train_index, test_index) in enumerate(skf.split(files, labels)):
        fold_train_dir = os.path.join(output_folder, f'fold_{fold+1}_train')
        train_folders.append(fold_train_dir)
        fold_test_dir = os.path.join(output_folder, f'fold_{fold+1}_val')
        test_folders.append(fold_test_dir)
        os.makedirs(fold_train_dir, exist_ok=True)
        os.makedirs(fold_test_dir, exist_ok=True)
        
        for train_idx in train_index:
            shutil.copy(files[train_idx], fold_train_dir)
        for test_idx in test_index:
            shutil.copy(files[test_idx], fold_test_dir)
            
    config.train_folders = train_folders
    config.test_folders = test_folders
    return train_folders, test_folders


# In[94]:


def poisson_subsample(config, las_file):
    min_distance = config.min_subsample_distance
    las = lp.read(las_file)
    points = np.vstack((las.x, las.y, las.z)).transpose()

    kdtree = cKDTree(points)

    selected = np.zeros(len(points), dtype=bool)
    selected_indices = []
    for i, point in enumerate(points):
        if selected[i]:
            continue

        selected_indices.append(i)
        selected[i] = True

        indices = kdtree.query_ball_point(point, min_distance)
        selected[indices] = True

    header = lp.LasHeader(point_format=las.header.point_format, version=las.header.version)
    header.offsets = las.header.offsets
    header.scales = las.header.scales

    subsampled_las = lp.LasData(header)
    subsampled_las.points = las.points[selected_indices]

    subsampled_las.write(las_file)


# In[95]:


def normalize_xy(las_file):
        las = lp.read(las_file)
        original_offset_x = las.header.offsets[0]
        original_offset_y = las.header.offsets[1]
        scale_x = las.header.scales[0]
        scale_y = las.header.scales[1]

        x = las.x
        y = las.y

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # Normalize coordinates
        normalized_x = x - mean_x
        normalized_y = y - mean_y

        new_header = lp.LasHeader(point_format=las.header.point_format, version=las.header.version)
        new_header.scales = las.header.scales

        new_header.offsets = [
            original_offset_x - mean_x,
            original_offset_y - mean_y,
            las.header.offsets[2]  # Same z
        ]

        new_las = lp.LasData(new_header)
        new_las.x = normalized_x
        new_las.y = normalized_y
        new_las.z = las.z

        for dim_name in las.point_format.dimension_names:
            if dim_name not in ["X", "Y", "Z"]:
                setattr(new_las, dim_name, getattr(las, dim_name))

        new_las.write(las_file)


# In[96]:


def rotate_z(pointcloud, degrees):
    theta = np.deg2rad(degrees)

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    return np.dot(pointcloud, rotation_matrix.T)


def rotate_las(las_file, rotations):
        las = lp.read(las_file)

        point_data = np.vstack((las.x, las.y, las.z)).T

        quadrant_ranges = []
        step = 360 // rotations
        for i in range(rotations):
            quadrant_ranges.append((i * step, (i + 1) * step))

        random_rotations = [np.random.uniform(low, high) for low, high in quadrant_ranges]

        for i, rotation in enumerate(random_rotations):

            rotated_points = rotate_z(point_data, rotation)

            header = lp.LasHeader(point_format=las.header.point_format, version=las.header.version)
            header.offsets = las.header.offsets
            header.scales = las.header.scales

            rotated_las = lp.LasData(header)
            rotated_las.x = rotated_points[:, 0]
            rotated_las.y = rotated_points[:, 1]
            rotated_las.z = rotated_points[:, 2]

            for dim_name in las.point_format.dimension_names:
                if dim_name not in ["X", "Y", "Z"]:
                    setattr(rotated_las, dim_name, getattr(las, dim_name))

            base_folder = os.path.dirname(las_file)
            base_name = os.path.basename(las_file).replace('.laz', f'_rot_{int(rotation)}.laz')
            output_file = os.path.join(base_folder, base_name)

            rotated_las.write(output_file)

# In[98]:


# Augment each folder

def augmentation(config):
     rotations=config.rotations
     train_folders = config.train_folders
     test_folders = config.test_folders
     min_subsample_distance = config.min_subsample_distance

     all_folders = train_folders + test_folders
     for folder in all_folders:
        for las_file in os.listdir(folder):
            if las_file.endswith('.laz'):

                normalize_xy(las_file=os.path.join(folder, las_file))

                poisson_subsample(config, las_file=os.path.join(folder, las_file))

                rotate_las(las_file=os.path.join(folder, las_file), rotations=rotations)


     for folder in all_folders:
        for las_file in os.listdir(folder):
            if las_file.endswith('.laz'):
                calculate_normals(config, os.path.join(folder, las_file))


# In[99]:


# Need to deal with nan normal values
def convert_to_txt(config):
    train_folders = config.train_folders
    test_folders = config.test_folders
    all_folders = train_folders + test_folders

    for folder in all_folders:
        for las_file in os.listdir(folder):
            if not las_file.endswith(('laz', 'las')):
                continue
            las = lp.read(os.path.join(folder, las_file))
            output_file = las_file.replace('.laz', '.txt')
            
            with open(os.path.join(folder, output_file), 'w') as f:
                for x, y, z, nx, ny, nz, intensity in zip(las.x, las.y, las.z, las.nx, las.ny, las.nz, las.intensity):
                    f.write(f'{x:.6f}, {y:.6f}, {z:.6f}, {nx}, {ny}, {nz}, {intensity}\n')


# In[100]:


# Convert to KPConv repository dataset format

def copy_to_datasets(config):
    train_folders = config.train_folders
    test_folders = config.test_folders
    dataset_dir = config.dataset_dir
    n_splits = config.n_splits
    min_subsample_distance = config.min_subsample_distance

    all_folders = train_folders + test_folders
    base_folder = train_folders[0]
    base_folder_test = test_folders[0]
    print(f'base test folder: {base_folder_test}')
    print(base_folder)
    
    # Get the folders of the species
    species_set = set()
    for folder in all_folders:
        for las_file in os.listdir(folder):
            species = las_file.split('_')[0]
            species_set.add(species)
    print(f'species_set: {species_set}')
    
    all_data_dirs = []
    for i in range(1, n_splits+1):
        
        # Make a folder for each dataset
        datadir = os.path.join(dataset_dir, f'data_{i}_subsample_{min_subsample_distance}')
        all_data_dirs.append(datadir)
        os.makedirs(datadir, exist_ok=True)
        
        # Make a subfolder for each species
        species_dirs = {species: os.path.join(datadir, species) for species in species_set}
        for species_dir in species_dirs.values():
            os.makedirs(species_dir, exist_ok=True)
        
        # Create filelist.txt, train.txt and test.txt files in the folder
        filelist_txt_file = os.path.join(datadir, 'filelist.txt')
        train_txt_file = os.path.join(datadir, 'train.txt')
        test_txt_file = os.path.join(datadir, 'test.txt')
        
        # Process training files
        train_folder = re.sub(r"fold_\d+_train", f'fold_{i}_train', base_folder)
        with open (train_txt_file, 'w') as f:
            for file in os.listdir(train_folder):
                if file.endswith('.txt'):
                    f.write(file[:-4] + '\n')
                else:
                    pass
            # print(f'Processed: {i} train files')
        
        for file in os.listdir(train_folder):
            species = file.split('_')[0]
            if file.endswith('.txt'):
                shutil.copy(os.path.join(train_folder, file), species_dirs[species])
            else:
                pass
                
        # Process test files
        test_folder = re.sub(r"fold_\d+_val", f'fold_{i}_val', base_folder_test)
        test_files = [file for file in os.listdir(test_folder) if file.endswith('.txt')]
        with open(test_txt_file, 'w') as f:
            for file in os.listdir(test_folder):
                if file.endswith('.txt'):
                    f.write(file[:-4] + '\n')
                else:
                    pass
                
        for file in os.listdir(test_folder):
            species = file.split('_')[0]
            if file.endswith('.txt'):
                shutil.copy(os.path.join(test_folder, file), species_dirs[species])
            else:
                pass
                
        with open (filelist_txt_file, 'w') as f:
            for species_dir in species_dirs.values():
                for file in os.listdir(species_dir):
                    relative_path = os.path.join(os.path.basename(species_dir), file)
                    f.write(relative_path + '\n')
        
    return all_data_dirs
        
# all_data_dirs = copy_to_datasets()
# Loop over each dataset and run trainNeuesPalaisTrees.py
# Need to pass config parameters here and update class_w


# In[102]:


def calculate_number_of_train_and_test_files(data_folder):
    train_path = os.path.join(data_folder, 'train.txt')
    test_path = os.path.join(data_folder, 'test.txt')

    with open(train_path) as f:
        num_train_files = len(f.readlines())
    
    with open(test_path) as f:
        num_test_files = len(f.readlines())
    return num_train_files, num_test_files


# In[103]:


# KPConv parameters
# parser.add_argument('--do_subsample', action='store_false', help='Enable subsampling or not')
# parser.add_argument('--first_subsampling_dl', type=float, default=None, help='The KPConv subsampling value')
# parser.add_argument('--max_epoch', type=int, default=None, help='Override max epoch')
# parser.add_argument('--data_path', type=str, default=None, help='Data path')
# parser.add_argument('--saving_path', type=str, default=None, help='Saving path')

# Need to add the len of files
# parser.add_argument('--num_train_models', type=int, help='# of trees')
# parser.add_argument('--num_test_models', type=int,  help='# of trees')
# 
# if args.num_train_models and self.mode == 'train':
#     self.num_models = args.num_train_models
# if args.num_test_models and self.mode == 'test':
#     self.num_models = args.num_test_models

def run_training(config, args=None):
    all_data_dirs = [os.path.join(config.dataset_dir, f"data_{i}_subsample_{config.min_subsample_distance}")
                     for i in range(1, config.n_splits + 1)]
    saving_path = config.saving_path
    print(f'Saving path: {saving_path}')
    min_subsample_distance = config.min_subsample_distance
    rotations = config.rotations
    n_splits = config.n_splits
    first_kpconv_subsampling_dl = config.first_kpconv_subsampling_dl
    max_epochs = config.max_epochs

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    # Save all parameters in results
    parameters = os.path.join(saving_path, 'parameters.txt')
    logfile = os.path.join(saving_path, 'logs.txt')

    with open(parameters, 'w') as f:
        f.write('Augmentation parameters:\n')
        f.write('-----------------------\n')
        f.write(f'Minimum distance between points: {min_subsample_distance}\n')
        f.write(f'Rotations: {rotations}\n')
        f.write('\n')

        # k-fold
        f.write(f'k-folds: {n_splits}\n')
        f.write('\n')

        #KPConv values
        f.write('KPConv Parameters:\n')
        f.write('-----------------------\n')
        f.write(f'max_epoch: {max_epochs}\n')
        f.write(f'first_kpconv_subsampling_dl: {first_kpconv_subsampling_dl}\n')


    with open(logfile, 'a') as logfile:

        for folder in all_data_dirs:
            print(f'folder: {folder}')
            # Save each fold within the results
            kfold_folder_name = folder.split('/')[-1]
            kfold_folder_path = os.path.join(saving_path, kfold_folder_name) # Save path
            if not os.path.exists(kfold_folder_path):
                os.makedirs(kfold_folder_path, exist_ok=True)

            # Set num train,test files
            num_train_files, num_test_files = calculate_number_of_train_and_test_files(folder)

            args = ['--max_epoch',str(max_epochs),
                    '--first_subsampling_dl',str(first_kpconv_subsampling_dl),
                    '--data_path', folder,
                    '--saving_path', kfold_folder_path,
                    '--num_train_models', str(num_train_files),
                    '--num_test_models', str(num_test_files)]

            process = subprocess.Popen(
                ['python', 'train_NeuesPalaisTrees.py'] + args,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            for line in process.stdout:
                print(line, end='')
                logfile.write(line)
            process.wait()
            logfile.write(f"Completed {kfold_folder_name}\n")

# In[104]:


def plot_train_and_val_accuracy_for_all_folds(config, num_classes=6):
    saving_path = config.saving_path
    folders = [os.path.join(saving_path, f) for f in os.listdir(saving_path) if os.path.isdir(os.path.join(saving_path, f))]

    fold=0
    alpha = .2
    fig, ax = plt.subplots(figsize=(14, 8))
    for folder in folders:
        training_file = os.path.join(folder, 'training.txt')
        validation_file = os.path.join(folder, 'val_confs.txt')
        df = pd.read_csv(training_file, sep='\s+')
        df = df.groupby('epochs').mean().reset_index()
        ax.plot(df['epochs'], df['train_accuracy'] * 100, 'k', label=f'Train: {fold+1}', alpha=alpha)

        # Val plot
        with open(validation_file, 'r') as f:
            val_confs = f.readlines()
        val_accuracies = []
        for val in val_confs:
            matrix = np.array(list(map(int, val.split()))).reshape(num_classes, num_classes)
            accuracy = np.sum(np.diag(matrix)) / np.sum(matrix)
            val_accuracies.append(accuracy * 100)
        ax.plot(df['epochs'], val_accuracies, label=f'Val: {fold+1}', color='b', linestyle='--', alpha=alpha)

        fold += 1
        alpha +=.2


    plt.legend(title='Metric and fold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.ylim([0, 100])
    plt.title(f'Training and validation accuracy for {n_splits} folds')
    plt.grid()
    plt.savefig(os.path.join(saving_path, 'train_accuracy.png'), bbox_inches='tight', dpi=400)
    plt.show();

    return folders

# results_folders = plot_train_and_val_accuracy_for_all_folds(saving_path);


# In[18]:


# parser.add_argument('--results_path', type=str, default=None, help='Results path')

# Pass each of the folders after training to test_models.py
# Within each folder, a new folder /test will be saved with test confusion matrices
# Should add an arg for the tester.py num_votes value. Current is 10 votes

# parser.add_argument('--results_path', type=str, default=None, help='Results path')

# # Pass each of the folders after training to test_models.py
# # Within each folder, a new folder /test will be saved with test confusion matrices
# # Should add an arg for the tester.py num_votes value. Current is 10 votes
#
# def test_models(data_folders):
#     for data_dir in data_folders:
#         print(results_folders)
#         args = ['--data_path', data_dir]
#
#     results = subprocess.run(
#         ['python', 'test_models.py'] + args,
#         capture_output=True,
#         text=True
#     )
#
#     test_results_folders = [os.path.join(saving_path, folder) for folder in saving_path]
#
#     return test_results_folders


# test_results_folders = test_models(data_folders)

def runpipeline(config):
    print(f'Copying to {config.copied_folder}')
    # copied_als_folder = copy_folder(config)

    print(f'\nConverting HAG to z')
    # convert_hag_to_z(config)

    print(f'\nSplitting into {config.n_splits} folds')
    train_folders, test_folders = stratified_k_fold_split(config)

    print(f'\nAugmenting')
    # augmentation(config)

    print(f'\nConverting to txt')
    # convert_to_txt(config)

    print(f'\nCopying to datasets')
    copy_to_datasets(config)

    print(f'\nRunning training')
    run_training(config)

def main():

    config = PipelineConfig(
    input_folder='/home/davidhersh/Dropbox/Uni/ThesisHersh/ALS_data',
    copied_folder = '/media/davidhersh/T7 Shield/pre-processing/tmp_Jan3',
    dataset_dir = '/media/davidhersh/T7 Shield/Datasets_Jan3',
    saving_path='/media/davidhersh/T7 Shield',
    # Running each
    max_epochs = 100,
    first_kpconv_subsampling_dl = 0.4,
    point_threshold = 500,
    # k-fold
    n_splits = 5,
    # Augmentation values
    min_subsample_distance = 0.2,
    rotations = 4,
    normals_search_radius = 2
    )

    runpipeline(config)

    # copied_als_folder, counter = copy_folder(input_folder=input_ALS_folder, output_folder=output_folder,
    #                                                  point_threshold=point_threshold)
    # convert_hag_to_z(input_folder=copied_als_folder)
    # #
    # train_folders, test_folders = stratified_k_fold_split(
    #     input_folder=copied_als_folder,
    #     n_splits=n_splits
    # )
    # augmentation(train_folders=train_folders,
    #              test_folders=test_folders,
    #              min_subsample_distance=min_subsample_distance,
    #              rotations=rotations,
    #              normals_search_radius=normals_search_radius,
    #              output_folder=output_folder)
    #
    # convert_to_txt(train_folders=train_folders, test_folders=test_folders)
    # all_data_dirs = copy_to_datasets(train_folders=train_folders, test_folders=test_folders, dataset_dir=dataset_dir)
    # run_training(script_name='train_NeuesPalaisTrees.py', args=[], all_data_dirs=all_data_dirs)
    # results_folder = plot_train_and_val_accuracy_for_all_folds(saving_path=saving_path)


if __name__ == '__main__':
    main()
# In[ ]:




