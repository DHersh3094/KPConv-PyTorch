import os
import uuid
import sys
import pandas as pd
# import open3d as o3d
from collections import defaultdict
import numpy as np
from datetime import datetime
import laspy as lp
from laspy import ExtraBytesParams
import subprocess
import glob
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.size'] = 14
import shutil
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler
from jakteristics import las_utils, compute_features, FEATURE_NAMES
from tqdm import tqdm
import re
import tempfile

def copy_folder(config, sample=True):
    input_folder = config.input_folder
    copied_folder = config.copied_folder
    figure_save_dir = config.dataset_dir
    min_point_threshold = config.min_point_threshold
    max_point_threshold = config.max_point_threshold
    plot = True
    redo = True

    # sample = getattr(config, "sample", False)

    """
    Copy single tree ALS files (ending with _g meaning ground classified) for a specific list of trees if they have a minimum point count.
    """
    if not os.path.exists(figure_save_dir):
        os.makedirs(figure_save_dir)

    if not os.path.exists(config.dataset_dir):
        os.mkdir(config.dataset_dir)

    if redo:

        try:
            if not os.path.exists(copied_folder):
                os.makedirs(copied_folder)
        except OSError:
            print(f'Unable to make folder')

        # Only copy certain species
        species_to_copy = list(config.labels_to_names.values())

        files = []
        species_counter = Counter()

        for root, dirs, filenames in os.walk(input_folder):
            for filename in filenames:
                try:
                    species_name = filename.split('_')[0]

                    # Study area BR06 has an issue with ground. Do not copy for further processing
                    # File format: FagSyl_BR02_04_2019-07-05_q2_ALS-on_g.laz
                    study_area = filename.split('_')[1]  # e.g. BR02

                    if filename.endswith('ALS-on_g.laz') and species_name in species_to_copy and study_area != "BR06":
                        # Limit if sampling
                        if sample and species_counter[species_name] >= 20:
                            continue

                        las = lp.read(os.path.join(root, filename))
                        number_of_nonground_points = len(las.points[las.classification != 2])
                        if number_of_nonground_points >= min_point_threshold and number_of_nonground_points <= max_point_threshold:
                            try:
                                species_counter[species_name] += 1
                                files.append(os.path.join(root, filename))  # Store full file paths
                                shutil.copyfile(os.path.join(root, filename), os.path.join(copied_folder, filename))
                            except Exception as e:
                                print(f'Error copying {filename}: {e}')
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    pass

        print(f'Copying {len(files)} files')
        print(f'Unique species: {species_counter}')

        # Calculate classes
        classes = sorted(list(species_counter.keys()))
        y = []
        for cls, count in species_counter.items():
            y.extend([cls] * count)
        y = np.array(y)
        class_weights = compute_class_weight('balanced',
                                             classes=np.array(classes),
                                             y=y)
        class_weights = list(class_weights)
        config.class_weights = class_weights
        print(f'Classes: {classes}')
        print(f'Class weights: {class_weights}')

        # Save class weights to .txt file to be loaded during training because copying folder and augmentation only needs to occur once in some cases
        class_weights_file = os.path.join(config.dataset_dir, 'class_weights.npy')
        np.save(class_weights_file, np.array(class_weights))

        if plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.bar(species_counter.keys(), species_counter.values())
            plt.ylabel('Number of trees')
            plt.xlabel('Species')
            plt.grid()
            plt.title(f"Number of trees with more than {min_point_threshold} non-ground points")
            plt.savefig(os.path.join(figure_save_dir, f'species_count_greaterthan_{min_point_threshold}_points.png'),
                        bbox_inches='tight')

        return copied_folder

    else:
        print('Skipping copying...')