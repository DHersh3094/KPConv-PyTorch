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

'''
Augmentation functions
'''

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
            
        # os.remove(las_file)


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
    
def decimate(config, las_file):
    decimation_percentage = config.decimation_percentage
    decimation_runs = config.decimation_runs

    las = lp.read(las_file)
    points = np.vstack((las.x, las.y, las.z)).transpose()

    for i in range(decimation_runs):
        # Calculate the number of points to sample based on the percentage
        num_points = points.shape[0]
        # print(f'Total number of points: {num_points}')
        point_samples = int(num_points * (decimation_percentage / 100.0))
        # print(f'Point samples: {point_samples}')

        # Randomly sample the points
        indices = np.random.choice(num_points, point_samples, replace=False)
        decimated_points = las.points[indices]

        header = lp.LasHeader(point_format=las.header.point_format, version=las.header.version)
        header.offsets = las.header.offsets
        header.scales = las.header.scales

        decimated_las = lp.LasData(header)
        decimated_las.points = decimated_points

        for dim_name in las.point_format.dimension_names:
            original_array = getattr(las, dim_name)
            decimated_array = original_array[indices]
            setattr(decimated_las, dim_name, decimated_array)

        output_file = las_file.replace('.laz', f'_decim_{i}.laz')

        decimated_las.write(output_file)

    # os.remove(las_file)
    
def jitter(config, las_file):
    amount = config.jitter_amount
    las = lp.read(las_file)
    xyz = np.vstack((las.x, las.y, las.z)).T
    
    #Bounds 
    min_bound = np.min(xyz, axis=0)
    max_bound = np.max(xyz, axis=0)
    extent = max_bound - min_bound
    noise = np.random.rand(*xyz.shape) * extent * amount
    xyz_noisy = xyz + noise
    
    new_header = lp.LasHeader(point_format=las.header.point_format, version=las.header.version)
    new_header.scales = las.header.scales
    new_header.offsets = las.header.offsets

    new_las = lp.LasData(new_header)

    new_las.x = xyz_noisy[:, 0]
    new_las.y = xyz_noisy[:, 1]
    new_las.z = xyz_noisy[:, 2]

    for dim_name in las.point_format.dimension_names:
        if dim_name not in ["X", "Y", "Z"]:
            setattr(new_las, dim_name, getattr(las, dim_name))

    new_las_name = las_file.replace('.laz', '_j.laz')
    new_las.write(new_las_name)
    
    os.remove(las_file)

def normalize_intensity(config, las_file):
    if 'intensity' in config.features:
        las = lp.read(las_file)

        # Skip if 'NormalizedIntensity' already exists
        if "NormalizedIntensity" in las.point_format.extra_dimension_names:
            print(f"'NormalizedIntensity' already exists in {las_file}, skipping.")
            return

        intensities = las.intensity.astype(np.float64)

        scalar = MinMaxScaler(feature_range=(0,1))

        intensities_reshaped = intensities.reshape(-1,1)
        normalized_intensities = scalar.fit_transform(intensities_reshaped).flatten()
        new_dim = ExtraBytesParams(name='NormalizedIntensity', type='float32')
        las.add_extra_dim(new_dim)
        las.NormalizedIntensity = normalized_intensities.astype(np.float32)
        las.write(las_file)
    else:
        pass