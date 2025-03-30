import json
import math
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2
from pylab import *
from scipy.optimize import curve_fit
from sklearn import mixture
import matplotlib.pyplot
import matplotlib.mlab
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
import h5py
from itertools import groupby
from operator import itemgetter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks, argrelextrema, savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from sklearn.cluster import AffinityPropagation
import seaborn as sns
import random
import subprocess

def get_tail_angles(df_tail, heading):
    xy = df_tail.values[:, ::2] + df_tail.values[:, 1::2] * 1j
    midline = -np.exp(1j * np.deg2rad(np.asarray(heading)))
    return -np.angle(np.diff(xy, axis=1) / midline[:, None])

def low_pass_filt(x, fs, cutoff, axis=0, order=2):

    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, x, axis=axis)

def get_bouts(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    bouts = []
    for line in lines:
        bouts.append([int(line.split('\n')[0].split()[0]), int(line.split('\n')[0].split()[1])])

    return bouts

def combine_episodes(events, gap=50):
    new_episodes = []
    start_index = 0
    start_episode = events[start_index]
    start = start_episode[0]
    end = start_episode[1]
    if events.shape[0] >= 2:
        for i in range(1, events.shape[0]):
            temp_start = events[i][0]
            temp_end = events[i][1]

            # if i < events.shape[0] - 1:
            if temp_start - end < gap:
                end = temp_end
                if i == events.shape[0] - 1:
                    new_episodes.append([start, end])
            elif temp_start - end >= gap:
                new_episodes.append([start, end])
                start = temp_start
                end = temp_end

                if i == events.shape[0] - 1:
                    new_episodes.append([start, end])
    else:
        new_episodes.append([start, end])
    return np.array(new_episodes)

def find_start_point(peak_index_1, peak_index_2, convolved, antimode_value):
    indices_before = [i for i in range(peak_index_1, peak_index_2 + 1)]
    index_temp = []
    index_temp_d = []
    for i in np.argwhere(convolved[indices_before] <= antimode_value): #derivative or convolved?
        index_temp.append(i[0])
    if not index_temp:
        final_start = peak_index_2 - 1
    else:
        rough_start = indices_before[index_temp[-1]]
        indices_between = [j for j in range(rough_start, peak_index_2 + 1)]
        for q in np.argwhere(convolved[indices_between] >= antimode_value):
            index_temp_d.append(q)
        if not index_temp_d:
            final_start = rough_start
        else:
            # print(index_temp_d)
            final_start = indices_between[index_temp_d[0][0]]

    return final_start

def find_start_point_v2(peak_index_1, peak_index_2, local_minimums):
    final_start = max([i for i in local_minimums if peak_index_1 < i < peak_index_2])

    return final_start

def find_nearest_bigger(A, B, num_list, convolved_result, antimode_value):
    # Filter the list to get only integers larger than A
    larger_numbers = [num for num in num_list if (A < num < B)]

    if not larger_numbers:
        print('no minima between: ', A, 'and', B)
        return None  # Return None if no larger number is found
    else:
        try:
            # stop_point = [num for num in np.arange(A, B) if
            #               # (convolved_result[num] >= -0.05) and (convolved_result[num - 1] <= -0.05)][0] - 1
            stop_point = [num for num in np.arange(A, B) if (convolved_result[num] <= antimode_value)][0]
        except:
            stop_point = B - 1

        return stop_point

def find_nearest_bigger_v2(A, B, num_list, convolved_result):
    # Filter the list to get only integers larger than A
    larger_numbers = [num for num in num_list if (A < num < B)]

    if not larger_numbers:
        print('no minima between: ', A, 'and', B)
        return None  # Return None if no larger number is found
    else:
        try:
            stop_point = [num for num in np.arange(A, B) if
                          (convolved_result[num] >= 0) and (convolved_result[num - 1] <= 0)][0] - 1
        except:
            stop_point = B - 1

        return stop_point

def find_nearest_bigger_v3(A, B, num_list, convolved_result, antimode_value):
    # Filter the list to get only integers larger than A
    larger_numbers = [num for num in num_list if (A < num < B)]

    if not larger_numbers:
        print('no minima between: ', A, 'and', B)
        return None  # Return None if no larger number is found
    else:
        try:
            stop_point_1 = min([num for num in larger_numbers])
            stop_point_2 = [num for num in np.arange(A, B) if (convolved_result[num] <= antimode_value)][0]

            stop_point = max([stop_point_1, stop_point_2])
        except:
            stop_point = B - 1

        return stop_point

def get_behavior_timestamps(filename):
    t = np.loadtxt(filename)[:, 1]
    t = (t - t[0]) / 3515839
    return t

def moving_average(a, n=4):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def find_valid_periods(arr, t=None):
    # Ensure the array is a NumPy array
    arr = np.array(arr)

    # Find where values are above 0.5
    above_0_5 = arr > 0.5

    # Find the segments of continuous True values
    segments = np.split(np.where(above_0_5)[0], np.where(np.diff(np.where(above_0_5)[0]) != 1)[0] + 1)

    valid_indexes = []

    for segment in segments:
        if len(segment) > 0:
            # Check if at least half are above 0.8
            if np.sum(arr[segment] > 0.6) >= len(segment) / 2:
                valid_indexes.append(segment)

    return valid_indexes

def superimpose_annotation(video_name, fps=200, ranges=None, output_path=None, frame_path=None):

    # Get a list of image file names in the specified folder
    image_folder = os.path.join(frame_path, video_name)
    if ranges is None:
        name = video_name
    else:
        name = f'{video_name}_{ranges[0]}_{ranges[1]}.mp4'

    print(f'Start superimposing: {name}')
    if output_path is None:
        output_video_path = f'D:/visual_fist_sift/annotated_visual_videos/{name}'
    else:
        output_video_path = os.path.join(output_path, name)

    total_frames = int(len(os.listdir(image_folder)))

    if ranges:
        image_files = [f'{i}.jpg' for i in range(ranges[0], min(ranges[1]+1, total_frames))]
    else:
        image_files = [f'{i}.jpg' for i in range(total_frames)]

    with h5py.File('D:/visual_fist_sift/sleap_tracked_results_h5/'+ video_name + '.h5', 'r') as f:
        tracks_matrix = f['tracks'][:]

    head_up_x = tracks_matrix[:,0,0,:].T.flatten()
    head_up_y = tracks_matrix[:,1,0,:].T.flatten()

    eye_mid_x = tracks_matrix[:,0,1,:].T.flatten()
    eye_mid_y = tracks_matrix[:,1,1,:].T.flatten()

    jaw_front_x = tracks_matrix[:,0,2,:].T.flatten()
    jaw_front_y = tracks_matrix[:,1,2,:].T.flatten()

    jaw_btm_x = tracks_matrix[:,0,3,:].T.flatten()
    jaw_btm_y = tracks_matrix[:,1,3,:].T.flatten()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4
    video_writer = None
    # Process each image
    count = 0
    for img_file in image_files:
        # print(f'Processing {video_name} {img_file}.')
        # Load the image
        temp_image_path = f'{image_folder}/{img_file}'
        image = cv2.imread(temp_image_path)

        # Check if image is loaded successfully
        if image is None:
            print(f"Error loading image: {img_file}")
            continue

        # Initialize video writer with the first image dimensions
        if video_writer is None:
            frame_height, frame_width, _ = image.shape
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))  # 1 frame per second

        try:
            coord_head_up = [int(head_up_x[count]), int(head_up_y[count])]
            coord_eye_mid = [int(eye_mid_x[count]), int(eye_mid_y[count])]
            coord_jaw_front = [int(jaw_front_x[count]), int(jaw_front_y[count])]
            coord_jaw_btm = [int(jaw_btm_x[count]), int(jaw_btm_y[count])]

            cv2.circle(image, coord_head_up, radius=5, color=(0, 0, 255), thickness=1)
            cv2.circle(image, coord_eye_mid, radius=5, color=(0, 0, 255), thickness=1)
            cv2.circle(image, coord_jaw_front, radius=5, color=(0, 0, 255), thickness=1)
            cv2.circle(image, coord_jaw_btm, radius=5, color=(0, 0, 255), thickness=1)
        except:
            continue


        # Write the modified image to the video
        video_writer.write(image)

        count += 1

    # Release the video writer
    video_writer.release()

    print(f"Video saved as {output_video_path}")
