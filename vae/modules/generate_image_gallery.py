import os
import yaml
import logging

import numpy as np
import pandas as pd

import math

import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib import pyplot as plt

from skimage.color import gray2rgb
from skimage.util import img_as_float

import zarr

from ..utils import log_banner, log_multiline

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# log_multiline(logger.info, pd.DataFrame().to_string(index=False))
# log_banner(logger.info, 'Boolean classifications')


def PlotInputImgs(numExamples, numColumns, imgs, seg, intensity_multiplier, labels, fontSize, channel_color_dict, fileName, contrast_limits, save_dir):

    numRows = math.ceil(numExamples / numColumns)
    grid_dims = (numRows, numColumns)

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(20, 10))

    custom_lines = []
    for e, row in enumerate(labels.iterrows()):

        plt.subplot(grid_dims[0], grid_dims[1], e + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
       
        # initialize array of zeros with shape of full-size image
        overlay = np.zeros((imgs.shape[2], imgs.shape[3]))

        # add centroid point at the center of the image
        # overlay[
        #     int(imgs.shape[2] / 2):int(imgs.shape[2] / 2) + 1,
        #     int(imgs.shape[3] / 2):int(imgs.shape[3] / 2) + 1
        # ] = 1
        
        overlay = gray2rgb(overlay)
        
        for name, (ch, color) in channel_color_dict.items():

            lyr = imgs[ch, row[0], :, :]

            # lyr = lyr.astype('float') # use for binary patches
            
            lyr = img_as_float(lyr)

            # apply image contrast settings
            lyr -= (contrast_limits[name][0] / 65535)
            lyr /= (
                (contrast_limits[name][1] / 65535) - 
                (contrast_limits[name][0] / 65535))

            lyr = np.clip(lyr, 0, 1)

            lyr = gray2rgb(lyr)

            lyr = lyr * intensity_multiplier
            
            lyr = lyr * to_rgb(color)
            overlay += lyr

            custom_lines.append(Line2D([0], [0], color=color, lw=5))

            # select segmentation outlines slice
            seg_slice = seg[0, row[0], :, :]

            # ensure segmentation outlines are normalized 0-1
            seg_slice = (seg_slice - np.min(seg_slice)) / np.ptp(seg_slice)

            # convert segmentation thumbnail to RGB
            seg_slice = gray2rgb(seg_slice) * 0.25  # decrease alpha

        # overlay += seg_slice

        label = row[1]['cluster_3d']

        plt.imshow(overlay, cmap=plt.cm.binary)
        plt.xlabel(label, size=fontSize, labelpad=1.5)

    legend_elements = []
    for name, (ch, color) in channel_color_dict.items():
        legend_elements.append(Line2D([0], [0], color=color, lw=5, label=name))

    fig.legend(
        handles=legend_elements, prop={'size': 11}, 
        bbox_to_anchor=(0.94, 0.99)
    )

    plt.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.85)
    plt.savefig(
        os.path.join(save_dir, f'{fileName}.png'), dpi=800, bbox_inches='tight'
    )
    plt.close('all')


def GENERATE_IMAGE_GALLERY(config):

    cellcutter_input_path = os.path.join(
        config.output_path, '1_cellcutter_input'
    )

    cellcutter_output_path = os.path.join(
        config.output_path, f'2_cellcutter_output_win{config.window_size}'
    )

    save_dir = os.path.join(config.output_path, '3_thumbnail_examples')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(
      os.path.join(config.output_path, 
                   'checkpoints/GENERATE_IMAGE_GALLERY.txt')):

        # read training labels
        labels_path = os.path.join(cellcutter_input_path, 'test.csv')
        labels = pd.read_csv(labels_path)

        # read training images
        z_path = os.path.join(
            cellcutter_output_path, f'test_thumbnails_{config.window_size}.zip'
        )
        store = zarr.ZipStore(z_path, mode='r')
        z = zarr.open(store=store)

        # read segmentation thumbnails for test data (16-bit unsigned integer)
        z1_test_path_seg = os.path.join(
            cellcutter_output_path, 
            f'test_thumbnails_{config.window_size}_seg.zip'
        )
        store = zarr.ZipStore(z1_test_path_seg, mode='r')
        z_seg = zarr.open(store=store)

        # contrast settings
        contrast_limits = yaml.safe_load(open(config.contrast_path))

        # pull random thumbnails from training data to check quality
        thumb_ids = np.random.RandomState(1).choice(
            range(0, z.shape[1]), config.gallery_size, replace=False
        )

        imgs = z.get_orthogonal_selection((slice(None), thumb_ids))
        seg = z_seg.get_orthogonal_selection((slice(None), thumb_ids))

        labels = labels.iloc[thumb_ids]
        labels.reset_index(drop=True, inplace=True)
        labels.sort_values(by='cluster_3d', inplace=True)

        PlotInputImgs(
            numExamples=config.gallery_size,
            numColumns=20,
            imgs=imgs,
            seg=seg,
            intensity_multiplier=1.1,
            labels=labels,
            fontSize=8,
            channel_color_dict=config.channel_colors,
            fileName='thumbnail_examples',
            contrast_limits=contrast_limits,
            save_dir=save_dir
        )
