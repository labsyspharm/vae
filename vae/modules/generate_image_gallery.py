import os
import yaml
import math
import logging

import zarr

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib import pyplot as plt

from skimage.color import gray2rgb
from skimage.util import img_as_float

from ..utils import log_banner, log_multiline

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# log_multiline(logger.info, pd.DataFrame().to_string(index=False))
# log_banner(logger.info, 'Boolean classifications')


def PlotInputImgs(numExamples, numColumns, imgs, seg, intensity_multiplier, labels, fontSize, tif_channels, channel_color_dict, fileName, cluster_column, contrast_limits, save_dir):

    numRows = math.ceil(numExamples / numColumns)
    grid_dims = (numRows, numColumns)

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(20, 10))

    custom_lines = []
    for e, (row, data) in enumerate(labels.iterrows()):

        plt.subplot(grid_dims[0], grid_dims[1], e + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
       
        # initialize array of zeros with shape of full-size image
        overlay = np.zeros((imgs.shape[2], imgs.shape[3]))

        # add centroid point at the center of the image
        overlay[
            int(imgs.shape[2] / 2):int(imgs.shape[2] / 2) + 1,
            int(imgs.shape[3] / 2):int(imgs.shape[3] / 2) + 1
        ] = 1
        
        overlay = gray2rgb(overlay)
        
        for name, color in channel_color_dict.items():
            
            ch = tif_channels.index(name)
            lyr = imgs[ch, row, :, :]

            # lyr = lyr.astype('float')  # use for binary patches
            
            lyr = img_as_float(lyr)

            # apply image contrast settings
            if str(imgs.dtype) == 'uint16':
                divisor = 65535
            elif str(imgs.dtype) == 'uint8':
                divisor = 255
            else:
                raise ValueError(f'Unsupported image dtype: {imgs.dtype}')

            lyr -= (contrast_limits[name][0] / divisor)
            lyr /= (
                (contrast_limits[name][1] / divisor) - 
                (contrast_limits[name][0] / divisor))

            lyr = np.clip(lyr, 0, 1)

            lyr = gray2rgb(lyr)

            lyr = lyr * intensity_multiplier
            lyr = lyr * to_rgb(color)
            overlay += lyr

            custom_lines.append(Line2D([0], [0], color=color, lw=5))

        # select segmentation outlines slice
        seg_slice = seg[0, row, :, :]

        # ensure segmentation outlines are normalized 0-1
        seg_slice = (seg_slice - np.min(seg_slice)) / np.ptp(seg_slice)

        # convert segmentation thumbnail to RGB
        seg_slice = gray2rgb(seg_slice) * 0.25  # decrease alpha

        overlay += seg_slice

        label = data[cluster_column]
        
        overlay = np.clip(overlay, 0, 1)
        plt.imshow(overlay, cmap=plt.cm.binary)
        plt.xlabel(label, size=fontSize, labelpad=1.5)

    legend_elements = []
    for name, color in channel_color_dict.items():
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

    if not os.path.exists(
        os.path.join(config.output_path, 
                     'checkpoints/GENERATE_IMAGE_GALLERY.txt')):
        
        cellcutter_input_path = os.path.join(
            config.output_path, '1_cellcutter_input'
        )

        cellcutter_output_path = os.path.join(
            config.output_path, f'3_cellcutter_output_win{config.window_size}'
        )

        save_dir = os.path.join(config.output_path, '4_patch_examples')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # read training labels
        csv_path = os.path.join(cellcutter_input_path, 'test_qc.csv')
        csv = pd.read_csv(csv_path)
        csv['Sample'] = csv['Sample'].astype(str)

        # read training images
        zip_store_path = os.path.join(
            cellcutter_output_path, 
            f'test_patches_{config.window_size}_qc.zip'
        )
        z = zarr.open(zarr.ZipStore(zip_store_path), mode='r')
        
        # read segmentation thumbnails for test data (16-bit unsigned integer)
        zip_store_path_seg = os.path.join(
            cellcutter_output_path, 
            f'test_patches_{config.window_size}_qc_seg.zip'
        )
        z_seg = zarr.open(zarr.ZipStore(zip_store_path_seg), mode='r')

        # contrast settings
        contrast_limits = yaml.safe_load(open(config.contrast_path))

        # pull random thumbnails from training data to check quality
        patch_ids = np.random.RandomState(1).choice(
            range(0, z.shape[1]), config.gallery_size, replace=False
        )

        imgs = z.get_orthogonal_selection((slice(None), patch_ids))
        seg = z_seg.get_orthogonal_selection((slice(None), patch_ids))
        
        labels = csv.iloc[patch_ids].copy()
        labels.reset_index(drop=True, inplace=True)
        if config.cluster_column:
            labels.sort_values(by=config.cluster_column, inplace=True)

        PlotInputImgs(
            numExamples=config.gallery_size,
            numColumns=20,
            imgs=imgs,
            seg=seg,
            intensity_multiplier=1.1,
            labels=labels,
            fontSize=2,
            tif_channels=config.tif_channels,
            channel_color_dict=config.channel_colors,
            fileName='patch_examples',
            cluster_column=config.cluster_column,
            contrast_limits=contrast_limits,
            save_dir=save_dir
        )
