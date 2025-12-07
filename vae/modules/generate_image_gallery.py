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


def PlotInputImgs(config, numExamples, numColumns, imgs, seg, intensity_multiplier, labels, fontSize, tif_channels, channel_color_dict, fileName, cluster_column, contrast_limits, save_dir):

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

        # Slice image patch from Zarr
        input_img = imgs[:, row, :, :]

        # Apply image contrast settings
        lower = np.array(
            [i[0] for i in contrast_limits.values()]
        ).reshape(input_img.shape[0], 1, 1)
        upper = np.array(
            [i[1] for i in contrast_limits.values()]
        ).reshape(input_img.shape[0], 1, 1)
        input_img = (input_img - lower) / (upper - lower)

        # use existing channel intensity ranges
        # lower = np.array(
        #     [input_img[i].min() for i in range(input_img.shape[0])]
        # ).reshape(input_img.shape[0], 1, 1)
        # upper = np.array(
        #     [input_img[i].max() for i in range(input_img.shape[0])]
        # ).reshape(input_img.shape[0], 1, 1)
        # input_img = (input_img - lower) / (upper - lower)

        # Slice out channels to visualize
        channel_indices = np.array(
            [tif_channels.index(i) for i in channel_color_dict.keys()]
        )
        input_img = input_img[channel_indices, :, :]

        # Segmentation outlines layer
        seg_layer = seg[0, row, :, :]
        seg_layer = img_as_float(seg_layer)
        seg_rgb = np.zeros((seg_layer.shape[0], seg_layer.shape[1], 4))  # RGBA
        seg_rgb[..., :3] = [1, 1, 1]  # color the full RGB array white
        seg_rgb[..., 3] = seg_layer  # only show values >0

        # Centroid marker layer
        patch_height, patch_width = (imgs.shape[2], imgs.shape[3])
        centroid_layer = np.zeros((patch_height, patch_width, 4))  # RGBA
        cy, cx = int(patch_height / 2), int(patch_width / 2)
        centroid_layer[cy, cx, :3] = [1, 1, 1]  # color the full RGB array white
        centroid_layer[cy, cx, 3] = 1  # only show the centroid value (>0)

        # for RGB images
        if config.RGB:
            overlay = np.transpose(input_img, (1, 2, 0))
            plt.imshow(overlay)
        else:
            # Convert to RGB, brighten, and colorize
            input_img = gray2rgb(input_img)
            input_img *= intensity_multiplier
            color_arr = np.array(
                [to_rgb(color) for _, color in channel_color_dict.items()]
            ).reshape(-1, 1, 1, 3)
            input_img *= color_arr

            # Sum images along channels axis to generate final RGB image patch
            overlay = np.sum(input_img, axis=0)
            overlay = np.clip(overlay, 0, 1)
            plt.imshow(overlay, cmap=plt.cm.binary)
        
        for name, color in channel_color_dict.items():

            custom_lines.append(Line2D([0], [0], color=color, lw=5))

        label = data['Sample']  # cluster_column

        # plt.imshow(seg_rgb, alpha=0.4)
        plt.imshow(centroid_layer)
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

        # read test labels
        csv_path = os.path.join(cellcutter_input_path, 'test_qc.csv')
        csv = pd.read_csv(csv_path)
        csv['Sample'] = csv['Sample'].astype(str)

        # Read test patches
        zip_store_path = os.path.join(
            cellcutter_output_path, 
            f'test_patches_{config.window_size}_qc.zip'
        )
        z = zarr.open(zarr.ZipStore(zip_store_path), mode='r')
        
        # Read test segmentation patches
        zip_store_path_seg = os.path.join(
            cellcutter_output_path, 
            f'test_patches_{config.window_size}_qc_seg.zip'
        )
        z_seg = zarr.open(zarr.ZipStore(zip_store_path_seg), mode='r')

        # Contrast settings
        contrast_limits = yaml.safe_load(
            open(config.contrast_path)
        )['setContrast']

        # Ensure keys in config.tif_channel order
        contrast_limits = {k: contrast_limits[k] for k in config.tif_channels}

        # Pull random patches from training data to check quality
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
            config=config,
            numExamples=config.gallery_size,
            numColumns=20,
            imgs=imgs,
            seg=seg,
            intensity_multiplier=1.0,
            labels=labels,
            fontSize=5,
            tif_channels=config.tif_channels,
            channel_color_dict=config.channel_colors,
            fileName='patch_examples',
            cluster_column=config.cluster_column,
            contrast_limits=contrast_limits,
            save_dir=save_dir
        )
