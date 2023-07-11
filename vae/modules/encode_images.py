import logging

import os
import sys
from datetime import datetime
import yaml
import pickle

import pandas as pd
import numpy as np
import math

from itertools import product
from natsort import natsorted

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.colors import ListedColormap
from matplotlib import colors
from matplotlib.colors import to_rgb
import matplotlib.gridspec as gridspec

from skimage.color import gray2rgb

from lazy_ops import DatasetView
import zarr

from keras.models import load_model

from sklearn.manifold import TSNE
from umap import UMAP
import hdbscan
from joblib import Memory

from ..utils import log_banner, log_multiline

logger = logging.getLogger(__name__)
# log_multiline(logger.info, pd.DataFrame().to_string(index=False))


def transposeZarr(z):
    """Re-organizes thumbnail dimensions as expected VAE input
       (i.e. cells, x, y, channels)."""
    view = DatasetView(z)
    result = view.lazy_transpose([1, 2, 3, 0])

    return result


def EncodeImgs(X, encoder):

    X_encoded = encoder.predict(X)

    return X_encoded


def reverse_log(cutoffs, channel_slice, channel_name, contrast_limits):
    """Reverses percentile normalization and log10-transformation,
       pixel outliers remained clipped)."""

    lower_cutoff_log, upper_cutoff_log = cutoffs[channel_name]

    # reverse percentile normalization
    channel_slice = (
        (((upper_cutoff_log-lower_cutoff_log)*(channel_slice-0)) /
         (1-0)
         ) + lower_cutoff_log)

    # reverse log10-transform
    channel_slice = np.rint(10 ** channel_slice)

    # Normalize linear pixel values between lower and upper percentile bounds
    # lower = np.rint(10**lower_cutoff_log)
    # upper = np.rint(10**upper_cutoff_log)
    # channel_slice = (channel_slice-lower) / (upper-lower)

    # Normalize linear pixel values between lower and upper contrast settings
    lower = contrast_limits[channel_name][0]
    upper = contrast_limits[channel_name][1]
    channel_slice = (channel_slice-lower) / (upper-lower)

    return channel_slice


def DecodeVectors(decoder, cutoffs, contrast_limits, X_encoded, X_seg, orig_input_dims, channel_color_dict, intensity_multiplier):

    # initialize a numpy array to store reconstructed thumbnails
    X_decoded = np.empty(
        shape=(0, orig_input_dims[0], orig_input_dims[1], 3))

    for encode, seg in zip(X_encoded, X_seg):

        # select segmentation outlines slice
        seg_slice = seg[:, :, 0]

        # ensure segmentation outlines are normalized 0-1
        seg_slice = (seg_slice - np.min(seg_slice))/np.ptp(seg_slice)

        # convert segmentation thumbnail to RGB
        # and add to blank image
        seg_slice = gray2rgb(seg_slice) * 0.25  # decrease alpha

        z_sample = np.array([encode])

        decoded = decoder.predict(z_sample)

        reconstructed_img = decoded.reshape(
            orig_input_dims[0], orig_input_dims[1], orig_input_dims[2])

        # initialize image overlay
        overlay = np.zeros(
            (reconstructed_img.shape[0],
             reconstructed_img.shape[1]))

        # add centroid point at the center of the image
        overlay[
            int(reconstructed_img.shape[0]/2):int(
                reconstructed_img.shape[0]/2)+1,
            int(reconstructed_img.shape[1]/2):int(
                reconstructed_img.shape[1]/2)+1
            ] = 1

        overlay = gray2rgb(overlay)

        for name, (ch, color) in channel_color_dict.items():

            channel_slice = reconstructed_img[:, :, ch]

            channel_slice = reverse_log(cutoffs, channel_slice, name, contrast_limits)

            channel_slice = gray2rgb(channel_slice)

            channel_slice = channel_slice * intensity_multiplier

            overlay += channel_slice * to_rgb(color)

        overlay += seg_slice

        overlay = overlay.reshape(
            (1, orig_input_dims[0], orig_input_dims[1], 3)
            )

        X_decoded = np.concatenate((X_decoded, overlay), axis=0)

    return X_decoded


def ScatterReconstructions(X_decoded, X_encoded_embedded, zoom, ax):

    def imscatter(x, y, ax, imageData, zoom):

        images = []
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            img = imageData[i]
            image = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(
                image, (x0, y0), xycoords='data', frameon=False)
            images.append(ax.add_artist(ab))

        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    imscatter(
        X_encoded_embedded[:, 0], X_encoded_embedded[:, 1],
        imageData=X_decoded, ax=ax, zoom=zoom)


def PlotLatentSpace(reconstructions, zoom, X_encoded_embedded, X_decoded, y, channel_color_dict, scatter_point_size, filename, save_dir):

    fig, ax = plt.subplots(figsize=(10, 10))

    if reconstructions:

        ScatterReconstructions(
            X_decoded=X_decoded, X_encoded_embedded=X_encoded_embedded,
            zoom=zoom, ax=ax
            )

        custom_lines = []
        for name, (ch, color) in channel_color_dict.items():

            custom_lines.append(
                Line2D([0], [0], color=color, lw=6, label=name)
                )

        ax.scatter(
            X_encoded_embedded[:, 0],
            X_encoded_embedded[:, 1],
            c='k', s=0.0, ec='k', lw=0.25, zorder=4)

        plt.legend(
            handles=custom_lines, prop={'size': 11}, labelspacing=0.7,
            bbox_to_anchor=(1.22, 1.0)
            )

        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=800)
        plt.close('all')

    else:
        # if VAE clustering
        if isinstance(y, np.ndarray):
            num_labels = len(np.unique(y))
            num_colors = plt.cm.get_cmap('tab20').N
            palette_multiplier = math.ceil(num_labels/num_colors)
            palette = []
            palette.extend(list(plt.cm.get_cmap('tab20').colors))
            palette = palette * palette_multiplier
            palette.insert(0, (0.0, 0.0, 0.0))
            trim = len(palette)-num_labels
            palette = palette[:-trim]

            label_color_dict = dict(zip(sorted(np.unique(y)), palette))
            c = [label_color_dict[i] for i in y]

            legend_elements = []
            for lbl, color in label_color_dict.items():

                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w',
                           label=lbl, markerfacecolor=color,
                           markeredgecolor='k', lw=0.25, markersize=7)
                           )

            plt.scatter(
                X_encoded_embedded[:, 0],
                X_encoded_embedded[:, 1],
                c=c, ec='k', lw=0.0, s=scatter_point_size
                )

            plt.legend(
                handles=legend_elements, fontsize=9.2, labelspacing=0.005,
                bbox_to_anchor=(1.12, 1.01)
                )
        else:
            # consensus HDBSCAN clustering
            # build cmap
            cmap = categorical_cmap(
                numUniqueSamples=len(y.unique()),
                numCatagories=10,
                cmap='tab10',
                continuous=False
                )

            label_color_dict = dict(
                zip(natsorted(y.unique()), [tuple(i) for i in cmap.colors])
                )

            if -1 in y.unique():
                # make black the first color to specify
                # cluster outliers (i.e. cluster -1 cells)
                cmap = ListedColormap(
                    np.insert(
                        arr=cmap.colors, obj=0,
                        values=[0.0, 0.0, 0.0], axis=0)
                        )

                # trim qualitative cmap to number of unique samples
                cmap = ListedColormap(cmap.colors[:-1])

            hue_dict = dict(
                zip(
                    natsorted(y.unique()),
                    list(range(len(y.unique()))))
                    )

            c = [hue_dict[i] for i in y]

            plt.scatter(
                X_encoded_embedded[:, 0],
                X_encoded_embedded[:, 1],
                c=c, cmap=cmap, ec='k',
                lw=0.0, s=scatter_point_size
                )

            legend_elements = []
            for e, i in enumerate(
                natsorted(y.unique())
              ):

                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', label=i,
                           markerfacecolor=cmap.colors[e], markeredgecolor='k',
                           lw=0.25, markersize=9)
                           )

            plt.legend(
                handles=legend_elements, labelspacing=0.15,
                bbox_to_anchor=(1.12, 1.0)
                )

        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=800)
        plt.close('all')

        return label_color_dict


def InterpolationGrid(orig_input_dims, grid_size, X_encoded, y, decoder, label_color_dict, channel_color_dict, frac_of_scatter_points, scatter_point_size, make_sample_sizes_equal, img_brightness_multiplier, scatter_point_alpha, save_dir):

    # make lists to store grid coordinates and their indices
    # for every latent space dimension
    grids = []
    indices = []

    # grab dimensions in reverse order (e.g. 3, 2, 1, 0) with grids.reverse()
    for d in range(2):

        # round minimum latent variable in dimension 'd' down to 100th place
        flr = math.floor(X_encoded[:, d].min() * 100.0) / 100.0

        # round maximum latent variable in dimension 'd' up to 100th place
        cel = math.ceil(X_encoded[:, d].max() * 100.0) / 100.0

        # construct grid of latent dimension values and their grid indices
        grid = np.array(np.linspace(flr, cel, grid_size))
        grids.append(grid)

        idx = np.array(range(0, len(grid)))
        idx = idx.astype(int)
        indices.append(idx)

    grids.reverse()

    # create an empty array that will fit the required number of
    # thumbnails given the chosen grid size
    y_dim, x_dim, channels = (orig_input_dims[0], orig_input_dims[1],
                              orig_input_dims[2]
                              )
    figure = np.zeros((y_dim * grid_size, x_dim * grid_size))

    # convert to RGB
    figure = gray2rgb(figure)

    # sample the grid
    for grid_tup, dim_tup in zip(product(*grids), product(*indices)):

        grid_tup = list(grid_tup)
        grid_tup.reverse()

        # get z sample at current grid spec
        z_sample = np.array([grid_tup])

        # decode z sample
        X_decoded = decoder.predict(z_sample)

        # reconstruct image
        reconstructed_img = X_decoded.reshape(
            y_dim, x_dim, channels)

        # create blank image to append channels to
        overlay = np.zeros(
            (reconstructed_img.shape[0],
             reconstructed_img.shape[1]))

        # convert to RGB
        overlay = gray2rgb(overlay)

        # append chanenels
        for name, (ch, color) in channel_color_dict.items():

            channel = reconstructed_img[:, :, ch]

            channel = gray2rgb(channel)

            channel = channel * img_brightness_multiplier

            overlay += channel * color

        figure[dim_tup[0] * y_dim: (dim_tup[0] + 1) * y_dim,
               dim_tup[1] * x_dim: (dim_tup[1] + 1) * x_dim] = overlay

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(figure)
    plt.grid(linestyle='dotted', linewidth=0.0)
    plt.gca().invert_yaxis()

    ax.set_xticks(list(range(x_dim, grid_size * x_dim+1, x_dim)))
    ax.set_xticklabels(list(np.round(grids[1], 2)), size=8)
    ax.set_yticks(list(range(y_dim, grid_size * y_dim+1, y_dim)))
    ax.set_yticklabels(list(np.round(grids[0], 2)), size=8)

    y = y.reset_index(drop=True)  # ensure y and X_encoded indices match
    y = y.sample(frac=frac_of_scatter_points)  # sample y
    data = X_encoded[y.index]  # get corresponding X_encoded samples
    y = y.reset_index(drop=True)  # reset y index to matche sampled X_encoded

    scatter_df = pd.concat([pd.DataFrame(y), pd.DataFrame(data)], axis=1)

    if make_sample_sizes_equal is True:
        lengths_list = []
        for i in scatter_df['cluster'].unique():

            lengths_list.append(len(scatter_df[scatter_df['cluster'] == i]))

        sample_size = min(lengths_list)

        sample_dfs = []
        for j in scatter_df['cluster'].unique():
            if len(scatter_df[scatter_df['cluster'] == j]) != sample_size:
                sample_dfs.append(
                    scatter_df[scatter_df['cluster'] == j].sample(n=sample_size))
            else:
                sample_dfs.append(scatter_df[scatter_df['cluster'] == j])

        scatter_df = pd.concat(sample_dfs, axis=0)
    else:
        pass

    # get global x and y ranges
    global_x_min = scatter_df[0].min()
    global_x_max = scatter_df[0].max()
    global_y_min = scatter_df[1].min()
    global_y_max = scatter_df[1].max()

    data = data[scatter_df.index]
    scatter_df = scatter_df.reset_index(drop=True)

    # filter latent vectors to isolate those between
    # the latent variable ranges used to generate the sweep grid
    scatter_points = scatter_df[
        (scatter_df[0] > grids[1].min())
        & (scatter_df[0] < grids[1].max())
        & (scatter_df[1] > grids[0].min())
        & (scatter_df[1] < grids[0].max())
        ].copy()

    # map latent space units (percent point function)
    # to the x, y pixel ranges of the sweep grid
    scatter_points[0] = np.interp(
        scatter_points[0],
        (global_x_min, global_x_max),
        (orig_input_dims[0]/2, (figure.shape[0] - orig_input_dims[0]/2))
        )
    scatter_points[1] = np.interp(
        scatter_points[1],
        (global_y_min, global_y_max),
        (orig_input_dims[0]/2, (figure.shape[0] - orig_input_dims[0]/2))
        )

    # plot latent vectors for images
    for i in natsorted(scatter_points['cluster'].unique()):

        ax.scatter(
            scatter_points[0][scatter_points['cluster'] == i],
            scatter_points[1][scatter_points['cluster'] == i],
            fc=[
                label_color_dict[i] for i in scatter_points['cluster'][
                    scatter_points['cluster'] == i]],
            marker='o',
            label=i, s=scatter_point_size,
            ec='k', lw=0.25, alpha=scatter_point_alpha)

    # channel legend
    legend_elements = []
    for name, (ch, color) in channel_color_dict.items():

        legend_elements.append(Line2D([0], [0], lw=6, color=color, label=name))

    leg = plt.legend(
        handles=legend_elements, loc='upper left', prop={'size': 11},
        markerscale=1, labelspacing=0.7, bbox_to_anchor=(1, 0.32)
        )
    ax.add_artist(leg)

    # cluster legend
    ax.legend(
        loc='upper left', prop={'size': 11}, labelspacing=0.6,
        markerscale=3, bbox_to_anchor=(1, 1.0))

    plt.xticks(rotation=90)

    plt.xlabel(
        'latent dimension 1', size=13,
        labelpad=10, fontweight='normal')
    plt.ylabel(
        'latent dimension 2', size=13,
        labelpad=10, fontweight='normal')

    plt.savefig(
        os.path.join(save_dir, 'InterpolationGrid.png'),
        dpi=800, bbox_inches='tight'
        )
    plt.close('all')

    return global_x_min, global_x_max, global_y_min, global_y_max, scatter_df


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def LassoVectors(decoder, clustering_labels, cutoffs, contrast_limits, orig_input_dims, imgs_instead_of_points, zoom, X, X_seg, X_encoded, X_encoded_embedded, X_decoded, y, numColumns, intensity_multiplier, max_examples, label_color_dict, channel_color_dict, thumbnail_font_size, save_dir):

    lasso_dict = {}

    data = pd.DataFrame(X_encoded_embedded, columns=['x', 'y'])

    if all(y == clustering_labels):
        y_df = pd.DataFrame(clustering_labels)
        y_df.rename(columns={0: 'cluster'}, inplace=True)
    else:
        y_df = y

    data = pd.merge(
        y_df.reset_index(drop=True), data, left_index=True, right_index=True
        )

    input_imgs = X[data.index]

    subplot_kw = dict(
        xlim=(data['x'].min(), data['x'].max()),
        ylim=(data['y'].min(), data['y'].max()),
        autoscale_on=False
        )

    fig, lasso_ax = plt.subplots(subplot_kw=subplot_kw, figsize=(9, 8))

    if imgs_instead_of_points is True:

        X_decoded = X_decoded[data.index]
        X_encoded_embedded = X_encoded_embedded[data.index]

        ScatterReconstructions(
            X_decoded=X_decoded, X_encoded_embedded=X_encoded_embedded,
            zoom=zoom, ax=lasso_ax
            )

        legend_elements = []
        for name, (ch, color) in channel_color_dict.items():

            legend_elements.append(
                Line2D([0], [0], color=color, lw=5, label=name)
                )

        pts = lasso_ax.scatter(
            data['x'], data['y'], c='k', s=0.0, ec='k', lw=0.25, zorder=4
            )

        plt.legend(
            handles=legend_elements, markerscale=1, labelspacing=0.7,
            prop={'size': 11}, bbox_to_anchor=(1.02, 0.99)
            )

    else:
        if all(y == clustering_labels):

            num_labels = len(np.unique(y))
            num_colors = plt.cm.get_cmap('tab20').N
            palette_multiplier = math.ceil(num_labels/num_colors)
            palette = []
            palette.extend(list(plt.cm.get_cmap('tab20').colors))
            palette = palette * palette_multiplier
            palette.insert(0, (0.0, 0.0, 0.0))
            trim = len(palette)-num_labels
            palette = palette[:-trim]

            label_color_dict = dict(zip(sorted(np.unique(y)), palette))
            c = [label_color_dict[i] for i in y]

            legend_elements = []
            for i in np.unique(clustering_labels):
                markerfacecolor = label_color_dict[i]
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w',
                           label=i, markerfacecolor=markerfacecolor,
                           markeredgecolor='k', lw=0.25, markersize=15)
                    )
        else:

            c = [label_color_dict[i] for i in data['cluster']]

            legend_elements = []
            for name, color in label_color_dict.items():
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w',
                           label=name, markerfacecolor=color,
                           markeredgecolor='k', lw=0.25, markersize=15)
                    )

        pts = lasso_ax.scatter(
            data['x'], data['y'], c=c, s=30.0, ec='k', lw=0.25, zorder=4
            )

        lasso_ax.update_datalim(np.column_stack([data['x'], data['y']]))
        lasso_ax.autoscale()

        plt.legend(
            handles=legend_elements, markerscale=1, labelspacing=0.7,
            prop={'size': 11}, bbox_to_anchor=(1.02, 0.99)
            )

    selector = SelectFromCollection(lasso_ax, pts)

    latent_vectors = X_encoded[data.index]
    data = data.reset_index(drop=True)

    def accept(event):
        if event.key == "enter":
            print("Selected points:")
            print(selector.xys[selector.ind])
            selector.disconnect()
            lasso_ax.set_title("")
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    lasso_ax.set_title("Press enter to accept selected points.")
    lasso_ax.set_aspect('equal')
    plt.show(block=True)

    selected_vectors = data.loc[selector.ind]
    selected_vectors['latent_vector'] = [
        i for i in latent_vectors[selector.ind]]
    selected_vectors['input_img'] = [
        i.flatten() for i in input_imgs[selector.ind]]

    if max_examples is not None:
        if len(selected_vectors) < max_examples:
            max_examples = len(selected_vectors)
        selected_vectors = selected_vectors.sample(
            n=max_examples, random_state=44)

    selected_vectors.sort_values(by='cluster', inplace=True)
    X_seg = X_seg[selected_vectors.index]

    # check cell images
    numSamples = len(selected_vectors)
    numRows = math.ceil(numSamples/numColumns)
    grid_dims = (numRows, numColumns)

    fig = plt.figure()

    fig.text(0.13, 0.97, 'Input Images', ha='left', fontsize='medium')
    fig.text(0.53, 0.97, 'Learned Representations', fontsize='medium')

    outer_grid_rows = 1
    outer_grid_cols = 2

    outer = gridspec.GridSpec(
        outer_grid_rows, outer_grid_cols, wspace=0.1, hspace=0.0)

    for panel in range(outer_grid_rows * outer_grid_cols):

        inner = gridspec.GridSpecFromSubplotSpec(
            grid_dims[0], grid_dims[1],
            subplot_spec=outer[panel], wspace=0.1, hspace=0.0)

        for e, (row, seg) in enumerate(
          zip(selected_vectors.iterrows(), X_seg)
          ):

            ax = plt.Subplot(fig, inner[e])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # select segmentation outlines slice
            seg_slice = seg[:, :, 0]

            # ensure segmentation outlines are normalized 0-1
            seg_slice = (seg_slice - np.min(seg_slice))/np.ptp(seg_slice)

            # convert segmentation thumbnail to RGB
            # and add to blank image
            seg_slice = gray2rgb(seg_slice) * 0.25  # decrease alpha

            if panel == 0:

                input_img = row[1]['input_img'].reshape(
                    orig_input_dims[0], orig_input_dims[1], orig_input_dims[2]
                    )

                overlay = np.zeros(
                    (input_img.shape[0],
                     input_img.shape[1]))

                # add centroid point at the center of the image
                overlay[
                    int(input_img.shape[0]/2):int(input_img.shape[0]/2)+1,
                    int(input_img.shape[1]/2):int(input_img.shape[1]/2)+1
                    ] = 1

                overlay = gray2rgb(overlay)

                for name, (ch, color) in channel_color_dict.items():

                    channel_slice = input_img[:, :, ch]

                    channel_slice = reverse_log(cutoffs, channel_slice, name, contrast_limits)

                    channel_slice = gray2rgb(channel_slice)

                    channel_slice = channel_slice * intensity_multiplier

                    overlay += channel_slice * to_rgb(color)

                overlay += seg_slice

            elif panel == 1:

                z_sample = np.array([list(row[1]['latent_vector'])])

                X_decoded = decoder.predict(z_sample)

                reconstructed_img = X_decoded.reshape(
                    orig_input_dims[0], orig_input_dims[1], orig_input_dims[2])

                overlay = np.zeros(
                    (reconstructed_img.shape[0],
                     reconstructed_img.shape[1])
                     )

                # add centroid point at the center of the image
                overlay[
                    int(reconstructed_img.shape[0]/2):int(
                        reconstructed_img.shape[0]/2)+1,
                    int(reconstructed_img.shape[1]/2):int(
                        reconstructed_img.shape[1]/2)+1
                    ] = 1

                overlay = gray2rgb(overlay)

                for name, (ch, color) in channel_color_dict.items():

                    channel_slice = reconstructed_img[:, :, ch]

                    channel_slice = reverse_log(cutoffs, channel_slice, name, contrast_limits)

                    channel_slice = gray2rgb(channel_slice)

                    channel_slice = channel_slice * intensity_multiplier

                    overlay += channel_slice * to_rgb(color)

                overlay += seg_slice

            ax.imshow(overlay, cmap=plt.cm.binary)

            ax.set_xlabel(
                row[1]['cluster'], fontsize=thumbnail_font_size, labelpad=0.75
                )
            fig.add_subplot(ax)

    fig.subplots_adjust(
        bottom=0.01, top=0.94,
        left=0.01, right=0.85,
        wspace=0.2, hspace=0.1
        )

    legend_elements = []
    for name, (ch, color) in channel_color_dict.items():
        legend_elements.append(
            Line2D([0], [0], color=color, lw=3, label=name)
            )

    fig.legend(
        handles=legend_elements, prop={'size': 5}, bbox_to_anchor=(0.98, 0.95))

    plt.savefig(
        os.path.join(save_dir, 'lassoed_cells.png'),
        dpi=800, bbox_inches='tight'
        )
    plt.close('all')


def PlotReconstructedImages(orig_input_dims, cutoffs, contrast_limits, decoder, X, X_seg, X_encoded, y, numColumns, label_color_dict, channel_color_dict, intensity_multiplier, thumbnail_font_size, filename, save_dir):

    numSamples = len(X)
    numRows = math.ceil(numSamples/numColumns)
    grid_dims = (numRows, numColumns)

    fig = plt.figure()

    fig.text(0.13, 0.97, 'Input Images', ha='left', fontsize='medium')
    fig.text(0.53, 0.97, 'Learned Representations', fontsize='medium')

    outer_grid_rows = 1
    outer_grid_cols = 2

    outer = gridspec.GridSpec(
        outer_grid_rows, outer_grid_cols, wspace=0.1, hspace=0.0
        )

    for panel in range(outer_grid_rows * outer_grid_cols):

        inner = gridspec.GridSpecFromSubplotSpec(
            grid_dims[0], grid_dims[1],
            subplot_spec=outer[panel], wspace=0.1, hspace=0.0)

        for e, (trans, encode, label, seg) in enumerate(
          zip(X, X_encoded, y.items(), X_seg)
          ):

            ax = plt.Subplot(fig, inner[e])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # select segmentation outlines slice
            seg_slice = seg[:, :, 0]

            # ensure segmentation outlines are normalized 0-1
            seg_slice = (seg_slice - np.min(seg_slice))/np.ptp(seg_slice)

            # convert segmentation thumbnail to RGB
            # and add to blank image
            seg_slice = gray2rgb(seg_slice) * 0.25  # decrease alpha

            if panel == 0:

                overlay = np.zeros((trans.shape[0], trans.shape[1]))

                # add centroid point at the center of the image
                overlay[
                    int(trans.shape[0]/2):int(trans.shape[0]/2)+1,
                    int(trans.shape[1]/2):int(trans.shape[1]/2)+1
                    ] = 1

                overlay = gray2rgb(overlay)

                for name, (ch, color) in channel_color_dict.items():

                    channel_slice = trans[:, :, ch]

                    channel_slice = reverse_log(cutoffs, channel_slice, name, contrast_limits)

                    channel_slice = gray2rgb(channel_slice)

                    channel_slice = channel_slice * intensity_multiplier

                    overlay += channel_slice * to_rgb(color)

                overlay += seg_slice

            elif panel == 1:

                z_sample = np.array([encode])

                x_decoded = decoder.predict(z_sample)

                reconstructed_img = x_decoded.reshape(
                    orig_input_dims[0], orig_input_dims[1], orig_input_dims[2])

                overlay = np.zeros(
                    (reconstructed_img.shape[0],
                     reconstructed_img.shape[1])
                     )

                # add centroid point at the center of the image
                overlay[
                    int(reconstructed_img.shape[0]/2):int(
                        reconstructed_img.shape[0]/2)+1,
                    int(reconstructed_img.shape[1]/2):int(
                        reconstructed_img.shape[1]/2)+1
                    ] = 1

                overlay = gray2rgb(overlay)

                for name, (ch, color) in channel_color_dict.items():

                    channel_slice = reconstructed_img[:, :, ch]

                    channel_slice = reverse_log(cutoffs, channel_slice, name, contrast_limits)

                    channel_slice = gray2rgb(channel_slice)

                    channel_slice = channel_slice * intensity_multiplier

                    overlay += channel_slice * to_rgb(color)

                overlay += seg_slice

            ax.imshow(overlay, cmap=plt.cm.binary)

            ax.set_xlabel(
                label[1], fontsize=thumbnail_font_size, labelpad=0.75
                )
            fig.add_subplot(ax)

    fig.subplots_adjust(
        bottom=0.01, top=0.94,
        left=0.01, right=0.85,
        wspace=0.2, hspace=0.1
        )

    legend_elements = []
    for name, (ch, color) in channel_color_dict.items():
        legend_elements.append(
            Line2D([0], [0], color=color, lw=3, label=name)
            )

    fig.legend(
        handles=legend_elements, prop={'size': 5}, bbox_to_anchor=(0.98, 0.95))

    plt.savefig(
        os.path.join(save_dir, f'{filename}.png'), dpi=800, bbox_inches='tight'
        )
    plt.close('all')


def mse(orig_input_dims, decoder, X, X_seg, X_encoded, y, mse_percentile_cutoff, filename, save_dir):

    errors = []

    for input_img, encoded_img in zip(X, X_encoded):

        z_sample = np.array([encoded_img])

        x_decoded = decoder.predict(z_sample)

        reconstructed_img = x_decoded.reshape(
            orig_input_dims[0], orig_input_dims[1], orig_input_dims[2])

        err = np.sum((input_img - reconstructed_img) ** 2)
        err /= float(input_img.shape[0] * input_img.shape[1])

        errors.append(err)

    average_error = np.mean(errors)
    print(f'average mean squared error is {average_error}')

    n, bins, pathes = plt.hist(errors, bins=50)
    plt.axvline(np.percentile(errors, mse_percentile_cutoff), c='r')
    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=800)

    outlier_idxs = [
        i for i, v in enumerate(errors)
        if v > np.percentile(errors, mse_percentile_cutoff)]

    X_outliers = X[outlier_idxs]
    X_outliers_seg = X_seg[outlier_idxs]
    X_encoded_outliers = X_encoded[outlier_idxs]
    y_outliers = y[outlier_idxs].reset_index(drop=True)

    plt.close('all')

    return average_error, errors, X_outliers, X_outliers_seg, X_encoded_outliers, y_outliers, outlier_idxs


def categorical_cmap(numUniqueSamples, numCatagories, cmap='tab10', continuous=False):

    numSubcatagories = math.ceil(numUniqueSamples/numCatagories)

    if numCatagories > plt.get_cmap(cmap).N:
        raise ValueError('Too many categories for colormap.')
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, numCatagories))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(numCatagories, dtype=int))
        # rearrange hue order to taste
        cd = {
            'B': 0, 'O': 1, 'G': 2, 'R': 3, 'Pu': 4,
            'Br': 5, 'Pi': 6, 'Gr': 7, 'Y': 8, 'Cy': 9,
            }
        myorder = [
            cd['B'], cd['O'], cd['G'], cd['Pu'], cd['Y'],
            cd['R'], cd['Cy'], cd['Br'], cd['Gr'], cd['Pi']
            ]
        ccolors = [ccolors[i] for i in myorder]

        # use Okabe and Ito color-safe palette for first 6 colors
        # ccolors[0] = np.array([0.91, 0.29, 0.235]) #E84A3C
        # ccolors[1] = np.array([0.18, 0.16, 0.15]) #2E2926
        ccolors[0] = np.array([0.0, 0.447, 0.698, 1.0])  # blue
        ccolors[1] = np.array([0.902, 0.624, 0.0, 1.0])  # orange
        ccolors[2] = np.array([0.0, 0.620, 0.451, 1.0])  # bluish green
        ccolors[3] = np.array([0.8, 0.475, 0.655, 1.0])  # reddish purple
        ccolors[4] = np.array([0.941, 0.894, 0.259, 1.0])  # yellow
        ccolors[5] = np.array([0.835, 0.369, 0.0, 1.0])  # vermillion

    cols = np.zeros((numCatagories * numSubcatagories, 3))
    for i, c in enumerate(ccolors):
        chsv = colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, numSubcatagories).reshape(numSubcatagories, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, numSubcatagories)
        arhsv[:, 2] = np.linspace(chsv[2], 1, numSubcatagories)
        rgb = colors.hsv_to_rgb(arhsv)
        cols[i * numSubcatagories:(i + 1) * numSubcatagories, :] = rgb
    cmap = colors.ListedColormap(cols)

    # trim colors if necessary
    if len(cmap.colors) > numUniqueSamples:
        trim = len(cmap.colors) - numUniqueSamples
        cmap_colors = cmap.colors[:-trim]
        cmap = colors.ListedColormap(cmap_colors, name='from_list', N=None)

    return cmap


def ENCODE_IMAGES(config):
    
    training_thumb_dims = (
        config.window_size, config.window_size, len(config.tif_channels)
        )

    with open(
      os.path.join(
        config.output_path, '4_feature_preprocessing_selections/cutoffs.pkl'),
      'rb') as handle:
        cutoffs = pickle.load(handle)

    channel_dict = dict(
        zip(config.tif_channels, range(len(config.tif_channels)))
        )

    save_dir = os.path.join(
        config.output_path, f'6_latent_space_LD{config.latent_dimension}'
        )
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    contrast_limits = yaml.safe_load(open(config.contrast_path))

    # load previously saved encoder and decoders
    try:
        encoder = load_model(
            os.path.join(config.output_path, '5_train_vae/encoder.hdf5')
            )
    except OSError:
        print('Encoder not found.')
        sys.exit()

    try:
        decoder = load_model(
            os.path.join(config.output_path, '5_train_vae/decoder.hdf5')
            )
    except OSError:
        print('Decoder not found.')
        sys.exit()

    ###########################################################################
    # read training labels
    y_train = pd.read_csv(
        os.path.join(config.output_path, '1_cellcutter_input/train.csv')
        )

    # read training thumbnails (16-bit unsigned integer format)
    z1_train_path = os.path.join(
        config.output_path,
        f'2_cellcutter_output_win{config.window_size}/train_thumbnails_{config.window_size}.zarr'
        )
    store = zarr.ZipStore(z1_train_path, mode='r')
    X_train = zarr.open(store=store)

    # read segmentation thumbnails for training data (16-bit unsigned integer)
    z1_train_path_seg = os.path.join(
        config.output_path,
        f'2_cellcutter_output_win{config.window_size}/train_thumbnails_{config.window_size}_seg.zarr'
        )
    store = zarr.ZipStore(z1_train_path_seg, mode='r')
    X_train_seg = zarr.open(store=store)

    # read validation labels
    y_validate = pd.read_csv(
        os.path.join(config.output_path, '1_cellcutter_input/validate.csv')
        )

    # read validation thumbnails (16-bit unsigned integer format)
    z1_validate_path = os.path.join(
        config.output_path,
        f'2_cellcutter_output_win{config.window_size}/validate_thumbnails_{config.window_size}.zarr'
        )
    store = zarr.ZipStore(z1_validate_path, mode='r')
    X_validate = zarr.open(store=store)

    # read segmentation thumbnails for validation data (16-bit unsigned int)
    z1_validate_path_seg = os.path.join(
        config.output_path,
        f'2_cellcutter_output_win{config.window_size}/validate_thumbnails_{config.window_size}_seg.zarr'
        )
    store = zarr.ZipStore(z1_validate_path_seg, mode='r')
    X_validate_seg = zarr.open(store=store)

    # read test labels
    y_test = pd.read_csv(
        os.path.join(config.output_path, '1_cellcutter_input/test.csv')
        )

    # read test thumbnails (16-bit unsigned integer format)
    z1_test_path = os.path.join(
        config.output_path,
        f'2_cellcutter_output_win{config.window_size}/test_thumbnails_{config.window_size}.zarr'
        )
    store = zarr.ZipStore(z1_test_path, mode='r')
    X_test = zarr.open(store=store)

    # read segmentation thumbnails for test data (16-bit unsigned integer)
    z1_test_path_seg = os.path.join(
        config.output_path,
        f'2_cellcutter_output_win{config.window_size}/test_thumbnails_{config.window_size}_seg.zarr'
        )
    store = zarr.ZipStore(z1_test_path_seg, mode='r')
    X_test_seg = zarr.open(store=store)

    ###########################################################################

    # take a sample of test thumbnail data for analysis

    # rearrange Zarr dimensions to fit shape of expected VAE input
    # (i.e. cells, x, y, channels)
    X_test1 = transposeZarr(z=X_test)
    X_test1 = X_test1[0:config.clustering_sample_size]

    X_test1_seg = transposeZarr(z=X_test_seg)
    X_test1_seg = X_test1_seg[0:config.clustering_sample_size]

    y_test1 = y_test[0:config.clustering_sample_size]

    ###########################################################################

    combo_dir = os.path.join(save_dir, 'combined_zarr')
    combo_dir_seg = os.path.join(save_dir, 'combined_zarr_seg')

    if not os.path.exists(combo_dir):
        os.makedirs(combo_dir)

        print('Combined data does not exist, creating...')

        # initialize combo zarr to store combined train, validate, test data
        X_combo = zarr.open(
            combo_dir,
            mode='w',
            shape=(
                X_train.shape[0], X_train.shape[1],
                X_train.shape[2], X_train.shape[3]
                ),
            chunks=(
                X_train.chunks[0], X_train.chunks[1],
                X_train.chunks[2], X_train.chunks[3]
                ),
            compressor=X_train.compressor,
            dtype=X_train.dtype
            )
        X_combo[:] = X_train
        # concatenate validation and test data to training data
        X_combo.append(X_validate, axis=1)
        X_combo.append(X_test, axis=1)
    else:
        # read combined thumbnails
        X_combo = zarr.open(combo_dir)

    if not os.path.exists(combo_dir_seg):
        os.makedirs(combo_dir_seg)

        print('Combined segmentation outlines does not exist, creating...')

        # initialize combo zarr to store combined train, validate, test data
        X_combo_seg = zarr.open(
            combo_dir_seg,
            mode='w',
            shape=(
                X_train_seg.shape[0], X_train_seg.shape[1],
                X_train_seg.shape[2], X_train_seg.shape[3]
                ),
            chunks=(
                X_train_seg.chunks[0], X_train_seg.chunks[1],
                X_train_seg.chunks[2], X_train_seg.chunks[3]
                ),
            compressor=X_train_seg.compressor,
            dtype=X_train_seg.dtype
            )
        X_combo_seg[:] = X_train_seg
        # concatenate validation and test data to training data
        X_combo_seg.append(X_validate_seg, axis=1)
        X_combo_seg.append(X_test_seg, axis=1)
    else:
        # read combined thumbnails
        X_combo_seg = zarr.open(combo_dir_seg)

    ###########################################################################

    if config.cluster_full_dataset:
        print()
        print('Aggregating combined training, validation, and test data.')

        # rearrange Zarr dimensions to fit shape of expected VAE input
        # (i.e. cells, x, y, channels)
        X_test1 = transposeZarr(z=X_combo)
        X_test1_seg = transposeZarr(z=X_combo_seg)

        # load data into memory
        X_test1 = X_test1[:]
        X_test1_seg = X_test1_seg[:]

        # combine labels for train, validate, and test data
        y_test1 = pd.concat([y_train, y_validate, y_test], axis=0)

    y_test1['cluster'].replace(
        to_replace={47: '4.7', 413: '4.13', 416: '4.16'}, inplace=True
        )
    y_test1['cluster'] = y_test1['cluster'].astype('str')

    ###########################################################################

    # log10 transform
    X_test1 = np.log10(X_test1, where=(X_test1 != 0))

    for i in range(X_test1.shape[0]):

        for e, (lower_cutoff_log, upper_cutoff_log) in enumerate(
          cutoffs.values()):

            # scale 0.17th and 99.99th percentile between 0 and 1
            X_test1[i, :, :, e] = (
                (((1-0)*(X_test1[i, :, :, e]-lower_cutoff_log)) /
                 (upper_cutoff_log-lower_cutoff_log)
                 ) + 0)

            # clip lower and upper outliers to 0 and 1, respectively
            X_test1[i, :, :, e] = np.clip(
                a=X_test1[i, :, :, e], a_min=0, a_max=1
                )

    ###########################################################################

    # encode test images
    X_encoded = EncodeImgs(X=X_test1, encoder=encoder)

    ###########################################################################
    # embed latent vectors if they are greater than 2D

    embedding_path = os.path.join(save_dir, 'embedding.npy')

    if (config.latent_dimension > 2) and not os.path.exists(embedding_path):

        startTime = datetime.now()

        print('Embedding data...')

        if config.embedding_algorithm == 'TSNE':
            print('Computing TSNE embedding.')
            X_encoded_embedded = TSNE(
                n_components=2,
                perplexity=27,
                early_exaggeration=19,
                learning_rate=200.0,
                metric='euclidean',
                random_state=5,
                init='pca', n_jobs=-1).fit_transform(X_encoded)

        elif config.embedding_algorithm == 'UMAP':
            print('Computing UMAP embedding.')
            X_encoded_embedded = UMAP(
                n_components=2,
                n_neighbors=30,
                learning_rate=1.0,
                output_metric='euclidean',
                min_dist=0.1,
                repulsion_strength=3,
                random_state=3,
                n_epochs=1000,
                init='spectral',
                metric='euclidean',
                metric_kwds=None,
                output_metric_kwds=None,
                n_jobs=-1,
                low_memory=False,
                spread=1.0,
                local_connectivity=1.0,
                set_op_mix_ratio=0.5,
                negative_sample_rate=5,
                transform_queue_size=4.0,
                a=None,
                b=None,
                angular_rp_forest=False,
                target_n_neighbors=-1,
                target_metric='categorical',
                target_metric_kwds=None,
                target_weight=0.5,
                transform_seed=42,
                transform_mode='embedding',
                force_approximation_algorithm=False,
                verbose=False,
                unique=False,
                densmap=False,
                dens_lambda=2.0,
                dens_frac=0.6,
                dens_var_shift=0.1,
                disconnection_distance=None,
                output_dens=False).fit_transform(X_encoded)

            print('Embedding completed in ' + str(datetime.now() - startTime))

        # save embedding
        np.save(os.path.join(save_dir, 'embedding'), X_encoded_embedded)

    elif (config.latent_dimension > 2) and os.path.exists(embedding_path):
        print('Loading saved embedding.')

        # load previously saved embedding
        X_encoded_embedded = np.load(embedding_path)

    else:
        # simply assign the 2D X_encoded the variable X_encoded_embedded
        X_encoded_embedded = X_encoded.copy()

    ###########################################################################

    # cluster the data with HDBSCAN
    print(f'Minimum_cluster_size is {config.hdbscan_min_cluster_size}')

    clustering = hdbscan.HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size, min_samples=None,
        metric='euclidean', alpha=1.0, p=None, algorithm='best',
        leaf_size=40,
        memory=Memory(location=None),
        approx_min_span_tree=True,
        gen_min_span_tree=False, core_dist_n_jobs=4,
        cluster_selection_method='eom',
        allow_single_cluster=False,
        prediction_data=False,
        match_reference_implementation=False).fit(X_encoded_embedded)

    print(np.unique(clustering.labels_))

    ###########################################################################

    # plot latent vectors colored according to prior clustering
    label_color_dict = PlotLatentSpace(
        reconstructions=False,
        zoom=None,
        X_encoded_embedded=X_encoded_embedded,
        X_decoded=None,
        y=y_test1['cluster'],
        channel_color_dict=None,
        scatter_point_size=144000/len(X_encoded_embedded),
        filename='consensus_clustering',
        save_dir=save_dir
        )

    # remove noisy cells (cluster -1)
    # plot latent vectors colored by HDBSCAN clustering of latent space
    # filter_indices = np.where(clustering.labels_ != -1)[0].tolist()
    # X_encoded_embedded_filt = np.take(X_encoded_embedded, filter_indices, 0)
    # clustering_labels = clustering.labels_[filter_indices]

    PlotLatentSpace(
        reconstructions=False,
        zoom=None,
        X_encoded_embedded=X_encoded_embedded,
        X_decoded=None,
        y=clustering.labels_,
        channel_color_dict=None,
        scatter_point_size=144000/len(X_encoded_embedded),
        filename='latent_clustering',
        save_dir=save_dir
        )

    # reconstruct thumbnail images from latent vectors
    X_decoded = DecodeVectors(
        decoder=decoder, cutoffs=cutoffs, contrast_limits=contrast_limits,
        X_encoded=X_encoded, X_seg=X_test1_seg,
        orig_input_dims=training_thumb_dims,
        channel_color_dict=config.channel_colors,
        intensity_multiplier=1.1
        )

    # plot latent vectors represented as their learned reconstructions
    PlotLatentSpace(
        reconstructions=True,
        zoom=0.5,
        X_encoded_embedded=X_encoded_embedded,
        X_decoded=X_decoded,
        y=y_test1['cluster'],
        channel_color_dict=config.channel_colors,
        scatter_point_size=144000/len(X_encoded_embedded),
        filename='thumbnails',
        save_dir=save_dir
        )

    # display learned representations of input thumbnail images
    if config.latent_dimension == 2:

        InterpolationGrid(
            orig_input_dims=training_thumb_dims,
            grid_size=50,
            X_encoded=X_encoded,
            y=y_test1['cluster'],
            decoder=decoder,
            label_color_dict=label_color_dict,
            channel_color_dict=config.channel_colors,
            frac_of_scatter_points=1.0,
            scatter_point_size=144000/len(X_encoded_embedded),
            make_sample_sizes_equal=False,
            img_brightness_multiplier=1.2,
            scatter_point_alpha=1.0,
            save_dir=save_dir
            )

    # get input and output images of lassoed latent vectors for targeted analysis of regions 
    if config.lasso_vector_tool:
        LassoVectors(
            decoder=decoder,
            clustering_labels=clustering.labels_,
            cutoffs=cutoffs,
            contrast_limits=contrast_limits,
            orig_input_dims=training_thumb_dims,
            imgs_instead_of_points=True,
            zoom=0.5,
            X=X_test1,
            X_seg=X_test1_seg,
            X_encoded=X_encoded,
            X_encoded_embedded=X_encoded_embedded,
            X_decoded=X_decoded,
            y=y_test1['cluster'],
            numColumns=10,
            intensity_multiplier=1.1,
            label_color_dict=label_color_dict,
            channel_color_dict=config.channel_colors,
            max_examples=1000,
            thumbnail_font_size=3.0,
            save_dir=save_dir
            )

    PlotReconstructedImages(
        orig_input_dims=training_thumb_dims,
        cutoffs=cutoffs,
        contrast_limits=contrast_limits,
        decoder=decoder,
        X=X_test1[0:100],
        X_seg=X_test1_seg[0:100],
        X_encoded=X_encoded[0:100],
        y=y_test1['cluster'][0:100],
        numColumns=10,
        label_color_dict=label_color_dict,
        channel_color_dict=config.channel_colors,
        intensity_multiplier=1.1,
        thumbnail_font_size=3.0,
        filename='learned_reconstructions',
        save_dir=save_dir
        )

    # compute mean squared error between thumbnail image inputs and outputs
    (average_error,
        errors,
        X_outliers,
        X_outliers_seg,
        X_encoded_outliers,
        y_outliers,
        outlier_idxs) = mse(
            orig_input_dims=training_thumb_dims,
            decoder=decoder,
            X=X_test1,
            X_seg=X_test1_seg,
            X_encoded=X_encoded,
            y=y_test1['cluster'],
            mse_percentile_cutoff=99,
            filename='mse_dist',
            save_dir=save_dir
            )

    # get input thumbnails associated with poor learned reconstruction
    PlotReconstructedImages(
        orig_input_dims=training_thumb_dims,
        cutoffs=cutoffs,
        contrast_limits=contrast_limits,
        decoder=decoder,
        X=X_outliers,
        X_seg=X_outliers_seg,
        X_encoded=X_encoded_outliers,
        y=y_outliers,
        numColumns=10,
        label_color_dict=label_color_dict,
        channel_color_dict=config.channel_colors,
        intensity_multiplier=1.0,
        thumbnail_font_size=3.0,
        filename='outliers',
        save_dir=save_dir
        )
