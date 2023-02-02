'''
A virtualenv with tensorflow 2.6.3 is needed to run the
VAE training module (train_vae) in this script.
'''

import argparse
import os
import re
import yaml
import random
import datetime

import pandas as pd
import numpy as np
import math

from subprocess import call
from subprocess import run

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from skimage.color import gray2rgb
from skimage.util import img_as_float

import zarr
from tifffile import imread

import pickle

from lazy_ops import DatasetView
from tensorflow.python.framework.ops import disable_eager_execution
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import save_model


def gen_cellcutter_input(root_output_path, csv_path, F):

    save_dir = os.path.join(root_output_path, '1_cellcutter_input')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        extension = os.path.splitext(csv_path)[1]
        ext = extension.split('.')[1]
        if ext == 'parquet':
            csv = pd.read_parquet(csv_path)
        elif ext == 'csv':
            csv = pd.read_csv(csv_path)
        else:
            raise ValueError(
                f'Note: extension type {extension} is not yet supported.'
                )

        # drop cells without a consensus cluster (i.e. noisy cells)
        csv = csv[csv['cluster'] != -1]

        #######################################################################

        # calculate weighted random sample by cluster size (class balance)
        groups = csv.groupby('cluster')
        sample_weights = pd.DataFrame(
            {'weights': 1 / (groups.size() * len(groups))}
            )
        weights = pd.merge(
            csv[['cluster']], sample_weights, left_on='cluster', right_index=True
            )

        csv = csv.sample(
            frac=F, replace=False, weights=weights['weights'],
            random_state=0, axis=0
            )
        print()
        print('Cells per cluster after cluster-weighted random sampling:')
        print(csv.groupby('cluster').size().sort_values(ascending=False))

        ###########################################################################

        # shuffle csv data
        csv = csv.sample(frac=1.0, random_state=0)

        # reserve 10% of data for testing and 10% for validation
        split = round(len(csv) * 0.10)
        test = csv[0:split]
        validate = csv[split:split*2]
        train = csv[split*2:]

        # reset row indexes of each dataframe
        test.reset_index(drop=True, inplace=True)
        validate.reset_index(drop=True, inplace=True)
        train.reset_index(drop=True, inplace=True)

        ###########################################################################

        # save testing, validation, and training dataframes for cellcutter
        test.to_csv(os.path.join(save_dir, 'test.csv'), index=False)
        validate.to_csv(os.path.join(save_dir, 'validate.csv'), index=False)
        train.to_csv(os.path.join(save_dir, 'train.csv'), index=False)

        return save_dir

    else:
        return save_dir


def run_cellcutter(root_output_path, cellcutter_input_path, image_path, mask_path, markers_path, cellcutter_markers, window_size, cells_per_chunk):

    save_dir = os.path.join(
        root_output_path, f'2_cellcutter_output_win{window_size}'
        )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        markers = pd.read_csv(markers_path)

        cellcutter_marker_ids = []
        for i in cellcutter_markers:
            id = markers['channel_number'][markers['marker_name'] == i].values[0]
            cellcutter_marker_ids.append(str(id))

        for name in ['train', 'validate', 'test']:
            print()
            print(f'Cutting {name} data...')
            run(
                ["cut_cells", "-z", "--window-size", window_size,
                 "--cells-per-chunk", cells_per_chunk, "--cache-size", "57711",
                 image_path, mask_path,
                 os.path.join(cellcutter_input_path, f"{name}.csv"),
                 os.path.join(save_dir, f"{name}_thumbnails_{window_size}.zarr"),
                 "--channels"] + cellcutter_marker_ids
                )

            run(
                ["cut_cells", "-z", "--window-size", window_size,
                 "--cells-per-chunk", cells_per_chunk, "--cache-size", "57711",
                 image_path, mask_path, os.path.join(cellcutter_input_path,
                 f"{name}.csv"), os.path.join(
                    save_dir, f"{name}_thumbnails_{window_size}_seg.zarr"),
                 "--channels", "1"
                 ]
                )
        return save_dir

    else:
        return save_dir


def gen_img_gallery(root_output_path, cellcutter_input_path, cellcutter_output_path, markers_path, num_examples, cellcutter_markers, viz_markers, window_size, contrast_path):

    save_dir = os.path.join(root_output_path, '3_thumbnail_examples')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        viz_marker_ids = [cellcutter_markers.index(i) for i in viz_markers]

        # inner function for thumbnail generation
        def PlotInputImgs(numExamples, numColumns, imgs, labels, fontSize, colors, channelNames, channelIDs, fileName, contrast_limits):

            numSamples = len(imgs)
            numRows = math.ceil(numExamples/numColumns)
            grid_dims = (numRows, numColumns)

            # numColumns = math.ceil(numExamples/numRows)
            # grid_dims = (numRows, numColumns)

            sns.set_style('whitegrid')
            fig = plt.figure(figsize=(13, 10))

            custom_lines = []
            for e, row in enumerate(labels.iterrows()):

                plt.subplot(grid_dims[0], grid_dims[1], e + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)

                # initialize array of zeros with shape of full-size image
                overlay = np.zeros((imgs.shape[2], imgs.shape[3]))

                overlay = gray2rgb(overlay)

                for d, ch, color in zip(channelIDs, channelNames, colors):

                    lyr = imgs[d, row[0], :, :]

                    lyr = img_as_float(lyr)

                    # apply image contrast settings
                    lyr -= (contrast_limits[ch][0]/65535)
                    lyr /= (
                        (contrast_limits[ch][1]/65535)
                        - (contrast_limits[ch][0]/65535))

                    lyr = np.clip(lyr, 0, 1)

                    lyr = gray2rgb(lyr)
                    lyr = lyr * color
                    overlay += lyr

                    custom_lines.append(Line2D([0], [0], color=color, lw=5))

                label = row[1]['cluster']

                # pass "thumb" variable to imshow (below) to check whether
                # thumbnail zarr file and CSV indices line up

                # img = imread(
                #     '/Volumes/My Book/cylinter_input/sardana-097/tif/' +
                #     'WD-76845-097.ome.tif', key=0)
                #
                # thumb = img[
                #     round(labels.loc[row[0], "Y_centroid"]) -
                #     32:round(labels.loc[row[0], "Y_centroid"]) + 32,
                #     round(labels.loc[row[0], "X_centroid"]) -
                #     32:round(labels.loc[row[0], "X_centroid"]) + 32,
                #     ]

                plt.imshow(overlay, cmap=plt.cm.binary)
                plt.xlabel(label, size=fontSize, labelpad=1.5)

            fig.legend(
                custom_lines, channelNames, prop={'size': 11},
                bbox_to_anchor=(0.98, 0.99)
                )

            plt.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.85)
            plt.savefig(os.path.join(save_dir, f'{fileName}.pdf'))
            plt.close('all')

        #######################################################################

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # read training labels
        labels_path = os.path.join(cellcutter_input_path, 'train.csv')
        labels = pd.read_csv(labels_path)

        # read training images
        z_path = os.path.join(
            cellcutter_output_path, f'train_thumbnails_{window_size}.zarr'
            )
        store = zarr.ZipStore(z_path, mode='r')
        z = zarr.open(store=store)

        #######################################################################

        # contrast settings
        contrast_limits = yaml.safe_load(open(contrast_path))

        #######################################################################

        # pull random thumbnails from training data to check quality
        thumb_ids = np.random.RandomState(1).choice(
            range(0, z.shape[1]), num_examples, replace=False)

        imgs = z.get_orthogonal_selection((slice(None), thumb_ids))

        labels = labels.iloc[thumb_ids]
        labels.reset_index(drop=True, inplace=True)
        labels.sort_values(by='cluster', inplace=True)

        colors = plt.get_cmap('tab10').colors * math.ceil(imgs.shape[3]/10)

        PlotInputImgs(
            numExamples=num_examples,
            numColumns=16,
            imgs=imgs,
            labels=labels,
            fontSize=8,
            colors=colors,
            channelNames=viz_markers,
            channelIDs=viz_marker_ids,
            fileName='thumbnail_examples',
            contrast_limits=contrast_limits
            )


def feature_preprocessing_selections(root_output_path, markers_path, cellcutter_markers, image_path):

    save_dir = os.path.join(
        root_output_path, '4_feature_preprocessing_selections'
        )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        markers = pd.read_csv(markers_path)

        # format plot grid
        numRows = 4
        numColumns = 6
        grid_dims = (numRows, numColumns)

        # initialize figure canvas
        fig_orig = plt.figure(figsize=(12, 8.5))
        fig_log = plt.figure(figsize=(12, 8.5))
        fig_clip = plt.figure(figsize=(12, 8.5))

        # loop over cellcutter channels
        cutoffs = {}
        for e, marker in enumerate(cellcutter_markers):

            print(marker)

            # get channel number from markers.csv
            channel_number = markers['channel_number'][
                markers['marker_name'] == marker].values[0]

            # read channel
            img = imread(image_path, key=channel_number-1)

            # log-transform image
            log_img = np.log10(img, where=(img != 0))

            # specify lower and upper percentile cutoffs
            lower_cutoff_log = np.percentile(log_img.ravel(), 0.17)
            upper_cutoff_log = np.percentile(log_img.ravel(), 99.99)

            # add channel cutoffs to dict
            cutoffs[marker] = (lower_cutoff_log, upper_cutoff_log)

            # scale 0.17th and 99.99th percentile between 0 and 1
            # Note: this will cause outlier pixels below the 0.17th percentile
            # and above the 99.99th to take values <0 and >1, respectively
            rescaled_log_img = (
                (((1-0)*(log_img-lower_cutoff_log)) /
                 (upper_cutoff_log-lower_cutoff_log)
                 ) + 0
                )

            # clip outliers to lower and upper percentile cutoffs (i.e., 0-1)
            clip_rescaled_log_img = np.clip(
                a=rescaled_log_img, a_min=0, a_max=1
                )

            # add channel subplot to figures
            ax_orig = fig_orig.add_subplot(grid_dims[0], grid_dims[1], e + 1)
            ax_log = fig_log.add_subplot(grid_dims[0], grid_dims[1], e + 1)
            ax_clip = fig_clip.add_subplot(grid_dims[0], grid_dims[1], e + 1)

            # plot original channel histogram
            vals, bins, patches = ax_orig.hist(
                img.ravel(), bins=60, color='tab:blue',
                alpha=0.7, rwidth=0.85
                )
            ax_orig.title.set_text(marker)

            # plot log-transformed channel histogram
            vals, bins, patches = ax_log.hist(
                log_img.ravel(), bins=60, color='tab:blue',
                alpha=0.7, rwidth=0.85
                )
            ax_log.vlines(
                x=[np.percentile(log_img.ravel(), 0.17),
                   np.percentile(log_img.ravel(), 99.99)],
                ymin=0, ymax=vals.max(), color='tab:red'
                )
            ax_log.title.set_text(marker)

            # plot normalized channel histogram
            vals, bins, patches = ax_clip.hist(
                clip_rescaled_log_img.ravel(), bins=60,
                color='tab:blue', alpha=0.7, rwidth=0.85
                )
            ax_clip.title.set_text(marker)

        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.subplots_adjust(
            bottom=0.01, top=0.99, left=0.01, right=0.99, hspace=0.2
            )
        plt.tight_layout()
        fig_orig.savefig(os.path.join(save_dir, 'log_hists_orig.pdf'))
        fig_log.savefig(os.path.join(save_dir, 'log_hists_log.pdf'))
        fig_clip.savefig(os.path.join(save_dir, 'log_hists_clip.pdf'))
        plt.close('all')

        # save cutoffs to disk
        with open(os.path.join(save_dir, 'cutoffs.pkl'), 'wb') as handle:
            pickle.dump(cutoffs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return save_dir

    else:
        return save_dir


def train_vae(root_output_path, cellcutter_output_path, feature_preprocessing_path, latent_dim, batch_size):

    save_dir = os.path.join(root_output_path, '5_train_vae')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        # clear backend, set random state seed
        K.clear_session()
        np.random.seed(237)

        disable_eager_execution()
        # needed for keras.Model not to throw
        # "You are passing KerasTensor(type_spec=TensorSpec(shape=()..." error
        # this fix is compatible with tensorflow 2.6.3

        def chunks(lst, thumbs_per_batch):
            """Yield successive n-sized chunks from a list."""
            for i in range(0, len(lst), thumbs_per_batch):
                # print(lst[i:i + thumbs_per_batch])
                yield lst[i:i + thumbs_per_batch]

        def transposeZarr(z):
            """Re-organizes thumbnail dimensions as expected VAE input
               (i.e. cells, x, y, channels)."""
            view = DatasetView(z)
            result = view.lazy_transpose([1, 2, 3, 0])

            return result

        # Inner functions for VAE training
        class ShuffleData(keras.callbacks.Callback):

            def on_epoch_end(self, epoch, logs=None):

                keys = list(logs.keys())

                print()

                X = X_train1

                if isinstance(X, np.ndarray):

                    # convert X to Zarr format if not already
                    z = zarr.zeros(
                        shape=(X.shape[0], X.shape[1], X.shape[2], X.shape[3]),
                        chunks=(X_train.chunks[0], X_train.chunks[1],
                                X_train.chunks[2], X_train.chunks[3])
                                )
                    z[:] = X
                    X = z

                # shuffle thumbnails in batches and
                # save as new zarr arrays to avoid memory overload
                rng = np.random.default_rng()

                for e, batch in enumerate(
                  chunks(lst=list(range(X.shape[1])),
                         thumbs_per_batch=8000)
                ):

                    print(
                        f'Shuffling batch {e+1} of thumbnails '
                        f'of length {len(batch)}...')

                    X_shuffle = zarr.open(
                        os.path.join(shuffled_batch_dir, f'batch_{e+1}'),
                        mode='w',
                        shape=(
                            X.shape[0], len(batch),
                            X.shape[2], X.shape[3]
                            ),
                        chunks=(
                            X_train.chunks[0], X_train.chunks[1],
                            X_train.chunks[2], X_train.chunks[3]
                            ),
                        compressor=X.compressor,
                        dtype='float32'
                        )

                    shuffle = X[:, batch[0]:(batch[-1]+1)]

                    rng.shuffle(shuffle, axis=1)
                    X_shuffle[:] = shuffle

                batches = [i for i in os.listdir(shuffled_batch_dir)]
                random.shuffle(batches)

                for e, i in enumerate(batches):

                    print(i)

                    z = zarr.open(
                        os.path.join(shuffled_batch_dir, i), mode='r'
                        )

                    if e == 0:

                        # initialize zarr to store shuffled batches
                        shuffle_final = zarr.open(
                            concatenated_batch_dir,
                            mode='w',
                            shape=(
                                z.shape[0], z.shape[1],
                                z.shape[2], z.shape[3]
                                ),
                            chunks=(
                                z.chunks[0], z.chunks[1],  # cellcutter chunks
                                z.chunks[2], z.chunks[3]
                                ),
                            compressor=z.compressor,
                            dtype='float32'
                            )

                        shuffle_final[:] = z
                        continue

                    shuffle_final.append(z, axis=1)

        def batch_generator(X, batch_size, steps):

            idx_start = 0
            idx_stop = batch_size
            step_counter = 1

            # load shuffled thumbnails if they exist
            if os.path.exists(concatenated_batch_dir):
                X = zarr.open(concatenated_batch_dir)

            while True:

                if step_counter <= steps:

                    # isolate batch of zarr thumbnails (unit16) as numpy array
                    batch = X[:, idx_start:idx_stop, :, :]

                    # convert batch back to Zarr format, but save as float32
                    z = zarr.zeros(
                        shape=(batch.shape[0], batch.shape[1],
                               batch.shape[2], batch.shape[3]),
                        chunks=(X_train.chunks[0], X_train.chunks[0],
                                X_train.chunks[2], X_train.chunks[3]),
                        compressor=X_train.compressor,
                        dtype='float32'
                        )

                    z[:] = batch

                    # rearrange Zarr dimensions to fit shape of VAE input
                    # (i.e. cells, x, y, channels)
                    batch = transposeZarr(z=z)

                    # load data into memory
                    batch = batch[:]

                    # augment batch data
                    for i in range(batch.shape[0]):

                        # # one of 4 rotation angles
                        # rand_rot = random.choice([0, 1, 2, 3])
                        # batch[i] = np.rot90(batch[i], k=rand_rot, axes=(0, 1))
                        #
                        # # flip left or right
                        # flip_lr = random.choice([0, 1])
                        # if flip_lr:
                        #     batch[i] = np.fliplr(batch[i])
                        #
                        # # flip up or down
                        # flip_ud = random.choice([0, 1])
                        # if flip_ud:
                        #     batch[i] = np.flipud(batch[i])

                        # log-transform
                        batch[i] = np.log10(batch[i], where=(batch[i] != 0))

                        # normalize 0.17 and 99.99 percentiles 0-1, per channel
                        for e, (lower_cutoff_log, upper_cutoff_log) in enumerate(
                          cutoffs.values()):

                            batch[i, :, :, e] = (
                                (((1-0)*(batch[i, :, :, e].ravel()-lower_cutoff_log)) /
                                (upper_cutoff_log-lower_cutoff_log)
                                ) + 0).reshape(batch[i, :, :, e].shape)

                            # clip lower and upper outliers to 0 and 1, respectively
                            batch[i, :, :, e] = np.clip(
                                a=batch[i, :, :, e], a_min=0, a_max=1
                                )

                    yield (batch, None)

                    idx_start += batch_size
                    idx_stop += batch_size
                    step_counter += 1

                else:

                    idx_start = 0
                    idx_stop = batch_size
                    step_counter = 1

        def TrainVAE(img_shape, training_epochs, learning_rate):

            # ENCODER NETWORK: Input -> Conv2D*4 -> Flatten -> Dense
            input_img = keras.Input(shape=img_shape)

            opt = RMSprop(learning_rate=learning_rate)

            x = layers.Conv2D(
                filters=32, kernel_size=3,
                padding='same', activation='relu'
                )(input_img)
            x = layers.Conv2D(
                filters=64, kernel_size=3, padding='same',
                activation='relu', strides=(2, 2)
                )(x)
            x = layers.Conv2D(
                filters=64, kernel_size=3,
                padding='same', activation='relu'
                )(x)
            # MAX POOL?
            # x = layers.MaxPooling2D(pool_size=(2, 2),
            #                         strides=2,
            #                         padding='valid')(x)
            x = layers.Conv2D(
                filters=64, kernel_size=3,
                padding='same', activation='relu'
                )(x)

            # need to know the shape of the network here for the decoder
            shape_before_flattening = K.int_shape(x)

            x = layers.Flatten()(x)
            # 850 was hardcoded here instead of latent_dim
            x = layers.Dense(latent_dim, activation='relu')(x)  # 850, was 32

            # two outputs, latent mean and (log)variance
            z_mu = layers.Dense(latent_dim, name='z_mu')(x)
            z_log_sigma = layers.Dense(latent_dim, name='z_log_sigma')(x)

            # SAMPLING FUNCTION
            def sampling(args):
                z_mu, z_log_sigma = args
                epsilon = K.random_normal(
                    shape=(K.shape(z_mu)[0], latent_dim),
                    mean=0.0, stddev=1.0
                    )
                return z_mu + K.exp(z_log_sigma) * epsilon

            # sample vector from the latent distribution
            z = layers.Lambda(sampling)([z_mu, z_log_sigma])

            # DECODER NETWORK
            # decoder takes the latent distribution sample as input
            decoder_input = layers.Input(K.int_shape(z)[1:])

            # expand to N total pixels
            x = layers.Dense(
                np.prod(shape_before_flattening[1:]),
                activation='relu'
                )(decoder_input)

            # reshape
            x = layers.Reshape(shape_before_flattening[1:])(x)

            # use Conv2DTranspose to reverse the conv layers from the encoder
            x = layers.Conv2DTranspose(
                filters=32, kernel_size=3, padding='same',
                activation='relu', strides=(2, 2)
                )(x)
            x = layers.Conv2D(
                filters=X_train1.shape[0], kernel_size=3,
                padding='same', activation='sigmoid'
                )(x)

            # decoder model statement
            decoder = Model(decoder_input, x)

            # apply the decoder to the sample from the latent distribution
            z_decoded = decoder(z)

            # construct a custom layer to calculate the loss
            class CustomVariationalLayer(keras.layers.Layer):

                def vae_loss(self, x, z_decoded):
                    x = K.flatten(x)
                    z_decoded = K.flatten(z_decoded)
                    # reconstruction loss
                    # xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
                    xent_loss = keras.metrics.mean_squared_error(x, z_decoded)

                    # KL divergence (vary this coefficient)  -0.5 -5e-4
                    kl_loss = -5e-4 * K.mean(
                        1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma),
                        axis=-1
                        )
                    return K.mean(xent_loss + kl_loss)

                # adds the custom loss to the class
                def call(self, inputs):
                    x = inputs[0]
                    z_decoded = inputs[1]
                    loss = self.vae_loss(x, z_decoded)
                    self.add_loss(loss, inputs=inputs)
                    return x

            # apply the custom loss to the input images and the
            # decoded latent distribution sample
            y = CustomVariationalLayer()([input_img, z_decoded])

            # VAE model statement
            vae = Model(input_img, y)
            vae.compile(optimizer='rmsprop', loss=None)
            vae.summary()

            checkpoint_path = f"{save_dir}/checkpoints"
            model_checkpoint = ModelCheckpoint(
                filepath=os.path.join(
                    checkpoint_path, 'val_loss-{val_loss:.5f}-cp.ckpt'),
                monitor='val_loss', verbose=1, save_best_only=True,
                save_weights_only=True
                )

            log_dir = os.path.join(
                tensorboard_log_dir,
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                )
            tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0)

            if os.path.exists(checkpoint_path):
                print(f'Loading existing weights at '
                    f'{tf.train.latest_checkpoint(checkpoint_path)}.'
                    )
                vae.load_weights(
                    tf.train.latest_checkpoint(checkpoint_path)
                    ).expect_partial()

            vae.fit(training_batch_generator,
                    epochs=training_epochs,
                    steps_per_epoch=steps_per_epoch,
                    verbose=1, use_multiprocessing=False,
                    validation_data=validation_batch_generator,
                    validation_steps=validation_steps,
                    callbacks=[model_checkpoint, tensorboard, ShuffleData()]
                    )

            # encoder model statement
            encoder = Model(input_img, z_mu)

            # save the encoder and decoder models after training
            save_model(
                encoder, f'{save_dir}/encoder.hdf5',
                overwrite=True, include_optimizer=True)

            save_model(
                decoder, f'{save_dir}/decoder.hdf5',
                overwrite=True, include_optimizer=True)

            return z_mu

        #######################################################################

        # create output directories
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        shuffled_batch_dir = os.path.join(
            save_dir, 'shuffled_thumbnail_batches'
            )

        concatenated_batch_dir = os.path.join(
            save_dir, 'concatenated_shuffled_thumbnails'
            )

        tensorboard_log_dir = os.path.join(save_dir, 'tensorboard_logs/fit')

        #######################################################################

        # read training thumbnails (16-bit unsigned integer format)
        path_numbers = re.findall(r'\d+', cellcutter_output_path)
        window_size = [int(i) for i in path_numbers][-1]

        z1_train_path = (
            os.path.join(cellcutter_output_path,
                         f"train_thumbnails_{window_size}.zarr")
            )
        store = zarr.ZipStore(z1_train_path, mode='r')
        X_train = zarr.open(store=store)

        # read validation thumbnails (16-bit unsigned integer format)
        z1_validate_path = (
            os.path.join(cellcutter_output_path,
                         f"validate_thumbnails_{window_size}.zarr")
            )
        store = zarr.ZipStore(z1_validate_path, mode='r')
        X_valid = zarr.open(store=store)

        #######################################################################

        # elect to hold a slice of data into memory
        # X_train1 = X_train[:, 0:10000, :, :]
        # X_valid1 = X_valid[:, 0:10000, :, :]

        # or read the full training and validation datasets
        X_train1 = X_train
        X_valid1 = X_valid

        #######################################################################

        # read percent cutoffs selected in feature_preprocessing_selections()
        with open(
            os.path.join(feature_preprocessing_path, 'cutoffs.pkl'), 'rb'
          ) as handle:
            cutoffs = pickle.load(handle)

        #######################################################################

        # compute number of training and validation steps per epoch
        steps_per_epoch = int(np.ceil(X_train1.shape[1]/batch_size))
        validation_steps = int(np.ceil(X_valid1.shape[1]/batch_size))

        # initialize batch generators
        training_batch_generator = batch_generator(
            X=X_train1, batch_size=batch_size, steps=steps_per_epoch
            )
        validation_batch_generator = batch_generator(
            X=X_valid1, batch_size=batch_size, steps=validation_steps
            )

        # train VAE
        (encoder, decoder, z_mu) = TrainVAE(
            img_shape=(
                X_train1.shape[2], X_train1.shape[3], X_train1.shape[0]),
            learning_rate=0.001,  # 0.0008 try higher, adaptive learning rates
            training_epochs=100
            )

        return save_dir

    else:
        return save_dir


def main(args):
    root_output_path = args.root_output_path

    # 1_gen_cellcutter_input
    csv_path = args.single_cell_data_path
    frac_sample = args.frac_sample
    cellcutter_input_path = gen_cellcutter_input(
        root_output_path, csv_path, frac_sample
        )

    # 2_run_cellcutter
    image_path = args.image_path
    mask_path = args.mask_path
    window_size = args.window_size
    cells_per_chunk = args.cells_per_chunk
    markers_path = args.markers_path
    cellcutter_markers = args.cellcutter_markers
    cellcutter_markers = [str(a) for a in cellcutter_markers.split(', ')]
    cellcutter_output_path = run_cellcutter(
        root_output_path, cellcutter_input_path, image_path, mask_path,
        markers_path, cellcutter_markers, window_size, cells_per_chunk
        )

    # 3_gen_img_gallery
    num_examples = args.num_examples
    contrast_path = args.contrast_path
    viz_markers = args.viz_markers
    viz_markers = [str(a) for a in viz_markers.split(', ')]
    gen_img_gallery(
        root_output_path, cellcutter_input_path, cellcutter_output_path,
        markers_path, num_examples, cellcutter_markers, viz_markers,
        window_size, contrast_path
        )

    # 4_feature_preprocessing_selections
    feature_preprocessing_path = feature_preprocessing_selections(
        root_output_path, markers_path, cellcutter_markers, image_path
        )

    # 5_train_vae
    latent_dim = args.latent_dim
    batch_size = args.batch_size
    train_vae(
        root_output_path, cellcutter_output_path,
        feature_preprocessing_path, latent_dim, batch_size
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VAE Pipeline')
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # required args  # replace "default="" with "required=True" after debugging
    required.add_argument(
        "--root_output_path", dest="root_output_path",
        help="Path to arbitrary save directory.", type=str, default='/Users/greg/Desktop/vae_test',
        )

    required.add_argument(
        "--single_cell_data_path", dest="single_cell_data_path",
        help="Path of single-cell data.", type=str, default='/Volumes/My Book/cylinter_input/clean_quant/output_3d_v2/consensus_clustering.parquet'
        )
    required.add_argument(
        "--image_path", dest="image_path",
        help="Path of image data.", type=str, default='/Volumes/My Book/cylinter_input/sardana-097/tif/WD-76845-097.ome.tif'
        )
    required.add_argument(
        "--mask_path", dest="mask_path",
        help="Path of segmentation mask.", type=str, default='/Volumes/My Book/cylinter_input/sardana-097/mask/WD-76845-097.tif'
        )
    required.add_argument(
        "--markers_path", dest="markers_path",
        help="Path to marker channel data.", type=str, default='/Volumes/My Book/cylinter_input/sardana-097/markers.csv'
        )
    required.add_argument(
        "--cellcutter_markers", dest="cellcutter_markers",
        help=(
            "Comma-delimited list of marker channels" +
            "for cellcutter processing."), type=str, default="anti_CD3, anti_CD45RO, Keratin_570, aSMA_660, CD4_488, CD45_PE, PD1_647, CD20_488, CD68_555, CD8a_660, CD163_488, FOXP3_570, PDL1_647, Ecad_488, Vimentin_555, CDX2_647, LaminABC_488, Desmin_555, CD31_647, PCNA_488, CollagenIV_647"
            )
    required.add_argument(
        "--viz_markers", dest="viz_markers",
        help=(
            "Comma-delimited list of channel names to" +
            "plot representative thumbnails."), type=str, default="Ecad_488, Keratin_570, aSMA_660, CD4_488, CD20_488, CD8a_660, FOXP3_570, Vimentin_555"
             )
    required.add_argument(
        "--contrast_path", dest="contrast_path",
        help="Path to CyLinter contrast limits for thumbnail generation.",
        type=str, default='/Volumes/My Book/cylinter_input/clean_quant/output_3d_v2/contrast/contrast_limits.yml'
        )

    # optional args
    optional.add_argument(
        "-frac_sample", dest="frac_sample",
        help="Fraction of single-cell data to analyze.",
        type=float, default=0.01
        )
    optional.add_argument(
        "-window_size", dest="window_size",
        help="Square dimension of cellcutter window size (in pixels).",
        type=str, default="30"
        )
    optional.add_argument(
        "-cells_per_chunk", dest="cells_per_chunk",
        help="Number of Cellcutter cells per Zarr chunk.",
        type=str, default="200"
        )
    optional.add_argument(
        "-num_examples", dest="num_examples",
        help="Number of Cellcutter image patches to view.",
        type=int, default=240
        )
    optional.add_argument(
        "-latent_dim", dest="latent_dim", help="VAE latent space dimension.",
        type=int, default=850
        )
    optional.add_argument(
        "-batch_size", dest="batch_size", help="VAE training batch size.",
        type=int, default=32
        )

    args = parser.parse_args()

    # DEBUG
    # required args
    # args.root_output_path = '/Users/greg/Desktop/vae_test'
    # args.single_cell_data_path = '/Volumes/My Book/cylinter_input/clean_quant/output_3d_v2/consensus_clustering.parquet'
    # args.image_path = '/Volumes/My Book/cylinter_input/sardana-097/tif/WD-76845-097.ome.tif'
    # args.mask_path = '/Volumes/My Book/cylinter_input/sardana-097/mask/WD-76845-097.tif'
    # args.markers_path = '/Volumes/My Book/cylinter_input/sardana-097/markers.csv'
    # args.cellcutter_markers = "anti_CD3, anti_CD45RO, Keratin_570, aSMA_660, CD4_488, CD45_PE, PD1_647, CD20_488, CD68_555, CD8a_660, CD163_488, FOXP3_570, PDL1_647, Ecad_488, Vimentin_555, CDX2_647, LaminABC_488, Desmin_555, CD31_647, PCNA_488, CollagenIV_647"
    # args.viz_markers = "Ecad_488, Keratin_570, aSMA_660, CD4_488, CD20_488, CD8a_660, FOXP3_570, Vimentin_555"
    # args.contrast_path = '/Volumes/My Book/cylinter_input/clean_quant/output_3d_v2/contrast/contrast_limits.yml'

    # optional args
    # args.frac_sample = 0.01
    # args.window_size = "30"
    # args.cells_per_chunk = "200"
    # args.num_examples = 240
    # args.latent_dim = 850
    # args.batch_size = 32

    # Run the VAE pipeline
    main(args)