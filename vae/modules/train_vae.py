import logging

import os
import re
import random
import datetime
import pickle

import pandas as pd
import numpy as np

from lazy_ops import DatasetView
import zarr

from tensorflow.python.framework.ops import disable_eager_execution
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import save_model

from ..utils import log_banner, log_multiline

logger = logging.getLogger(__name__)
# log_multiline(logger.info, pd.DataFrame().to_string(index=False))

# to run this script interactively:
# srun --pty -t 12:00:00 --mem=200G -p gpu --gres=gpu:4 bash
# source ~/venvs/vae/bin/activate
# module load gcc/6.2.0 cuda/11.2
# python ~/scripts/vae/5_train_vae.py

# to run this script with sbatch:
# sh ~/scripts/vae/submit.sh

# disable_eager_execution()
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


class ShuffleData(keras.callbacks.Callback):
    def __init__(self, X_train1, X_train, shuffled_batch_dir, concatenated_batch_dir):
        self.X_train1 = X_train1
        self.X_train = X_train
        self.shuffled_batch_dir = shuffled_batch_dir
        self.concatenated_batch_dir = concatenated_batch_dir

    def on_epoch_begin(self, epoch, logs=None):

        keys = list(logs.keys())

        print()

        X = self.X_train1

        if isinstance(X, np.ndarray):
            # convert X to Zarr format if not already
            z = zarr.zeros(
                shape=(X.shape[0], X.shape[1], X.shape[2], X.shape[3]),
                chunks=(self.X_train.chunks[0], self.X_train.chunks[1],
                        self.X_train.chunks[2], self.X_train.chunks[3])
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
                os.path.join(self.shuffled_batch_dir, f'batch_{e+1}'),
                mode='w',
                shape=(
                    X.shape[0], len(batch),
                    X.shape[2], X.shape[3]
                    ),
                chunks=(
                    self.X_train.chunks[0], self.X_train.chunks[1],
                    self.X_train.chunks[2], self.X_train.chunks[3]
                    ),
                compressor=X.compressor,
                dtype='float32'
                )

            shuffle = X[:, batch[0]:(batch[-1]+1)]

            rng.shuffle(shuffle, axis=1)
            X_shuffle[:] = shuffle

        batches = [i for i in os.listdir(self.shuffled_batch_dir)]
        random.shuffle(batches)

        for e, i in enumerate(batches):

            print(i)

            z = zarr.open(
                os.path.join(self.shuffled_batch_dir, i), mode='r'
                )

            if e == 0:

                # initialize zarr to store shuffled batches
                shuffle_final = zarr.open(
                    self.concatenated_batch_dir,
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


def batch_generator(X, X_train, batch_size, steps, cutoffs, concatenated_batch_dir):

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


def train(X_train1, X_train, img_shape, training_batch_generator, validation_batch_generator, steps_per_epoch, validation_steps, training_epochs, learning_rate, latent_dimension, save_dir, concatenated_batch_dir):

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
    # 850 was hardcoded here instead of latent_dimension
    x = layers.Dense(latent_dimension, activation='relu')(x)  # 850, was 32

    # two outputs, latent mean and (log)variance
    z_mu = layers.Dense(latent_dimension, name='z_mu')(x)
    z_log_sigma = layers.Dense(latent_dimension, name='z_log_sigma')(x)

    # SAMPLING FUNCTION
    def sampling(args):
        z_mu, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(K.shape(z_mu)[0], latent_dimension),
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
        filters=img_shape[2], kernel_size=3,
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

    tensorboard_log_dir = os.path.join(save_dir, 'tensorboard_logs/fit')
    log_dir = os.path.join(
        tensorboard_log_dir,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0)

    print(
        'To monitor model training: Open new terminal window, ' +
        'step into VAE virtual environment, ' +
        'run tensorboard --logdir <path_to_tensorboard_fit>'
        )
    print()

    if os.path.exists(checkpoint_path):
        print(f'Loading existing weights at '
            f'{tf.train.latest_checkpoint(checkpoint_path)}.'
            )
        vae.load_weights(
            tf.train.latest_checkpoint(checkpoint_path)
            ).expect_partial()

    shuffled_batch_dir = os.path.join(
        save_dir, 'shuffled_thumbnail_batches'
        )

    vae.fit(training_batch_generator,
            epochs=training_epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=1, use_multiprocessing=False,
            validation_data=validation_batch_generator,
            validation_steps=validation_steps,
            callbacks=[model_checkpoint, tensorboard,
                       ShuffleData(X_train1, X_train,
                                   shuffled_batch_dir, concatenated_batch_dir)]
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


def TRAIN_VAE(config):
    if not os.path.isfile(
      os.path.join(config.output_path, 'checkpoints/TRAIN_VAE.txt')):

        cellcutter_output_path = os.path.join(
            config.output_path, f'2_cellcutter_output_win{config.window_size}'
            )

        feature_preprocessing_path = os.path.join(
            config.output_path, '4_feature_preprocessing_selections'
            )

        save_dir = os.path.join(config.output_path, '5_train_vae')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # clear backend, set random state seed
        K.clear_session()
        np.random.seed(237)

        disable_eager_execution()
        # needed for keras.Model not to throw
        # "You are passing KerasTensor(type_spec=TensorSpec(shape=()..." error
        # this fix is compatible with tensorflow 2.6.3

        # create output directories
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        concatenated_batch_dir = os.path.join(
            save_dir, 'concatenated_shuffled_thumbnails'
            )

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

        # compute number of training and validation steps per epoch
        steps_per_epoch = int(np.ceil(X_train1.shape[1]/config.batch_size))
        validation_steps = int(np.ceil(X_valid1.shape[1]/config.batch_size))

        # initialize batch generators
        training_batch_generator = batch_generator(
            X=X_train1, X_train=X_train, batch_size=config.batch_size,
            steps=steps_per_epoch, cutoffs=cutoffs,
            concatenated_batch_dir=concatenated_batch_dir
            )
        validation_batch_generator = batch_generator(
            X=X_valid1, X_train=X_train, batch_size=config.batch_size,
            steps=validation_steps, cutoffs=cutoffs,
            concatenated_batch_dir=concatenated_batch_dir
            )

        # train VAE
        train(
            X_train1=X_train1,
            X_train=X_train,
            img_shape=(
                X_train1.shape[2], X_train1.shape[3], X_train1.shape[0]),
            training_batch_generator=training_batch_generator,
            validation_batch_generator=validation_batch_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            learning_rate=config.learning_rate,
            training_epochs=config.training_epochs,
            latent_dimension=config.latent_dimension,
            save_dir=save_dir,
            concatenated_batch_dir=concatenated_batch_dir
            )
