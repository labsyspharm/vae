import os
import re
import sys
import pickle
import logging

import datetime

import pandas as pd
import numpy as np

import zarr
import dask.array as da

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence  # works with generator
# from tensorflow.python.keras.utils.data_utils import Sequence  # tf dataset
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from ..utils import (
    log_banner, log_multiline, log_transform, 
    align_histograms, compute_vignette_mask
)

###############################################################################
# to run this script interactively on HMS o2 GPU partition:

# srun --pty -p gpu_requeue -t 0-24:00 --gres=gpu:1 --mem=100G bash

# scontrol show node compute-gc-17-152 (to see specs for node)
# nvidia-smi (to see specs for the resourced GPU(s))

# conda activate vae

# module load gcc/9.2.0 python/3.10.11 cuda/12.1 
# (compatible with tensorflow=2.15.0 and L40S, teslaX100, 
#  RTX8000, A100, and maybe other GPUs)

# cd to VAE I/O directory

# vae --module TRAIN_VAE config.yml
# requires tensorflow-gpu to be installed

# to run this script with sbatch:
# sh ~/scripts/vae/submit.sh
###############################################################################

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# log_multiline(logger.info, pd.DataFrame().to_string(index=False))
# log_banner(logger.info, 'Boolean classifications')


class DataGenerator(Sequence):
    'generates data for Keras model fit'
    
    def __init__(
            self, name, zarr, y, batch_size, limits, 
            masked_model, mask, shuffle=True):
        'initialization'
        self.name = name
        self.zarr = zarr
        self.y = y
        self.batch_size = batch_size
        self.limits = limits
        self.masked_model = masked_model
        self.mask = mask
        self.shuffle = shuffle
        self.indices = np.arange(zarr.shape[1])
        self.on_epoch_end()

    def __len__(self):
        'denotes number of batches per epoch'
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        'generates indices for one batch of data'
        batch_indices = self.indices[
            index * self.batch_size:(index + 1) * self.batch_size
        ]
        batch_data = self.__data_generation(batch_indices)
        return batch_data
    
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            
            if i == self.__len__() - 1:
                self.on_epoch_end()
    
    def on_epoch_end(self):
        'updates indices after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices):
        'yields data containing batch_size samples'
        X = self.zarr.oindex[:, batch_indices, :, :].transpose([1, 2, 3, 0])
        
        # X = X.astype('float')  # use for binary patches
        
        X = da.from_array(X)
  
        labels = self.y[batch_indices]
        dask_labels = da.from_array(labels.values, chunks=(X.chunksize[0],))
        dask_labels = dask_labels.reshape((-1, 1, 1, 1))

        # preprocess image patches
        X = da.map_blocks(
            align_histograms, X, 
            dask_labels, self.limits, dtype=np.float32,
        ).compute()

        # apply mask
        if self.masked_model:
            X *= self.mask

        # rotational data augmention
        if self.name == 'train':
            X = rotate_batch(X)

        # NaN test
        if np.isnan(X).any():
            print('Batch contains NaNs!!!')
            sys.exit(1)
        
        return X


def rotate_batch(batch):
    num_images = batch.shape[0]
    rotation_angles = np.random.randint(0, 4, size=num_images)
    for i in range(num_images):
        batch[i] = np.rot90(batch[i], rotation_angles[i], axes=(0, 1))
    return batch


def latest_keras_model_checkpoint(checkpoint_path):
    # list all files in the checkpoint directory
    checkpoints = [f for f in os.listdir(checkpoint_path) if 
                   f.endswith('.keras')]
    
    if not checkpoints:
        raise ValueError("No checkpoints found in the directory.")
    
    # get the full path of each checkpoint
    checkpoints = [os.path.join(checkpoint_path, f) for f in checkpoints]
    
    # sort checkpoints based on modification time
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    
    return latest_checkpoint


def build_and_fit_model(img_shape, latent_dimension, learning_rate, training_epochs, training_data_generator, steps_per_epoch, validation_data_generator, validation_steps, save_dir):
      
    # ENCODER NETWORK: Input -> Conv2D*4 -> Flatten -> Dense
    input_img = keras.Input(shape=img_shape)

    RMSprop(learning_rate=learning_rate)

    x = layers.Conv2D(
        filters=32, kernel_size=3, padding='same', activation='relu'
    )(input_img)
    x = layers.Conv2D(
        filters=64, kernel_size=3, padding='same', activation='relu', 
        strides=(2, 2)
    )(x)
    x = layers.Conv2D(
        filters=64, kernel_size=3, padding='same', activation='relu'
    )(x)

    # MAX POOL
    # x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    x = layers.Conv2D(
        filters=64, kernel_size=3, padding='same', activation='relu'
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
            shape=(K.shape(z_mu)[0], latent_dimension), mean=0.0, stddev=1.0
        )
        return z_mu + K.exp(z_log_sigma) * epsilon

    # sample vector from the latent distribution
    z = layers.Lambda(sampling)([z_mu, z_log_sigma])

    # DECODER NETWORK
    # decoder takes the latent distribution sample as input
    decoder_input = layers.Input(K.int_shape(z)[1:])

    # expand to N total pixels
    x = layers.Dense(
        np.prod(shape_before_flattening[1:]), activation='relu'
    )(decoder_input)

    # reshape
    x = layers.Reshape(shape_before_flattening[1:])(x)

    # use Conv2DTranspose to reverse the conv layers from the encoder
    x = layers.Conv2DTranspose(
        filters=32, kernel_size=3, padding='same', activation='relu', 
        strides=(2, 2)
    )(x)
    x = layers.Conv2D(
        filters=img_shape[2], kernel_size=3, padding='same', 
        activation='sigmoid'
    )(x)

    # decoder model statement
    decoder = Model(decoder_input, x)

    # apply the decoder to the sample from the latent distribution
    z_decoded = decoder(z)

    # construct a custom layer to calculate the loss
    @keras.saving.register_keras_serializable()
    class CustomVariationalLayer(keras.layers.Layer):

        def vae_loss(self, x, z_decoded, z_mu, z_log_sigma):
            x = K.flatten(x)
            z_decoded = K.flatten(z_decoded)
            
            # reconstruction loss
            xent_loss = keras.metrics.mean_squared_error(x, z_decoded)
            # xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)

            # KL divergence
            kl_loss = -5e-4 * K.mean(
                1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1
            )
            return K.mean(xent_loss + kl_loss)

        # adds the custom loss to the class
        def call(self, inputs):
            x = inputs[0]
            z_decoded = inputs[1]
            z_mu = inputs[2]
            z_log_sigma = inputs[3]
            loss = self.vae_loss(x, z_decoded, z_mu, z_log_sigma)
            self.add_loss(loss, inputs=inputs)
            return x

    # apply the custom loss to the input images and the
    # decoded latent distribution sample
    y = CustomVariationalLayer()([input_img, z_decoded, z_mu, z_log_sigma])

    # VAE model statement
    vae = Model(input_img, y)
    vae.compile(optimizer='RMSprop', loss=None)
    vae.summary()

    # initialize tensorboard
    tensorboard_log_dir = os.path.join(save_dir, 'tensorboard_logs/fit')
    log_dir = os.path.join(
        tensorboard_log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0)
    logger.info(
        'To monitor model training: Open new terminal window, '
        'step into VAE virtual environment, '
        'run tensorboard --logdir <path_to_tensorboard_fit>'
    )
    print()

    checkpoint_path = f"{save_dir}/checkpoints"
    
    # load existing model
    if os.path.exists(checkpoint_path):

        latest_checkpoint = latest_keras_model_checkpoint(
            checkpoint_path=checkpoint_path
        )

        # load an arbitrary model (optional)
        latest_checkpoint = (
            '/n/scratch/users/g/gjb15/binary_patches/'
            '5_train_vae/checkpoints/val_loss-0.00412-model.keras'
        )
        
        print(f'Loading latest model at {latest_checkpoint}')
        print()
        
        # load latest model
        vae = load_model(latest_checkpoint, compile=True, safe_mode=False) 
        
        # set initial_value_threshold
        pattern = r'val_loss-(\d+\.\d+)-model\.keras'
        match = re.search(pattern, latest_checkpoint)
        initial_value_threshold = float(match.group(1))
    
    else:
        initial_value_threshold = None

    # initialize model checkpoint callback
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            checkpoint_path, 'val_loss-{val_loss:.5f}-model.keras'),
        monitor='val_loss', verbose=1, save_best_only=True, 
        save_weights_only=False, 
        initial_value_threshold=initial_value_threshold
    )

    # fit model
    vae.fit(
        x=training_data_generator, steps_per_epoch=steps_per_epoch,
        validation_data=validation_data_generator, 
        validation_steps=validation_steps, epochs=training_epochs, 
        use_multiprocessing=False, workers=4, verbose=1, 
        callbacks=[model_checkpoint, tensorboard]  
    )

    if os.path.exists(checkpoint_path):
        
        # encoder model statement
        encoder_input = vae.input
        encoder_output = vae.get_layer('z_mu').output
        encoder = Model(encoder_input, encoder_output)

        # extract decoder from vae model
        decoder = vae.get_layer('model')

    else:
        # encoder model statement
        encoder = Model(input_img, z_mu)

        # decoder model statement is specified in VAE built above in this case

    # save the encoder and decoder models after training
    save_model(
        encoder, f'{save_dir}/encoder.hdf5', overwrite=True, 
        include_optimizer=True
    )
    save_model(
        decoder, f'{save_dir}/decoder.hdf5', overwrite=True, 
        include_optimizer=True
    )


def TRAIN_VAE(config):

    if not os.path.isfile(
      os.path.join(config.output_path, 'checkpoints/TRAIN_VAE.txt')):

        # clear backend, set random state seed
        K.clear_session()
        np.random.seed(237)

        cellcutter_output_path = os.path.join(
            config.output_path, f'2_cellcutter_output_win{config.window_size}'
        )

        histogram_alignment_path = os.path.join(
            config.output_path, '4_histogram_alignment'
        )

        save_dir = os.path.join(config.output_path, '5_train_vae')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        path_numbers = re.findall(r'\d+', cellcutter_output_path)
        window_size = [int(i) for i in path_numbers][-1]

        # read training and validation thumbnails (16-bit unsigned integers)
        z1_train_path = (
            os.path.join(cellcutter_output_path, 
                         f'train_thumbnails_{window_size}.zip')
        )
        store = zarr.ZipStore(z1_train_path, mode='r')
        X_train = zarr.open(store=store)
        # X_train = X_train[:, 0:1000, :, :]
        # X_train = zarr.array(
        #     X_train, chunks=(21, 1, 62, 62), dtype=X_train.dtype
        # )
        
        # read validation thumbnails (16-bit unsigned integers)
        z1_validate_path = (
            os.path.join(cellcutter_output_path, 
                         f'validate_thumbnails_{window_size}.zip')
        )
        store = zarr.ZipStore(z1_validate_path, mode='r')
        X_valid = zarr.open(store=store)
        # X_valid = X_valid[:, 0:1000, :, :]
        # X_valid = zarr.array(
        #     X_valid, chunks=(21, 1, 62, 62), dtype=X_valid.dtype
        # )

        # read training labels
        y_train = pd.read_csv(
            os.path.join(config.output_path, '1_cellcutter_input/train.csv')
        )
        y_train = y_train['Sample']
        
        # read validation labels
        y_validate = pd.read_csv(
            os.path.join(config.output_path, '1_cellcutter_input/validate.csv')
        )
        y_validate = y_validate['Sample']

        # read histogram scaling functions computed in align_histograms.py
        with open(
          os.path.join(
           histogram_alignment_path, 
           'limits.pkl'), 'rb') as handle:
            limits = pickle.load(handle)

        # compute steps per epoch for training data
        steps_per_epoch = X_train.shape[1] // config.batch_size 

        # compute steps per epoch for validation data 
        validation_steps = X_valid.shape[1] // config.batch_size

        # compute vignette mask
        mask, vmin, vmax = compute_vignette_mask(
            window_size=config.window_size, std_dev=config.mask_std_dev
        )

        # initialize training data generator
        # (does not seem to work with multi-GPU processing)
        training_data_generator = DataGenerator(
            name='train', zarr=X_train, y=y_train, 
            batch_size=config.batch_size, limits=limits, 
            masked_model=config.masked_model, mask=mask, shuffle=True
        )

        # initialize validation data generator
        validation_data_generator = DataGenerator(
            name='valid', zarr=X_valid, y=y_validate, 
            batch_size=config.batch_size, limits=limits, 
            masked_model=config.masked_model, mask=mask, shuffle=False
        )

        # CONVERT BATCH GENERATOR TO TENSORFLOW DATASET
        # def generator_to_tfdata(data_generator):
        #     output_signature = tf.TensorSpec(
        #         shape=(data_generator.batch_size,) + 
        #         (X_train.shape[2], X_train.shape[3], X_train.shape[0]), dtype=tf.float32
        #     )
        #     dataset = tf.data.Dataset.from_generator(
        #         data_generator, output_signature=output_signature
        #     )
        #     return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # tf_training_dataset = generator_to_tfdata(data_generator=training_data_generator)
        # tf_validation_dataset = generator_to_tfdata(data_generator=validation_data_generator)
        
        print()
        print('GPUs available: ', len(tf.config.list_physical_devices('GPU')))
        print()
        
        # build model
        if len(tf.config.list_physical_devices('GPU')) > 1:
            
            # use distributed GPU strategy
            strategy = tf.distribute.MirroredStrategy()  # use all GPUs

            with strategy.scope():
                build_and_fit_model(
                    img_shape=(X_train.shape[2], X_train.shape[3], 
                               X_train.shape[0]),
                    latent_dimension=config.latent_dimension, 
                    learning_rate=config.learning_rate,
                    training_epochs=config.training_epochs,
                    training_data_generator=training_data_generator,
                    steps_per_epoch=steps_per_epoch, 
                    validation_data_generator=validation_data_generator,
                    validation_steps=validation_steps, 
                    save_dir=save_dir
                )
        else:
            build_and_fit_model(
                img_shape=(X_train.shape[2], X_train.shape[3], 
                           X_train.shape[0]),
                latent_dimension=config.latent_dimension, 
                learning_rate=config.learning_rate,
                training_epochs=config.training_epochs,
                training_data_generator=training_data_generator,
                steps_per_epoch=steps_per_epoch, 
                validation_data_generator=validation_data_generator,
                validation_steps=validation_steps, 
                save_dir=save_dir
            )
