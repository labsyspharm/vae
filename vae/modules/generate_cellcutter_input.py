import os
import logging

import pandas as pd

from ..utils import log_banner, log_multiline

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# log_multiline(logger.info, pd.DataFrame().to_string(index=False))
# log_banner(logger.info, 'Boolean classifications')


def GENERATE_CELLCUTTER_INPUT(config):

    save_dir = os.path.join(config.output_path, '1_cellcutter_input')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        extension = os.path.splitext(config.csv_path)[1]
        if extension == '.parquet':
            csv = pd.read_parquet(config.csv_path)
        elif extension == '.csv':
            csv = pd.read_csv(config.csv_path)
        else:
            raise ValueError(f'Note: extension type {extension} not supported.')

        # drop noisy cells from HDBSCAN clustering
        csv = csv[csv['cluster_2d'] != -1]

        #######################################################################

        # calculate weighted random sample by cluster size (class balance)
        groups = csv.groupby('cluster_2d')
        sample_weights = pd.DataFrame({'weights': 1 / (groups.size() * len(groups))})
        weights = pd.merge(
            csv[['cluster_2d']], sample_weights, left_on='cluster_2d', right_index=True
        )

        csv = csv.sample(
            frac=config.percent_cells, replace=False, weights=weights['weights'],
            random_state=0, axis=0
        )

        # remove cells affected by artifacts missed during initial QC
        # these artifacts were identified by VAE clustering 
        # (cluster 27, 29, and 30 in 20x20um analysis)
        
        # residual_artifact_cellids = main['CellID'][main['VAE20_ROT_res2.0'].isin([27, 29, 30])]
        residual_artifact_cellids = pd.read_csv(
            '/Users/greg/projects/vae-paper/src/input/residual_artifact_cellids.csv'
        )
        csv = csv[~csv['CellID'].isin(residual_artifact_cellids['CellID'])]

        print()
        print('Cells per cluster after cluster-weighted random sampling:')
        print(csv.groupby('cluster_2d').size().sort_values(ascending=False))

        #######################################################################

        # shuffle csv data
        csv = csv.sample(frac=1.0, random_state=0)

        # reserve 10% of data for testing after model training 
        split = round(len(csv) * 0.10)
        test = csv[0:split]
        
        # reserve 10% of data for validation at the end of each training epoch
        validate = csv[split:split * 2]
        
        # use remaining data for model training
        train = csv[split * 2:]

        # reset row indexes of each dataframe
        test.reset_index(drop=True, inplace=True)
        validate.reset_index(drop=True, inplace=True)
        train.reset_index(drop=True, inplace=True)

        #######################################################################

        # save testing, validation, and training dataframes for cellcutter
        test.to_csv(os.path.join(save_dir, 'test.csv'), index=False)
        validate.to_csv(os.path.join(save_dir, 'validate.csv'), index=False)
        train.to_csv(os.path.join(save_dir, 'train.csv'), index=False)

        return save_dir

    else:
        return save_dir
