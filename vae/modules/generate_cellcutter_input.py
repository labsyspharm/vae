import os

import logging

import numpy as np
import pandas as pd

from ..utils import log_banner, log_multiline

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# log_multiline(logger.info, pd.DataFrame().to_string(index=False))
# log_banner(logger.info, 'Boolean classifications')


def GENERATE_CELLCUTTER_INPUT(config):
    
    if not os.path.isfile(
       os.path.join(config.output_path,
                    'checkpoints/GENERATE_CELLCUTTER_INPUT.txt')):
        
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

        #######################################################################
        # weighted random sampling
        
        cluster_col = (
            "cluster_2d" if "cluster_2d" in csv.columns else
            "cluster_3d" if "cluster_3d" in csv.columns else
            None
        )

        n_samples = int(len(csv) * config.percent_cells)

        if cluster_col is not None:

            csv = csv[csv[cluster_col] != -1].copy()

            cluster_sizes = csv[cluster_col].value_counts()

            weights = csv[cluster_col].map(
                lambda c: 1.0 / cluster_sizes[c]
            ).to_numpy()

            weights = weights / weights.sum()

            chosen_idx = np.random.choice(
                csv.index,
                size=n_samples,
                replace=False,
                p=weights
            )

            csv = csv.loc[chosen_idx]

            print("Cells per cluster after cluster-weighted sampling:")
            print(csv.groupby(cluster_col).size().sort_values(ascending=False))

        else:
            chosen_idx = np.random.choice(
                csv.index,
                size=n_samples,
                replace=False
            )
            csv = csv.loc[chosen_idx]

        #######################################################################
        
        print()
        logger.info(
            'Partitioning CyLinter dataframe into training, ' 
            'validation, and test sets...'
        )
        print()
        
        # Shuffle csv data
        csv = csv.sample(frac=config.percent_cells, random_state=0)

        # Reserve 10% of data for testing after model training 
        split = round(len(csv) * 0.10)
        test = csv[0:split].copy()
        test.sort_values(by=['Sample', 'CellID'], inplace=True)
        
        # Reserve 10% of data for validation at the end of each training epoch
        validate = csv[split:split * 2].copy()
        validate.sort_values(by=['Sample', 'CellID'], inplace=True)
        
        # Use remaining data for model training
        train = csv[split * 2:].copy()
        train.sort_values(by=['Sample', 'CellID'], inplace=True)
    
        # Reset row indexes of each dataframe
        test.reset_index(drop=True, inplace=True)
        validate.reset_index(drop=True, inplace=True)
        train.reset_index(drop=True, inplace=True)

        #######################################################################

        # Save testing, validation, and training dataframes for cellcutter
        test.to_csv(os.path.join(save_dir, 'test_raw.csv'), index=False)
        validate.to_csv(os.path.join(save_dir, 'validate_raw.csv'), index=False)
        train.to_csv(os.path.join(save_dir, 'train_raw.csv'), index=False)
