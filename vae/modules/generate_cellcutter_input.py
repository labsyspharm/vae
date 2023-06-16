import logging

import os
import pandas as pd

from ..utils import log_banner, log_multiline

logger = logging.getLogger(__name__)
# log_multiline(logger.info, pd.DataFrame().to_string(index=False))


def GENERATE_CELLCUTTER_INPUT(config):

    save_dir = os.path.join(config.output_path, '1_cellcutter_input')
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
                f'Note: extension type {extension} not yet supported.'
                )

        # drop cells without a consensus cluster (i.e. noisy cells)
        # specific to HDBSCAN clustered data for now
        csv = csv[csv['cluster'] != -1]

        #######################################################################

        # calculate weighted random sample by cluster size (class balance)
        groups = csv.groupby('cluster')
        sample_weights = pd.DataFrame(
            {'weights': 1 / (groups.size() * len(groups))}
            )
        weights = pd.merge(
            csv[['cluster']], sample_weights,
            left_on='cluster', right_index=True
            )

        csv = csv.sample(
            frac=F, replace=False, weights=weights['weights'],
            random_state=0, axis=0
            )
        print()
        print('Cells per cluster after cluster-weighted random sampling:')
        print(csv.groupby('cluster').size().sort_values(ascending=False))

        #######################################################################

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

        #######################################################################

        # save testing, validation, and training dataframes for cellcutter
        test.to_csv(os.path.join(save_dir, 'test.csv'), index=False)
        validate.to_csv(os.path.join(save_dir, 'validate.csv'), index=False)
        train.to_csv(os.path.join(save_dir, 'train.csv'), index=False)

        return save_dir

    else:
        return save_dir
