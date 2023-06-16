import logging

import os
import pandas as pd

from subprocess import run

from ..utils import log_banner, log_multiline

logger = logging.getLogger(__name__)
# log_multiline(logger.info, pd.DataFrame().to_string(index=False))


def RUN_CELLCUTTER(config):

    cellcutter_input_path = os.path.join(
        config.output_path, '1_cellcutter_input'
        )

    save_dir = os.path.join(
        config.output_path, f'2_cellcutter_output_win{config.window_size}'
        )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        markers = pd.read_csv(markers_path)

        cellcutter_marker_ids = []
        for i in cellcutter_markers:
            id = markers['channel_number'][
                markers['marker_name'] == i].values[0]
            cellcutter_marker_ids.append(str(id))

        for name in ['train', 'validate', 'test']:
            print()
            print(f'Cutting {name} data...')
            run(
                ["cut_cells", "-z", "--window-size", window_size,
                 "--cells-per-chunk", config.cells_per_chunk,
                 "--cache-size", "57711", image_path, mask_path,
                 os.path.join(cellcutter_input_path, f"{name}.csv"),
                 os.path.join(
                    save_dir, f"{name}_thumbnails_{config.window_size}.zarr"),
                 "--channels"] + cellcutter_marker_ids
                )

            run(
                ["cut_cells", "-z", "--window-size", window_size,
                 "--cells-per-chunk", cells_per_chunk, "--cache-size", "57711",
                 seg_path, mask_path,
                 os.path.join(cellcutter_input_path, f"{name}.csv"),
                 os.path.join(
                    save_dir,
                    f"{name}_thumbnails_{config.window_size}_seg.zarr"),
                 "--channels", "1"
                 ]
                )
