import yaml
import pathlib


class Config:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_path(cls, path):
        config = cls()
        with open(path) as f:
            data = yaml.safe_load(f)
        config.output_path = pathlib.Path(data['output_path']).resolve()
        config.csv_path = pathlib.Path(data['csv_path']).resolve()
        config.tif_path = pathlib.Path(data['tif_path']).resolve()
        config.outlines_path = pathlib.Path(data['outlines_path']).resolve()
        config.mask_path = pathlib.Path(data['mask_path']).resolve()
        config.markers_path = pathlib.Path(data['markers_path']).resolve()
        config.contrast_path = pathlib.Path(data['contrast_path']).resolve()
        config.tif_channels = list(data['tif_channels'])
        config.percent_cells = float(data['percent_cells'])
        config.window_size = int(data['window_size'])
        config.cache_size_cellcutter = int(data['cache_size_cellcutter'])
        config.cells_per_chunk = int(data['cells_per_chunk'])
        config.gallery_size = int(data['gallery_size'])
        config.latent_dimension = int(data['latent_dimension'])
        config.cutoffs = list(data['cutoffs'])
        config.batch_size = int(data['batch_size'])
        config.masked_model = bool(data['masked_model'])
        config.mask_std_dev = int(data['mask_std_dev'])
        config.learning_rate = float(data['learning_rate'])
        config.training_epochs = int(data['training_epochs'])
        config.cluster_full_dataset = bool(data['cluster_full_dataset'])
        config.clustering_sample_size = int(data['clustering_sample_size'])
        config.embedding_algorithm = str(data['embedding_algorithm'])
        config.channel_colors = dict(data['channel_colors'])
        config.hdbscan_min_cluster_size = int(data['hdbscan_min_cluster_size'])
        config.lasso_vector_tool = bool(data['lasso_vector_tool'])
        return config

    @property
    def checkpoint_path(self):
        return self.output_path / 'checkpoints'

    def __repr__(self):
        kwargs_str = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Config({kwargs_str})"
