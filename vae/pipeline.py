import pyarrow
import pyarrow.parquet
import pandas as pd
from . import components


def save_checkpoint(config, module):
    module_name = module.__name__
    path = config.checkpoint_path / f"{module_name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    open(path, mode='a').close()


def run_pipeline(config, start_module_name):
    if (
        start_module_name is None
        or start_module_name == components.pipeline_module_names[0]
    ):
        start_index = 0
        data = None
    else:
        start_index = components.pipeline_module_names.index(start_module_name)
        previous_module_name = components.pipeline_module_names[start_index - 1]
        checkpoint_file_path = (
            config.checkpoint_path / f"{previous_module_name}.txt"
        )
        if not checkpoint_file_path.exists():
            raise Exception(
                f"Checkpoint file for module {previous_module_name} not found"
            )

    for module in components.pipeline_modules[start_index:]:
        module(config)
        save_checkpoint(config, module)
