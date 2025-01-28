import sys
import logging
import pathlib
import argparse

from ..config import Config
from .. import pipeline, components

logger = logging.getLogger(__name__)


def main(argv=sys.argv):

    epilog = 'Pipeline modules:\n'
    epilog += '\n'.join(f"    {n}" for n in components.pipeline_module_names)
    parser = argparse.ArgumentParser(
        description='Train a VAE model on cropped image patches and cluster their latent encodings',
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser = argparse.ArgumentParser(description='VAE Pipeline')
    parser._action_groups.pop()
    
    # required = parser.add_argument_group('required arguments')
    # optional = parser.add_argument_group('optional arguments')

    parser.add_argument("config", type=path_resolved, help="Path to YAML configuration file")
    parser.add_argument("--module", type=str, help="Pipeline module at which to begin processing")

    args = parser.parse_args(argv[1:])
    if not validate_paths(args):
        return 1
    if args.module and args.module not in components.pipeline_module_names:
        print(f"Error: --module {args.module} is not a pipeline module", file=sys.stderr)
        return 1

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    logger.info("Reading configuration file")
    config = Config.from_path(args.config)
    create_output_directory(config)

    logger.info("Executing pipeline")
    pipeline.run_pipeline(config, args.module)

    logger.info("Finished")

    return 0


def path_resolved(path_str):
    """Return a resolved Path for a string."""
    path = pathlib.Path(path_str)
    path = path.resolve()
    return path


def validate_paths(args):
    """Validate the Path entries in the argument list."""
    ok = True
    if not args.config.exists():
        print(
            f"Config path does not exist:\n     {args.config}\n",
            file=sys.stderr
        )
        ok = False
    return ok


def create_output_directory(config):
    """Create the output directory structure given the configuration object."""
    config.output_path.mkdir(parents=True, exist_ok=True)
