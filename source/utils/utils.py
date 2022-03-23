import argparse
from pathlib import Path


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        required=True,
        help='configuration file path')
    argparser.add_argument(
        '-s', '--save_root',
        metavar='S',
        required=True,
        help='Checkpoint and summary saving directory path')
    args = argparser.parse_args()
    return args


def get_project_root() -> Path:
    return Path(__file__).parent.parent
