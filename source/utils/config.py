import os
import json
import pathlib
import datetime

from types import SimpleNamespace


def get_config_from_json(path: str):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(path, mode='r', encoding='utf-8') as config_file:
        config = json.load(
            config_file, object_hook=lambda d: SimpleNamespace(**d))
    return config


def process_config(args):
    config = get_config_from_json(args.config)
    config.save.path = SimpleNamespace()
    now = datetime.datetime.now()
    pat_now = f'{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}'
    config.save.path.log_dir = os.path.join(
        args.save_root, config.exp_name, pat_now, 'log')
    config.save.path.checkpoint_dir = os.path.join(
        args.save_root, config.exp_name, pat_now, 'checkpoint')
    config.dataset.path = pathlib.Path(args.dataset)
    return config
