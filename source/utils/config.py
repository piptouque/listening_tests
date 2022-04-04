import os
import json

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
    config.save.path.log_dir = os.path.join(
        args.save_root, config.exp_name, "log/")
    config.save.path.checkpoint_dir = os.path.join(
        args.save_root, config.exp_name, "checkpoint/")
    return config
