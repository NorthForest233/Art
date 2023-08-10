from omegaconf import OmegaConf
import importlib
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type=str,
        help='path to config',
    )
    return parser

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    return model

def instantiate_from_config(config):
    if not 'target' in config:
        return None
    return get_obj_from_str(config['target'])(**config.get('params', dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
