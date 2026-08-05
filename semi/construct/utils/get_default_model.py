"""
A helper function to get a default model for quick testing
"""
import os
from omegaconf import open_dict
from hydra import compose, initialize

import torch
from construct.model.construct import Construct
from construct.inference.utils.args_utils import get_dataset_cfg
from construct.utils.download_models import download_models_if_needed


def get_default_model() -> Construct:
    initialize(version_base='1.3.2', config_path="../config", job_name="eval_config")
    cfg = compose(config_name="eval_config")

    weight_dir = download_models_if_needed()
    with open_dict(cfg):
        cfg['weights'] = os.path.join(weight_dir, 'construct-base-mega.pth')
    get_dataset_cfg(cfg)

    # Load the network weights
    construct = Construct(cfg).cuda().eval()
    model_weights = torch.load(cfg.weights)
    construct.load_weights(model_weights)

    return construct
