from .convnextv2 import convnextv2_nano
from .convnextv2 import convnextv2_tiny
import timm
import torch
import os


def build_convnextv2_model(model_type, freeze_at=0):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'pretrain_models')

    if model_type == 'convnextv2_nano.fcmae_ft_in22k_in1k':
        model = convnextv2_nano(drop_path_rate=0.4,
                                head_init_scale=1.0)
        checkpoint = torch.load(os.path.join(model_dir, 'convnextv2_nano_22k_224_ema.pt'))
        model.load_state_dict(checkpoint["model"], strict=False)

    elif model_type == 'convnextv2_tiny.fcmae_ft_in22k_in1k':
        model = convnextv2_tiny(drop_path_rate=0.4,
                                head_init_scale=1.0)
        checkpoint = torch.load(os.path.join(model_dir, 'convnextv2_tiny_22k_224_ema.pt'))
        model.load_state_dict(checkpoint["model"], strict=False)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
