import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import config
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator

def get_model_diff():
    """
    This function defines U-net baseline of the DDPM
    """
    unet = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=1,
        num_res_blocks=2,
        channels=(128, 256, 512),
        attention_levels=(False, True, True),
        num_head_channels=(0, 256, 512),
    )

    return unet

