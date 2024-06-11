import torch
import torch.nn as nn
import torch.nn.functional as F

import monai
from monai.networks.blocks.convolutions import ResidualUnit
from monai.networks.layers.factories import Act, Norm

from functools import partial

from .base import SegmentorWithTTA
from .sa_block import SABlock

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeat=1):
        super().__init__()
        
        self.downsampling_layer = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
        )
        
        self.block = nn.Sequential(*[
            ResidualUnit(
            3,
            in_channels,
            in_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
            act=Act.GELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            last_conv_only=False,
            adn_ordering="NDA",
        ) for _ in range(repeat)
        ])
    
    def forward(self, x):
        x = self.block(x)
        return x, self.downsampling_layer(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        self.upsampling_layer = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2),
        )
        
        
        self.block = ResidualUnit(
            3,
            in_channels + skip_channels,
            out_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
            act=Act.GELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            last_conv_only=False,
            adn_ordering="NDA",
        )
    
    def forward(self, x, skip):
        x = self.upsampling_layer(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        
        return x


'''
@register_model
def uniformer_small(pretrained=True, **kwargs):
    model = UniFormer(
        depth=[3, 4, 8, 3],
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
'''


class CenterBlock(nn.Module):
    def __init__(self, in_channels, depth=1):
        super().__init__()
        
        self.block = nn.Sequential(*[
            SABlock(
                dim=in_channels, num_heads=in_channels // 64, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            for _ in range(depth)
        ])
    
    def forward(self, x):
        return self.block(x)

class CNN_SA_UNET(nn.Module):
    def __init__(self):
        super().__init__()
        
        list_enc_channels = [32, 64, 128, 256, 320, 512]
        list_enc_repeat = [1, 1, 1, 2, 4, 3]
        
        list_dec_channels = [16, 32, 64, 128, 256, 512]
        
        self.pre_conv = nn.Sequential(
            nn.InstanceNorm3d(4),
            nn.Conv3d(4, list_enc_channels[0], kernel_size=1),
        )
        
        self.encoders = nn.ModuleList([
            EncoderBlock(list_enc_channels[i], list_enc_channels[i+1], repeat=repeat) \
            for i, repeat in enumerate(list_enc_repeat[:-1])
        ])
        
        self.center = CenterBlock(list_enc_channels[-1], depth=list_enc_repeat[-1])
        
        self.decoders = nn.ModuleList([
            DecoderBlock(list_dec_channels[-i], list_enc_channels[-i-1], list_dec_channels[-i-1]) \
            for i in range(1, len(list_dec_channels))
        ])
        
        self.last_conv = nn.Sequential(
            nn.Conv3d(list_dec_channels[0], 3, kernel_size=1),
        )
    
    def forward(self, x):
        x = self.pre_conv(x)
        
        skip_features = []
        for encoder in self.encoders:
            f, x = encoder(x)
            skip_features.append(f)
        
        x = self.center(x)
        
        out_features = []
        for decoder, skip_f in zip(self.decoders, skip_features[::-1]):
            x = decoder(x, skip_f)
            out_features.append(x)
        
        y = self.last_conv(out_features[-1])
        
        return y