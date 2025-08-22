

# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .swin_transformer import SwinTransformer
from .swin_transformerv2 import SwinTransformerV2


def norm_targets(targets, patch_size):
    assert patch_size % 2 == 1
    
    targets_ = targets
    targets_count = torch.ones_like(targets)

    targets_square = targets ** 2.
    
    targets_mean = F.avg_pool2d(targets, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_square_mean = F.avg_pool2d(targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_count = F.avg_pool2d(targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=True) * (patch_size ** 2)
    
    targets_var = (targets_square_mean - targets_mean ** 2.) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.)
    
    targets_ = (targets_ - targets_mean) / (targets_var + 1.e-6) ** 0.5
    
    return targets_


class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, masks=None, is_training=False):
        if is_training and masks is None:
            return super().forward(x, is_training=True)
        
        # Handle list input (multi-crop training)
        if isinstance(x, list):
            return super().forward_features(x, masks)
        
        # SimMIM reconstruction mode
        x = self.patch_embed(x)
        
        if masks is not None:
            B, L, _ = x.shape
            mask_tokens = self.mask_token.expand(B, L, -1)
            w = masks.flatten(1).unsqueeze(-1).type_as(mask_tokens)
            x = x * (1. - w) + mask_tokens * w

        if self.ape:
            # Use original position embedding without cls token for SimMIM
            if self.absolute_pos_embed.shape[1] > x.shape[1]:
                # Skip cls token position embedding
                x = x + self.absolute_pos_embed[:, 1:]
            else:
                x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # Reshape for reconstruction
        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class SwinTransformerV2ForSimMIM(SwinTransformerV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, masks=None, is_training=False):
        if is_training and masks is None:
            return super().forward(x, is_training=True)
        
        # Handle list input (multi-crop training)
        if isinstance(x, list):
            return super().forward_features(x, masks)
        
        # SimMIM reconstruction mode
        x = self.patch_embed(x)

        if masks is not None:
            B, L, _ = x.shape
            mask_tokens = self.mask_token.expand(B, L, -1)
            w = masks.flatten(1).unsqueeze(-1).type_as(mask_tokens)
            x = x * (1. - w) + mask_tokens * w

        if self.ape:
            # Use original position embedding without cls token for SimMIM
            if self.absolute_pos_embed.shape[1] > x.shape[1]:
                # Skip cls token position embedding
                x = x + self.absolute_pos_embed[:, 1:]
            else:
                x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class SimMIM(nn.Module):
    def __init__(self, config, encoder, encoder_stride, in_chans, patch_size):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = in_chans
        self.patch_size = patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        
        # norm target as prompted
        if self.config.NORM_TARGET.ENABLE:
            x = norm_targets(x, self.config.NORM_TARGET.PATCH_SIZE)
        
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}
