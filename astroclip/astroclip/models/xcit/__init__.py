# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from .xcit import XCiT

logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    
    if "xcit" in args.arch:
        xcit_kwargs = dict(
            img_size=args.img_size,
            num_classes=0,  
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            qkv_bias=True,
            eta=args.eta,
            cls_attn_layers=args.cls_attn_layers,
            use_pos=args.use_pos,
            tokens_norm=args.tokens_norm,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path_rate,
        )
        teacher = XCiT(**xcit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        
        student = XCiT(
            **xcit_kwargs,
        )
        embed_dim = student.embed_dim

        return student, teacher, embed_dim
    else:
        logger.warning(f"Model architecture {args.arch} is not supported.")
        
    
    # if "vit" in args.arch:
    #     vit_kwargs = dict(
    #         img_size=img_size,
    #         patch_size=args.patch_size,
    #         init_values=args.layerscale,
    #         ffn_layer=args.ffn_layer,
    #         block_chunks=args.block_chunks,
    #         qkv_bias=args.qkv_bias,
    #         proj_bias=args.proj_bias,
    #         ffn_bias=args.ffn_bias,
    #         num_register_tokens=args.num_register_tokens,
    #         interpolate_offset=args.interpolate_offset,
    #         interpolate_antialias=args.interpolate_antialias,
    #     )
    #     teacher = vits.__dict__[args.arch](**vit_kwargs)
    #     if only_teacher:
    #         return teacher, teacher.embed_dim
    #     student = vits.__dict__[args.arch](
    #         **vit_kwargs,
    #         drop_path_rate=args.drop_path_rate,
    #         drop_path_uniform=args.drop_path_uniform,
    #     )
    #     embed_dim = student.embed_dim
    # return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
