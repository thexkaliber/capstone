# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from .vim import VisionMamba


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
            
    if "vim" in args.arch:
        vim_kwargs = dict(
            img_size=args.img_size,
            pretrained=False,
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            stride=args.stride,
            embed_dim=args.embed_dim,
            depth=args.depth,
            rms_norm=args.rms_norm,
            residual_in_fp32=args.residual_in_fp32,
            fused_add_norm=args.fused_add_norm,
            if_abs_pos_embed=args.if_abs_pos_embed,
            if_cls_token=args.if_cls_token,
            use_middle_cls_token=args.use_middle_cls_token,
            return_features=True,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path_rate,
        )
        teacher = VisionMamba(**vim_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = VisionMamba(**vim_kwargs, drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
        return student, teacher, embed_dim
    else:
        raise NotImplementedError(f"Model architecture {args.arch} is not supported.")
    

def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
