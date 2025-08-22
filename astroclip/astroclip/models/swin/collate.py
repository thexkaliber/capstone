# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for Swin Transformer hierarchical patch structure
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


def collate_data_and_cast_swin(samples_list, mask_ratio_tuple, mask_probability, dtype, 
                               n_tokens=None, mask_generator=None, 
                               patch_size=4, depths=[2, 2, 6, 2]):
    """
    Swin Transformer specific collate function that handles hierarchical patch structure.
    Automatically handles both fixed and variable image sizes.

    Dictionary with collated data and corrected upperbound calculation
    """
    
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    
    # Calculate Swin patch counts for each crop (handles variable sizes automatically)
    def swin_patch_count(img_size):
        # Start with initial patches, apply merging (2x reduction per stage except first)
        patches_per_dim = img_size // patch_size
        for _ in range(len(depths) - 1):  # Number of merging stages
            patches_per_dim = patches_per_dim // 2
        return patches_per_dim ** 2
    
    # Get patch count for each crop
    patch_counts = [swin_patch_count(crop.shape[-1]) for crop in collated_global_crops]
    
    upperbound = 0
    masks_list = []
    
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        
        N_i = patch_counts[i]
        num_masked_patches = int(N_i * random.uniform(prob_min, prob_max))
        masks_list.append(torch.BoolTensor(mask_generator(num_masked_patches)))
        upperbound += int(N_i * prob_max)
    
    # No masking for remaining samples
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
