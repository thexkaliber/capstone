import logging
import os
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union, List

import h5py
import numpy as np
import torch
from PIL import Image as im
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
from datasets import load_from_disk

logger = logging.getLogger("astrodino")
_Target = float


class _SplitFull(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _SplitFull.TRAIN: 74_500_000,
            _SplitFull.VAL: 100_000,
            _SplitFull.TEST: 400_000,
        }
        return split_lengths[self]


def _open_h5_files(prefix: str, root: str, num: int) -> List[h5py.File]:
    """Open all available h5 files and skip missing ones."""
    files = []
    for i in range(num):
        file_path = os.path.join(
            root, f"{prefix}/images_npix152_0{i:02d}000000_0{i+1:02d}000000.h5"
        )
        try:
            f = h5py.File(file_path, "r")
            files.append(f)
        except (OSError, FileNotFoundError) as e:
            logger.warning(f"Could not open file {file_path}: {e}")
    return files


class LegacySurvey(VisionDataset):
    Target = Union[_Target]
    Split = Union[_SplitFull]

    def __init__(
        self,
        *,
        split: "LegacySurvey.Split",
        root: str,
        extra: str = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        # Open available hdf5 files in north and south
        self._files = _open_h5_files("north", root, 14)
        self._files += _open_h5_files("south", root, 61)

        # Track the number of images in each file for indexing
        self._file_lengths = [f["images"].shape[0] for f in self._files]
        self._cumulative_lengths = np.cumsum([0] + self._file_lengths)

        total_images = self._cumulative_lengths[-1]

        # Create randomized array of indices
        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(total_images)

        # Define split sizes based on available data
        train_len = int(total_images * 0.9933)
        val_len = int(total_images * 0.00133)
        test_len = total_images - train_len - val_len

        if split == _SplitFull.TRAIN:
            self._indices = indices[:train_len]
        elif split == _SplitFull.VAL:
            self._indices = indices[train_len:train_len + val_len]
        else:
            self._indices = indices[-test_len:]

    @property
    def split(self) -> "LegacySurvey.Split":
        return self._split

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        true_index = self._indices[index]
        file_idx = np.searchsorted(self._cumulative_lengths, true_index, side="right") - 1
        local_idx = true_index - self._cumulative_lengths[file_idx]
        image = self._files[file_idx]["images"][local_idx].astype("float32")
        image = torch.tensor(image)
        target = None

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._indices)


class _SplitNorth(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _SplitNorth.TRAIN: 13_500_000,
            _SplitNorth.VAL: 100_000,
            _SplitNorth.TEST: 400_000,
        }
        return split_lengths[self]


class LegacySurveyNorth(VisionDataset):
    Target = Union[_Target]
    Split = Union[_SplitNorth]

    def __init__(
        self,
        *,
        split: "LegacySurveyNorth.Split",
        root: str,
        extra: str = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        # Only open files in north
        self._files = _open_h5_files("north", root, 14)

        # Track the number of images in each file for indexing
        self._file_lengths = [f["images"].shape[0] for f in self._files]
        self._cumulative_lengths = np.cumsum([0] + self._file_lengths)

        total_images = self._cumulative_lengths[-1]

        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(total_images)

        train_len = int(total_images * 0.964)
        val_len = int(total_images * 0.0071)
        test_len = total_images - train_len - val_len

        if split == _SplitNorth.TRAIN:
            self._indices = indices[:train_len]
        elif split == _SplitNorth.VAL:
            self._indices = indices[train_len:train_len + val_len]
        else:
            self._indices = indices[-test_len:]

    @property
    def split(self) -> "LegacySurveyNorth.Split":
        return self._split

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        true_index = self._indices[index]
        file_idx = np.searchsorted(self._cumulative_lengths, true_index, side="right") - 1
        local_idx = true_index - self._cumulative_lengths[file_idx]
        image = self._files[file_idx]["images"][local_idx].astype("float32")
        image = torch.tensor(image)
        target = None

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._indices)


class AstroClip(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        ds = load_from_disk(root)
        # If split is not present, raise a clear error
        if split not in ds:
            raise ValueError(f"Split '{split}' not found in dataset at {root}. Available splits: {list(ds.keys())}")
        self.dataset = ds[split]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = torch.tensor(item["image"], dtype=torch.float32) / 255.0

        if image.ndim != 3:
            raise ValueError(f"Expected 3D image tensor, got shape {image.shape}")

        # If the middle dimension is 3, probably (H, C, W) -> transpose
        if image.shape[1] == 3 and image.shape[0] != 3:
            image = image.permute(1, 0, 2)
        # If last dimension is 3, probably (H, W, C)
        elif image.shape[2] == 3 and image.shape[0] != 3:
            image = image.permute(2, 0, 1)

        if image.shape[0] != 3:
            raise ValueError(f"Expected 3 channels at dim 0, got shape {image.shape}")

        target = None

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        return image, target