import torch
import numpy as np
import sys
from torch import Tensor
from PIL.Image import Image as PILImage
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

def to_binary_tensor(pic: Union[PILImage, np.ndarray]) -> Tensor:

    mode_to_nptype = {"I": np.int32, "I;16" if sys.byteorder == "little" else "I;16B": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))

    # PIL image is (W, H, C)
    if pic.mode == "L":
        img = img.view(pic.size[1], pic.size[0], 1)
    elif pic.mode == "RGB":
        img = img.view(pic.size[1], pic.size[0], 3)
    
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()

    if isinstance(img, torch.ByteTensor):
        img = img.to(dtype=torch.uint8)

    num_channels = img.shape[0]
    height, width = img.shape[1], img.shape[2]
    binary_tensor = torch.zeros((num_channels*8, height, width), dtype=torch.float32)

    for ch in range(num_channels):
        for i in range(8):
            bit_mask = 1 << (7 - i)
            binary_tensor[ch*8+i, :, :] = (img[ch] & bit_mask) >> (7 - i)
    
    return binary_tensor

class ToBinaryTensor:
    """Convert a PIL Image or ndarray to binary tensor.

    This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x 8 x H x W)
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_binary_tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"