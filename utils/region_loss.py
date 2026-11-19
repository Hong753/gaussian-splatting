"""Optional region supervision, independent of alpha compositing and ray culling.

Masks select *loss pixels*. They never erase rendered geometry or change the
rasterizer. SSIM windows must fit inside the mask, preventing outside-image
supervision from leaking across the selected region boundary.
"""
import hashlib
from pathlib import Path


def mask_path(directory, name):
    return Path(directory) / (hashlib.sha256(name.encode('utf8')).hexdigest()[:24] + '.png')


def load_mask(directory, name, resolution):
    import numpy as np
    from PIL import Image
    import torch
    path = mask_path(directory, name)
    with Image.open(path) as image:
        array = np.asarray(image.convert('L').resize(resolution, Image.Resampling.NEAREST)).copy()
    mask = torch.from_numpy(array > 127).unsqueeze(0)
    if not mask.any():
        raise ValueError(f'Empty supervision mask: {path}')
    return mask


def _mask(image, mask):
    import torch
    if image.ndim != 3 or mask.shape != (1, *image.shape[-2:]):
        raise ValueError('Region loss requires CHW images and a 1HW mask')
    mask = mask.to(device=image.device)
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all() or not mask.any():
        raise ValueError('Region mask must be nonempty, finite and binary')
    return mask.to(image.dtype)


def photometric_terms(image, target, mask):
    import torch.nn.functional as F
    from utils.loss_utils import ssim
    mask = _mask(image, mask)
    l1 = ((image-target).abs()*mask).sum() / (mask.sum()*image.shape[0])
    # Include outside-image pixels as invalid support, too.
    interior = 1 - F.max_pool2d(F.pad(1-mask[None], (5,5,5,5), value=1), 11, stride=1)[0]
    if not interior.any():
        raise ValueError('Regional mask has no complete 11x11 SSIM window; enlarge the selected region')
    value = (ssim(image, target, return_map=True)*interior).sum() / (interior.sum()*image.shape[0])
    return l1, value


def score(image, target, mask=None):
    """Native per-channel PSNR, optionally normalized over region pixels only."""
    import torch
    if mask is None:
        mse = (image-target).square().flatten(1).mean(1)
    else:
        mask = _mask(image, mask)
        mse = ((image-target).square()*mask).flatten(1).sum(1) / mask.sum()
    return (-10*torch.log10(mse)).mean()


def display(image, mask):
    """Dim unsupervised pixels in previews; metrics always use unmodified RGB."""
    if mask is None:
        return image
    return image * (.2 + .8*mask.to(device=image.device, dtype=image.dtype))



def load_rgb_weight(directory, camera):
    """A native-resolution 8-bit map encodes 1x (black) to 2x (white) RGB supervision."""
    import numpy as np
    from PIL import Image
    import torch
    path = mask_path(directory, camera.image_name)
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (camera.image_width, camera.image_height):
            raise ValueError(f"Expected a native-resolution grayscale loss weight map at {path}")
        array = np.asarray(image).copy()
    weights = 1 + torch.from_numpy(array).float() / 255.
    return weights / weights.mean()


def weighted_l1(image, target, normalized_weight):
    """Use the whole image and preserve the mean scale of the usual RGB loss."""
    return ((image - target).abs() * normalized_weight.to(image.device)).mean()
