import torch
from torch.nn import functional as nnF

def conv_op(images: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Applies 2D correlation with reflective padding.
    
    This operation preserves spatial dimensions. The kernel is applied with
    reflective boundary conditions to avoid edge artifacts.

    Args:
        images (torch.Tensor): Tensor of shape (NB, Nc, Ny, Nx).
        kernel (torch.Tensor): Tensor of shape (no, ni, ny, nx).
            `ni` must be either 1 or `Nc`.

    Raises:
        ValueError: If `images.ndim` != 4.
        ValueError: If `kernel.ndim` != 4.
        ValueError: If `ni` not in {1, Nc}.

    Returns:
        torch.Tensor: Tensor of shape (NB, no, Ny, Nx) when `ni` = `Nc`, else
        (NB, no*Nc, Ny, Nx) when `ni` = 1.
    """
    if images.ndim != 4:
        raise ValueError(
            f"images must have 4 dimensions instead of {images.ndim}"
        )
    if kernel.ndim != 4:
        raise ValueError(
            f"kernel must have 4 dimensions instead of {kernel.ndim}"
        )
    if kernel.shape[1] not in [1, images.shape[1]]:
        raise ValueError(
            f"{kernel.shape[1]} must only be either 1 or equal to "
            f"{images.shape[1]}"
        )
    shape = images.shape
    same_filter_for_multi_colors = kernel.shape[1] == 1 and shape[1] > 1
    if same_filter_for_multi_colors:
        # Our special situation where `images` is a color image (`Nc` > 1) and
        # only one kernel is provided (`ni` = 1), move all the "channels" into
        #  the "batches" so that the same filter is applied to all the 
        # "batches".
        images = images.reshape(-1, 1, *shape[2:])
    ny, nx = kernel.shape[2:]
    pad = (ny//2, ny//2, nx//2, nx//2)
    images = nnF.pad(images, pad, mode='reflect')
    out = nnF.conv2d(images, kernel)
    if same_filter_for_multi_colors:
        # Handling the output for our special situation. In this case, the
        # output would have a shape of (NBNc, no, Ny, Nx). We unflatten it
        # first to (NB, Nc, no, Ny, Nx), then permute it to (NB, no, Nc,
        # Ny, Nx) and then flatten the second and third dimensions to obtain
        # an image with shape (NB, noNc, Ny, Nx). 
        out = out.unflatten(0, shape[:2]).permute(0, 2, 1, 3, 4).flatten(1, 2)
    return out