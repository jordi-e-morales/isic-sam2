from typing import Tuple

import numpy as np
from PIL import Image
from skimage.morphology import binary_erosion, disk


def overlay_mask_with_contour(
    image_pil: Image.Image,
    mask: np.ndarray,
    mask_color: Tuple[int, int, int] = (0, 255, 0),
    contour_color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> Image.Image:
    """Overlay a binary mask on top of an RGB PIL image with semi-transparent color and draw contour.

    Args:
        image_pil: Input RGB image (PIL Image).
        mask: Boolean array (H, W) or uint8 (0/1) mask.
        mask_color: RGB color for filled region.
        contour_color: RGB color for the contour.
        alpha: Blending factor for mask fill.

    Returns:
        PIL Image with overlay.
    """
    assert image_pil.mode == "RGB", "image_pil must be RGB"

    img = np.array(image_pil).astype(np.float32)
    H, W = img.shape[:2]

    m = mask.astype(bool)
    if m.shape[0] != H or m.shape[1] != W:
        # Resize mask to image size via nearest
        from PIL import Image as PILImage
        m_img = PILImage.fromarray((m.astype(np.uint8) * 255))
        m_img = m_img.resize((W, H))
        m = np.array(m_img) > 127

    # Blend mask color
    fill = np.zeros_like(img)
    fill[:, :, 0] = mask_color[0]
    fill[:, :, 1] = mask_color[1]
    fill[:, :, 2] = mask_color[2]

    out = img.copy()
    out[m] = (alpha * fill[m] + (1.0 - alpha) * img[m])

    # Contour via morphological erosion difference
    try:
        edge = m & (~binary_erosion(m, footprint=disk(1)))
    except Exception:
        # Fallback: 4-neighborhood erosion using a simple kernel
        from scipy.ndimage import binary_erosion as be
        edge = m & (~be(m))

    # Draw contour as solid color (no alpha)
    rr, cc = np.where(edge)
    out[rr, cc, 0] = contour_color[0]
    out[rr, cc, 1] = contour_color[1]
    out[rr, cc, 2] = contour_color[2]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")
