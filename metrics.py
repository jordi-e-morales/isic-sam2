import math
from typing import Dict, Any

import numpy as np
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans


def _safe_perimeter(region) -> float:
    # Prefer Crofton perimeter if available for better estimation
    p = getattr(region, "perimeter_crofton", None)
    if p is not None:
        return float(p)
    return float(region.perimeter)


def _compute_basic_shape(mask: np.ndarray) -> Dict[str, Any]:
    h, w = mask.shape
    lab = label(mask.astype(bool))
    if lab.max() == 0:
        return {
            "area_px": 0,
            "area_fraction": 0.0,
            "perimeter_px": 0.0,
            "equiv_diameter_px": 0.0,
            "circularity": 0.0,
            "border_irregularity": 0.0,
            "bbox": None,
            "centroid_xy": None,
            "image_size": [int(w), int(h)],
        }

    # Use largest component
    regions = regionprops(lab)
    regions.sort(key=lambda r: r.area, reverse=True)
    r = regions[0]

    A = float(r.area)
    P = _safe_perimeter(r)
    area_fraction = A / float(h * w)

    # Shape metrics
    equiv_diameter = math.sqrt(4.0 * A / math.pi) if A > 0 else 0.0
    circularity = float((4.0 * math.pi * A) / (P * P)) if P > 0 else 0.0
    # Border irregularity as inverse of circularity
    border_irregularity = float((P * P) / (4.0 * math.pi * A)) if A > 0 else 0.0

    minr, minc, maxr, maxc = r.bbox
    bbox = [int(minc), int(minr), int(maxc), int(maxr)]  # x0,y0,x1,y1
    cy, cx = r.centroid  # (row, col)

    return {
        "area_px": int(A),
        "area_fraction": float(area_fraction),
        "perimeter_px": float(P),
        "equiv_diameter_px": float(equiv_diameter),
        "circularity": float(np.clip(circularity, 0.0, 1.0)),
        "border_irregularity": float(border_irregularity),
        "bbox": bbox,
        "centroid_xy": [float(cx), float(cy)],
        "image_size": [int(w), int(h)],
    }


def _compute_asymmetry(mask: np.ndarray, cx: float, cy: float) -> Dict[str, float]:
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]

    left = mask & (xx < cx)
    right = mask & (xx >= cx)
    top = mask & (yy < cy)
    bottom = mask & (yy >= cy)

    A = float(mask.sum())
    if A <= 0:
        return {"asymmetry_horizontal": 0.0, "asymmetry_vertical": 0.0, "asymmetry_score": 0.0}

    asym_h = float(abs(left.sum() - right.sum())) / A
    asym_v = float(abs(top.sum() - bottom.sum())) / A
    asym_score = float(max(asym_h, asym_v))
    return {
        "asymmetry_horizontal": float(np.clip(asym_h, 0.0, 1.0)),
        "asymmetry_vertical": float(np.clip(asym_v, 0.0, 1.0)),
        "asymmetry_score": float(np.clip(asym_score, 0.0, 1.0)),
    }


def _compute_color_variegation(mask: np.ndarray, image_rgb: np.ndarray, k: int = 3) -> Dict[str, float]:
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        return {"color_variegation": 0.0}

    pixels = image_rgb[mask.astype(bool)]
    if pixels.size < max(50, k * 10):
        return {"color_variegation": 0.0}

    X = pixels.astype(np.float32) / 255.0
    try:
        km = KMeans(n_clusters=k, n_init=5, random_state=0)
        labels = km.fit_predict(X)
        centers = km.cluster_centers_
        # Mean distance of each pixel to its cluster center
        dists = np.linalg.norm(X - centers[labels], axis=1)
        # Normalize: distances in RGB are in [0, sqrt(3)]; scale to ~[0,1]
        score = float(np.mean(dists) / math.sqrt(3))
        score = float(np.clip(score, 0.0, 1.0))
        return {"color_variegation": score}
    except Exception:
        return {"color_variegation": 0.0}


def compute_metrics(mask: np.ndarray, image_rgb: np.ndarray) -> Dict[str, Any]:
    """Compute ABCD-style metrics for a segmentation mask and its image.

    Returns a JSON-serializable dict.
    """
    mask = mask.astype(bool)
    shape = _compute_basic_shape(mask)

    # Asymmetry
    if shape["centroid_xy"] is not None:
        cx, cy = shape["centroid_xy"]
        asym = _compute_asymmetry(mask, cx=cx, cy=cy)
    else:
        asym = {"asymmetry_horizontal": 0.0, "asymmetry_vertical": 0.0, "asymmetry_score": 0.0}

    # Color variegation
    color = _compute_color_variegation(mask, image_rgb)

    return {
        **shape,
        **asym,
        **color,
    }
