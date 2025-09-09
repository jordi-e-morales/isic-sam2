import os
import re
import logging
from typing import Optional, Tuple

import numpy as np

# Optional deps for fallback segmentation
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, remove_small_holes, remove_small_objects, disk
from skimage.segmentation import flood
from skimage.measure import label, regionprops


log = logging.getLogger(__name__)


def sam2_available() -> bool:
    try:
        import sam2  # type: ignore
        return True
    except Exception:
        return False


def _infer_model_type_from_ckpt(ckpt_path: str) -> str:
    """Best-effort guess of SAM 2 model_type from checkpoint filename."""
    name = os.path.basename(ckpt_path).lower()
    table = {
        "tiny": "sam2_hiera_t",
        "small": "sam2_hiera_s",
        "base": "sam2_hiera_b",
        "large": "sam2_hiera_l",
        "xlarge": "sam2_hiera_xl",
    }
    for k, v in table.items():
        if k in name:
            return v
    # default
    return "sam2_hiera_l"


class Sam2Wrapper:
    """Thin wrapper around SAM 2 predictor with a robust CPU fallback segmentation.

    Usage:
        sam = Sam2Wrapper(device="cuda", ckpt_path="models/sam2_hiera_large.pt")
        mask = sam.segment(image_rgb_uint8, point=(x, y))  # or box=(x0,y0,x1,y1)
    """

    def __init__(self, device: str = "cpu", ckpt_path: Optional[str] = None, model_type: Optional[str] = None):
        self.device = device
        self.ckpt_path = ckpt_path
        self.model_type = model_type
        self.model = None
        self.predictor = None
        self.ready = False

        if ckpt_path is None:
            log.info("SAM2 checkpoint not provided; wrapper will use fallback only.")
            return

        if not sam2_available():
            log.warning("SAM2 package not available; using fallback.")
            return

        if not os.path.isfile(ckpt_path):
            log.warning("SAM2 checkpoint not found at %s; using fallback.", ckpt_path)
            return

        # Attempt to build model and predictor (supports minor API changes)
        try:
            # Newer API
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore


            cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # package-relative path for Hydra
            self.model = build_sam2(cfg, ckpt_path, device=self.device)
            self.predictor = SAM2ImagePredictor(self.model)
            self.ready = True
            log.info("Initialized SAM2 on device=%s", self.device)
            return
        except Exception as e:
            log.warning("SAM2 new API load failed: %s", e)

        try:
            # Fallback: try original SAM-style registry (unlikely for SAM2, but safe)
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore

            model_type = (self.model_type or "vit_h").lower()
            self.model = sam_model_registry[model_type](checkpoint=self.ckpt_path)
            if self.device == "cuda":
                import torch
                self.model.to(device=self.device)
            self.predictor = SamPredictor(self.model)
            self.ready = True
            log.info("Initialized SAM (fallback) model_type=%s on device=%s", model_type, self.device)
        except Exception as e:
            log.error("Failed to initialize any SAM backend: %s", e)
            self.model = None
            self.predictor = None
            self.ready = False

    # ------------------------
    # Inference
    # ------------------------
    def segment(self, image_rgb: np.ndarray, point: Optional[Tuple[float, float]] = None,
                box: Optional[Tuple[float, float, float, float]] = None) -> Optional[np.ndarray]:
        """Run segmentation on an RGB uint8 image using SAM2/SAM. Returns a boolean mask (H, W)."""
        if image_rgb is None:
            return None
        if not self.ready or self.predictor is None:
            raise RuntimeError("SAM backend not initialized. Use segment_fallback instead or check ckpt and install.")

        h, w = image_rgb.shape[:2]

        # Ensure uint8
        if image_rgb.dtype != np.uint8:
            img = image_rgb.astype(np.uint8)
        else:
            img = image_rgb

        # Set image once per inference
        try:
            self.predictor.set_image(img)
        except Exception as e:
            log.error("Predictor.set_image failed: %s", e)
            raise

        masks = None
        scores = None

        # Try SAM2 signature
        try:
            if point is not None:
                point_coords = np.array([[float(point[0]), float(point[1])]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)
                out = self.predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                             box=None, multimask_output=False)
            elif box is not None:
                # SAM expects [x0, y0, x1, y1]
                box_arr = np.array([float(box[0]), float(box[1]), float(box[2]), float(box[3])], dtype=np.float32)
                out = self.predictor.predict(point_coords=None, point_labels=None,
                                             box=box_arr, multimask_output=False)
            else:
                raise ValueError("Either point or box must be provided.")

            # Possible outputs: (masks, scores, logits) or dict
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                masks, scores = out[0], out[1]
            elif isinstance(out, dict) and "masks" in out:
                masks, scores = out.get("masks"), out.get("scores")
            else:
                raise RuntimeError("Unexpected predictor output type.")
        except Exception as e:
            # Try SAM (v1) API as a fallback
            log.warning("SAM2 predict failed, trying SAM v1 API: %s", e)
            try:
                if point is not None:
                    point_coords = np.array([[float(point[0]), float(point[1])]], dtype=np.float32)
                    point_labels = np.array([1], dtype=np.int32)
                    masks, scores, _ = self.predictor.predict(
                        point_coords=point_coords, point_labels=point_labels, multimask_output=False
                    )
                elif box is not None:
                    box_arr = np.array([float(box[0]), float(box[1]), float(box[2]), float(box[3])], dtype=np.float32)
                    masks, scores, _ = self.predictor.predict(
                        box=box_arr, multimask_output=False
                    )
                else:
                    raise ValueError("Either point or box must be provided.")
            except Exception as e2:
                log.error("All SAM predict attempts failed: %s", e2)
                return None

        # masks -> select best
        if masks is None:
            return None
        arr = np.array(masks)
        if arr.ndim == 2:
            m = arr.astype(bool)
        elif arr.ndim == 3:
            # choose highest score if available
            idx = 0
            if scores is not None:
                try:
                    idx = int(np.argmax(np.array(scores)))
                except Exception:
                    idx = 0
            m = arr[idx].astype(bool)
        else:
            return None

        # Ensure size
        if m.shape[0] != h or m.shape[1] != w:
            # simple resize via nearest (avoid skimage dependency here)
            from PIL import Image as PILImage
            m_img = PILImage.fromarray(m.astype(np.uint8) * 255)
            m_img = m_img.resize((w, h))
            m = (np.array(m_img) > 127)

        # Post-process: largest component
        m = Sam2Wrapper._largest_component(m)
        return m

    # ------------------------
    # Classical fallback
    # ------------------------
    @staticmethod
    def segment_fallback(image_rgb: np.ndarray, point: Optional[Tuple[float, float]] = None,
                          box: Optional[Tuple[float, float, float, float]] = None) -> Optional[np.ndarray]:
        if image_rgb is None:
            return None
        img = image_rgb
        if img.ndim == 2:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = rgb2gray(img)
        h, w = gray.shape

        mask = None
        if point is not None:
            px, py = int(round(point[0])), int(round(point[1]))
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            # Estimate local tolerance from 15x15 patch
            r = 7
            y0, y1 = max(0, py - r), min(h, py + r + 1)
            x0, x1 = max(0, px - r), min(w, px + r + 1)
            patch = gray[y0:y1, x0:x1]
            tol = float(np.std(patch) * 2.0 + 0.05)
            tol = max(0.02, min(0.25, tol))
            try:
                m = flood(gray, (py, px), tolerance=tol)
            except Exception:
                # fallback: global threshold around patch median
                med = float(np.median(patch))
                thr = med
                m = gray < thr if gray[py, px] < med else gray > thr
            mask = m
        elif box is not None:
            x0, y0, x1, y1 = box
            x0 = int(max(0, min(w - 1, round(x0))))
            x1 = int(max(0, min(w - 1, round(x1))))
            y0 = int(max(0, min(h - 1, round(y0))))
            y1 = int(max(0, min(h - 1, round(y1))))
            if x1 <= x0 or y1 <= y0:
                return None
            roi = gray[y0:y1, x0:x1]
            try:
                thr = threshold_otsu(roi)
                roi_mask = roi < thr if np.mean(roi) > thr else roi > thr
            except Exception:
                thr = np.mean(roi)
                roi_mask = roi > thr
            mask = np.zeros_like(gray, dtype=bool)
            mask[y0:y1, x0:x1] = roi_mask
        else:
            return None

        # Morphology cleanup
        mask = binary_closing(mask, footprint=disk(3))
        mask = remove_small_holes(mask, area_threshold=128)
        mask = remove_small_objects(mask, min_size=128)

        # Largest component
        mask = Sam2Wrapper._largest_component(mask)
        return mask

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        if mask is None:
            return mask
        lab = label(mask.astype(bool))
        if lab.max() == 0:
            return mask.astype(bool)
        regions = regionprops(lab)
        regions.sort(key=lambda r: r.area, reverse=True)
        keep = regions[0].label
        return (lab == keep)

