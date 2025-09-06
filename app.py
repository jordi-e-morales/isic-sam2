import os
import io
import json
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests

# Local modules
from sam2_infer import Sam2Wrapper, sam2_available
from metrics import compute_metrics
from viz import overlay_mask_with_contour


APP_TITLE = "ISIC Interactive Segmentation (SAM 2 Demo)"
SAMPLES_DIR = os.path.join("data", "samples")
DEFAULT_CKPT_PATH = os.path.join("models", "sam2_hiera_large.pt")


def ensure_dirs() -> None:
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("exports", exist_ok=True)


def list_sample_images() -> List[str]:
    if not os.path.isdir(SAMPLES_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(SAMPLES_DIR) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files


def download_checkpoint(url: str, dest_path: str) -> Tuple[bool, str]:
    """Download a checkpoint file from URL to dest_path. Returns (ok, message)."""
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            chunk = 1 << 20  # 1 MiB
            written = 0
            with open(dest_path, "wb") as f:
                for part in r.iter_content(chunk_size=chunk):
                    if not part:
                        continue
                    f.write(part)
                    written += len(part)
            # Basic size sanity check
            if total and written != total:
                return False, f"Downloaded size mismatch: {written} vs {total} bytes"
        return True, f"Downloaded to {dest_path}"
    except Exception as e:
        return False, f"Download failed: {e}"


def load_pil_image(uploaded_file, sample_file: Optional[str]) -> Optional[Image.Image]:
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            return img
        except Exception as e:
            st.error(f"Failed to open uploaded image: {e}")
            return None
    if sample_file:
        path = os.path.join(SAMPLES_DIR, sample_file)
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception as e:
            st.error(f"Failed to open sample image '{sample_file}': {e}")
            return None
    return None


def get_device_selection() -> str:
    choice = st.sidebar.selectbox("Device", ["Auto", "CPU", "CUDA"], index=0,
                                  help="Choose CUDA if you have a compatible GPU. Auto will try CUDA, then CPU.")
    torch_cuda = False
    try:
        import torch
        torch_cuda = torch.cuda.is_available()
    except Exception:
        torch_cuda = False

    if choice == "CUDA":
        if torch_cuda:
            st.sidebar.success("CUDA is available.")
            return "cuda"
        else:
            st.sidebar.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    elif choice == "CPU":
        return "cpu"
    else:
        # Auto
        if torch_cuda:
            st.sidebar.info("Auto: using CUDA")
            return "cuda"
        else:
            st.sidebar.info("Auto: using CPU")
            return "cpu"


def get_prompt_from_canvas(canvas_result, prompt_mode: str, image_size: Tuple[int, int]):
    """Parse the drawable canvas JSON to extract either a point (x, y) or a bounding box (x0,y0,x1,y1).
    Returns a dict with keys: {"point": (x,y)} or {"box": (x0,y0,x1,y1)} or None if not found.
    Coordinates are in image pixel space.
    """
    if canvas_result is None or canvas_result.json_data is None:
        return None
    data = canvas_result.json_data
    if "objects" not in data or not data["objects"]:
        return None
    W, H = image_size

    # We use the last object drawn
    obj = data["objects"][-1]
    obj_type = obj.get("type")
    # The canvas uses Fabric.js coordinates; handle circle and rect
    if prompt_mode == "Point" and obj_type == "circle":
        # Fabric circle stores center at (left + radius, top + radius)
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))
        radius = float(obj.get("radius", 0.0))
        x = left + radius
        y = top + radius
        # Clamp to bounds
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        return {"point": (x, y)}
    if prompt_mode == "Box" and obj_type == "rect":
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))
        width = float(obj.get("width", 0.0))
        height = float(obj.get("height", 0.0))
        x0, y0 = left, top
        x1, y1 = left + width, top + height
        # Normalize and clamp
        x0, x1 = sorted([max(0, min(W - 1, x0)), max(0, min(W - 1, x1))])
        y0, y1 = sorted([max(0, min(H - 1, y0)), max(0, min(H - 1, y1))])
        return {"box": (x0, y0, x1, y1)}
    return None


def to_bytes_png(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ensure_dirs()

    st.title(APP_TITLE)
    st.caption("For research/education only. Not a medical device or diagnostic tool.")

    # Sidebar controls
    device = get_device_selection()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model")
    ckpt_path = st.sidebar.text_input("SAM 2 checkpoint (.pt)", value=DEFAULT_CKPT_PATH,
                                      help="Path to SAM 2 model weights. Place your file under ./models/")
    weights_url = st.sidebar.text_input(
        "Weights URL (optional)", value="",
        help="Paste a direct download URL to a SAM 2 .pt checkpoint and click Download to save under ./models/",
    )
    if st.sidebar.button("Download weights to models/", disabled=(not weights_url.strip())):
        target = ckpt_path if ckpt_path.strip() else DEFAULT_CKPT_PATH
        with st.spinner("Downloading weights..."):
            ok, msg = download_checkpoint(weights_url.strip(), target)
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)
    use_sam2 = st.sidebar.checkbox("Use SAM 2 backend (if available)", value=True,
                                   help="If disabled or unavailable, a simple fallback segmentation will be used.")

    if use_sam2:
        if sam2_available():
            st.sidebar.success("SAM 2 Python package detected.")
        else:
            st.sidebar.warning("SAM 2 package not detected. Fallback segmentation will be used.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Image Source")
    sample_files = list_sample_images()
    sample_choice = None
    if sample_files:
        sample_choice = st.sidebar.selectbox("Pick a sample (./data/samples)", ["(none)"] + sample_files)
        if sample_choice == "(none)":
            sample_choice = None
    uploaded = st.sidebar.file_uploader("Or upload an image", type=["jpg", "jpeg", "png", "bmp"])

    image = load_pil_image(uploaded, sample_choice)
    if image is None:
        st.info("Upload an image or add some files to ./data/samples to get started.")
        return

    # Prompt mode
    prompt_mode = st.radio("Prompt Type", ["Point", "Box"], horizontal=True)

    # Display canvas with the background image (with robust fallback)
    W, H = image.size
    st.write("Draw a small circle for a point, or a rectangle for a box prompt.")
    canvas_result = None
    canvas_error = None
    try:
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",  # fill for circles/rectangles
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=image,
            update_streamlit=True,
            height=H,
            width=W,
            drawing_mode="circle" if prompt_mode == "Point" else "rect",
            key="canvas",
        )
    except Exception as e:
        canvas_error = str(e)

    prompt = get_prompt_from_canvas(canvas_result, prompt_mode, (W, H)) if canvas_result is not None else None

    prompt_fallback = None
    if canvas_error is not None:
        st.warning(
            "Interactive canvas unavailable (likely due to Streamlit/plugin version mismatch). "
            "Using fallback numeric inputs for the prompt."
        )
        if prompt_mode == "Point":
            c1, c2 = st.columns(2)
            with c1:
                px = st.slider("Point X", min_value=0, max_value=W - 1, value=int(W // 2), step=1)
            with c2:
                py = st.slider("Point Y", min_value=0, max_value=H - 1, value=int(H // 2), step=1)
            prompt_fallback = {"point": (float(px), float(py))}
        else:
            c1, c2 = st.columns(2)
            with c1:
                x0 = st.slider("Box X0", min_value=0, max_value=W - 2, value=int(W * 0.25), step=1)
                x1 = st.slider("Box X1", min_value=x0 + 1, max_value=W - 1, value=int(W * 0.75), step=1)
            with c2:
                y0 = st.slider("Box Y0", min_value=0, max_value=H - 2, value=int(H * 0.25), step=1)
                y1 = st.slider("Box Y1", min_value=y0 + 1, max_value=H - 1, value=int(H * 0.75), step=1)
            prompt_fallback = {"box": (float(x0), float(y0), float(x1), float(y1))}

    # If canvas didn't produce a prompt, try fallback prompt
    if prompt is None and prompt_fallback is not None:
        prompt = prompt_fallback

    run = st.button("Run Segmentation", type="primary")

    if run:
        if prompt is None:
            st.warning("Please draw a point (circle) or a box on the image.")
            st.stop()

        # Prepare SAM2 wrapper (lazy init)
        sam2 = Sam2Wrapper(device=device, ckpt_path=ckpt_path if use_sam2 else None)

        np_img = np.array(image)
        mask = None
        used_backend = "fallback"

        if use_sam2 and sam2.ready:
            try:
                if "point" in prompt:
                    px, py = prompt["point"]
                    mask = sam2.segment(np_img, point=(px, py))
                else:
                    box = prompt["box"]
                    mask = sam2.segment(np_img, box=box)
                used_backend = "sam2"
            except Exception as e:
                st.error(f"SAM 2 inference failed: {e}. Using fallback segmentation.")

        if mask is None:
            # Fallback segmentation
            if "point" in prompt:
                px, py = prompt["point"]
                mask = Sam2Wrapper.segment_fallback(np_img, point=(px, py))
            else:
                mask = Sam2Wrapper.segment_fallback(np_img, box=prompt["box"]) 

        if mask is None:
            st.error("Segmentation failed.")
            st.stop()

        # Visualize overlay and contour
        overlay = overlay_mask_with_contour(image, mask, mask_color=(0, 255, 0), contour_color=(255, 0, 0), alpha=0.35)
        st.subheader("Segmentation Result")
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            st.image(overlay, caption=f"Overlay ({used_backend})", use_column_width=True)
        with col2:
            st.markdown("#### Metrics (ABCD-style)")
            metrics = compute_metrics(mask, np_img)
            st.json(metrics)

            # Export buttons
            st.markdown("#### Export")
            # Mask PNG (binary 0/255)
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            st.download_button(
                label="Download Mask (PNG)",
                data=to_bytes_png(mask_img),
                file_name="mask.png",
                mime="image/png",
            )
            st.download_button(
                label="Download Overlay (PNG)",
                data=to_bytes_png(overlay),
                file_name="overlay.png",
                mime="image/png",
            )
            st.download_button(
                label="Download Metrics (JSON)",
                data=json.dumps(metrics, indent=2).encode("utf-8"),
                file_name="metrics.json",
                mime="application/json",
            )

    with st.expander("About"):
        st.markdown(
            """
            This app demonstrates interactive lesion segmentation with an emphasis on border quality and basic ABCD-style metrics. 
            If SAM 2 is not installed or a checkpoint is not provided, a simple classical fallback segmentation is used for demonstration.
            """
        )


if __name__ == "__main__":
    main()
