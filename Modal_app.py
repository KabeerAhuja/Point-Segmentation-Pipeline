"""
BoastIQ Player Tracker — Modal Web Application (SAM2 + Robust Handoff)

Deploy:  modal deploy modal_app.py
Run dev: modal serve modal_app.py

Architecture:
  Browser  ←WebSocket→  FastAPI (CPU)  ←.remote()→  GPU (A100-80GB, SAM2)
                              ↕                           ↕
                        Firebase Storage            Modal Volume (scratch)
"""

import modal
import os
import json
import uuid
import base64
import math
import time
import re
import asyncio
from pathlib import Path

# ====================================================================
# Modal App Setup
# ====================================================================

app = modal.App("boastiq-tracker")

scratch_vol = modal.Volume.from_name("boastiq-scratch", create_if_missing=True)
model_vol = modal.Volume.from_name("boastiq-models", create_if_missing=True)
ball_weights_vol = modal.Volume.from_name("ball-inferencing-weights")

gcs_secret = modal.Secret.from_name("googlecloud-secret")
hf_secret = modal.Secret.from_name("huggingface-secret")
gemini_secret = modal.Secret.from_name("Gemini-API-Key")

DATA_DIR = "/data"
MODEL_CACHE = "/models"
SAM2_DIR = "/opt/sam2"

# ====================================================================
# Modal Images
# ====================================================================

sam2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "ninja-build")
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "opencv-contrib-python-headless", "numpy", "Pillow",
        "google-cloud-storage",
        "huggingface-hub",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/sam2.git /opt/sam2",
        "cd /opt/sam2 && pip install -e '.[notebooks]'",
        # Build SAM2 CUDA extensions (_C module) — nvcc available from devel image
        "cd /opt/sam2 && python setup.py build_ext --inplace",
    )
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "fastapi[standard]", "uvicorn", "websockets", "python-multipart",
        "google-cloud-storage",
        "opencv-contrib-python-headless", "numpy", "Pillow",
    )
    .add_local_file(
        local_path=str(Path(__file__).parent / "index.html"),
        remote_path="/app/index.html",
    )
)

gemini_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "google-genai>=1.0.0",
        "google-cloud-storage",
        "opencv-contrib-python-headless", "numpy", "Pillow",
    )
    .add_local_file(
        local_path=str(Path(__file__).parent / "squash_court_lines.jpeg"),
        remote_path="/app/squash_court_lines.jpeg",
    )
)

yolo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "ultralytics",
        "opencv-python-headless",
        "google-cloud-storage",
        "numpy",
    )
)

BALL_WEIGHTS_DIR = "/ball_weights"

# ====================================================================
# Shared Configuration
# ====================================================================

DEFAULT_CONFIG = {
    "frames_per_segment": 1500,
    "overlap_frames": 150,           # 15s at 10fps — wide handoff window
    "target_fps": 10,
    "target_height": 480,
    "min_player_separation": 120,
    "min_mask_area_px": 500,
    "max_mask_area_px": 60000,
    "handoff_search_zone": 600,      # Score last 600 frames for best handoff
    "handoff_prompt_retries": 5,     # Retry prompt on different frames
    "handoff_multi_points": 5,       # Interior mask points for prompt
    "handoff_min_mask_iou": 0.10,    # Min IoU for validation
}


def compute_proc_params(src_width, src_height, src_fps, config=None):
    """Compute processing parameters from source video properties."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    proc_h = cfg["target_height"]
    proc_w = int(round(src_width * proc_h / src_height / 2) * 2)
    scale_x = src_width / proc_w
    scale_y = src_height / proc_h
    frame_step = src_fps / cfg["target_fps"]
    area_scale = (proc_h / src_height) ** 2

    return {
        **cfg,
        "src_width": src_width,
        "src_height": src_height,
        "src_fps": src_fps,
        "proc_width": proc_w,
        "proc_height": proc_h,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "frame_step": frame_step,
        "min_mask_area_proc": max(100, int(cfg["min_mask_area_px"] * area_scale)),
        "max_mask_area_proc": int(cfg["max_mask_area_px"] * area_scale),
        "min_separation_proc": int(cfg["min_player_separation"] / scale_x),
    }


# ====================================================================
# Google Cloud Storage Helpers
# ====================================================================

STORAGE_BUCKET = "boastiq.firebasestorage.app"
UPLOAD_PREFIX = "uploaded-videos"
OUTPUT_PREFIX = "video-data"


def sanitize_filename(name: str) -> str:
    """Sanitize a filename to lowercase + underscores, no extension."""
    stem = Path(name).stem  # strip extension
    clean = stem.lower()
    clean = re.sub(r'[^a-z0-9]+', '_', clean)  # replace non-alphanumeric with _
    clean = clean.strip('_')
    return clean or "video"


def make_video_key(original_filename: str) -> str:
    """Create a unique video key: sanitized_name_uuid_timestamp."""
    from datetime import datetime, timezone
    sanitized = sanitize_filename(original_filename)
    short_id = str(uuid.uuid4())[:8]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{sanitized}_{short_id}_{ts}"


def _get_gcs_bucket():
    from google.cloud import storage
    from google.oauth2 import service_account
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    client = storage.Client(credentials=credentials)
    return client.bucket(STORAGE_BUCKET)


def download_from_gcs(gcs_path: str, local_path: str):
    bucket = _get_gcs_bucket()
    blob = bucket.blob(gcs_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)


def upload_to_gcs(local_path: str, gcs_path: str) -> str:
    import urllib.parse
    bucket = _get_gcs_bucket()
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    encoded_path = urllib.parse.quote(gcs_path, safe="")
    url = f"https://firebasestorage.googleapis.com/v0/b/{STORAGE_BUCKET}/o/{encoded_path}?alt=media"
    return url


def generate_signed_upload_url(gcs_path: str) -> str:
    import datetime
    bucket = _get_gcs_bucket()
    blob = bucket.blob(gcs_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(hours=2),
        method="PUT",
        content_type="video/mp4",
    )
    return url


# ====================================================================
# Mask / Geometry / Handoff Helpers (used by GPU function)
# ====================================================================

def _mask_helpers():
    """Return a dict of helper functions. Called inside GPU context."""
    import numpy as np
    import cv2

    def euclidean(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    def mask_centroid(mask):
        ys, xs = np.where(mask > 0.5)
        return (int(np.mean(xs)), int(np.mean(ys))) if len(xs) > 0 else None

    def mask_area(mask):
        return int(np.sum(mask > 0.5))

    def mask_bbox(mask):
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0: return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def mask_foot(mask):
        bbox = mask_bbox(mask)
        if bbox is None: return None
        return ((bbox[0]+bbox[2])//2, bbox[3])

    def mask_quality_ok(mask, min_a, max_a):
        area = mask_area(mask)
        if area < min_a: return False, area, f"small({area})"
        if area > max_a: return False, area, f"large({area})"
        bbox = mask_bbox(mask)
        if bbox is None: return False, 0, "empty"
        bw, bh = bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1
        aspect = bw / max(bh, 1)
        if aspect > 4.0: return False, area, f"wide({aspect:.1f})"
        if aspect < 0.05: return False, area, f"narrow({aspect:.2f})"
        return True, area, "ok"

    def mask_compactness(mask):
        bbox = mask_bbox(mask)
        if bbox is None: return 0.0
        bw, bh = bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1
        return mask_area(mask) / max(bw * bh, 1)

    def mask_iou(mask_a, mask_b):
        a, b = mask_a > 0.5, mask_b > 0.5
        inter = np.sum(a & b)
        union = np.sum(a | b)
        return inter / max(union, 1)

    def point_in_mask(pt, mask):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            return mask[y, x] > 0.5
        return False

    def sample_interior_points(mask, n=5, rng=None):
        """Sample N interior points from a mask for multi-point prompts."""
        if rng is None:
            rng = np.random.default_rng(42)
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return []
        # Try eroded mask for interior points
        try:
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode((mask > 0.5).astype(np.uint8), kernel, iterations=2)
            ey, ex = np.where(eroded > 0)
            if len(ex) >= n:
                idx = rng.choice(len(ex), size=n, replace=False)
                return [(int(ex[i]), int(ey[i])) for i in idx]
        except Exception:
            pass
        # Fallback: centroid + bbox quadrants
        points = [(int(np.mean(xs)), int(np.mean(ys)))]
        bbox = mask_bbox(mask)
        if bbox:
            x0, y0, x1, y1 = bbox
            dx, dy = max(1, int((x1-x0)*0.25)), max(1, int((y1-y0)*0.25))
            for pt in [(x0+dx,y0+dy),(x1-dx,y0+dy),(x0+dx,y1-dy),(x1-dx,y1-dy)]:
                if point_in_mask(pt, mask) and len(points) < n:
                    points.append(pt)
        while len(points) < n and len(xs) > 0:
            idx = rng.integers(0, len(xs))
            points.append((int(xs[idx]), int(ys[idx])))
        return points[:n]

    def build_handoff_prompt(mask, centroid, proc_w, proc_h, n_points=5):
        """Build box + multi-point prompt arrays from a handoff mask."""
        result = {"points": None, "labels": None, "box": None}
        bbox = mask_bbox(mask)
        if bbox:
            x0, y0, x1, y1 = bbox
            pad = 5
            result["box"] = np.array([
                max(0, x0-pad), max(0, y0-pad),
                min(proc_w-1, x1+pad), min(proc_h-1, y1+pad)
            ], dtype=np.float32)
        pts = sample_interior_points(mask, n=n_points)
        if not pts:
            pts = [centroid]
        result["points"] = np.array(pts, dtype=np.float32)
        result["labels"] = np.ones(len(pts), dtype=np.int32)
        return result

    def score_handoff(m1, m2, c1, c2, local_idx, total, min_sep_proc):
        """Score a handoff candidate. Higher = better."""
        sep = euclidean(c1, c2)
        sep_score = min(1.0, max(0.0, sep / (min_sep_proc * 2.0)))
        comp1, comp2 = mask_compactness(m1), mask_compactness(m2)
        a1, a2 = mask_area(m1), mask_area(m2)
        area_ratio = min(a1, a2) / max(a1, a2, 1)
        quality = (comp1 + comp2) / 2.0 * 0.5 + area_ratio * 0.5
        recency = local_idx / max(total - 1, 1)
        return sep_score * 0.50 + quality * 0.35 + recency * 0.15, sep

    def validate_prompt_mask(new_mask, expected_bbox, expected_centroid, min_a, max_a, proc_w):
        """Validate a prompt mask is reasonable."""
        ok, area, reason = mask_quality_ok(new_mask, min_a, max_a)
        if not ok:
            return False, f"quality: {reason}"
        c = mask_centroid(new_mask)
        if c is None:
            return False, "empty"
        drift = euclidean(c, expected_centroid)
        # Allow generous drift — players move between end of prev segment and overlap prompt
        if drift > proc_w * 0.6:
            return False, f"drift({drift:.0f}px)"
        return True, f"ok(area={area})"

    return {
        "euclidean": euclidean, "centroid": mask_centroid, "area": mask_area,
        "bbox": mask_bbox, "foot": mask_foot, "quality": mask_quality_ok,
        "compactness": mask_compactness, "iou": mask_iou,
        "point_in": point_in_mask, "sample_interior": sample_interior_points,
        "build_handoff_prompt": build_handoff_prompt,
        "score_handoff": score_handoff,
        "validate_prompt": validate_prompt_mask,
    }


# ====================================================================
# GPU Tracker Class
# ====================================================================

@app.cls(
    gpu="A100-80GB",
    image=sam2_image,
    volumes={DATA_DIR: scratch_vol, MODEL_CACHE: model_vol},
    secrets=[hf_secret, gcs_secret],
    timeout=1800,
    scaledown_window=300,
)
class TrackerGPU:

    @modal.enter()
    def load_model(self):
        import torch
        import sys
        sys.path.insert(0, SAM2_DIR)
        from sam2.build_sam import build_sam2_video_predictor

        ckpt = f"{MODEL_CACHE}/sam2.1_hiera_small.pt"
        cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

        if not os.path.exists(ckpt):
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="facebook/sam2.1-hiera-small",
                filename="sam2.1_hiera_small.pt",
                local_dir=MODEL_CACHE,
            )
            model_vol.commit()

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        self.predictor = build_sam2_video_predictor(cfg, ckpt, device=torch.device("cuda"))
        self.dtype = torch.bfloat16  # bfloat16 for ~2-3x speedup on A100
        print("✓ SAM2 model loaded (bfloat16 autocast enabled)")

    @modal.method()
    def extract_first_frame(self, job_id: str, video_key: str = "") -> dict:
        import cv2
        scratch_vol.reload()
        video_path = f"{DATA_DIR}/jobs/{job_id}/source.mp4"
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return {"error": "Failed to read video"}

        # Encode as JPEG at source resolution
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode()

        # Always save first frame to volume (needed by homography function)
        local_frame = f"{DATA_DIR}/jobs/{job_id}/first_frame.jpg"
        cv2.imwrite(local_frame, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        scratch_vol.commit()

        # Upload to Firebase if video_key provided
        first_frame_url = None
        if video_key:
            try:
                gcs_path = f"{OUTPUT_PREFIX}/{video_key}/first_frame.jpg"
                first_frame_url = upload_to_gcs(local_frame, gcs_path)
                print(f"✓ First frame uploaded to {gcs_path}")
            except Exception as e:
                print(f"⚠ First frame upload failed: {e}")

        return {
            "frame_b64": b64,
            "src_width": w, "src_height": h,
            "src_fps": round(fps, 2),
            "total_frames": total,
            "duration_sec": round(total / fps, 2),
            "first_frame_url": first_frame_url,
        }

    @modal.method()
    def process_segment(self, job_id: str, seg_idx: int, src_start: int,
                        src_num_frames: int, seed_p1_proc: list, seed_p2_proc: list,
                        prompt_local_idx: int, params: dict,
                        prev_handoff: dict = None,
                        yolo_box_p1_proc: list = None,
                        yolo_box_p2_proc: list = None) -> dict:
        """
        Process one segment with SAM2 + robust handoff.

        prev_handoff: if not None, contains 'bbox_p1', 'bbox_p2', 'mask_p1_pts',
                      'mask_p2_pts' from the previous segment for box+multipoint prompts.
        yolo_box_p1_proc: if not None, [x1,y1,x2,y2] from YOLO person detection (proc resolution)
        yolo_box_p2_proc: if not None, [x1,y1,x2,y2] from YOLO person detection (proc resolution)
        """
        import cv2
        import numpy as np
        import gc
        import torch
        import subprocess
        import shutil
        import time

        scratch_vol.reload()
        h = _mask_helpers()
        p = params

        # GPU diagnostics
        try:
            gpu_mem = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU: {gpu_mem.name}, {gpu_mem.total_memory / 1e9:.1f} GB total")
            print(f"  VRAM: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            import subprocess as _sp
            smi = _sp.run(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True)
            print(f"  nvidia-smi: {smi.stdout.strip()}")
        except Exception as e:
            print(f"  GPU diag failed: {e}")

        seed_p1 = tuple(seed_p1_proc)
        seed_p2 = tuple(seed_p2_proc)

        job_dir = Path(f"{DATA_DIR}/jobs/{job_id}")
        seg_dir = job_dir / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)
        out_dir = job_dir / "output_segments"
        out_dir.mkdir(parents=True, exist_ok=True)

        video_path = str(job_dir / "source.mp4")
        frame_step = p["frame_step"]
        sam2_frame_count = int(src_num_frames / frame_step)

        # ---- Step 1: Extract frames to JPEG directory ----
        frames_dir = str(seg_dir / f"frames_{seg_idx:04d}")
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir)

        cap = cv2.VideoCapture(video_path)
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Seek once to start, then use grab() to skip — avoids costly random seeks
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_start)
        current_pos = src_start
        frames_written = 0
        for j in range(sam2_frame_count):
            target_sf = src_start + int(round(j * frame_step))
            if target_sf >= total_src_frames:
                break
            # Skip intermediate frames without decoding
            while current_pos < target_sf:
                cap.grab()
                current_pos += 1
            ret, frame = cap.read()
            current_pos += 1
            if not ret: break
            resized = cv2.resize(frame, (p["proc_width"], p["proc_height"]),
                                 interpolation=cv2.INTER_AREA)
            cv2.imwrite(f"{frames_dir}/{j:05d}.jpg", resized)
            frames_written += 1
        cap.release()
        sam2_frame_count = frames_written

        if sam2_frame_count == 0:
            return {"status": "error", "message": "No frames extracted"}

        # ---- Step 2: Build prompts ----
        is_first_seg = prev_handoff is None

        if is_first_seg:
            # Use YOLO bounding box + user point click for robust full-body tracking
            if yolo_box_p1_proc:
                pad = 5
                prompt_p1 = {
                    "points": np.array([seed_p1], dtype=np.float32),
                    "labels": np.array([1], dtype=np.int32),
                    "box": np.array([
                        max(0, yolo_box_p1_proc[0] - pad),
                        max(0, yolo_box_p1_proc[1] - pad),
                        min(p["proc_width"] - 1, yolo_box_p1_proc[2] + pad),
                        min(p["proc_height"] - 1, yolo_box_p1_proc[3] + pad),
                    ], dtype=np.float32),
                }
                print(f"  P1 prompt: point {seed_p1} + YOLO box {yolo_box_p1_proc}")
            else:
                prompt_p1 = {
                    "points": np.array([seed_p1], dtype=np.float32),
                    "labels": np.array([1], dtype=np.int32),
                    "box": None,
                }
                print(f"  P1 prompt: point only {seed_p1} (no YOLO box)")

            if yolo_box_p2_proc:
                pad = 5
                prompt_p2 = {
                    "points": np.array([seed_p2], dtype=np.float32),
                    "labels": np.array([1], dtype=np.int32),
                    "box": np.array([
                        max(0, yolo_box_p2_proc[0] - pad),
                        max(0, yolo_box_p2_proc[1] - pad),
                        min(p["proc_width"] - 1, yolo_box_p2_proc[2] + pad),
                        min(p["proc_height"] - 1, yolo_box_p2_proc[3] + pad),
                    ], dtype=np.float32),
                }
                print(f"  P2 prompt: point {seed_p2} + YOLO box {yolo_box_p2_proc}")
            else:
                prompt_p2 = {
                    "points": np.array([seed_p2], dtype=np.float32),
                    "labels": np.array([1], dtype=np.int32),
                    "box": None,
                }
                print(f"  P2 prompt: point only {seed_p2} (no YOLO box)")

            expected_bbox_p1 = yolo_box_p1_proc
            expected_bbox_p2 = yolo_box_p2_proc
        else:
            # Handoff: build box + multi-point from previous mask data
            expected_bbox_p1 = prev_handoff.get("bbox_p1")
            expected_bbox_p2 = prev_handoff.get("bbox_p2")

            print(f"  Handoff debug: seed_p1={seed_p1}, seed_p2={seed_p2}")
            print(f"  Handoff debug: bbox_p1={expected_bbox_p1}, bbox_p2={expected_bbox_p2}")
            print(f"  Handoff debug: pts_p1={prev_handoff.get('mask_p1_pts')}")
            print(f"  Handoff debug: pts_p2={prev_handoff.get('mask_p2_pts')}")

            if expected_bbox_p1 and prev_handoff.get("mask_p1_pts"):
                pts = prev_handoff["mask_p1_pts"]
                bbox = expected_bbox_p1
                pad = 5
                prompt_p1 = {
                    "points": np.array(pts, dtype=np.float32),
                    "labels": np.ones(len(pts), dtype=np.int32),
                    "box": np.array([
                        max(0, bbox[0]-pad), max(0, bbox[1]-pad),
                        min(p["proc_width"]-1, bbox[2]+pad),
                        min(p["proc_height"]-1, bbox[3]+pad)
                    ], dtype=np.float32),
                }
            else:
                prompt_p1 = {
                    "points": np.array([seed_p1], dtype=np.float32),
                    "labels": np.array([1], dtype=np.int32),
                    "box": None,
                }

            if expected_bbox_p2 and prev_handoff.get("mask_p2_pts"):
                pts = prev_handoff["mask_p2_pts"]
                bbox = expected_bbox_p2
                pad = 5
                prompt_p2 = {
                    "points": np.array(pts, dtype=np.float32),
                    "labels": np.ones(len(pts), dtype=np.int32),
                    "box": np.array([
                        max(0, bbox[0]-pad), max(0, bbox[1]-pad),
                        min(p["proc_width"]-1, bbox[2]+pad),
                        min(p["proc_height"]-1, bbox[3]+pad)
                    ], dtype=np.float32),
                }
            else:
                prompt_p2 = {
                    "points": np.array([seed_p2], dtype=np.float32),
                    "labels": np.array([1], dtype=np.int32),
                    "box": None,
                }

        # ---- Step 3: SAM2 init + prompt with validation + retry ----
        gc.collect()
        torch.cuda.empty_cache()

        if not is_first_seg:
            _b1 = prompt_p1['box'].tolist() if prompt_p1.get('box') is not None else None
            _b2 = prompt_p2['box'].tolist() if prompt_p2.get('box') is not None else None
            print(f"  Prompt P1: points={prompt_p1['points'].tolist()}, box={_b1}")
            print(f"  Prompt P2: points={prompt_p2['points'].tolist()}, box={_b2}")

        p1_id, p2_id = 1, 2
        status = "ok"
        warnings_list = []

        # Build list of frames to try for prompt — start at handoff frame, alternate outward
        frames_to_try = [prompt_local_idx]
        if not is_first_seg:
            overlap_sam2 = p["overlap_frames"]
            step = max(1, overlap_sam2 // (p["handoff_prompt_retries"] + 1))
            for i in range(1, p["handoff_prompt_retries"]):
                for alt in [prompt_local_idx - i * step, prompt_local_idx + i * step]:
                    if 0 <= alt < sam2_frame_count:
                        frames_to_try.append(alt)
            # DON'T sort — keep prompt_local_idx first, then alternate outward
            # Dedup while preserving order
            seen = set()
            deduped = []
            for f in frames_to_try:
                if f not in seen:
                    seen.add(f)
                    deduped.append(f)
            frames_to_try = deduped
        print(f"  Frames to try: {frames_to_try}")

        prompt_accepted = False
        state = None

        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            # Load frames ONCE — this is the expensive step (~28s)
            state = self.predictor.init_state(video_path=frames_dir)
            print(f"  ✓ SAM2 state initialized ({sam2_frame_count} frames loaded)")

            for attempt, try_frame in enumerate(frames_to_try):
                # On retries, just reset prompts (keeps frames in VRAM — instant)
                if attempt > 0:
                    self.predictor.reset_state(state)

                # Add prompts
                p1_kwargs = dict(
                    inference_state=state, frame_idx=try_frame,
                    obj_id=p1_id,
                    points=prompt_p1["points"], labels=prompt_p1["labels"],
                )
                if prompt_p1.get("box") is not None:
                    p1_kwargs["box"] = prompt_p1["box"]
                _, out_ids_1, logits_p1 = self.predictor.add_new_points_or_box(**p1_kwargs)
                # P1 is the only object so far → always index 0
                mask_p1 = (logits_p1[0][0].cpu().numpy() > 0.0).astype(np.uint8)

                p2_kwargs = dict(
                    inference_state=state, frame_idx=try_frame,
                    obj_id=p2_id,
                    points=prompt_p2["points"], labels=prompt_p2["labels"],
                )
                if prompt_p2.get("box") is not None:
                    p2_kwargs["box"] = prompt_p2["box"]
                _, out_ids_2, logits_p2 = self.predictor.add_new_points_or_box(**p2_kwargs)
                # CRITICAL: After adding P2, logits_p2 contains masks for ALL objects:
                #   logits_p2[0] = P1's mask, logits_p2[1] = P2's mask
                # Use out_ids_2 to find the correct index for P2
                p2_mask_idx = list(out_ids_2).index(p2_id)
                mask_p2 = (logits_p2[p2_mask_idx][0].cpu().numpy() > 0.0).astype(np.uint8)

                if attempt == 0:
                    print(f"  Debug masks: out_ids_1={list(out_ids_1)} out_ids_2={list(out_ids_2)} p2_idx={p2_mask_idx}")
                    print(f"  Debug areas: P1={int(mask_p1.sum())} P2={int(mask_p2.sum())}")

                # First segment: trust user click
                if is_first_seg:
                    p1_area = float(mask_p1.sum())
                    p2_area = float(mask_p2.sum())
                    if p1_area <= 10 and p2_area <= 10:
                        reprompt_b64 = self._get_reprompt_frame(
                            video_path, src_start, try_frame, frame_step, p)
                        self.predictor.reset_state(state)
                        gc.collect(); torch.cuda.empty_cache()
                        return {
                            "status": "both_lost",
                            "message": "Could not detect players at selected positions",
                            "reprompt_frame_b64": reprompt_b64,
                            "reprompt_src_frame": src_start + int(round(try_frame * frame_step)),
                        }
                    if p1_area <= 10:
                        warnings_list.append("P1 not detected"); status = "p1_lost"; p1_id = None
                    if p2_area <= 10:
                        warnings_list.append("P2 not detected"); status = "p2_lost"; p2_id = None
                    prompt_accepted = True
                    prompt_local_idx = try_frame
                    break

                # Handoff: validate
                ok1, r1 = h["validate_prompt"](
                    mask_p1, expected_bbox_p1, seed_p1,
                    p["min_mask_area_proc"], p["max_mask_area_proc"], p["proc_width"])
                ok2, r2 = h["validate_prompt"](
                    mask_p2, expected_bbox_p2, seed_p2,
                    p["min_mask_area_proc"], p["max_mask_area_proc"], p["proc_width"])

                if ok1 and ok2:
                    # Cross-check: reject if both masks are the same player
                    iou_cross = h["iou"](mask_p1, mask_p2)
                    if iou_cross > 0.3:
                        print(f"  ✗ Attempt {attempt} frame {try_frame}: masks overlap (IoU={iou_cross:.2f}) — same player detected")
                    else:
                        prompt_accepted = True
                        prompt_local_idx = try_frame
                        print(f"  ✓ Prompt accepted attempt {attempt} frame {try_frame}: P1={r1} P2={r2} cross_iou={iou_cross:.3f}")
                        break
                else:
                    print(f"  ✗ Attempt {attempt} frame {try_frame}: P1={r1} P2={r2}")

                    # Midway: retry with box + centroid-only point (single point instead of multi)
                    if attempt == len(frames_to_try) // 2:
                        self.predictor.reset_state(state)
                        # Single centroid point + box — simpler prompt, less ambiguous
                        p1_mid_pts = np.array([seed_p1], dtype=np.float32)
                        p2_mid_pts = np.array([seed_p2], dtype=np.float32)
                        p1_mid_kwargs = dict(
                            inference_state=state, frame_idx=try_frame, obj_id=p1_id,
                            points=p1_mid_pts, labels=np.array([1], dtype=np.int32))
                        p2_mid_kwargs = dict(
                            inference_state=state, frame_idx=try_frame, obj_id=p2_id,
                            points=p2_mid_pts, labels=np.array([1], dtype=np.int32))
                        # Keep boxes if available
                        if prompt_p1.get("box") is not None:
                            p1_mid_kwargs["box"] = prompt_p1["box"]
                        if prompt_p2.get("box") is not None:
                            p2_mid_kwargs["box"] = prompt_p2["box"]
                        _, _, logits_p1 = self.predictor.add_new_points_or_box(**p1_mid_kwargs)
                        mask_p1 = (logits_p1[0][0].cpu().numpy() > 0.0).astype(np.uint8)
                        _, mid_ids_2, logits_p2 = self.predictor.add_new_points_or_box(**p2_mid_kwargs)
                        mid_p2_idx = list(mid_ids_2).index(p2_id)
                        mask_p2 = (logits_p2[mid_p2_idx][0].cpu().numpy() > 0.0).astype(np.uint8)
                        ok1, r1 = h["validate_prompt"](
                            mask_p1, expected_bbox_p1, seed_p1,
                            p["min_mask_area_proc"], p["max_mask_area_proc"], p["proc_width"])
                        ok2, r2 = h["validate_prompt"](
                            mask_p2, expected_bbox_p2, seed_p2,
                            p["min_mask_area_proc"], p["max_mask_area_proc"], p["proc_width"])
                        if ok1 and ok2:
                            iou_cross = h["iou"](mask_p1, mask_p2)
                            if iou_cross <= 0.3:
                                prompt_accepted = True
                                prompt_local_idx = try_frame
                                print(f"  ✓ Centroid+box fallback accepted frame {try_frame} cross_iou={iou_cross:.3f}")
                                break
                            else:
                                print(f"  ✗ Centroid+box fallback overlap (IoU={iou_cross:.2f})")

            if not prompt_accepted:
                print("  ⚠ All handoff prompt attempts failed — requesting user reprompt")
                # Get a frame to show the user for re-clicking
                reprompt_b64 = self._get_reprompt_frame(
                    video_path, src_start, prompt_local_idx, frame_step, p)
                self.predictor.reset_state(state)
                gc.collect(); torch.cuda.empty_cache()
                return {
                    "status": "handoff_failed",
                    "message": "Automatic handoff could not find both players. Please click on both players.",
                    "reprompt_frame_b64": reprompt_b64,
                    "reprompt_src_frame": src_start + int(round(prompt_local_idx * frame_step)),
                }

            # ---- Step 4: Propagate ----
            tracking_data = []
            mask_cache = {}
            last_progress_time = time.time()

            for frame_idx, obj_ids, video_res_masks in self.predictor.propagate_in_video(state):
                local_idx = frame_idx
                src_frame_idx = src_start + int(round(local_idx * frame_step))
                timestamp = src_frame_idx / p["src_fps"]

                entry = {
                    "frame_number": src_frame_idx,
                    "timestamp_sec": round(timestamp, 4),
                    "player_1": None, "player_2": None,
                }

                masks_for_frame = {}
                for i, oid in enumerate(obj_ids):
                    oid_int = int(oid)
                    m = (video_res_masks[i][0].cpu().numpy() > 0.0).astype(np.uint8)

                    if oid_int == p1_id:
                        key, pid = "player_1", "p1"
                    elif oid_int == p2_id:
                        key, pid = "player_2", "p2"
                    else:
                        continue

                    foot = h["foot"](m)
                    if foot:
                        entry[key] = {
                            "x": int(round(foot[0] * p["scale_x"])),
                            "y": int(round(foot[1] * p["scale_y"])),
                        }
                    masks_for_frame[pid] = m

                tracking_data.append(entry)
                mask_cache[local_idx] = masks_for_frame

                now = time.time()
                if now - last_progress_time >= 10.0:
                    prog = {"frames_done": local_idx + 1, "total": sam2_frame_count}
                    prog_path = str(job_dir / "progress.json")
                    with open(prog_path, "w") as f:
                        json.dump(prog, f)
                    scratch_vol.commit()
                    last_progress_time = now

        # Final progress
        prog = {"frames_done": sam2_frame_count, "total": sam2_frame_count}
        with open(str(job_dir / "progress.json"), "w") as f:
            json.dump(prog, f)
        scratch_vol.commit()

        # ---- Step 5: SCORED handoff selection ----
        handoff = None
        cap_check = cv2.VideoCapture(video_path)
        total_src = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_check.release()
        is_last = (src_start + src_num_frames >= total_src)

        if not is_last:
            try:
                ho = self._find_best_handoff(mask_cache, p1_id, p2_id,
                                             sam2_frame_count, h, p)
                handoff_local = ho["local_idx"]
                handoff_src = src_start + int(round(handoff_local * frame_step))
                overlap_src = int(round(p["overlap_frames"] * frame_step))
                next_src = max(0, handoff_src - overlap_src)
                pli = int(round(overlap_src / frame_step))

                handoff = {
                    "seed_p1_proc": list(ho["c1"]),
                    "seed_p2_proc": list(ho["c2"]),
                    "next_src_start": next_src,
                    "prompt_local_idx": pli,
                    # Handoff data for box+multipoint on next segment
                    "prev_handoff": {
                        "bbox_p1": ho.get("bbox_p1"),
                        "bbox_p2": ho.get("bbox_p2"),
                        "mask_p1_pts": ho.get("pts_p1"),
                        "mask_p2_pts": ho.get("pts_p2"),
                    },
                }
            except RuntimeError as e:
                warnings_list.append(f"Handoff failed: {e}")

        # Release GPU aggressively
        self.predictor.reset_state(state)
        del state
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        vram_after = torch.cuda.memory_reserved() / 1e9
        print(f"  VRAM after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {vram_after:.2f} GB reserved")

        # ---- Step 6: Save masks to volume (fast) instead of rendering (slow) ----
        # Rendering moves to a parallel CPU function
        import numpy as np
        masks_dir = str(out_dir / f"masks_{seg_idx:04d}")
        os.makedirs(masks_dir, exist_ok=True)
        for local_idx in sorted(mask_cache.keys()):
            masks = mask_cache[local_idx]
            src_frame_idx = src_start + int(round(local_idx * frame_step))
            for pid in ("p1", "p2"):
                m = masks.get(pid)
                if m is not None:
                    np.save(f"{masks_dir}/{local_idx:05d}_{pid}.npy", m)
        # Save frame mapping for the render function
        frame_map = {
            local_idx: src_start + int(round(local_idx * frame_step))
            for local_idx in sorted(mask_cache.keys())
        }
        with open(f"{masks_dir}/frame_map.json", "w") as f:
            json.dump(frame_map, f)

        del mask_cache

        # Keep JPEG frames for render_segment (it will clean up after)
        scratch_vol.commit()

        return {
            "status": status if status in ("p1_lost", "p2_lost") else "ok",
            "tracking_data": tracking_data,
            "handoff": handoff,
            "warnings": warnings_list,
            "seg_idx": seg_idx,
            "frames_processed": sam2_frame_count,
        }

    def _find_best_handoff(self, mask_cache, p1_id, p2_id, frame_count, h, p):
        """Scored handoff: evaluate last N frames, pick best.
        Strongly prefers frames where both players are moving slowly."""
        import numpy as np
        search_start = max(0, frame_count - p["handoff_search_zone"])

        # ---- Pre-compute centroids for velocity calculation ----
        centroids = {}  # idx → (c1, c2)
        for idx in range(search_start, frame_count):
            masks = mask_cache.get(idx)
            if masks is None:
                continue
            m1 = masks.get("p1")
            m2 = masks.get("p2")
            if m1 is None or m2 is None:
                continue
            c1, c2 = h["centroid"](m1), h["centroid"](m2)
            if c1 is not None and c2 is not None:
                centroids[idx] = (c1, c2)

        # ---- Compute per-frame velocity (pixels/frame) ----
        velocities = {}  # idx → (v1, v2)
        sorted_idxs = sorted(centroids.keys())
        for i in range(1, len(sorted_idxs)):
            curr = sorted_idxs[i]
            prev = sorted_idxs[i - 1]
            if curr - prev > 3:  # Skip if frames are too far apart
                continue
            c1_curr, c2_curr = centroids[curr]
            c1_prev, c2_prev = centroids[prev]
            gap = curr - prev
            v1 = h["euclidean"](c1_curr, c1_prev) / gap
            v2 = h["euclidean"](c2_curr, c2_prev) / gap
            velocities[curr] = (v1, v2)

        # ---- Smooth velocities with a 5-frame window ----
        smooth_vel = {}
        for idx in velocities:
            v1_sum, v2_sum, count = 0, 0, 0
            for j in range(idx - 2, idx + 3):
                if j in velocities:
                    v1_sum += velocities[j][0]
                    v2_sum += velocities[j][1]
                    count += 1
            if count > 0:
                smooth_vel[idx] = (v1_sum / count, v2_sum / count)

        candidates = []

        for idx in range(frame_count - 1, search_start - 1, -1):
            masks = mask_cache.get(idx)
            if masks is None: continue
            m1 = masks.get("p1") if p1_id else None
            m2 = masks.get("p2") if p2_id else None
            if m1 is None or m2 is None: continue
            c1, c2 = h["centroid"](m1), h["centroid"](m2)
            if c1 is None or c2 is None: continue
            ok1, _, _ = h["quality"](m1, p["min_mask_area_proc"], p["max_mask_area_proc"])
            ok2, _, _ = h["quality"](m2, p["min_mask_area_proc"], p["max_mask_area_proc"])
            if not ok1 or not ok2: continue

            score, sep = h["score_handoff"](m1, m2, c1, c2, idx, frame_count,
                                            p["min_separation_proc"])

            # ---- HARD FILTER: reject if players overlap or are too close ----
            if sep < p["min_separation_proc"]:
                continue  # Players too close — handoff would confuse SAM2

            # Check mask overlap — if IoU > 0.1, masks are merging
            iou = h["iou"](m1, m2)
            if iou > 0.1:
                continue  # Masks overlapping — players indistinguishable

            # ---- Velocity bonus: prefer low-velocity frames (but separation is king) ----
            if idx in smooth_vel:
                v1, v2 = smooth_vel[idx]
                combined_vel = v1 + v2
                # Low velocity (< 2 px/frame) = full bonus (0.15)
                # High velocity (> 10 px/frame) = zero bonus
                vel_bonus = max(0.0, 0.15 * (1.0 - combined_vel / 10.0))
            else:
                vel_bonus = 0.0

            # Scoring: separation is most important, velocity is a tiebreaker
            # Original score = sep(0.50) + quality(0.35) + recency(0.15)
            adjusted_score = score * 0.85 + vel_bonus

            # Collect interior points for next segment's prompt
            pts1 = h["sample_interior"](m1, n=p["handoff_multi_points"])
            pts2 = h["sample_interior"](m2, n=p["handoff_multi_points"])
            bbox1 = h["bbox"](m1)
            bbox2 = h["bbox"](m2)

            candidates.append({
                "score": adjusted_score, "sep": sep, "c1": c1, "c2": c2,
                "local_idx": idx,
                "bbox_p1": bbox1, "bbox_p2": bbox2,
                "pts_p1": pts1, "pts_p2": pts2,
                "vel": smooth_vel.get(idx, (None, None)),
            })

        if not candidates:
            raise RuntimeError("No valid handoff frame found")

        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        v_info = f"vel=({best['vel'][0]:.1f},{best['vel'][1]:.1f})px/f" if best['vel'][0] is not None else "vel=unknown"
        print(f"  ✓ Best handoff: local={best['local_idx']} score={best['score']:.3f} "
              f"sep={best['sep']:.0f}px {v_info}")
        return best

    def _get_reprompt_frame(self, video_path, src_start, prompt_idx, frame_step, p):
        import cv2
        src_frame = src_start + int(round(prompt_idx * frame_step))
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
        ret, frame = cap.read()
        cap.release()
        if not ret: return ""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode()


# ====================================================================
# CPU Helper Functions
# ====================================================================

@app.function(
    image=gemini_image,
    volumes={DATA_DIR: scratch_vol},
    secrets=[gemini_secret, gcs_secret],
    timeout=1800,
)
def compute_court_homography(job_id: str, video_key: str, src_width: int, src_height: int):
    """Call Gemini to find court landmarks, compute homography matrix, upload to Firebase.
    Runs in parallel with SAM2 tracking — only needs the first frame."""
    import cv2
    import numpy as np
    from PIL import Image as PILImage
    import re
    import json as json_mod
    import base64
    import os

    scratch_vol.reload()

    # ---- Load first frame from volume ----
    first_frame_path = f"{DATA_DIR}/jobs/{job_id}/first_frame.jpg"
    if not os.path.exists(first_frame_path):
        # Fall back to extracting from source video
        video_path = f"{DATA_DIR}/jobs/{job_id}/source.mp4"
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return {"status": "error", "message": "Could not read first frame"}
        cv2.imwrite(first_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    else:
        frame = cv2.imread(first_frame_path)

    # ---- Resize to 1000x1000 for Gemini ----
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = PILImage.fromarray(frame_rgb)
    resized = pil_frame.resize((1000, 1000), PILImage.Resampling.LANCZOS)

    import io
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=95)
    frame_b64 = base64.b64encode(buf.getvalue()).decode()

    # ---- Load reference image ----
    with open("/app/squash_court_lines.jpeg", "rb") as f:
        ref_b64 = base64.b64encode(f.read()).decode()

    # ---- Call Gemini API ----
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    REQUIRED_KEYS = [
        "T_junction",
        "Left_Service_Box_Inner_Front",
        "Left_Service_Box_Inner_Back",
        "Right_Service_Box_Inner_Front",
        "Right_Service_Box_Inner_Back",
    ]
    LINE_KEYS = [
        "Front_Wall_Floor_Line",
        "Left_Wall_Floor_Line",
        "Right_Wall_Floor_Line",
    ]
    ALL_KNOWN_KEYS = REQUIRED_KEYS  # Lines validated separately

    prompt = """
Role: You are a computer vision and geometry expert specializing in sports analytics.

Task: Analyze the two attached images to extract precise pixel coordinates (x, y) for specific landmarks on a squash court floor.

Image 1 (Reference): squash_court_lines.jpeg - This is a schematic diagram showing the standard layout of squash court lines (Short Line, Half Court Line, Service Boxes).

Image 2 (Target): first_frame_1000x1000.jpg - This is a real-world camera frame of a squash court filmed from behind the back wall glass.

Instructions:
1. Analyze the Schematic: Use Image 1 to understand the geometric relationship between the "Short Line" (horizontal line across the court), the "Half Court Line" (vertical line running from the Short Line to the back), and the "Service Boxes" (rectangular areas on the left and right).

2. Map to Target: Identify these corresponding red lines on the floor of Image 2. Note that perspective distorts the rectangle shapes into trapezoids.

3. Locate Specific Points and Lines: Find the following in Image 2:

   SERVICE BOX LANDMARKS (5 points — intersections of red lines on the floor):
   - The "T": The central intersection where the Short Line meets the Half Court Line.
   - Left Service Box (Inner-Front): The corner of the left service box that touches the Short Line (closest to the center "T").
   - Left Service Box (Inner-Back): The corner of the left service box closest to the camera and the center line (the back-right corner of the left box).
   - Right Service Box (Inner-Front): The corner of the right service box that touches the Short Line (closest to the center "T").
   - Right Service Box (Inner-Back): The corner of the right service box closest to the camera and the center line (the back-left corner of the right box).

   WALL-FLOOR BOUNDARY LINES (3 lines — each defined by TWO points on that line):
   For each wall, identify the line where the wall meets the floor. Provide two clearly visible points along each line.

   - Front Wall Floor Line: The horizontal line where the FRONT WALL meets the FLOOR. This is visible as the base of the front wall. Pick two points along this line — one on the left side, one on the right side.

   - Left Wall Floor Line: The line where the LEFT SIDE WALL meets the FLOOR. This runs from the back of the court (near the camera, bottom of image) toward the front wall (top of image). The left wall may be glass or solid. Pick two points along where this wall meets the floor — one near the front of the court, one near the back.

   - Right Wall Floor Line: The line where the RIGHT SIDE WALL meets the FLOOR. Same as above but on the right side. Pick two points along where this wall meets the floor — one near the front, one near the back.

   TIPS for finding wall-floor lines:
   - The front wall floor line is the base of the front wall — look for where the white/painted wall surface ends and the wooden floor begins.
   - The side wall floor lines may follow the edge of the glass panels or the base of the side walls. Look for the transition between wall material and wooden floor.
   - Each line only needs two points — pick points that are clearly visible and as far apart as possible for accuracy.

Output Format: Provide the result as a strictly valid JSON object. Do not include markdown formatting or explanations.

{
  "T_junction": [x, y],
  "Left_Service_Box_Inner_Front": [x, y],
  "Left_Service_Box_Inner_Back": [x, y],
  "Right_Service_Box_Inner_Front": [x, y],
  "Right_Service_Box_Inner_Back": [x, y],
  "Front_Wall_Floor_Line": [[x1, y1], [x2, y2]],
  "Left_Wall_Floor_Line": [[x1, y1], [x2, y2]],
  "Right_Wall_Floor_Line": [[x1, y1], [x2, y2]]
}
"""

    ref_bytes = base64.b64decode(ref_b64)
    frame_bytes = base64.b64decode(frame_b64)

    contents = [
        prompt,
        types.Part.from_bytes(data=ref_bytes, mime_type="image/jpeg"),
        types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg"),
    ]

    generation_config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.95,
        top_k=40,
        max_output_tokens=200048,
    )

    def validate_gemini_response(text):
        """Parse and validate Gemini response. Returns (coords_dict, error_msg)."""
        # Try to extract JSON from response
        json_str = None

        # Try raw JSON first (no markdown)
        text_stripped = text.strip()
        if text_stripped.startswith("{") and text_stripped.endswith("}"):
            json_str = text_stripped

        # Try markdown code block
        if not json_str:
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if json_match:
                json_str = json_match.group(1).strip()

        # Try finding JSON object in text
        if not json_str:
            json_match = re.search(r'\{[^{}]*"T_junction"[^{}]*"Right_Wall_Floor_Line"[^}]*\}', text, re.DOTALL)
            if not json_match:
                # Try without wall lines (backwards compat)
                json_match = re.search(r'\{[^{}]*"T_junction"[^{}]*\}', text)
            if json_match:
                json_str = json_match.group(0)

        if not json_str:
            return None, f"No JSON found in response: {text[:200]}"

        try:
            coords = json_mod.loads(json_str)
        except json_mod.JSONDecodeError as e:
            return None, f"Invalid JSON: {e} — raw: {json_str[:200]}"

        # Check all required keys present
        missing = [k for k in REQUIRED_KEYS if k not in coords]
        if missing:
            return None, f"Missing keys: {missing}"

        # Check line keys
        missing_lines = [k for k in LINE_KEYS if k not in coords]
        has_all_lines = len(missing_lines) == 0
        if missing_lines:
            print(f"  ⚠ Missing wall lines: {missing_lines} — front corners won't be computed")

        # Validate each point coordinate
        present_point_keys = [k for k in REQUIRED_KEYS if k in coords]
        for key in present_point_keys:
            val = coords[key]
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                return None, f"{key} is not [x, y]: {val}"
            x, y = val[0], val[1]
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return None, f"{key} has non-numeric values: {val}"

            # Check if values are in pixel range (0-1000)
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                coords[key] = [int(round(x)), int(round(y))]
            # Detect normalized 0-1 values and scale up
            elif 0 <= x <= 1 and 0 <= y <= 1:
                coords[key] = [int(round(x * 1000)), int(round(y * 1000))]
                print(f"  ⚠ {key} appeared normalized ({val}), scaled to {coords[key]}")
            else:
                return None, f"{key} out of range (0-1000): {val}"

        # Validate line keys — each should be [[x1,y1], [x2,y2]]
        if has_all_lines:
            for key in LINE_KEYS:
                val = coords[key]
                if not isinstance(val, (list, tuple)) or len(val) != 2:
                    return None, f"{key} is not [[x1,y1],[x2,y2]]: {val}"
                for i, pt in enumerate(val):
                    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                        return None, f"{key}[{i}] is not [x, y]: {pt}"
                    x, y = pt[0], pt[1]
                    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                        return None, f"{key}[{i}] has non-numeric values: {pt}"
                    # Scale if needed
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        coords[key][i] = [int(round(x * 1000)), int(round(y * 1000))]
                    elif 0 <= x <= 1000 and 0 <= y <= 1000:
                        coords[key][i] = [int(round(x)), int(round(y))]
                    else:
                        return None, f"{key}[{i}] out of range (0-1000): {pt}"

            # Compute front corner intersections from lines
            def line_intersect(p1, p2, p3, p4):
                """Find intersection of line (p1→p2) and line (p3→p4). Returns [x, y] or None."""
                x1, y1 = p1; x2, y2 = p2
                x3, y3 = p3; x4, y4 = p4
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-6:
                    return None  # Parallel lines
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                ix = x1 + t * (x2 - x1)
                iy = y1 + t * (y2 - y1)
                return [int(round(ix)), int(round(iy))]

            front_line = coords["Front_Wall_Floor_Line"]
            left_line = coords["Left_Wall_Floor_Line"]
            right_line = coords["Right_Wall_Floor_Line"]

            fl_corner = line_intersect(front_line[0], front_line[1], left_line[0], left_line[1])
            fr_corner = line_intersect(front_line[0], front_line[1], right_line[0], right_line[1])

            if fl_corner and fr_corner:
                # Validate: corners should be above T junction and on correct sides
                t_y = coords["T_junction"][1]
                t_x = coords["T_junction"][0]
                if fl_corner[1] > t_y + 20:
                    print(f"  ⚠ Computed Front_Left_Corner y={fl_corner[1]} below T y={t_y} — discarding corners")
                elif fr_corner[1] > t_y + 20:
                    print(f"  ⚠ Computed Front_Right_Corner y={fr_corner[1]} below T y={t_y} — discarding corners")
                elif fl_corner[0] > t_x + 50:
                    print(f"  ⚠ Computed Front_Left_Corner x={fl_corner[0]} right of T — discarding corners")
                elif fr_corner[0] < t_x - 50:
                    print(f"  ⚠ Computed Front_Right_Corner x={fr_corner[0]} left of T — discarding corners")
                else:
                    coords["Front_Left_Corner"] = fl_corner
                    coords["Front_Right_Corner"] = fr_corner
                    print(f"  ✓ Computed front corners from line intersections: FL={fl_corner} FR={fr_corner}")
            else:
                print(f"  ⚠ Could not compute front corners — lines may be parallel")

            # Remove line keys from coords (not needed for homography)
            for key in LINE_KEYS:
                coords.pop(key, None)

        # Sanity check: T_junction should be roughly center-ish, not at edges
        t = coords["T_junction"]
        if t[0] < 50 or t[0] > 950 or t[1] < 50 or t[1] > 950:
            return None, f"T_junction at extreme edge ({t}), likely wrong"

        # Sanity check: no two point-type entries should be identical
        point_entries = {k: v for k, v in coords.items() if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (int, float))}
        all_pts = list(point_entries.values())
        all_keys = list(point_entries.keys())
        for i in range(len(all_pts)):
            for j in range(i + 1, len(all_pts)):
                dist = ((all_pts[i][0] - all_pts[j][0])**2 + (all_pts[i][1] - all_pts[j][1])**2) ** 0.5
                if dist < 20:
                    return None, f"{all_keys[i]} and {all_keys[j]} are too close ({dist:.0f}px apart) — likely same point"

        # Sanity check: Left points should be left of T, Right points should be right
        t_x = coords["T_junction"][0]
        for key in ["Left_Service_Box_Inner_Front", "Left_Service_Box_Inner_Back"]:
            if coords[key][0] > t_x + 50:
                return None, f"{key} ({coords[key]}) is right of T_junction ({t_x}) — L/R swapped?"
        for key in ["Right_Service_Box_Inner_Front", "Right_Service_Box_Inner_Back"]:
            if coords[key][0] < t_x - 50:
                return None, f"{key} ({coords[key]}) is left of T_junction ({t_x}) — L/R swapped?"

        # Sanity check: Front points should be at similar Y (on the short line)
        front_y_diff = abs(coords["Left_Service_Box_Inner_Front"][1] - coords["Right_Service_Box_Inner_Front"][1])
        if front_y_diff > 80:
            return None, f"Front points Y differ by {front_y_diff}px — should be on same line"

        # Sanity check: Back points should be below (larger Y) than front points
        for side in ["Left", "Right"]:
            front_y = coords[f"{side}_Service_Box_Inner_Front"][1]
            back_y = coords[f"{side}_Service_Box_Inner_Back"][1]
            if back_y <= front_y:
                return None, f"{side} back ({back_y}) is above front ({front_y}) — should be below (closer to camera)"

        return coords, None

    # ---- Retry loop (indefinite until valid response) ----
    MAX_RETRIES = 50  # effectively unlimited — Gemini must succeed
    resized_coords = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        print(f"  → Gemini attempt {attempt + 1}/{MAX_RETRIES}...")
        try:
            response = client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=contents,
                config=generation_config,
            )

            # Debug: check finish reason and full response
            try:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                print(f"  ← finish_reason={finish_reason}")
            except Exception as dbg:
                print(f"  ← Could not read finish_reason: {dbg}")

            response_text = response.text
            print(f"  ← Response ({len(response_text)} chars): {response_text[:200]}")

            coords, error = validate_gemini_response(response_text)
            if coords:
                resized_coords = coords
                print(f"  ✓ Valid coordinates on attempt {attempt + 1}: {coords}")
                break
            else:
                last_error = error
                print(f"  ✗ Validation failed: {error}")
                # Increase temperature slightly on retries for diversity
                generation_config = types.GenerateContentConfig(
                    temperature=min(0.3 + attempt * 0.15, 0.8),
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=200048,
                )
        except Exception as e:
            last_error = str(e)
            print(f"  ✗ Gemini API error: {e}")
            import time as _time
            _time.sleep(2)  # Brief pause before retry

    if resized_coords is None:
        print(f"  ✗ All {MAX_RETRIES} Gemini attempts failed. Last error: {last_error}")
        return {"status": "error", "message": f"Gemini failed after {MAX_RETRIES} attempts: {last_error}"}

    # ---- Scale coordinates from 1000x1000 → source resolution ----
    scale_x = src_width / 1000
    scale_y = src_height / 1000
    scaled_coords = {}
    for name, (x, y) in resized_coords.items():
        scaled_coords[name] = (int(round(x * scale_x)), int(round(y * scale_y)))

    print(f"  ✓ Court landmarks (source res): {scaled_coords}")

    # ---- Compute homography matrix ----
    REAL_WORLD_COORDS = {
        "T_junction": (3.2, 4.31),
        "Left_Service_Box_Inner_Front": (1.6, 4.31),
        "Left_Service_Box_Inner_Back": (1.6, 2.71),
        "Right_Service_Box_Inner_Front": (4.8, 4.31),
        "Right_Service_Box_Inner_Back": (4.8, 2.71),
        "Front_Left_Corner": (0.0, 9.75),
        "Front_Right_Corner": (6.4, 9.75),
    }

    common_keys = sorted(set(scaled_coords.keys()) & set(REAL_WORLD_COORDS.keys()))
    if len(common_keys) < 4:
        return {"status": "error", "message": f"Only {len(common_keys)} matching landmarks found"}
    print(f"  Using {len(common_keys)} calibration points: {common_keys}")

    src_points = np.array([scaled_coords[k] for k in common_keys], dtype=np.float32)
    dst_points = np.array([REAL_WORLD_COORDS[k] for k in common_keys], dtype=np.float32)

    H, h_status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if H is None or H.shape != (3, 3):
        print(f"  ✗ findHomography returned degenerate result: {H}")
        return {"status": "error", "message": "Homography computation failed — points may be collinear"}

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        print(f"  ✗ Homography matrix is singular")
        return {"status": "error", "message": "Homography matrix is singular"}

    # Reprojection error
    projected = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.sqrt(np.sum((projected - dst_points) ** 2, axis=1))
    mean_error = float(np.mean(errors))

    print(f"  ✓ Homography computed (reprojection error: {mean_error:.4f}m)")

    # ---- Build output JSON ----
    output = {
        "homography_matrix": H.tolist(),
        "homography_matrix_inverse": H_inv.tolist(),
        "reprojection_error_meters": round(mean_error, 6),
        "court_landmarks_pixels": {k: list(scaled_coords[k]) for k in common_keys},
        "court_landmarks_meters": {k: list(REAL_WORLD_COORDS[k]) for k in common_keys},
        "num_calibration_points": len(common_keys),
        "source_resolution": {"width": src_width, "height": src_height},
        "gemini_model": "gemini-3-pro-preview",
    }

    # ---- Upload to Firebase ----
    import tempfile
    gcs_path = f"{OUTPUT_PREFIX}/{video_key}/homography.json"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json_mod.dump(output, tf, indent=2)
        tf_path = tf.name

    url = upload_to_gcs(tf_path, gcs_path)
    os.unlink(tf_path)
    print(f"  ✓ Homography uploaded to {gcs_path}")

    return {"status": "ok", "url": url, "reprojection_error": mean_error}


# ====================================================================
# Ball Tracking (YOLO) — runs in parallel on A10G
# ====================================================================

@app.function(
    image=yolo_image,
    volumes={DATA_DIR: scratch_vol},
    timeout=120,
)
def detect_person_boxes(job_id: str, frame_idx: int = 0):
    """
    Run YOLO person detection on a single frame to get bounding boxes.
    Returns list of [x1, y1, x2, y2] boxes at source resolution,
    sorted by area (largest first — most likely full-body detections).
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO

    scratch_vol.reload()
    video_path = f"{DATA_DIR}/jobs/{job_id}/source.mp4"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"  ⚠ Could not read frame {frame_idx}")
        return []

    # Run YOLOv8 person detection (class 0 = person in COCO)
    model = YOLO("yolov8n.pt")
    results = model(frame, classes=[0], conf=0.3, verbose=False)

    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            area = (x2 - x1) * (y2 - y1)
            boxes.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 3),
                "area": int(area),
            })

    # Sort by area descending (largest = most likely full body)
    boxes.sort(key=lambda b: b["area"], reverse=True)
    print(f"  ✓ YOLO person detection: {len(boxes)} people found on frame {frame_idx}")
    for i, b in enumerate(boxes):
        print(f"    Person {i+1}: {b['box']} conf={b['confidence']} area={b['area']}")

    return boxes


@app.function(
    image=yolo_image,
    volumes={DATA_DIR: scratch_vol, BALL_WEIGHTS_DIR: ball_weights_vol},
    secrets=[gcs_secret],
    gpu="A10G",
    timeout=1800,
)
def track_ball(job_id: str, video_key: str, src_fps: float, src_width: int, src_height: int):
    """Run YOLO ball detection on every frame of the source video.
    Uses batched inference + ffmpeg decode + FP16 for high GPU utilization."""
    import cv2
    import numpy as np
    import json as json_mod
    import os
    import time as _time
    import subprocess
    import threading
    from collections import deque

    scratch_vol.reload()

    video_path = f"{DATA_DIR}/jobs/{job_id}/source.mp4"
    if not os.path.exists(video_path):
        print(f"  ✗ Ball tracking: source video not found at {video_path}")
        return {"status": "error", "message": "Source video not found"}

    # ---- Load YOLO weights from volume ----
    weights_path = f"{BALL_WEIGHTS_DIR}/squash_v1/best.pt"
    if not os.path.exists(weights_path):
        print(f"  ✗ Ball tracking: weights not found at {weights_path}")
        return {"status": "error", "message": "YOLO weights not found"}

    from ultralytics import YOLO
    model = YOLO(weights_path)
    print(f"  ✓ YOLO model loaded from {weights_path}")

    # ---- Get video info ----
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"  Ball tracking: {width}x{height} @ {fps:.1f} FPS — {total_frames} frames ({round(total_frames/fps, 1)}s)")

    # ---- Decode frames with ffmpeg in background thread ----
    BATCH_SIZE = 32
    frame_queue = deque(maxlen=BATCH_SIZE * 3)  # buffer ahead
    decode_done = threading.Event()

    def decode_frames():
        """Decode all frames via ffmpeg pipe — much faster than OpenCV."""
        cmd = [
            "ffmpeg", "-i", video_path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-v", "quiet", "-"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=width * height * 3 * 4)
        frame_size = width * height * 3
        idx = 0
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            frame_queue.append((idx, frame))
            idx += 1
            # Throttle if queue is full
            while len(frame_queue) >= BATCH_SIZE * 3:
                _time.sleep(0.001)
        proc.stdout.close()
        proc.wait()
        decode_done.set()

    decoder_thread = threading.Thread(target=decode_frames, daemon=True)
    decoder_thread.start()

    # ---- Batched inference loop ----
    frames_result = {}
    detected = 0
    frame_idx = 0
    start_time = _time.time()

    while True:
        # Collect a batch
        batch_frames = []
        batch_indices = []

        while len(batch_frames) < BATCH_SIZE:
            if frame_queue:
                idx, frame = frame_queue.popleft()
                batch_frames.append(frame)
                batch_indices.append(idx)
            elif decode_done.is_set():
                break  # No more frames coming
            else:
                _time.sleep(0.001)  # Wait for decoder

        if not batch_frames:
            break

        # Run batch inference — GPU processes all frames at once
        results = model.predict(
            batch_frames,
            imgsz=1280,
            conf=0.25,
            half=True,       # FP16 — 2x throughput on A10G
            verbose=False,
        )

        # Process results
        for i, result in enumerate(results):
            idx = batch_indices[i]
            ts = round(idx / fps, 4)
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                best = int(boxes.conf.argmax())
                x_center = float(boxes.xywh[best][0])
                y_center = float(boxes.xywh[best][1])
                conf = float(boxes.conf[best])

                frames_result[str(idx)] = {
                    "frame_number": idx,
                    "timestamp_sec": ts,
                    "detected": True,
                    "pixel_x": round(x_center, 2),
                    "pixel_y": round(y_center, 2),
                    "confidence": round(conf, 4),
                }
                detected += 1
            else:
                frames_result[str(idx)] = {
                    "frame_number": idx,
                    "timestamp_sec": ts,
                    "detected": False,
                    "pixel_x": None,
                    "pixel_y": None,
                    "confidence": None,
                }

        frame_idx = batch_indices[-1]
        if frame_idx % 500 < BATCH_SIZE:
            elapsed = _time.time() - start_time
            fps_actual = (frame_idx + 1) / max(elapsed, 0.1)
            pct = round((frame_idx + 1) / total_frames * 100, 1)
            print(f"  Ball tracking: {frame_idx+1}/{total_frames} ({pct}%) — {detected} detections — {fps_actual:.1f} fps")

    decoder_thread.join(timeout=10)
    elapsed = _time.time() - start_time
    det_rate = round(detected / max(total_frames, 1) * 100, 2)
    print(f"  ✓ Ball tracking complete: {len(frames_result)} frames, {detected} detections ({det_rate}%), took {elapsed:.1f}s")

    # ---- Build output JSON ----
    output = {
        "source_resolution": {"width": width, "height": height},
        "source_fps": fps,
        "total_frames": total_frames,
        "frames_with_detection": detected,
        "detection_rate_pct": det_rate,
        "model": "YOLOv8 squash_v1",
        "confidence_threshold": 0.5,
        "coordinate_description": "Midpoint of bounding box, source resolution pixels",
        "frames": frames_result,
    }

    # ---- Upload to Firebase ----
    import tempfile
    gcs_path = f"{OUTPUT_PREFIX}/{video_key}/ball_tracking.json"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json_mod.dump(output, tf, indent=2)
        tf_path = tf.name

    url = upload_to_gcs(tf_path, gcs_path)
    os.unlink(tf_path)
    print(f"  ✓ Ball tracking JSON uploaded to {gcs_path}")

    return {"status": "ok", "url": url, "total_frames": total_frames, "detected": detected}


@app.function(
    image=web_image,
    volumes={DATA_DIR: scratch_vol},
    secrets=[gcs_secret],
    timeout=600,
)
def prepare_job(job_id: str, gcs_path: str):
    local_path = f"{DATA_DIR}/jobs/{job_id}/source.mp4"
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    download_from_gcs(gcs_path, local_path)
    scratch_vol.commit()
    return {"status": "ok", "job_id": job_id}


@app.function(
    image=web_image,
    volumes={DATA_DIR: scratch_vol},
    cpu=8,
    memory=16384,
    timeout=1200,
)
def render_segment(job_id: str, seg_idx: int, params: dict):
    """Render one segment's masks into an overlay video at proc resolution.
    Copies files to local /tmp first to avoid network volume latency."""
    import cv2
    import numpy as np
    import subprocess
    import shutil
    import time as _time

    p = params
    job_dir = Path(f"{DATA_DIR}/jobs/{job_id}")
    seg_dir = job_dir / "segments"
    out_dir = job_dir / "output_segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / f"masks_{seg_idx:04d}"
    frames_dir = seg_dir / f"frames_{seg_idx:04d}"

    # Wait for volume to sync — masks might not be visible immediately
    frame_map = None
    for sync_attempt in range(10):
        scratch_vol.reload()
        frame_map_path = masks_dir / "frame_map.json"
        if frame_map_path.exists():
            try:
                with open(str(frame_map_path), "r") as f:
                    frame_map = json.load(f)
                if frame_map:
                    break
            except:
                pass
        print(f"  ⏳ Render seg {seg_idx}: waiting for volume sync (attempt {sync_attempt + 1}/10)...")
        _time.sleep(5)

    if not frame_map:
        print(f"  ⚠ No frame_map for segment {seg_idx} after 10 retries")
        return {"status": "no_masks"}

    sorted_entries = sorted(frame_map.keys(), key=int)
    total_frames = len(sorted_entries)

    if not (frames_dir).exists():
        for sync_attempt in range(5):
            scratch_vol.reload()
            if (frames_dir).exists():
                break
            print(f"  ⏳ Render seg {seg_idx}: waiting for frame JPEGs (attempt {sync_attempt + 1}/5)...")
            _time.sleep(5)

    # ---- Bulk copy to local /tmp to avoid per-file network latency ----
    local_tmp = Path(f"/tmp/render_{job_id}_{seg_idx}")
    local_masks = local_tmp / "masks"
    local_frames = local_tmp / "frames"
    local_masks.mkdir(parents=True, exist_ok=True)
    local_frames.mkdir(parents=True, exist_ok=True)

    copy_start = _time.time()
    print(f"  Render seg {seg_idx}: copying {total_frames} frames + masks to local disk...")

    # Copy masks
    mask_count = 0
    for local_idx_str in sorted_entries:
        local_idx = int(local_idx_str)
        for suffix in [f"{local_idx:05d}_p1.npy", f"{local_idx:05d}_p2.npy"]:
            src = masks_dir / suffix
            if src.exists():
                shutil.copy2(str(src), str(local_masks / suffix))
                mask_count += 1
        # Copy JPEG
        jpg_name = f"{local_idx:05d}.jpg"
        src_jpg = frames_dir / jpg_name
        if src_jpg.exists():
            shutil.copy2(str(src_jpg), str(local_frames / jpg_name))

    copy_time = _time.time() - copy_start
    print(f"  Render seg {seg_idx}: copied {mask_count} masks + {total_frames} JPEGs in {copy_time:.1f}s")

    # ---- Render from local disk ----
    COLOR_P1 = (0, 0, 255)
    COLOR_P2 = (255, 0, 0)

    # Render at lower fps for speed (every Nth frame)
    render_fps = min(5, p["target_fps"])
    frame_skip = max(1, p["target_fps"] // render_fps)
    render_entries = sorted_entries[::frame_skip]
    total_render = len(render_entries)
    print(f"  Render seg {seg_idx}: {total_render}/{total_frames} frames at {render_fps}fps (skip {frame_skip})")

    # Single-pass H.264 via ffmpeg pipe (no double encode)
    local_out_path = str(local_tmp / f"out_{seg_idx:04d}.mp4")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{p['proc_width']}x{p['proc_height']}",
        "-pix_fmt", "bgr24",
        "-r", str(render_fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        local_out_path,
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frames_written = 0
    frames_skipped = 0
    render_start = _time.time()

    for local_idx_str in render_entries:
        local_idx = int(local_idx_str)

        jpeg_path = str(local_frames / f"{local_idx:05d}.jpg")
        if not os.path.exists(jpeg_path):
            frames_skipped += 1
            continue
        frame = cv2.imread(jpeg_path)
        if frame is None:
            frames_skipped += 1
            continue

        for pid, color, label in [("p1", COLOR_P1, "P1"), ("p2", COLOR_P2, "P2")]:
            mask_path = str(local_masks / f"{local_idx:05d}_{pid}.npy")
            if not os.path.exists(mask_path):
                continue
            m = np.load(mask_path)
            mask_bool = m.astype(bool)
            overlay = frame.copy()
            overlay[mask_bool] = color
            frame = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)

            ys, xs = np.where(m > 0)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                cv2.circle(frame, (cx, cy), 6, color, -1)
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 2)
                cv2.putText(frame, label, (cx + 10, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        try:
            ffmpeg_proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print(f"  ⚠ ffmpeg pipe broken at frame {frames_written}")
            break
        frames_written += 1

        if frames_written % 300 == 0:
            elapsed = _time.time() - render_start
            fps_actual = frames_written / max(elapsed, 0.1)
            print(f"  Render seg {seg_idx}: {frames_written}/{total_render} frames ({fps_actual:.1f} fps)")

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait(timeout=300)
    render_time = _time.time() - render_start
    print(f"  Render seg {seg_idx}: wrote {frames_written} frames, skipped {frames_skipped}, "
          f"render {render_time:.1f}s ({frames_written/max(render_time,0.1):.1f} fps)")

    if frames_written == 0:
        shutil.rmtree(str(local_tmp), ignore_errors=True)
        return {"status": "error", "message": "No frames rendered"}

    # Copy final output back to volume
    final_vol_path = str(out_dir / f"out_{seg_idx:04d}.mp4")
    shutil.copy2(local_out_path, final_vol_path)

    # Clean up volume masks/frames and local tmp
    shutil.rmtree(str(masks_dir), ignore_errors=True)
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    shutil.rmtree(str(local_tmp), ignore_errors=True)

    scratch_vol.commit()
    print(f"  ✓ Segment {seg_idx} rendered ({frames_written} frames)")
    return {"status": "ok", "seg_idx": seg_idx}


# ====================================================================
# Unified World Coord Conversion + Point Detection
# (runs once after finalize_and_upload compiles all segments)
# ====================================================================

@app.function(
    image=web_image,
    secrets=[gcs_secret],
    timeout=900,
)
def convert_and_detect_points(video_key: str):
    """
    Unified post-processing pipeline:
      1. Download player_tracking.json (pixel coords, all segments combined)
      2. Download homography.json (wait if needed)
      3. Convert to world coordinates → upload player_locations.json
      4. Download ball_tracking.json (wait if needed)
      5. Run point detection on full match
      6. Upload point_starts.json
    """
    import numpy as np
    import json as json_mod
    import tempfile
    import time as _time

    COURT_WIDTH = 6.4
    HALF_COURT_X = COURT_WIDTH / 2

    # Detection thresholds (tuned on ground truth data)
    LOW_THRESH = 1.0
    MIN_LOW_DURATION = 0.5
    BURST_THRESH = 2.0
    MIN_BETWEEN = 5.0

    print("=" * 60)
    print(f"Convert + Detect: {video_key}")
    print("=" * 60)

    gcs_dir = f"{OUTPUT_PREFIX}/{video_key}"

    # ================================================================
    # Step 1: Download player_tracking.json
    # ================================================================
    print("\n📥 Downloading player tracking...")
    local_tracking = f"/tmp/player_tracking_{video_key}.json"
    try:
        download_from_gcs(f"{gcs_dir}/player_tracking.json", local_tracking)
        with open(local_tracking, "r") as f:
            tracking_data = json_mod.load(f)
        print(f"  ✓ {tracking_data.get('total_frames_tracked', '?')} frames")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return {"status": "error", "message": f"Player tracking not found: {e}"}

    # ================================================================
    # Step 2: Download homography (wait if needed)
    # ================================================================
    print("\n📥 Downloading homography...")
    H = None
    h_data = {}
    for attempt in range(30):
        try:
            local_h = f"/tmp/homography_{video_key}.json"
            download_from_gcs(f"{gcs_dir}/homography.json", local_h)
            with open(local_h, "r") as f:
                h_data = json_mod.load(f)
            H = np.array(h_data["homography_matrix"], dtype=np.float64)
            if H.shape == (3, 3):
                print(f"  ✓ Homography loaded (error: {h_data.get('reprojection_error_meters', '?')}m)")
                break
            H = None
        except Exception:
            pass
        if attempt < 29:
            if attempt % 6 == 0:
                print(f"  ⏳ Waiting for homography (attempt {attempt + 1}/30)...")
            _time.sleep(10)

    if H is None:
        print("  ✗ Homography not available")
        return {"status": "error", "message": "Homography not available"}

    # ================================================================
    # Step 3: Convert to world coordinates
    # ================================================================
    print("\n🌍 Converting to world coordinates...")
    src_fps = tracking_data.get("source_fps", 30)
    world_frames = []
    converted = 0

    for entry in tracking_data.get("frames", []):
        world_entry = {
            "frame_number": entry["frame_number"],
            "timestamp_sec": entry["timestamp_sec"],
            "player_1": None,
            "player_2": None,
        }

        for player_key in ("player_1", "player_2"):
            pdata = entry.get(player_key)
            if pdata is None or pdata.get("x") is None:
                continue

            px = float(pdata["x"])
            py = float(pdata["y"])

            pt = np.array([px, py, 1.0], dtype=np.float64)
            transformed = H @ pt
            if abs(transformed[2]) < 1e-10:
                continue
            court_x = round(float(transformed[0] / transformed[2]), 4)
            court_y = round(float(transformed[1] / transformed[2]), 4)

            # Reject coordinates far outside court bounds (bad homography extrapolation)
            # Court is 6.4m × 9.75m; allow ~1m margin for near-wall positions
            if court_x < -1.5 or court_x > 7.9 or court_y < -2.0 or court_y > 11.75:
                continue

            world_entry[player_key] = {
                "pixel_x": int(pdata["x"]),
                "pixel_y": int(pdata["y"]),
                "court_x": court_x,
                "court_y": court_y,
            }
            converted += 1

        world_frames.append(world_entry)

    # Build and upload player_locations.json
    world_output = {
        "source_resolution": tracking_data.get("source_resolution"),
        "source_fps": src_fps,
        "total_frames": len(world_frames),
        "frames_with_world_coords": converted,
        "time_range": {
            "start_sec": world_frames[0]["timestamp_sec"] if world_frames else 0,
            "end_sec": world_frames[-1]["timestamp_sec"] if world_frames else 0,
        },
        "coordinate_description": "court_x/court_y in meters from front-left corner of court (6.4m x 9.75m)",
        "homography_reprojection_error_m": h_data.get("reprojection_error_meters"),
        "frames": world_frames,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json_mod.dump(world_output, tf, indent=2)
        tf_path = tf.name
    world_url = upload_to_gcs(tf_path, f"{gcs_dir}/player_locations.json")
    os.unlink(tf_path)
    print(f"  ✓ player_locations.json uploaded ({converted} coords in {len(world_frames)} frames)")

    # ================================================================
    # Step 4: Download ball tracking (wait if needed)
    # ================================================================
    print("\n📥 Downloading ball tracking...")
    ball_data = None
    for attempt in range(60):
        try:
            local_ball = f"/tmp/ball_{video_key}.json"
            download_from_gcs(f"{gcs_dir}/ball_tracking.json", local_ball)
            with open(local_ball, "r") as f:
                ball_data = json_mod.load(f)
            print(f"  ✓ Ball tracking loaded ({ball_data.get('frames_with_detection', '?')} detections)")
            break
        except Exception:
            pass
        if attempt < 59:
            if attempt % 6 == 0:
                print(f"  ⏳ Waiting for ball tracking (attempt {attempt + 1}/60)...")
            _time.sleep(10)

    if ball_data is None:
        print("  ⚠ Ball tracking not available — running detection without ball data")

    # ================================================================
    # Step 5: Run point detection on full match
    # ================================================================
    print("\n⚙️  Running point detection...")

    # Parse world coords into arrays
    valid_frames = [fr for fr in world_frames
                    if fr.get("player_1") and fr.get("player_2")
                    and fr["player_1"].get("court_x") is not None
                    and fr["player_2"].get("court_x") is not None]

    if len(valid_frames) < 10:
        print(f"  ✗ Not enough valid frames ({len(valid_frames)})")
        return {"status": "error", "message": "Not enough valid frames for point detection"}

    ts = np.array([fr["timestamp_sec"] for fr in valid_frames])
    p1x = np.array([fr["player_1"]["court_x"] for fr in valid_frames])
    p1y = np.array([fr["player_1"]["court_y"] for fr in valid_frames])
    p2x = np.array([fr["player_2"]["court_x"] for fr in valid_frames])
    p2y = np.array([fr["player_2"]["court_y"] for fr in valid_frames])

    # Velocities with capping — players physically can't exceed ~8 m/s in squash
    MAX_VELOCITY = 8.0  # m/s — anything higher is tracking noise/homography error
    dt = np.diff(ts, prepend=ts[0] - 0.1)
    dt[dt <= 0] = 0.1
    p1v = np.sqrt(np.diff(p1x, prepend=p1x[0])**2 + np.diff(p1y, prepend=p1y[0])**2) / dt
    p2v = np.sqrt(np.diff(p2x, prepend=p2x[0])**2 + np.diff(p2y, prepend=p2y[0])**2) / dt
    p1v = np.clip(p1v, 0, MAX_VELOCITY)
    p2v = np.clip(p2v, 0, MAX_VELOCITY)

    def smooth(arr, w=5):
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode='same')

    p1v_s = smooth(p1v)
    p2v_s = smooth(p2v)

    # Detect frozen tracking — if a player barely moves for 30+ frames, they're lost
    FROZEN_THRESH = 0.05  # m/s — effectively stationary
    FROZEN_WINDOW = 30    # frames
    p1_frozen = np.zeros(len(ts), dtype=bool)
    p2_frozen = np.zeros(len(ts), dtype=bool)
    for i in range(FROZEN_WINDOW, len(ts)):
        if np.mean(p1v[i - FROZEN_WINDOW:i]) < FROZEN_THRESH:
            p1_frozen[i] = True
        if np.mean(p2v[i - FROZEN_WINDOW:i]) < FROZEN_THRESH:
            p2_frozen[i] = True

    n_p1_frozen = p1_frozen.sum()
    n_p2_frozen = p2_frozen.sum()
    if n_p1_frozen > 0 or n_p2_frozen > 0:
        print(f"  ⚠️  Frozen tracking detected: P1={n_p1_frozen} frames ({100*n_p1_frozen/len(ts):.1f}%), "
              f"P2={n_p2_frozen} frames ({100*n_p2_frozen/len(ts):.1f}%)")

    # Build adaptive combined velocity:
    # - When both tracked: P1 + P2 (normal)
    # - When one frozen: double the other player's velocity to compensate
    combined = np.zeros(len(ts))
    for i in range(len(ts)):
        if p1_frozen[i] and p2_frozen[i]:
            combined[i] = 0.0  # Both lost — can't detect anything
        elif p1_frozen[i]:
            combined[i] = p2v_s[i] * 2.0  # Only P2 active
        elif p2_frozen[i]:
            combined[i] = p1v_s[i] * 2.0  # Only P1 active
        else:
            combined[i] = p1v_s[i] + p2v_s[i]  # Both active

    # Ball density helper
    ball_frames_dict = ball_data.get("frames", {}) if ball_data else {}
    ball_fps = ball_data.get("source_fps", 30) if ball_data else 30

    def ball_density(t_sec, window_sec=2.0):
        if not ball_frames_dict:
            return -1.0
        cf = int(t_sec * ball_fps)
        hw = int(window_sec * ball_fps / 2)
        count = total = 0
        for fn in range(max(0, cf - hw), cf + hw):
            bf = ball_frames_dict.get(str(fn))
            if bf:
                total += 1
                if bf.get("detected"):
                    count += 1
        return count / max(total, 1)

    # Find low-activity periods
    low_periods = []
    in_low = False
    low_start = None

    for i in range(len(ts)):
        if combined[i] < LOW_THRESH:
            if not in_low:
                in_low = True
                low_start = i
        else:
            if in_low:
                duration = ts[i] - ts[low_start]
                if duration >= MIN_LOW_DURATION:
                    low_periods.append((low_start, i, duration))
                in_low = False

    print(f"  Low-activity periods: {len(low_periods)}")

    # Find burst after each low period
    point_starts = []
    last_point_time = -999.0

    for si, ei, gap_dur in low_periods:
        burst_idx = None
        for j in range(ei, min(ei + 50, len(ts))):
            if combined[j] > BURST_THRESH:
                burst_idx = j
                break

        if burst_idx is None:
            continue

        point_time = float(ts[burst_idx])
        if point_time - last_point_time < MIN_BETWEEN:
            continue

        # Confirming signals
        opposite_sides = bool((p1x[burst_idx] < HALF_COURT_X) != (p2x[burst_idx] < HALF_COURT_X))
        x_sep = float(abs(p1x[burst_idx] - p2x[burst_idx]))
        bd = ball_density(point_time)
        bd_pause = ball_density((ts[si] + ts[ei]) / 2)

        confidence = 0.0
        signals = []

        if gap_dur >= 3.0:
            confidence += 0.3; signals.append("long_pause")
        elif gap_dur >= 1.5:
            confidence += 0.2; signals.append("medium_pause")
        else:
            confidence += 0.1; signals.append("short_pause")

        if opposite_sides and x_sep > 2.0:
            confidence += 0.3; signals.append("opposite_sides_wide")
        elif opposite_sides:
            confidence += 0.2; signals.append("opposite_sides")

        if bd >= 0 and bd_pause >= 0:
            if bd_pause < 0.3 and bd > 0.3:
                confidence += 0.2; signals.append("ball_reappears")
            elif bd > 0.3:
                confidence += 0.1; signals.append("ball_present")

        if combined[burst_idx] > 2.5:
            confidence += 0.2; signals.append("strong_burst")
        else:
            confidence += 0.1; signals.append("burst")

        confidence = min(confidence, 1.0)
        p1_side = "left" if p1x[burst_idx] < HALF_COURT_X else "right"
        p2_side = "left" if p2x[burst_idx] < HALF_COURT_X else "right"

        point_starts.append({
            "point_number": len(point_starts) + 1,
            "timestamp_sec": round(point_time, 3),
            "frame_number": int(point_time * src_fps),
            "confidence": round(confidence, 3),
            "signals": signals,
            "gap_before_sec": round(gap_dur, 3),
            "player_1": {
                "court_x": round(float(p1x[burst_idx]), 3),
                "court_y": round(float(p1y[burst_idx]), 3),
                "side": p1_side,
            },
            "player_2": {
                "court_x": round(float(p2x[burst_idx]), 3),
                "court_y": round(float(p2y[burst_idx]), 3),
                "side": p2_side,
            },
            "opposite_sides": opposite_sides,
            "ball_density": round(bd, 3),
        })
        last_point_time = point_time

    # ================================================================
    # Step 6: Upload point_starts.json
    # ================================================================
    points_output = {
        "video_key": video_key,
        "total_points_detected": len(point_starts),
        "time_range": {
            "start_sec": float(ts[0]),
            "end_sec": float(ts[-1]),
        },
        "detection_config": {
            "low_activity_threshold": LOW_THRESH,
            "min_low_duration_sec": MIN_LOW_DURATION,
            "burst_threshold": BURST_THRESH,
            "min_between_points_sec": MIN_BETWEEN,
        },
        "point_starts": point_starts,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json_mod.dump(points_output, tf, indent=2)
        tf_path = tf.name
    points_url = upload_to_gcs(tf_path, f"{gcs_dir}/point_starts.json")
    os.unlink(tf_path)

    print(f"\n  ✅ Point detection complete: {len(point_starts)} points")
    for ps in point_starts:
        print(f"    Point {ps['point_number']:3d} | "
              f"{ps['timestamp_sec']:6.1f}s | "
              f"conf={ps['confidence']:.2f} | "
              f"opp={ps['opposite_sides']} | "
              f"{', '.join(ps['signals'])}")

    return {
        "status": "ok",
        "world_coords_url": world_url,
        "point_starts_url": points_url,
        "total_points": len(point_starts),
    }


@app.function(
    image=web_image,
    volumes={DATA_DIR: scratch_vol},
    secrets=[gcs_secret],
    cpu=8,
    memory=16384,
    timeout=1800,
)
def finalize_and_upload(job_id: str, tracking_data: list, video_key: str):
    """Wait for renders, concatenate segments, export JSON, upload to Firebase."""
    import subprocess
    import cv2

    job_dir = Path(f"{DATA_DIR}/jobs/{job_id}")
    out_dir = job_dir / "output_segments"
    scratch_vol.reload()

    deduped = {}
    for entry in tracking_data:
        deduped[entry["frame_number"]] = entry
    frames_sorted = [deduped[k] for k in sorted(deduped.keys())]

    cap = cv2.VideoCapture(str(job_dir / "source.mp4"))
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sfps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    output_json = {
        "source_resolution": {"width": sw, "height": sh},
        "source_fps": sfps,
        "total_frames_tracked": len(frames_sorted),
        "model": "SAM2.1 Hiera Small",
        "prompt_type": "point_prompts + box_handoff",
        "coordinate_description": "Midpoint of lower edge of bounding box, source resolution pixels",
        "frames": frames_sorted,
    }
    json_path = str(job_dir / "player_tracking.json")
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)

    seg_files = sorted(out_dir.glob("out_*.mp4"))
    if seg_files:
        concat_list = str(job_dir / "concat.txt")
        with open(concat_list, "w") as f:
            for sf in seg_files:
                f.write(f"file '{sf}'\n")

        final_video = str(job_dir / "tracked_output.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", final_video
        ], check=True, capture_output=True)
    else:
        final_video = None

    scratch_vol.commit()

    gcs_dir = f"{OUTPUT_PREFIX}/{video_key}"
    json_url = upload_to_gcs(json_path, f"{gcs_dir}/player_tracking.json")
    video_url = None
    if final_video and Path(final_video).exists():
        video_url = upload_to_gcs(final_video, f"{gcs_dir}/tracked_output.mp4")

    # Spawn unified world-coord conversion + point detection (fire-and-forget)
    try:
        convert_and_detect_points.spawn(video_key)
        print(f"  ✓ World coord conversion + point detection spawned")
    except Exception as e:
        print(f"  ⚠ Convert + detect spawn failed: {e}")

    return {"json_url": json_url, "video_url": video_url}


# ====================================================================
# FastAPI Web Server
# ====================================================================

@app.function(
    image=web_image,
    volumes={DATA_DIR: scratch_vol},
    secrets=[gcs_secret],
    timeout=3600,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI()
    api.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    tracker_gpu = TrackerGPU()

    @api.get("/")
    async def index():
        html = Path("/app/index.html").read_text()
        return HTMLResponse(html)

    @api.post("/api/create-job")
    async def create_job(body: dict = None):
        job_id = str(uuid.uuid4())[:8]
        filename = (body or {}).get("filename", "video.mp4")
        video_key = make_video_key(filename)
        return JSONResponse({"job_id": job_id, "video_key": video_key})

    @api.post("/api/get-upload-url")
    async def get_upload_url(body: dict):
        job_id = body["job_id"]
        video_key = body["video_key"]
        gcs_path = f"{UPLOAD_PREFIX}/{video_key}.mp4"
        url = generate_signed_upload_url(gcs_path)
        return JSONResponse({"upload_url": url, "gcs_path": gcs_path})

    @api.post("/api/prepare")
    async def prepare(body: dict):
        job_id = body["job_id"]
        gcs_path = body["gcs_path"]
        video_key = body.get("video_key", "")
        await asyncio.to_thread(prepare_job.remote, job_id, gcs_path)
        result = await asyncio.to_thread(
            tracker_gpu.extract_first_frame.remote, job_id, video_key)

        # Spawn court homography computation in parallel (fire-and-forget)
        if video_key and result.get("src_width"):
            try:
                compute_court_homography.spawn(
                    job_id, video_key,
                    result["src_width"], result["src_height"])
                print(f"✓ Homography computation spawned for {video_key}")
            except Exception as e:
                print(f"⚠ Homography spawn failed: {e}")

            # Spawn ball tracking in parallel (fire-and-forget)
            try:
                track_ball.spawn(
                    job_id, video_key,
                    result["src_fps"], result["src_width"], result["src_height"])
                print(f"✓ Ball tracking spawned for {video_key}")
            except Exception as e:
                print(f"⚠ Ball tracking spawn failed: {e}")

        return JSONResponse(result)

    @api.websocket("/ws/{job_id}")
    async def ws_process(ws: WebSocket, job_id: str):
        await ws.accept()
        try:
            data = await ws.receive_json()
            if data.get("type") != "start_processing":
                await ws.send_json({"type": "error", "message": "Expected start_processing"})
                return

            player_1_src = data["player_1"]
            player_2_src = data["player_2"]
            src_width = data["src_width"]
            src_height = data["src_height"]
            src_fps = data["src_fps"]
            total_frames = data["total_frames"]
            gcs_path = data["gcs_path"]
            video_key = data.get("video_key", "")

            params = compute_proc_params(src_width, src_height, src_fps)

            def to_proc(pt):
                return [int(round(pt[0] / params["scale_x"])),
                        int(round(pt[1] / params["scale_y"]))]

            seed_p1_proc = to_proc(player_1_src)
            seed_p2_proc = to_proc(player_2_src)
            prompt_local_idx = 0
            current_src_start = 0
            prev_handoff = None  # Carries box+pts data between segments

            # ---- YOLO person detection for initial box prompts ----
            yolo_box_p1_proc = None
            yolo_box_p2_proc = None
            try:
                await ws.send_json({"type": "status", "message": "Detecting player bounding boxes (YOLO)..."})
                person_boxes = await asyncio.to_thread(
                    lambda: detect_person_boxes.remote(job_id, 0)
                )
                if person_boxes:
                    # Match each clicked point to the YOLO box that contains it
                    def find_box_for_point(pt_src, boxes):
                        """Find the YOLO box containing the clicked point."""
                        px, py = pt_src
                        for b in boxes:
                            x1, y1, x2, y2 = b["box"]
                            if x1 <= px <= x2 and y1 <= py <= y2:
                                return b["box"]
                        # Fallback: find nearest box by center distance
                        best = None
                        best_dist = float('inf')
                        for b in boxes:
                            x1, y1, x2, y2 = b["box"]
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                            if dist < best_dist:
                                best_dist = dist
                                best = b["box"]
                        return best

                    box_p1_src = find_box_for_point(player_1_src, person_boxes)
                    box_p2_src = find_box_for_point(player_2_src, person_boxes)

                    # Make sure P1 and P2 got different boxes
                    if box_p1_src and box_p2_src and box_p1_src == box_p2_src:
                        # Both matched same box — only use for the closer click
                        boxes_list = [b["box"] for b in person_boxes]
                        if len(boxes_list) >= 2:
                            box_p1_src = boxes_list[0]
                            box_p2_src = boxes_list[1]
                        else:
                            box_p2_src = None  # Can't distinguish

                    # Convert boxes to proc resolution
                    if box_p1_src:
                        yolo_box_p1_proc = [
                            int(round(box_p1_src[0] / params["scale_x"])),
                            int(round(box_p1_src[1] / params["scale_y"])),
                            int(round(box_p1_src[2] / params["scale_x"])),
                            int(round(box_p1_src[3] / params["scale_y"])),
                        ]
                        print(f"  ✓ YOLO box P1: {box_p1_src} → proc {yolo_box_p1_proc}")

                    if box_p2_src:
                        yolo_box_p2_proc = [
                            int(round(box_p2_src[0] / params["scale_x"])),
                            int(round(box_p2_src[1] / params["scale_y"])),
                            int(round(box_p2_src[2] / params["scale_x"])),
                            int(round(box_p2_src[3] / params["scale_y"])),
                        ]
                        print(f"  ✓ YOLO box P2: {box_p2_src} → proc {yolo_box_p2_proc}")

                    if not box_p1_src and not box_p2_src:
                        print("  ⚠ No YOLO boxes matched — using point prompts only")
                else:
                    print("  ⚠ YOLO found no people — using point prompts only")
            except Exception as e:
                print(f"  ⚠ YOLO person detection failed: {e} — using point prompts only")

            src_per_seg = int(math.ceil(params["frames_per_segment"] * params["frame_step"]))
            est_segments = math.ceil(total_frames / src_per_seg)

            all_tracking_data = []
            seg_idx = 0
            render_tasks = []  # Parallel CPU render jobs

            await ws.send_json({
                "type": "status",
                "message": f"Starting processing: {est_segments} estimated segments",
                "params": {
                    "proc_resolution": f"{params['proc_width']}x{params['proc_height']}",
                    "target_fps": params["target_fps"],
                    "frames_per_segment": params["frames_per_segment"],
                    "overlap_frames": params["overlap_frames"],
                    "handoff_strategy": "scored + box+multipoint + validation",
                }
            })

            while current_src_start < total_frames:
                seg_idx += 1
                src_remaining = total_frames - current_src_start
                src_num = min(src_per_seg, src_remaining)
                is_last = (current_src_start + src_num >= total_frames)

                await ws.send_json({
                    "type": "segment_start",
                    "segment": seg_idx,
                    "est_total": est_segments,
                    "src_start": current_src_start,
                    "src_frames": src_num,
                    "is_last": is_last,
                })

                # Capture prev_handoff and yolo boxes for lambda closure
                _prev_handoff = prev_handoff
                _yolo_p1 = yolo_box_p1_proc if seg_idx == 1 else None
                _yolo_p2 = yolo_box_p2_proc if seg_idx == 1 else None

                gpu_task = asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: tracker_gpu.process_segment.remote(
                        job_id, seg_idx, current_src_start, src_num,
                        seed_p1_proc, seed_p2_proc, prompt_local_idx, params,
                        _prev_handoff, _yolo_p1, _yolo_p2
                    )
                )

                import time as _time
                start_time = _time.time()
                prog_path = f"{DATA_DIR}/jobs/{job_id}/progress.json"
                last_frames_done = 0
                sam2_total = int(src_num / params["frame_step"])

                while True:
                    done, _ = await asyncio.wait({gpu_task}, timeout=10.0)
                    if done:
                        result = gpu_task.result()
                        break
                    elapsed = _time.time() - start_time
                    frames_done = last_frames_done
                    try:
                        scratch_vol.reload()
                        with open(prog_path, "r") as pf:
                            prog_data = json.load(pf)
                            frames_done = prog_data.get("frames_done", 0)
                            sam2_total = prog_data.get("total", sam2_total)
                            last_frames_done = frames_done
                    except: pass

                    pct = round(frames_done / max(sam2_total, 1) * 100, 1)
                    eta_str = ""
                    if frames_done > 0 and elapsed > 5:
                        rate = frames_done / elapsed
                        remaining = (sam2_total - frames_done) / max(rate, 0.01)
                        mins, secs = divmod(int(remaining), 60)
                        eta_str = f" • ~{mins}m {secs}s remaining"

                    await ws.send_json({
                        "type": "progress",
                        "segment": seg_idx,
                        "frames_done": frames_done,
                        "frames_total": sam2_total,
                        "pct": pct,
                        "elapsed_sec": round(elapsed),
                        "message": f"Segment {seg_idx}: {frames_done}/{sam2_total} frames ({pct}%){eta_str}",
                    })

                status = result.get("status", "error")

                if status == "both_lost":
                    await ws.send_json({
                        "type": "needs_reprompt",
                        "segment": seg_idx,
                        "message": result.get("message", "Both players lost"),
                        "frame_b64": result.get("reprompt_frame_b64", ""),
                        "src_frame": result.get("reprompt_src_frame", current_src_start),
                        "lost_players": [1, 2],
                    })

                    reprompt = await ws.receive_json()
                    if reprompt.get("type") == "reprompt":
                        player_1_src = reprompt["player_1"]
                        player_2_src = reprompt["player_2"]
                        seed_p1_proc = to_proc(player_1_src)
                        seed_p2_proc = to_proc(player_2_src)
                        prompt_local_idx = 0
                        prev_handoff = None
                        continue
                    elif reprompt.get("type") == "skip":
                        current_src_start += src_num
                        prompt_local_idx = 0
                        prev_handoff = None
                        continue
                    else:
                        break

                if status == "handoff_failed":
                    await ws.send_json({
                        "type": "needs_reprompt",
                        "segment": seg_idx,
                        "message": result.get("message", "Handoff failed — please click both players"),
                        "frame_b64": result.get("reprompt_frame_b64", ""),
                        "src_frame": result.get("reprompt_src_frame", current_src_start),
                        "lost_players": [1, 2],
                    })

                    reprompt = await ws.receive_json()
                    if reprompt.get("type") == "reprompt":
                        player_1_src = reprompt["player_1"]
                        player_2_src = reprompt["player_2"]
                        seed_p1_proc = to_proc(player_1_src)
                        seed_p2_proc = to_proc(player_2_src)
                        # Retry same segment with user clicks — no handoff data
                        prompt_local_idx = 0
                        prev_handoff = None
                        continue
                    elif reprompt.get("type") == "skip":
                        current_src_start += src_num
                        prompt_local_idx = 0
                        prev_handoff = None
                        continue
                    else:
                        break

                if status in ("p1_lost", "p2_lost"):
                    await ws.send_json({
                        "type": "needs_reprompt",
                        "segment": seg_idx,
                        "message": f"Player {'1' if 'p1' in status else '2'} lost",
                        "lost_players": [1] if "p1" in status else [2],
                        "tracking_partial": True,
                    })
                    reprompt = await ws.receive_json()
                    if reprompt.get("type") == "reprompt":
                        player_1_src = reprompt["player_1"]
                        player_2_src = reprompt["player_2"]
                        seed_p1_proc = to_proc(player_1_src)
                        seed_p2_proc = to_proc(player_2_src)

                if result.get("tracking_data"):
                    all_tracking_data.extend(result["tracking_data"])

                # Upload segment JSON to Firebase immediately
                seg_tracking = result.get("tracking_data", [])
                if seg_tracking and video_key:
                    try:
                        seg_json = {
                            "segment": seg_idx,
                            "source_resolution": {"width": src_width, "height": src_height},
                            "source_fps": src_fps,
                            "frames_tracked": len(seg_tracking),
                            "time_range": {
                                "start_sec": seg_tracking[0].get("timestamp_sec"),
                                "end_sec": seg_tracking[-1].get("timestamp_sec"),
                            },
                            "model": "SAM2.1 Hiera Small",
                            "coordinate_description": "Midpoint of lower edge of bounding box, source resolution pixels",
                            "frames": seg_tracking,
                        }
                        seg_json_str = json.dumps(seg_json, indent=2)
                        seg_gcs_path = f"{OUTPUT_PREFIX}/{video_key}/segments/segment_{seg_idx:03d}.json"
                        # Write to temp file and upload
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                            tf.write(seg_json_str)
                            tf_path = tf.name
                        seg_url = upload_to_gcs(tf_path, seg_gcs_path)
                        os.unlink(tf_path)
                        print(f"  ✓ Segment {seg_idx} JSON uploaded ({len(seg_tracking)} frames)")
                    except Exception as e:
                        print(f"  ⚠ Segment JSON upload failed: {e}")

                # Spawn CPU render in parallel — GPU is free for next segment
                if result.get("status") not in ("both_lost", "handoff_failed"):
                    _seg = seg_idx  # capture for lambda
                    render_handle = render_segment.spawn(job_id, _seg, params)
                    render_tasks.append(render_handle)

                await ws.send_json({
                    "type": "segment_complete",
                    "segment": seg_idx,
                    "frames_processed": result.get("frames_processed", 0),
                    "warnings": result.get("warnings", []),
                })

                # Advance to next segment
                ho = result.get("handoff")
                if ho:
                    seed_p1_proc = ho["seed_p1_proc"]
                    seed_p2_proc = ho["seed_p2_proc"]
                    current_src_start = ho["next_src_start"]
                    prompt_local_idx = ho["prompt_local_idx"]
                    prev_handoff = ho.get("prev_handoff")  # box+pts for next segment
                else:
                    current_src_start += src_num
                    prompt_local_idx = 0
                    prev_handoff = None

            # ---- Wait for parallel renders to finish ----
            if render_tasks:
                await ws.send_json({"type": "status", "message": f"Rendering {len(render_tasks)} segments..."})
                for i, handle in enumerate(render_tasks):
                    try:
                        await asyncio.to_thread(handle.get)
                    except Exception as e:
                        print(f"  ⚠ Render task {i} failed: {e}")
                    await ws.send_json({
                        "type": "status",
                        "message": f"Rendered {i+1}/{len(render_tasks)} segments",
                    })

            # ---- Finalization ----
            await ws.send_json({"type": "status", "message": "Finalizing output..."})

            finalize_task = asyncio.get_event_loop().run_in_executor(
                None,
                lambda: finalize_and_upload.remote(
                    job_id, all_tracking_data, video_key
                )
            )
            elapsed_f = 0
            while True:
                done, _ = await asyncio.wait({finalize_task}, timeout=15.0)
                if done:
                    urls = finalize_task.result()
                    break
                elapsed_f += 15
                await ws.send_json({
                    "type": "status",
                    "message": f"Finalizing output... ({elapsed_f}s)",
                })

            await ws.send_json({
                "type": "complete",
                "json_url": urls.get("json_url"),
                "video_url": urls.get("video_url"),
                "total_segments": seg_idx,
                "total_frames_tracked": len(all_tracking_data),
            })

        except WebSocketDisconnect:
            print(f"Client disconnected: {job_id}")
        except Exception as e:
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except:
                pass

    return api
