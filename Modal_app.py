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
    .add_local_file(
        local_path=str(Path(__file__).parent / "court_reference_annotated.jpg"),
        remote_path="/app/court_reference_annotated.jpg",
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

        # ---- Load pre-extracted PTS timestamps for VFR-correct timing ----
        frame_timestamps = {}
        timestamps_path = job_dir / "frame_timestamps.json"
        try:
            import json as json_mod
            with open(str(timestamps_path), "r") as f:
                ts_data = json_mod.load(f)
            frame_timestamps = ts_data.get("timestamps", {})
            src_fps_from_ts = ts_data.get("fps", p["src_fps"])
            print(f"  ✓ Loaded {len(frame_timestamps)} PTS timestamps")
        except Exception as e:
            print(f"  ⚠ Could not load timestamps ({e}), using calculated")
            # Fallback to calculated timestamps
            frame_timestamps = {}

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
                # Use actual PTS timestamp if available, fall back to calculated
                timestamp = frame_timestamps.get(str(src_frame_idx), src_frame_idx / p["src_fps"])

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

                    # Get all mask properties
                    centroid = h["centroid"](m)
                    bbox = h["bbox"](m)
                    foot = h["foot"](m)
                    area = h["area"](m)
                    
                    if centroid and bbox and foot:
                        # Scale from proc resolution to source resolution
                        sx, sy = p["scale_x"], p["scale_y"]
                        entry[key] = {
                            # Centroid (center of mass)
                            "centroid": {
                                "x": int(round(centroid[0] * sx)),
                                "y": int(round(centroid[1] * sy)),
                            },
                            # Bounding box
                            "bbox": {
                                "x_min": int(round(bbox[0] * sx)),
                                "y_min": int(round(bbox[1] * sy)),
                                "x_max": int(round(bbox[2] * sx)),
                                "y_max": int(round(bbox[3] * sy)),
                            },
                            # Foot position (midpoint of lower bbox edge)
                            "foot_position": {
                                "x": int(round(foot[0] * sx)),
                                "y": int(round(foot[1] * sy)),
                            },
                            # Mask pixel count (scaled to source resolution)
                            "pixel_count": int(round(area * sx * sy)),
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
    """Call Gemini to find court landmarks, compute homography matrices for all surfaces, upload to Firebase.
    Runs in parallel with SAM2 tracking — only needs the first frame.
    
    Computes homographies for:
    - Floor plane (pixel → court meters)
    - Front wall (pixel → wall position/height)
    - Left wall (pixel → depth/height)
    - Right wall (pixel → depth/height)
    """
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

    # ---- Prepare image for Gemini (preserve aspect ratio) ----
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = PILImage.fromarray(frame_rgb)
    
    # Get actual image dimensions
    img_width, img_height = pil_frame.size
    print(f"  Original image dimensions: {img_width}x{img_height}")
    
    # Scale down if larger than 3072 (Gemini's max), preserving aspect ratio
    MAX_DIM = 3072
    if img_width > MAX_DIM or img_height > MAX_DIM:
        scale = MAX_DIM / max(img_width, img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        pil_frame = pil_frame.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
        img_width, img_height = new_width, new_height
        print(f"  Scaled to: {img_width}x{img_height}")

    import io
    buf = io.BytesIO()
    pil_frame.save(buf, format="JPEG", quality=95)
    frame_b64 = base64.b64encode(buf.getvalue()).decode()

    # ---- Load reference schematic ----
    with open("/app/squash_court_lines.jpeg", "rb") as f:
        schematic_b64 = base64.b64encode(f.read()).decode()

    # ---- Call Gemini API ----
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Required floor landmarks
    REQUIRED_FLOOR_KEYS = [
        "T_junction",
        "Left_Service_Box_Inner_Front",
        "Left_Service_Box_Inner_Back",
        "Right_Service_Box_Inner_Front",
        "Right_Service_Box_Inner_Back",
    ]
    
    # Optional floor landmarks (computed from lines or detected directly)
    OPTIONAL_FLOOR_KEYS = [
        "Front_Left_Floor_Point",
        "Front_Right_Floor_Point",
        "Left_Short_Point",
        "Right_Short_Point",
        "Left_Service_Box_Outer_Back",
        "Right_Service_Box_Outer_Back",
    ]
    
    # Front wall landmarks
    FRONT_WALL_KEYS = [
        "Left_Tin_Point",
        "Right_Tin_Point",
        "Left_Service_Point",
        "Right_Service_Point",
        "Front_Left_Out_Point",
        "Front_Right_Out_Point",
    ]
    
    # Line definitions
    LINE_KEYS = [
        "Front_Wall_Floor_Line",
        "Left_Wall_Floor_Line",
        "Right_Wall_Floor_Line",
    ]

    prompt = """
You are a computer vision expert analyzing a squash court image to extract precise landmark coordinates.

COORDINATE SYSTEM: Return all coordinates in normalized 0-1000 scale:
- (0, 0) = top-left corner of image
- (1000, 1000) = bottom-right corner of image

You are provided with TWO images:
1. IMAGE 1 (Schematic): A diagram showing standard squash court layout with labeled lines
2. IMAGE 2 (Target): The actual court photo to analyze

---

## STEP 1: IDENTIFY THE FOUR SURFACES

Look at IMAGE 2 and identify these four surfaces:
- FRONT WALL: The wall facing the camera, contains horizontal red lines (tin, service line, out line)
- LEFT WALL: The side wall on the left side of the image
- RIGHT WALL: The side wall on the right side of the image  
- FLOOR: The wooden playing surface with red court lines

---

## STEP 2: TRACE THE EDGE LINES WHERE SURFACES MEET

Find these three boundary lines and report TWO points along each:

A) Front_Wall_Floor_Line: The horizontal line where the front wall meets the floor
   - This is at the BASE of the front wall, where white wall meets wooden floor
   - Find the leftmost and rightmost visible points of this line

B) Left_Wall_Floor_Line: The diagonal line where the left wall meets the floor
   - Runs from front-left toward back-left of court
   - Find two points along this edge

C) Right_Wall_Floor_Line: The diagonal line where the right wall meets the floor
   - Runs from front-right toward back-right of court
   - Find two points along this edge

---

## STEP 3: FIND FRONT WALL CORNERS (from line intersections)

Calculate where the edge lines meet:
- Front_Left_Floor_Point = intersection of Front_Wall_Floor_Line and Left_Wall_Floor_Line
- Front_Right_Floor_Point = intersection of Front_Wall_Floor_Line and Right_Wall_Floor_Line

---

## STEP 4: FIND FLOOR COURT LINES

On the FLOOR, locate these RED painted lines:
- SHORT LINE: Horizontal red line running across the court (parallel to front wall)
- HALF-COURT LINE: Vertical red line running from short line toward camera
- SERVICE BOX LINES: The squares on either side of the half-court line

From these, identify:
- T_junction: Where short line meets half-court line (the "T")
- Left_Short_Point: Where short line meets left wall
- Right_Short_Point: Where short line meets right wall
- Left_Service_Box_Inner_Front: Inner corner of left service box on short line
- Left_Service_Box_Inner_Back: Inner corner of left service box toward camera
- Left_Service_Box_Outer_Back: Where service box back line meets left wall
- Right_Service_Box_Inner_Front: Inner corner of right service box on short line
- Right_Service_Box_Inner_Back: Inner corner of right service box toward camera
- Right_Service_Box_Outer_Back: Where service box back line meets right wall

---

## STEP 5: FIND FRONT WALL HORIZONTAL LINES

On the FRONT WALL, locate these horizontal RED lines (from bottom to top):
- TIN LINE: Lowest red line on front wall (at ~0.43m height)
- SERVICE LINE: Middle red line (at ~1.83m height)
- OUT LINE: Highest red line (at ~4.57m height, may extend onto side walls)

For each line, find the left and right endpoints:
- Left_Tin_Point, Right_Tin_Point
- Left_Service_Point, Right_Service_Point
- Front_Left_Out_Point, Front_Right_Out_Point

---

## OUTPUT FORMAT

Return ONLY valid JSON. No markdown, no explanation. Start with { and end with }.

{
  "edge_lines": {
    "Front_Wall_Floor_Line": [[x1, y1], [x2, y2]],
    "Left_Wall_Floor_Line": [[x1, y1], [x2, y2]],
    "Right_Wall_Floor_Line": [[x1, y1], [x2, y2]]
  },
  "landmarks": {
    "Front_Left_Floor_Point": [x, y],
    "Front_Right_Floor_Point": [x, y],
    "T_junction": [x, y],
    "Left_Service_Box_Inner_Front": [x, y],
    "Left_Service_Box_Inner_Back": [x, y],
    "Left_Service_Box_Outer_Back": [x, y],
    "Right_Service_Box_Inner_Front": [x, y],
    "Right_Service_Box_Inner_Back": [x, y],
    "Right_Service_Box_Outer_Back": [x, y],
    "Left_Short_Point": [x, y],
    "Right_Short_Point": [x, y],
    "Left_Tin_Point": [x, y],
    "Right_Tin_Point": [x, y],
    "Left_Service_Point": [x, y],
    "Right_Service_Point": [x, y],
    "Front_Left_Out_Point": [x, y],
    "Front_Right_Out_Point": [x, y]
  }
}
"""

    # Prepare image bytes for API call
    schematic_bytes = base64.b64decode(schematic_b64)
    frame_bytes = base64.b64decode(frame_b64)

    # Send schematic and target image only (no reference image)
    contents = [
        prompt,
        types.Part.from_bytes(data=schematic_bytes, mime_type="image/jpeg"),  # IMAGE 1: Schematic
        types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg"),       # IMAGE 2: Target
    ]

    generation_config = types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=16384,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    )

    # All point keys we might receive
    ALL_POINT_KEYS = REQUIRED_FLOOR_KEYS + OPTIONAL_FLOOR_KEYS + FRONT_WALL_KEYS

    def validate_gemini_response(text):
        """Parse and validate Gemini response. Returns (coords_dict, error_msg).
        
        Handles the new nested JSON format with 'landmarks' and 'edge_lines' keys,
        and flattens it for compatibility with downstream code.
        """
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
            json_match = re.search(r'\{[\s\S]*"landmarks"[\s\S]*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)

        # Fallback: try old format with T_junction at top level
        if not json_str:
            json_match = re.search(r'\{[\s\S]*"T_junction"[\s\S]*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)

        if not json_str:
            return None, f"No JSON found in response: {text[:200]}"

        try:
            raw_data = json_mod.loads(json_str)
        except json_mod.JSONDecodeError as e:
            return None, f"Invalid JSON: {e} — raw: {json_str[:200]}"

        # Handle new nested format: extract landmarks and edge_lines
        if "landmarks" in raw_data:
            # New format - flatten it
            coords = dict(raw_data.get("landmarks", {}))
            edge_lines = raw_data.get("edge_lines", {})
            
            # Copy edge_lines to coords for compatibility
            for key, val in edge_lines.items():
                coords[key] = val
            
            # Log reasoning if present
            reasoning = raw_data.get("reasoning", {})
            if reasoning:
                print(f"  Gemini reasoning: {list(reasoning.keys())}")
            
            # Check if Gemini returned image dimensions
            returned_dims = raw_data.get("image_dimensions", None)
            if returned_dims:
                print(f"  Gemini reports image dimensions: {returned_dims}")
        else:
            # Old format - use as-is
            coords = raw_data

        # Check required floor keys present
        missing = [k for k in REQUIRED_FLOOR_KEYS if k not in coords]
        if missing:
            return None, f"Missing required keys: {missing}"

        # Check line keys
        missing_lines = [k for k in LINE_KEYS if k not in coords]
        has_all_lines = len(missing_lines) == 0
        if missing_lines:
            print(f"  ⚠ Missing wall lines: {missing_lines}")

        # ---- Scale ALL coordinates from 0-1000 normalized to actual pixels ----
        # img_width/img_height = dimensions of image sent to Gemini (from outer scope)
        max_x, max_y = img_width, img_height
        scale_x = max_x / 1000.0
        scale_y = max_y / 1000.0
        
        # Points that can be skipped if out of bounds (may be off-screen in some camera angles)
        SKIPPABLE_POINTS = {
            "Left_Service_Box_Outer_Back",
            "Right_Service_Box_Outer_Back",
            "Left_Short_Point",
            "Right_Short_Point",
        }
        
        # Scale all point coordinates
        for key in ALL_POINT_KEYS:
            if key not in coords:
                continue
            val = coords[key]
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                return None, f"{key} is not [x, y]: {val}"
            x, y = val[0], val[1]
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return None, f"{key} has non-numeric values: {val}"
            
            # Scale from 0-1000 to actual pixels
            px = int(round(x * scale_x))
            py = int(round(y * scale_y))
            
            # Check bounds after scaling
            if 0 <= px <= max_x and 0 <= py <= max_y:
                coords[key] = [px, py]
            elif key in SKIPPABLE_POINTS:
                print(f"  ⚠ {key} out of bounds after scaling ({val} → [{px}, {py}]), skipping")
                del coords[key]
            else:
                return None, f"{key} out of range after scaling: {val} → [{px}, {py}]"

        # Scale all line coordinates
        for key in LINE_KEYS:
            if key not in coords:
                continue
            val = coords[key]
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                return None, f"{key} is not [[x1,y1],[x2,y2]]: {val}"
            scaled_line = []
            for i, pt in enumerate(val):
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    return None, f"{key}[{i}] is not [x, y]: {pt}"
                x, y = pt[0], pt[1]
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    return None, f"{key}[{i}] has non-numeric values: {pt}"
                px = int(round(x * scale_x))
                py = int(round(y * scale_y))
                scaled_line.append([px, py])
            coords[key] = scaled_line

        # Compute front corner intersections from lines if not already present
        if has_all_lines:
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

            # Compute front corners if not detected directly
            if "Front_Left_Floor_Point" not in coords:
                fl_corner = line_intersect(front_line[0], front_line[1], left_line[0], left_line[1])
                if fl_corner and 0 <= fl_corner[0] <= max_x and 0 <= fl_corner[1] <= max_y:
                    coords["Front_Left_Floor_Point"] = fl_corner
                    print(f"  ✓ Computed Front_Left_Floor_Point from lines: {fl_corner}")

            if "Front_Right_Floor_Point" not in coords:
                fr_corner = line_intersect(front_line[0], front_line[1], right_line[0], right_line[1])
                if fr_corner and 0 <= fr_corner[0] <= max_x and 0 <= fr_corner[1] <= max_y:
                    coords["Front_Right_Floor_Point"] = fr_corner
                    print(f"  ✓ Computed Front_Right_Floor_Point from lines: {fr_corner}")

        # Sanity check: T_junction shouldn't be at the very edge (2% margin)
        margin_x = int(max_x * 0.02)
        margin_y = int(max_y * 0.02)
        t = coords["T_junction"]
        if t[0] < margin_x or t[0] > max_x - margin_x or t[1] < margin_y or t[1] > max_y - margin_y:
            return None, f"T_junction at extreme edge ({t}), likely wrong"

        # Sanity check: Left points should be left of T, Right points should be right
        t_x = coords["T_junction"][0]
        tolerance = int(max_x * 0.05)  # 5% tolerance
        for key in ["Left_Service_Box_Inner_Front", "Left_Service_Box_Inner_Back"]:
            if coords[key][0] > t_x + tolerance:
                return None, f"{key} ({coords[key]}) is right of T_junction ({t_x}) — L/R swapped?"
        for key in ["Right_Service_Box_Inner_Front", "Right_Service_Box_Inner_Back"]:
            if coords[key][0] < t_x - tolerance:
                return None, f"{key} ({coords[key]}) is left of T_junction ({t_x}) — L/R swapped?"

        # Sanity check: Front points should be at similar Y (on the short line)
        front_y_diff = abs(coords["Left_Service_Box_Inner_Front"][1] - coords["Right_Service_Box_Inner_Front"][1])
        max_y_diff = int(max_y * 0.08)  # Allow 8% difference
        if front_y_diff > max_y_diff:
            return None, f"Front points Y differ by {front_y_diff}px — should be on same line"

        # Sanity check: Back points should be below (larger Y) than front points
        for side in ["Left", "Right"]:
            front_y = coords[f"{side}_Service_Box_Inner_Front"][1]
            back_y = coords[f"{side}_Service_Box_Inner_Back"][1]
            if back_y <= front_y:
                return None, f"{side} back ({back_y}) is above front ({front_y}) — should be below (closer to camera)"

        return coords, None

    # ---- Collect multiple successful API calls and pick the best ----
    NUM_SUCCESSFUL_CALLS = 5
    MAX_RETRIES = 50  # Total attempts allowed across all successful calls
    successful_responses = []  # List of (coords_dict, raw_coords_with_lines)
    last_error = None
    attempt = 0

    print(f"  Collecting {NUM_SUCCESSFUL_CALLS} successful Gemini responses...")
    
    while len(successful_responses) < NUM_SUCCESSFUL_CALLS and attempt < MAX_RETRIES:
        attempt += 1
        print(f"  → Gemini attempt {attempt}/{MAX_RETRIES} (success: {len(successful_responses)}/{NUM_SUCCESSFUL_CALLS})...")
        try:
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
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
            print(f"  ← Response ({len(response_text)} chars): {response_text[:300]}")

            coords, error = validate_gemini_response(response_text)
            if coords:
                resized = {k: v for k, v in coords.items() if k not in LINE_KEYS}
                successful_responses.append((resized, coords))
                print(f"  ✓ Valid response #{len(successful_responses)} on attempt {attempt}")
            else:
                last_error = error
                print(f"  ✗ Validation failed: {error}")
        except Exception as e:
            last_error = str(e)
            print(f"  ✗ Gemini API error: {e}")
            import time as _time
            _time.sleep(2)  # Brief pause before retry

    if len(successful_responses) == 0:
        print(f"  ✗ All {MAX_RETRIES} Gemini attempts failed. Last error: {last_error}")
        return {"status": "error", "message": f"Gemini failed after {MAX_RETRIES} attempts: {last_error}"}
    
    if len(successful_responses) < NUM_SUCCESSFUL_CALLS:
        print(f"  ⚠ Only got {len(successful_responses)}/{NUM_SUCCESSFUL_CALLS} successful responses, proceeding anyway")

    # ---- Scale coordinates from Gemini image dimensions → source resolution ----
    # img_width/img_height = dimensions of image sent to Gemini
    # src_width/src_height = original source video dimensions
    scale_x = src_width / img_width
    scale_y = src_height / img_height
    print(f"  Coordinate scaling: Gemini image ({img_width}x{img_height}) → Source ({src_width}x{src_height})")
    print(f"  Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
    
    def scale_coords(gemini_coords):
        """Scale coordinates from Gemini image dimensions to source resolution."""
        scaled = {}
        for name, val in gemini_coords.items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], (int, float)):
                scaled[name] = (int(round(val[0] * scale_x)), int(round(val[1] * scale_y)))
        return scaled

    # ---- Real-world coordinates for each surface ----
    
    # FLOOR: X = left-right (0-6.4m), Y = distance from front wall (0 at front, 9.75 at back)
    FLOOR_COORDS = {
        "Front_Left_Floor_Point": (0.0, 0.0),
        "Front_Right_Floor_Point": (6.4, 0.0),
        "T_junction": (3.2, 4.31),
        "Left_Service_Box_Inner_Front": (1.6, 4.31),
        "Left_Service_Box_Inner_Back": (1.6, 2.71),
        "Right_Service_Box_Inner_Front": (4.8, 4.31),
        "Right_Service_Box_Inner_Back": (4.8, 2.71),
        "Left_Short_Point": (0.0, 4.31),
        "Right_Short_Point": (6.4, 4.31),
    }

    # FRONT WALL: X = left-right (0-6.4m), Y = height from floor
    FRONT_WALL_COORDS = {
        "Front_Left_Floor_Point": (0.0, 0.0),
        "Front_Right_Floor_Point": (6.4, 0.0),
        "Left_Tin_Point": (0.0, 0.43),
        "Right_Tin_Point": (6.4, 0.43),
        "Left_Service_Point": (0.0, 1.83),
        "Right_Service_Point": (6.4, 1.83),
        "Front_Left_Out_Point": (0.0, 4.57),
        "Front_Right_Out_Point": (6.4, 4.57),
    }

    # LEFT WALL: X = depth from front wall (0 at front), Y = height from floor
    LEFT_WALL_COORDS = {
        "Front_Left_Floor_Point": (0.0, 0.0),
        "Left_Short_Point": (4.31, 0.0),
        "Left_Service_Box_Outer_Back": (2.71, 0.0),  # Service box back line meets left wall
        "Front_Left_Out_Point": (0.0, 4.57),
        "Left_Tin_Point": (0.0, 0.43),
        "Left_Service_Point": (0.0, 1.83),
    }

    # RIGHT WALL: X = depth from front wall (0 at front), Y = height from floor
    RIGHT_WALL_COORDS = {
        "Front_Right_Floor_Point": (0.0, 0.0),
        "Right_Short_Point": (4.31, 0.0),
        "Right_Service_Box_Outer_Back": (2.71, 0.0),  # Service box back line meets right wall
        "Front_Right_Out_Point": (0.0, 4.57),
        "Right_Tin_Point": (0.0, 0.43),
        "Right_Service_Point": (0.0, 1.83),
    }

    def compute_single_homography(pixel_coords, real_coords, surface_name, verbose=True):
        """Compute homography for a single surface. Returns dict with matrix and stats."""
        common_keys = sorted(set(pixel_coords.keys()) & set(real_coords.keys()))
        
        if verbose:
            print(f"  {surface_name} - matched points: {common_keys}")
        
        result = {
            "homography_matrix": None,
            "homography_matrix_inverse": None,
            "reprojection_error_meters": None,
            "status": "",
            "landmarks_used": common_keys,
            "num_points": len(common_keys),
        }
        
        if len(common_keys) < 4:
            result["status"] = f"insufficient_points ({len(common_keys)} < 4)"
            if verbose:
                print(f"  ⚠ {surface_name}: only {len(common_keys)} points — need ≥4")
            return result
        
        src_pts = np.array([pixel_coords[k] for k in common_keys], dtype=np.float32)
        dst_pts = np.array([real_coords[k] for k in common_keys], dtype=np.float32)
        
        if verbose:
            print(f"    Pixel coords: {src_pts.tolist()}")
            print(f"    Real coords:  {dst_pts.tolist()}")
        
        # Try RANSAC first
        H, h_status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # If RANSAC fails, try without RANSAC (regular least squares)
        if H is None or H.shape != (3, 3):
            if verbose:
                print(f"    RANSAC failed, trying least squares...")
            H, h_status = cv2.findHomography(src_pts, dst_pts, 0)  # 0 = regular method
        
        # If still failing, try LMEDS
        if H is None or H.shape != (3, 3):
            if verbose:
                print(f"    Least squares failed, trying LMEDS...")
            H, h_status = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
        
        if H is None or H.shape != (3, 3):
            result["status"] = "computation_failed"
            if verbose:
                print(f"  ✗ {surface_name}: all homography methods failed (H={H}, status={h_status})")
            return result
        
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            result["status"] = "singular_matrix"
            if verbose:
                print(f"  ✗ {surface_name}: singular matrix")
            return result
        
        # Reprojection error
        projected = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.sqrt(np.sum((projected - dst_pts) ** 2, axis=1))
        mean_error = float(np.mean(errors))
        
        result["homography_matrix"] = H.tolist()
        result["homography_matrix_inverse"] = H_inv.tolist()
        result["reprojection_error_meters"] = round(mean_error, 6)
        result["status"] = "success"
        
        if verbose:
            quality = "✓" if mean_error < 0.1 else "⚠" if mean_error < 0.3 else "✗"
            print(f"  {quality} {surface_name}: error={mean_error:.4f}m ({len(common_keys)} points)")
        
        return result

    # ---- Compute homography for each successful response and find the best ----
    print(f"\n  Evaluating {len(successful_responses)} responses to find best homography...")
    
    best_response_idx = -1
    best_floor_error = float('inf')
    all_floor_errors = []
    
    for idx, (resized_coords, raw_coords) in enumerate(successful_responses):
        scaled_coords = scale_coords(resized_coords)
        floor_result = compute_single_homography(scaled_coords, FLOOR_COORDS, f"Response {idx+1} Floor", verbose=False)
        
        floor_error = floor_result.get("reprojection_error_meters")
        if floor_error is not None:
            all_floor_errors.append((idx, floor_error))
            quality = "✓" if floor_error < 0.1 else "⚠" if floor_error < 0.3 else "✗"
            print(f"  {quality} Response {idx+1}: floor error = {floor_error:.4f}m")
            
            if floor_error < best_floor_error:
                best_floor_error = floor_error
                best_response_idx = idx
        else:
            print(f"  ✗ Response {idx+1}: floor homography failed ({floor_result.get('status')})")
            all_floor_errors.append((idx, None))
    
    if best_response_idx < 0:
        return {"status": "error", "message": "All responses failed to produce valid floor homography"}
    
    print(f"\n  ★ Best response: #{best_response_idx + 1} with floor error = {best_floor_error:.4f}m")
    
    # Use the best response
    resized_coords, raw_coords_with_lines = successful_responses[best_response_idx]
    scaled_coords = scale_coords(resized_coords)

    print(f"  ✓ Court landmarks (source res): {len(scaled_coords)} points")

    # ---- Compute all surface homographies for the best response ----
    print("  Computing all surface homographies for best response...")
    
    floor_result = compute_single_homography(scaled_coords, FLOOR_COORDS, "Floor")
    front_wall_result = compute_single_homography(scaled_coords, FRONT_WALL_COORDS, "Front Wall")
    left_wall_result = compute_single_homography(scaled_coords, LEFT_WALL_COORDS, "Left Wall")
    right_wall_result = compute_single_homography(scaled_coords, RIGHT_WALL_COORDS, "Right Wall")

    # ---- Compute pixel regions covered by each surface ----
    def compute_pixel_region(corner_keys, coords_dict, src_w, src_h):
        """
        Compute the pixel region (polygon + bounding box) for a surface.
        Returns dict with polygon vertices and bounding box, or None if insufficient points.
        """
        # Get available corner points
        points = []
        for key in corner_keys:
            if key in coords_dict:
                points.append(coords_dict[key])
        
        if len(points) < 3:
            return None
        
        # Convert to numpy for calculations
        pts = np.array(points, dtype=np.float32)
        
        # Compute convex hull to get proper polygon ordering
        try:
            hull = cv2.convexHull(pts)
            polygon = hull.reshape(-1, 2).tolist()
        except:
            polygon = points
        
        # Bounding box
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        bbox = {
            "x_min": int(max(0, min(x_coords))),
            "y_min": int(max(0, min(y_coords))),
            "x_max": int(min(src_w, max(x_coords))),
            "y_max": int(min(src_h, max(y_coords))),
        }
        
        # Compute area
        area = (bbox["x_max"] - bbox["x_min"]) * (bbox["y_max"] - bbox["y_min"])
        
        return {
            "polygon": [[int(p[0]), int(p[1])] for p in polygon],
            "bounding_box": bbox,
            "area_pixels": area,
        }

    # Define corner points for each surface region
    # Floor: quadrilateral from front corners to back of court
    floor_corners = ["Front_Left_Floor_Point", "Front_Right_Floor_Point", 
                     "Right_Service_Box_Inner_Back", "Left_Service_Box_Inner_Back"]
    # Extend floor to estimated back corners if we have service box points
    if "Left_Service_Box_Inner_Back" in scaled_coords and "Right_Service_Box_Inner_Back" in scaled_coords:
        # Estimate back corners by extrapolating from service box
        left_back = scaled_coords["Left_Service_Box_Inner_Back"]
        right_back = scaled_coords["Right_Service_Box_Inner_Back"]
        t_junction = scaled_coords.get("T_junction", (src_width // 2, 0))
        
        # Back of court is further down (larger Y) - estimate based on perspective
        # The service box back is at Y=2.71m, court back is at Y=9.75m
        # So we need to extrapolate ~3.6x further from front wall
        if "Front_Left_Floor_Point" in scaled_coords:
            front_y = scaled_coords["Front_Left_Floor_Point"][1]
            service_y = left_back[1]
            # Extrapolate to image bottom (or a reasonable estimate)
            back_y = min(src_height - 10, int(service_y + (service_y - front_y) * 1.5))
            
            # Estimate X spread at back (wider due to perspective)
            x_spread_service = right_back[0] - left_back[0]
            x_spread_back = int(x_spread_service * 1.3)
            center_x = (left_back[0] + right_back[0]) // 2
            
            estimated_back_left = (max(0, center_x - x_spread_back // 2), back_y)
            estimated_back_right = (min(src_width, center_x + x_spread_back // 2), back_y)
            
            floor_corners_extended = [
                scaled_coords.get("Front_Left_Floor_Point"),
                scaled_coords.get("Front_Right_Floor_Point"),
                estimated_back_right,
                estimated_back_left,
            ]
            floor_corners_extended = [p for p in floor_corners_extended if p is not None]
        else:
            floor_corners_extended = None
    else:
        floor_corners_extended = None

    # Front wall: quadrilateral from floor line to out line
    front_wall_corners = ["Front_Left_Floor_Point", "Front_Right_Floor_Point",
                          "Front_Right_Out_Point", "Front_Left_Out_Point"]
    
    # Left wall: from front corner along left edge
    left_wall_corners = ["Front_Left_Floor_Point", "Front_Left_Out_Point",
                         "Left_Service_Point", "Left_Tin_Point", "Left_Short_Point"]
    
    # Right wall: from front corner along right edge  
    right_wall_corners = ["Front_Right_Floor_Point", "Front_Right_Out_Point",
                          "Right_Service_Point", "Right_Tin_Point", "Right_Short_Point"]

    # Compute regions
    floor_region = None
    if floor_corners_extended:
        # Use extended corners for floor region
        temp_coords = {f"pt_{i}": p for i, p in enumerate(floor_corners_extended)}
        floor_region = compute_pixel_region([f"pt_{i}" for i in range(len(floor_corners_extended))], 
                                            temp_coords, src_width, src_height)
    if floor_region is None:
        floor_region = compute_pixel_region(floor_corners, scaled_coords, src_width, src_height)
    
    front_wall_region = compute_pixel_region(front_wall_corners, scaled_coords, src_width, src_height)
    left_wall_region = compute_pixel_region(left_wall_corners, scaled_coords, src_width, src_height)
    right_wall_region = compute_pixel_region(right_wall_corners, scaled_coords, src_width, src_height)

    print(f"  ✓ Pixel regions computed: floor={floor_region is not None}, front_wall={front_wall_region is not None}, "
          f"left_wall={left_wall_region is not None}, right_wall={right_wall_region is not None}")

    # ---- Build output JSON ----
    output = {
        "source_resolution": {"width": src_width, "height": src_height},
        "gemini_image_resolution": {"width": img_width, "height": img_height},
        "gemini_model": "gemini-3.1-pro-preview",
        "num_candidates_evaluated": len(successful_responses),
        "best_candidate_index": best_response_idx + 1,
        "court_landmarks_pixels": {k: list(v) for k, v in scaled_coords.items()},
        
        # Primary floor homography (backwards compatible)
        "homography_matrix": floor_result.get("homography_matrix"),
        "homography_matrix_inverse": floor_result.get("homography_matrix_inverse"),
        "reprojection_error_meters": floor_result.get("reprojection_error_meters"),
        "num_calibration_points": floor_result.get("num_points"),
        "court_landmarks_meters": {k: list(FLOOR_COORDS[k]) for k in floor_result.get("landmarks_used", [])},
        
        # All surface homographies with pixel regions
        "homographies": {
            "floor": {
                **floor_result,
                "coordinate_system": "X=left-right (0-6.4m), Y=depth from front wall (0-9.75m)",
                "real_world_coords": {k: list(FLOOR_COORDS[k]) for k in floor_result.get("landmarks_used", [])},
                "pixel_region": floor_region,
            },
            "front_wall": {
                **front_wall_result,
                "coordinate_system": "X=left-right (0-6.4m), Y=height from floor (0-4.57m)",
                "real_world_coords": {k: list(FRONT_WALL_COORDS[k]) for k in front_wall_result.get("landmarks_used", [])},
                "pixel_region": front_wall_region,
            },
            "left_wall": {
                **left_wall_result,
                "coordinate_system": "X=depth from front wall (0-9.75m), Y=height from floor",
                "real_world_coords": {k: list(LEFT_WALL_COORDS[k]) for k in left_wall_result.get("landmarks_used", [])},
                "pixel_region": left_wall_region,
            },
            "right_wall": {
                **right_wall_result,
                "coordinate_system": "X=depth from front wall (0-9.75m), Y=height from floor",
                "real_world_coords": {k: list(RIGHT_WALL_COORDS[k]) for k in right_wall_result.get("landmarks_used", [])},
                "pixel_region": right_wall_region,
            },
        },
    }

    # ---- Upload to Firebase ----
    import tempfile
    gcs_path = f"{OUTPUT_PREFIX}/{video_key}/homography.json"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json_mod.dump(output, tf, indent=2)
        tf_path = tf.name

    url = upload_to_gcs(tf_path, gcs_path)
    os.unlink(tf_path)
    print(f"  ✓ Multi-surface homography uploaded to {gcs_path}")

    return {
        "status": "ok", 
        "url": url, 
        "reprojection_error": floor_result.get("reprojection_error_meters"),
        "surfaces_computed": sum(1 for h in [floor_result, front_wall_result, left_wall_result, right_wall_result] 
                                  if h.get("status") == "success"),
    }


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
    volumes={DATA_DIR: scratch_vol},
    secrets=[gcs_secret],
    gpu="A10G",
    timeout=600,
)
def find_both_players_frame(job_id: str, start_frame: int = 0, max_frames: int = 300, step: int = 5):
    """
    Scan frames to find the first frame where both players are detected.
    Returns the frame number and timestamp where both players are first visible.
    
    Args:
        job_id: The job ID with the source video
        start_frame: Frame to start scanning from
        max_frames: Maximum frames to scan
        step: Check every Nth frame for speed
    
    Returns:
        {
            "found": bool,
            "frame": int,
            "timestamp_sec": float,
            "player_1_box": [x1, y1, x2, y2],
            "player_2_box": [x1, y1, x2, y2],
            "frames_scanned": int
        }
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO
    
    scratch_vol.reload()
    video_path = f"{DATA_DIR}/jobs/{job_id}/source.mp4"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ✗ Could not open video: {video_path}")
        return {"found": False, "error": "Could not open video"}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Scanning for both players: frames {start_frame} to {min(start_frame + max_frames, total_frames)}, step={step}")
    
    # Load YOLO model
    model = YOLO("yolov8n.pt")
    
    # Minimum area for a valid player detection (avoid small partial detections)
    MIN_PLAYER_AREA = 5000  # pixels
    
    frames_scanned = 0
    frame_idx = start_frame
    
    while frame_idx < min(start_frame + max_frames, total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            frame_idx += step
            continue
        
        frames_scanned += 1
        
        # Run YOLO person detection
        results = model(frame, classes=[0], conf=0.4, verbose=False)
        
        # Collect valid person detections
        persons = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                area = (x2 - x1) * (y2 - y1)
                
                if area >= MIN_PLAYER_AREA:
                    persons.append({
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": conf,
                        "area": area,
                        "center_x": (x1 + x2) / 2,
                    })
        
        # Check if we have at least 2 distinct people
        if len(persons) >= 2:
            # Sort by x-coordinate to identify left/right players
            persons.sort(key=lambda p: p["center_x"])
            
            # Check they are reasonably separated (not overlapping detections)
            p1, p2 = persons[0], persons[1]
            separation = abs(p1["center_x"] - p2["center_x"])
            
            # Players should be separated by at least 100 pixels
            if separation > 100:
                timestamp_sec = frame_idx / fps
                print(f"  ✓ Both players found at frame {frame_idx} ({timestamp_sec:.2f}s)")
                print(f"    Player 1: {p1['box']} (left)")
                print(f"    Player 2: {p2['box']} (right)")
                
                cap.release()
                return {
                    "found": True,
                    "frame": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "player_1_box": p1["box"],
                    "player_2_box": p2["box"],
                    "frames_scanned": frames_scanned,
                    "fps": fps,
                }
        
        frame_idx += step
    
    cap.release()
    print(f"  ✗ Could not find both players in {frames_scanned} frames scanned")
    return {
        "found": False,
        "frames_scanned": frames_scanned,
        "fps": fps,
    }


@app.function(
    image=yolo_image,
    volumes={DATA_DIR: scratch_vol, BALL_WEIGHTS_DIR: ball_weights_vol},
    secrets=[gcs_secret],
    gpu="A10G",
    timeout=1800,
)
def track_ball(job_id: str, video_key: str, src_fps: float, src_width: int, src_height: int, start_frame: int = 0):
    """Run YOLO ball detection on every frame of the source video.
    Uses batched inference + ffmpeg decode + FP16 for high GPU utilization.
    
    Args:
        start_frame: Frame to start tracking from (for trimmed videos). 
                     Frame indices in output will be relative to the original video.
    """
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
    
    # Calculate frames to process (from start_frame to end)
    frames_to_process = total_frames - start_frame
    if start_frame > 0:
        print(f"  Ball tracking: starting from frame {start_frame}, processing {frames_to_process} frames")

    # ---- Load pre-extracted PTS timestamps (handles VFR correctly) ----
    timestamps_path = f"{DATA_DIR}/jobs/{job_id}/frame_timestamps.json"
    try:
        with open(timestamps_path, "r") as f:
            ts_data = json_mod.load(f)
        frame_timestamps = ts_data.get("timestamps", {})
        print(f"  ✓ Loaded {len(frame_timestamps)} pre-extracted timestamps")
    except Exception as e:
        print(f"  ⚠ Could not load timestamps ({e}), extracting with ffprobe...")
        # Fallback: extract timestamps here
        pts_cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "frame=pts_time",
            "-of", "csv=p=0",
            video_path
        ]
        try:
            pts_result = subprocess.run(pts_cmd, capture_output=True, text=True, timeout=300)
            pts_lines = pts_result.stdout.strip().split('\n')
            frame_timestamps = {}
            for i, line in enumerate(pts_lines):
                line = line.strip()
                if line:
                    try:
                        frame_timestamps[str(i)] = float(line)
                    except ValueError:
                        frame_timestamps[str(i)] = i / fps
                else:
                    frame_timestamps[str(i)] = i / fps
            print(f"  ✓ Extracted {len(frame_timestamps)} frame timestamps")
        except Exception as e2:
            print(f"  ⚠ ffprobe failed ({e2}), using calculated timestamps")
            frame_timestamps = {str(i): i / fps for i in range(total_frames)}

    # ---- Decode frames with ffmpeg in background thread ----
    BATCH_SIZE = 32
    frame_queue = deque(maxlen=BATCH_SIZE * 3)  # buffer ahead
    decode_done = threading.Event()

    def decode_frames():
        """Decode frames via ffmpeg pipe — frame-accurate.

        -fps_mode passthrough emits source frames 1:1 (no dup/drop).

        Timing comes from ffmpeg's own `showinfo` filter, parsed off
        stderr in lockstep with stdout frames. Each emitted frame
        therefore carries its own PTS, so the timestamp written to JSON
        is the PTS of the exact frame we ran detection on — independent
        of how many frames ffprobe (used to build frame_timestamps.json)
        thought the stream had. That removes the cross-tool index
        alignment that was producing drift over long videos: any
        single-frame skip on either side used to shift every subsequent
        timestamp lookup, and the offset grew monotonically with time.
        """
        import re as _re
        cmd = [
            "ffmpeg", "-i", video_path,
            "-fps_mode", "passthrough",
            "-vf", "showinfo",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-loglevel", "info",
            "-"
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=width * height * 3 * 4,
        )

        # showinfo writes one line per emitted frame to stderr. Pair it
        # with stdout bytes by order — both are produced in lockstep per
        # frame by the filter graph.
        pts_buffer = deque()
        pts_lock = threading.Lock()
        pts_eof = threading.Event()
        pts_re = _re.compile(r"Parsed_showinfo.*?pts_time:(\S+)")

        def read_stderr():
            for raw_line in iter(proc.stderr.readline, b""):
                try:
                    line = raw_line.decode("utf-8", errors="replace")
                except Exception:
                    continue
                m = pts_re.search(line)
                if not m:
                    continue
                try:
                    pts = float(m.group(1))
                except ValueError:
                    continue
                with pts_lock:
                    pts_buffer.append(pts)
            pts_eof.set()

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()

        frame_size = width * height * 3
        idx = 0
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break

            # Pull the matching PTS for this frame in emission order. If
            # the parser falls behind, wait briefly. If ffmpeg never
            # emits a PTS for this frame, fall back to frame_timestamps
            # then idx/fps so we still produce a usable timestamp.
            ts = None
            for _ in range(2000):  # up to ~2 s
                with pts_lock:
                    if pts_buffer:
                        ts = pts_buffer.popleft()
                        break
                if pts_eof.is_set():
                    break
                _time.sleep(0.001)

            # Skip frames before start_frame
            if idx < start_frame:
                idx += 1
                continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            # idx is the actual frame number in the original video
            frame_queue.append((idx, ts, frame))
            idx += 1
            # Throttle if queue is full
            while len(frame_queue) >= BATCH_SIZE * 3:
                _time.sleep(0.001)
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.stderr.close()
        except Exception:
            pass
        proc.wait()
        stderr_thread.join(timeout=5)
        decode_done.set()

    decoder_thread = threading.Thread(target=decode_frames, daemon=True)
    decoder_thread.start()

    # Sanity check: source frame count from ffprobe should match what cv2 reports.
    # If they diverge significantly, idx/timestamp alignment is at risk.
    if frame_timestamps and abs(len(frame_timestamps) - total_frames) > 2:
        print(f"  ⚠ Frame count mismatch: ffprobe={len(frame_timestamps)} cv2={total_frames} — using ffprobe count")
        total_frames = len(frame_timestamps)
        frames_to_process = total_frames - start_frame

    # ---- Batched inference loop ----
    # Collect ALL candidate detections per frame (not just argmax). Final
    # per-frame selection happens after the loop so the stuck-ball filter
    # can use the full history to blacklist stationary blobs (e.g. a ball
    # wedged in the ceiling that YOLO confidently tracks every frame).
    raw_detections = {}
    frame_idx = 0
    start_time = _time.time()

    while True:
        # Collect a batch
        batch_frames = []
        batch_indices = []

        while len(batch_frames) < BATCH_SIZE:
            if frame_queue:
                idx, ts_raw, frame = frame_queue.popleft()
                batch_frames.append(frame)
                batch_indices.append((idx, ts_raw))
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

        # Process results — store ALL candidates per frame
        for i, result in enumerate(results):
            idx, ts_raw = batch_indices[i]
            # PTS comes directly from ffmpeg's showinfo for the frame we
            # actually ran detection on. Fall back to the ffprobe table,
            # then idx/fps, only if showinfo had nothing for this frame.
            if ts_raw is None:
                ts_raw = frame_timestamps.get(str(idx), idx / fps)
            ts = round(ts_raw, 4)
            boxes = result.boxes

            candidates = []
            if boxes is not None and len(boxes) > 0:
                xywh = boxes.xywh.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for j in range(len(confs)):
                    candidates.append((
                        float(xywh[j][0]),
                        float(xywh[j][1]),
                        float(confs[j]),
                    ))
            raw_detections[idx] = {"ts": ts, "candidates": candidates}

        frame_idx = batch_indices[-1][0]
        frames_processed = frame_idx - start_frame + 1
        if frames_processed % 500 < BATCH_SIZE:
            elapsed = _time.time() - start_time
            fps_actual = frames_processed / max(elapsed, 0.1)
            pct = round(frames_processed / frames_to_process * 100, 1)
            raw_detected = sum(1 for d in raw_detections.values() if d["candidates"])
            print(f"  Ball tracking: {frames_processed}/{frames_to_process} ({pct}%) — {raw_detected} raw detections — {fps_actual:.1f} fps")

    decoder_thread.join(timeout=10)
    elapsed = _time.time() - start_time

    # ---- Stuck-ball filter: blacklist stationary detection clusters ----
    # Conservative thresholds: a region is only blacklisted if a detection
    # persists there for ≥30% of frames-with-any-detection with positional
    # standard deviation ≤ 8 px on both axes.
    STUCK_BUCKET_PX = 30
    STUCK_MIN_FRAME_FRACTION = 0.30
    STUCK_MAX_POSITION_STD = 8.0
    STUCK_BLACKLIST_RADIUS = 25.0

    from collections import defaultdict
    bucket_points = defaultdict(list)
    frames_with_any = sum(1 for d in raw_detections.values() if d["candidates"])
    for d in raw_detections.values():
        for x, y, _ in d["candidates"]:
            bx = int(x // STUCK_BUCKET_PX)
            by = int(y // STUCK_BUCKET_PX)
            bucket_points[(bx, by)].append((x, y))

    threshold_count = max(1, int(STUCK_MIN_FRAME_FRACTION * frames_with_any))
    blacklist = []  # list of (cx, cy, r)
    for pts in bucket_points.values():
        if len(pts) < threshold_count:
            continue
        arr = np.array(pts)
        std_x = float(arr[:, 0].std())
        std_y = float(arr[:, 1].std())
        if std_x > STUCK_MAX_POSITION_STD or std_y > STUCK_MAX_POSITION_STD:
            continue
        cx = float(arr[:, 0].mean())
        cy = float(arr[:, 1].mean())
        # Merge into an existing nearby region instead of stacking overlaps
        merged = False
        for k, (ecx, ecy, er) in enumerate(blacklist):
            if (cx - ecx) ** 2 + (cy - ecy) ** 2 <= (er + STUCK_BLACKLIST_RADIUS) ** 2:
                blacklist[k] = ((ecx + cx) / 2, (ecy + cy) / 2,
                                max(er, STUCK_BLACKLIST_RADIUS))
                merged = True
                break
        if not merged:
            blacklist.append((cx, cy, STUCK_BLACKLIST_RADIUS))

    if blacklist:
        print(f"  ✓ Stuck-ball filter: blacklisted {len(blacklist)} stationary region(s):")
        for cx, cy, r in blacklist:
            print(f"      center=({cx:.0f},{cy:.0f}) radius={r:.0f}px")
    else:
        print(f"  ✓ Stuck-ball filter: no stationary blobs detected")

    def _is_blacklisted(x, y):
        for cx, cy, r in blacklist:
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                return True
        return False

    # ---- Pass 2: choose best non-blacklisted candidate per frame ----
    frames_result = {}
    detected = 0
    rejected_count = 0
    for idx in sorted(raw_detections.keys()):
        d = raw_detections[idx]
        ts = d["ts"]
        valid = [(x, y, c) for (x, y, c) in d["candidates"] if not _is_blacklisted(x, y)]
        rejected_count += len(d["candidates"]) - len(valid)
        if valid:
            x, y, conf = max(valid, key=lambda t: t[2])
            frames_result[str(idx)] = {
                "frame_number": idx,
                "timestamp_sec": ts,
                "detected": True,
                "pixel_x": round(x, 2),
                "pixel_y": round(y, 2),
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

    det_rate = round(detected / max(frames_to_process, 1) * 100, 2)
    print(f"  ✓ Ball tracking complete: {len(frames_result)} frames, {detected} detections "
          f"({det_rate}%), {rejected_count} candidates filtered, took {elapsed:.1f}s")

    # ---- Build output JSON ----
    output = {
        "source_resolution": {"width": width, "height": height},
        "source_fps": fps,
        "total_frames": total_frames,
        "start_frame": start_frame,
        "frames_processed": len(frames_result),
        "frames_with_detection": detected,
        "detection_rate_pct": det_rate,
        "model": "YOLOv8 squash_v1",
        "confidence_threshold": 0.5,
        "stuck_ball_filter": {
            "bucket_size_px": STUCK_BUCKET_PX,
            "min_frame_fraction": STUCK_MIN_FRAME_FRACTION,
            "max_position_std_px": STUCK_MAX_POSITION_STD,
            "blacklist_radius_px": STUCK_BLACKLIST_RADIUS,
            "blacklisted_regions": [
                {"center_x": round(cx, 1), "center_y": round(cy, 1), "radius": round(r, 1)}
                for cx, cy, r in blacklist
            ],
            "candidates_rejected": rejected_count,
        },
        "coordinate_description": "Midpoint of bounding box, source resolution pixels. Frame numbers are relative to original video.",
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
    import subprocess
    import cv2
    
    local_path = f"{DATA_DIR}/jobs/{job_id}/source.mp4"
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    download_from_gcs(gcs_path, local_path)
    
    # Extract PTS timestamps for all frames (handles VFR correctly)
    # This is used by both ball tracking and player tracking for consistency
    print(f"  Extracting frame PTS timestamps...")
    pts_cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "frame=pts_time",
        "-of", "csv=p=0",
        local_path
    ]
    
    # Get video FPS for fallback
    cap = cv2.VideoCapture(local_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    frame_timestamps = {}
    try:
        pts_result = subprocess.run(pts_cmd, capture_output=True, text=True, timeout=300)
        pts_lines = pts_result.stdout.strip().split('\n')
        
        for i, line in enumerate(pts_lines):
            line = line.strip()
            if line:
                try:
                    frame_timestamps[str(i)] = float(line)
                except ValueError:
                    frame_timestamps[str(i)] = i / fps
            else:
                frame_timestamps[str(i)] = i / fps
        
        # Check for VFR
        if len(frame_timestamps) > 10:
            timestamps = [frame_timestamps[str(i)] for i in range(min(100, len(frame_timestamps)))]
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            max_deviation = max(abs(iv - avg_interval) for iv in intervals)
            
            if max_deviation > 0.005:
                print(f"  ⚠ VFR detected: max deviation={max_deviation*1000:.2f}ms from avg")
            else:
                print(f"  ✓ CFR video: consistent {1/avg_interval:.2f} fps")
                
        print(f"  ✓ Extracted {len(frame_timestamps)} frame timestamps")
        
    except Exception as e:
        print(f"  ⚠ ffprobe failed ({e}), using calculated timestamps")
        frame_timestamps = {str(i): i / fps for i in range(total_frames)}
    
    # Save timestamps to volume for use by other functions
    timestamps_path = f"{DATA_DIR}/jobs/{job_id}/frame_timestamps.json"
    with open(timestamps_path, "w") as f:
        json.dump({
            "fps": fps,
            "total_frames": total_frames,
            "timestamps": frame_timestamps,
        }, f)
    
    scratch_vol.commit()
    return {"status": "ok", "job_id": job_id}


@app.function(
    image=web_image,
    volumes={DATA_DIR: scratch_vol},
    timeout=600,
)
def cleanup_old_jobs(max_age_hours: int = 24):
    """Clean up old job directories to free inodes.
    Call this periodically or when inode usage is high."""
    import time
    import shutil
    
    jobs_dir = Path(f"{DATA_DIR}/jobs")
    if not jobs_dir.exists():
        return {"status": "no_jobs_dir"}
    
    now = time.time()
    max_age_sec = max_age_hours * 3600
    
    deleted = 0
    freed_files = 0
    errors = []
    
    scratch_vol.reload()
    
    for job_dir in jobs_dir.iterdir():
        if not job_dir.is_dir():
            continue
        
        try:
            # Check age based on directory mtime
            mtime = job_dir.stat().st_mtime
            age_hours = (now - mtime) / 3600
            
            if age_hours > max_age_hours:
                # Count files before deletion
                file_count = sum(1 for _ in job_dir.rglob("*") if _.is_file())
                
                shutil.rmtree(str(job_dir), ignore_errors=True)
                deleted += 1
                freed_files += file_count
                print(f"  Cleaned up {job_dir.name} ({file_count} files, {age_hours:.1f}h old)")
        except Exception as e:
            errors.append(f"{job_dir.name}: {e}")
    
    try:
        scratch_vol.commit()
    except:
        pass
    
    print(f"✓ Cleanup complete: {deleted} jobs, {freed_files} files freed")
    return {
        "status": "ok",
        "jobs_deleted": deleted,
        "files_freed": freed_files,
        "errors": errors,
    }


@app.function(
    image=web_image,
    volumes={DATA_DIR: scratch_vol},
    secrets=[gcs_secret],
    cpu=8,
    memory=16384,
    timeout=1800,
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

    # Single reload at start to sync volume state from TrackerGPU
    try:
        scratch_vol.reload()
    except Exception as e:
        print(f"  ⚠ Render seg {seg_idx}: initial reload failed: {e}, continuing anyway...")

    # Wait for masks to be visible (TrackerGPU commits before spawning render)
    frame_map = None
    for sync_attempt in range(15):
        frame_map_path = masks_dir / "frame_map.json"
        if frame_map_path.exists():
            try:
                with open(str(frame_map_path), "r") as f:
                    frame_map = json.load(f)
                if frame_map:
                    break
            except:
                pass
        if sync_attempt < 14:
            print(f"  ⏳ Render seg {seg_idx}: waiting for masks (attempt {sync_attempt + 1}/15)...")
            _time.sleep(1)
            # Retry reload on later attempts
            if sync_attempt in [4, 9]:
                try:
                    scratch_vol.reload()
                    print(f"  ↻ Render seg {seg_idx}: volume reloaded")
                except:
                    pass
            _time.sleep(1)

    if not frame_map:
        print(f"  ⚠ No frame_map for segment {seg_idx} after 15 retries")
        return {"status": "no_masks"}

    sorted_entries = sorted(frame_map.keys(), key=int)
    total_frames = len(sorted_entries)

    if not (frames_dir).exists():
        for sync_attempt in range(5):
            if (frames_dir).exists():
                break
            print(f"  ⏳ Render seg {seg_idx}: waiting for frame JPEGs (attempt {sync_attempt + 1}/5)...")
            _time.sleep(1)

    # ---- Bulk copy to local /tmp using parallel threads ----
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    local_tmp = Path(f"/tmp/render_{job_id}_{seg_idx}")
    local_masks = local_tmp / "masks"
    local_frames = local_tmp / "frames"
    local_masks.mkdir(parents=True, exist_ok=True)
    local_frames.mkdir(parents=True, exist_ok=True)

    copy_start = _time.time()
    print(f"  Render seg {seg_idx}: copying {total_frames} frames + masks to local disk (parallel)...")

    def copy_file(src_dst):
        src, dst = src_dst
        if src.exists():
            shutil.copy2(str(src), str(dst))
            return True
        return False

    # Build list of all files to copy
    copy_tasks = []
    for local_idx_str in sorted_entries:
        local_idx = int(local_idx_str)
        # Masks
        for suffix in [f"{local_idx:05d}_p1.npy", f"{local_idx:05d}_p2.npy"]:
            copy_tasks.append((masks_dir / suffix, local_masks / suffix))
        # JPEG
        jpg_name = f"{local_idx:05d}.jpg"
        copy_tasks.append((frames_dir / jpg_name, local_frames / jpg_name))

    # Copy in parallel with 16 threads
    mask_count = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(copy_file, copy_tasks))
        mask_count = sum(1 for r in results if r)

    copy_time = _time.time() - copy_start
    print(f"  Render seg {seg_idx}: copied {mask_count} files in {copy_time:.1f}s ({mask_count/max(copy_time,0.1):.0f} files/s)")

    # ---- Render from local disk ----
    COLOR_P1 = (0, 0, 255)
    COLOR_P2 = (255, 0, 0)

    # Render at lower fps for speed (every Nth frame)
    render_fps = min(5, p["target_fps"])
    frame_skip = max(1, p["target_fps"] // render_fps)
    render_entries = sorted_entries[::frame_skip]
    total_render = len(render_entries)
    print(f"  Render seg {seg_idx}: {total_render}/{total_frames} frames at {render_fps}fps (skip {frame_skip})")

    # Single-pass H.264 via ffmpeg pipe with multi-threading
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
        "-preset", "ultrafast",
        "-threads", "8",
        "-crf", "26",
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

    # ---- Save masks to GCS (compressed NPZ for smaller file size) ----
    try:
        p1_masks = {}
        p2_masks = {}
        
        for local_idx_str in sorted_entries:
            local_idx = int(local_idx_str)
            src_frame = int(frame_map[local_idx_str])
            
            p1_path = local_masks / f"{local_idx:05d}_p1.npy"
            p2_path = local_masks / f"{local_idx:05d}_p2.npy"
            
            if p1_path.exists():
                p1_masks[f"frame_{src_frame:06d}"] = np.load(str(p1_path))
            if p2_path.exists():
                p2_masks[f"frame_{src_frame:06d}"] = np.load(str(p2_path))
        
        print(f"  Masks collected: P1={len(p1_masks)} P2={len(p2_masks)} frames")
        
        # Save as compressed NPZ (binary masks compress very well ~10:1)
        npz_local = local_tmp / f"masks_seg_{seg_idx:04d}.npz"
        npz_start = _time.time()
        np.savez_compressed(
            str(npz_local),
            p1_masks=p1_masks,
            p2_masks=p2_masks,
            frame_map=frame_map,
            proc_width=p.get("proc_width", 960),
            proc_height=p.get("proc_height", 540),
        )
        npz_time = _time.time() - npz_start
        
        npz_size_mb = npz_local.stat().st_size / (1024 * 1024)
        print(f"  NPZ created: {npz_local.name} ({npz_size_mb:.2f} MB) in {npz_time:.1f}s")
        
        # Upload to GCS
        upload_start = _time.time()
        video_key = p.get("video_key", job_id)
        gcs_masks_path = f"{OUTPUT_PREFIX}/{video_key}/masks/segment_{seg_idx:04d}.npz"
        masks_url = upload_to_gcs(str(npz_local), gcs_masks_path)
        upload_time = _time.time() - upload_start
        print(f"  ✓ Masks uploaded to GCS in {upload_time:.1f}s: {gcs_masks_path}")
        
    except Exception as e:
        import traceback
        print(f"  ⚠ Mask save failed: {e}")
        print(f"  Traceback: {traceback.format_exc()}")

    # Copy final output back to volume
    final_vol_path = str(out_dir / f"out_{seg_idx:04d}.mp4")
    shutil.copy2(local_out_path, final_vol_path)

    # Clean up volume masks/frames to free inodes
    shutil.rmtree(str(masks_dir), ignore_errors=True)
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    
    # Clean up local tmp
    shutil.rmtree(str(local_tmp), ignore_errors=True)

    # Best-effort commit
    try:
        scratch_vol.commit()
    except Exception as e:
        print(f"  ⚠ Render seg {seg_idx}: commit skipped: {e}")
    
    print(f"  ✓ Segment {seg_idx} rendered ({frames_written} frames)")
    return {"status": "ok", "seg_idx": seg_idx}


# ====================================================================
# Bounce and Hit Detection
# ====================================================================

def detect_bounces_and_hits(ball_data: dict, homography_data: dict) -> dict:
    """
    Detect ball bounces and player hits from ball tracking data.
    
    Uses structural analysis:
    - Bounces: Recursive binary segmentation of ball trajectory
    - Hits: FW→FW structural search (between front wall bounces, find the hit)
    
    Args:
        ball_data: Ball tracking JSON with frames dict
        homography_data: Homography JSON with surface matrices
        
    Returns:
        Dict with events list and metadata
    """
    import numpy as np
    from collections import Counter
    
    # ═══════════════════════════════════════════════════════════════════
    # Configuration
    # ═══════════════════════════════════════════════════════════════════
    
    # Bounce detection
    RESIDUAL_THRESH = 5.0       # Min residual to consider splitting
    MIN_SEGMENT_POINTS = 4      # Min points to attempt split
    IMPROVEMENT_THRESH = 1.0    # Min improvement to accept split
    CLUSTER_WINDOW = 3          # Frames to merge nearby events
    
    # Gap detector
    GAP_FRAMES = 15             # Frame gap threshold
    GAP_SPEED = 150             # Speed threshold for gap
    ANGLE_CHANGE_MIN = 10       # Degrees
    SPEED_DIFF_MIN = 3          # px/frame
    
    # Launch detector
    LAUNCH_SLOW = 4             # Speed before launch
    LAUNCH_FAST = 15            # Speed after launch
    
    # Front wall detector
    FW_PY_MAX = 350             # py threshold for front wall
    FW_MIN_SPEED = 3            # Minimum speed
    FW_MAX_GAP = 5              # Max frame gap
    
    # Hit detection
    ACCEL_STRONG = 2.5          # Speed ratio for confident hit
    ACCEL_WEAK = 1.5            # Speed ratio for fallback
    VOLLEY_ACCEL = 5.0          # Speed ratio for volleys
    MIN_GAP_FRAMES = 3          # Min gap for tracking gap method
    PY_MIN_HIT = 350            # Ball must be at py > this for hit
    PY_WALL_CUTOFF = 300        # Ball at py < this is on wall
    PY_REVERSAL_MIN = 5         # Min py trend change
    HIT_DEDUP_WINDOW = 8        # Min frames between hits
    MAX_FRAME_GAP = 5           # Max gap for speed calc
    
    # Court bounds
    WALL_BOUNDS = (-0.5, 7.0, -0.5, 5.0)
    FLOOR_BOUNDS = (-0.5, 7.0, -0.5, 10.5)
    
    # ═══════════════════════════════════════════════════════════════════
    # Load homography matrices
    # ═══════════════════════════════════════════════════════════════════
    
    homographies = homography_data.get("homographies", {})
    
    H_FW = np.array(homographies.get("front_wall", {}).get("homography_matrix", np.eye(3)))
    H_FL = np.array(homographies.get("floor", {}).get("homography_matrix", np.eye(3)))
    H_LW = np.array(homographies.get("left_wall", {}).get("homography_matrix", np.eye(3)))
    H_RW = np.array(homographies.get("right_wall", {}).get("homography_matrix", np.eye(3)))
    
    # ═══════════════════════════════════════════════════════════════════
    # Utility functions
    # ═══════════════════════════════════════════════════════════════════
    
    def project(H, px, py):
        """Project pixel coordinates through homography."""
        r = H @ np.array([px, py, 1.0])
        if abs(r[2]) < 1e-10:
            return None, None
        return r[0] / r[2], r[1] / r[2]
    
    def in_bounds(x, y, bounds):
        return x is not None and bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]
    
    def get_surface(px, py):
        """Determine which court surface a pixel maps to."""
        results = {}
        for name, H, bounds in [
            ('FW', H_FW, WALL_BOUNDS),
            ('LW', H_LW, WALL_BOUNDS),
            ('RW', H_RW, WALL_BOUNDS),
            ('FL', H_FL, FLOOR_BOUNDS),
        ]:
            x, y = project(H, px, py)
            if in_bounds(x, y, bounds):
                results[name] = (x, y)
        
        if not results:
            return 'NONE'
        
        # Disambiguate by py
        if py < 350:
            for w in ['FW', 'LW', 'RW']:
                if w in results:
                    return w
        elif py > 450:
            if 'FL' in results:
                return 'FL'
        
        if py < 400:
            for w in ['FW', 'LW', 'RW']:
                if w in results:
                    return w
        if 'FL' in results:
            return 'FL'
        return list(results.keys())[0]
    
    def fit_quadratic(xs, ys):
        """Fit quadratic and return coeffs + residual."""
        if len(xs) < 3:
            return None, float('inf')
        try:
            coeffs = np.polyfit(xs, ys, 2)
            fitted = np.polyval(coeffs, xs)
            residual = np.max(np.abs(ys - fitted))
            return coeffs, residual
        except:
            return None, float('inf')
    
    # ═══════════════════════════════════════════════════════════════════
    # Load and preprocess ball tracking
    # ═══════════════════════════════════════════════════════════════════
    
    frames_data = ball_data.get('frames', {})
    if not frames_data:
        return {"events": [], "error": "No ball tracking frames"}
    
    # Build detection list
    detections_raw = []
    for k, v in frames_data.items():
        if v.get('detected'):
            detections_raw.append((
                int(k),
                v.get('pixel_x', v.get('x', 0)),
                v.get('pixel_y', v.get('y', 0)),
                v.get('confidence', 0.5),
                v.get('timestamp_sec', int(k) / 30.0)
            ))
    
    detections_raw.sort(key=lambda x: x[0])
    
    if len(detections_raw) < 10:
        return {"events": [], "error": "Too few ball detections"}
    
    # Ghost filtering
    positions = [(round(d[1]), round(d[2])) for d in detections_raw]
    if positions:
        ghost_pos, ghost_count = Counter(positions).most_common(1)[0]
        gx, gy = ghost_pos
        
        if ghost_count / len(detections_raw) >= 0.05:
            detections_raw = [
                d for d in detections_raw
                if abs(d[1] - gx) > 30 or abs(d[2] - gy) > 30
            ]
    
    # Build detection dict
    detections = {}
    for f, x, y, c, t in detections_raw:
        detections[f] = (x, y, t, c)
    
    all_frames = sorted(detections.keys())
    
    print(f"    Ball detections after ghost filter: {len(all_frames)}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Stage 1: Bounce Detection
    # ═══════════════════════════════════════════════════════════════════
    
    bounce_candidates = []
    
    # 1.1 Recursive binary segmentation
    def segment_bounces(frames_subset):
        """Recursively find bounce points by curve fitting."""
        if len(frames_subset) < MIN_SEGMENT_POINTS:
            return []
        
        xs = np.array([detections[f][0] for f in frames_subset])
        ys = np.array([detections[f][1] for f in frames_subset])
        ts = np.array(frames_subset, dtype=float)
        
        _, residual = fit_quadratic(ts, ys)
        
        if residual <= RESIDUAL_THRESH:
            return []
        
        # Try splitting at each point
        best_split = None
        best_improvement = 0
        
        for i in range(2, len(frames_subset) - 2):
            left_frames = frames_subset[:i]
            right_frames = frames_subset[i:]
            
            _, res_left = fit_quadratic(
                np.array(left_frames, dtype=float),
                np.array([detections[f][1] for f in left_frames])
            )
            _, res_right = fit_quadratic(
                np.array(right_frames, dtype=float),
                np.array([detections[f][1] for f in right_frames])
            )
            
            combined_res = max(res_left, res_right)
            improvement = residual - combined_res
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_split = i
        
        if best_split and best_improvement > IMPROVEMENT_THRESH:
            split_frame = frames_subset[best_split]
            left_bounces = segment_bounces(frames_subset[:best_split])
            right_bounces = segment_bounces(frames_subset[best_split:])
            return left_bounces + [split_frame] + right_bounces
        
        return []
    
    # Find continuous segments
    segments = []
    seg_start = 0
    for i in range(1, len(all_frames)):
        if all_frames[i] - all_frames[i-1] > 5:
            if i - seg_start >= MIN_SEGMENT_POINTS:
                segments.append(all_frames[seg_start:i])
            seg_start = i
    if len(all_frames) - seg_start >= MIN_SEGMENT_POINTS:
        segments.append(all_frames[seg_start:])
    
    # Run segmentation on each continuous segment
    for seg in segments:
        bounces = segment_bounces(seg)
        for b in bounces:
            bounce_candidates.append((b, 'segment', 1.0))
    
    # 1.2 Gap detector
    for i in range(1, len(all_frames)):
        f1, f2 = all_frames[i-1], all_frames[i]
        gap = f2 - f1
        
        if gap > GAP_FRAMES:
            bounce_candidates.append((f1, 'gap_end', 0.8))
            bounce_candidates.append((f2, 'gap_start', 0.8))
    
    # 1.3 Front wall detector (py minima)
    for i in range(2, len(all_frames) - 2):
        f = all_frames[i]
        py = detections[f][1]
        
        if py > FW_PY_MAX:
            continue
        
        # Check if local minimum
        py_before = detections[all_frames[i-1]][1]
        py_after = detections[all_frames[i+1]][1]
        
        if py < py_before and py < py_after:
            # Check speed
            gap1 = f - all_frames[i-1]
            gap2 = all_frames[i+1] - f
            if gap1 <= FW_MAX_GAP and gap2 <= FW_MAX_GAP:
                d1 = np.hypot(
                    detections[f][0] - detections[all_frames[i-1]][0],
                    detections[f][1] - detections[all_frames[i-1]][1]
                )
                if d1 / gap1 >= FW_MIN_SPEED:
                    bounce_candidates.append((f, 'fw_min', 0.9))
    
    # 1.4 Cluster and deduplicate
    bounce_candidates.sort(key=lambda x: x[0])
    clustered = []
    
    i = 0
    while i < len(bounce_candidates):
        cluster = [bounce_candidates[i]]
        j = i + 1
        while j < len(bounce_candidates) and bounce_candidates[j][0] - cluster[0][0] <= CLUSTER_WINDOW:
            cluster.append(bounce_candidates[j])
            j += 1
        
        # Keep the one with highest score
        best = max(cluster, key=lambda x: x[2])
        clustered.append(best)
        i = j
    
    # 1.5 Surface classification and physics filter
    bounces = []
    for frame, method, score in clustered:
        if frame not in detections:
            continue
        px, py, ts, conf = detections[frame]
        surface = get_surface(px, py)
        bounces.append({
            "frame": frame,
            "timestamp_sec": round(ts, 3),
            "pixel_x": round(px, 1),
            "pixel_y": round(py, 1),
            "surface": surface,
            "method": method,
            "score": round(score, 2),
        })
    
    # Wall dedup: max 1 per wall per 30 frames
    final_bounces = []
    wall_last = {"FW": -999, "LW": -999, "RW": -999}
    
    for b in bounces:
        surf = b["surface"]
        if surf in wall_last:
            if b["frame"] - wall_last[surf] < 30:
                continue
            wall_last[surf] = b["frame"]
        final_bounces.append(b)
    
    print(f"    Bounces detected: {len(final_bounces)}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Stage 2: Hit Detection
    # ═══════════════════════════════════════════════════════════════════
    
    def find_best_hit(start_f, end_f):
        """Find best hit candidate in frame range."""
        frames = [f for f in all_frames if start_f <= f <= end_f]
        if len(frames) < 3:
            return None, '', 0
        
        # Compute speeds
        speed_pairs = []
        for k in range(1, len(frames)):
            f1, f2 = frames[k-1], frames[k]
            gap = f2 - f1
            if gap <= MAX_FRAME_GAP:
                spd = np.hypot(
                    detections[f2][0] - detections[f1][0],
                    detections[f2][1] - detections[f1][1]
                ) / gap
            else:
                spd = 0
            speed_pairs.append((f1, f2, spd, gap))
        
        # Method 1: Speed explosion
        best_expl_f = None
        best_ratio = 0
        for k in range(1, len(speed_pairs)):
            _, f1, s0, g0 = speed_pairs[k-1]
            _, f2, s1, g1 = speed_pairs[k]
            if g0 > MAX_FRAME_GAP or g1 > MAX_FRAME_GAP:
                continue
            if s0 < 0.5:
                continue
            if detections[f1][1] < PY_WALL_CUTOFF:
                continue
            ratio = s1 / s0
            if ratio > best_ratio:
                best_ratio = ratio
                best_expl_f = f1
        
        # Method 2: Tracking gap
        best_gap_f = None
        best_gap = 0
        for f1, f2, s, gap in speed_pairs:
            if gap >= MIN_GAP_FRAMES and gap > best_gap:
                if detections[f1][1] > PY_WALL_CUTOFF:
                    best_gap = gap
                    best_gap_f = f1
        
        # Method 3: py reversal
        best_rev_f = None
        best_rev_score = 0
        for k in range(2, len(frames) - 2):
            f = frames[k]
            if detections[f][1] < PY_MIN_HIT:
                continue
            py_before = [detections[frames[j]][1] for j in range(max(0, k-3), k)]
            py_after = [detections[frames[j]][1] for j in range(k+1, min(len(frames), k+4))]
            if len(py_before) < 2 or len(py_after) < 2:
                continue
            trend_before = py_before[-1] - py_before[0]
            trend_after = py_after[-1] - py_after[0]
            if trend_before > PY_REVERSAL_MIN and trend_after < -PY_REVERSAL_MIN:
                score = abs(trend_before) + abs(trend_after)
                if score > best_rev_score:
                    best_rev_score = score
                    best_rev_f = f
        
        # Priority selection
        if best_expl_f and best_ratio > ACCEL_STRONG:
            return best_expl_f, 'accel', best_ratio
        if best_gap_f and best_gap >= MIN_GAP_FRAMES:
            return best_gap_f, 'gap', best_gap
        if best_rev_f:
            return best_rev_f, 'py_rev', best_rev_score
        if best_expl_f and best_ratio > ACCEL_WEAK:
            return best_expl_f, 'weak_accel', best_ratio
        return None, '', 0
    
    # Build FW bounce list
    fw_frames = sorted([b["frame"] for b in final_bounces if b["surface"] == "FW"])
    bounce_frames_set = set(b["frame"] for b in final_bounces)
    
    hits_raw = []
    
    # FW→FW structural search
    for i in range(len(fw_frames)):
        fw2 = fw_frames[i]
        fw1 = fw_frames[i-1] if i > 0 else 0
        
        # Find non-FW bounces between
        non_fw = [b["frame"] for b in final_bounces 
                  if fw1 < b["frame"] < fw2 and b["surface"] != "FW"]
        
        if non_fw:
            hit_f, method, score = find_best_hit(fw1 + 1, fw2 - 1)
            if hit_f:
                hits_raw.append((hit_f, fw2, method, score))
        elif fw2 - fw1 < 100:
            hit_f, method, score = find_best_hit(fw1 + 1, fw2 - 1)
            if hit_f and method == 'accel' and score > VOLLEY_ACCEL:
                hits_raw.append((hit_f, fw2, 'volley', score))
    
    # Serve detection
    if fw_frames:
        first_fw = fw_frames[0]
        hit_f, method, score = find_best_hit(0, first_fw - 1)
        if hit_f:
            hits_raw.append((hit_f, first_fw, f'serve_{method}', score))
    
    # Deduplicate
    hits_raw.sort(key=lambda x: x[0])
    hits = []
    used = set()
    
    for hit_f, fw_f, method, score in hits_raw:
        if any(abs(hit_f - u) <= HIT_DEDUP_WINDOW for u in used):
            continue
        
        # Avoid hit on bounce frame
        if hit_f in bounce_frames_set:
            for offset in [-1, 1, -2, 2, -3, 3]:
                candidate = hit_f + offset
                if candidate not in bounce_frames_set and candidate in detections:
                    hit_f = candidate
                    break
        
        used.add(hit_f)
        
        if hit_f in detections:
            hx, hy, ht, _ = detections[hit_f]
        else:
            nf = min(all_frames, key=lambda f: abs(f - hit_f))
            hx, hy, ht, _ = detections[nf]
        
        hits.append({
            "frame": hit_f,
            "timestamp_sec": round(ht, 3),
            "pixel_x": round(hx, 1),
            "pixel_y": round(hy, 1),
            "method": method,
            "score": round(score, 2),
        })
    
    print(f"    Hits detected: {len(hits)}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Deconflict bounces and hits - remove bounces within 3 frames of hits
    # ═══════════════════════════════════════════════════════════════════
    
    HIT_BOUNCE_EXCLUSION = 3  # Frames before/after a hit where bounces are suppressed
    
    hit_frames = set(h["frame"] for h in hits)
    
    # Build exclusion zones around each hit
    exclusion_frames = set()
    for hf in hit_frames:
        for offset in range(-HIT_BOUNCE_EXCLUSION, HIT_BOUNCE_EXCLUSION + 1):
            exclusion_frames.add(hf + offset)
    
    # Filter bounces that fall within exclusion zones
    bounces_before = len(final_bounces)
    final_bounces = [b for b in final_bounces if b["frame"] not in exclusion_frames]
    bounces_removed = bounces_before - len(final_bounces)
    
    if bounces_removed > 0:
        print(f"    Removed {bounces_removed} bounces near hits (±{HIT_BOUNCE_EXCLUSION} frames)")
    
    # ═══════════════════════════════════════════════════════════════════
    # Calculate speeds for all events
    # ═══════════════════════════════════════════════════════════════════
    
    def calc_speeds(frame):
        """Calculate speed_in and speed_out for a given frame."""
        idx = all_frames.index(frame) if frame in all_frames else -1
        if idx < 0:
            return 0, 0
        
        # Speed in (before event)
        speed_in = 0
        if idx > 0:
            f_prev = all_frames[idx - 1]
            gap = frame - f_prev
            if gap <= MAX_FRAME_GAP and gap > 0:
                speed_in = np.hypot(
                    detections[frame][0] - detections[f_prev][0],
                    detections[frame][1] - detections[f_prev][1]
                ) / gap
        
        # Speed out (after event)
        speed_out = 0
        if idx < len(all_frames) - 1:
            f_next = all_frames[idx + 1]
            gap = f_next - frame
            if gap <= MAX_FRAME_GAP and gap > 0:
                speed_out = np.hypot(
                    detections[f_next][0] - detections[frame][0],
                    detections[f_next][1] - detections[frame][1]
                ) / gap
        
        return round(speed_in, 2), round(speed_out, 2)
    
    # ═══════════════════════════════════════════════════════════════════
    # Combine and format output
    # ═══════════════════════════════════════════════════════════════════
    
    # Structured events list
    events = []
    
    # Array overlay format: [timestamp, pixel_x, pixel_y, speed_ratio, speed_in, speed_out, frame]
    overlay = []
    
    for b in final_bounces:
        speed_in, speed_out = calc_speeds(b["frame"])
        # speed_ratio <= 1.0 for bounces
        speed_ratio = round(speed_out / max(speed_in, 0.1), 2) if speed_in > 0.1 else 0.5
        speed_ratio = min(speed_ratio, 1.0)  # Cap at 1.0 for bounces
        
        # Calculate world coordinates based on surface
        world_x, world_y = None, None
        surface = b["surface"]
        px, py = b["pixel_x"], b["pixel_y"]
        
        if surface == "FW":
            world_x, world_y = project(H_FW, px, py)
        elif surface == "FL":
            world_x, world_y = project(H_FL, px, py)
        elif surface == "LW":
            world_x, world_y = project(H_LW, px, py)
        elif surface == "RW":
            world_x, world_y = project(H_RW, px, py)
        
        events.append({
            "type": "bounce",
            "frame": b["frame"],
            "timestamp_sec": b["timestamp_sec"],
            "pixel_x": b["pixel_x"],
            "pixel_y": b["pixel_y"],
            "surface": b["surface"],
            "method": b["method"],
            "speed_in": speed_in,
            "speed_out": speed_out,
            "speed_ratio": speed_ratio,
            "world_x": round(world_x, 3) if world_x is not None else None,
            "world_y": round(world_y, 3) if world_y is not None else None,
        })
        
        overlay.append([
            b["timestamp_sec"],
            b["pixel_x"],
            b["pixel_y"],
            speed_ratio,
            speed_in,
            speed_out,
            b["frame"]
        ])
    
    for h in hits:
        speed_in, speed_out = calc_speeds(h["frame"])
        # speed_ratio > 1.0 for hits (use the detection score as proxy)
        speed_ratio = max(h["score"], 2.0)  # Ensure > 1.0 for hits
        
        events.append({
            "type": "hit",
            "frame": h["frame"],
            "timestamp_sec": h["timestamp_sec"],
            "pixel_x": h["pixel_x"],
            "pixel_y": h["pixel_y"],
            "method": h["method"],
            "speed_in": speed_in,
            "speed_out": speed_out,
            "speed_ratio": speed_ratio,
        })
        
        overlay.append([
            h["timestamp_sec"],
            h["pixel_x"],
            h["pixel_y"],
            speed_ratio,
            speed_in,
            speed_out,
            h["frame"]
        ])
    
    # Sort both by timestamp
    events.sort(key=lambda e: e["timestamp_sec"])
    overlay.sort(key=lambda e: e[0])
    
    return {
        "total_events": len(events),
        "total_bounces": len(final_bounces),
        "total_hits": len(hits),
        "bounce_surfaces": {
            "FW": sum(1 for b in final_bounces if b["surface"] == "FW"),
            "FL": sum(1 for b in final_bounces if b["surface"] == "FL"),
            "LW": sum(1 for b in final_bounces if b["surface"] == "LW"),
            "RW": sum(1 for b in final_bounces if b["surface"] == "RW"),
        },
        "events": events,
        "overlay": overlay,  # Array format for compatibility
    }


def attribute_shots(
    bounce_hit_events: dict,
    player_locations: dict,
    homography_data: dict,
    player_1_handedness: str = "right",
    player_2_handedness: str = "right",
    player_1_name: str = "Player 1",
    player_2_name: str = "Player 2",
) -> dict:
    """
    Attribute each hit to a player based on proximity, and determine shot type.
    
    Args:
        bounce_hit_events: Output from detect_bounces_and_hits
        player_locations: Player world coordinates per frame
        homography_data: Homography matrices for coordinate conversion
        player_1_handedness: "right" or "left"
        player_2_handedness: "right" or "left"
        player_1_name: Display name for player 1
        player_2_name: Display name for player 2
        
    Returns:
        Dict with shot attributions including forehand/backhand classification
    """
    import numpy as np
    
    events = bounce_hit_events.get("events", [])
    frames_data = player_locations.get("frames", [])
    
    if not events or not frames_data:
        return {"shots": [], "error": "Missing events or player data"}
    
    # Build frame lookup for player positions
    player_pos_by_frame = {}
    for entry in frames_data:
        frame = entry.get("frame_number")
        if frame is None:
            continue
        player_pos_by_frame[frame] = {
            "player_1": entry.get("player_1"),
            "player_2": entry.get("player_2"),
        }
    
    # Get floor homography for ball projection
    homographies = homography_data.get("homographies", {})
    H_floor = homographies.get("floor", {}).get("homography_matrix")
    
    if H_floor:
        H_floor = np.array(H_floor)
    else:
        # Fallback to main homography
        H_floor = np.array(homography_data.get("homography_matrix", np.eye(3)))
    
    def pixel_to_court(px, py):
        """Convert pixel coordinates to court coordinates."""
        r = H_floor @ np.array([px, py, 1.0])
        if abs(r[2]) < 1e-10:
            return None, None
        return r[0] / r[2], r[1] / r[2]
    
    def get_player_at_frame(frame, player_key):
        """Get player position at or near the given frame."""
        # Try exact frame
        if frame in player_pos_by_frame:
            p = player_pos_by_frame[frame].get(player_key)
            if p and p.get("court_x") is not None:
                return p
        
        # Search nearby frames (within 5 frames)
        for offset in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
            nearby = frame + offset
            if nearby in player_pos_by_frame:
                p = player_pos_by_frame[nearby].get(player_key)
                if p and p.get("court_x") is not None:
                    return p
        return None
    
    def distance(p1_court, ball_court):
        """Euclidean distance between two court positions."""
        if p1_court is None or ball_court[0] is None:
            return 999
        return np.hypot(
            p1_court.get("court_x", 0) - ball_court[0],
            p1_court.get("court_y", 0) - ball_court[1]
        )
    
    def determine_ball_side(player_court_x, ball_court_x):
        """
        Determine which side of the player the ball is on.
        
        Assumes player is facing the front wall (y=0).
        - Ball x < player x → ball is on player's LEFT
        - Ball x > player x → ball is on player's RIGHT
        """
        if player_court_x is None or ball_court_x is None:
            return "unknown"
        
        diff = ball_court_x - player_court_x
        if abs(diff) < 0.3:  # Within 30cm, consider it center
            return "center"
        elif diff < 0:
            return "left"
        else:
            return "right"
    
    def determine_shot_type(ball_side, handedness):
        """
        Determine if shot is forehand or backhand based on ball side and handedness.
        
        Right-handed: right side = forehand, left side = backhand
        Left-handed: left side = forehand, right side = backhand
        """
        if ball_side == "unknown" or ball_side == "center":
            return "unknown"
        
        if handedness == "right":
            return "forehand" if ball_side == "right" else "backhand"
        else:  # left-handed
            return "forehand" if ball_side == "left" else "backhand"
    
    # Process each hit event
    shots = []
    hit_events = [e for e in events if e.get("type") == "hit"]
    last_hitter = None
    
    # Court bounds (with some margin)
    COURT_WIDTH = 6.4
    COURT_LENGTH = 9.75
    MARGIN = 1.0  # Allow 1m outside court for measurement error
    
    def is_valid_court_pos(x, y):
        """Check if position is within valid court bounds."""
        if x is None or y is None:
            return False
        return (-MARGIN <= x <= COURT_WIDTH + MARGIN and 
                -MARGIN <= y <= COURT_LENGTH + MARGIN)
    
    def pixel_distance(p1_pixel, ball_px, ball_py):
        """Calculate pixel distance between player foot and ball."""
        if p1_pixel is None:
            return 9999
        foot = p1_pixel.get("foot_position", p1_pixel.get("centroid", {}))
        if not foot:
            return 9999
        fx = foot.get("x", 0)
        fy = foot.get("y", 0)
        return np.hypot(ball_px - fx, ball_py - fy)
    
    print(f"    Attributing {len(hit_events)} hits to players...")
    
    for idx, event in enumerate(hit_events):
        frame = event["frame"]
        ball_px = event["pixel_x"]
        ball_py = event["pixel_y"]
        
        # Convert ball to court coordinates
        ball_court_x, ball_court_y = pixel_to_court(ball_px, ball_py)
        
        # Get player positions (world coords)
        p1 = get_player_at_frame(frame, "player_1")
        p2 = get_player_at_frame(frame, "player_2")
        
        # Get player pixel positions for fallback
        p1_pixel = None
        p2_pixel = None
        for entry in frames_data:
            if entry.get("frame_number") == frame:
                p1_pixel = entry.get("player_1")
                p2_pixel = entry.get("player_2")
                break
        # Search nearby if not found
        if p1_pixel is None:
            for offset in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
                for entry in frames_data:
                    if entry.get("frame_number") == frame + offset:
                        p1_pixel = entry.get("player_1")
                        p2_pixel = entry.get("player_2")
                        break
                if p1_pixel:
                    break
        
        # Decide which distance metric to use
        ball_on_court = is_valid_court_pos(ball_court_x, ball_court_y)
        
        if ball_on_court:
            # Use court-based distance
            dist_p1 = distance(p1, (ball_court_x, ball_court_y))
            dist_p2 = distance(p2, (ball_court_x, ball_court_y))
            distance_method = "court"
        else:
            # Ball is off-court (likely on front wall) - use pixel distance
            dist_p1 = pixel_distance(p1_pixel, ball_px, ball_py)
            dist_p2 = pixel_distance(p2_pixel, ball_px, ball_py)
            distance_method = "pixel"
        
        # Determine hitter by proximity
        if dist_p1 < dist_p2:
            hitter = 1
            hitter_pos = p1
            hitter_dist = dist_p1
            handedness = player_1_handedness
        else:
            hitter = 2
            hitter_pos = p2
            hitter_dist = dist_p2
            handedness = player_2_handedness
        
        # Attribution confidence
        dist_diff = abs(dist_p1 - dist_p2)
        if distance_method == "court":
            if dist_diff > 2.0:
                confidence = "high"
            elif dist_diff > 1.0:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            # Pixel-based: use different thresholds (in pixels)
            if dist_diff > 200:
                confidence = "high"
            elif dist_diff > 100:
                confidence = "medium"
            else:
                confidence = "low"
        
        # Check alternation rule
        expected_hitter = 2 if last_hitter == 1 else 1 if last_hitter == 2 else None
        alternation_conflict = expected_hitter is not None and hitter != expected_hitter
        
        # If low confidence and alternation conflict, trust alternation
        if alternation_conflict and confidence == "low":
            hitter = expected_hitter
            hitter_pos = p1 if hitter == 1 else p2
            hitter_dist = dist_p1 if hitter == 1 else dist_p2
            handedness = player_1_handedness if hitter == 1 else player_2_handedness
            confidence = "alternation"
        
        # Determine ball side relative to hitter
        hitter_court_x = hitter_pos.get("court_x") if hitter_pos else None
        ball_side = determine_ball_side(hitter_court_x, ball_court_x)
        
        # Determine shot type (forehand/backhand)
        shot_type = determine_shot_type(ball_side, handedness)
        
        # Determine if this is a serve (first shot of a point)
        is_serve = "serve" in event.get("method", "")
        
        shot = {
            "shot_number": idx + 1,
            "frame": frame,
            "timestamp_sec": event["timestamp_sec"],
            "player": hitter,
            "attribution_confidence": confidence,
            "alternation_conflict": alternation_conflict,
            
            # Ball info
            "ball_pixel": {"x": ball_px, "y": ball_py},
            "ball_court": {"x": round(ball_court_x, 3) if ball_court_x else None, 
                          "y": round(ball_court_y, 3) if ball_court_y else None},
            
            # Player info
            "player_court": {
                "x": round(hitter_pos.get("court_x"), 3) if hitter_pos and hitter_pos.get("court_x") else None,
                "y": round(hitter_pos.get("court_y"), 3) if hitter_pos and hitter_pos.get("court_y") else None,
            } if hitter_pos else None,
            "player_distance_m": round(hitter_dist, 2),
            
            # Shot classification
            "ball_side": ball_side,
            "handedness": handedness,
            "shot_type": shot_type,
            "is_serve": is_serve,
            
            # Detection method
            "hit_detection_method": event.get("method", "unknown"),
            "distance_method": distance_method,
        }
        
        shots.append(shot)
        last_hitter = hitter
    
    # Summary stats
    p1_shots = [s for s in shots if s["player"] == 1]
    p2_shots = [s for s in shots if s["player"] == 2]
    
    p1_forehands = sum(1 for s in p1_shots if s["shot_type"] == "forehand")
    p1_backhands = sum(1 for s in p1_shots if s["shot_type"] == "backhand")
    p2_forehands = sum(1 for s in p2_shots if s["shot_type"] == "forehand")
    p2_backhands = sum(1 for s in p2_shots if s["shot_type"] == "backhand")
    
    print(f"    Player 1 ({player_1_name}): {len(p1_shots)} shots ({p1_forehands} FH, {p1_backhands} BH)")
    print(f"    Player 2 ({player_2_name}): {len(p2_shots)} shots ({p2_forehands} FH, {p2_backhands} BH)")
    
    return {
        "total_shots": len(shots),
        "player_1": {
            "name": player_1_name,
            "handedness": player_1_handedness,
            "total_shots": len(p1_shots),
            "forehands": p1_forehands,
            "backhands": p1_backhands,
        },
        "player_2": {
            "name": player_2_name,
            "handedness": player_2_handedness,
            "total_shots": len(p2_shots),
            "forehands": p2_forehands,
            "backhands": p2_backhands,
        },
        "shots": shots,
    }


def classify_shot(
    side_wall_contact: dict,
    floor_bounce: dict,
    front_wall_contact: dict,
    hit_time: float,
    prev_floor_bounce_time: float
) -> str:
    """
    Classify a shot as drive, drop, boast, or volley based on ball trajectory.
    
    Classification rules:
    - Boast: Ball hits side wall (side_wall_contact exists with valid coords)
    - Volley: Ball hit before it bounces (hit_time < prev_floor_bounce_time or no floor bounce after hit)
    - Drop: Ball lands in front court (floor_bounce y < 4.0m from front wall)
    - Drive: Ball lands in back court (floor_bounce y > 6.0m from front wall) or default
    
    Court is 9.75m long, front wall at y=0.
    """
    # Check for boast (ball hits side wall)
    if side_wall_contact and side_wall_contact.get("x") is not None:
        return "boast"
    
    # Check for volley (hit before previous bounce lands, or no floor bounce after this hit)
    if prev_floor_bounce_time is not None and hit_time < prev_floor_bounce_time:
        return "volley"
    
    # If no floor bounce recorded, might be a volley or incomplete data
    if not floor_bounce or floor_bounce.get("y") is None:
        # Check if front wall contact is low (likely a kill/nick attempt)
        if front_wall_contact and front_wall_contact.get("y") is not None:
            fw_height = front_wall_contact.get("y", 2.0)
            if fw_height < 0.8:  # Below ~80cm, likely a kill shot
                return "drop"
        return "drive"  # Default to drive if we can't determine
    
    # Classify based on where ball lands
    bounce_y = floor_bounce.get("y", 5.0)  # Distance from front wall
    
    if bounce_y < 4.0:
        # Front court - drop shot
        return "drop"
    elif bounce_y > 6.5:
        # Back court - drive (length)
        return "drive"
    else:
        # Mid court - could be either, check front wall height
        if front_wall_contact and front_wall_contact.get("y") is not None:
            fw_height = front_wall_contact.get("y", 2.0)
            if fw_height < 1.2:  # Low on front wall
                return "drop"
        return "drive"


def build_point_rallies(
    point_starts: list,
    shot_attribution: dict,
    bounce_hit_events: dict,
    player_locations: dict,
    player_config: dict,
    src_fps: float,
    total_duration_sec: float,
) -> list:
    """
    Build per-point rally breakdowns with full shot sequences.
    
    For each point, creates a JSON structure showing:
    - Point timing (start, end, duration)
    - Player info (names, handedness)
    - Full rally sequence with each shot showing:
      - Who hit it
      - Player positions at hit time
      - Ball trajectory (front wall, side wall, floor contacts)
    
    Returns:
        List of point rally dictionaries
    """
    import numpy as np
    
    if not point_starts:
        print("    ⚠ Missing point_starts")
        return []
    
    if not shot_attribution:
        print("    ⚠ Missing shot_attribution")
        return []
    
    shots = shot_attribution.get("shots", []) if shot_attribution else []
    events = bounce_hit_events.get("events", []) if bounce_hit_events else []
    frames_data = player_locations.get("frames", []) if player_locations else []
    
    print(f"    Building rallies: {len(point_starts)} points, {len(shots)} shots, {len(events)} events, {len(frames_data)} frames")
    
    # Build frame lookup for player positions
    player_pos_by_frame = {}
    for entry in frames_data:
        if not entry:
            continue
        frame = entry.get("frame_number")
        if frame is not None:
            p1_data = entry.get("player_1") or {}
            p2_data = entry.get("player_2") or {}
            player_pos_by_frame[frame] = {
                "player_1": {
                    "world_x": p1_data.get("court_x") if p1_data else None,
                    "world_y": p1_data.get("court_y") if p1_data else None,
                },
                "player_2": {
                    "world_x": p2_data.get("court_x") if p2_data else None,
                    "world_y": p2_data.get("court_y") if p2_data else None,
                },
            }
    
    # Sort everything by time
    shots_sorted = sorted(shots, key=lambda s: s.get("timestamp_sec", 0))
    events_sorted = sorted(events, key=lambda e: e.get("timestamp_sec", 0))
    points_sorted = sorted(point_starts, key=lambda p: p.get("timestamp_sec", 0))
    
    # Get player info
    p1_info = player_config.get("player_1", {})
    p2_info = player_config.get("player_2", {})
    
    point_rallies = []
    
    for i, point in enumerate(points_sorted):
        point_num = point.get("point_number", i + 1)
        point_start = point.get("timestamp_sec", 0)
        
        # Determine point end time (next point start or video end)
        if i + 1 < len(points_sorted):
            point_end = points_sorted[i + 1].get("timestamp_sec", total_duration_sec)
        else:
            point_end = total_duration_sec
        
        point_duration = round(point_end - point_start, 2)
        
        # Find all shots within this point
        point_shots = [
            s for s in shots_sorted 
            if point_start <= s.get("timestamp_sec", 0) < point_end
        ]
        
        # Find all bounce events within this point
        point_events = [
            e for e in events_sorted 
            if point_start <= e.get("timestamp_sec", 0) < point_end
        ]
        
        # Build rally sequence
        rally = []
        
        for shot_idx, shot in enumerate(point_shots):
            hit_time = shot.get("timestamp_sec", 0)
            hit_frame = shot.get("frame", 0)
            
            # Determine next hit time (for finding bounces between shots)
            if shot_idx + 1 < len(point_shots):
                next_hit_time = point_shots[shot_idx + 1].get("timestamp_sec", point_end)
            else:
                next_hit_time = point_end
            
            # Get player positions at hit time
            player_positions = player_pos_by_frame.get(hit_frame, {})
            
            # Search nearby frames if not found
            if not player_positions:
                for offset in range(1, 10):
                    if hit_frame + offset in player_pos_by_frame:
                        player_positions = player_pos_by_frame[hit_frame + offset]
                        break
                    if hit_frame - offset in player_pos_by_frame:
                        player_positions = player_pos_by_frame[hit_frame - offset]
                        break
            
            p1_pos = player_positions.get("player_1", {})
            p2_pos = player_positions.get("player_2", {})
            
            # Find bounces between this hit and next hit
            bounces_after_hit = [
                e for e in point_events
                if e.get("type") == "bounce" and hit_time < e.get("timestamp_sec", 0) <= next_hit_time
            ]
            
            # Categorize bounces by surface
            front_wall_contact = None
            side_wall_contact = None
            floor_bounce = None
            
            for bounce in bounces_after_hit:
                surface = bounce.get("surface", "")
                bounce_info = {
                    "x": round(bounce.get("world_x", 0), 2) if bounce.get("world_x") else None,
                    "y": round(bounce.get("world_y", 0), 2) if bounce.get("world_y") else None,
                    "timestamp_sec": round(bounce.get("timestamp_sec", 0), 2),
                    "frame": bounce.get("frame"),
                }
                
                if surface == "FW" and front_wall_contact is None:
                    front_wall_contact = bounce_info
                elif surface in ["LW", "RW"] and side_wall_contact is None:
                    side_wall_contact = {
                        **bounce_info,
                        "wall": "left" if surface == "LW" else "right",
                    }
                elif surface == "FL" and floor_bounce is None:
                    floor_bounce = bounce_info
            
            # Get previous floor bounce time for volley detection
            prev_floor_bounce_time = None
            if rally:
                prev_shot = rally[-1]
                prev_fb = prev_shot.get("floor_bounce")
                if prev_fb and isinstance(prev_fb, dict):
                    prev_floor_bounce_time = prev_fb.get("timestamp_sec")
            
            # Classify shot type (drive, drop, boast, volley)
            shot_class = classify_shot(
                side_wall_contact=side_wall_contact,
                floor_bounce=floor_bounce,
                front_wall_contact=front_wall_contact,
                hit_time=hit_time,
                prev_floor_bounce_time=prev_floor_bounce_time
            )
            
            # Build shot entry
            rally_shot = {
                "shot_number": shot_idx + 1,
                "hit_by": f"player_{shot.get('player', 1)}",
                "player_name": shot.get("player_name", f"Player {shot.get('player', 1)}"),
                "hit_time_sec": round(hit_time, 2),
                "frame": hit_frame,
                "shot_type": shot.get("shot_type", "unknown"),
                "shot_class": shot_class,
                "confidence": shot.get("confidence", "unknown"),
                
                "player_1_position": {
                    "x": round(p1_pos.get("world_x"), 2) if p1_pos.get("world_x") else None,
                    "y": round(p1_pos.get("world_y"), 2) if p1_pos.get("world_y") else None,
                },
                "player_2_position": {
                    "x": round(p2_pos.get("world_x"), 2) if p2_pos.get("world_x") else None,
                    "y": round(p2_pos.get("world_y"), 2) if p2_pos.get("world_y") else None,
                },
                
                "front_wall_contact": front_wall_contact,
                "side_wall_contact": side_wall_contact,
                "floor_bounce": floor_bounce,
            }
            
            rally.append(rally_shot)
        
        # Build point summary
        point_rally = {
            "point_number": point_num,
            "start_time_sec": round(point_start, 2),
            "end_time_sec": round(point_end, 2),
            "duration_sec": point_duration,
            "start_frame": point.get("frame", 0),
            
            "player_1": {
                "name": p1_info.get("name", "Player 1"),
                "handedness": p1_info.get("handedness", "right"),
            },
            "player_2": {
                "name": p2_info.get("name", "Player 2"),
                "handedness": p2_info.get("handedness", "right"),
            },
            
            "total_shots": len(rally),
            "rally": rally,
        }
        
        point_rallies.append(point_rally)
    
    return point_rallies


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
            if pdata is None:
                continue

            # Handle both old format (x, y) and new format (foot_position, centroid, bbox)
            if "foot_position" in pdata:
                # New format
                foot = pdata["foot_position"]
                px = float(foot["x"])
                py = float(foot["y"])
            elif "x" in pdata:
                # Old format (backward compatible)
                px = float(pdata["x"])
                py = float(pdata["y"])
            else:
                continue

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

            # Build world entry with all available data
            player_world = {
                "court_x": court_x,
                "court_y": court_y,
            }
            
            # Include pixel data from new format
            if "foot_position" in pdata:
                player_world["foot_position"] = pdata["foot_position"]
                player_world["centroid"] = pdata.get("centroid")
                player_world["bbox"] = pdata.get("bbox")
                player_world["pixel_count"] = pdata.get("pixel_count")
            else:
                # Old format - just foot position
                player_world["foot_position"] = {"x": int(px), "y": int(py)}
            
            world_entry[player_key] = player_world
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

    # ================================================================
    # Step 6: Bounce and Hit Detection
    # ================================================================
    print("\n🏓 Running bounce and hit detection...")
    
    bounce_hit_result = None
    try:
        bounce_hit_result = detect_bounces_and_hits(ball_data, h_data)
        
        if bounce_hit_result and bounce_hit_result.get("events"):
            # Upload bounce/hit overlay
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                json_mod.dump(bounce_hit_result, tf, indent=2)
                tf_path = tf.name
            bounce_hit_url = upload_to_gcs(tf_path, f"{gcs_dir}/bounce_hit_events.json")
            os.unlink(tf_path)
            
            n_bounces = bounce_hit_result.get("total_bounces", 0)
            n_hits = bounce_hit_result.get("total_hits", 0)
            surfaces = bounce_hit_result.get("bounce_surfaces", {})
            
            print(f"  ✅ Bounce/hit detection complete:")
            print(f"     Bounces: {n_bounces} (FW:{surfaces.get('FW',0)} FL:{surfaces.get('FL',0)} LW:{surfaces.get('LW',0)} RW:{surfaces.get('RW',0)})")
            print(f"     Hits: {n_hits}")
        else:
            error_msg = bounce_hit_result.get("error", "Unknown") if bounce_hit_result else "No result"
            print(f"  ⚠ No bounce/hit events detected: {error_msg}")
            bounce_hit_url = None
    except Exception as e:
        print(f"  ⚠ Bounce/hit detection failed: {e}")
        import traceback
        traceback.print_exc()
        bounce_hit_url = None

    # ================================================================
    # Step 7: Shot Attribution (who hit each shot, forehand/backhand)
    # ================================================================
    print("\n🎯 Running shot attribution...")
    
    shot_attribution_url = None
    try:
        # Try to load player config (handedness) from GCS if it exists
        player_1_handedness = "right"
        player_2_handedness = "right"
        player_1_name = "Player 1"
        player_2_name = "Player 2"
        
        try:
            local_config = f"/tmp/player_config_{video_key}.json"
            config_gcs_path = f"{gcs_dir}/player_config.json"
            print(f"  Downloading player config from: {config_gcs_path}")
            download_from_gcs(config_gcs_path, local_config)
            with open(local_config, "r") as f:
                player_config = json_mod.load(f)
            print(f"  Raw player_config loaded: {player_config}")
            player_1_handedness = player_config.get("player_1", {}).get("handedness", "right")
            player_2_handedness = player_config.get("player_2", {}).get("handedness", "right")
            player_1_name = player_config.get("player_1", {}).get("name", "Player 1")
            player_2_name = player_config.get("player_2", {}).get("name", "Player 2")
            print(f"  Loaded player config: {player_1_name} ({player_1_handedness}), {player_2_name} ({player_2_handedness})")
        except Exception as e:
            print(f"  Using default player config: both right-handed (error: {e})")
        
        if bounce_hit_result and bounce_hit_result.get("events") and world_output.get("frames"):
            shot_result = attribute_shots(
                bounce_hit_result,
                world_output,
                h_data,
                player_1_handedness=player_1_handedness,
                player_2_handedness=player_2_handedness,
                player_1_name=player_1_name,
                player_2_name=player_2_name,
            )
            
            if shot_result and shot_result.get("shots"):
                # Upload shot attribution
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                    json_mod.dump(shot_result, tf, indent=2)
                    tf_path = tf.name
                shot_attribution_url = upload_to_gcs(tf_path, f"{gcs_dir}/shot_attribution.json")
                os.unlink(tf_path)
                
                p1 = shot_result.get("player_1", {})
                p2 = shot_result.get("player_2", {})
                print(f"  ✅ Shot attribution complete:")
                print(f"     Player 1 ({p1.get('handedness', 'R')}): {p1.get('total_shots', 0)} shots "
                      f"({p1.get('forehands', 0)} FH, {p1.get('backhands', 0)} BH)")
                print(f"     Player 2 ({p2.get('handedness', 'R')}): {p2.get('total_shots', 0)} shots "
                      f"({p2.get('forehands', 0)} FH, {p2.get('backhands', 0)} BH)")
            else:
                print(f"  ⚠ No shots attributed")
                shot_result = None
        else:
            print(f"  ⚠ Skipping shot attribution (missing bounce/hit events or player data)")
            shot_result = None
            
    except Exception as e:
        print(f"  ⚠ Shot attribution failed: {e}")
        import traceback
        traceback.print_exc()
        shot_result = None

    # ================================================================
    # Step 8: Build Per-Point Rally Breakdown
    # ================================================================
    print("\n📋 Building per-point rally breakdowns...")
    
    point_rally_urls = []
    try:
        # Get total video duration from tracking data
        total_duration_sec = 0
        if tracking_data.get("frames"):
            last_frame = tracking_data["frames"][-1]
            total_duration_sec = last_frame.get("timestamp_sec", 0) + 1.0
        
        # Load player config for names
        player_config_data = {
            "player_1": {"name": player_1_name, "handedness": player_1_handedness},
            "player_2": {"name": player_2_name, "handedness": player_2_handedness},
        }
        
        if point_starts and shot_result:
            point_rallies = build_point_rallies(
                point_starts=point_starts,
                shot_attribution=shot_result,
                bounce_hit_events=bounce_hit_result,
                player_locations=world_output,
                player_config=player_config_data,
                src_fps=src_fps,
                total_duration_sec=total_duration_sec,
            )
            
            if point_rallies:
                print(f"  ✅ Built {len(point_rallies)} point rallies")
                
                # Upload each point as a separate file
                for point_rally in point_rallies:
                    point_num = point_rally.get("point_number", 0)
                    point_filename = f"point_{point_num:03d}.json"
                    
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                        json_mod.dump(point_rally, tf, indent=2)
                        tf_path = tf.name
                    
                    point_url = upload_to_gcs(tf_path, f"{gcs_dir}/points/{point_filename}")
                    point_rally_urls.append(point_url)
                    os.unlink(tf_path)
                    
                    # Print summary for this point
                    rally_len = len(point_rally.get("rally", []))
                    duration = point_rally.get("duration_sec", 0)
                    print(f"     Point {point_num}: {rally_len} shots, {duration}s")
                
                # Also upload an index file listing all points
                points_index = {
                    "total_points": len(point_rallies),
                    "points": [
                        {
                            "point_number": pr["point_number"],
                            "start_time_sec": pr["start_time_sec"],
                            "end_time_sec": pr["end_time_sec"],
                            "duration_sec": pr["duration_sec"],
                            "total_shots": pr["total_shots"],
                            "file": f"point_{pr['point_number']:03d}.json",
                        }
                        for pr in point_rallies
                    ],
                    "player_1": player_config_data["player_1"],
                    "player_2": player_config_data["player_2"],
                }
                
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                    json_mod.dump(points_index, tf, indent=2)
                    tf_path = tf.name
                points_index_url = upload_to_gcs(tf_path, f"{gcs_dir}/points/index.json")
                os.unlink(tf_path)
                print(f"  ✅ Uploaded points index: {points_index_url}")
            else:
                print(f"  ⚠ No point rallies built")
        else:
            print(f"  ⚠ Skipping rally breakdown (missing point_starts or shot_result)")
            
    except Exception as e:
        print(f"  ⚠ Rally breakdown failed: {e}")
        import traceback
        traceback.print_exc()

    return {
        "status": "ok",
        "world_coords_url": world_url,
        "point_starts_url": points_url,
        "bounce_hit_url": bounce_hit_url,
        "shot_attribution_url": shot_attribution_url,
        "total_points": len(point_starts),
        "point_rally_urls": point_rally_urls,
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
        print(f"  🚀 Spawning convert_and_detect_points for video_key: {video_key}")
        convert_and_detect_points.spawn(video_key)
        print(f"  ✓ World coord conversion + point detection spawned")
    except Exception as e:
        print(f"  ⚠ Convert + detect spawn failed: {e}")
        import traceback
        traceback.print_exc()

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

    @api.get("/api/points/{video_key}")
    async def get_points(video_key: str):
        """Fetch point_starts.json from GCS and return it."""
        import tempfile
        import json as json_mod
        try:
            local_path = f"/tmp/point_starts_{video_key}.json"
            download_from_gcs(f"video-data/{video_key}/point_starts.json", local_path)
            with open(local_path, "r") as f:
                data = json_mod.load(f)
            return JSONResponse(data)
        except Exception as e:
            return JSONResponse({"error": str(e), "status": "not_ready"}, status_code=404)

    @api.get("/api/points/{video_key}/index")
    async def get_points_index(video_key: str):
        """Fetch points/index.json from GCS - summary of all points."""
        import json as json_mod
        try:
            local_path = f"/tmp/points_index_{video_key}.json"
            download_from_gcs(f"video-data/{video_key}/points/index.json", local_path)
            with open(local_path, "r") as f:
                data = json_mod.load(f)
            return JSONResponse(data)
        except Exception as e:
            return JSONResponse({"error": str(e), "status": "not_ready"}, status_code=404)

    @api.get("/api/points/{video_key}/{point_number}")
    async def get_point_rally(video_key: str, point_number: int):
        """Fetch individual point rally data (point_XXX.json)."""
        import json as json_mod
        try:
            local_path = f"/tmp/point_{video_key}_{point_number}.json"
            download_from_gcs(f"video-data/{video_key}/points/point_{point_number:03d}.json", local_path)
            with open(local_path, "r") as f:
                data = json_mod.load(f)
            return JSONResponse(data)
        except Exception as e:
            return JSONResponse({"error": str(e), "status": "not_ready"}, status_code=404)

    @api.post("/api/rebuild-rallies/{video_key}")
    async def rebuild_rallies(video_key: str):
        """Rebuild point rallies for an existing video (re-runs rally breakdown step)."""
        import json as json_mod
        import tempfile
        import os
        
        gcs_dir = f"video-data/{video_key}"
        
        try:
            # Download required files
            print(f"Rebuilding rallies for {video_key}...")
            
            # 1. Point starts
            local_points = f"/tmp/rebuild_points_{video_key}.json"
            try:
                download_from_gcs(f"{gcs_dir}/point_starts.json", local_points)
                with open(local_points, "r") as f:
                    points_data = json_mod.load(f)
                point_starts = points_data.get("point_starts", [])
                print(f"  Loaded {len(point_starts)} point starts")
            except Exception as e:
                return JSONResponse({"error": f"Failed to load point_starts.json: {e}", "status": "failed"}, status_code=500)
            
            # 2. Shot attribution
            local_shots = f"/tmp/rebuild_shots_{video_key}.json"
            try:
                download_from_gcs(f"{gcs_dir}/shot_attribution.json", local_shots)
                with open(local_shots, "r") as f:
                    shot_result = json_mod.load(f)
                print(f"  Loaded {len(shot_result.get('shots', []))} shots")
            except Exception as e:
                return JSONResponse({"error": f"Failed to load shot_attribution.json: {e}", "status": "failed"}, status_code=500)
            
            # 3. Bounce/hit events
            local_events = f"/tmp/rebuild_events_{video_key}.json"
            try:
                download_from_gcs(f"{gcs_dir}/bounce_hit_events.json", local_events)
                with open(local_events, "r") as f:
                    bounce_hit_result = json_mod.load(f)
                print(f"  Loaded {len(bounce_hit_result.get('events', []))} events")
            except Exception as e:
                print(f"  Warning: Could not load bounce_hit_events.json: {e}")
                bounce_hit_result = {"events": []}
            
            # 4. Player locations (world coords)
            local_locs = f"/tmp/rebuild_locs_{video_key}.json"
            try:
                download_from_gcs(f"{gcs_dir}/player_locations.json", local_locs)
                with open(local_locs, "r") as f:
                    world_output = json_mod.load(f)
                print(f"  Loaded {len(world_output.get('frames', []))} location frames")
            except Exception as e:
                print(f"  Warning: Could not load player_locations.json: {e}")
                world_output = {"frames": []}
            
            # 5. Player config
            local_config = f"/tmp/rebuild_config_{video_key}.json"
            try:
                download_from_gcs(f"{gcs_dir}/player_config.json", local_config)
                with open(local_config, "r") as f:
                    player_config = json_mod.load(f)
                print(f"  Loaded player config: {player_config}")
            except Exception as e:
                print(f"  Warning: Could not load player_config.json: {e}")
                player_config = {
                    "player_1": {"name": "Player 1", "handedness": "right"},
                    "player_2": {"name": "Player 2", "handedness": "right"},
                }
            
            # Get total duration
            total_duration_sec = 0
            frames = world_output.get("frames", [])
            if frames and len(frames) > 0:
                last_frame = frames[-1]
                if last_frame:
                    total_duration_sec = last_frame.get("timestamp_sec", 0) + 1.0
            
            src_fps = world_output.get("source_fps", 30)
            
            print(f"  Total duration: {total_duration_sec}s, FPS: {src_fps}")
            
            # Build rallies
            point_rallies = build_point_rallies(
                point_starts=point_starts,
                shot_attribution=shot_result,
                bounce_hit_events=bounce_hit_result,
                player_locations=world_output,
                player_config=player_config,
                src_fps=src_fps,
                total_duration_sec=total_duration_sec,
            )
            
            if not point_rallies:
                return JSONResponse({"error": "No rallies built - check logs", "status": "failed"}, status_code=500)
            
            print(f"  Built {len(point_rallies)} point rallies")
            
            # Upload each point
            point_rally_urls = []
            for point_rally in point_rallies:
                point_num = point_rally.get("point_number", 0)
                point_filename = f"point_{point_num:03d}.json"
                
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                    json_mod.dump(point_rally, tf, indent=2)
                    tf_path = tf.name
                
                point_url = upload_to_gcs(tf_path, f"{gcs_dir}/points/{point_filename}")
                point_rally_urls.append(point_url)
                os.unlink(tf_path)
            
            # Upload index
            points_index = {
                "total_points": len(point_rallies),
                "points": [
                    {
                        "point_number": pr["point_number"],
                        "start_time_sec": pr["start_time_sec"],
                        "end_time_sec": pr["end_time_sec"],
                        "duration_sec": pr["duration_sec"],
                        "total_shots": pr["total_shots"],
                        "file": f"point_{pr['point_number']:03d}.json",
                    }
                    for pr in point_rallies
                ],
                "player_1": player_config.get("player_1", {}),
                "player_2": player_config.get("player_2", {}),
            }
            
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                json_mod.dump(points_index, tf, indent=2)
                tf_path = tf.name
            index_url = upload_to_gcs(tf_path, f"{gcs_dir}/points/index.json")
            os.unlink(tf_path)
            
            return JSONResponse({
                "status": "ok",
                "total_points": len(point_rallies),
                "index_url": index_url,
                "point_urls": point_rally_urls,
            })
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error: {error_trace}")
            return JSONResponse({"error": str(e), "trace": error_trace, "status": "failed"}, status_code=500)

    @api.post("/api/regenerate-bounce-events/{video_key}")
    async def regenerate_bounce_events(video_key: str):
        """
        Regenerate bounce_hit_events.json from scratch using ball_tracking.json and homography.json.
        This fixes missing world coordinates in events.
        Then rebuilds point rallies with the new data.
        """
        import json as json_mod
        import tempfile
        import os
        
        gcs_dir = f"video-data/{video_key}"
        
        try:
            print(f"Regenerating bounce events for {video_key}...")
            
            # 1. Download ball_tracking.json
            local_ball = f"/tmp/regen_ball_{video_key}.json"
            try:
                download_from_gcs(f"{gcs_dir}/ball_tracking.json", local_ball)
                with open(local_ball, "r") as f:
                    ball_data = json_mod.load(f)
                print(f"  Loaded ball tracking: {len(ball_data.get('frames', []))} frames")
            except Exception as e:
                return JSONResponse({"error": f"Failed to load ball_tracking.json: {e}", "status": "failed"}, status_code=500)
            
            # 2. Download homography.json
            local_homo = f"/tmp/regen_homo_{video_key}.json"
            try:
                download_from_gcs(f"{gcs_dir}/homography.json", local_homo)
                with open(local_homo, "r") as f:
                    homography_data = json_mod.load(f)
                print(f"  Loaded homography data")
            except Exception as e:
                return JSONResponse({"error": f"Failed to load homography.json: {e}", "status": "failed"}, status_code=500)
            
            # 3. Run detect_bounces_and_hits
            bounce_hit_result = detect_bounces_and_hits(ball_data, homography_data)
            
            if not bounce_hit_result or not bounce_hit_result.get("events"):
                return JSONResponse({"error": "No bounce/hit events detected", "status": "failed"}, status_code=500)
            
            n_events = len(bounce_hit_result.get("events", []))
            n_with_coords = sum(1 for e in bounce_hit_result.get("events", []) if e.get("world_x") is not None)
            print(f"  Detected {n_events} events, {n_with_coords} with world coordinates")
            
            # 4. Upload new bounce_hit_events.json
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                json_mod.dump(bounce_hit_result, tf, indent=2)
                tf_path = tf.name
            
            bounce_hit_url = upload_to_gcs(tf_path, f"{gcs_dir}/bounce_hit_events.json")
            os.unlink(tf_path)
            print(f"  Uploaded new bounce_hit_events.json")
            
            # 5. Now rebuild point rallies with the new data
            # Load other required files
            local_points = f"/tmp/regen_points_{video_key}.json"
            try:
                download_from_gcs(f"{gcs_dir}/point_starts.json", local_points)
                with open(local_points, "r") as f:
                    points_data = json_mod.load(f)
                point_starts = points_data.get("point_starts", [])
            except:
                point_starts = []
            
            local_shots = f"/tmp/regen_shots_{video_key}.json"
            try:
                download_from_gcs(f"{gcs_dir}/shot_attribution.json", local_shots)
                with open(local_shots, "r") as f:
                    shot_result = json_mod.load(f)
            except:
                shot_result = {"shots": []}
            
            local_locs = f"/tmp/regen_locs_{video_key}.json"
            try:
                download_from_gcs(f"{gcs_dir}/player_locations.json", local_locs)
                with open(local_locs, "r") as f:
                    world_output = json_mod.load(f)
            except:
                world_output = {"frames": []}
            
            local_config = f"/tmp/regen_config_{video_key}.json"
            try:
                download_from_gcs(f"{gcs_dir}/player_config.json", local_config)
                with open(local_config, "r") as f:
                    player_config = json_mod.load(f)
            except:
                player_config = {"player_1": {"name": "Player 1"}, "player_2": {"name": "Player 2"}}
            
            # Get total duration
            total_duration_sec = 0
            frames = world_output.get("frames", [])
            if frames and len(frames) > 0:
                last_frame = frames[-1]
                if last_frame:
                    total_duration_sec = last_frame.get("timestamp_sec", 0) + 1.0
            
            src_fps = world_output.get("source_fps", 30)
            
            # Build rallies with new bounce data
            point_rallies = build_point_rallies(
                point_starts=point_starts,
                shot_attribution=shot_result,
                bounce_hit_events=bounce_hit_result,
                player_locations=world_output,
                player_config=player_config,
                src_fps=src_fps,
                total_duration_sec=total_duration_sec,
            )
            
            # Upload point rallies
            point_rally_urls = []
            for point_rally in point_rallies:
                point_num = point_rally.get("point_number", 0)
                point_filename = f"point_{point_num:03d}.json"
                
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                    json_mod.dump(point_rally, tf, indent=2)
                    tf_path = tf.name
                
                point_url = upload_to_gcs(tf_path, f"{gcs_dir}/points/{point_filename}")
                point_rally_urls.append(point_url)
                os.unlink(tf_path)
            
            # Upload index
            points_index = {
                "total_points": len(point_rallies),
                "points": [{"point_number": pr.get("point_number"), "duration_sec": pr.get("duration_sec"), "total_shots": pr.get("total_shots")} for pr in point_rallies],
                "player_1": player_config.get("player_1", {}),
                "player_2": player_config.get("player_2", {}),
            }
            
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                json_mod.dump(points_index, tf, indent=2)
                tf_path = tf.name
            index_url = upload_to_gcs(tf_path, f"{gcs_dir}/points/index.json")
            os.unlink(tf_path)
            
            return JSONResponse({
                "status": "ok",
                "bounce_hit_events": n_events,
                "events_with_world_coords": n_with_coords,
                "total_points": len(point_rallies),
                "bounce_hit_url": bounce_hit_url,
                "index_url": index_url,
            })
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error: {error_trace}")
            return JSONResponse({"error": str(e), "trace": error_trace, "status": "failed"}, status_code=500)

    @api.post("/api/create-job")
    async def create_job(body: dict = None):
        job_id = str(uuid.uuid4())[:8]
        filename = (body or {}).get("filename", "video.mp4")
        video_key = make_video_key(filename)
        return JSONResponse({"job_id": job_id, "video_key": video_key})

    @api.post("/api/player-config")
    async def save_player_config(body: dict):
        """Save player names and handedness configuration to GCS."""
        video_key = body.get("video_key")
        if not video_key:
            return JSONResponse({"error": "video_key required"}, status_code=400)
        
        # Log what we received
        print(f"📝 Player config request for {video_key}:")
        print(f"   Received: player_1_name='{body.get('player_1_name')}', player_2_name='{body.get('player_2_name')}'")
        print(f"   Received: player_1_handedness='{body.get('player_1_handedness')}', player_2_handedness='{body.get('player_2_handedness')}'")
        
        player_config = {
            "player_1": {
                "name": body.get("player_1_name", "Player 1"),
                "handedness": body.get("player_1_handedness", "right"),
            },
            "player_2": {
                "name": body.get("player_2_name", "Player 2"),
                "handedness": body.get("player_2_handedness", "right"),
            },
        }
        
        print(f"   Saving: {player_config}")
        
        # Save to GCS
        import tempfile
        gcs_path = f"{OUTPUT_PREFIX}/{video_key}/player_config.json"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(player_config, tf, indent=2)
            tf_path = tf.name
        
        try:
            url = upload_to_gcs(tf_path, gcs_path)
            import os
            os.unlink(tf_path)
            print(f"✓ Player config saved: {gcs_path}")
            return JSONResponse({
                "status": "ok",
                "player_config": player_config,
                "gcs_path": gcs_path,
            })
        except Exception as e:
            print(f"✗ Player config save failed: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    @api.post("/api/get-upload-url")
    async def get_upload_url(body: dict):
        job_id = body["job_id"]
        video_key = body["video_key"]
        gcs_path = f"{UPLOAD_PREFIX}/{video_key}.mp4"
        url = generate_signed_upload_url(gcs_path)
        return JSONResponse({"upload_url": url, "gcs_path": gcs_path})

    @api.post("/api/detect-trim-point")
    async def detect_trim_point(body: dict):
        """
        Detect the first frame where both players are visible.
        This can be used to auto-trim the video to start when the match begins.
        
        Request body:
        {
            "job_id": "...",
            "start_frame": 0,      // optional, default 0
            "max_frames": 300,     // optional, default 300 (10 seconds at 30fps)
            "step": 5              // optional, check every Nth frame
        }
        
        Returns:
        {
            "found": true/false,
            "frame": 72,
            "timestamp_sec": 2.4,
            "player_1_box": [x1, y1, x2, y2],
            "player_2_box": [x1, y1, x2, y2],
            "frames_scanned": 60
        }
        """
        job_id = body.get("job_id")
        if not job_id:
            return JSONResponse({"error": "job_id required"}, status_code=400)
        
        start_frame = body.get("start_frame", 0)
        max_frames = body.get("max_frames", 300)
        step = body.get("step", 5)
        
        try:
            result = await asyncio.to_thread(
                find_both_players_frame.remote,
                job_id,
                start_frame,
                max_frames,
                step
            )
            return JSONResponse(result)
        except Exception as e:
            import traceback
            return JSONResponse({
                "error": str(e),
                "trace": traceback.format_exc()
            }, status_code=500)

    @api.get("/api/frame/{job_id}/{frame_idx}")
    async def get_frame(job_id: str, frame_idx: int):
        """
        Get a specific frame from the video as base64 JPEG.
        Used for frame navigation during player selection.
        """
        import cv2
        import base64
        
        scratch_vol.reload()
        video_path = f"{DATA_DIR}/jobs/{job_id}/source.mp4"
        
        if not os.path.exists(video_path):
            return JSONResponse({"error": "Video not found"}, status_code=404)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return JSONResponse({"error": "Could not open video"}, status_code=500)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return JSONResponse({"error": f"Could not read frame {frame_idx}"}, status_code=404)
        
        # Encode as JPEG
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
        
        return JSONResponse({
            "frame_idx": frame_idx,
            "frame_b64": frame_b64
        })

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

            # NOTE: Ball tracking is now spawned after trim selection in websocket handler
            # to ensure it starts from the correct frame

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
            trim_start_frame = data.get("trim_start_frame", 0)  # Optional trim offset
            
            # Keep original total_frames for loop bounds, calculate trimmed count for progress
            original_total_frames = total_frames
            trimmed_frames_count = total_frames - trim_start_frame
            
            if trim_start_frame > 0:
                print(f"  ✓ Video trimmed: starting from frame {trim_start_frame} ({trim_start_frame / src_fps:.2f}s)")
                print(f"  Processing {trimmed_frames_count} frames (frames {trim_start_frame} to {original_total_frames})")
            
            # Spawn ball tracking now that we know the trim point
            if video_key:
                try:
                    track_ball.spawn(
                        job_id, video_key,
                        src_fps, src_width, src_height,
                        trim_start_frame)
                    print(f"✓ Ball tracking spawned for {video_key} starting at frame {trim_start_frame}")
                except Exception as e:
                    print(f"⚠ Ball tracking spawn failed: {e}")
            
            # ---- Load existing player config (names and handedness) and add seed points ----
            player_1_name = "Player 1"
            player_2_name = "Player 2"
            player_1_handedness = "right"
            player_2_handedness = "right"
            
            # Try to load existing config saved by /api/player-config
            try:
                existing_config_path = f"/tmp/existing_player_config_{video_key}.json"
                download_from_gcs(f"video-data/{video_key}/player_config.json", existing_config_path)
                with open(existing_config_path, "r") as f:
                    existing_config = json.load(f)
                player_1_name = existing_config.get("player_1", {}).get("name", "Player 1")
                player_2_name = existing_config.get("player_2", {}).get("name", "Player 2")
                player_1_handedness = existing_config.get("player_1", {}).get("handedness", "right")
                player_2_handedness = existing_config.get("player_2", {}).get("handedness", "right")
                os.unlink(existing_config_path)
                print(f"  ✓ Loaded existing player config: {player_1_name}, {player_2_name}")
            except Exception as e:
                print(f"  ⚠ No existing player config found, using defaults: {e}")
            
            # Save player config with seed points to GCS
            player_config_data = {
                "player_1": {
                    "name": player_1_name,
                    "handedness": player_1_handedness,
                    "seed_point": player_1_src,
                },
                "player_2": {
                    "name": player_2_name,
                    "handedness": player_2_handedness,
                    "seed_point": player_2_src,
                },
            }
            
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                    json.dump(player_config_data, tf, indent=2)
                    config_path = tf.name
                gcs_config_path = f"video-data/{video_key}/player_config.json"
                upload_to_gcs(config_path, gcs_config_path)
                os.unlink(config_path)
                print(f"  ✓ Player config saved: {player_1_name} ({player_1_handedness}), {player_2_name} ({player_2_handedness})")
            except Exception as e:
                print(f"  ⚠ Failed to save player config: {e}")

            params = compute_proc_params(src_width, src_height, src_fps)
            params["video_key"] = video_key  # Include for mask uploads

            def to_proc(pt):
                return [int(round(pt[0] / params["scale_x"])),
                        int(round(pt[1] / params["scale_y"]))]

            seed_p1_proc = to_proc(player_1_src)
            seed_p2_proc = to_proc(player_2_src)
            prompt_local_idx = 0
            current_src_start = trim_start_frame  # Start from trimmed frame
            prev_handoff = None  # Carries box+pts data between segments

            # ---- YOLO person detection for initial box prompts ----
            yolo_box_p1_proc = None
            yolo_box_p2_proc = None
            try:
                await ws.send_json({"type": "status", "message": "Detecting player bounding boxes (YOLO)..."})
                person_boxes = await asyncio.to_thread(
                    lambda: detect_person_boxes.remote(job_id, trim_start_frame)
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
            est_segments = math.ceil(trimmed_frames_count / src_per_seg)

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

            while current_src_start < original_total_frames:
                seg_idx += 1
                src_remaining = original_total_frames - current_src_start
                src_num = min(src_per_seg, src_remaining)
                is_last = (current_src_start + src_num >= original_total_frames)

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

            # Generate Firebase storage URL for point_starts.json (will be created by convert_and_detect_points)
            point_starts_url = None
            try:
                import urllib.parse
                gcs_path = f"video-data/{video_key}/point_starts.json"
                encoded_path = urllib.parse.quote(gcs_path, safe="")
                point_starts_url = f"https://firebasestorage.googleapis.com/v0/b/boastiq.firebasestorage.app/o/{encoded_path}?alt=media"
                print(f"Generated point_starts_url: {point_starts_url}")
            except Exception as e:
                print(f"Warning: Could not generate point_starts_url: {e}")

            complete_msg = {
                "type": "complete",
                "json_url": urls.get("json_url"),
                "video_url": urls.get("video_url"),
                "total_segments": seg_idx,
                "total_frames_tracked": len(all_tracking_data),
            }
            if point_starts_url:
                complete_msg["point_starts_url"] = point_starts_url
            
            await ws.send_json(complete_msg)

        except WebSocketDisconnect:
            print(f"Client disconnected: {job_id}")
        except Exception as e:
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except:
                pass

    return api