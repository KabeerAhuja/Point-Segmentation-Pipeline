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
# Gemini API key for the court-landmark seed call (Flash by default).
# Provision this secret in the Modal dashboard with GEMINI_API_KEY=...
gemini_secret = modal.Secret.from_name("Gemini-API-Key")
# Firebase service-account JSON for backend admin gating (ID token verification +
# custom-claim checks). Provision a Modal secret named `firebase-admin-creds`
# with key FIREBASE_SERVICE_ACCOUNT_JSON = the full JSON contents of a Firebase
# service-account private key (Firebase Console → Project Settings → Service
# Accounts → Generate New Private Key).
firebase_secret = modal.Secret.from_name("firebase-admin-creds")
# Stripe credentials for the subscription paywall. Provision a Modal secret named
# `stripe-secret` with keys STRIPE_SECRET_KEY (sk_...), STRIPE_WEBHOOK_SECRET
# (whsec_..., from the webhook endpoint you register on /api/stripe-webhook), and
# STRIPE_PRICE_ID (price_... for the recurring monthly plan). This is the LAST
# setup step — all code paths tolerate its absence until then.
# The `stripe-secret` Modal secret is provisioned (STRIPE_SECRET_KEY,
# STRIPE_WEBHOOK_SECRET, STRIPE_PRICE_ID), so use it directly. Do NOT alias this
# to an empty placeholder — an empty STRIPE_SECRET_KEY makes create-subscription
# return `stripe_not_configured`.
stripe_secret = modal.Secret.from_name("stripe-secret")
# Free-trial length for new subscriptions. During the trial the subscription is
# `trialing` (grants access); the card is collected upfront and charged at end.
TRIAL_PERIOD_DAYS = 7
# Recurring YEARLY price ($300/yr = $25/mo billed annually) for the annual plan.
# Created via `modal run Modal_app.py::create_annual_price` on the same Stripe
# product as the monthly STRIPE_PRICE_ID. Price ids are not secret (they're sent
# to the browser), so this is hardcoded rather than kept in the secret. An env
# override (STRIPE_PRICE_ID_ANNUAL) takes precedence if ever set.
STRIPE_PRICE_ID_ANNUAL = "price_1TsmKWLhz0zSQC5ZOYogkDde"
# Pin coupon/promotion-code + discount calls to a stable API version. The account
# default (2026-06-24.dahlia) restructured promotion codes to point at a new
# `Promotion` object and rejects the classic `coupon` param, so we pin these
# specific calls to the classic model (Coupon → PromotionCode → discounts).
STRIPE_PROMO_API_VERSION = "2024-06-20"

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
        "firebase-admin>=6.5.0",
        "scikit-learn>=1.7",   # learned event spotter (spotter_detect.py)
        "stripe>=9",           # subscription paywall (create-subscription + webhook)
    )
    .add_local_file(
        local_path=str(Path(__file__).parent / "index.html"),
        remote_path="/app/index.html",
    )
    .add_local_file(
        local_path=str(Path(__file__).parent / "Tag.html"),
        remote_path="/app/Tag.html",
    )
    .add_local_file(
        local_path=str(Path(__file__).parent / "Annotate.html"),
        remote_path="/app/Annotate.html",
    )
    # Learned event spotter model (primary engine for detect_bounces_and_hits).
    # Re-export from the training repo with: python3 spotter_multi.py train
    .add_local_file(
        local_path=str(Path(__file__).parent / "spotter_model.pkl"),
        remote_path="/app/spotter_model.pkl",
    )
    .add_local_python_source("spotter_detect")
    # Vendored squash event detector (legacy fallback engine).
    .add_local_python_source("squashev")
)

# GPU-accelerated video transcode image for the background playback normalizer
# (normalize_playback_source). Built on the CUDA runtime so ffmpeg can reach the
# NVENC hardware H.264 encoder — the NVIDIA driver is injected by Modal at run
# time. Ubuntu 22.04's ffmpeg ships with h264_nvenc enabled; the function still
# falls back to libx264 if NVENC is unavailable, so correctness never depends on
# the GPU. Kept lean (no opencv/torch) since it only shells out to ffmpeg.
transcode_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg")
    .pip_install("google-cloud-storage")
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

tracknet_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "opencv-python-headless",
        "google-cloud-storage",
        "numpy",
    )
)

BALL_WEIGHTS_DIR = "/ball_weights"
TRACKNET_WEIGHTS_PATH = f"{BALL_WEIGHTS_DIR}/tracknet_v3/squash_best.pt"
TRACKNET_SEQ_LEN = 8
TRACKNET_IN_W = 512
TRACKNET_IN_H = 288
TRACKNET_HEATMAP_THRESHOLD = 0.3
TRACKNET_INPAINT_WEIGHTS_PATH = f"{BALL_WEIGHTS_DIR}/tracknet_v3/InpaintNet_best.pt"
TRACKNET_INPAINT_SEQ_LEN = 16     # upstream window size
TRACKNET_INPAINT_MAX_GAP = 5      # only fill detection gaps up to this many frames

# ====================================================================
# Shared Configuration
# ====================================================================

DEFAULT_CONFIG = {
    "frames_per_segment": 1500,
    "overlap_frames": 150,
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


def _is_faststart(local_path: str) -> bool:
    """True if the MP4's moov atom comes before mdat (progressive/faststart),
    which lets browsers (esp. Safari) start playback without fetching the tail."""
    import struct
    try:
        with open(local_path, "rb") as f:
            head = f.read(1 << 20)  # 1 MiB is plenty to see ftyp + (moov|mdat)
    except Exception:
        return False
    i, names = 0, []
    while i + 8 <= len(head):
        size = struct.unpack(">I", head[i:i+4])[0]
        name = head[i+4:i+8].decode("ascii", "ignore")
        if not name.isalpha():
            break
        names.append(name)
        if size == 1:
            if i + 16 > len(head):
                break
            size = struct.unpack(">Q", head[i+8:i+16])[0]
        if size < 8:
            break
        i += size
    return "moov" in names and ("mdat" not in names or names.index("moov") < names.index("mdat"))


def _probe_web_safe(local_path: str):
    """Probe a video's codec / pixel format / faststart layout.

    Returns (is_web_safe, codec, pixfmt) where is_web_safe means the file plays
    in every browser as-is (H.264 / 8-bit 4:2:0 / faststart). Raw phone uploads
    usually fail this: iPhones record HEVC (H.265), often 10-bit, which Chrome
    cannot decode, and phone MP4/MOV put the `moov` atom at the END of the file,
    which Safari streams poorly over HTTP.
    """
    import subprocess

    def _probe(entry):
        try:
            return subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", f"stream={entry}", "-of", "csv=p=0", local_path],
                capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception:
            return ""

    codec = _probe("codec_name")
    pixfmt = _probe("pix_fmt")
    is_h264 = codec == "h264"
    is_8bit_420 = pixfmt in ("yuv420p", "yuvj420p")
    web_safe = is_h264 and is_8bit_420 and _is_faststart(local_path)
    return web_safe, codec, pixfmt


def _transcode_web_safe(local_path: str, out_path: str, codec: str, pixfmt: str,
                        use_gpu: bool = True):
    """Produce a universally browser-playable encode of `local_path` at
    `out_path` (H.264 / 8-bit 4:2:0 / faststart).

    Frame timing is preserved (`-vsync passthrough`) so the output stays
    frame-aligned with the original the detector ran on. If only the container
    layout is wrong we losslessly remux; otherwise we transcode with the GPU
    NVENC encoder when available, falling back to CPU libx264. Raises on failure.
    """
    import subprocess

    if codec == "h264" and pixfmt in ("yuv420p", "yuvj420p"):
        # Codec/bit-depth are fine — only the container layout is wrong.
        print(f"  → source h264/{pixfmt} but moov-at-end; lossless faststart remux")
        subprocess.run(
            ["ffmpeg", "-y", "-i", local_path, "-c", "copy",
             "-movflags", "+faststart", out_path],
            check=True, capture_output=True, text=True, timeout=1800)
        return

    # Wrong codec / bit depth (e.g. iPhone HEVC 10-bit) — transcode to H.264 8-bit.
    if use_gpu:
        print(f"  → source {codec or '?'}/{pixfmt or '?'}; NVENC transcode to H.264 8-bit + faststart")
        gpu_cmd = ["ffmpeg", "-y", "-i", local_path,
                   "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0",
                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                   "-movflags", "+faststart", "-vsync", "passthrough", out_path]
        try:
            subprocess.run(gpu_cmd, check=True, capture_output=True, text=True, timeout=3600)
            return
        except Exception as e:
            print(f"  ⚠ NVENC transcode failed ({e}); falling back to libx264")

    print(f"  → transcoding {codec or '?'}/{pixfmt or '?'} to H.264 8-bit + faststart (libx264)")
    subprocess.run(
        ["ffmpeg", "-y", "-i", local_path,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", "-vsync", "passthrough", out_path],
        check=True, capture_output=True, text=True, timeout=3600)


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


def generate_signed_read_url(gcs_path: str, hours: int = 2) -> str:
    import datetime
    bucket = _get_gcs_bucket()
    blob = bucket.blob(gcs_path)
    return blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(hours=hours),
        method="GET",
    )


def list_pending_landmark_jobs() -> list:
    """List videos that have a first frame uploaded but no homography yet.

    Returns a list of {video_key, first_frame_url, uploaded_at_iso, has_homography}
    for the admin tagging queue.
    """
    bucket = _get_gcs_bucket()
    pending = []
    seen_keys = set()
    for blob in bucket.list_blobs(prefix=f"{OUTPUT_PREFIX}/"):
        if not blob.name.endswith("/first_frame.png"):
            continue
        video_key = blob.name[len(f"{OUTPUT_PREFIX}/"):-len("/first_frame.png")]
        if not video_key or video_key in seen_keys:
            continue
        seen_keys.add(video_key)
        homog_blob = bucket.blob(f"{OUTPUT_PREFIX}/{video_key}/homography.json")
        if homog_blob.exists():
            continue
        pending.append({
            "video_key": video_key,
            "first_frame_url": generate_signed_read_url(blob.name, hours=4),
            "uploaded_at_iso": (blob.time_created.isoformat()
                                if blob.time_created else None),
        })
    pending.sort(key=lambda j: j["uploaded_at_iso"] or "", reverse=True)
    return pending


def backfill_thumbnail_paths(force: bool = False) -> dict:
    """Set videos/{key}.thumbnailPath = video-data/{key}/first_frame.png for
    existing docs that have a first frame in Storage but no thumbnailPath yet.

    The My Videos dashboard reads `thumbnailPath` to show a card thumbnail;
    videos created before that field was populated fall back to a placeholder.
    This walks the first_frame.png blobs in GCS and patches the matching
    Firestore doc. With force=False, docs that already have a thumbnailPath are
    left untouched. Returns a summary of what changed.
    """
    app = _init_firebase_admin()
    from firebase_admin import firestore as fb_firestore
    # Firestore lives in the named database "default" (literal string), NOT the
    # special "(default)" DB — must be passed explicitly or writes 404.
    db = fb_firestore.client(app, database_id="default")
    bucket = _get_gcs_bucket()

    updated, skipped, missing_doc = [], [], []
    seen_keys = set()
    for blob in bucket.list_blobs(prefix=f"{OUTPUT_PREFIX}/"):
        if not blob.name.endswith("/first_frame.png"):
            continue
        video_key = blob.name[len(f"{OUTPUT_PREFIX}/"):-len("/first_frame.png")]
        if not video_key or video_key in seen_keys:
            continue
        seen_keys.add(video_key)

        doc_ref = db.collection("videos").document(video_key)
        snap = doc_ref.get()
        if not snap.exists:
            missing_doc.append(video_key)
            continue
        if snap.to_dict().get("thumbnailPath") and not force:
            skipped.append(video_key)
            continue
        doc_ref.update({"thumbnailPath": blob.name})
        updated.append(video_key)

    return {
        "updated_count": len(updated),
        "skipped_count": len(skipped),
        "missing_doc_count": len(missing_doc),
        "updated": updated,
        "missing_doc": missing_doc,
    }


def _first_nonblack_frame(video_path: str, max_scan: int = 90, luma_thresh: float = 12.0):
    """Return the first non-near-black frame of a video as (frame_bgr, w, h).

    Scans up to `max_scan` frames and picks the first whose mean pixel value
    clears `luma_thresh`; falls back to frame 0 if every scanned frame is dark
    (e.g. a genuinely black opening). Returns (None, w, h) if unreadable.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = None
    chosen = None
    for _ in range(max_scan):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if first is None:
            first = frame
        if float(frame.mean()) >= luma_thresh:
            chosen = frame
            break
    cap.release()
    return (chosen if chosen is not None else first), w, h


@app.function(
    image=web_image,
    secrets=[gcs_secret, firebase_secret],
    timeout=600,
    cpu=4,
    memory=8192,
)
def regenerate_first_frame(video_key: str, gcs_path: str = "") -> dict:
    """Server-side (cv2) extraction of a real first frame for court homography.

    The browser-side extraction is unreliable on iOS Safari (it can grab a black
    frame before the decoder has painted). This reads the ACTUAL uploaded MP4 —
    which has no rendering constraints — and writes an authoritative
    first_frame.png + thumbnailPath. Safe to run any time after upload.
    """
    import cv2, tempfile, os
    src = gcs_path or f"{UPLOAD_PREFIX}/{video_key}.mp4"
    local_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    local_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    try:
        download_from_gcs(src, local_vid)
        frame, w, h = _first_nonblack_frame(local_vid)
        if frame is None:
            return {"video_key": video_key, "error": "no_readable_frame"}
        cv2.imwrite(local_png, frame)
        dst = f"{OUTPUT_PREFIX}/{video_key}/first_frame.png"
        upload_to_gcs(local_png, dst)
        # Point the My Videos thumbnail at the freshly extracted frame.
        try:
            fb_app = _init_firebase_admin()
            from firebase_admin import firestore as fb_firestore
            db = fb_firestore.client(fb_app, database_id="default")
            db.collection("videos").document(video_key).update({"thumbnailPath": dst})
        except Exception as e:
            print(f"⚠ thumbnailPath patch failed for {video_key}: {e}")
        print(f"✓ Server first frame written for {video_key} ({w}x{h})")
        return {"video_key": video_key, "ok": True, "src_width": w, "src_height": h, "gcs_path": dst}
    except Exception as e:
        return {"video_key": video_key, "error": str(e)}
    finally:
        for p in (local_vid, local_png):
            try:
                os.unlink(p)
            except Exception:
                pass


@app.function(
    image=web_image,
    secrets=[gcs_secret, firebase_secret],
    timeout=1800,
    cpu=2,
)
def regenerate_all_pending() -> dict:
    """Re-extract first_frame.png (server-side) for every video still awaiting
    court landmarks. Fixes black iOS frames already sitting in the admin queue."""
    jobs = list_pending_landmark_jobs()
    results = []
    for j in jobs:
        vk = j.get("video_key")
        if not vk:
            continue
        results.append(regenerate_first_frame.local(vk))
    ok = sum(1 for r in results if r.get("ok"))
    return {"pending": len(jobs), "regenerated_ok": ok, "results": results}


@app.local_entrypoint()
def backfill_pending_frames():
    """`modal run Modal_app.py::backfill_pending_frames` — regenerate server-side
    first frames for all videos awaiting court landmarks."""
    import json
    print(json.dumps(regenerate_all_pending.remote(), indent=2))


# ====================================================================
# Manual court-landmark tagging (admin-only) — build homography.json
# from 17 hand-clicked landmark pixel coords.
# ====================================================================

# Real-world coordinates per surface (WSF singles standard). Mirrors the
# tables inside `compute_court_homography` — kept in sync so the manual
# path produces the same schema downstream code expects.
_COURT_SHORT_LINE_Y       = 5.44
_COURT_BOX_SIZE           = 1.6
_COURT_BOX_BACK_Y         = _COURT_SHORT_LINE_Y + _COURT_BOX_SIZE     # 7.04 (boxes extend toward back wall)
_COURT_WIDTH_M            = 6.4
_COURT_LENGTH_M           = 9.75
_COURT_OUT_HEIGHT_M       = 4.57
_COURT_TIN_HEIGHT_M       = 0.48
_COURT_SERVICE_HEIGHT_M   = 1.78

_FLOOR_REAL = {
    "Front_Left_Floor_Point":         (0.0, 0.0),
    "Front_Right_Floor_Point":        (_COURT_WIDTH_M, 0.0),
    "T_junction":                      (_COURT_WIDTH_M / 2.0, _COURT_SHORT_LINE_Y),
    "Left_Service_Box_Inner_Front":   (_COURT_BOX_SIZE, _COURT_SHORT_LINE_Y),
    "Left_Service_Box_Inner_Back":    (_COURT_BOX_SIZE, _COURT_BOX_BACK_Y),
    "Right_Service_Box_Inner_Front":  (_COURT_WIDTH_M - _COURT_BOX_SIZE, _COURT_SHORT_LINE_Y),
    "Right_Service_Box_Inner_Back":   (_COURT_WIDTH_M - _COURT_BOX_SIZE, _COURT_BOX_BACK_Y),
    "Left_Short_Point":                (0.0, _COURT_SHORT_LINE_Y),
    "Right_Short_Point":               (_COURT_WIDTH_M, _COURT_SHORT_LINE_Y),
    "Left_Service_Box_Outer_Back":    (0.0, _COURT_BOX_BACK_Y),
    "Right_Service_Box_Outer_Back":   (_COURT_WIDTH_M, _COURT_BOX_BACK_Y),
}
_FRONT_WALL_REAL = {
    "Front_Left_Floor_Point":   (0.0, 0.0),
    "Front_Right_Floor_Point":  (_COURT_WIDTH_M, 0.0),
    "Left_Tin_Point":           (0.0, _COURT_TIN_HEIGHT_M),
    "Right_Tin_Point":          (_COURT_WIDTH_M, _COURT_TIN_HEIGHT_M),
    "Left_Service_Point":       (0.0, _COURT_SERVICE_HEIGHT_M),
    "Right_Service_Point":      (_COURT_WIDTH_M, _COURT_SERVICE_HEIGHT_M),
    "Front_Left_Out_Point":     (0.0, _COURT_OUT_HEIGHT_M),
    "Front_Right_Out_Point":    (_COURT_WIDTH_M, _COURT_OUT_HEIGHT_M),
}
_LEFT_WALL_REAL = {
    "Front_Left_Floor_Point":         (0.0, 0.0),
    "Left_Short_Point":                (_COURT_SHORT_LINE_Y, 0.0),
    "Left_Service_Box_Outer_Back":    (_COURT_BOX_BACK_Y, 0.0),
    "Front_Left_Out_Point":           (0.0, _COURT_OUT_HEIGHT_M),
    "Left_Tin_Point":                  (0.0, _COURT_TIN_HEIGHT_M),
    "Left_Service_Point":              (0.0, _COURT_SERVICE_HEIGHT_M),
}
_RIGHT_WALL_REAL = {
    "Front_Right_Floor_Point":         (0.0, 0.0),
    "Right_Short_Point":                (_COURT_SHORT_LINE_Y, 0.0),
    "Right_Service_Box_Outer_Back":    (_COURT_BOX_BACK_Y, 0.0),
    "Front_Right_Out_Point":           (0.0, _COURT_OUT_HEIGHT_M),
    "Right_Tin_Point":                  (0.0, _COURT_TIN_HEIGHT_M),
    "Right_Service_Point":              (0.0, _COURT_SERVICE_HEIGHT_M),
}
_SURFACE_REAL = {
    "floor":      _FLOOR_REAL,
    "front_wall": _FRONT_WALL_REAL,
    "left_wall":  _LEFT_WALL_REAL,
    "right_wall": _RIGHT_WALL_REAL,
}
_SURFACE_LABEL = {
    "floor":      "X=left-right (0-6.4m), Y=depth from front wall (0-9.75m)",
    "front_wall": "X=left-right (0-6.4m), Y=height from floor (0-4.57m)",
    "left_wall":  "X=depth from front wall (0-9.75m), Y=height from floor",
    "right_wall": "X=depth from front wall (0-9.75m), Y=height from floor",
}

# Ordered list the tagging UI walks through.
COURT_LANDMARK_ORDER = [
    "Front_Left_Out_Point", "Front_Right_Out_Point",
    "Left_Service_Point", "Right_Service_Point",
    "Left_Tin_Point", "Right_Tin_Point",
    "Front_Left_Floor_Point", "Front_Right_Floor_Point",
    "Left_Short_Point", "Right_Short_Point",
    "T_junction",
    "Left_Service_Box_Outer_Back", "Left_Service_Box_Inner_Back",
    "Left_Service_Box_Inner_Front",
    "Right_Service_Box_Outer_Back", "Right_Service_Box_Inner_Back",
    "Right_Service_Box_Inner_Front",
]


def build_and_upload_manual_homography(
    video_key: str,
    landmarks_pixels: dict,
    src_width: int,
    src_height: int,
    tagged_by: str = "admin",
) -> dict:
    """Take 17 manually-tagged landmark pixel coords, fit per-surface
    homographies, and upload the resulting JSON to GCS at the same path
    `compute_court_homography` uses. Returns a summary dict.
    """
    import cv2
    import numpy as np
    import json as json_mod
    import tempfile
    import os as _os

    scaled = {}
    for name, val in landmarks_pixels.items():
        if not (isinstance(val, (list, tuple)) and len(val) >= 2):
            continue
        try:
            x = float(val[0]); y = float(val[1])
        except (TypeError, ValueError):
            continue
        scaled[name] = (int(round(x)), int(round(y)))

    if len(scaled) < 4:
        raise ValueError(
            f"Need ≥4 landmarks, got {len(scaled)}: {sorted(scaled)}")

    def _fit(real_coords):
        common = sorted(set(scaled) & set(real_coords))
        out = {
            "homography_matrix": None,
            "homography_matrix_inverse": None,
            "reprojection_error_meters": None,
            "status": "",
            "landmarks_used": common,
            "num_points": len(common),
            "source": "manual_tagging",
        }
        if len(common) < 4:
            out["status"] = f"insufficient_points ({len(common)} < 4)"
            return out
        src_pts = np.array([scaled[k] for k in common], dtype=np.float32)
        dst_pts = np.array([real_coords[k] for k in common], dtype=np.float32)
        H = None
        for method in (cv2.RANSAC, 0, cv2.LMEDS):
            H, _ = cv2.findHomography(src_pts, dst_pts, method, 5.0)
            if H is not None and H.shape == (3, 3):
                break
        if H is None or H.shape != (3, 3):
            out["status"] = "computation_failed"
            return out
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            out["status"] = "singular_matrix"
            return out
        projected = cv2.perspectiveTransform(
            src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.sqrt(np.sum((projected - dst_pts) ** 2, axis=1))
        out["homography_matrix"] = H.tolist()
        out["homography_matrix_inverse"] = H_inv.tolist()
        out["reprojection_error_meters"] = round(float(errors.mean()), 6)
        out["status"] = "success"
        return out

    surface_results = {name: _fit(coords) for name, coords in _SURFACE_REAL.items()}
    floor = surface_results["floor"]

    def _region(corner_keys):
        pts = [scaled[k] for k in corner_keys if k in scaled]
        if len(pts) < 3:
            return None
        arr = np.array(pts, dtype=np.float32)
        try:
            hull = cv2.convexHull(arr).reshape(-1, 2).tolist()
        except Exception:
            hull = pts
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        bbox = {
            "x_min": int(max(0, min(xs))),
            "y_min": int(max(0, min(ys))),
            "x_max": int(min(src_width, max(xs))),
            "y_max": int(min(src_height, max(ys))),
        }
        return {
            "polygon": [[int(p[0]), int(p[1])] for p in hull],
            "bounding_box": bbox,
            "area_pixels": (bbox["x_max"] - bbox["x_min"]) * (bbox["y_max"] - bbox["y_min"]),
        }

    regions = {
        "floor": _region(["Front_Left_Floor_Point", "Front_Right_Floor_Point",
                          "Right_Short_Point", "Left_Short_Point"]),
        "front_wall": _region(["Front_Left_Floor_Point", "Front_Right_Floor_Point",
                               "Front_Right_Out_Point", "Front_Left_Out_Point"]),
        "left_wall": _region(["Front_Left_Floor_Point", "Front_Left_Out_Point",
                              "Left_Service_Point", "Left_Tin_Point", "Left_Short_Point"]),
        "right_wall": _region(["Front_Right_Floor_Point", "Front_Right_Out_Point",
                               "Right_Service_Point", "Right_Tin_Point", "Right_Short_Point"]),
    }

    output = {
        "source_resolution": {"width": src_width, "height": src_height},
        "landmark_detector": "manual_tagging",
        "tagged_by": tagged_by,
        "court_landmarks_pixels": {k: list(v) for k, v in scaled.items()},
        "landmark_source": {k: "detected" for k in scaled},
        "homography_matrix": floor.get("homography_matrix"),
        "homography_matrix_inverse": floor.get("homography_matrix_inverse"),
        "reprojection_error_meters": floor.get("reprojection_error_meters"),
        "num_calibration_points": floor.get("num_points"),
        "court_landmarks_meters": {
            k: list(_FLOOR_REAL[k]) for k in floor.get("landmarks_used") or []
        },
        "homographies": {
            name: {
                **res,
                "coordinate_system": _SURFACE_LABEL[name],
                "real_world_coords": {
                    k: list(_SURFACE_REAL[name][k])
                    for k in res.get("landmarks_used") or []
                },
                "pixel_region": regions[name],
            }
            for name, res in surface_results.items()
        },
    }

    gcs_path = f"{OUTPUT_PREFIX}/{video_key}/homography.json"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json_mod.dump(output, tf, indent=2)
        tf_path = tf.name
    url = upload_to_gcs(tf_path, gcs_path)
    _os.unlink(tf_path)

    return {
        "status": "ok",
        "url": url,
        "gcs_path": gcs_path,
        "reprojection_error": floor.get("reprojection_error_meters"),
        "surfaces_computed": sum(
            1 for h in surface_results.values() if h.get("status") == "success"
        ),
    }


# ====================================================================
# Admin auth: Firebase ID token verification + custom-claim check.
# Used by /admin/* FastAPI routes.
# ====================================================================

def _init_firebase_admin():
    """Initialise firebase-admin once per process from the Modal secret."""
    import firebase_admin
    if firebase_admin._apps:
        return firebase_admin.get_app()
    from firebase_admin import credentials
    sa_blob = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if not sa_blob:
        raise RuntimeError(
            "FIREBASE_SERVICE_ACCOUNT_JSON env var missing — provision the "
            "`firebase-admin-creds` Modal secret on this function.")
    sa = json.loads(sa_blob)
    return firebase_admin.initialize_app(credentials.Certificate(sa))


def _mark_video_status(video_key: str, status: str):
    """Write the given status to videos/{video_key} in Firestore.

    Called from the WebSocket handler after tracking completes (or errors) so
    the video's status updates even if the user has closed the browser tab.
    Silently no-ops on any failure — status writes are best-effort; the client
    still writes on complete when the tab is open.
    """
    if not video_key:
        return
    try:
        app = _init_firebase_admin()
        from firebase_admin import firestore as fb_firestore
        # This project uses a NAMED Firestore database ("default" as a literal
        # string, NOT the special "(default)" db). Omitting database_id here
        # 404s. See _init_firebase_admin docstring + line 334 pattern.
        db = fb_firestore.client(app, database_id="default")
        db.collection("videos").document(video_key).update({"status": status})
    except Exception as e:
        print(f"⚠ _mark_video_status({video_key}, {status}) failed: {e}")


# ============================================================================
# Decoupled tracking status file: written by the spawned run_tracking_loop
# function, read by the WS handler (and the /api/tracking-status endpoint).
# Lives at gs://.../video-data/{video_key}/tracking_status.json so both the
# writer (spawn, on a GPU container) and reader (web, on CPU) can share it
# without needing shared volumes. Small JSON, cheap to rewrite each segment.
# ============================================================================
def _tracking_status_gcs_path(video_key: str) -> str:
    return f"{OUTPUT_PREFIX}/{video_key}/tracking_status.json"


def write_tracking_status(video_key: str, status: dict) -> None:
    """Upload the given status dict as tracking_status.json to GCS.

    Called after every segment (and on every state transition) by the spawned
    tracking function. Idempotent — always overwrites. Best-effort: failure
    just gets logged (the spawn keeps grinding; the WS just won't see updates).
    """
    if not video_key:
        return
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(status, tf)
            tmp_path = tf.name
        upload_to_gcs(tmp_path, _tracking_status_gcs_path(video_key))
        os.unlink(tmp_path)
    except Exception as e:
        print(f"⚠ write_tracking_status({video_key}) failed: {e}")


def read_tracking_status(video_key: str):
    """Download tracking_status.json for the given video from GCS.

    Returns the parsed dict, or None if the file doesn't exist yet (spawn
    hasn't gotten around to writing it, or the video was never tracked).
    """
    if not video_key:
        return None
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="rb", suffix=".json", delete=False) as tf:
            tmp_path = tf.name
        download_from_gcs(_tracking_status_gcs_path(video_key), tmp_path)
        with open(tmp_path, "r") as f:
            data = json.load(f)
        os.unlink(tmp_path)
        return data
    except Exception:
        # Includes "file doesn't exist yet" — silent by design; caller decides.
        return None


def verify_admin_token(authorization_header: str) -> dict:
    """Verify a Firebase ID token from an `Authorization: Bearer …` header
    and confirm the user has admin role. Returns decoded token claims on
    success, raises ValueError on any failure (caller maps to HTTP 401/403)."""
    if not authorization_header or not authorization_header.startswith("Bearer "):
        raise ValueError("missing_bearer_token")
    token = authorization_header[len("Bearer "):].strip()
    if not token:
        raise ValueError("empty_bearer_token")
    _init_firebase_admin()
    from firebase_admin import auth as fb_auth
    try:
        decoded = fb_auth.verify_id_token(token, check_revoked=False)
    except Exception as e:
        raise ValueError(f"invalid_token:{e}")
    if decoded.get("role") != "admin":
        raise ValueError("not_admin")
    return decoded


def verify_token(authorization_header: str) -> dict:
    """Verify a Firebase ID token from an `Authorization: Bearer …` header for a
    regular (non-admin) signed-in user. Returns decoded claims (uid, email, …) on
    success, raises ValueError on any failure (caller maps to HTTP 401)."""
    if not authorization_header or not authorization_header.startswith("Bearer "):
        raise ValueError("missing_bearer_token")
    token = authorization_header[len("Bearer "):].strip()
    if not token:
        raise ValueError("empty_bearer_token")
    _init_firebase_admin()
    from firebase_admin import auth as fb_auth
    try:
        return fb_auth.verify_id_token(token, check_revoked=False)
    except Exception as e:
        raise ValueError(f"invalid_token:{e}")


# ---- Subscription / billing state on users/{uid} --------------------------
# Written ONLY by the Stripe webhook (Admin SDK). The client reads these fields
# but must never write them (future strict Firestore rules must enforce that).

_ACTIVE_SUB_STATUSES = {"active", "trialing"}


def get_user_billing(uid: str) -> dict:
    """Read the billing fields from users/{uid}. Returns a dict with at least
    `subscriptionStatus` (defaults to "none" for users who never subscribed)."""
    default = {"subscriptionStatus": "none", "plan": "free"}
    if not uid:
        return default
    try:
        app = _init_firebase_admin()
        from firebase_admin import firestore as fb_firestore
        db = fb_firestore.client(app, database_id="default")
        snap = db.collection("users").document(uid).get()
        data = snap.to_dict() if snap.exists else None
        if not data:
            return default
        return {
            "stripeCustomerId": data.get("stripeCustomerId"),
            "subscriptionId": data.get("subscriptionId"),
            "subscriptionStatus": data.get("subscriptionStatus") or "none",
            "plan": data.get("plan") or "free",
            "currentPeriodEnd": data.get("currentPeriodEnd"),
        }
    except Exception as e:
        print(f"⚠ get_user_billing({uid}) failed: {e}")
        return default


def set_user_billing(uid: str, fields: dict):
    """Merge billing fields into users/{uid}. Best-effort (mirrors
    _mark_video_status). Called from the Stripe webhook handler."""
    if not uid:
        return
    try:
        app = _init_firebase_admin()
        from firebase_admin import firestore as fb_firestore
        db = fb_firestore.client(app, database_id="default")
        db.collection("users").document(uid).set(fields, merge=True)
    except Exception as e:
        print(f"⚠ set_user_billing({uid}) failed: {e}")


def has_active_subscription(billing: dict) -> bool:
    """True if the given billing dict represents an access-granting subscription."""
    return (billing or {}).get("subscriptionStatus") in _ACTIVE_SUB_STATUSES


def find_uid_by_stripe_customer(customer_id: str):
    """Reverse-lookup a Firebase uid from a Stripe customer id, using the
    denormalized `stripeCustomerId` field on user docs. Fallback path for
    webhook events whose object lacks our firebaseUid metadata."""
    if not customer_id:
        return None
    try:
        app = _init_firebase_admin()
        from firebase_admin import firestore as fb_firestore
        db = fb_firestore.client(app, database_id="default")
        docs = (db.collection("users")
                  .where("stripeCustomerId", "==", customer_id)
                  .limit(1).stream())
        for d in docs:
            return d.id
    except Exception as e:
        print(f"⚠ find_uid_by_stripe_customer({customer_id}) failed: {e}")
    return None


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
    # Allow up to N game-containers alive at once so the games of a multi-game
    # match (and concurrent uploads from other users) run on separate GPUs in
    # parallel. Bounded to cap cost / stay within the account GPU quota; games
    # beyond the cap simply queue. Tune to your Modal A100 quota.
    max_containers=20,
)
class TrackerGPU:
    # Parametrized by the game's job_id. Modal maintains a SEPARATE autoscaling
    # container pool per distinct parameter value, so each game's segments all
    # route to their own dedicated A100 container — the games in a multi-game
    # match process in PARALLEL on separate GPUs instead of being multiplexed
    # onto one shared warm container. Defaults to "" for non-per-game callers
    # (e.g. extract_first_frame), which share the default pool.
    job_key: str = modal.parameter(default="")

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
        local_frame = f"{DATA_DIR}/jobs/{job_id}/first_frame.png"
        cv2.imwrite(local_frame, frame)
        scratch_vol.commit()

        # Upload to Firebase if video_key provided
        first_frame_url = None
        if video_key:
            try:
                gcs_path = f"{OUTPUT_PREFIX}/{video_key}/first_frame.png"
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
            mask_data = []  # Per-frame polygon contours (source-resolution)
            mask_cache = {}
            last_progress_time = time.time()

            # Polygon simplification ε at proc resolution. Stored in source-resolution px.
            # 2.0 → ~27 vertices/polygon median (~3 MB/2615 frames)
            # 0.5 → ~80-100 vertices/polygon median (~10 MB/2615 frames) — finer detail
            POLY_EPS_PROC = 2.0

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

                mask_entry = {
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

                        # ---- Polygon contours (source-resolution px) ----
                        try:
                            contours, _ = cv2.findContours(
                                m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            polygons = []
                            for c in contours:
                                if len(c) < 3:
                                    continue
                                simplified = cv2.approxPolyDP(c, POLY_EPS_PROC, True)
                                if len(simplified) < 3:
                                    continue
                                pts_src = [
                                    [int(round(float(pt[0][0]) * sx)),
                                     int(round(float(pt[0][1]) * sy))]
                                    for pt in simplified
                                ]
                                polygons.append(pts_src)

                            if polygons:
                                mask_entry[key] = {
                                    "polygons": polygons,
                                    "bbox_xyxy": [
                                        int(round(bbox[0] * sx)),
                                        int(round(bbox[1] * sy)),
                                        int(round(bbox[2] * sx)),
                                        int(round(bbox[3] * sy)),
                                    ],
                                    "area_px": int(round(area * sx * sy)),
                                }
                        except Exception as _e:
                            # Don't fail tracking if polygon extraction hits an edge case
                            pass
                    masks_for_frame[pid] = m

                tracking_data.append(entry)
                mask_data.append(mask_entry)
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
            "mask_data": mask_data,
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

# --- Court landmark detection: cheap Gemini seed + local line fitter ---
# One small Gemini vision call gives ROUGH seed pixels (the model is
# bad at precision but okay at "which line is which"). The deterministic
# pipeline below refines those seeds by fitting painted lines and
# intersecting them — that's where accuracy comes from. One bounded
# corrective re-seed if the guards flag a problem.

_COURT_LANDMARKS = [
    "Front_Left_Floor_Point", "Front_Right_Floor_Point",
    "Left_Service_Point", "Right_Service_Point",
    "Left_Tin_Point", "Right_Tin_Point",
    "Front_Left_Out_Point", "Front_Right_Out_Point",
    "Left_Short_Point", "Right_Short_Point",
    "T_junction",
    "Left_Service_Box_Inner_Front", "Left_Service_Box_Inner_Back",
    "Left_Service_Box_Outer_Back",
    "Right_Service_Box_Inner_Front", "Right_Service_Box_Inner_Back",
    "Right_Service_Box_Outer_Back",
]
_COURT_SEED_NAMES = [
    "Front_Left_Out_Point", "Front_Right_Out_Point",
    "Left_Service_Point", "Right_Service_Point",
    "Left_Tin_Point", "Right_Tin_Point",
    "Front_Left_Floor_Point", "Front_Right_Floor_Point",
    "Left_Short_Point", "Right_Short_Point", "T_junction",
    "Left_Service_Box_Outer_Back", "Left_Service_Box_Inner_Back",
    "Right_Service_Box_Outer_Back", "Right_Service_Box_Inner_Back",
]
_COURT_PRICES = {  # USD per token (input, output) — Gemini API rates
    "gemini-3-flash":   (3e-7, 25e-7),
    "gemini-2.5-flash": (3e-7, 25e-7),
    "gemini-2.5-pro":   (125e-8, 1e-5),
}
_COURT_SEED_PROMPT = """You are seeding a deterministic squash-court line detector. Look at the FRAME (a court photo, {W}x{H} px) and the REFERENCE DIAGRAM (line names). Give ROUGH pixel estimates only — the code refines them precisely, so +/-30px is fine. Do NOT try to be exact.

Return ONLY this JSON (coordinates in the FRAME's {W}x{H} pixel space):
{{"SEEDS": {{ {names} }}, "PLAYERS": [[x_min,x_max], ...]}}

- Each SEED is [x,y]. The front wall is the FAR wall; its Out/Service/Tin/Floor lines are top->bottom on that wall (often small/distant — seed them ON that wall, never at the image top edge).
- PLAYERS = column x-ranges covering each player's body so the fitter skips them; [] if no players.
{feedback}Return only the JSON object, no prose."""


def _court_two_pt(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    m = (y2 - y1) / (x2 - x1)
    return m, y1 - m * x1


def _court_fit_h(gray, x0, x1, m, b, band=24, thr=150, exclude=None, res=2.5):
    import numpy as np
    H = gray.shape[0]; Q = None
    for _ in range(8):
        P = []
        for x in range(int(x0), int(x1)):
            if exclude and any(a <= x <= c for a, c in exclude):
                continue
            yp = m * x + b
            lo, hi = max(0, int(yp - band)), min(H, int(yp + band))
            col = gray[lo:hi, x]
            if len(col) < 3:
                continue
            j = int(np.argmin(col))
            if col[j] < thr and col[j] < col.mean() - 10:
                P.append((x, lo + j))
        if len(P) < 8:
            break
        P = np.array(P); m, b = np.polyfit(P[:, 0], P[:, 1], 1)
        r = np.abs(P[:, 1] - (m * P[:, 0] + b)); Q = P[r < res]
        if len(Q) >= 8:
            m, b = np.polyfit(Q[:, 0], Q[:, 1], 1)
        band = max(5, band - 2)
    return float(m), float(b), (0 if Q is None else len(Q))


def _court_fit_v(gray, y0, y1, m, b, band=18, thr=150, res=2.5):
    import numpy as np
    W = gray.shape[1]; Q = None
    for _ in range(8):
        P = []
        for y in range(int(y0), int(y1)):
            xp = m * y + b
            lo, hi = max(0, int(xp - band)), min(W, int(xp + band))
            row = gray[y, lo:hi]
            if len(row) < 3:
                continue
            j = int(np.argmin(row))
            if row[j] < thr:
                P.append((lo + j, y))
        if len(P) < 8:
            break
        P = np.array(P); m, b = np.polyfit(P[:, 1], P[:, 0], 1)
        r = np.abs(P[:, 0] - (m * P[:, 1] + b)); Q = P[r < res]
        if len(Q) >= 8:
            m, b = np.polyfit(Q[:, 1], Q[:, 0], 1)
        band = max(4, band - 2)
    return float(m), float(b), (0 if Q is None else len(Q))


def _court_fit_seam(gray, wood, x0, x1, m, b, band=34):
    import numpy as np
    H = gray.shape[0]; best = None
    for mode in ("color", "dark"):
        mm, bb, Q, bd = m, b, None, band
        for _ in range(8):
            P = []
            for x in range(int(x0), int(x1)):
                yp = mm * x + bb
                lo, hi = max(0, int(yp - bd)), min(H, int(yp + bd))
                if hi - lo < 5:
                    continue
                if mode == "color":
                    seg = wood[lo:hi, x]
                    g = np.gradient(np.convolve(seg, np.ones(5) / 5, "same"))
                    j = int(np.argmax(g))
                    if g[j] > 3 and seg[j:j + 6].mean() > 30:
                        P.append((x, lo + j))
                else:
                    seg = gray[lo:hi, x]
                    j = int(np.argmin(seg))
                    if seg[j] < 172:
                        P.append((x, lo + j))
            if len(P) < 10:
                break
            P = np.array(P); mm, bb = np.polyfit(P[:, 0], P[:, 1], 1)
            r = np.abs(P[:, 1] - (mm * P[:, 0] + bb)); Q = P[r < 3]
            if len(Q) >= 10:
                mm, bb = np.polyfit(Q[:, 0], Q[:, 1], 1)
            bd = max(8, bd - 3)
        n = 0 if Q is None else len(Q)
        if abs(mm) > 0.15 and (best is None or n > best[2]):
            best = (float(mm), float(bb), n)
    return best if best else (m, b, 0)


def _court_ih(a, b):
    m1, b1 = a[:2]; m2, b2 = b[:2]
    x = (b2 - b1) / (m1 - m2)
    return x, m1 * x + b1


def _court_iv(h, v):
    mh, bh = h[:2]; mv, bv = v[:2]
    y = (mh * bv + bh) / (1 - mh * mv)
    return mv * y + bv, y


def _court_homog(s, d):
    import numpy as np
    A = []
    for (x, y), (u, v) in zip(s, d):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    h = np.linalg.lstsq(np.array(A), np.array(d, float).reshape(-1), rcond=None)[0]
    return np.append(h, 1).reshape(3, 3)


def _court_ap(H, X, Y):
    import numpy as np
    w = H @ np.array([X, Y, 1.0])
    return w[0] / w[2], w[1] / w[2]


def _court_fit_pipeline(image, S, players):
    import numpy as np
    im = np.asarray(image.convert("RGB")).astype(float)
    gray = im.mean(2); wood = im[:, :, 0] - im[:, :, 2]
    Hh, Ww = gray.shape; ex = players

    def xr(a, b, pad=40):
        return (min(S[a][0], S[b][0]) - pad, max(S[a][0], S[b][0]) + pad)

    L = {}
    L["front_out"] = _court_fit_h(
        gray, *xr("Front_Left_Out_Point", "Front_Right_Out_Point"),
        *_court_two_pt(S["Front_Left_Out_Point"], S["Front_Right_Out_Point"]),
        exclude=ex)
    L["front_service"] = _court_fit_h(
        gray, *xr("Left_Service_Point", "Right_Service_Point"),
        *_court_two_pt(S["Left_Service_Point"], S["Right_Service_Point"]),
        exclude=ex)
    L["front_tin"] = _court_fit_h(
        gray, *xr("Left_Tin_Point", "Right_Tin_Point"),
        *_court_two_pt(S["Left_Tin_Point"], S["Right_Tin_Point"]),
        exclude=ex)
    L["front_base"] = _court_fit_h(
        gray, *xr("Front_Left_Floor_Point", "Front_Right_Floor_Point"),
        *_court_two_pt(S["Front_Left_Floor_Point"], S["Front_Right_Floor_Point"]),
        exclude=ex)
    L["left_wall_floor"] = _court_fit_seam(
        gray, wood,
        min(S["Left_Short_Point"][0], S["Front_Left_Floor_Point"][0]) - 30,
        max(S["Left_Short_Point"][0], S["Front_Left_Floor_Point"][0]) + 10,
        *_court_two_pt(S["Front_Left_Floor_Point"], S["Left_Short_Point"]))
    L["right_wall_floor"] = _court_fit_seam(
        gray, wood,
        min(S["Front_Right_Floor_Point"][0], S["Right_Short_Point"][0]) - 10,
        max(S["Front_Right_Floor_Point"][0], S["Right_Short_Point"][0]) + 30,
        *_court_two_pt(S["Front_Right_Floor_Point"], S["Right_Short_Point"]))
    L["short"] = _court_fit_h(
        gray, *xr("Left_Short_Point", "Right_Short_Point", 10),
        *_court_two_pt(S["Left_Short_Point"], S["Right_Short_Point"]),
        exclude=ex)
    L["half_court"] = _court_fit_v(
        gray, S["T_junction"][1] + 8, S["T_junction"][1] + 200,
        0.0, S["T_junction"][0])
    L["left_box_back"] = _court_fit_h(
        gray,
        min(S["Left_Service_Box_Outer_Back"][0], S["Left_Service_Box_Inner_Back"][0]) - 10,
        max(S["Left_Service_Box_Outer_Back"][0], S["Left_Service_Box_Inner_Back"][0]) + 10,
        *_court_two_pt(S["Left_Service_Box_Outer_Back"], S["Left_Service_Box_Inner_Back"]))
    L["right_box_back"] = _court_fit_h(
        gray,
        min(S["Right_Service_Box_Inner_Back"][0], S["Right_Service_Box_Outer_Back"][0]) - 10,
        max(S["Right_Service_Box_Inner_Back"][0], S["Right_Service_Box_Outer_Back"][0]) + 10,
        *_court_two_pt(S["Right_Service_Box_Inner_Back"], S["Right_Service_Box_Outer_Back"]))

    warn = []
    EDGE = max(35, int(0.04 * Hh))
    HORIZ = ("front_out", "front_service", "front_tin", "front_base",
             "short", "left_box_back", "right_box_back")
    for k, ln in L.items():
        if k in HORIZ:
            yc = ln[0] * (Ww / 2) + ln[1]
            if yc < EDGE or yc > Hh - EDGE:
                warn.append(f"{k} near image border (y={yc:.0f}); re-seed on the painted line")
            if abs(ln[0]) > 0.10:
                warn.append(f"{k} slope {ln[0]:.3f} too steep; re-seed")
        if "wall_floor" in k and abs(ln[0]) < 0.20:
            warn.append(f"{k} slope {ln[0]:.3f} too shallow; re-seed endpoints")
        if ln[2] < 50:
            warn.append(f"{k} low inliers ({ln[2]})")
    ob, sv, tn, bs = L["front_out"][1], L["front_service"][1], L["front_tin"][1], L["front_base"][1]
    if not (ob < sv < tn < bs):
        warn.append("front-wall lines out of order; check out/service/tin/base seeds")
    elif (sv - ob) > 0:
        r2, r3 = (tn - sv) / (sv - ob), (bs - tn) / (sv - ob)
        if not (0.35 < r2 < 0.75):
            warn.append(f"front service->tin spacing off (ratio {r2:.2f}, expect ~0.51); out line likely mis-seeded")
        if not (0.08 < r3 < 0.30):
            warn.append(f"front tin->base spacing off (ratio {r3:.2f}, expect ~0.16)")

    FLF = _court_ih(L["front_base"], L["left_wall_floor"])
    FRF = _court_ih(L["front_base"], L["right_wall_floor"])
    VP  = _court_ih(L["left_wall_floor"], L["right_wall_floor"])
    xL, xR = FLF[0], FRF[0]
    at = lambda ln, x: (x, ln[0] * x + ln[1])
    P = {
        "Front_Left_Out_Point":    at(L["front_out"], xL),
        "Front_Right_Out_Point":   at(L["front_out"], xR),
        "Left_Service_Point":      at(L["front_service"], xL),
        "Right_Service_Point":     at(L["front_service"], xR),
        "Left_Tin_Point":          at(L["front_tin"], xL),
        "Right_Tin_Point":         at(L["front_tin"], xR),
        "Front_Left_Floor_Point":  FLF,
        "Front_Right_Floor_Point": FRF,
        "Left_Short_Point":        _court_ih(L["short"], L["left_wall_floor"]),
        "Right_Short_Point":       _court_ih(L["short"], L["right_wall_floor"]),
        "T_junction":              _court_iv(L["short"], L["half_court"]),
        "Left_Service_Box_Outer_Back":  _court_ih(L["left_box_back"], L["left_wall_floor"]),
        "Right_Service_Box_Outer_Back": _court_ih(L["right_box_back"], L["right_wall_floor"]),
    }
    src = {k: "detected" for k in P}
    fr = {
        "Front_Left_Floor_Point":  (0, 0),
        "Front_Right_Floor_Point": (6.4, 0),
        "Left_Short_Point":        (0, 5.49),
        "Right_Short_Point":       (6.4, 5.49),
        "T_junction":              (3.2, 5.49),
    }
    Hm = _court_homog([fr[k] for k in fr], [P[k] for k in fr])
    for k, (X, Y) in {
        "Left_Service_Box_Inner_Front":  (1.6, 5.49),
        "Left_Service_Box_Inner_Back":   (1.6, 7.09),
        "Right_Service_Box_Inner_Front": (4.8, 5.49),
        "Right_Service_Box_Inner_Back":  (4.8, 7.09),
    }.items():
        P[k] = _court_ap(Hm, X, Y); src[k] = "mapped"
    # Snap inner-back points onto the detected back lines (combines both cues).
    P["Left_Service_Box_Inner_Back"] = (
        P["Left_Service_Box_Inner_Back"][0],
        L["left_box_back"][0] * P["Left_Service_Box_Inner_Back"][0] + L["left_box_back"][1],
    )
    P["Right_Service_Box_Inner_Back"] = (
        P["Right_Service_Box_Inner_Back"][0],
        L["right_box_back"][0] * P["Right_Service_Box_Inner_Back"][0] + L["right_box_back"][1],
    )

    checks = {
        "vp_x_vs_T_x_px": round(abs(VP[0] - P["T_junction"][0]), 1),
        "back_lines_parallel": abs(L["left_box_back"][0] - L["right_box_back"][0]) < 0.02,
        "corners_symmetric": abs((xL + xR) / 2 - P["T_junction"][0]) < 25,
    }
    if checks["vp_x_vs_T_x_px"] > 30:
        warn.append(f"VP.x vs T.x off by {checks['vp_x_vs_T_x_px']}px")

    return {
        "source_resolution": {"width": Ww, "height": Hh},
        "court_landmarks_pixels": {
            k: [round(P[k][0], 1), round(P[k][1], 1)] for k in _COURT_LANDMARKS
        },
        "landmark_source": src,
        "fitted_lines": {
            k: {"m": round(v[0], 5), "b": round(v[1], 2), "inliers": v[2]}
            for k, v in L.items()
        },
        "depth_vanishing_point": [round(VP[0], 1), round(VP[1], 1)],
        "consistency_checks": checks,
        "warnings": warn,
    }


class _CourtUsage:
    """Token-count shim so the caller can keep using ``usage.input_tokens``
    / ``usage.output_tokens`` regardless of which SDK we're on."""
    __slots__ = ("input_tokens", "output_tokens")

    def __init__(self, input_tokens=0, output_tokens=0):
        self.input_tokens = int(input_tokens or 0)
        self.output_tokens = int(output_tokens or 0)


def _court_ask_seeds(client, model, frame, diagram, W, H, feedback=""):
    import re, json as json_mod
    from google.genai import types
    names = ", ".join(f'"{n}": [x,y]' for n in _COURT_SEED_NAMES)
    fb = (f"\nNOTE: the previous attempt produced these warnings — fix the "
          f"named seeds:\n{feedback}\n" if feedback else "")
    prompt = _COURT_SEED_PROMPT.format(W=W, H=H, names=names, feedback=fb)
    contents = [prompt, "FRAME:", frame]
    if diagram is not None:
        contents += ["REFERENCE DIAGRAM:", diagram]
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            response_mime_type="application/json",
        ),
    )
    txt = (resp.text or "").strip()
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON in seed response: {txt[:500]}")
    data = json_mod.loads(m.group(0))
    seeds = {k: [float(a) for a in v] for k, v in data["SEEDS"].items()}
    players = [[int(a), int(b)] for a, b in data.get("PLAYERS", [])]
    meta = getattr(resp, "usage_metadata", None)
    usage = _CourtUsage(
        input_tokens=getattr(meta, "prompt_token_count", 0) if meta else 0,
        output_tokens=getattr(meta, "candidates_token_count", 0) if meta else 0,
    )
    return seeds, players, usage


@app.function(
    image=gemini_image,
    volumes={DATA_DIR: scratch_vol},
    secrets=[gemini_secret, gcs_secret],
    timeout=1800,
)
def compute_court_homography(job_id: str, video_key: str, src_width: int, src_height: int):
    """Detect court landmarks via a cheap Gemini Flash seed call +
    deterministic local line-fitting pipeline, then derive plane
    homographies and upload to Firebase.

    Flow:
      * Gemini Flash (default; override with COURT_LANDMARK_MODEL env var)
        sees the frame + reference diagram and returns ~rough seed pixels
        for 15 landmarks plus player x-ranges to exclude;
      * the local pipeline fits painted lines and intersects them to land
        every landmark within a few pixels;
      * if guards flag a problem, one corrective re-seed is attempted with
        the warnings fed back as feedback.
    """
    import cv2
    import numpy as np
    import json as json_mod
    import os
    import tempfile

    scratch_vol.reload()

    # ---- Load first frame from volume ----
    first_frame_path = f"{DATA_DIR}/jobs/{job_id}/first_frame.png"
    if not os.path.exists(first_frame_path):
        video_path = f"{DATA_DIR}/jobs/{job_id}/source.mp4"
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return {"status": "error", "message": "Could not read first frame"}
        cv2.imwrite(first_frame_path, frame)

    # ---- Real-world coordinates per surface (WSF singles standard) ----
    # Short line: 5.44m from front wall; service box: 1.6m × 1.6m squares
    # whose front edge lies on the short line and which extend toward the
    # back wall (back edge at 7.04m from front wall).
    SHORT_LINE_Y = 5.44
    BOX_SIZE     = 1.6
    BOX_BACK_Y   = SHORT_LINE_Y + BOX_SIZE   # 7.04m from front wall
    COURT_WIDTH  = 6.4
    COURT_LENGTH = 9.75
    WALL_HEIGHT_OUT = 4.57
    WALL_TIN_HEIGHT = 0.48
    WALL_SERVICE_HEIGHT = 1.78

    FLOOR_COORDS = {
        "Front_Left_Floor_Point":         (0.0, 0.0),
        "Front_Right_Floor_Point":        (COURT_WIDTH, 0.0),
        "T_junction":                      (COURT_WIDTH / 2.0, SHORT_LINE_Y),
        "Left_Service_Box_Inner_Front":   (BOX_SIZE, SHORT_LINE_Y),
        "Left_Service_Box_Inner_Back":    (BOX_SIZE, BOX_BACK_Y),
        "Right_Service_Box_Inner_Front":  (COURT_WIDTH - BOX_SIZE, SHORT_LINE_Y),
        "Right_Service_Box_Inner_Back":   (COURT_WIDTH - BOX_SIZE, BOX_BACK_Y),
        "Left_Short_Point":                (0.0, SHORT_LINE_Y),
        "Right_Short_Point":               (COURT_WIDTH, SHORT_LINE_Y),
        "Left_Service_Box_Outer_Back":    (0.0, BOX_BACK_Y),
        "Right_Service_Box_Outer_Back":   (COURT_WIDTH, BOX_BACK_Y),
    }

    FRONT_WALL_COORDS = {
        "Front_Left_Floor_Point":   (0.0, 0.0),
        "Front_Right_Floor_Point":  (COURT_WIDTH, 0.0),
        "Left_Tin_Point":           (0.0, WALL_TIN_HEIGHT),
        "Right_Tin_Point":          (COURT_WIDTH, WALL_TIN_HEIGHT),
        "Left_Service_Point":       (0.0, WALL_SERVICE_HEIGHT),
        "Right_Service_Point":      (COURT_WIDTH, WALL_SERVICE_HEIGHT),
        "Front_Left_Out_Point":     (0.0, WALL_HEIGHT_OUT),
        "Front_Right_Out_Point":    (COURT_WIDTH, WALL_HEIGHT_OUT),
    }

    LEFT_WALL_COORDS = {
        "Front_Left_Floor_Point":         (0.0, 0.0),
        "Left_Short_Point":                (SHORT_LINE_Y, 0.0),
        "Left_Service_Box_Outer_Back":    (BOX_BACK_Y, 0.0),
        "Front_Left_Out_Point":           (0.0, WALL_HEIGHT_OUT),
        "Left_Tin_Point":                  (0.0, WALL_TIN_HEIGHT),
        "Left_Service_Point":              (0.0, WALL_SERVICE_HEIGHT),
    }

    RIGHT_WALL_COORDS = {
        "Front_Right_Floor_Point":         (0.0, 0.0),
        "Right_Short_Point":                (SHORT_LINE_Y, 0.0),
        "Right_Service_Box_Outer_Back":    (BOX_BACK_Y, 0.0),
        "Front_Right_Out_Point":           (0.0, WALL_HEIGHT_OUT),
        "Right_Tin_Point":                  (0.0, WALL_TIN_HEIGHT),
        "Right_Service_Point":              (0.0, WALL_SERVICE_HEIGHT),
    }

    SURFACE_COORDS = {
        "floor":      FLOOR_COORDS,
        "front_wall": FRONT_WALL_COORDS,
        "left_wall":  LEFT_WALL_COORDS,
        "right_wall": RIGHT_WALL_COORDS,
    }
    SURFACE_LABEL = {
        "floor":      ("X=left-right (0-6.4m), Y=depth from front wall (0-9.75m)"),
        "front_wall": ("X=left-right (0-6.4m), Y=height from floor (0-4.57m)"),
        "left_wall":  ("X=depth from front wall (0-9.75m), Y=height from floor"),
        "right_wall": ("X=depth from front wall (0-9.75m), Y=height from floor"),
    }

    # ---- Load frame + reference diagram ----
    from PIL import Image
    frame_pil = Image.open(first_frame_path).convert("RGB")
    diagram_path = "/app/court_reference_annotated.jpg"
    diagram_pil = (Image.open(diagram_path).convert("RGB")
                   if os.path.exists(diagram_path) else None)
    if diagram_pil is None:
        print("  ⚠ Reference diagram not bundled; sending frame only.")
    W_seed, H_seed = frame_pil.size

    # ---- Cheap Gemini Flash seed call + deterministic local refinement ----
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ.get("COURT_LANDMARK_MODEL", "gemini-2.5-flash")
    print(f"  Seeding court landmarks with {model} + local line fitter...")

    MAX_RETRIES = 1
    feedback = ""
    parsed = None
    total_in = total_out = 0
    try:
        for attempt in range(MAX_RETRIES + 1):
            seeds, players, usage = _court_ask_seeds(
                client, model, frame_pil, diagram_pil, W_seed, H_seed, feedback,
            )
            total_in += usage.input_tokens
            total_out += usage.output_tokens
            parsed = _court_fit_pipeline(frame_pil, seeds, players)
            if not parsed.get("warnings"):
                break
            feedback = "\n".join("- " + w for w in parsed["warnings"])
            print(f"  attempt {attempt+1}: warnings → re-seeding\n    "
                  + feedback.replace("\n", "\n    "))
    except Exception as e:
        return {"status": "error", "message": f"Seed + fit pipeline failed: {e}"}

    pin, pout = _COURT_PRICES.get(model, (3e-7, 25e-7))
    est_cost = total_in * pin + total_out * pout
    print(f"  ✓ Landmarks fitted ({len(parsed['court_landmarks_pixels'])} pts, "
          f"~${est_cost:.4f}, warnings={len(parsed.get('warnings', []))})")

    # ---- Normalize landmark coords + surface homographies ----
    landmarks_raw = parsed.get("court_landmarks_pixels") or {}
    landmark_source = parsed.get("landmark_source") or {}
    homographies_raw = parsed.get("homographies") or {}

    # The model is told to output pixels in the source frame's own
    # resolution. If it reports a source_resolution that doesn't match
    # we rescale before consuming.
    reported_res = parsed.get("source_resolution") or {}
    rep_w = int(reported_res.get("width") or src_width)
    rep_h = int(reported_res.get("height") or src_height)
    scale_x = src_width / max(rep_w, 1)
    scale_y = src_height / max(rep_h, 1)
    if abs(scale_x - 1.0) > 1e-3 or abs(scale_y - 1.0) > 1e-3:
        print(f"  Rescaling landmarks {rep_w}x{rep_h} → {src_width}x{src_height}")

    scaled_coords = {}
    for name, val in landmarks_raw.items():
        if not (isinstance(val, (list, tuple)) and len(val) >= 2):
            continue
        try:
            x = float(val[0]) * scale_x
            y = float(val[1]) * scale_y
        except (TypeError, ValueError):
            continue
        scaled_coords[name] = (int(round(x)), int(round(y)))

    if len(scaled_coords) < 4:
        return {"status": "error",
                "message": f"Pipeline returned only {len(scaled_coords)} landmarks "
                           "(need ≥4 for a homography)",
                "raw_excerpt": full_text[:2000]}

    # ---- Pull each surface homography (or recompute as a safety net) ----
    def _matrix_or_none(m):
        if not m:
            return None
        try:
            arr = np.array(m, dtype=np.float64)
        except Exception:
            return None
        if arr.shape != (3, 3):
            return None
        return arr

    def _fit_homography(pixel_coords, real_coords, label):
        common = sorted(set(pixel_coords) & set(real_coords))
        result = {
            "homography_matrix": None,
            "homography_matrix_inverse": None,
            "reprojection_error_meters": None,
            "status": "",
            "landmarks_used": common,
            "num_points": len(common),
        }
        if len(common) < 4:
            result["status"] = f"insufficient_points ({len(common)} < 4)"
            return result
        src_pts = np.array([pixel_coords[k] for k in common], dtype=np.float32)
        dst_pts = np.array([real_coords[k] for k in common], dtype=np.float32)
        H = None
        for method in (cv2.RANSAC, 0, cv2.LMEDS):
            H, _ = cv2.findHomography(src_pts, dst_pts, method, 5.0)
            if H is not None and H.shape == (3, 3):
                break
        if H is None or H.shape != (3, 3):
            result["status"] = "computation_failed"
            return result
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            result["status"] = "singular_matrix"
            return result
        projected = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.sqrt(np.sum((projected - dst_pts) ** 2, axis=1))
        result["homography_matrix"] = H.tolist()
        result["homography_matrix_inverse"] = H_inv.tolist()
        result["reprojection_error_meters"] = round(float(errors.mean()), 6)
        result["status"] = "success"
        return result

    def _build_surface(name, real_coords):
        rh = homographies_raw.get(name) or {}
        H = _matrix_or_none(rh.get("homography_matrix"))
        if H is not None:
            # Trust the reported matrix. Compute the inverse + reprojection error
            # locally against the same landmark set so the schema stays uniform.
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = None
            common = sorted(set(scaled_coords) & set(real_coords))
            err = rh.get("reprojection_error_meters")
            if H_inv is not None and common:
                src_pts = np.array([scaled_coords[k] for k in common], dtype=np.float32)
                dst_pts = np.array([real_coords[k] for k in common], dtype=np.float32)
                projected = cv2.perspectiveTransform(
                    src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
                local_err = float(np.sqrt(np.sum((projected - dst_pts) ** 2, axis=1)).mean())
                # Prefer the locally-computed error when none was reported.
                if err is None:
                    err = round(local_err, 6)
            return {
                "homography_matrix": H.tolist(),
                "homography_matrix_inverse": (H_inv.tolist() if H_inv is not None else None),
                "reprojection_error_meters": err,
                "status": "success",
                "landmarks_used": common,
                "num_points": len(common),
                "source": "gemini_seed_local_fit",
            }
        # Fallback: fit locally from the reported landmarks.
        fitted = _fit_homography(scaled_coords, real_coords, name)
        fitted["source"] = "local_fit"
        return fitted

    surface_results = {
        name: _build_surface(name, coords)
        for name, coords in SURFACE_COORDS.items()
    }
    floor_result = surface_results["floor"]
    front_wall_result = surface_results["front_wall"]
    left_wall_result = surface_results["left_wall"]
    right_wall_result = surface_results["right_wall"]

    for name, res in surface_results.items():
        err = res.get("reprojection_error_meters")
        quality = "✓" if (err is not None and err < 0.1) else ("⚠" if err is not None and err < 0.3 else "✗")
        print(f"  {quality} {name:11s}: error={err}  pts={res.get('num_points')}  "
              f"src={res.get('source','?')}  status={res.get('status')}")

    # ---- Pixel-region polygons (informational; kept for output-shape parity) ----
    def _pixel_region(corner_keys, coords_dict):
        pts = []
        for k in corner_keys:
            if k in coords_dict:
                pts.append(coords_dict[k])
        if len(pts) < 3:
            return None
        arr = np.array(pts, dtype=np.float32)
        try:
            hull = cv2.convexHull(arr).reshape(-1, 2).tolist()
        except Exception:
            hull = pts
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        bbox = {
            "x_min": int(max(0, min(xs))),
            "y_min": int(max(0, min(ys))),
            "x_max": int(min(src_width, max(xs))),
            "y_max": int(min(src_height, max(ys))),
        }
        return {
            "polygon": [[int(p[0]), int(p[1])] for p in hull],
            "bounding_box": bbox,
            "area_pixels": (bbox["x_max"] - bbox["x_min"]) * (bbox["y_max"] - bbox["y_min"]),
        }

    floor_region = _pixel_region(
        ["Front_Left_Floor_Point", "Front_Right_Floor_Point",
         "Right_Short_Point", "Left_Short_Point"], scaled_coords)
    front_wall_region = _pixel_region(
        ["Front_Left_Floor_Point", "Front_Right_Floor_Point",
         "Front_Right_Out_Point", "Front_Left_Out_Point"], scaled_coords)
    left_wall_region = _pixel_region(
        ["Front_Left_Floor_Point", "Front_Left_Out_Point",
         "Left_Service_Point", "Left_Tin_Point", "Left_Short_Point"], scaled_coords)
    right_wall_region = _pixel_region(
        ["Front_Right_Floor_Point", "Front_Right_Out_Point",
         "Right_Service_Point", "Right_Tin_Point", "Right_Short_Point"], scaled_coords)

    # ---- Build the final output blob (schema preserved for downstream) ----
    output = {
        "source_resolution": {"width": src_width, "height": src_height},
        "landmark_detector": f"{model} seed + local_line_fit",
        "court_landmarks_pixels": {k: list(v) for k, v in scaled_coords.items()},
        "landmark_source": landmark_source or None,
        "consistency_checks": parsed.get("consistency_checks"),
        "fitted_lines": parsed.get("fitted_lines"),
        "depth_vanishing_point": parsed.get("depth_vanishing_point"),

        # Floor primary (backwards-compatible top-level keys).
        "homography_matrix": floor_result.get("homography_matrix"),
        "homography_matrix_inverse": floor_result.get("homography_matrix_inverse"),
        "reprojection_error_meters": floor_result.get("reprojection_error_meters"),
        "num_calibration_points": floor_result.get("num_points"),
        "court_landmarks_meters": {
            k: list(FLOOR_COORDS[k]) for k in floor_result.get("landmarks_used") or []
        },

        # All surface homographies + pixel regions + coord systems.
        "homographies": {
            "floor": {
                **floor_result,
                "coordinate_system": SURFACE_LABEL["floor"],
                "real_world_coords": {
                    k: list(FLOOR_COORDS[k]) for k in floor_result.get("landmarks_used") or []
                },
                "pixel_region": floor_region,
            },
            "front_wall": {
                **front_wall_result,
                "coordinate_system": SURFACE_LABEL["front_wall"],
                "real_world_coords": {
                    k: list(FRONT_WALL_COORDS[k]) for k in front_wall_result.get("landmarks_used") or []
                },
                "pixel_region": front_wall_region,
            },
            "left_wall": {
                **left_wall_result,
                "coordinate_system": SURFACE_LABEL["left_wall"],
                "real_world_coords": {
                    k: list(LEFT_WALL_COORDS[k]) for k in left_wall_result.get("landmarks_used") or []
                },
                "pixel_region": left_wall_region,
            },
            "right_wall": {
                **right_wall_result,
                "coordinate_system": SURFACE_LABEL["right_wall"],
                "real_world_coords": {
                    k: list(RIGHT_WALL_COORDS[k]) for k in right_wall_result.get("landmarks_used") or []
                },
                "pixel_region": right_wall_region,
            },
        },
    }

    # ---- Upload to Firebase ----
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
        "surfaces_computed": sum(
            1 for h in surface_results.values() if h.get("status") == "success"
        ),
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
    image=tracknet_image,
    volumes={DATA_DIR: scratch_vol, BALL_WEIGHTS_DIR: ball_weights_vol},
    secrets=[gcs_secret],
    gpu="A10G",
    timeout=1800,
)
def track_ball(job_id: str, video_key: str, src_fps: float, src_width: int, src_height: int, start_frame: int = 0, end_frame: int = None):
    """Run TrackNetV3 ball detection on every frame of the source video.

    TrackNetV3 is a U-Net that takes 8 consecutive RGB frames stacked along the
    channel axis (24 channels at 288x512) and emits 8 sigmoid heatmaps — one per
    input frame. We slide the window with stride=seq_len so every source frame
    is predicted exactly once.

    Args:
        start_frame: Frame to start tracking from (for trimmed videos).
                     Frame indices in output will be relative to the original video.
        end_frame: Frame to stop tracking at (exclusive). Defaults to total_frames.
                   Used for multi-game segmentation so each game's ball track
                   only covers its own frame range.
    """
    import cv2
    import numpy as np
    import json as json_mod
    import os
    import time as _time
    import subprocess
    import threading
    from collections import deque

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    scratch_vol.reload()

    video_path = f"{DATA_DIR}/jobs/{job_id}/source.mp4"
    # prepare_job commits source.mp4 as soon as the GCS download finishes, but
    # spawn ordering + volume propagation can race — wait up to a few minutes
    # for the file to appear before treating it as a real miss.
    if not os.path.exists(video_path):
        import time as _time_wait
        deadline = _time_wait.time() + 300  # 5 min budget
        while _time_wait.time() < deadline:
            _time_wait.sleep(5)
            scratch_vol.reload()
            if os.path.exists(video_path):
                print(f"  ✓ Ball tracking: source video found after wait")
                break
        else:
            print(f"  ✗ Ball tracking: source video not found at {video_path} after 5min wait")
            return {"status": "error", "message": "Source video not found"}

    # ---- TrackNetV3 model definition (qaz812345/TrackNetV3 — model.py) ----
    # Module / attribute names match the released checkpoint exactly so the
    # state_dict loads cleanly (down_block_*/up_block_*/bottleneck, each with
    # conv_1/conv_2(/conv_3), each containing conv + bn + relu). Sigmoid is
    # inside forward; do NOT apply it again at the call site.
    class _Conv2DBlock(nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding="same", bias=False)
            self.bn = nn.BatchNorm2d(c_out)
            self.relu = nn.ReLU()
        def forward(self, x): return self.relu(self.bn(self.conv(x)))

    class _Double2DConv(nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            self.conv_1 = _Conv2DBlock(c_in, c_out)
            self.conv_2 = _Conv2DBlock(c_out, c_out)
        def forward(self, x): return self.conv_2(self.conv_1(x))

    class _Triple2DConv(nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            self.conv_1 = _Conv2DBlock(c_in, c_out)
            self.conv_2 = _Conv2DBlock(c_out, c_out)
            self.conv_3 = _Conv2DBlock(c_out, c_out)
        def forward(self, x): return self.conv_3(self.conv_2(self.conv_1(x)))

    class TrackNet(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.down_block_1 = _Double2DConv(in_dim, 64)
            self.down_block_2 = _Double2DConv(64, 128)
            self.down_block_3 = _Triple2DConv(128, 256)
            self.bottleneck = _Triple2DConv(256, 512)
            self.up_block_1 = _Triple2DConv(768, 256)
            self.up_block_2 = _Double2DConv(384, 128)
            self.up_block_3 = _Double2DConv(192, 64)
            self.predictor = nn.Conv2d(64, out_dim, kernel_size=1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x1 = self.down_block_1(x)
            x = F.max_pool2d(x1, 2, 2)
            x2 = self.down_block_2(x)
            x = F.max_pool2d(x2, 2, 2)
            x3 = self.down_block_3(x)
            x = F.max_pool2d(x3, 2, 2)
            x = self.bottleneck(x)
            x = torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), x3], dim=1)
            x = self.up_block_1(x)
            x = torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), x2], dim=1)
            x = self.up_block_2(x)
            x = torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), x1], dim=1)
            x = self.up_block_3(x)
            x = self.predictor(x)
            return self.sigmoid(x)

    # ---- Load TrackNetV3 weights from volume ----
    weights_path = TRACKNET_WEIGHTS_PATH
    if not os.path.exists(weights_path):
        print(f"  ✗ Ball tracking: TrackNet weights not found at {weights_path}")
        return {"status": "error", "message": "TrackNet weights not found"}

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    param_dict = ckpt.get("param_dict", {}) if isinstance(ckpt, dict) else {}
    seq_len = int(param_dict.get("seq_len", TRACKNET_SEQ_LEN))
    bg_mode = param_dict.get("bg_mode", "")
    # in_dim per qaz812345/TrackNetV3 utils/general.py::get_model
    if bg_mode == "concat":
        model_in_dim = (seq_len + 1) * 3
    elif bg_mode == "subtract":
        model_in_dim = seq_len * 1  # single channel diff per frame
    elif bg_mode == "subtract_concat":
        model_in_dim = seq_len * 4
    else:
        model_in_dim = seq_len * 3

    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    model = TrackNet(in_dim=model_in_dim, out_dim=seq_len).eval()
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        # Retry non-strict to surface a useful key diff for debugging.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"  ⚠ TrackNet load (non-strict): {len(missing)} missing, {len(unexpected)} unexpected")
            print(f"     sample missing: {list(missing)[:3]}")
            print(f"     sample unexpected: {list(unexpected)[:3]}")
        else:
            raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  ✓ TrackNetV3 loaded from {weights_path} (seq_len={seq_len}, bg_mode='{bg_mode}', in_dim={model_in_dim}, device={device})")

    # ---- Compute background image for bg_mode='concat' ----
    # The released checkpoint was trained with the per-rally median frame
    # concatenated to the seq_len input frames (3 extra channels).
    # We approximate per-rally with a per-video median over N evenly-spaced
    # samples at TrackNet input resolution. RGB uint8 float, NO /255.
    bg_tensor = None
    if bg_mode in ("concat", "subtract", "subtract_concat"):
        BG_SAMPLES = 30
        cap_bg = cv2.VideoCapture(video_path)
        bg_total = int(cap_bg.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_idxs = np.linspace(0, max(bg_total - 1, 0), BG_SAMPLES).astype(int)
        bg_frames = []
        for si in sample_idxs:
            cap_bg.set(cv2.CAP_PROP_POS_FRAMES, int(si))
            ok, fr = cap_bg.read()
            if not ok:
                continue
            small = cv2.resize(fr, (TRACKNET_IN_W, TRACKNET_IN_H), interpolation=cv2.INTER_AREA)
            bg_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        cap_bg.release()
        if not bg_frames:
            print(f"  ✗ Background sampling failed — couldn't read any frames")
            return {"status": "error", "message": "Background sampling failed"}
        bg_np = np.median(np.stack(bg_frames, axis=0), axis=0)  # (H,W,3) float64
        bg_tensor = torch.from_numpy(bg_np.astype(np.float32)).permute(2, 0, 1).contiguous()  # (3,H,W)
        print(f"  ✓ Background median computed from {len(bg_frames)}/{BG_SAMPLES} samples")

    # ---- Get video info ----
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"  Ball tracking: {width}x{height} @ {fps:.1f} FPS — {total_frames} frames ({round(total_frames/fps, 1)}s)")

    # Clamp end_frame — client may not know exact video length, and multi-game
    # segmentation uses this to bound each game's ball track to its own range.
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    # Calculate frames to process (from start_frame to end_frame)
    frames_to_process = end_frame - start_frame
    if start_frame > 0 or end_frame < total_frames:
        print(f"  Ball tracking: frames {start_frame} → {end_frame} ({frames_to_process} frames)")

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
    # For TrackNet we feed seq_len=8 frames at 288x512 (input axis stacked along
    # channel dim) and get one heatmap per frame back. We batch BATCH_WINDOWS
    # windows together per forward, so the GPU sees BATCH_WINDOWS*seq_len frames
    # of work at a time.
    BATCH_WINDOWS = 4
    BATCH_SIZE = seq_len * BATCH_WINDOWS  # for queue sizing only
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
            # Stop at end_frame — bounds each game's tracking to its own range
            if idx >= end_frame:
                break
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
        # Re-clamp end_frame against the corrected total, then recompute range.
        if end_frame > total_frames:
            end_frame = total_frames
        frames_to_process = end_frame - start_frame

    # ---- TrackNet sliding-window inference loop ----
    # We collect seq_len consecutive frames into a window, batch BATCH_WINDOWS
    # windows per forward, and harvest one heatmap per input frame. Stride is
    # seq_len (non-overlapping) so each source frame is predicted exactly once.
    #
    # As with the YOLO version we keep ALL contour-derived candidates per frame
    # (not just argmax) so the stuck-ball filter downstream can blacklist
    # stationary blobs using the full detection history.
    raw_detections = {}
    start_time = _time.time()

    scale_x_out = src_width / TRACKNET_IN_W
    scale_y_out = src_height / TRACKNET_IN_H

    def _run_batch(windows_tensors, windows_meta):
        """Run model on a batch of windows; populate raw_detections."""
        if not windows_tensors:
            return
        x = torch.stack(windows_tensors, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            y = model(x)  # (B, seq_len, H, W) sigmoid
        y_np = y.detach().cpu().numpy()
        thresh = TRACKNET_HEATMAP_THRESHOLD
        for w_i, meta in enumerate(windows_meta):
            heatmaps = y_np[w_i]  # (seq_len, H, W)
            for f_i, (idx, ts_raw) in enumerate(meta):
                if idx is None:
                    continue  # padded slot
                if ts_raw is None:
                    ts_raw = frame_timestamps.get(str(idx), idx / fps)
                ts = round(ts_raw, 4)
                hm = heatmaps[f_i]
                # Threshold + contour extraction (matches author predict_location).
                bin_map = (hm >= thresh).astype(np.uint8) * 255
                candidates = []
                if bin_map.any():
                    contours, _ = cv2.findContours(
                        bin_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for c in contours:
                        x_r, y_r, w_r, h_r = cv2.boundingRect(c)
                        cx_l = x_r + w_r / 2.0
                        cy_l = y_r + h_r / 2.0
                        # Confidence: heatmap value at the contour centroid.
                        cyi = min(max(int(round(cy_l)), 0), hm.shape[0] - 1)
                        cxi = min(max(int(round(cx_l)), 0), hm.shape[1] - 1)
                        conf = float(hm[cyi, cxi])
                        cx_src = cx_l * scale_x_out
                        cy_src = cy_l * scale_y_out
                        candidates.append((cx_src, cy_src, conf))
                raw_detections[idx] = {"ts": ts, "candidates": candidates}

    # State: buffers for current window + pending batch.
    cur_window_tensors = []  # list of per-frame (3, H, W) tensors
    cur_window_meta = []     # list of (idx, ts_raw)
    pending_window_tensors = []  # list of stacked (3*seq_len, H, W)
    pending_window_meta = []
    last_frame_tensor = None  # for padding tail window

    def _flush_window():
        nonlocal cur_window_tensors, cur_window_meta
        # Pad shorter windows by repeating the last real frame.
        while len(cur_window_tensors) < seq_len:
            pad_t = last_frame_tensor if last_frame_tensor is not None else torch.zeros(
                (3, TRACKNET_IN_H, TRACKNET_IN_W), dtype=torch.float32
            )
            cur_window_tensors.append(pad_t)
            cur_window_meta.append((None, None))
        if bg_mode == "concat":
            # Author convention: median background prepended (3 + L*3 channels).
            stacked = torch.cat([bg_tensor] + cur_window_tensors, dim=0)
        else:
            stacked = torch.cat(cur_window_tensors, dim=0)
        pending_window_tensors.append(stacked)
        pending_window_meta.append(cur_window_meta)
        cur_window_tensors = []
        cur_window_meta = []

    last_logged_pct = -1

    while True:
        if frame_queue:
            idx, ts_raw, frame = frame_queue.popleft()
            # Resize to TrackNet input, BGR→RGB, to (3, H, W) float32.
            # NOTE: the qaz812345 dataset.py does NOT divide by 255; the
            # heatmap target is normalized, but input frames stay in [0,255].
            small = cv2.resize(frame, (TRACKNET_IN_W, TRACKNET_IN_H), interpolation=cv2.INTER_AREA)
            small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(small).to(torch.float32).permute(2, 0, 1).contiguous()
            last_frame_tensor = t
            cur_window_tensors.append(t)
            cur_window_meta.append((idx, ts_raw))
            if len(cur_window_tensors) == seq_len:
                _flush_window()
        elif decode_done.is_set():
            # Drain: flush partial window then stop.
            if cur_window_tensors:
                _flush_window()
            break
        else:
            _time.sleep(0.001)
            continue

        # Run the model once we have a full batch of windows.
        if len(pending_window_tensors) >= BATCH_WINDOWS:
            _run_batch(pending_window_tensors, pending_window_meta)
            pending_window_tensors = []
            pending_window_meta = []

            # Progress log.
            frames_processed = len(raw_detections)
            if frames_to_process > 0:
                pct = int(frames_processed / frames_to_process * 100)
                if pct // 5 != last_logged_pct // 5:
                    elapsed = _time.time() - start_time
                    fps_actual = frames_processed / max(elapsed, 0.1)
                    raw_detected = sum(1 for d in raw_detections.values() if d["candidates"])
                    print(f"  Ball tracking: {frames_processed}/{frames_to_process} ({pct}%) — {raw_detected} raw detections — {fps_actual:.1f} fps")
                    last_logged_pct = pct

    # Flush any remaining windows in the final partial batch.
    if pending_window_tensors:
        _run_batch(pending_window_tensors, pending_window_meta)

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

    det_rate_pre = round(detected / max(frames_to_process, 1) * 100, 2)
    print(f"  ✓ TrackNet detections: {detected}/{len(frames_result)} ({det_rate_pre}%) "
          f"after stuck-ball filter, took {elapsed:.1f}s")

    # ---- InpaintNet trajectory rectification ----
    # qaz812345/TrackNetV3 InpaintNet (model.py): 1-D U-Net, in (N, L, 3) → out (N, L, 2).
    # Module names below MUST match the released checkpoint exactly (note the
    # upstream "buttleneck" typo) so the state_dict loads with strict=True.
    inpaint_filled = 0
    inpaint_skipped_long_gap = 0
    inpaint_used = False
    if os.path.exists(TRACKNET_INPAINT_WEIGHTS_PATH):
        try:
            class _IPConv(nn.Module):
                def __init__(self, c_in, c_out):
                    super().__init__()
                    self.conv = nn.Conv1d(c_in, c_out, kernel_size=3, padding="same")
                def forward(self, x): return F.leaky_relu(self.conv(x), 0.1)

            class _IPDouble(nn.Module):
                def __init__(self, c_in, c_out):
                    super().__init__()
                    self.conv_1 = _IPConv(c_in, c_out)
                    self.conv_2 = _IPConv(c_out, c_out)
                def forward(self, x): return self.conv_2(self.conv_1(x))

            class InpaintNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.down_1 = _IPConv(3, 32)
                    self.down_2 = _IPConv(32, 64)
                    self.down_3 = _IPConv(64, 128)
                    self.buttleneck = _IPDouble(128, 256)  # sic, matches checkpoint
                    self.up_1 = _IPConv(384, 128)          # 256 + 128 skip
                    self.up_2 = _IPConv(192, 64)           # 128 + 64 skip
                    self.up_3 = _IPConv(96, 32)            # 64 + 32 skip
                    self.predictor = nn.Conv1d(32, 2, kernel_size=3, padding="same")

                def forward(self, x):
                    # x: (N, L, 3) → (N, 3, L)
                    x = x.transpose(1, 2)
                    x1 = self.down_1(x);          p1 = F.max_pool1d(x1, 2)
                    x2 = self.down_2(p1);         p2 = F.max_pool1d(x2, 2)
                    x3 = self.down_3(p2);         p3 = F.max_pool1d(x3, 2)
                    b = self.buttleneck(p3)
                    u1 = F.interpolate(b,  scale_factor=2, mode="nearest")
                    u1 = self.up_1(torch.cat([u1, x3], dim=1))
                    u2 = F.interpolate(u1, scale_factor=2, mode="nearest")
                    u2 = self.up_2(torch.cat([u2, x2], dim=1))
                    u3 = F.interpolate(u2, scale_factor=2, mode="nearest")
                    u3 = self.up_3(torch.cat([u3, x1], dim=1))
                    out = self.predictor(u3)              # (N, 2, L)
                    return out.transpose(1, 2)            # (N, L, 2)

            ckpt_ip = torch.load(TRACKNET_INPAINT_WEIGHTS_PATH, map_location="cpu", weights_only=False)
            sd_ip = ckpt_ip.get("model", ckpt_ip.get("state_dict", ckpt_ip)) if isinstance(ckpt_ip, dict) else ckpt_ip
            inpaint = InpaintNet().eval()
            inpaint.load_state_dict(sd_ip, strict=True)
            inpaint = inpaint.to(device)
            print(f"  ✓ InpaintNet loaded from {TRACKNET_INPAINT_WEIGHTS_PATH}")
            inpaint_used = True

            # Build ordered sequence of (x, y, vis). Coords go in TrackNet
            # processing resolution (288x512) — InpaintNet was trained on those.
            ordered_idxs = sorted(int(k) for k in frames_result.keys())
            L = TRACKNET_INPAINT_SEQ_LEN
            seq = np.zeros((len(ordered_idxs), 3), dtype=np.float32)
            for i, idx in enumerate(ordered_idxs):
                e = frames_result[str(idx)]
                if e["detected"]:
                    seq[i, 0] = e["pixel_x"] / scale_x_out  # → 512 wide
                    seq[i, 1] = e["pixel_y"] / scale_y_out  # → 288 tall
                    seq[i, 2] = 1.0

            # Process in non-overlapping chunks of L. Pad final chunk with zeros.
            corrected = np.zeros((len(ordered_idxs), 2), dtype=np.float32)
            with torch.no_grad():
                for s in range(0, len(ordered_idxs), L):
                    chunk = seq[s : s + L]
                    if len(chunk) < 2:
                        continue
                    if not chunk[:, 2].any():
                        continue   # all-zero visibility → don't hallucinate
                    if len(chunk) < L:
                        pad = np.zeros((L - len(chunk), 3), dtype=np.float32)
                        chunk_in = np.concatenate([chunk, pad], axis=0)
                    else:
                        chunk_in = chunk
                    x = torch.from_numpy(chunk_in).unsqueeze(0).to(device)
                    y = inpaint(x).cpu().numpy()[0]   # (L, 2)
                    corrected[s : s + len(chunk)] = y[: len(chunk)]

            # For undetected frames within a short gap, swap in inpaint output.
            # MAX_GAP is measured as min(distance-to-prev-detected, distance-to-next-detected).
            n = len(ordered_idxs)
            visible = seq[:, 2] > 0.5
            for i in range(n):
                if visible[i]:
                    continue
                # Distance to nearest detected frame on each side.
                back = next((d for d in range(1, TRACKNET_INPAINT_MAX_GAP + 1)
                             if i - d >= 0 and visible[i - d]), None)
                fwd = next((d for d in range(1, TRACKNET_INPAINT_MAX_GAP + 1)
                            if i + d < n and visible[i + d]), None)
                if back is None or fwd is None:
                    inpaint_skipped_long_gap += 1
                    continue
                cx_proc, cy_proc = corrected[i]
                if cx_proc <= 0 and cy_proc <= 0:
                    continue
                if _is_blacklisted(cx_proc * scale_x_out, cy_proc * scale_y_out):
                    continue
                idx = ordered_idxs[i]
                frames_result[str(idx)] = {
                    **frames_result[str(idx)],
                    "detected": True,
                    "pixel_x": round(float(cx_proc * scale_x_out), 2),
                    "pixel_y": round(float(cy_proc * scale_y_out), 2),
                    "confidence": 0.5,
                    "interpolated": True,
                }
                detected += 1
                inpaint_filled += 1
            print(f"  ✓ InpaintNet filled {inpaint_filled} gap frames "
                  f"(skipped {inpaint_skipped_long_gap} long gaps > {TRACKNET_INPAINT_MAX_GAP}f)")
        except Exception as e:
            print(f"  ⚠ InpaintNet failed ({e}); using raw TrackNet detections")
    else:
        print(f"  ⚠ InpaintNet weights not found at {TRACKNET_INPAINT_WEIGHTS_PATH}; skipping rectification")

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
        "model": "TrackNetV3 + InpaintNet" if inpaint_used else "TrackNetV3",
        "confidence_threshold": TRACKNET_HEATMAP_THRESHOLD,
        "inpaint": {
            "applied": inpaint_used,
            "gaps_filled": inpaint_filled,
            "long_gaps_skipped": inpaint_skipped_long_gap,
            "max_gap_frames": TRACKNET_INPAINT_MAX_GAP,
        },
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
    image=transcode_image,
    gpu="A10G",
    secrets=[gcs_secret],
    timeout=3600,
    scaledown_window=60,
    max_containers=6,
    retries=1,
)
def normalize_playback_source(gcs_path: str):
    """Background: make the uploaded video universally browser-playable (H.264
    8-bit + faststart) and OVERWRITE the GCS object the web player loads.

    Runs OFF the user's critical path — `prepare_job` commits the original source
    immediately and the detector decodes it directly, so nothing waits on this.
    GPU NVENC keeps a long clip fast (falls back to libx264 if NVENC is missing);
    frame timing is preserved so the playback file stays frame-aligned with the
    original the detector ran on. No-op when the source is already web-safe.
    """
    import tempfile, os

    tmpdir = tempfile.mkdtemp()
    local_in = os.path.join(tmpdir, "source.mp4")
    out_path = os.path.join(tmpdir, "websafe.mp4")
    try:
        download_from_gcs(gcs_path, local_in)
        web_safe, codec, pixfmt = _probe_web_safe(local_in)
        if web_safe:
            print(f"  ✓ {gcs_path} already web-safe (h264/{pixfmt}/faststart); skip normalize")
            return {"status": "skipped"}

        _transcode_web_safe(local_in, out_path, codec, pixfmt, use_gpu=True)

        bucket = _get_gcs_bucket()
        blob = bucket.blob(gcs_path)
        blob.cache_control = "public, max-age=86400"
        blob.upload_from_filename(out_path, content_type="video/mp4")
        print(f"  ✓ overwrote {gcs_path} with web-safe H.264 source")
        return {"status": "ok"}
    except Exception as e:
        print(f"  ⚠ playback normalization failed ({e}); web player keeps original source")
        return {"status": "error", "error": str(e)}
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _extract_frame_timestamps(local_path: str, fps: float, total_frames: int) -> dict:
    """Build the {frame_index: timestamp_sec} map used by ball + player tracking.

    Fast path: sample the first ~120 frames' PTS. If they're evenly spaced the
    video is constant-frame-rate and every timestamp is exactly i/fps — instant
    even for a multi-hour clip, so we skip the full per-frame ffprobe dump that
    used to take minutes on long uploads. Only genuine variable-frame-rate (VFR)
    sources fall back to the full PTS extraction.
    """
    import subprocess

    sample_pts = []
    try:
        sample_cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-read_intervals", "%+#120",
            "-show_entries", "frame=pts_time", "-of", "csv=p=0", local_path,
        ]
        out = subprocess.run(sample_cmd, capture_output=True, text=True, timeout=60).stdout
        sample_pts = [float(x) for x in out.strip().split("\n") if x.strip()]
    except Exception as e:
        print(f"  ⚠ VFR probe failed ({e}); assuming CFR")

    is_cfr = True
    if len(sample_pts) > 10:
        intervals = [sample_pts[i + 1] - sample_pts[i] for i in range(len(sample_pts) - 1)]
        avg = sum(intervals) / len(intervals)
        if avg <= 0 or max(abs(iv - avg) for iv in intervals) > 0.005:
            is_cfr = False

    if is_cfr:
        print(f"  ✓ CFR video: timestamps = i/fps ({fps:.2f} fps), skipping full PTS dump")
        return {str(i): i / fps for i in range(total_frames)}

    # VFR: need every frame's real PTS. Rare for phone video; pay the full cost.
    print(f"  ⚠ VFR detected; extracting full per-frame PTS map")
    frame_timestamps = {}
    try:
        pts_cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "frame=pts_time", "-of", "csv=p=0", local_path,
        ]
        pts_lines = subprocess.run(pts_cmd, capture_output=True, text=True, timeout=300).stdout.strip().split("\n")
        for i, line in enumerate(pts_lines):
            line = line.strip()
            try:
                frame_timestamps[str(i)] = float(line) if line else i / fps
            except ValueError:
                frame_timestamps[str(i)] = i / fps
        print(f"  ✓ Extracted {len(frame_timestamps)} frame timestamps")
    except Exception as e:
        print(f"  ⚠ ffprobe PTS dump failed ({e}); using calculated timestamps")
        frame_timestamps = {str(i): i / fps for i in range(total_frames)}
    return frame_timestamps


@app.function(
    image=web_image,
    volumes={DATA_DIR: scratch_vol},
    secrets=[gcs_secret],
    timeout=600,
    cpu=8,
    memory=8192,
)
def prepare_job(job_id: str, gcs_path: str):
    import cv2

    local_path = f"{DATA_DIR}/jobs/{job_id}/source.mp4"
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    download_from_gcs(gcs_path, local_path)

    # Commit the ORIGINAL source immediately so the detector (which decodes HEVC
    # directly) can start the moment its spawn lands. The browser-safe H.264
    # transcode is decoupled to a background GPU function and does NOT gate
    # processing — this is what keeps "getting ready" fast regardless of length.
    scratch_vol.commit()

    # Kick off the browser-playback transcode in the background (HEVC→H.264 on
    # GPU/NVENC). Fire-and-forget: overwrites the GCS object the web player loads
    # while detection runs. No-op inside if the source is already web-safe.
    try:
        normalize_playback_source.spawn(gcs_path)
    except Exception as e:
        print(f"  ⚠ could not spawn playback normalizer: {e}")

    # Video metadata (instant from the container header).
    cap = cv2.VideoCapture(local_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Sidecar with the per-video metadata so /api/prepare can answer the client
    # without re-opening the video or booting a GPU container.
    meta_sidecar = {
        "src_width": width,
        "src_height": height,
        "src_fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 2) if fps > 0 else 0.0,
    }
    meta_path = f"{DATA_DIR}/jobs/{job_id}/video_meta.json"
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta_sidecar, f)

    # Frame timestamps — fast for the common CFR case (see helper).
    frame_timestamps = _extract_frame_timestamps(local_path, fps, total_frames)

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


def _read_prepare_metadata(job_id: str) -> dict:
    """Read the sidecar written by `prepare_job` and return it as the /api/prepare
    response payload. Used when the client extracted the first frame locally and
    doesn't need the GPU first-frame round-trip.
    """
    scratch_vol.reload()
    meta_path = f"{DATA_DIR}/jobs/{job_id}/video_meta.json"
    if not os.path.exists(meta_path):
        return {"error": f"video_meta.json missing for job {job_id}"}
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta


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

def detect_bounces_and_hits(
    ball_data: dict,
    homography_data: dict,
    player_locations: dict = None,
    player_masks: dict = None,
) -> dict:
    """
    Detect ball bounces and player hits from ball tracking data.

    Detection engine: the ``squashev`` package (piecewise-parabola
    trajectory segmentation + a player-mask-aware 9-rule classifier).
    See ``squashev/events.py`` and ``squashev/segments.py``.

      1. ``squashev`` fits piecewise parametric segments (x linear,
         y quadratic) to the ball series; each boundary between two
         segments is a candidate event.
      2. Three velocity "gates" (reversal, speed change, gap
         divergence) fire on each boundary; gravity-consistent changes
         across gaps are suppressed.
      3. An ordered decision tree classifies each surviving boundary,
         using **SAM2 player-mask proximity** as the top-priority HIT
         signal, homography region confirmation for BOUNCE surface, and
         squash rally grammar (a hit must reach the front wall).
      4. Post-process: dedup, drop physically-impossible same-wall
         repeats, insert front-wall bounces hidden in gaps.

    This function is a thin adapter: it hands ``ball_data`` /
    ``homography_data`` / ``player_masks`` to ``squashev.pipeline.run``
    as temp JSON files (the interface it expects), then maps the
    returned predictions back into the boastiq event schema used by the
    rest of the pipeline. Surfaces are reported as FW / FL / LW / RW.
    Player attribution comes from the masks (and is refined downstream
    by ``attribute_shots``).

    Args:
        ball_data: Ball tracking JSON with ``frames`` dict and
            ``source_resolution`` {width, height}.
        homography_data: Homography JSON with per-surface matrices.
        player_locations: kept for signature compatibility; attribution
            now comes from ``player_masks``.
        player_masks: player_masks.json dict (per-frame SAM2 polygons).
            Without it, ``squashev`` cannot detect racket hits, so pass
            it whenever available.

    Returns:
        Dict with ``events`` list, ``overlay`` array, and metadata.
    """
    import numpy as np

    # ═══════════════════════════════════════════════════════════════════
    # Resolution & zone derivation
    # ═══════════════════════════════════════════════════════════════════
    src_res = ball_data.get("source_resolution") or {}
    W = int(src_res.get("width") or 1280)
    H = int(src_res.get("height") or 720)
    # Detector tunables now live in squashev/config.py; the old inline
    # 720p-baseline constants were removed with the legacy engine.

    # ═══════════════════════════════════════════════════════════════════
    # Homography matrices (for world-coord projection of bounces)
    # ═══════════════════════════════════════════════════════════════════
    homographies = homography_data.get("homographies", {}) if homography_data else {}
    H_FW = np.array(homographies.get("front_wall", {}).get("homography_matrix", np.eye(3)))
    H_FL = np.array(homographies.get("floor", {}).get("homography_matrix", np.eye(3)))
    H_LW = np.array(homographies.get("left_wall", {}).get("homography_matrix", np.eye(3)))
    H_RW = np.array(homographies.get("right_wall", {}).get("homography_matrix", np.eye(3)))

    def project(Hmat, px, py):
        r = Hmat @ np.array([px, py, 1.0])
        if abs(r[2]) < 1e-10:
            return None, None
        return float(r[0] / r[2]), float(r[1] / r[2])

    SURFACE_MAP = {
        "front_wall": "FW",
        "floor":      "FL",
        "left_wall":  "LW",
        "right_wall": "RW",
    }

    # Player attribution is handled inside ``squashev`` via mask
    # proximity (``player_masks``); the old foot-point lookup that used
    # ``player_locations`` has been removed. ``player_locations`` is kept
    # only for signature compatibility with existing call sites.

    # ═══════════════════════════════════════════════════════════════════
    # Load ball series into per-frame arrays
    # ═══════════════════════════════════════════════════════════════════
    frames_data = ball_data.get("frames", {})
    if not frames_data:
        return {"events": [], "overlay": [], "total_events": 0,
                "total_bounces": 0, "total_hits": 0,
                "bounce_surfaces": {"FW": 0, "FL": 0, "LW": 0, "RW": 0},
                "error": "No ball tracking frames"}

    total_frames = int(ball_data.get("total_frames") or 0)
    if total_frames <= 0:
        try:
            total_frames = max(int(k) for k in frames_data.keys()) + 1
        except ValueError:
            total_frames = 0
    if total_frames < 10:
        return {"events": [], "overlay": [], "total_events": 0,
                "total_bounces": 0, "total_hits": 0,
                "bounce_surfaces": {"FW": 0, "FL": 0, "LW": 0, "RW": 0},
                "error": "Too few frames"}

    bx = np.full(total_frames, np.nan, dtype=np.float64)
    by = np.full(total_frames, np.nan, dtype=np.float64)
    bt = np.full(total_frames, np.nan, dtype=np.float64)
    bc = np.zeros(total_frames, dtype=np.float64)
    for k, v in frames_data.items():
        if not v.get("detected"):
            continue
        f = int(k)
        if not (0 <= f < total_frames):
            continue
        px = v.get("pixel_x", v.get("x"))
        py = v.get("pixel_y", v.get("y"))
        if px is None or py is None:
            continue
        # Drop out-of-frame inpaint extrapolations.
        if not (0 <= px < W and 0 <= py < H):
            continue
        bx[f] = float(px)
        by[f] = float(py)
        bc[f] = float(v.get("confidence", 0.5))
        bt[f] = float(v.get("timestamp_sec", f / 30.0))

    n_det = int(np.sum(~np.isnan(bx)))
    if n_det < 10:
        return {"events": [], "overlay": [], "total_events": 0,
                "total_bounces": 0, "total_hits": 0,
                "bounce_surfaces": {"FW": 0, "FL": 0, "LW": 0, "RW": 0},
                "error": "Too few ball detections"}

    # Fill timestamps for frames without detections (linear by frame index
    # against the median fps inferred from detected timestamps).
    valid_ts = ~np.isnan(bt)
    if valid_ts.sum() >= 2:
        idx = np.where(valid_ts)[0]
        ts = bt[idx]
        # Per-frame interval (robust to noise)
        df = np.diff(idx)
        dt = np.diff(ts)
        good = df > 0
        per_frame = float(np.median(dt[good] / df[good])) if good.any() else 1.0 / 30.0
        t0 = float(ts[0]) - float(idx[0]) * per_frame
        all_t = t0 + np.arange(total_frames, dtype=np.float64) * per_frame
        # Keep the measured timestamps where present.
        bt = np.where(valid_ts, bt, all_t)
        fps_inv = per_frame
    else:
        fps_inv = 1.0 / 30.0
        bt = np.arange(total_frames, dtype=np.float64) * fps_inv

    # ═══════════════════════════════════════════════════════════════════
    # Event detection & classification via the squashev package.
    # squashev reads ball / homography / mask JSON from files, so the
    # in-memory dicts are written to temp files for the handoff. Its
    # velocity thresholds are tuned for ~REF_FPS; scale the fps-dependent
    # ones so behaviour is preserved at other frame rates.
    # ═══════════════════════════════════════════════════════════════════
    import json as _json
    import tempfile as _tempfile
    import os as _os
    from squashev.config import Config, DataPaths, config_from_overrides
    from squashev.pipeline import run as _squashev_run

    REF_FPS = 59.89
    src_fps = float(ball_data.get("source_fps") or 0.0)
    if src_fps <= 1e-6:
        src_fps = (1.0 / fps_inv) if fps_inv else 30.0

    overrides = {"fps": src_fps}
    if abs(src_fps - REF_FPS) / REF_FPS > 0.15:
        vscale = REF_FPS / src_fps          # px/frame velocities scale with 1/fps
        fscale = src_fps / REF_FPS          # frame-window counts scale with fps
        _base = Config()
        _VEL_FIELDS = ("vx_flip_delta", "vy_flip_delta", "vel_jump_min",
                       "fw_min_pre_speed", "hit_speed_gain_min",
                       "side_wall_vy_preserve_px", "gravity_vx_tol",
                       "gravity_vy_tol", "gap_tol_px_per_frame",
                       "seed_flip_delta_px")
        _FRAME_FIELDS = ("max_intra_gap", "gap_confirm_min",
                         "max_event_gap_frames", "mask_search_radius",
                         "min_event_separation", "gravity_suppress_min_gap",
                         "intersect_pad_frames", "grammar_fw_min_gap",
                         "grammar_fw_dedup_frames")
        for _f in _VEL_FIELDS:
            overrides[_f] = float(getattr(_base, _f)) * vscale
        for _f in _FRAME_FIELDS:
            overrides[_f] = max(1, int(round(getattr(_base, _f) * fscale)))
    cfg = config_from_overrides(Config(), overrides=overrides)

    _tmp_paths = []

    def _dump(obj):
        tf = _tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        _json.dump(obj, tf)
        tf.close()
        _tmp_paths.append(tf.name)
        return tf.name

    # ═══════════════════════════════════════════════════════════════════
    # PRIMARY ENGINE: learned event spotter (spotter_detect.py) — a
    # gradient-boosted temporal classifier trained on hand-labeled rallies.
    # Emits the same `preds` shape the formatter below consumes. The legacy
    # squashev rule engine remains as an automatic fallback if the model
    # file or scikit-learn is unavailable.
    # ═══════════════════════════════════════════════════════════════════
    preds = None
    _model_path = _os.environ.get("SPOTTER_MODEL") or (
        "/app/spotter_model.pkl" if _os.path.exists("/app/spotter_model.pkl")
        else str(Path(__file__).parent / "spotter_model.pkl"))
    try:
        from spotter_detect import spot_events
        preds = spot_events(ball_data, homography_data, player_masks,
                            _model_path)
        print(f"    spotter: {len(preds)} events (model={_model_path})")
    except Exception as _spot_err:  # noqa: BLE001 — any failure -> legacy engine
        print(f"    ⚠ spotter unavailable ({type(_spot_err).__name__}: "
              f"{_spot_err}); falling back to squashev rules")

    if preds is None:
        try:
            _ball_path = _dump(ball_data)
            _homo_path = _dump(homography_data or {})
            _masks_path = _dump(player_masks if player_masks else {"frames": []})
            paths = DataPaths(ball_json=_ball_path, homography_json=_homo_path,
                              masks_json=_masks_path, width=W, height=H)
            predictions, _diag = _squashev_run(paths, cfg)
        finally:
            for _p in _tmp_paths:
                try:
                    _os.unlink(_p)
                except OSError:
                    pass

        if not player_masks:
            print("    ⚠ No player masks supplied — squashev cannot detect "
                  "racket hits; only bounces will be produced.")

        # Map squashev predictions into the `preds` shape the formatter below
        # consumes. squashev bounce.surface is already a full name
        # (front_wall/floor/left_wall/right_wall); hits carry player +
        # hit_by (→ hitter, used to attribute following bounces).
        preds = []
        for _p in predictions:
            q = {
                "frame": int(_p["frame"]),
                "type": _p["type"],
                "pixel_x": float(_p["pixel_x"]),
                "pixel_y": float(_p["pixel_y"]),
                "method": ("rally_grammar" if _p.get("grammar")
                           else "fallback" if _p.get("fallback") else "squashev"),
            }
            if _p["type"] == "bounce":
                q["surface"] = _p.get("surface", "")
            else:
                q["player"] = _p.get("player")
                _hit_by = _p.get("hit_by")
                if _hit_by not in ("player_1", "player_2"):
                    _hit_by = _p.get("player") if _p.get("player") in ("player_1", "player_2") else None
                q["hitter"] = _hit_by
            preds.append(q)
    preds.sort(key=lambda e: e["frame"])


    # ═══════════════════════════════════════════════════════════════════
    # Format output (preserve boastiq event schema)
    # ═══════════════════════════════════════════════════════════════════
    def ts_for_frame(f):
        if 0 <= f < total_frames and not np.isnan(bt[f]):
            return float(bt[f])
        return float(f) * fps_inv

    def calc_speeds(frame):
        """speed_in / speed_out estimated from nearest valid neighbours."""
        if not (0 <= frame < total_frames):
            return 0.0, 0.0
        fp = None
        for d in range(1, 8):
            ff = frame - d
            if 0 <= ff < total_frames and not np.isnan(bx[ff]):
                fp = ff; break
        fn = None
        for d in range(1, 8):
            ff = frame + d
            if 0 <= ff < total_frames and not np.isnan(bx[ff]):
                fn = ff; break
        # If the event frame itself has a detection use it; otherwise use
        # the bracket frames directly.
        if not np.isnan(bx[frame]):
            ex, ey = float(bx[frame]), float(by[frame])
        else:
            ex = ey = None
        speed_in = 0.0
        if fp is not None:
            df = frame - fp
            if df > 0 and df <= 5:
                if ex is not None:
                    speed_in = float(np.hypot(ex - bx[fp], ey - by[fp])) / df
                elif fn is not None:
                    # interpolate position at frame
                    span = fn - fp
                    if span > 0:
                        ix = bx[fp] + (bx[fn] - bx[fp]) * (frame - fp) / span
                        iy = by[fp] + (by[fn] - by[fp]) * (frame - fp) / span
                        speed_in = float(np.hypot(ix - bx[fp], iy - by[fp])) / df
        speed_out = 0.0
        if fn is not None:
            df = fn - frame
            if df > 0 and df <= 5:
                if ex is not None:
                    speed_out = float(np.hypot(bx[fn] - ex, by[fn] - ey)) / df
                elif fp is not None:
                    span = fn - fp
                    if span > 0:
                        ix = bx[fp] + (bx[fn] - bx[fp]) * (frame - fp) / span
                        iy = by[fp] + (by[fn] - by[fp]) * (frame - fp) / span
                        speed_out = float(np.hypot(bx[fn] - ix, by[fn] - iy)) / df
        return round(float(speed_in), 2), round(float(speed_out), 2)

    SURFACE_H = {"FW": H_FW, "FL": H_FL, "LW": H_LW, "RW": H_RW}

    events_out = []
    overlay = []
    bounce_surface_counts = {"FW": 0, "FL": 0, "LW": 0, "RW": 0}
    n_bounces = 0
    n_hits = 0
    last_hitter_player = None  # the player who last struck the ball

    for p in preds:
        frame = int(p["frame"])
        ts = round(ts_for_frame(frame), 3)
        px = round(float(p["pixel_x"]), 1)
        py = round(float(p["pixel_y"]), 1)
        speed_in, speed_out = calc_speeds(frame)
        method = p.get("method", "")
        if p["type"] == "bounce":
            surf_code = SURFACE_MAP.get(p.get("surface", ""), "NONE")
            speed_ratio = round(speed_out / max(speed_in, 0.1), 2) if speed_in > 0.1 else 0.5
            speed_ratio = min(speed_ratio, 1.0)
            wx = wy = None
            Hmat = SURFACE_H.get(surf_code)
            if Hmat is not None:
                wx, wy = project(Hmat, px, py)
            bounce_event = {
                "type": "bounce",
                "frame": frame,
                "timestamp_sec": ts,
                "pixel_x": px, "pixel_y": py,
                "surface": surf_code,
                "method": method,
                "speed_in": speed_in,
                "speed_out": speed_out,
                "speed_ratio": speed_ratio,
                "world_x": round(wx, 3) if wx is not None else None,
                "world_y": round(wy, 3) if wy is not None else None,
            }
            # Any bounce that follows a hit belongs to the hitter.
            if last_hitter_player is not None:
                bounce_event["player"] = last_hitter_player
            events_out.append(bounce_event)
            overlay.append([ts, px, py, speed_ratio, speed_in, speed_out, frame])
            if surf_code in bounce_surface_counts:
                bounce_surface_counts[surf_code] += 1
            n_bounces += 1
        else:  # hit
            speed_A = float(p.get("speed_A", speed_in))
            speed_B = float(p.get("speed_B", speed_out))
            if speed_A > 0.1:
                ratio_raw = speed_B / speed_A
            else:
                ratio_raw = 2.0
            speed_ratio = max(round(ratio_raw, 2), 1.01)
            # `hitter` is the resolved player_1/player_2 even when the display
            # `player` is "serve"; fall back to `player` for older preds.
            hitter = p.get("hitter") or p.get("player")
            hitter = hitter if hitter in ("player_1", "player_2") else None
            events_out.append({
                "type": "hit",
                "frame": frame,
                "timestamp_sec": ts,
                "pixel_x": px, "pixel_y": py,
                "method": method,
                "player": p.get("player"),
                "hitter": hitter,
                "speed_in": speed_in,
                "speed_out": speed_out,
                "speed_ratio": speed_ratio,
            })
            overlay.append([ts, px, py, speed_ratio, speed_in, speed_out, frame])
            n_hits += 1
            # Advance the running hitter so the next FW bounce can inherit it.
            if hitter is not None:
                last_hitter_player = hitter

    events_out.sort(key=lambda e: e["timestamp_sec"])
    overlay.sort(key=lambda e: e[0])

    print(f"    Bounces detected: {n_bounces}")
    print(f"    Hits detected: {n_hits}")

    return {
        "total_events": len(events_out),
        "total_bounces": n_bounces,
        "total_hits": n_hits,
        "bounce_surfaces": bounce_surface_counts,
        "events": events_out,
        "overlay": overlay,
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


def shots_from_bounce_hits(bounce_hit_events: dict, player_config: dict) -> dict:
    """Build the shot list straight from the detector's hit events, using
    the detector's own (mask-based) attribution.

    This replaces the former ``attribute_shots`` pass. Each detected hit
    becomes a shot owned by the player the detector says struck it
    (``hitter``, falling back to ``player``); ``build_point_rallies`` then
    attributes each bounce to the most recent such hit — i.e. bounces are
    owned by the last player that hit the ball.

    Forehand/backhand and drive/drop/boast classification that used to come
    from ``attribute_shots`` are intentionally dropped for now (``shot_type``
    is left ``"unknown"``); shot attribution will be revisited separately.

    Returns a dict shaped like the old ``shot_attribution`` (``{"shots": [...]}``)
    so ``build_point_rallies`` consumes it unchanged.
    """
    events = (bounce_hit_events or {}).get("events", []) if bounce_hit_events else []
    pc = player_config or {}
    names = {
        1: (pc.get("player_1") or {}).get("name", "Player 1"),
        2: (pc.get("player_2") or {}).get("name", "Player 2"),
    }
    shots = []
    for e in events:
        if e.get("type") != "hit":
            continue
        # Accept the detector's string ids ("player_1"), manual integer
        # labels (1/2), or a serve/unknown that alternates from the prior shot.
        who = e.get("hitter") or e.get("player")
        if who in ("player_1", 1, "1"):
            pid = 1
        elif who in ("player_2", 2, "2"):
            pid = 2
        else:
            pid = 2 if (shots and shots[-1]["player"] == 1) else 1
        shots.append({
            "shot_number": len(shots) + 1,
            "frame": e.get("frame"),
            "timestamp_sec": e.get("timestamp_sec", 0),
            "player": pid,
            "player_name": names[pid],
            "confidence": "detector",
        })
    return {"shots": shots}


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

    # Pre-sort frames by timestamp for fast per-point movement slicing.
    # Each entry: (timestamp_sec, p1_x, p1_y, p2_x, p2_y) with None if missing.
    frames_by_time = []
    for entry in frames_data:
        if not entry:
            continue
        ts = entry.get("timestamp_sec")
        if ts is None:
            continue
        p1 = entry.get("player_1") or {}
        p2 = entry.get("player_2") or {}
        frames_by_time.append((
            float(ts),
            p1.get("court_x"), p1.get("court_y"),
            p2.get("court_x"), p2.get("court_y"),
        ))
    frames_by_time.sort(key=lambda r: r[0])
    
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

        # Point end: prefer the calm-period start from the v2 detector
        # (point_end_timestamp_sec). That's the *real* end of the rally —
        # not "the moment the next point begins". Stats per point only
        # count shots/movement inside [point_start, point_end_calm).
        # Fallbacks: next-point-start, then video end. Also clamp so a
        # bogus calm-end can't run past the next point's start.
        calm_end = point.get("point_end_timestamp_sec")
        if i + 1 < len(points_sorted):
            next_point_start = points_sorted[i + 1].get("timestamp_sec", total_duration_sec)
        else:
            next_point_start = total_duration_sec

        if calm_end is not None and calm_end > point_start:
            point_end = min(float(calm_end), next_point_start)
        else:
            point_end = next_point_start

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
        
        # Build downsampled movement track for heatmaps.
        # Walk frames in [point_start, point_end), keep one sample per
        # SAMPLE_INTERVAL_SEC. Court-meter coords; null where missing.
        SAMPLE_INTERVAL_SEC = 0.2  # 5 Hz
        p1_x_track, p1_y_track = [], []
        p2_x_track, p2_y_track = [], []
        track_t0 = None
        next_sample_t = point_start
        for ts, p1x, p1y, p2x, p2y in frames_by_time:
            if ts < point_start:
                continue
            if ts >= point_end:
                break
            if ts < next_sample_t:
                continue
            if track_t0 is None:
                track_t0 = ts
            p1_x_track.append(round(p1x, 2) if p1x is not None else None)
            p1_y_track.append(round(p1y, 2) if p1y is not None else None)
            p2_x_track.append(round(p2x, 2) if p2x is not None else None)
            p2_y_track.append(round(p2y, 2) if p2y is not None else None)
            next_sample_t = ts + SAMPLE_INTERVAL_SEC

        movement = {
            "fps": round(1.0 / SAMPLE_INTERVAL_SEC, 2),
            "start_sec": round(track_t0, 2) if track_t0 is not None else None,
            "samples": len(p1_x_track),
            "player_1": {"x": p1_x_track, "y": p1_y_track},
            "player_2": {"x": p2_x_track, "y": p2_y_track},
        }

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
                    "x": round(bounce.get("world_x"), 2) if bounce.get("world_x") is not None else None,
                    "y": round(bounce.get("world_y"), 2) if bounce.get("world_y") is not None else None,
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
            
            # Build shot entry. (Shot classification — drive/drop/boast/volley
            # and forehand/backhand — has been removed; to be revisited.)
            rally_shot = {
                "shot_number": shot_idx + 1,
                "hit_by": f"player_{shot.get('player', 1)}",
                "player_name": shot.get("player_name", f"Player {shot.get('player', 1)}"),
                "hit_time_sec": round(hit_time, 2),
                "frame": hit_frame,
                "confidence": shot.get("confidence", "unknown"),
                
                "player_1_position": {
                    "x": round(p1_pos.get("world_x"), 2) if p1_pos.get("world_x") is not None else None,
                    "y": round(p1_pos.get("world_y"), 2) if p1_pos.get("world_y") is not None else None,
                },
                "player_2_position": {
                    "x": round(p2_pos.get("world_x"), 2) if p2_pos.get("world_x") is not None else None,
                    "y": round(p2_pos.get("world_y"), 2) if p2_pos.get("world_y") is not None else None,
                },
                
                "front_wall_contact": front_wall_contact,
                "side_wall_contact": side_wall_contact,
                "floor_bounce": floor_bounce,
            }
            
            rally.append(rally_shot)
        
        # Build point summary. end_source records whether the rally
        # cutoff came from the v2 calm-period detector or from the
        # next-point fallback — useful for debugging stat drift.
        end_source = "calm" if (point.get("point_end_timestamp_sec") is not None
                                and float(point.get("point_end_timestamp_sec")) > point_start
                                and float(point.get("point_end_timestamp_sec")) <= next_point_start
                                ) else "next_point_or_video_end"
        point_rally = {
            "point_number": point_num,
            "start_time_sec": round(point_start, 2),
            "end_time_sec": round(point_end, 2),
            "end_source": end_source,
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
            "movement": movement,
        }

        point_rallies.append(point_rally)

    return point_rallies


def compute_match_summary(point_rallies: list, player_config: dict) -> dict:
    """
    Aggregate per-point rallies into a single match-level summary the
    client can render without re-downloading each point_NNN.json.

    Mirrors the JS aggregation that used to run in Index.html
    loadMatchAggregates(): total duration, total shots, avg rally,
    per-player distance from the dense movement track, and per-player
    shot-class counts (drive/drop/boast/volley).
    """
    import math

    def _fmt_time(secs: float) -> str:
        if secs is None or not math.isfinite(secs):
            return "0:00"
        s = int(round(secs))
        return f"{s // 60}:{s % 60:02d}"

    def _movement_distance(rd: dict, player: str):
        m = (rd or {}).get("movement") or {}
        p = m.get(player) or {}
        xs = p.get("x") or []
        ys = p.get("y") or []
        if len(xs) < 2:
            return None
        total = 0.0
        last_x = last_y = None
        n = min(len(xs), len(ys))
        for i in range(n):
            x, y = xs[i], ys[i]
            if x is None or y is None:
                last_x = last_y = None
                continue
            if last_x is not None:
                dx = x - last_x
                dy = y - last_y
                total += (dx * dx + dy * dy) ** 0.5
            last_x, last_y = x, y
        return total

    p1_name = (player_config.get("player_1") or {}).get("name") or "Player 1"
    p2_name = (player_config.get("player_2") or {}).get("name") or "Player 2"

    durations = [float(pr.get("duration_sec") or 0) for pr in point_rallies]
    total_dur = sum(durations)
    avg_dur = (total_dur / len(durations)) if durations else 0.0

    p1_shots = p2_shots = 0
    p1_dist = p2_dist = 0.0

    for rd in point_rallies:
        for shot in rd.get("rally") or []:
            if shot.get("hit_by") == "player_1":
                p1_shots += 1
            else:
                p2_shots += 1
        d1 = _movement_distance(rd, "player_1")
        d2 = _movement_distance(rd, "player_2")
        if d1 is not None:
            p1_dist += d1
        if d2 is not None:
            p2_dist += d2

    return {
        "duration": _fmt_time(total_dur),
        "totalShots": p1_shots + p2_shots,
        "avgRally": f"{avg_dur:.1f} s",
        "distP1": f"{p1_dist:.0f} m",
        "distP2": f"{p2_dist:.0f} m",
        # Shot classification removed; keys kept at 0 to preserve the schema
        # the client reads (drive/drop/boast/volley cards).
        "drives":  {"p1": 0, "p2": 0},
        "drops":   {"p1": 0, "p2": 0},
        "boasts":  {"p1": 0, "p2": 0},
        "volleys": {"p1": 0, "p2": 0},
        "player_1_name": p1_name,
        "player_2_name": p2_name,
    }


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
    import bisect
    import math

    # Court geometry (meters from front-left floor corner)
    COURT_WIDTH_M = 6.4
    COURT_DEPTH_M = 9.75
    HALF_COURT_X = COURT_WIDTH_M / 2
    SHORT_LINE_Y_M = 5.44
    SERVICE_BOX_BACK_Y_M = 7.04
    LEFT_SERVICE_BOX_X_RANGE = (0.0, 1.6)
    RIGHT_SERVICE_BOX_X_RANGE = (4.8, 6.4)

    # Stage 1 — front-wall flight
    MIN_FLIGHT_FRAMES = 5
    MIN_FLIGHT_Y_DROP_PX = 45
    MIN_FLIGHT_MEAN_SPEED = 9.0
    FLIGHT_MAX_INNER_GAP = 6
    FLIGHT_EPS_PX = 4.0

    # Stage 2 — pre-hit calm
    PRE_HIT_LOOKBACK_FRAMES = 90
    PRE_HIT_CALM_MAX_MEAN_SPEED = 8.0
    PRE_HIT_CALM_MAX_X_SPREAD = 250.0
    PRE_HIT_GAP_FRAMES_OK = 18

    # Stage 3 — player verification
    SERVICE_BOX_MARGIN_M = 0.50
    SHORT_LINE_MARGIN_M = 0.15
    STATIONARITY_WINDOW_FRAMES = 45
    MAX_STATIONARITY_STD_M = 1.10
    PRE_HIT_VERIFY_LOOKBACK = 60

    # Suppression / scoring
    SUPPRESS_WINDOW_FRAMES = 120
    W_FLIGHT_DROP = 1.0
    W_FLIGHT_SPEED = 0.5
    W_PRE_CALM = 0.5
    W_VERIFIED = 2.0

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
    # Step 2: Download homography (wait if needed — admin manually tags)
    # ================================================================
    # Homography is produced by the admin tagging UI (/admin/tag) at an
    # unpredictable time after upload. Poll generously: 360 × 30s = 3 hours.
    print("\n📥 Downloading homography (waits for admin tagging)...")
    H = None
    h_data = {}
    # Poll every 2s so the pipeline resumes ~immediately after the admin
    # submits landmarks. Total budget stays at 3 hours (5400 × 2s).
    POLL_ATTEMPTS = 5400
    POLL_INTERVAL_S = 2
    for attempt in range(POLL_ATTEMPTS):
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
        if attempt < POLL_ATTEMPTS - 1:
            # Log a status line every 60s (30 polls × 2s) so the log
            # doesn't get spammed at the new 2s cadence.
            if attempt % 30 == 0:
                mins_waited = (attempt * POLL_INTERVAL_S) // 60
                print(f"  ⏳ Waiting for homography ({mins_waited} min elapsed; "
                      f"admin should tag at /admin/tag)...")
            _time.sleep(POLL_INTERVAL_S)

    if H is None:
        print("  ✗ Homography not available after 3 hours")
        return {"status": "error", "message": "Homography not available — admin tagging timed out"}

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

    # Player masks (SAM2 polygons) — required by the squashev detector to
    # classify racket hits. Built by the earlier tracking stage; load best
    # effort (detection still runs bounces-only if unavailable).
    player_masks = None
    try:
        local_masks = f"/tmp/masks_{video_key}.json"
        download_from_gcs(f"{gcs_dir}/player_masks.json", local_masks)
        with open(local_masks, "r") as f:
            player_masks = json_mod.load(f)
        print(f"  ✓ Player masks loaded ({len(player_masks.get('frames', []))} frames)")
    except Exception as e:
        print(f"  ⚠ Player masks unavailable ({e}); hits will not be detected")

    if ball_data is None:
        print("  ⚠ Ball tracking not available — running detection without ball data")

    # ================================================================
    # Step 5: Run point detection on full match
    # Player-activity detector v2: calm → burst transitions in combined
    # player velocity. Ball-track is consulted only as a confidence signal.
    # See: "Squash point-start detector — portable spec (v2)".
    # ================================================================
    print("\n⚙️  Running point detection (player-activity v2)...")

    src_res = tracking_data.get("source_resolution") or {}
    src_w = int(src_res.get("width") or 1920)
    src_h = int(src_res.get("height") or 1080)

    # ---- v2 spec constants ----
    COURT_WIDTH_M_V2          = 6.4
    HALF_COURT_X_V2           = 3.2

    LOW_THRESH                = 1.3     # m/s; combined velocity below this = "calm"
    MIN_LOW_DURATION          = 0.5     # seconds; min calm-window length
    BURST_THRESH              = 2.0     # m/s; combined velocity above this = "burst"
    MIN_BETWEEN               = 3.5     # seconds; min spacing between emitted points
    LOOKAHEAD_SAMPLES         = 60      # samples after calm for the burst

    MAX_VELOCITY              = 8.0     # m/s; cap (rejects homography noise)
    FROZEN_THRESH             = 0.05    # m/s; below for FROZEN_WINDOW = "lost"
    FROZEN_WINDOW             = 30      # samples of zero motion = frozen tracking
    SMOOTH_WINDOW             = 5       # box-filter window for velocity smoothing

    BALL_DENSITY_WINDOW_SEC   = 2.0     # ± window around moment for ball density

    GAP_WEIGHT_PER_SEC        = 0.10
    GAP_CAP                   = 0.30
    SEP_BASE                  = 0.10
    SEP_PER_M_OVER_2          = 0.05
    SEP_CAP                   = 0.25
    BALL_REAPPEAR_BASE        = 0.15
    BALL_REAPPEAR_PER_DENSITY = 0.30
    BALL_CAP                  = 0.30
    BURST_PER_M_OVER_1_5      = 0.10
    BURST_CAP                 = 0.20
    MIN_EMIT_SCORE            = 0.45    # spec target ≥85% precision

    # ---- Ball detection map for ball_density_at (optional) ----
    ball_frames_dict = ball_data.get("frames", {}) if ball_data else {}
    ball_detected_set = set()
    have_ball = False
    if ball_frames_dict:
        have_ball = True
        for fn_str, bf in ball_frames_dict.items():
            if bf and bf.get("detected"):
                try:
                    ball_detected_set.add(int(fn_str))
                except (TypeError, ValueError):
                    continue

    def ball_density_at(t_sec, window_sec):
        """Fraction of frames in [t-window/2, t+window/2] where the ball
        was detected. Returns -1.0 if ball data isn't available."""
        if not have_ball:
            return -1.0
        center_frame = int(round(t_sec * src_fps))
        half = int(round(window_sec * src_fps / 2))
        lo = max(0, center_frame - half)
        hi = center_frame + half
        total = hi - lo + 1
        if total <= 0:
            return -1.0
        # Counting set-membership across a [lo, hi] range is O(window);
        # set lookups are O(1) so this is cheap even on long rallies.
        detected = sum(1 for f in range(lo, hi + 1) if f in ball_detected_set)
        return detected / total

    # ---- Step 1: Build per-sample series from world_frames ----
    # world_frames was produced by Step 3 (pixel→court via floor homography).
    # Skip any sample missing either player's court_x/court_y.
    frame_nums_arr = []
    ts_arr = []
    p1x = []; p1y = []
    p2x = []; p2y = []
    for entry in world_frames:
        if not entry:
            continue
        p1 = entry.get("player_1") or {}
        p2 = entry.get("player_2") or {}
        cx1 = p1.get("court_x"); cy1 = p1.get("court_y")
        cx2 = p2.get("court_x"); cy2 = p2.get("court_y")
        if cx1 is None or cy1 is None or cx2 is None or cy2 is None:
            continue
        ts_val = entry.get("timestamp_sec")
        fn_val = entry.get("frame_number")
        if ts_val is None or fn_val is None:
            continue
        frame_nums_arr.append(int(fn_val))
        ts_arr.append(float(ts_val))
        p1x.append(float(cx1)); p1y.append(float(cy1))
        p2x.append(float(cx2)); p2y.append(float(cy2))

    n = len(frame_nums_arr)
    print(f"  Per-sample series: {n} samples (both players present)")

    if n < 10:
        print(f"  ✗ Not enough valid samples ({n}) — need ≥ 10")
        point_starts = []
        ts = np.array([0.0]) if n == 0 else np.array(ts_arr)
        detection_funnel = {
            "algorithm": "player_activity_v2",
            "samples": n,
            "low_periods": 0,
            "candidates_with_burst": 0,
            "emitted": 0,
            "rejected": 0,
            "rejection_reasons": {},
            "ball_signal_available": have_ball,
            "note": "insufficient samples (n < 10)",
        }
    else:
        ts = np.array(ts_arr)

        # ---- Step 2: Per-player velocity ----
        v1 = [0.0] * n
        v2 = [0.0] * n
        for i in range(1, n):
            dt = max(ts_arr[i] - ts_arr[i - 1], 0.1)
            d1 = math.hypot(p1x[i] - p1x[i - 1], p1y[i] - p1y[i - 1]) / dt
            d2 = math.hypot(p2x[i] - p2x[i - 1], p2y[i] - p2y[i - 1]) / dt
            v1[i] = min(MAX_VELOCITY, d1)
            v2[i] = min(MAX_VELOCITY, d2)
        if n > 1:
            v1[0] = v1[1]; v2[0] = v2[1]

        # ---- Step 3: Smooth velocities (box filter, width SMOOTH_WINDOW) ----
        # Spec uses [i-2 .. i+2] inclusive (width 5 centered).
        half_w = SMOOTH_WINDOW // 2
        v1_s = [0.0] * n
        v2_s = [0.0] * n
        for i in range(n):
            lo = max(0, i - half_w)
            hi = min(n, i + half_w + 1)
            v1_s[i] = sum(v1[lo:hi]) / (hi - lo)
            v2_s[i] = sum(v2[lo:hi]) / (hi - lo)

        # ---- Step 4: Frozen-tracking detection (mean of RAW v over prior FROZEN_WINDOW) ----
        p1_frozen = [False] * n
        p2_frozen = [False] * n
        for i in range(n):
            lo = max(0, i - FROZEN_WINDOW)
            window = max(1, i - lo)
            m1 = sum(v1[lo:i + 1]) / (i - lo + 1)
            m2 = sum(v2[lo:i + 1]) / (i - lo + 1)
            p1_frozen[i] = (m1 < FROZEN_THRESH)
            p2_frozen[i] = (m2 < FROZEN_THRESH)

        combined = [0.0] * n
        for i in range(n):
            if p1_frozen[i] and p2_frozen[i]:
                combined[i] = 0.0
            elif p1_frozen[i]:
                combined[i] = v2_s[i] * 2.0
            elif p2_frozen[i]:
                combined[i] = v1_s[i] * 2.0
            else:
                combined[i] = v1_s[i] + v2_s[i]

        # ---- Step 5: Find low-activity periods ----
        low_periods = []  # (start_idx, end_idx, duration_sec)
        in_low = False
        low_start = 0
        for i in range(n):
            if combined[i] < LOW_THRESH:
                if not in_low:
                    in_low = True
                    low_start = i
            else:
                if in_low:
                    dur = ts_arr[i] - ts_arr[low_start]
                    if dur >= MIN_LOW_DURATION:
                        low_periods.append((low_start, i, dur))
                    in_low = False
        # Trailing low period
        if in_low and (ts_arr[n - 1] - ts_arr[low_start]) >= MIN_LOW_DURATION:
            low_periods.append((low_start, n - 1, ts_arr[n - 1] - ts_arr[low_start]))

        print(f"  Low-activity periods: {len(low_periods)}")

        # ---- Step 6: For each low period, find the burst that follows ----
        candidates_all = []   # all (post low+burst) candidates, regardless of score
        emitted = []          # surviving (post score gate + min-between)
        last_emitted_t = float("-inf")

        for (si, ei, gap_dur) in low_periods:
            # Find first burst within LOOKAHEAD_SAMPLES after the calm window
            burst_idx = None
            for j in range(ei, min(ei + LOOKAHEAD_SAMPLES, n)):
                if combined[j] > BURST_THRESH:
                    burst_idx = j
                    break
            if burst_idx is None:
                continue

            point_t = ts_arr[burst_idx]

            # MIN_BETWEEN suppression of duplicates
            if point_t - last_emitted_t < MIN_BETWEEN:
                # Still record as a candidate for the funnel debug.
                candidates_all.append({
                    "burst_idx": burst_idx,
                    "gap_dur": gap_dur,
                    "rejected": "min_between",
                })
                continue

            # Point end = start of the NEXT sustained calm period after this burst
            point_end_idx = None
            run_start = None
            for j in range(burst_idx + 1, n):
                if combined[j] < LOW_THRESH:
                    if run_start is None:
                        run_start = j
                    elif ts_arr[j] - ts_arr[run_start] >= MIN_LOW_DURATION:
                        point_end_idx = run_start
                        break
                else:
                    run_start = None

            # ---- Confidence signals ----
            opp_sides = (p1x[burst_idx] < HALF_COURT_X_V2) != (p2x[burst_idx] < HALF_COURT_X_V2)
            x_sep = abs(p1x[burst_idx] - p2x[burst_idx])
            bd_burst = ball_density_at(point_t, BALL_DENSITY_WINDOW_SEC)
            pause_t = (ts_arr[si] + ts_arr[ei]) / 2.0
            bd_pause = ball_density_at(pause_t, BALL_DENSITY_WINDOW_SEC)

            # ---- Continuous score (sum, capped at 1.0) ----
            score = 0.0
            score += min(GAP_CAP, gap_dur * GAP_WEIGHT_PER_SEC)
            if opp_sides:
                score += SEP_BASE + min(SEP_CAP - SEP_BASE,
                                        max(0.0, x_sep - 2.0) * SEP_PER_M_OVER_2)
            if bd_burst >= 0 and bd_pause >= 0:
                if bd_burst > bd_pause + 0.1:
                    jump = max(0.0, bd_burst - bd_pause)
                    score += min(BALL_CAP,
                                 BALL_REAPPEAR_BASE + jump * BALL_REAPPEAR_PER_DENSITY)
                # ball-present-throughout: no bonus (FP signature)
            score += min(BURST_CAP,
                         max(0.0, combined[burst_idx] - 1.5) * BURST_PER_M_OVER_1_5)
            score = min(1.0, score)

            candidate_record = {
                "burst_idx": burst_idx,
                "gap_dur": gap_dur,
                "burst_v": combined[burst_idx],
                "x_sep": x_sep,
                "opp_sides": opp_sides,
                "bd_burst": bd_burst,
                "bd_pause": bd_pause,
                "score": score,
                "point_end_idx": point_end_idx,
            }
            candidates_all.append(candidate_record)

            if score < MIN_EMIT_SCORE:
                candidate_record["rejected"] = "score_below_threshold"
                continue

            emitted.append(candidate_record)
            last_emitted_t = point_t

        # ---- Build legacy-shaped point_starts entries ----
        point_starts = []
        for c in emitted:
            bi = c["burst_idx"]
            pe = c["point_end_idx"]
            point_t = ts_arr[bi]
            end_t = ts_arr[pe] if pe is not None else None
            rally_dur = (end_t - point_t) if end_t is not None else None
            confidence_legacy = round(c["score"], 3)

            entry = {
                # Legacy fields downstream consumers rely on
                "point_number": len(point_starts) + 1,
                "timestamp_sec": round(point_t, 3),
                "frame_number": int(frame_nums_arr[bi]),
                "frame": int(frame_nums_arr[bi]),
                "confidence": confidence_legacy,
                # v2 spec fields
                "point_start_frame": int(frame_nums_arr[bi]),
                "point_start_timestamp_sec": round(point_t, 3),
                "point_end_frame": int(frame_nums_arr[pe]) if pe is not None else None,
                "point_end_timestamp_sec": round(end_t, 3) if end_t is not None else None,
                "rally_duration_sec": round(rally_dur, 3) if rally_dur is not None else None,
                "gap_before_sec": round(c["gap_dur"], 3),
                "burst_combined_velocity": round(c["burst_v"], 3),
                "x_separation_m": round(c["x_sep"], 3),
                "opposite_sides": bool(c["opp_sides"]),
                "ball_density_pause": round(c["bd_pause"], 3) if c["bd_pause"] >= 0 else None,
                "ball_density_burst": round(c["bd_burst"], 3) if c["bd_burst"] >= 0 else None,
                "score": round(c["score"], 3),
            }
            point_starts.append(entry)

        # ---- Funnel for debug ----
        rejection_counts = {}
        for c in candidates_all:
            r = c.get("rejected")
            if r:
                rejection_counts[r] = rejection_counts.get(r, 0) + 1
        detection_funnel = {
            "algorithm": "player_activity_v2",
            "samples": n,
            "low_periods": len(low_periods),
            "candidates_with_burst": len(candidates_all),
            "emitted": len(point_starts),
            "rejected": len(candidates_all) - len(point_starts),
            "rejection_reasons": rejection_counts,
            "ball_signal_available": have_ball,
        }
        print(f"  Detection funnel: {detection_funnel}")

    # ---- Build + upload point_starts.json (common to both branches) ----
    points_output = {
        "video_key": video_key,
        "total_points_detected": len(point_starts),
        "detection_funnel": detection_funnel,
        "time_range": {
            "start_sec": float(ts[0]) if len(ts) > 0 else 0.0,
            "end_sec": float(ts[-1]) if len(ts) > 0 else 0.0,
        },
        "detection_config": {
            "algorithm": "player_activity_v2",
            "low_thresh_m_per_s": LOW_THRESH,
            "min_low_duration_sec": MIN_LOW_DURATION,
            "burst_thresh_m_per_s": BURST_THRESH,
            "min_between_sec": MIN_BETWEEN,
            "lookahead_samples": LOOKAHEAD_SAMPLES,
            "max_velocity_m_per_s": MAX_VELOCITY,
            "frozen_thresh_m_per_s": FROZEN_THRESH,
            "frozen_window_samples": FROZEN_WINDOW,
            "smooth_window_samples": SMOOTH_WINDOW,
            "ball_density_window_sec": BALL_DENSITY_WINDOW_SEC,
            "min_emit_score": MIN_EMIT_SCORE,
            "score_caps": {
                "gap_cap": GAP_CAP,
                "sep_cap": SEP_CAP,
                "ball_cap": BALL_CAP,
                "burst_cap": BURST_CAP,
            },
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
        bv = ps.get("burst_combined_velocity") or 0.0
        gap = ps.get("gap_before_sec") or 0.0
        rd = ps.get("rally_duration_sec")
        rd_str = f"{rd:5.2f}s" if rd is not None else "  n/a"
        print(f"    Point {ps['point_number']:3d} | "
              f"start={ps['timestamp_sec']:7.2f}s (f{ps['frame_number']:6d}) | "
              f"rally={rd_str} | "
              f"gap_before={gap:5.2f}s | burst_v={bv:4.2f}m/s | "
              f"score={ps['score']:.2f}")

    # ================================================================
    # Step 6: Bounce and Hit Detection
    # ================================================================
    print("\n🏓 Running bounce and hit detection...")
    
    bounce_hit_result = None
    try:
        bounce_hit_result = detect_bounces_and_hits(
            ball_data, h_data, player_locations=world_output,
            player_masks=player_masks,
        )

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
    # Step 7: Hit list for rally breakdown (from detector attribution)
    # ================================================================
    # The separate shot-attribution pass has been removed. Rallies and
    # heatmaps now use the detector's own mask-based hit attribution
    # straight from bounce_hit_events: each hit becomes a shot owned by
    # the player the detector says struck it, and bounces inherit the
    # most recent hit's player. Forehand/backhand + shot-type
    # classification are deferred (to be revisited).
    print("\n🎯 Building hit list from detector attribution...")

    shot_attribution_url = None
    player_1_handedness = "right"
    player_2_handedness = "right"
    player_1_name = "Player 1"
    player_2_name = "Player 2"
    try:
        local_config = f"/tmp/player_config_{video_key}.json"
        config_gcs_path = f"{gcs_dir}/player_config.json"
        download_from_gcs(config_gcs_path, local_config)
        with open(local_config, "r") as f:
            player_config = json_mod.load(f)
        player_1_handedness = player_config.get("player_1", {}).get("handedness", "right")
        player_2_handedness = player_config.get("player_2", {}).get("handedness", "right")
        player_1_name = player_config.get("player_1", {}).get("name", "Player 1")
        player_2_name = player_config.get("player_2", {}).get("name", "Player 2")
        print(f"  Loaded player config: {player_1_name} ({player_1_handedness}), {player_2_name} ({player_2_handedness})")
    except Exception as e:
        print(f"  Using default player config: both right-handed (error: {e})")

    player_config_data = {
        "player_1": {"name": player_1_name, "handedness": player_1_handedness},
        "player_2": {"name": player_2_name, "handedness": player_2_handedness},
    }

    if bounce_hit_result and bounce_hit_result.get("events"):
        shot_result = shots_from_bounce_hits(bounce_hit_result, player_config_data)
        print(f"  ✅ {len(shot_result['shots'])} hits from detector attribution")
    else:
        print(f"  ⚠ No hit events; skipping rally breakdown")
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
                    "match_summary": compute_match_summary(point_rallies, player_config_data),
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
def finalize_and_upload(job_id: str, tracking_data: list, video_key: str, mask_data: list = None):
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

    # ---- Build player_masks.json (per-frame polygon contours) ----
    masks_json_path = None
    if mask_data:
        mask_deduped = {}
        for m_entry in mask_data:
            mask_deduped[m_entry["frame_number"]] = m_entry
        mask_frames_sorted = [mask_deduped[k] for k in sorted(mask_deduped.keys())]

        masks_output = {
            "source_resolution": {"width": sw, "height": sh},
            "source_fps": sfps,
            "total_frames_tracked": len(mask_frames_sorted),
            "model": "SAM2.1 Hiera Small",
            "polygon_simplification_epsilon_px": 2.0,
            "coordinate_description": (
                "Polygon vertices in source-resolution pixels. "
                "Each player may have multiple disjoint polygons (rare; usually 1)."
            ),
            "frames": mask_frames_sorted,
        }
        masks_json_path = str(job_dir / "player_masks.json")
        with open(masks_json_path, "w") as f:
            json.dump(masks_output, f)
        try:
            size_mb = os.path.getsize(masks_json_path) / (1024 * 1024)
            print(f"  ✓ player_masks.json built: {len(mask_frames_sorted)} frames, {size_mb:.1f} MB")
        except Exception:
            pass

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
    masks_url = None
    if masks_json_path and Path(masks_json_path).exists():
        try:
            masks_url = upload_to_gcs(masks_json_path, f"{gcs_dir}/player_masks.json")
            print(f"  ✓ player_masks.json uploaded to {gcs_dir}/player_masks.json")
        except Exception as e:
            print(f"  ⚠ player_masks.json upload failed: {e}")
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

    return {"json_url": json_url, "video_url": video_url, "masks_url": masks_url}


# ====================================================================
# Per-point clip renderer — produces a shareable mp4 of a single rally.
# Idempotent: a cached clip is returned without re-running ffmpeg.
# ====================================================================

@app.function(
    image=web_image,
    secrets=[gcs_secret],
    cpu=2,
    memory=4096,
    timeout=300,
)
def render_point_clip(video_key: str, point_number: int, pad_sec: float = 0.4) -> dict:
    """Trim the source mp4 to [start, end] of a single point and upload
    as clips/point_NNN.mp4. Returns the public Firebase Storage URL.
    Cached: re-rendering the same clip is a no-op."""
    import json as json_mod
    import subprocess
    import os as _os
    import urllib.parse

    gcs_dir = f"{OUTPUT_PREFIX}/{video_key}"
    clip_gcs_path = f"{gcs_dir}/clips/point_{point_number:03d}.mp4"

    bucket = _get_gcs_bucket()
    clip_blob = bucket.blob(clip_gcs_path)

    def _public_url(path):
        return (f"https://firebasestorage.googleapis.com/v0/b/{STORAGE_BUCKET}"
                f"/o/{urllib.parse.quote(path, safe='')}?alt=media")

    if clip_blob.exists():
        return {
            "status": "ok",
            "cached": True,
            "url": _public_url(clip_gcs_path),
            "video_key": video_key,
            "point_number": point_number,
        }

    # Fetch the point's start/end timestamps from points/point_NNN.json.
    local_point = f"/tmp/clip_point_{video_key}_{point_number}.json"
    try:
        download_from_gcs(f"{gcs_dir}/points/point_{point_number:03d}.json", local_point)
    except Exception as e:
        return {"status": "error", "message": f"Point JSON not found: {e}"}
    with open(local_point, "r") as f:
        pdata = json_mod.load(f)
    _os.unlink(local_point)

    start_sec = float(pdata.get("start_time_sec", 0))
    end_sec = float(pdata.get("end_time_sec") or (start_sec + 10.0))
    if end_sec <= start_sec:
        end_sec = start_sec + 10.0

    # Pad slightly each side so the clip doesn't feel guillotined.
    start_sec = max(0.0, start_sec - pad_sec)
    duration_sec = max(0.5, (end_sec - start_sec) + 2 * pad_sec)

    # Source video lives at uploaded-videos/{key}.mp4.
    local_source = f"/tmp/clip_source_{video_key}.mp4"
    try:
        download_from_gcs(f"uploaded-videos/{video_key}.mp4", local_source)
    except Exception as e:
        return {"status": "error", "message": f"Source video not found: {e}"}

    local_clip = f"/tmp/clip_{video_key}_{point_number}.mp4"
    # Re-encode (libx264) instead of stream-copy: stream-copy is keyframe-aligned
    # so it usually starts late or includes filler. Transcoding is ~3-5× realtime
    # for a 10-second clip on 2 vCPUs — fast enough for an interactive share.
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", local_source,
            "-t", f"{duration_sec:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",  # metadata at front → instant browser playback
            local_clip,
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        try:
            _os.unlink(local_source)
        except Exception:
            pass
        return {"status": "error", "message": f"ffmpeg failed: {e.stderr.decode()[-500:]}"}

    url = upload_to_gcs(local_clip, clip_gcs_path)
    try:
        _os.unlink(local_source); _os.unlink(local_clip)
    except Exception:
        pass

    return {
        "status": "ok",
        "cached": False,
        "url": url,
        "video_key": video_key,
        "point_number": point_number,
        "start_sec": round(start_sec, 3),
        "duration_sec": round(duration_sec, 3),
    }


# ====================================================================
# FastAPI Web Server
# ====================================================================

# Multi-game matches run as N concurrent WebSocket connections that all share
# the single TrackerGPU container, so each game runs at ~1/N speed. A 4-game
# match therefore needs well over an hour of wall-clock. The old 3600s timeout
# hard-cancelled every game mid-processing (CancelledError, uncatchable by the
# except-Exception path below) → cards stuck on "PROCESSING" forever. 2h gives
# a 4-game match room to finish; the in-handler soft deadline guarantees a
# terminal status is written before this hard limit is ever reached.
WEB_HARD_TIMEOUT_SEC = 7200  # Modal will hard-cancel the input at this point
PROCESSING_SOFT_DEADLINE_SEC = WEB_HARD_TIMEOUT_SEC - 300  # mark failed before the kill

# Independent, longer timeout for the spawned tracking loop. Because the loop
# runs decoupled from the WebSocket handler, its lifetime isn't bounded by
# client connection state, and it can safely take much longer than the web
# endpoint. Sized to fit a full 60-minute match at current per-segment rates.
TRACKING_LOOP_TIMEOUT_SEC = 14400  # 4 hours
TRACKING_LOOP_SOFT_DEADLINE_SEC = TRACKING_LOOP_TIMEOUT_SEC - 300


@app.function(
    image=web_image,
    volumes={DATA_DIR: scratch_vol},
    secrets=[gcs_secret, firebase_secret],
    timeout=TRACKING_LOOP_TIMEOUT_SEC,
)
def run_tracking_loop(
    job_id: str,
    video_key: str,
    src_width: int,
    src_height: int,
    src_fps: float,
    original_total_frames: int,
    trim_start_frame: int,
    trim_end_frame: int,
    seed_p1_proc: list,
    seed_p2_proc: list,
    yolo_box_p1_proc,   # list or None
    yolo_box_p2_proc,   # list or None
    params: dict,
) -> dict:
    """Long-running tracking loop, decoupled from any WebSocket.

    Runs the full segment loop, spawns render tasks, calls finalize + upload,
    and writes progress to gs://.../tracking_status.json throughout so the WS
    handler (and the /api/tracking-status endpoint) can surface progress to
    the client. On terminal state, writes the Firestore video status too, so
    cards flip to complete/failed even if the user closed the tab.

    Reprompt (both_lost / handoff_failed / p1_lost / p2_lost) is NOT
    supported in this decoupled path — the segment is logged as a warning
    and skipped. Interactive reprompt would require a bidirectional
    client<->spawn channel that we don't have here yet.

    Returns a dict with the final URLs on success, or {"error": "..."} on
    failure. Callers can `.get()` if they want the return, but the primary
    signal is the tracking_status.json state field.
    """
    import time as _time
    import tempfile
    import threading

    started_at_ts = _time.time()
    warnings_list: list = []

    def _status_dict(state, message, current_segment, est_total_segments,
                     frames_done, frames_total, pct, urls=None, error=None):
        return {
            "state": state,  # "starting" | "running" | "complete" | "error"
            "video_key": video_key,
            "job_id": job_id,
            "current_segment": current_segment,
            "est_total_segments": est_total_segments,
            "current_frames_done": frames_done,
            "current_frames_total": frames_total,
            "pct": pct,
            "message": message,
            "elapsed_sec": round(_time.time() - started_at_ts),
            "warnings": warnings_list,
            "urls": urls,
            "error": error,
        }

    try:
        trimmed_frames_count = trim_end_frame - trim_start_frame
        src_per_seg = int(math.ceil(params["frames_per_segment"] * params["frame_step"]))
        est_segments = math.ceil(trimmed_frames_count / src_per_seg)

        # Initial "starting" write so the client sees something within seconds.
        write_tracking_status(video_key, _status_dict(
            "starting",
            f"Starting: {est_segments} estimated segments",
            0, est_segments, 0, 0, 0.0,
        ))

        all_tracking_data: list = []
        all_mask_data: list = []
        seg_idx = 0
        render_tasks: list = []
        current_src_start = trim_start_frame
        prev_handoff = None
        prompt_local_idx = 0
        current_seed_p1 = seed_p1_proc
        current_seed_p2 = seed_p2_proc

        # Per-job GPU container (mirrors the ws_process pattern so multi-game
        # matches get separate containers per game and run in parallel).
        tracker_gpu = TrackerGPU(job_key=job_id)

        while current_src_start < trim_end_frame:
            seg_idx += 1

            # Soft deadline: bail cleanly before the hard timeout.
            if _time.time() - started_at_ts > TRACKING_LOOP_SOFT_DEADLINE_SEC:
                raise TimeoutError(
                    f"Processing exceeded soft deadline "
                    f"({TRACKING_LOOP_SOFT_DEADLINE_SEC}s) after {seg_idx - 1} segments."
                )

            src_remaining = trim_end_frame - current_src_start
            src_num = min(src_per_seg, src_remaining)

            _yolo_p1 = yolo_box_p1_proc if seg_idx == 1 else None
            _yolo_p2 = yolo_box_p2_proc if seg_idx == 1 else None

            sam2_total = int(src_num / params["frame_step"])

            # Announce segment start.
            write_tracking_status(video_key, _status_dict(
                "running",
                f"Segment {seg_idx}/{est_segments}",
                seg_idx, est_segments, 0, sam2_total,
                round((seg_idx - 1) / max(est_segments, 1) * 100, 1),
            ))

            # Fire the segment on a thread so we can poll progress.json while
            # SAM2 propagates. .remote() blocks until the Modal call resolves.
            gpu_result_container: dict = {}

            def _worker(_prev_handoff=prev_handoff,
                        _seed_p1=current_seed_p1, _seed_p2=current_seed_p2,
                        _prompt_local_idx=prompt_local_idx,
                        _current_src_start=current_src_start, _src_num=src_num,
                        _seg_idx=seg_idx, _yolo1=_yolo_p1, _yolo2=_yolo_p2):
                try:
                    gpu_result_container["result"] = tracker_gpu.process_segment.remote(
                        job_id, _seg_idx, _current_src_start, _src_num,
                        _seed_p1, _seed_p2, _prompt_local_idx, params,
                        _prev_handoff, _yolo1, _yolo2,
                    )
                except Exception as e:
                    gpu_result_container["exception"] = e

            worker = threading.Thread(target=_worker, daemon=True)
            worker.start()

            seg_start_time = _time.time()
            prog_path = f"{DATA_DIR}/jobs/{job_id}/progress.json"

            while worker.is_alive():
                worker.join(timeout=10.0)
                if not worker.is_alive():
                    break
                # Poll the GPU-side progress file and update the status JSON.
                frames_done = 0
                try:
                    scratch_vol.reload()
                    with open(prog_path, "r") as pf:
                        prog_data = json.load(pf)
                        frames_done = prog_data.get("frames_done", 0)
                        sam2_total = prog_data.get("total", sam2_total)
                except Exception:
                    pass

                elapsed_seg = _time.time() - seg_start_time
                # Overall pct = fully-done segments + fraction of current.
                overall_pct = round(
                    (seg_idx - 1 + frames_done / max(sam2_total, 1)) / max(est_segments, 1) * 100,
                    1,
                )
                write_tracking_status(video_key, _status_dict(
                    "running",
                    f"Segment {seg_idx}/{est_segments}: {frames_done}/{sam2_total} frames",
                    seg_idx, est_segments, frames_done, sam2_total, overall_pct,
                ))

            if "exception" in gpu_result_container:
                raise gpu_result_container["exception"]
            result = gpu_result_container.get("result") or {}

            status = result.get("status", "error")

            # Reprompt states: log warning + skip. See docstring.
            if status in ("both_lost", "handoff_failed", "p1_lost", "p2_lost"):
                warnings_list.append({
                    "segment": seg_idx,
                    "status": status,
                    "message": result.get("message", f"Segment {seg_idx}: {status}"),
                })
                print(f"  ⚠ Segment {seg_idx} lost tracking ({status}) — skipping without reprompt")
                if result.get("tracking_data"):
                    all_tracking_data.extend(result["tracking_data"])
                if result.get("mask_data"):
                    all_mask_data.extend(result["mask_data"])
                current_src_start += src_num
                prompt_local_idx = 0
                prev_handoff = None
                continue

            # Normal segment: collect data, upload per-seg JSON, spawn render.
            if result.get("tracking_data"):
                all_tracking_data.extend(result["tracking_data"])
            if result.get("mask_data"):
                all_mask_data.extend(result["mask_data"])

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
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                        tf.write(seg_json_str)
                        tf_path = tf.name
                    upload_to_gcs(tf_path, seg_gcs_path)
                    os.unlink(tf_path)
                    print(f"  ✓ Segment {seg_idx} JSON uploaded ({len(seg_tracking)} frames)")
                except Exception as e:
                    print(f"  ⚠ Segment JSON upload failed: {e}")

            if result.get("status") not in ("both_lost", "handoff_failed"):
                render_handle = render_segment.spawn(job_id, seg_idx, params)
                render_tasks.append(render_handle)

            # Advance via handoff (or fall through to next segment).
            ho = result.get("handoff")
            if ho:
                current_seed_p1 = ho["seed_p1_proc"]
                current_seed_p2 = ho["seed_p2_proc"]
                current_src_start = ho["next_src_start"]
                prompt_local_idx = ho["prompt_local_idx"]
                prev_handoff = ho.get("prev_handoff")
            else:
                current_src_start += src_num
                prompt_local_idx = 0
                prev_handoff = None

            # Segment-complete status update.
            write_tracking_status(video_key, _status_dict(
                "running",
                f"Segment {seg_idx}/{est_segments} complete",
                seg_idx, est_segments, sam2_total, sam2_total,
                round(seg_idx / max(est_segments, 1) * 100, 1),
            ))

        # ---- Wait for parallel renders ----
        write_tracking_status(video_key, _status_dict(
            "running",
            f"Rendering {len(render_tasks)} segments...",
            seg_idx, est_segments, 0, 0, 95.0,
        ))
        for i, handle in enumerate(render_tasks):
            try:
                handle.get()
            except Exception as e:
                print(f"  ⚠ Render task {i} failed: {e}")

        # ---- Finalize ----
        write_tracking_status(video_key, _status_dict(
            "running",
            "Finalizing output...",
            seg_idx, est_segments, 0, 0, 97.0,
        ))
        urls = finalize_and_upload.remote(
            job_id, all_tracking_data, video_key, all_mask_data
        )

        # Compute point_starts_url (mirrors old WS handler behavior).
        point_starts_url = None
        try:
            import urllib.parse
            gcs_path = f"video-data/{video_key}/point_starts.json"
            encoded_path = urllib.parse.quote(gcs_path, safe="")
            point_starts_url = f"https://firebasestorage.googleapis.com/v0/b/boastiq.firebasestorage.app/o/{encoded_path}?alt=media"
        except Exception as e:
            print(f"Warning: Could not generate point_starts_url: {e}")

        final_urls = {
            "json_url": urls.get("json_url"),
            "video_url": urls.get("video_url"),
            "masks_url": urls.get("masks_url"),
            "point_starts_url": point_starts_url,
            "total_segments": seg_idx,
            "total_frames_tracked": len(all_tracking_data),
        }

        # Terminal "complete" write.
        write_tracking_status(video_key, _status_dict(
            "complete",
            "Complete",
            seg_idx, est_segments, 0, 0, 100.0, urls=final_urls,
        ))

        # Server-side Firestore status write so the video card flips even if
        # the tab is long gone.
        if video_key:
            _mark_video_status(video_key, "complete")

        return final_urls

    except Exception as e:
        err_msg = str(e)
        print(f"⚠ run_tracking_loop failed for {video_key}: {err_msg}")
        try:
            write_tracking_status(video_key, {
                "state": "error",
                "video_key": video_key,
                "job_id": job_id,
                "error": err_msg,
                "warnings": warnings_list,
                "elapsed_sec": round(_time.time() - started_at_ts),
            })
        except Exception:
            pass
        if video_key:
            _mark_video_status(video_key, "failed")
        return {"error": err_msg}


@app.function(
    image=web_image,
    volumes={DATA_DIR: scratch_vol},
    secrets=[gcs_secret, firebase_secret, stripe_secret],
    timeout=WEB_HARD_TIMEOUT_SEC,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI()
    api.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    tracker_gpu = TrackerGPU()

    def require_admin(request: Request) -> dict:
        """FastAPI dependency: verify Firebase ID token + admin custom claim."""
        try:
            return verify_admin_token(request.headers.get("authorization", ""))
        except ValueError as e:
            code = str(e)
            if code == "not_admin":
                raise HTTPException(status_code=403, detail="admin_only")
            raise HTTPException(status_code=401, detail=code)

    @api.get("/")
    async def index():
        html = Path("/app/index.html").read_text()
        return HTMLResponse(html)

    @api.get("/admin/tag")
    async def admin_tag_page():
        """Admin court-landmark tagging UI. Auth happens client-side (the
        page redirects non-admins after sign-in); the underlying admin API
        calls are gated by `require_admin`."""
        html = Path("/app/Tag.html").read_text()
        return HTMLResponse(html)

    @api.get("/admin/api/pending")
    async def admin_list_pending(claims: dict = Depends(require_admin)):
        try:
            jobs = await asyncio.to_thread(list_pending_landmark_jobs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"list_failed: {e}")
        return JSONResponse({"pending": jobs, "tagger_uid": claims.get("uid")})

    @api.post("/admin/api/backfill-thumbnails")
    async def admin_backfill_thumbnails(body: dict = None, claims: dict = Depends(require_admin)):
        """One-off: set thumbnailPath on existing video docs that have a
        first_frame.png in Storage but no thumbnailPath. Pass {"force": true}
        to also overwrite docs that already have one."""
        force = bool((body or {}).get("force"))
        try:
            result = await asyncio.to_thread(backfill_thumbnail_paths, force)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"backfill_failed: {e}")
        return JSONResponse(result)

    @api.get("/admin/api/frame/{video_key}")
    async def admin_get_frame(video_key: str, claims: dict = Depends(require_admin)):
        """Return a signed read URL for this video's first_frame.png plus
        its natural pixel dimensions (so the browser can map clicks back to
        source-resolution coords)."""
        import cv2
        import tempfile
        gcs_path = f"{OUTPUT_PREFIX}/{video_key}/first_frame.png"
        url = generate_signed_read_url(gcs_path, hours=4)
        # Probe the natural dimensions so the tagger can scale clicks correctly.
        local = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        try:
            await asyncio.to_thread(download_from_gcs, gcs_path, local)
            img = cv2.imread(local)
            if img is None:
                raise HTTPException(status_code=404, detail="first_frame missing")
            h, w = img.shape[:2]
        finally:
            try:
                os.unlink(local)
            except Exception:
                pass
        return JSONResponse({
            "video_key": video_key,
            "frame_url": url,
            "src_width": int(w),
            "src_height": int(h),
            "landmark_order": COURT_LANDMARK_ORDER,
        })

    @api.post("/admin/api/landmarks/{video_key}")
    async def admin_submit_landmarks(video_key: str, request: Request,
                                     claims: dict = Depends(require_admin)):
        """Receive 17 hand-tagged landmark coords, build per-surface
        homographies, upload `homography.json` to GCS. The blocked
        downstream pipeline picks it up automatically on its next poll."""
        body = await request.json()
        landmarks = body.get("landmarks") or {}
        src_width = int(body.get("src_width") or 0)
        src_height = int(body.get("src_height") or 0)
        if src_width <= 0 or src_height <= 0:
            raise HTTPException(status_code=400,
                                detail="src_width and src_height required")
        missing = [n for n in COURT_LANDMARK_ORDER if n not in landmarks]
        if missing:
            raise HTTPException(status_code=400,
                                detail=f"missing_landmarks:{missing}")
        try:
            result = await asyncio.to_thread(
                build_and_upload_manual_homography,
                video_key, landmarks, src_width, src_height,
                claims.get("email") or claims.get("uid") or "admin",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"build_failed: {e}")
        return JSONResponse(result)

    # ---- Manual hit/bounce annotation (admin-only) ----------------------
    @api.get("/admin/annotate")
    async def admin_annotate_page():
        """Serve the hit & bounce annotator UI. Auth happens client-side;
        the underlying /state and /save endpoints are gated by require_admin."""
        html = Path("/app/Annotate.html").read_text()
        return HTMLResponse(html)

    @api.get("/admin/annotate/{video_key}/state")
    async def admin_annotate_state(video_key: str,
                                   claims: dict = Depends(require_admin)):
        """Return everything the annotator needs on load: signed URL for the
        source video, fps + duration, existing bounce_hit_events, homography
        (for pixel→world projection), and player config (for display names)."""
        import json as json_mod
        import tempfile
        import os
        gcs_dir = f"{OUTPUT_PREFIX}/{video_key}"

        def _try_load(rel_path):
            local = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
            try:
                download_from_gcs(f"{gcs_dir}/{rel_path}", local)
                with open(local, "r") as f:
                    return json_mod.load(f)
            except Exception:
                return None
            finally:
                try:
                    os.unlink(local)
                except Exception:
                    pass

        events_data = _try_load("bounce_hit_events.json") or {"events": []}
        homography  = _try_load("homography.json")
        player_cfg  = _try_load("player_config.json") or {
            "player_1": {"name": "Player 1", "handedness": "right"},
            "player_2": {"name": "Player 2", "handedness": "right"},
        }
        locs = _try_load("player_locations.json") or {"frames": []}

        # fps + duration: prefer player_locations metadata (already computed
        # at processing time). Fall back to inspecting the video with cv2.
        fps = locs.get("source_fps") or 30
        duration = 0.0
        frames = locs.get("frames") or []
        if frames:
            last = frames[-1] or {}
            duration = float(last.get("timestamp_sec") or 0) + (1.0 / fps)
        if duration <= 0:
            # cv2 probe fallback
            import cv2
            local_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            try:
                download_from_gcs(f"uploaded-videos/{video_key}.mp4", local_vid)
                cap = cv2.VideoCapture(local_vid)
                if cap.isOpened():
                    cap_fps = cap.get(cv2.CAP_PROP_FPS) or fps
                    cap_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    if cap_fps > 0:
                        fps = cap_fps
                    if cap_frames > 0 and cap_fps > 0:
                        duration = cap_frames / cap_fps
                cap.release()
            except Exception as e:
                print(f"  cv2 probe failed: {e}")
            finally:
                try:
                    os.unlink(local_vid)
                except Exception:
                    pass

        video_url = generate_signed_read_url(
            f"uploaded-videos/{video_key}.mp4", hours=6,
        )

        return JSONResponse({
            "video_key": video_key,
            "video_url": video_url,
            "fps": float(fps),
            "duration_sec": float(duration),
            "events": events_data.get("events", []),
            "homographies": (homography or {}).get("homographies") if homography else None,
            "player_config": player_cfg,
        })

    @api.post("/admin/annotate/{video_key}/save")
    async def admin_annotate_save(video_key: str, request: Request,
                                  claims: dict = Depends(require_admin)):
        """Accept the edited event list, overwrite bounce_hit_events.json,
        re-run shot attribution + rally build + match_summary. Everything
        the results page reads is refreshed."""
        import json as json_mod
        import tempfile
        import os
        gcs_dir = f"{OUTPUT_PREFIX}/{video_key}"

        body = await request.json()
        events = body.get("events") or []
        if not isinstance(events, list):
            raise HTTPException(status_code=400, detail="events must be a list")

        # Normalize + sort. Downstream code assumes sort-by-timestamp.
        for e in events:
            e["timestamp_sec"] = float(e.get("timestamp_sec") or 0)
            e["frame"] = int(e.get("frame") or 0)
        events.sort(key=lambda e: e["timestamp_sec"])

        # 1. Load dependencies.
        def _load_or_fail(rel_path, label):
            local = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
            try:
                download_from_gcs(f"{gcs_dir}/{rel_path}", local)
                with open(local, "r") as f:
                    return json_mod.load(f)
            except Exception as ex:
                raise HTTPException(
                    status_code=400,
                    detail=f"missing {label} ({rel_path}): {ex}",
                )
            finally:
                try:
                    os.unlink(local)
                except Exception:
                    pass

        world_output   = _load_or_fail("player_locations.json", "player locations")
        homography_data = _load_or_fail("homography.json",     "homography")
        point_starts_j = _load_or_fail("point_starts.json",    "point starts")
        player_config  = _load_or_fail("player_config.json",   "player config")

        # 2. Write edited bounce_hit_events.json (write-through).
        bounce_hit_result = {
            "total_events": len(events),
            "events": events,
            "overlay": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json_mod.dump(bounce_hit_result, tf, indent=2)
            tf_path = tf.name
        upload_to_gcs(tf_path, f"{gcs_dir}/bounce_hit_events.json")
        os.unlink(tf_path)

        # 3. Build shots straight from the (edited) hit events using the
        #    detector's attribution. Manually-labeled hits already carry
        #    `player` (1/2), which shots_from_bounce_hits honors directly, so
        #    the user's ground-truth wins. (The separate shot_attribution.json
        #    pass has been removed.)
        shot_result = shots_from_bounce_hits(bounce_hit_result, player_config)

        # 4. Rebuild rallies.
        point_starts = point_starts_j.get("point_starts", []) if isinstance(point_starts_j, dict) else []
        frames = world_output.get("frames", [])
        total_duration_sec = 0.0
        if frames:
            last = frames[-1] or {}
            total_duration_sec = float(last.get("timestamp_sec") or 0) + 1.0
        src_fps = world_output.get("source_fps", 30)

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
            raise HTTPException(status_code=500, detail="no rallies built after save")

        for pr in point_rallies:
            pn = pr.get("point_number", 0)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                json_mod.dump(pr, tf, indent=2)
                tf_path = tf.name
            upload_to_gcs(tf_path, f"{gcs_dir}/points/point_{pn:03d}.json")
            os.unlink(tf_path)

        # 5. Rewrite points/index.json with fresh match_summary.
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
            "match_summary": compute_match_summary(point_rallies, player_config),
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json_mod.dump(points_index, tf, indent=2)
            tf_path = tf.name
        upload_to_gcs(tf_path, f"{gcs_dir}/points/index.json")
        os.unlink(tf_path)

        return JSONResponse({
            "status": "ok",
            "video_key": video_key,
            "total_events": len(events),
            "total_points": len(point_rallies),
            "match_summary": points_index["match_summary"],
        })

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

    @api.get("/api/clip/{video_key}/{point_number}")
    async def get_or_render_clip(video_key: str, point_number: int):
        """Return public URL of a single-point clip. Triggers an ffmpeg
        render on the worker if the clip doesn't exist yet (cached
        afterward). No auth — the URL is meant to be shared."""
        try:
            result = render_point_clip.remote(video_key, point_number)
            if result.get("status") == "error":
                return JSONResponse(result, status_code=500)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

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
            
            # 2. Bounce/hit events
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

            # Shots come from the detector's own attribution in the events.
            shot_result = shots_from_bounce_hits(bounce_hit_result, player_config)
            print(f"  Built {len(shot_result.get('shots', []))} shots from detector attribution")

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
                "match_summary": compute_match_summary(point_rallies, player_config),
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

    @api.post("/api/backfill-match-summary/{video_key}")
    async def backfill_match_summary(video_key: str, force: bool = False):
        """
        Compute match_summary for a video that was processed before the
        summary-embedding change, and rewrite points/index.json with it
        included. Reads the existing per-point JSONs; does NOT re-run
        tracking. Idempotent — with force=false, skips videos that
        already have a summary.
        """
        import json as json_mod
        import tempfile
        import os

        gcs_dir = f"video-data/{video_key}"

        try:
            local_index = f"/tmp/bfsum_index_{video_key}.json"
            try:
                download_from_gcs(f"{gcs_dir}/points/index.json", local_index)
                with open(local_index, "r") as f:
                    index_data = json_mod.load(f)
            except Exception as e:
                return JSONResponse(
                    {"error": f"index.json not found: {e}", "status": "not_ready"},
                    status_code=404,
                )

            if index_data.get("match_summary") and not force:
                return JSONResponse({
                    "status": "skipped",
                    "reason": "match_summary already present (pass force=true to recompute)",
                    "video_key": video_key,
                })

            point_entries = index_data.get("points") or []
            point_rallies = []
            missing = []
            for entry in point_entries:
                pn = entry.get("point_number")
                if pn is None:
                    continue
                local_pt = f"/tmp/bfsum_pt_{video_key}_{pn}.json"
                try:
                    download_from_gcs(
                        f"{gcs_dir}/points/point_{int(pn):03d}.json",
                        local_pt,
                    )
                    with open(local_pt, "r") as f:
                        point_rallies.append(json_mod.load(f))
                except Exception as e:
                    missing.append({"point_number": pn, "error": str(e)})
                finally:
                    try:
                        os.unlink(local_pt)
                    except Exception:
                        pass

            if not point_rallies:
                return JSONResponse(
                    {"error": "no point rallies could be loaded", "missing": missing, "status": "failed"},
                    status_code=500,
                )

            player_config = {
                "player_1": index_data.get("player_1") or {},
                "player_2": index_data.get("player_2") or {},
            }
            index_data["match_summary"] = compute_match_summary(point_rallies, player_config)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                json_mod.dump(index_data, tf, indent=2)
                tf_path = tf.name
            index_url = upload_to_gcs(tf_path, f"{gcs_dir}/points/index.json")
            os.unlink(tf_path)

            return JSONResponse({
                "status": "ok",
                "video_key": video_key,
                "points_included": len(point_rallies),
                "points_missing": missing,
                "index_url": index_url,
                "match_summary": index_data["match_summary"],
            })
        except Exception as e:
            import traceback
            return JSONResponse(
                {"error": str(e), "trace": traceback.format_exc(), "status": "failed"},
                status_code=500,
            )

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

            # 2b. Download player_masks.json (optional; enables hit detection)
            player_masks = None
            try:
                local_masks = f"/tmp/regen_masks_{video_key}.json"
                download_from_gcs(f"{gcs_dir}/player_masks.json", local_masks)
                with open(local_masks, "r") as f:
                    player_masks = json_mod.load(f)
                print(f"  Loaded player masks: {len(player_masks.get('frames', []))} frames")
            except Exception as e:
                print(f"  ⚠ Player masks unavailable ({e}); hits will not be detected")

            # 3. Run detect_bounces_and_hits
            bounce_hit_result = detect_bounces_and_hits(
                ball_data, homography_data, player_masks=player_masks
            )
            
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

            # Shots come from the freshly regenerated detector attribution.
            shot_result = shots_from_bounce_hits(bounce_hit_result, player_config)

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
                "match_summary": compute_match_summary(point_rallies, player_config),
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

    # ---- Subscription paywall -------------------------------------------
    def require_user(request: Request) -> dict:
        """FastAPI helper: verify a Firebase ID token (any signed-in user).
        Raises HTTP 401 on failure."""
        try:
            return verify_token(request.headers.get("authorization", ""))
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))

    def require_subscription(request: Request) -> dict:
        """Verify the caller is signed in AND has an active subscription.
        Raises 401 if unauthenticated, 402 if unsubscribed. Returns claims.

        Anonymous (trial-funnel) users are exempt — they get the single free
        upload the client caps via localStorage, matching the pre-paywall trial.
        """
        claims = require_user(request)
        if (claims.get("firebase") or {}).get("sign_in_provider") == "anonymous":
            return claims
        billing = get_user_billing(claims.get("uid"))
        if not has_active_subscription(billing):
            raise HTTPException(status_code=402, detail="subscription_required")
        return claims

    @api.post("/api/create-subscription")
    async def create_subscription(request: Request):
        """Create (or reuse) a Stripe customer + an incomplete monthly
        subscription for the signed-in user, and return the first invoice's
        PaymentIntent client_secret so the browser's Payment Element can
        confirm the card in-page."""
        import stripe
        claims = require_user(request)
        uid = claims.get("uid")
        email = claims.get("email")

        # Plan selection: "annual"/"yearly" → yearly price; anything else → monthly.
        # Both plans carry the same 7-day trial, so the rest of the flow is shared.
        try:
            body = await request.json()
        except Exception:
            body = {}
        plan = str((body or {}).get("plan", "monthly")).lower()
        is_annual = plan in ("annual", "yearly", "year")

        stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
        if is_annual:
            price_id = os.environ.get("STRIPE_PRICE_ID_ANNUAL", "") or STRIPE_PRICE_ID_ANNUAL
        else:
            price_id = os.environ.get("STRIPE_PRICE_ID", "")
        if not stripe.api_key or not price_id:
            raise HTTPException(status_code=503, detail="stripe_not_configured")

        billing = get_user_billing(uid)
        if has_active_subscription(billing):
            # Nothing to pay for — client should just refresh its state.
            return JSONResponse({"status": "already_active"}, status_code=409)

        try:
            customer_id = billing.get("stripeCustomerId")
            if not customer_id:
                customer = await asyncio.to_thread(
                    lambda: stripe.Customer.create(
                        email=email, metadata={"firebaseUid": uid}))
                customer_id = customer.id
                await asyncio.to_thread(
                    set_user_billing, uid, {"stripeCustomerId": customer_id})

            sub = await asyncio.to_thread(
                lambda: stripe.Subscription.create(
                    customer=customer_id,
                    items=[{"price": price_id}],
                    # 7-day free trial: the subscription starts in `trialing`
                    # (which grants access) and the first real charge lands when
                    # the trial ends. We still collect the card upfront via a
                    # SetupIntent so we can charge automatically at trial end.
                    trial_period_days=TRIAL_PERIOD_DAYS,
                    trial_settings={
                        "end_behavior": {"missing_payment_method": "cancel"}},
                    payment_behavior="default_incomplete",
                    payment_settings={
                        "save_default_payment_method": "on_subscription"},
                    expand=["pending_setup_intent",
                            "latest_invoice.payment_intent"],
                    metadata={"firebaseUid": uid,
                              "plan": "annual" if is_annual else "monthly"}))
        except Exception as e:
            print(f"✗ create-subscription failed for {uid}: {e}")
            raise HTTPException(status_code=502, detail=f"stripe_error:{e}")

        # With a trial, the first invoice is $0 so there's no PaymentIntent —
        # collect + save the card via the SetupIntent instead (confirmSetup on
        # the client). Fall back to the PaymentIntent path for a trial-less
        # subscription (e.g. if the trial is ever removed).
        client_secret = None
        mode = None
        setup_intent = getattr(sub, "pending_setup_intent", None)
        if setup_intent:
            client_secret = setup_intent.client_secret
            mode = "setup"
        else:
            try:
                client_secret = sub.latest_invoice.payment_intent.client_secret
                mode = "payment"
            except Exception:
                pass
        if not client_secret:
            raise HTTPException(status_code=502, detail="no_client_secret")
        return JSONResponse({
            "subscriptionId": sub.id,
            "clientSecret": client_secret,
            "mode": mode,
            "plan": "annual" if is_annual else "monthly",
        })

    @api.get("/api/subscription-status")
    async def subscription_status(request: Request):
        """Return the caller's billing state (convenience; the client can also
        read users/{uid} from Firestore directly)."""
        claims = require_user(request)
        billing = get_user_billing(claims.get("uid"))
        return JSONResponse({
            "active": has_active_subscription(billing),
            **billing,
        })

    @api.post("/api/apply-promo")
    async def apply_promo(request: Request, body: dict):
        """Validate a customer-facing promotion code and attach its coupon to the
        caller's just-created (incomplete/trialing) subscription, so the discount
        applies to invoices after the trial. Returns the resulting monthly total
        for display. 404 if the code is invalid/expired."""
        import stripe
        claims = require_user(request)
        uid = claims.get("uid")
        sub_id = (body or {}).get("subscriptionId")
        code = ((body or {}).get("code") or "").strip()
        if not sub_id or not code:
            raise HTTPException(status_code=400, detail="missing_params")

        stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
        if not stripe.api_key:
            raise HTTPException(status_code=503, detail="stripe_not_configured")

        try:
            promos = await asyncio.to_thread(
                lambda: stripe.PromotionCode.list(
                    code=code, active=True, limit=1,
                    stripe_version=STRIPE_PROMO_API_VERSION))
            promo = promos.data[0] if promos.data else None
            if not promo:
                raise HTTPException(status_code=404, detail="invalid_promo")

            # Ownership check: the subscription's customer must match this user's.
            sub = await asyncio.to_thread(
                lambda: stripe.Subscription.retrieve(
                    sub_id, stripe_version=STRIPE_PROMO_API_VERSION))
            billing = get_user_billing(uid)
            if sub.customer != billing.get("stripeCustomerId"):
                raise HTTPException(status_code=403, detail="not_your_subscription")

            await asyncio.to_thread(
                lambda: stripe.Subscription.modify(
                    sub_id, discounts=[{"promotion_code": promo.id}],
                    stripe_version=STRIPE_PROMO_API_VERSION))
        except HTTPException:
            raise
        except Exception as e:
            print(f"✗ apply-promo failed for {uid} code={code!r}: {e}")
            raise HTTPException(status_code=502, detail=f"stripe_error:{e}")

        # Compute the resulting monthly total (base price − coupon) for display.
        price_id = os.environ.get("STRIPE_PRICE_ID", "")
        try:
            base = await asyncio.to_thread(
                lambda: stripe.Price.retrieve(
                    price_id, stripe_version=STRIPE_PROMO_API_VERSION))
            base_cents = base.unit_amount or 0
        except Exception:
            base_cents = 0
        coupon = promo.coupon
        new_cents = base_cents
        if getattr(coupon, "amount_off", None):
            new_cents = max(0, base_cents - coupon.amount_off)
        elif getattr(coupon, "percent_off", None):
            new_cents = int(round(base_cents * (1 - coupon.percent_off / 100.0)))
        return JSONResponse({
            "applied": True,
            "code": code,
            "newMonthlyCents": new_cents,
        })

    @api.post("/api/stripe-webhook")
    async def stripe_webhook(request: Request):
        """Stripe → server webhook. Verifies the signature, then writes the
        subscription state into users/{uid} (the source of truth the app gates
        on). Server-to-server, no browser CORS involved."""
        import stripe
        stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
        webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        payload = await request.body()
        sig = request.headers.get("stripe-signature", "")
        try:
            # Verify the signature (raises on tampering). We then read fields from
            # the raw JSON payload as plain dicts — modern stripe-python resource
            # objects (v15) do NOT support dict-style .get(), which would raise
            # AttributeError('get') on every event.
            stripe.Webhook.construct_event(payload, sig, webhook_secret)
        except Exception as e:
            print(f"✗ stripe webhook signature verification failed: {e}")
            raise HTTPException(status_code=400, detail="invalid_signature")

        event = json.loads(payload)
        etype = event["type"]
        obj = event["data"]["object"]

        def _uid_for(o):
            uid = (o.get("metadata") or {}).get("firebaseUid")
            if uid:
                return uid
            return find_uid_by_stripe_customer(o.get("customer"))

        try:
            if etype.startswith("customer.subscription."):
                uid = _uid_for(obj)
                status = obj.get("status")  # active | past_due | canceled | ...
                if etype == "customer.subscription.deleted":
                    status = "canceled"
                await asyncio.to_thread(set_user_billing, uid, {
                    "subscriptionId": obj.get("id"),
                    "stripeCustomerId": obj.get("customer"),
                    "subscriptionStatus": status,
                    "plan": "pro" if status in _ACTIVE_SUB_STATUSES else "free",
                    "currentPeriodEnd": obj.get("current_period_end"),
                })
                print(f"✓ webhook {etype}: uid={uid} status={status}")
            elif etype in ("invoice.paid", "invoice.payment_succeeded"):
                uid = _uid_for(obj)
                if uid:
                    await asyncio.to_thread(set_user_billing, uid, {
                        "subscriptionStatus": "active",
                        "plan": "pro",
                        "stripeCustomerId": obj.get("customer"),
                    })
                print(f"✓ webhook {etype}: uid={uid}")
            elif etype == "invoice.payment_failed":
                uid = _uid_for(obj)
                if uid:
                    await asyncio.to_thread(set_user_billing, uid, {
                        "subscriptionStatus": "past_due",
                    })
                print(f"✓ webhook {etype}: uid={uid}")
        except Exception as e:
            print(f"⚠ webhook {etype} handling failed: {e}")

        return JSONResponse({"received": True})

    @api.post("/api/create-job")
    async def create_job(request: Request, body: dict = None):
        # Paywall: uploading/analyzing a video requires an active subscription.
        require_subscription(request)
        body = body or {}
        job_id = str(uuid.uuid4())[:8]
        # Accept client-provided video_key (new auth flow generates it client-side so
        # the Firestore record can be created before upload). Fall back to server-side
        # generation for any caller that doesn't supply one.
        video_key = body.get("video_key") or make_video_key(body.get("filename", "video.mp4"))
        return JSONResponse({"job_id": job_id, "video_key": video_key})

    @api.post("/api/extract-first-frame")
    async def extract_first_frame_endpoint(body: dict = None):
        """Kick off authoritative server-side first-frame extraction for a video
        whose browser-side grab was black/missing (iOS Safari). Fire-and-forget:
        regenerate_first_frame writes first_frame.png + thumbnailPath itself."""
        body = body or {}
        video_key = body.get("video_key")
        if not video_key:
            return JSONResponse({"error": "video_key required"}, status_code=400)
        gcs_path = body.get("gcs_path", "")
        regenerate_first_frame.spawn(video_key, gcs_path)
        return JSONResponse({"status": "spawned", "video_key": video_key})

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
                "uid": body.get("player_1_uid") or None,
            },
            "player_2": {
                "name": body.get("player_2_name", "Player 2"),
                "handedness": body.get("player_2_handedness", "right"),
                "uid": body.get("player_2_uid") or None,
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
        # Return proper 400 JSON instead of a 500 traceback when the client
        # sends a partial body (e.g. after create-job returned 402 upstream
        # and the client kept going with an undefined job_id).
        video_key = body.get("video_key")
        if not video_key:
            return JSONResponse(
                {"error": "video_key is required"}, status_code=400
            )
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
    async def prepare(request: Request, body: dict):
        """
        Returns immediately with a call_id. prepare_job runs in the background
        via Modal's .spawn(); the client polls /api/prepare-status until done.

        This avoids Modal's HTTP request idle timeout on long videos where the
        GCS download + ffprobe per-frame PTS extraction can exceed ~150s.
        """
        # Paywall: kicking off processing requires an active subscription.
        require_subscription(request)
        job_id = body["job_id"]
        gcs_path = body["gcs_path"]
        video_key = body.get("video_key", "")
        skip_first_frame = bool(body.get("skip_first_frame", False))

        # Fast path: prepare_job already ran for this job_id and wrote the sidecar.
        # Return its result inline so the client gets a hit without another spawn.
        cached = await asyncio.to_thread(_read_prepare_metadata, job_id)
        if cached and not cached.get("error"):
            return JSONResponse({"status": "done", **cached})

        # Spawn the slow work in the background. .spawn() returns a FunctionCall
        # immediately without blocking the HTTP request.
        call = prepare_job.spawn(job_id, gcs_path)

        # If the client wants the (old) GPU first-frame extraction, also spawn it.
        # That call gets polled via its own call_id below; default path is the
        # client-extracted first frame so we usually skip this.
        first_frame_call_id = None
        if not skip_first_frame and video_key:
            ff_call = tracker_gpu.extract_first_frame.spawn(job_id, video_key)
            first_frame_call_id = ff_call.object_id

        return JSONResponse({
            "status": "pending",
            "call_id": call.object_id,
            "first_frame_call_id": first_frame_call_id,
            "job_id": job_id,
        })

    @api.post("/api/prepare-status")
    async def prepare_status(body: dict):
        """Poll a previously spawned prepare_job (and optionally the GPU first-frame
        call). Returns the metadata sidecar contents once prepare_job is complete."""
        from modal.functions import FunctionCall
        job_id = body["job_id"]
        call_id = body.get("call_id")
        first_frame_call_id = body.get("first_frame_call_id")

        # Check the prepare_job call first.
        if call_id:
            try:
                call = FunctionCall.from_id(call_id)
                # Non-blocking: timeout=0 returns immediately if not done.
                try:
                    call.get(timeout=0)
                    done = True
                except TimeoutError:
                    done = False
            except Exception as e:
                return JSONResponse({"status": "error", "error": str(e)})
            if not done:
                return JSONResponse({"status": "pending"})

        # prepare_job is done — read the sidecar metadata it wrote.
        meta = await asyncio.to_thread(_read_prepare_metadata, job_id)
        if meta.get("error"):
            return JSONResponse({"status": "error", "error": meta["error"]})

        # If the GPU first-frame extraction was spawned, fold its result in.
        if first_frame_call_id:
            try:
                ff_call = FunctionCall.from_id(first_frame_call_id)
                try:
                    ff_result = ff_call.get(timeout=0)
                    meta = {**meta, **(ff_result or {})}
                except TimeoutError:
                    # Frame call still running — return prepare result anyway so
                    # the client can proceed; client extracted its own frame.
                    pass
            except Exception:
                pass

        return JSONResponse({"status": "done", **meta})

    @api.get("/api/ball-stats/{video_key}")
    async def ball_stats(video_key: str):
        """Return the header block of ball_tracking.json for a given video.

        Useful for admin debugging — shows detection_rate_pct, InpaintNet
        gap-fill counts, stuck-ball rejections, etc. Skips the giant frames
        array. Public (no auth) because it exposes nothing sensitive: just
        aggregate quality numbers about a specific video.
        """
        def _fetch_header():
            import tempfile as _tf
            gcs_path = f"{OUTPUT_PREFIX}/{video_key}/ball_tracking.json"
            with _tf.NamedTemporaryFile(mode="rb", suffix=".json", delete=False) as tf:
                tmp = tf.name
            try:
                download_from_gcs(gcs_path, tmp)
                with open(tmp, "r") as f:
                    d = json.load(f)
                return {k: v for k, v in d.items() if k != "frames"}
            finally:
                try: os.unlink(tmp)
                except Exception: pass

        try:
            header = await asyncio.to_thread(_fetch_header)
            return JSONResponse(header)
        except Exception as e:
            return JSONResponse(
                {"error": f"Could not fetch ball_tracking.json: {e}"},
                status_code=404,
            )

    @api.get("/api/tracking-status/{video_key}")
    async def tracking_status(video_key: str):
        """Read the current tracking_status.json for a given video from GCS.

        Used by the client as a fallback when the WebSocket is closed (browser
        tab was closed and re-opened, WS timed out, etc.) — the client polls
        this endpoint every few seconds and forwards the state to the same UI
        update code that handles WS messages.

        Returns:
            The parsed status dict on success, or {"state": "unknown"} if
            there's no status file yet (spawn hasn't started writing, or the
            video was never tracked).
        """
        status = await asyncio.to_thread(read_tracking_status, video_key)
        if status is None:
            return JSONResponse({"state": "unknown", "video_key": video_key})
        return JSONResponse(status)

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
            # Optional trim end (multi-game segmentation). Defaults to total_frames
            # (full video). Clamped so client overshoot can't push past video length.
            trim_end_frame_raw = data.get("trim_end_frame")
            trim_end_frame = int(trim_end_frame_raw) if trim_end_frame_raw else total_frames
            trim_end_frame = min(trim_end_frame, total_frames)

            # Keep original total_frames for reference; trim bounds drive the loop.
            original_total_frames = total_frames
            trimmed_frames_count = trim_end_frame - trim_start_frame

            if trim_start_frame > 0 or trim_end_frame < original_total_frames:
                print(f"  ✓ Video trimmed: frames {trim_start_frame} → {trim_end_frame} "
                      f"({trim_start_frame / src_fps:.2f}s → {trim_end_frame / src_fps:.2f}s)")
                print(f"  Processing {trimmed_frames_count} frames")

            # ---- Wait for prepare_job outputs to be available on the volume ----
            # The client no longer blocks on prepare — it fires prepare and opens
            # the WebSocket immediately. For long uploads, prepare_job may still
            # be downloading the source and running ffprobe. We poll for its
            # output files (frame_timestamps.json + source.mp4) before starting
            # any tracking work. Ping the client periodically so it can render
            # a "preparing" state instead of a silent hang.
            timestamps_path = f"{DATA_DIR}/jobs/{job_id}/frame_timestamps.json"
            source_path = f"{DATA_DIR}/jobs/{job_id}/source.mp4"
            max_wait_sec = 25 * 60  # 25 min hard cap
            wait_start = time.time()
            last_ping = 0.0
            while True:
                try:
                    scratch_vol.reload()
                except Exception:
                    pass
                if os.path.exists(timestamps_path) and os.path.exists(source_path):
                    break
                elapsed = time.time() - wait_start
                if elapsed > max_wait_sec:
                    await ws.send_json({
                        "type": "error",
                        "message": "Video preparation timed out. Try re-uploading.",
                    })
                    if video_key:
                        _mark_video_status(video_key, "failed")
                    return
                # Ping the client every ~5s so a still-open tab shows progress.
                if elapsed - last_ping >= 5:
                    await ws.send_json({
                        "type": "status",
                        "message": f"Preparing video on server… ({int(elapsed)}s)",
                    })
                    last_ping = elapsed
                await asyncio.sleep(2)

            # Spawn ball tracking now that we know the trim range
            if video_key:
                try:
                    track_ball.spawn(
                        job_id, video_key,
                        src_fps, src_width, src_height,
                        trim_start_frame, trim_end_frame)
                    print(f"✓ Ball tracking spawned for {video_key} "
                          f"frames {trim_start_frame}→{trim_end_frame}")
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

            # ---- Kick off decoupled tracking + poll status for progress ----
            # Instead of running the segment loop inline (which dies if the WS
            # closes — Cloudflare, browser sleep, Modal ASGI auto-cancel — and
            # is bounded by the web endpoint's own timeout), we spawn
            # run_tracking_loop as an independent Modal function with its own
            # long timeout. The WS handler just polls tracking_status.json
            # from GCS every ~2s and forwards updates to the client. If the WS
            # dies mid-processing, the spawn keeps grinding — the tracking
            # data + final Firestore status write survive the disconnect.
            tracking_call = run_tracking_loop.spawn(
                job_id,
                video_key,
                src_width,
                src_height,
                src_fps,
                original_total_frames,
                trim_start_frame,
                trim_end_frame,
                seed_p1_proc,
                seed_p2_proc,
                yolo_box_p1_proc,
                yolo_box_p2_proc,
                params,
            )
            print(f"✓ Spawned run_tracking_loop for {video_key} "
                  f"(call_id={tracking_call.object_id})")

            # Poll the GCS status file every ~2s and forward the relevant bits
            # to the client via the existing wire message types. If any
            # ws.send_json fails (client gone), we stop trying — the spawn is
            # unaffected. We cap the poll loop just short of the web
            # endpoint's own hard timeout so we exit cleanly (letting the
            # spawn finish in the background) instead of being force-killed.
            last_seg_seen = 0
            client_alive = True
            poll_start = time.time()
            while True:
                if time.time() - poll_start > (WEB_HARD_TIMEOUT_SEC - 120):
                    if client_alive:
                        try:
                            await ws.send_json({
                                "type": "status",
                                "message": (
                                    "Tracking still running — you can close "
                                    "this tab; the card will update when it "
                                    "finishes."
                                ),
                            })
                        except Exception:
                            pass
                    break

                await asyncio.sleep(2)

                status_json = await asyncio.to_thread(
                    read_tracking_status, video_key
                )
                if status_json is None:
                    continue

                state = status_json.get("state")
                cur_seg = int(status_json.get("current_segment") or 0)
                est_total = int(status_json.get("est_total_segments") or 0)
                frames_done = int(status_json.get("current_frames_done") or 0)
                frames_total = int(status_json.get("current_frames_total") or 0)
                pct = status_json.get("pct") or 0
                msg = status_json.get("message", "")

                # Emit segment_start whenever we cross into a new segment so
                # the client's progress bar knows to reset for this seg.
                if cur_seg > last_seg_seen and state == "running":
                    if client_alive:
                        try:
                            await ws.send_json({
                                "type": "segment_start",
                                "segment": cur_seg,
                                "est_total": est_total,
                                "src_start": 0,   # not published by decoupled loop
                                "src_frames": frames_total,
                                "is_last": (cur_seg == est_total),
                            })
                        except Exception:
                            client_alive = False
                    last_seg_seen = cur_seg

                # Forward the running message so the client sees frame-by-frame
                # progress inside a segment too.
                if state in ("starting", "running") and client_alive:
                    try:
                        await ws.send_json({
                            "type": "progress",
                            "segment": cur_seg,
                            "frames_done": frames_done,
                            "frames_total": frames_total,
                            "pct": pct,
                            "message": msg,
                        })
                    except Exception:
                        client_alive = False

                if state == "complete":
                    urls = status_json.get("urls") or {}
                    complete_msg = {
                        "type": "complete",
                        "json_url": urls.get("json_url"),
                        "video_url": urls.get("video_url"),
                        "masks_url": urls.get("masks_url"),
                        "point_starts_url": urls.get("point_starts_url"),
                        "total_segments": urls.get("total_segments", cur_seg),
                        "total_frames_tracked": urls.get(
                            "total_frames_tracked", 0
                        ),
                    }
                    if client_alive:
                        try:
                            await ws.send_json(complete_msg)
                        except Exception:
                            pass
                    break

                if state == "error":
                    if client_alive:
                        try:
                            await ws.send_json({
                                "type": "error",
                                "message": status_json.get(
                                    "error", "Tracking failed"
                                ),
                            })
                        except Exception:
                            pass
                    break

        except asyncio.CancelledError:
            # Modal hard-cancels the WS input when *this* function's (shorter)
            # timeout is hit or the container shuts down. But run_tracking_loop
            # is a separate, longer-lived spawn that owns the terminal video
            # status — it keeps grinding after the relay dies and will write
            # complete/failed itself. So only mark the video failed here if the
            # tracking loop shows no sign of life; otherwise leave its status
            # alone (a false "failed" was flipping cards on healthy jobs).
            print(f"Processing cancelled (hard timeout / shutdown): {job_id}")
            try:
                if 'video_key' in locals() and video_key:
                    st = await asyncio.to_thread(read_tracking_status, video_key)
                    state = (st or {}).get("state")
                    if state not in ("starting", "running", "complete"):
                        _mark_video_status(video_key, "failed")
            except Exception:
                pass
            raise
        except WebSocketDisconnect:
            print(f"Client disconnected: {job_id}")
        except Exception as e:
            # A failure in the *relay* loop (e.g. a transient GCS read blip) is
            # not a failure of tracking — run_tracking_loop owns the terminal
            # status and is still running. Surface the error to the client but
            # do NOT touch the video status here, or we'd flip a healthy job's
            # card to "failed" while it's still processing.
            print(f"WS relay error for {job_id}: {e}")
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except:
                pass

    return api


# ====================================================================
# Test entrypoint — sanity-check compute_court_homography on a single frame
# ====================================================================

@app.function(
    image=gemini_image,
    volumes={DATA_DIR: scratch_vol},
    timeout=120,
)
def _seed_test_frame(job_id: str, image_b64: str) -> dict:
    """Drop a base64-encoded PNG into the Modal volume at the path the
    homography function expects, so test_homography can run end-to-end
    without a prior pipeline step."""
    import base64
    import os
    scratch_vol.reload()
    job_dir = f"{DATA_DIR}/jobs/{job_id}"
    os.makedirs(job_dir, exist_ok=True)
    out_path = f"{job_dir}/first_frame.png"
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(image_b64))
    scratch_vol.commit()
    return {"path": out_path, "bytes": os.path.getsize(out_path)}


@app.function(
    image=web_image,
    secrets=[gcs_secret, firebase_secret],
    timeout=900,
)
def backfill_thumbnails_fn(force: bool = False) -> dict:
    """Run the thumbnailPath backfill over all existing video docs.

    Invoke with:  modal run Modal_app.py::backfill_thumbnails_fn
    (add --force to overwrite docs that already have a thumbnailPath).
    """
    result = backfill_thumbnail_paths(force=force)
    print(f"[backfill] {result['updated_count']} updated, "
          f"{result['skipped_count']} skipped, "
          f"{result['missing_doc_count']} first_frame blobs had no video doc")
    return result


@app.function(
    image=web_image,
    secrets=[gcs_secret],
    timeout=900,
)
def refit_and_reprocess(video_key: str, also_reprocess: bool = True) -> dict:
    """Re-fit the homography from the already-tagged pixel landmarks in
    GCS, then optionally re-spawn convert_and_detect_points.

    Use after the court-geometry constants changed (e.g. service-box back
    moved from 3.89 → 7.04 m). The admin's landmark pixels don't change;
    only the real-world meter table they're fit against does.
    """
    import json as json_mod
    import tempfile

    gcs_dir = f"{OUTPUT_PREFIX}/{video_key}"
    local_h = f"/tmp/homography_{video_key}_in.json"
    try:
        download_from_gcs(f"{gcs_dir}/homography.json", local_h)
    except Exception as e:
        return {"status": "error", "message": f"No existing homography.json: {e}"}

    with open(local_h, "r") as f:
        h_data = json_mod.load(f)

    pixels = h_data.get("court_landmarks_pixels") or {}
    if not pixels:
        return {"status": "error", "message": "homography.json has no court_landmarks_pixels"}

    src_res = h_data.get("source_resolution") or {}
    src_w = int(src_res.get("width") or 1920)
    src_h = int(src_res.get("height") or 1080)

    print(f"  Refitting homography for {video_key} ({len(pixels)} landmarks, {src_w}x{src_h})")
    try:
        summary = build_and_upload_manual_homography(
            video_key=video_key,
            landmarks_pixels=pixels,
            src_width=src_w,
            src_height=src_h,
            tagged_by=h_data.get("tagged_by", "refit"),
        )
    except Exception as e:
        return {"status": "error", "message": f"Refit failed: {e}"}

    print(f"  ✓ Refit complete; floor reprojection error: "
          f"{summary.get('reprojection_error_meters')}m")

    if also_reprocess:
        print(f"  🚀 Spawning convert_and_detect_points for {video_key}")
        convert_and_detect_points.spawn(video_key)

    return {
        "status": "ok",
        "video_key": video_key,
        "floor_reprojection_error_m": summary.get("reprojection_error_meters"),
        "reprocess_spawned": bool(also_reprocess),
    }


@app.local_entrypoint()
def refit_video(video_key: str, no_reprocess: bool = False):
    """Refit homography (using current court-geometry constants) and
    re-trigger convert_and_detect_points for an existing video_key.

    Usage:
        modal run Modal_app.py::refit_video --video-key <key>
        modal run Modal_app.py::refit_video --video-key <key> --no-reprocess
    """
    import json
    result = refit_and_reprocess.remote(video_key, also_reprocess=not no_reprocess)
    print("\n--- result ---")
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def test_homography(image_path: str, video_key: str = "homography-smoke-test",
                     job_id: str = "test_homography_job",
                     width: int = 0, height: int = 0):
    """Run compute_court_homography against a local PNG/JPG file.

    Usage:
        modal run Modal_app.py::test_homography --image-path /path/to/first_frame.png

    Optional flags:
        --video-key   GCS key prefix for the uploaded result (default: homography-smoke-test)
        --job-id      Modal volume scratch dir (default: test_homography_job)
        --width/--height  Override source resolution (default: read from image)

    Reads the local file, base64-encodes it, seeds the Modal volume,
    then invokes compute_court_homography.remote(...). Prints the
    returned dict (status, surface count, floor reprojection error,
    upload URL).
    """
    import base64
    import json
    import sys
    from pathlib import Path

    p = Path(image_path).expanduser()
    if not p.exists():
        sys.exit(f"Image not found: {p}")
    raw = p.read_bytes()

    # If width/height weren't given, peek with Pillow.
    if width <= 0 or height <= 0:
        try:
            from PIL import Image as _PILImage
            import io
            with _PILImage.open(io.BytesIO(raw)) as im:
                width = width or im.width
                height = height or im.height
        except Exception as e:
            sys.exit(f"Couldn't read dimensions from {p}: {e}. "
                     "Pass --width and --height.")

    b64 = base64.b64encode(raw).decode()
    print(f"Seeding {p.name} ({len(raw):,} bytes, {width}x{height}) "
          f"to volume as job_id={job_id} ...")
    seed = _seed_test_frame.remote(job_id, b64)
    print(f"  wrote {seed['bytes']:,} bytes to {seed['path']}")

    print(f"Invoking compute_court_homography.remote({job_id!r}, {video_key!r}, "
          f"{width}, {height}) ...")
    result = compute_court_homography.remote(job_id, video_key, width, height)
    print("\n--- result ---")
    print(json.dumps(result, indent=2))


# ====================================================================
# Admin claim management — set `role: admin` on a Firebase user so they
# can use the /admin/* routes. Run once per admin account.
# ====================================================================

@app.function(
    image=web_image,
    secrets=[firebase_secret],
    timeout=60,
)
def _set_admin_claim_remote(email: str = "", uid: str = "") -> dict:
    _init_firebase_admin()
    from firebase_admin import auth as fb_auth
    if not (email or uid):
        return {"status": "error", "message": "Provide --email or --uid"}
    user = fb_auth.get_user(uid) if uid else fb_auth.get_user_by_email(email)
    existing = dict(user.custom_claims or {})
    existing["role"] = "admin"
    fb_auth.set_custom_user_claims(user.uid, existing)
    return {
        "status": "ok",
        "uid": user.uid,
        "email": user.email,
        "claims": existing,
        "note": "User must sign out and back in (or call user.getIdToken(true)) "
                "to refresh their ID token with the new claim.",
    }


@app.local_entrypoint()
def set_admin_claim(email: str = "", uid: str = ""):
    """Grant `role: admin` custom claim on a Firebase user.

    Usage:
        modal run Modal_app.py::set_admin_claim --email you@example.com
        modal run Modal_app.py::set_admin_claim --uid <firebase-uid>

    After this runs, the user must sign out + back in (or the client app
    must call user.getIdToken(true)) for the new claim to land in the
    Firebase ID token sent to the backend.
    """
    import json as _json
    if not email and not uid:
        raise SystemExit("Provide --email or --uid")
    result = _set_admin_claim_remote.remote(email=email, uid=uid)
    print(_json.dumps(result, indent=2))


@app.function(
    image=web_image,
    secrets=[stripe_secret],
    timeout=60,
)
def _create_discount_remote(code: str, dollars_off: int = 10, months: int = 0,
                            max_redemptions: int = 0) -> dict:
    """Create (or reuse) a coupon worth `dollars_off`/mo and a promotion code
    `code` that customers redeem at checkout. Runs inside Modal so it uses the
    live STRIPE_SECRET_KEY from the `stripe-secret` secret."""
    import stripe
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]

    if months and int(months) > 0:
        coupon_id = f"boastiq_{dollars_off}off_{months}mo"
        coupon_kwargs = {"duration": "repeating", "duration_in_months": int(months)}
    else:
        coupon_id = f"boastiq_{dollars_off}off_forever"
        coupon_kwargs = {"duration": "forever"}

    ver = STRIPE_PROMO_API_VERSION
    try:
        coupon = stripe.Coupon.retrieve(coupon_id, stripe_version=ver)
        coupon_note = f"reused existing coupon {coupon_id}"
    except Exception:
        coupon = stripe.Coupon.create(
            id=coupon_id, amount_off=int(dollars_off) * 100, currency="usd",
            name=f"${dollars_off} off/mo", stripe_version=ver, **coupon_kwargs)
        coupon_note = f"created coupon {coupon_id}"

    promo_kwargs = {"coupon": coupon.id, "code": code, "stripe_version": ver}
    if max_redemptions and int(max_redemptions) > 0:
        promo_kwargs["max_redemptions"] = int(max_redemptions)
    promo = stripe.PromotionCode.create(**promo_kwargs)
    return {
        "status": "ok",
        "coupon_id": coupon.id,
        "coupon_note": coupon_note,
        "promo_code": promo.code,
        "promo_id": promo.id,
        "max_redemptions": max_redemptions or None,
    }


@app.local_entrypoint()
def create_discount(code: str = "", dollars_off: int = 10, months: int = 0,
                    max_redemptions: int = 0):
    """Create a redeemable discount code in Stripe (uses the live key in Modal).

    Usage:
        modal run Modal_app.py::create_discount --code COACH25
        modal run Modal_app.py::create_discount --code SUMMER --dollars-off 10 --months 3
        modal run Modal_app.py::create_discount --code VIP --max-redemptions 50

    Defaults: $10 off, forever (so $35 → $25/mo permanently), unlimited uses.
    """
    import json as _json
    if not code:
        raise SystemExit("Provide --code (the string customers will type)")
    result = _create_discount_remote.remote(
        code=code, dollars_off=dollars_off, months=months,
        max_redemptions=max_redemptions)
    print(_json.dumps(result, indent=2))


@app.function(
    image=web_image,
    secrets=[stripe_secret],
    timeout=60,
)
def _create_annual_price_remote(cents: int = 30000) -> dict:
    """Create (or reuse) a recurring YEARLY price of `cents` on the same product
    as the monthly plan (STRIPE_PRICE_ID). Runs inside Modal so it uses the live
    STRIPE_SECRET_KEY. Idempotent: reuses an existing active yearly price of the
    same amount on that product if one already exists."""
    import stripe
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
    monthly_price_id = os.environ.get("STRIPE_PRICE_ID", "")
    if not stripe.api_key or not monthly_price_id:
        return {"status": "error", "detail": "stripe_not_configured"}

    monthly = stripe.Price.retrieve(monthly_price_id, expand=["product"])
    product_id = monthly.product if isinstance(monthly.product, str) else monthly.product.id

    # Reuse an existing matching yearly price to stay idempotent across re-runs.
    for p in stripe.Price.list(product=product_id, active=True, limit=100).auto_paging_iter():
        rec = getattr(p, "recurring", None)
        if rec and rec.interval == "year" and p.unit_amount == int(cents) and p.currency == "usd":
            return {"status": "ok", "price_id": p.id, "product_id": product_id,
                    "note": "reused existing yearly price", "unit_amount": p.unit_amount}

    price = stripe.Price.create(
        product=product_id,
        currency="usd",
        unit_amount=int(cents),
        recurring={"interval": "year"},
        nickname="BoastIQ Player — Annual",
    )
    return {"status": "ok", "price_id": price.id, "product_id": product_id,
            "note": "created new yearly price", "unit_amount": price.unit_amount}


@app.local_entrypoint()
def create_annual_price(cents: int = 30000):
    """Create the recurring yearly Stripe price for the annual plan (uses the live
    key in Modal). Prints the price_... id to wire into STRIPE_PRICE_ID_ANNUAL.

    Usage:
        modal run Modal_app.py::create_annual_price            # $300/year
        modal run Modal_app.py::create_annual_price --cents 30000
    """
    import json as _json
    result = _create_annual_price_remote.remote(cents=cents)
    print(_json.dumps(result, indent=2))