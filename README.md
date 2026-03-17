# BoastIQ Player Tracker — Web Application

Track 2 players through match videos using SAM2.1 (Hiera Small) on A100-80GB GPUs with robust segment handoff.

## Architecture

```
Browser  ←WebSocket→  FastAPI on Modal (CPU)  ←.remote()→  SAM2 on Modal (A100-80GB)
   ↕                                                              ↕
Firebase Storage                                           Modal Volume (scratch)
(input video + output files)
```

### Robust Handoff System

Long videos are processed in segments (~42s each at default settings). Between segments, the handoff system ensures continuous tracking:

1. **Scored selection** — Evaluates the last 600 frames for separation, mask compactness, area similarity, and recency. Picks the best frame, not just the first acceptable one.
2. **Box + multi-point prompts** — Uses the previous segment's mask to build a bounding box prompt (SAM2's most reliable prompt type) plus 5 interior points.
3. **Validation + retry** — Checks the new segment's initial masks for quality, centroid drift, and IoU against the expected region. Retries on up to 5 alternative frames across the 6.25s overlap window.
4. **Points-only fallback** — If box prompts fail, drops back to multi-point only.

## Setup

### 1. Prerequisites

- [Modal account](https://modal.com) with CLI installed
- [Firebase project](https://console.firebase.google.com) with Storage enabled
- HuggingFace account (no gated access needed for SAM2)

```bash
pip install modal
modal setup
```

### 2. Firebase Service Account

1. Go to Firebase Console → Project Settings → Service Accounts
2. Click "Generate new private key" → downloads a JSON file
3. Also note your **Storage bucket name** (e.g., `your-project.firebasestorage.app`)

### 3. Configure Modal Secrets

Verify with `modal secret list`:

```bash
# HuggingFace token
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here

# Google Cloud service account
modal secret create googlecloud-secret \
  SERVICE_ACCOUNT_JSON='{ paste full JSON here }'
```

> The app uses the `boastiq` GCS bucket. Videos upload to `web-uploaded-videos/`,
> outputs go to `web-tracking-outputs/`.

### 4. Configure Firebase in Frontend

Edit `index.html` and replace the Firebase config block (around line 245):

```js
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "boastiq-dev.firebaseapp.com",
  projectId: "boastiq-dev",
  storageBucket: "boastiq.firebasestorage.app",
  messagingSenderId: "000000000000",
  appId: "YOUR_APP_ID"
};
```

Get `apiKey` and `appId` from Firebase Console → Project Settings → General → Your apps.

### 5. Firebase Storage Rules

For development (no auth):

```
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /{allPaths=**} {
      allow read, write: if true;
    }
  }
}
```

> ⚠️ Restrict these rules before going to production.

### 6. Deploy

```bash
cd boastiq-tracker
modal deploy modal_app.py
```

Modal will print the URL, e.g.:
```
https://your-workspace--boastiq-tracker-web.modal.run
```

### 7. Development Mode

```bash
modal serve modal_app.py
```

Hot-reload — changes to `modal_app.py` or `index.html` are picked up automatically.

## Usage

1. **Upload** — Drop your match video (MP4, 720p/1080p, 30/60fps)
2. **Select players** — Click Player 1 (red), right-click Player 2 (blue)
3. **Track** — SAM2 processes the video in segments with automatic handoff
4. **Re-prompt** — If tracking is lost, you'll see the frame and can click new positions
5. **Download** — Get the masked video and JSON with foot positions from Firebase

## Output Files

### Tracked Video (`tracked_output.mp4`)
- Source resolution with colored mask overlays (P1 = red, P2 = blue)
- Subsampled to 24fps from source
- H.264 encoded

### Tracking JSON (`player_tracking.json`)
```json
{
  "source_resolution": {"width": 1280, "height": 720},
  "source_fps": 30.0,
  "total_frames_tracked": 12000,
  "model": "SAM2.1 Hiera Small",
  "prompt_type": "point_prompts + box_handoff",
  "coordinate_description": "Midpoint of lower edge of bounding box, source resolution pixels",
  "frames": [
    {
      "frame_number": 0,
      "timestamp_sec": 0.0,
      "player_1": {"x": 510, "y": 485},
      "player_2": {"x": 725, "y": 492}
    }
  ]
}
```

## Configuration

Default processing parameters (in `modal_app.py` → `DEFAULT_CONFIG`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frames_per_segment` | 1000 | SAM2 frames per GPU call (~42s of video at 24fps) |
| `overlap_frames` | 150 | Overlap between segments (~6.25s at 24fps) |
| `target_fps` | 24 | Processing frame rate |
| `target_height` | 480 | Processing resolution height |
| `min_player_separation` | 120 | Minimum handoff separation (source px) |
| `handoff_search_zone` | 600 | Frames to evaluate for best handoff |
| `handoff_prompt_retries` | 5 | Prompt retry attempts on different frames |
| `handoff_multi_points` | 5 | Interior mask points per prompt |

## Costs

- **GPU**: A100-80GB at ~$3.50/hr on Modal. An 8-min video processes in ~10-15 min.
- **Firebase**: Storage costs for input/output videos. ~$0.026/GB/month.
- **Bandwidth**: Firebase egress for downloads.

## File Structure

```
boastiq-tracker/
├── modal_app.py     # Modal app (GPU + CPU functions + FastAPI server)
├── index.html       # Frontend (served by Modal)
└── README.md        # This file
```
