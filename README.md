# Hand Tracking System

Real-time 3D hand motion tracking for research and clinical assessment.
Captures quantitative kinematic metrics from hand movements using an Intel
RealSense depth camera and MediaPipe hand landmark detection.

---

## What it measures

| Metric | Description |
|---|---|
| **Distance** | Total 3-D path length (metres) traced by the hand centroid |
| **Rest ratio** | Fraction of recording time the hand was stationary (speed < 5 cm/s) |
| **Hover time** | Cumulative seconds the hand was close to the camera plane (< 0.75 m) while actively moving |
| **Average jerk** | Mean magnitude of the third time-derivative of filtered 3-D position — a proxy for movement smoothness |

All metrics are written to `tracking_logs.csv` at the end of each session and periodically auto-saved during long recordings.

---

## Dependencies

| Package | Tested version | Notes |
|---|---|---|
| Python | 3.10 – 3.13 | 3.11 recommended |
| `pyrealsense2` | ≥ 2.56 | Intel RealSense SDK Python bindings |
| `mediapipe` | ≥ 0.10.14 | Hand landmark model (Tasks API) |
| `opencv-python` | ≥ 4.9 | Frame rendering and video I/O |
| `numpy` | ≥ 1.24 | Numerical operations |

---

## Installation

```bash
# 1. Clone / download this repository
git clone https://github.com/LolMaple/HTS.git
cd hand-tracking-system

# 2. (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the MediaPipe hand landmark model
#    Place hand_landmarker.task in the same directory as hts.py
wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task     # Windows
curl -sO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task    # macOS / Linux
```

> **MediaPipe Hand Landmark Model** — or manually download from [Latest](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
> and place in same directory as hts.py

> **Intel RealSense drivers** — install the [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense/releases)
> before installing `pyrealsense2` via pip.  On Windows, plug in the camera
> *after* the SDK is installed.

---

## Usage

```bash
python hts.py
```

The application opens in a maximised window.  Three operating modes are available from the main menu:

### Live View
Streams directly from a connected RealSense camera.

1. Click **Start Live Camera**.
2. Adjust the participant ID slider.
3. Select **A** (AI-assisted) or **M** (Manual) scan type.
4. Click **START** to begin recording.
5. Click **STOP** to end the clip, then **End Session** to review metrics.

### Load Recording
Plays back a previously recorded `.bag` file through the same pipeline.

- The file is auto-paused after 1 second so you can confirm the source before tracking begins.
- Use the ▶/⏸ button to pause and resume playback at any point.

### Batch Process
Processes a queue of `.bag` files unattended.

1. Click **Batch Process** from the main menu.
2. Click **Add Files** to select one or more `.bag` files.
3. Assign participant IDs and scan types per file using the inline controls.
4. Click **Start** — files are processed sequentially at maximum throughput.
5. Results are saved automatically; click **Cancel** at any time to stop cleanly.

---

## Output files

All outputs are written relative to the script (or executable) directory:

```
tracking_logs.csv          # Session summary table (one row per clip)
data/
  clips/                   # Annotated AVI recordings (centre-cropped)
  heatmaps/                # 2-D spatial heatmaps (PNG)
  paths/                   # Hand path overlaid on first frame (PNG)
```

### CSV columns

`Timestamp`, `Session_ID`, `Status`, `Scan_Type`, `Clip_Number`,
`Recording_File`, `Duration_s`, `Distance_m`, `Rest_Ratio`,
`Avg_Jerk`, `Hover_Time_s`, `Path_Points`, `Process_Mode`

---

## Coloured glove

The tracking pipeline is tuned for a **bright cyan / blue nitrile glove** worn on the tracked hand.  The HSV masking step (constants `GLOVE_HSV_LOWER` / `GLOVE_HSV_UPPER` in `hts.py`) isolates the glove colour and applies a channel-mix transformation so MediaPipe — which was trained on bare skin — reliably detects the glove as a hand.

To use a different glove colour, adjust `GLOVE_HSV_LOWER` and `GLOVE_HSV_UPPER` at the top of `hts.py`.

---

## Building a standalone executable

A `hts.spec` file is included that handles all bundling correctly —
mediapipe runtime data, the model file, and the optional app icon.

**Before building**, place the following in the same directory as `hts.spec`:

| File | Required? | Notes |
|---|---|---|
| `hand_landmarker.task` | **Yes** | Download link in the Installation section above |
| `app_icon.ico` | No | Embedded as the window/taskbar icon if present; app runs fine without it |

```bash
# Install PyInstaller (once)
pip install pyinstaller

# Build — run from the directory containing hts.spec
pyinstaller hts.spec
```

The resulting `dist\HTS.exe` is self-contained — no Python installation
required on the target machine.  Output files (CSV logs, video clips, heatmaps) are
written to the folder containing the `.exe` at runtime.

> **Debugging tip:** If the `.exe` crashes silently on launch, open `hts.spec`,
> change `console=False` to `console=True`, and rebuild.  A console window will appear
> and print the error.

---

## Algorithm references

- **One Euro Filter** — Casiez, G., Roussel, N., & Vogel, D. (2012). 1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems. *Proc. CHI 2012*, 2527–2530. https://doi.org/10.1145/2207676.2208639

- **MediaPipe Hands** — Zhang, F., et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. *arXiv:2006.10214*. https://arxiv.org/abs/2006.10214

- **Intel RealSense SDK** — https://github.com/IntelRealSense/librealsense

---

## License

MIT — see `LICENSE` file for full text.
