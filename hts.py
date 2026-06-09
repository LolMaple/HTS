# SPDX-License-Identifier: MIT
#
# MIT License
# Copyright (c) 2024 Natalie H. and Shangzhe L.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""
Hand Tracking System (HTS) — hts.py
=====================================
Real-time 3D hand motion tracking using an Intel RealSense depth camera and
MediaPipe hand landmark detection.  Designed for research applications that
require quantitative measures of hand movement during a task.

What it measures
----------------
- Total 3D path length (metres) traced by the hand centroid over the session
- Rest ratio: fraction of recording time the hand was stationary (speed < 5 cm/s)
- Hover time: cumulative time the hand was close to the imaging plane (depth
  < HOVER_DEPTH_THRESHOLD) while actively moving
- Average jerk: mean magnitude of the third time-derivative of filtered 3D
  position — a proxy for movement smoothness

Application modes
-----------------
1. Live View   — streams directly from a connected RealSense camera
2. Load Recording — plays back a pre-recorded .bag file with the same pipeline
3. Batch Process  — processes a queue of .bag files unattended

Output files (written to the directory containing the script / executable)
---------------------------------------------------------------------------
- tracking_logs.csv          session-level summary (one row per clip)
- data/clips/*.avi           annotated video recordings (centre-cropped)
- data/heatmaps/*.png        2-D spatial heatmap of hand position over time
- data/paths/*.png           first-frame overlay showing the full hand path

Dependencies
------------
See requirements.txt.  Key packages:
    pyrealsense2 >= 2.56   Intel RealSense SDK Python bindings
    mediapipe >= 0.10.14      Hand landmark model (.task file, VIDEO mode)
    opencv-python >= 4.9   Frame capture, rendering, video I/O
    numpy >= 1.24          Numerical operations

Usage
-----
    python hts.py

Place hand_landmarker.task in the same directory as this script (or the
compiled executable).  The application opens a maximised OpenCV window.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import os
import time
import csv
import sys
import math
import tkinter as tk
from tkinter import filedialog
import threading
import re
import queue
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from enum import Enum


# ---------------------------------------------------------------------------
# Resource / output path helpers
# ---------------------------------------------------------------------------

def get_resource_path() -> str:
    """Return the directory that contains bundled resources (model file etc.).

    When running as a PyInstaller-frozen executable the runtime unpacks
    resources to a temporary ``_MEIPASS`` folder.  When running as a plain
    script, the directory containing this file is used, with a few fallback
    locations checked in order.
    """
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS

    candidates = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(sys.executable),
        os.getcwd(),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "hand_landmarker.task")):
            return c
    return os.path.dirname(os.path.abspath(__file__))


def get_output_path() -> str:
    """Return the directory where output files (logs, clips, heatmaps) are written.

    For a frozen executable, outputs are placed next to the .exe so they are
    easily accessible to the user.  For a plain script, the script directory
    is used.
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


RESOURCE_PATH = get_resource_path()
OUTPUT_PATH = get_output_path()


# ---------------------------------------------------------------------------
# File dialog helpers
# ---------------------------------------------------------------------------

def open_file_dialog(title: str = "Select File", filetypes=None) -> str:
    """Show a native file-open dialog and return the chosen path (or '' if cancelled).

    Creates and immediately destroys a hidden Tk root to avoid leaving a
    zombie window that can hang the process.

    Args:
        title:     Dialog window title.
        filetypes: List of (label, pattern) tuples, e.g. [("ROS Bag", "*.bag")].

    Returns:
        Absolute path of the selected file, or an empty string.
    """
    if filetypes is None:
        filetypes = [("All Files", "*.*")]
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def open_files_dialog(title: str = "Select Files", filetypes=None):
    """Show a native multi-file-open dialog and return the chosen paths.

    Args:
        title:     Dialog window title.
        filetypes: List of (label, pattern) tuples.

    Returns:
        Tuple of absolute paths, or an empty tuple.
    """
    if filetypes is None:
        filetypes = [("All Files", "*.*")]
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    paths = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    root.destroy()
    return paths


# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

# --- Application identity ---
APP_TITLE = "HAND TRACKING SYSTEM"
WINDOW_NAME = "Hand Tracking System"

# --- Session ID ---
# Prefix prepended to the zero-padded participant number, e.g. "P-03".
# Change this to match your study's naming convention.
SESSION_ID_PREFIX = "P"

# --- Camera / stream ---
STREAM_WIDTH = 848          # RealSense colour and depth stream width  (pixels)
STREAM_HEIGHT = 480         # RealSense colour and depth stream height (pixels)
STREAM_FPS = 30             # Target frame rate (frames per second)
FRAME_MS = 1000 // STREAM_FPS  # Nominal frame duration in milliseconds (~33 ms)

# --- Spatial processing ---
# Fraction of frame width masked on each side to exclude monitor reflections
# and out-of-field-of-view artefacts.
SIDE_MARGIN_RATIO = 0.25

# Fraction of frame width blanked on each side during glove-colour masking
# (slightly wider than SIDE_MARGIN_RATIO to suppress peripheral noise).
GLOVE_ROI_MARGIN = 0.30

# --- Depth thresholds ---
MAX_DEPTH_M = 3.0           # Ignore depth readings beyond this distance (metres)
HOVER_DEPTH_THRESHOLD = 0.75  # Depth below which a moving hand is counted as "hovering" (metres)

# --- Motion thresholds ---
IDLE_SPEED_THRESHOLD = 0.05   # 3D speed (m/s) below which the hand is considered idle
MAX_DIST_JUMP_M = 0.30        # Maximum plausible frame-to-frame displacement; larger jumps
                               # are assumed to be tracking artefacts and are discarded

# --- One Euro Filter parameters (see OneEuroFilter class docstring) ---
FILTER_MIN_CUTOFF = 0.01    # Minimum cutoff frequency (Hz); controls steady-state smoothing
FILTER_BETA = 0.10          # Speed coefficient; higher = less lag during fast movement

# --- Participant ID ---
MAX_PARTICIPANT_ID = 35     # Highest participant number supported by the ID slider

# --- Recording ---
VIDEO_BUFFER_FRAMES = 120   # Async video-writer queue depth (~4 s at 30 fps)
AUTOSAVE_INTERVAL_S = 60    # Session log is flushed to CSV at this interval (seconds)

# --- Visualisation ---
TRAIL_LENGTH = 50           # Number of recent positions drawn as a fading path trail

# --- MediaPipe detection thresholds ---
DETECTION_CONFIDENCE = 0.3  # Minimum confidence for hand detection, presence, and tracking

# --- Hand skeleton connectivity (MediaPipe landmark indices) ---
# Landmark numbering follows the MediaPipe hand model:
# 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),    # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),    # index finger
    (0, 9),  (9, 10), (10, 11),(11, 12),   # middle finger
    (0, 13), (13, 14),(14, 15),(15, 16),   # ring finger
    (0, 17), (17, 18),(18, 19),(19, 20),   # pinky
    (5, 9),  (9, 13), (13, 17),            # palm transverse connections
]

# --- HSV glove colour range ---
# Tuned for a bright cyan/blue nitrile or latex glove under typical lab lighting.
# Hue range 70–150 covers cyan (≈90) through blue (≈120).
# High saturation minimum (145) ensures only vivid, saturated colours are selected,
# rejecting skin tones, white backgrounds, and grey clothing.
GLOVE_HSV_LOWER = np.array([70,  145,  60])
GLOVE_HSV_UPPER = np.array([150, 255, 255])

# ---------------------------------------------------------------------------
# UI theme — dark professional
# ---------------------------------------------------------------------------
# All colours are BGR tuples (OpenCV convention).
# Primary accent is teal (≈ RGB 50, 185, 200).

UI_BG           = (40,  38,  35)   # Background: very dark charcoal
UI_PANEL        = (58,  55,  52)   # Panel / card background
UI_ACCENT       = (200, 185,  50)  # Teal accent (BGR of RGB 50,185,200)
UI_BORDER       = (110, 100,  65)  # Muted teal border
UI_TEXT         = (230, 230, 235)  # Primary text (near-white)
UI_TEXT_DIM     = (155, 155, 163)  # Secondary / dimmed text
UI_BTN_IDLE     = (65,  62,  58)   # Button at rest
UI_BTN_HOVER    = (115, 108,  72)  # Button on hover
UI_BTN_ACTIVE   = (185, 175,  55)  # Button active / pressed
UI_BTN_SHADOW   = (22,  21,  19)   # Button drop shadow
UI_SUCCESS      = (100, 200, 100)  # Status: done / success (BGR)
UI_WARNING      = (80,  160, 220)  # Status: partial / warning (BGR)
UI_PROGRESS     = (160, 150,  45)  # Progress bar fill (BGR)


# ---------------------------------------------------------------------------
# One Euro Filter
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """Adaptive low-pass filter for reducing jitter in noisy real-time signals.

    Reference
    ---------
    Casiez, G., Roussel, N., & Vogel, D. (2012). 1€ Filter: A Simple Speed-based
    Low-pass Filter for Noisy Input in Interactive Systems.  In *Proceedings of
    CHI 2012* (pp. 2527–2530). ACM. https://doi.org/10.1145/2207676.2208639

    The filter adapts its cutoff frequency based on the signal's speed:
    - At low speed the cutoff is low  → heavy smoothing, minimal jitter.
    - At high speed the cutoff is high → reduced lag, responsive to fast motion.

    Parameters
    ----------
    t0 : float
        Initial timestamp (seconds).
    min_cutoff : float
        Minimum cutoff frequency (Hz).  Lower values smooth more aggressively
        but increase lag at low speed.  Typical range: 0.01–1.0.
    beta : float
        Speed coefficient.  Higher values raise the cutoff faster when the
        signal is moving quickly, reducing lag.  Typical range: 0.0–1.0.
    d_cutoff : float
        Cutoff frequency (Hz) for the derivative estimate.  Rarely needs tuning.
    """

    def __init__(self, t0: float, min_cutoff: float = 1.0,
                 beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = t0

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        """Compute the exponential smoothing factor α for a given cutoff frequency."""
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def _exp_smooth(self, a: float, x: float, x_prev: float) -> float:
        """Apply one step of exponential smoothing: x_hat = a*x + (1-a)*x_prev."""
        return a * x + (1 - a) * x_prev

    def filter(self, t: float, x: float) -> float:
        """Filter a new sample.

        Args:
            t: Current timestamp (seconds).
            x: Raw signal value at time *t*.

        Returns:
            Filtered signal value.
        """
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

        # Estimate derivative and smooth it
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exp_smooth(a_d, dx, self.dx_prev)

        # Adaptive cutoff: rises with signal speed to reduce lag
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exp_smooth(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


# ---------------------------------------------------------------------------
# Threaded video writer
# ---------------------------------------------------------------------------

class ThreadedVideoWriter:
    """Writes video frames from a background thread to prevent UI stutter.

    OpenCV's ``VideoWriter.write()`` can block for several milliseconds while
    compressing each frame, which causes visible frame-rate drops when called
    on the main thread.  This class offloads the write to a daemon thread and
    buffers up to ``VIDEO_BUFFER_FRAMES`` frames in a queue.  If the queue
    fills (i.e. disk I/O cannot keep up), frames are dropped with a warning
    rather than blocking the UI.

    Args:
        filename:   Output video file path.
        fourcc:     FourCC codec code (e.g. ``cv2.VideoWriter_fourcc(*'MJPG')``).
        fps:        Frames per second for the output file.
        frame_size: (width, height) tuple of each frame.
    """

    def __init__(self, filename: str, fourcc: int, fps: float, frame_size: tuple):
        self._writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self._queue = queue.Queue(maxsize=VIDEO_BUFFER_FRAMES)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def write(self, frame: np.ndarray) -> None:
        """Queue *frame* for writing.  Non-blocking; drops the frame if the buffer is full."""
        if not self._stop_event.is_set():
            try:
                # Copy is critical: caller may mutate the frame array after this call
                self._queue.put(frame.copy(), timeout=0.005)
            except queue.Full:
                print("[WARNING] Video writer buffer full — dropping frame to preserve UI FPS")

    def _write_loop(self) -> None:
        """Background loop: dequeue frames and write them to disk."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.1)
                self._writer.write(frame)
                self._queue.task_done()
            except queue.Empty:
                continue

    def release(self) -> None:
        """Signal the writer thread to stop, drain the queue, then release the file."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        self._writer.release()


# ---------------------------------------------------------------------------
# Application state machine
# ---------------------------------------------------------------------------

class AppState(Enum):
    """Top-level states of the application's state machine.

    Transitions::

        CONNECTING ──► TRACKING ──► FINISHED ──► CONNECTING
             │                                        ▲
             └──────────► BATCH_PROCESSING ───────────┘
    """
    CONNECTING = 1        # Main menu; waiting for user to choose a source
    TRACKING = 2          # Live or bag-file playback with active UI
    FINISHED = 3          # Session summary screen shown after a recording
    BATCH_PROCESSING = 4  # Unattended queue processing


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class App:
    """Hand tracking application.

    Manages the RealSense pipeline, MediaPipe hand landmark detection, UI
    rendering, metric computation, and file output.  All UI is rendered into
    an OpenCV named window using plain NumPy/cv2 drawing primitives (no GUI
    framework dependency).

    Typical lifecycle::

        app = App()
        app.run()          # blocks until the window is closed
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, model_path: str = "hand_landmarker.task"):
        """Initialise the application.

        Loads the MediaPipe hand landmark model, prepares all state variables,
        and (optionally) attempts to use the GPU delegate for faster inference.

        Args:
            model_path: Path to the MediaPipe ``hand_landmarker.task`` model
                        file.  If not found at this path, ``RESOURCE_PATH`` is
                        searched before giving up.
        """
        self.state = AppState.CONNECTING

        # RealSense pipeline objects
        self.pipeline = None
        self.config = None
        self.align = None
        self.color_intrinsics = None
        self.current_bag_file = None
        self.is_rgb_input = False  # Some .bag files store colour as RGB8 not BGR8

        # --- Tracking state ---
        self.is_tracking = False
        self.total_distance = 0.0       # Cumulative 3D path length (metres)
        self.prev_position_3d = None    # Last filtered 3D position (numpy array)
        self.path_points = deque()      # 2-D pixel positions for path visualisation
        self.elapsed_time = 0.0
        self.last_frame_time = 0.0
        self.start_time = 0.0
        self.frame_count = 0

        # --- One Euro Filter instances (one per axis) ---
        # Initialised properly in reset_session_state(); None until first use.
        self.filter_x = None
        self.filter_y = None
        self.filter_z = None

        # --- Advanced motion metrics ---
        self.idle_time = 0.0            # Cumulative time hand was stationary
        self.hover_time = 0.0           # Cumulative time hand hovered close to camera
        self.cumulative_jerk = 0.0      # Sum of per-frame jerk magnitudes
        self.jerk_count = 0             # Number of jerk samples (for averaging)
        self.prev_velocity = None       # Previous velocity vector (m/s)
        self.prev_accel = None          # Previous acceleration vector (m/s²)
        self.prev_metric_time = None    # Wall-clock time of previous metric sample

        # --- MediaPipe hand landmarker ---
        resolved_model = os.path.join(RESOURCE_PATH, "hand_landmarker.task")
        print(f"[INFO] RESOURCE_PATH: {RESOURCE_PATH}")
        print(f"[INFO] Model path: {resolved_model}, exists: {os.path.exists(resolved_model)}")
        if not os.path.exists(resolved_model):
            resolved_model = "hand_landmarker.task"  # Fallback to CWD
        if not os.path.exists(resolved_model):
            print("[ERROR] hand_landmarker.task not found.  "
                  "Place it next to the script or executable.")
            sys.exit(1)

        self.model_asset_path = resolved_model

        self.landmarker = self._create_hand_landmarker()
        self.timestamp_ms = 0

        # --- UI state ---
        self.mouse_pos = (0, 0)
        self.clicked = False
        self.mouse_down = False
        self.window_name = WINDOW_NAME
        self.running = True

        # --- Participant / session controls ---
        self.id_slider_val = 0           # Integer 0–MAX_PARTICIPANT_ID
        self.user_id_input = "00"        # Zero-padded string kept in sync with slider
        self.live_scan_type = "AI"       # "AI" or "MAN" — affects output file naming
        self.show_id_error = False

        # --- Recording ---
        self.video_writer = None
        self.recording_filename = None
        self.session_id = None
        self.clip_number = 1
        self.is_saving = False

        # --- Reliability ---
        self.last_autosave_time = 0.0
        self.connection_attempted = False

        # --- Heatmap / path data ---
        self.all_hand_coords = []  # List of (x, y) pixel tuples for the current session
        self.first_frame = None    # First colour frame captured; used as heatmap background

        # --- Batch processing ---
        self.batch_queue = []          # List of dicts: {path, participant_id, scan_type}
        self.batch_progress = 0
        self.batch_status = "Idle"     # "Idle" | "Running" | "Cancelled"
        self.batch_cancel_requested = False
        self.batch_file_results = {}   # index → "DONE" | "PARTIAL"
        self.batch_file_stats = {}     # index → {distance, hover, jerk, duration}
        self.batch_scroll_offset = 0
        self.batch_default_participant_id = "00"
        self.batch_default_scan_type = "AI"
        self.batch_frame_count = 0

        # --- Async pipeline loading ---
        self.loading_pipeline = False
        self.loading_result = None
        self._pending_bag_file = None

        # --- Bag-file playback control ---
        self.bag_playback_paused = False
        self.bag_auto_pause_done = False
        self.bag_playback_start_time = None

    def _create_hand_landmarker(self):
        """Create a MediaPipe hand landmarker using the CPU delegate."""
        base_options = python.BaseOptions(model_asset_path=self.model_asset_path)
        try:
            base_options.delegate = python.BaseOptions.Delegate.CPU
        except Exception:
            pass

        print("[INFO] Using CPU delegate for inference.")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=DETECTION_CONFIDENCE,
            min_hand_presence_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=DETECTION_CONFIDENCE,
        )
        return vision.HandLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def draw_button(self, img: np.ndarray, rect: tuple, text: str,
                    is_hovered: bool, is_active: bool = False) -> None:
        """Draw a labelled button with the application's dark theme.

        Args:
            img:        Frame to draw on (modified in place).
            rect:       (x, y, w, h) bounding rectangle of the button.
            text:       Button label.
            is_hovered: Whether the mouse cursor is currently over the button.
            is_active:  Whether the button is in an active / pressed state
                        (e.g. "STOP" while recording).
        """
        x, y, w, h = rect

        if is_active:
            bg_color = UI_BTN_ACTIVE
        elif is_hovered:
            bg_color = UI_BTN_HOVER
        else:
            bg_color = UI_BTN_IDLE

        # Drop shadow
        sd = 2
        cv2.rectangle(img, (x + sd, y + sd), (x + w + sd, y + h + sd), UI_BTN_SHADOW, -1)
        # Body
        cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
        # Border
        cv2.rectangle(img, (x, y), (x + w, y + h), UI_BORDER, 2)

        # Centred label
        font = cv2.FONT_HERSHEY_PLAIN
        fs, thick = 1.5, 2
        (tw, th), _ = cv2.getTextSize(text, font, fs, thick)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        cv2.putText(img, text, (tx + 1, ty + 1), font, fs, UI_BTN_SHADOW, thick)
        cv2.putText(img, text, (tx, ty), font, fs, UI_TEXT, thick)

    def is_inside_button(self, pos: tuple, btn_pos: tuple, btn_size: tuple) -> bool:
        """Return True if *pos* (x, y) falls within the button rectangle.

        Args:
            pos:      (x, y) cursor position.
            btn_pos:  (x, y) top-left corner of the button.
            btn_size: (w, h) dimensions of the button.
        """
        px, py = pos
        bx, by = btn_pos
        bw, bh = btn_size
        return bx <= px <= bx + bw and by <= py <= by + bh

    def draw_landmarks(self, rgb_image: np.ndarray, detection_result) -> np.ndarray:
        """Overlay hand skeleton landmarks on *rgb_image*.

        Draws bones as teal lines and joints as small white circles with a
        teal border.  If tracking is active but no hand is detected, a
        "SEARCHING…" status message is rendered instead.

        Args:
            rgb_image:        The colour frame to annotate (not modified in place;
                              a copy is returned).
            detection_result: MediaPipe ``HandLandmarkerResult`` from the most
                              recent ``detect_for_video`` call.

        Returns:
            Annotated copy of *rgb_image*.
        """
        annotated = np.copy(rgb_image)
        landmarks_list = detection_result.hand_landmarks

        if not landmarks_list:
            if self.is_tracking:
                h, w = annotated.shape[:2]
                cv2.putText(annotated, "SEARCHING...",
                            (w // 2 - 80, h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 1, UI_TEXT_DIM, 2)
            return annotated

        height, width = annotated.shape[:2]
        for hand_landmarks in landmarks_list:
            # Draw bones
            for start_idx, end_idx in HAND_CONNECTIONS:
                s = hand_landmarks[start_idx]
                e = hand_landmarks[end_idx]
                p1 = (int(s.x * width), int(s.y * height))
                p2 = (int(e.x * width), int(e.y * height))
                cv2.line(annotated, p1, p2, UI_ACCENT, 3)

            # Draw joint dots
            for lm in hand_landmarks:
                cx = int(lm.x * width)
                cy = int(lm.y * height)
                cv2.circle(annotated, (cx, cy), 5, UI_ACCENT, -1)
                cv2.circle(annotated, (cx, cy), 3, UI_TEXT, -1)

        return annotated

    def _draw_title(self, img: np.ndarray, text: str, y: int,
                    scale: float = 2.5, thickness: int = 3) -> None:
        """Draw a horizontally centred title in the accent colour with a drop shadow.

        Args:
            img:       Frame to draw on (in place).
            text:      Title string.
            y:         Baseline y-coordinate.
            scale:     Font scale.
            thickness: Stroke thickness.
        """
        font = cv2.FONT_HERSHEY_PLAIN
        w = img.shape[1]
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        tx = (w - tw) // 2
        cv2.putText(img, text, (tx + 2, y + 2), font, scale, UI_BTN_SHADOW, thickness)
        cv2.putText(img, text, (tx, y), font, scale, UI_ACCENT, thickness)

    def _draw_subtitle(self, img: np.ndarray, text: str, y: int) -> None:
        """Draw a horizontally centred subtitle in the dimmed text colour."""
        font = cv2.FONT_HERSHEY_PLAIN
        w = img.shape[1]
        (tw, _), _ = cv2.getTextSize(text, font, 1.1, 1)
        tx = (w - tw) // 2
        cv2.putText(img, text, (tx, y), font, 1.1, UI_TEXT_DIM, 1)

    # ------------------------------------------------------------------
    # Pipeline initialisation
    # ------------------------------------------------------------------

    def init_pipeline(self, bag_file: str = None) -> bool:
        """Initialise (or reinitialise) the RealSense pipeline.

        Performs a complete state reset before opening a new source so that
        metrics and recording state from a previous session do not bleed through.
        For bag files, real-time playback is enabled for interactive viewing and
        disabled (fast-forward) for batch processing.

        Args:
            bag_file: Path to a .bag recording, or None to open the live camera.

        Returns:
            True on success, False if the device/file could not be opened.
        """
        self.reset_session_state()

        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        # Reset MediaPipe timestamp counter — required for each new video source
        self.timestamp_ms = 0
        try:
            self.landmarker = self._create_hand_landmarker()
        except Exception as exc:
            print(f"[WARN] Could not recreate landmarker: {exc}")

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if bag_file:
            if not os.path.exists(bag_file):
                print(f"[ERROR] Bag file not found: {bag_file}")
                return False
            try:
                rs.config.enable_device_from_file(self.config, bag_file,
                                                   repeat_playback=False)
                self.current_bag_file = bag_file
                print(f"[INFO] Loading bag: {bag_file}")
            except Exception as exc:
                print(f"[ERROR] Failed to load bag file: {exc}")
                return False
        else:
            self.current_bag_file = None
            ctx = rs.context()
            if len(ctx.query_devices()) == 0:
                print("[ERROR] No Intel RealSense device detected.")
                return False
            print("[INFO] Connecting to live RealSense camera...")
            self.config.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT,
                                      rs.format.z16, STREAM_FPS)
            self.config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT,
                                      rs.format.bgr8, STREAM_FPS)

        try:
            profile = self.pipeline.start(self.config)
            color_profile = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            self.align = rs.align(rs.stream.color)

            # Detect whether the colour stream uses RGB8 ordering (common in bag files)
            self.is_rgb_input = (color_profile.format() == rs.format.rgb8)
            if self.is_rgb_input:
                print("[INFO] Colour stream is RGB8; will convert to BGR for display.")

            if bag_file:
                playback = profile.get_device().as_playback()
                if self.state == AppState.BATCH_PROCESSING:
                    # Disable real-time to process as fast as possible
                    playback.set_real_time(False)
                    print("[BATCH] Real-time playback disabled for maximum throughput.")
                else:
                    # Match recorded frame rate for interactive review
                    playback.set_real_time(True)

                self.bag_playback_paused = False
                self.bag_auto_pause_done = False
                self.bag_playback_start_time = None
            else:
                self.bag_playback_paused = False

            return True

        except RuntimeError as exc:
            print(f"[ERROR] Pipeline start failed: {exc}")
            self.pipeline = None
            return False

    def _init_pipeline_async(self) -> None:
        """Worker called from a background thread to keep the UI responsive during init."""
        bag_file = getattr(self, '_pending_bag_file', None)
        result = self.init_pipeline(bag_file=bag_file)
        self.loading_result = result

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """OpenCV mouse event callback — updates cursor position and click state.

        Also handles mousewheel scrolling for the batch file list.
        """
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = True
            self.mouse_down = True
            self.mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.state == AppState.BATCH_PROCESSING and len(self.batch_queue) > 5:
                if flags > 0:
                    self.batch_scroll_offset = max(0, self.batch_scroll_offset - 1)
                else:
                    max_scroll = max(0, len(self.batch_queue) - 5)
                    self.batch_scroll_offset = min(max_scroll, self.batch_scroll_offset + 1)

    # ------------------------------------------------------------------
    # Session logging
    # ------------------------------------------------------------------

    def log_session(self, status: str = "COMPLETED", process_mode: str = None) -> None:
        """Append a summary row for the current session to tracking_logs.csv.

        Creates the file with a header row if it does not yet exist.  Safe to
        call multiple times per session (e.g. for auto-save); each call appends
        a row with the current metric values.

        Args:
            status:       Session outcome label ("COMPLETED" or "PARTIAL").
            process_mode: Override for the Process_Mode CSV column.  If None,
                          inferred from the application state.
        """
        if not process_mode:
            if self.state == AppState.BATCH_PROCESSING:
                process_mode = "Batch Processed"
            elif self.current_bag_file:
                process_mode = "Load Recording"
            else:
                process_mode = "Live View"

        rest_ratio = (self.idle_time / self.elapsed_time) if self.elapsed_time > 0 else 0.0
        avg_jerk = (self.cumulative_jerk / self.jerk_count) if self.jerk_count > 0 else 0.0

        scan_label = "MANUAL" if getattr(self, 'live_scan_type', 'AI') == "MAN" else "AI"
        sid = self.session_id or f"{SESSION_ID_PREFIX}-{self.user_id_input}"
        rec_file = self.recording_filename or "N/A"
        num_points = len(self.all_hand_coords) if hasattr(self, 'all_hand_coords') else 0

        log_path = os.path.join(OUTPUT_PATH, "tracking_logs.csv")
        file_exists = os.path.isfile(log_path)

        with open(log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Session_ID", "Status", "Scan_Type", "Clip_Number",
                    "Recording_File", "Duration_s", "Distance_m", "Rest_Ratio",
                    "Avg_Jerk", "Hover_Time_s", "Path_Points", "Process_Mode",
                ])
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                sid, status, scan_label, self.clip_number, rec_file,
                f"{self.elapsed_time:.2f}", f"{self.total_distance:.3f}",
                f"{rest_ratio:.4f}", f"{avg_jerk:.4f}",
                f"{self.hover_time:.2f}", num_points, process_mode,
            ])

        print(f"[LOG] {status} | Session: {sid} ({scan_label}) | "
              f"Clip: {self.clip_number} | Duration: {self.elapsed_time:.1f}s | "
              f"Distance: {self.total_distance:.3f}m")

    # ------------------------------------------------------------------
    # Clip numbering
    # ------------------------------------------------------------------

    def get_next_clip_number(self, user_id: str) -> int:
        """Return the next available clip number for *user_id* by scanning the log and clips directory.

        Searches both ``tracking_logs.csv`` and the ``data/clips/`` folder for
        existing clip numbers assigned to *user_id*, then returns max + 1.

        Args:
            user_id: Zero-padded participant ID string (e.g. "03").

        Returns:
            Next clip number (1-based integer).
        """
        target_sid = f"{SESSION_ID_PREFIX}-{int(user_id):02d}"
        max_clip = 0

        log_file = os.path.join(OUTPUT_PATH, "tracking_logs.csv")
        if os.path.exists(log_file):
            try:
                with open(log_file) as f:
                    for row in csv.DictReader(f):
                        if row.get("Session_ID") == target_sid:
                            try:
                                c = int(row.get("Clip_Number", 0))
                                if c > max_clip:
                                    max_clip = c
                            except ValueError:
                                pass
            except Exception:
                pass

        clips_dir = os.path.join(OUTPUT_PATH, "data", "clips")
        if os.path.exists(clips_dir):
            pattern = re.compile(rf"session_{re.escape(target_sid)}_clip_(\d+)\.avi")
            for fname in os.listdir(clips_dir):
                m = pattern.match(fname)
                if m:
                    c = int(m.group(1))
                    if c > max_clip:
                        max_clip = c

        return max_clip + 1

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_recording(self, width: int, height: int) -> None:
        """Create a new ``ThreadedVideoWriter`` and set recording state.

        The output video is cropped to the centre ``(1 - 2*SIDE_MARGIN_RATIO)``
        fraction of the frame to match the tracking region of interest.

        Args:
            width:  Full frame width (pixels).
            height: Full frame height (pixels).
        """
        self.session_id = f"{SESSION_ID_PREFIX}-{int(self.user_id_input):02d}"
        self.clip_number = self.get_next_clip_number(self.user_id_input)

        output_dir = os.path.join(OUTPUT_PATH, "data", "clips")
        os.makedirs(output_dir, exist_ok=True)

        scan_tag = getattr(self, 'live_scan_type', 'AI')
        if len(scan_tag) >= 3:
            scan_tag = scan_tag[0]  # Shorten "MAN" → "M"

        self.recording_filename = (
            f"session_{self.session_id}_{scan_tag}_clip_{self.clip_number:02d}.avi"
        )
        full_path = os.path.join(output_dir, self.recording_filename)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        margin = int(width * SIDE_MARGIN_RATIO)
        crop_width = width - 2 * margin

        self.video_writer = ThreadedVideoWriter(full_path, fourcc, float(STREAM_FPS),
                                                (crop_width, height))
        print(f"[REC] Started: {full_path}  ({crop_width}×{height})")

    def stop_recording(self) -> None:
        """Finalise the current recording asynchronously to avoid blocking the UI."""
        if not getattr(self, 'is_saving', False):
            self.is_saving = True
            threading.Thread(target=self._async_save_worker, daemon=True).start()

    def _async_save_worker(self) -> None:
        """Background worker: flushes the video writer then generates output images."""
        try:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                print("[REC] Video saved.")
            self.generate_heatmap()
            self.generate_path_image()
        except Exception as exc:
            print(f"[ERROR] Save worker: {exc}")
        finally:
            self.is_saving = False

    # ------------------------------------------------------------------
    # Output image generation
    # ------------------------------------------------------------------

    def generate_heatmap(self) -> None:
        """Generate and save a 2-D spatial heatmap of hand positions.

        Uses a 2-D histogram of all recorded (x, y) pixel positions, applies
        logarithmic scaling to make both dense hover regions and sparse movement
        paths visible, then overlays the result on the first captured frame.

        The output is cropped to the centre ROI (matching the recorded video)
        and saved as a PNG in ``data/heatmaps/``.
        """
        if not self.all_hand_coords:
            return

        print("[INFO] Generating heatmap...")
        w, h = STREAM_WIDTH, STREAM_HEIGHT

        xs = [p[0] for p in self.all_hand_coords]
        ys = [p[1] for p in self.all_hand_coords]

        # 2-D histogram — fewer bins produce larger, more readable blobs
        heatmap, _, _ = np.histogram2d(xs, ys, bins=30,
                                        range=[[0, w], [0, h]])
        # histogram2d returns H[x, y]; transpose to get H[row, col] for image coords
        hm = heatmap.T

        # Logarithmic scale: compresses the dynamic range so both hover hotspots
        # (very high counts) and movement trails (low counts) are visible
        hm = np.log1p(hm)

        hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_CUBIC)
        hm = cv2.GaussianBlur(hm, (31, 31), 0)

        norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        if self.first_frame is not None:
            bg = self.first_frame
            if bg.shape[:2] != coloured.shape[:2]:
                bg = cv2.resize(bg, (w, h))
            # Blend: heatmap dominant (60 %) with faded first frame background (40 %)
            coloured = cv2.addWeighted(coloured, 0.6, bg, 0.4, 0)

        # Crop to centre ROI
        margin = int(w * SIDE_MARGIN_RATIO)
        coloured = coloured[:, margin:w - margin]

        output_dir = os.path.join(OUTPUT_PATH, "data", "heatmaps")
        os.makedirs(output_dir, exist_ok=True)
        scan_tag = getattr(self, 'live_scan_type', 'AI')
        if len(scan_tag) >= 3:
            scan_tag = scan_tag[0]
        fname = f"heatmap_{self.session_id}_{scan_tag}_clip_{self.clip_number:02d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), coloured)
        print(f"[INFO] Heatmap saved: {fname}")

    def generate_path_image(self) -> None:
        """Draw the full recorded hand path overlaid on the first captured frame.

        Path segments are drawn in gold.  ``None`` entries in ``path_points``
        represent hand-loss events and cause gaps in the drawn line.

        Output is saved as a PNG in ``data/paths/``.
        """
        coords = (self.all_hand_coords
                  if self.all_hand_coords else list(self.path_points))
        if len(coords) < 2 or self.first_frame is None:
            print("[INFO] Path image: not enough data.")
            return

        h, w = self.first_frame.shape[:2]
        path_img = self.first_frame.copy()

        for i in range(1, len(coords)):
            p1, p2 = coords[i - 1], coords[i]
            if p1 is not None and p2 is not None:
                cv2.line(path_img, p1, p2, (0, 215, 255), 2)  # Gold BGR

        margin = int(w * SIDE_MARGIN_RATIO)
        path_img = path_img[:, margin:w - margin]

        output_dir = os.path.join(OUTPUT_PATH, "data", "paths")
        os.makedirs(output_dir, exist_ok=True)
        scan_tag = getattr(self, 'live_scan_type', 'AI')
        if len(scan_tag) >= 3:
            scan_tag = scan_tag[0]
        fname = f"path_{self.session_id}_{scan_tag}_clip_{self.clip_number:02d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), path_img)
        print(f"[INFO] Path image saved: {fname}")

    # ------------------------------------------------------------------
    # Glove masking for MediaPipe
    # ------------------------------------------------------------------

    def apply_strict_glove_mask(self, image: np.ndarray) -> np.ndarray:
        """Isolate the coloured tracking glove and recolour it for MediaPipe detection.

        MediaPipe's hand model was trained on skin-coloured hands.  This method
        performs two transformations so that a bright cyan/blue nitrile glove
        is reliably detected:

        1. **HSV colour mask** — pixels outside the glove HSV range are zeroed,
           leaving only the glove visible.

        2. **Channel-mix "orange trick"** — the retained pixels are channel-mixed
           so that MediaPipe's RGB input sees an orange-ish colour rather than
           cyan, which more closely resembles skin tone:
           - Green channel halved  (suppresses the greenish component of cyan)
           - Red channel zeroed    (removes noise; MP's blue channel)
           In BGR terms: [B, G, R] → [B, G*0.5, 0]
           MediaPipe reads this as [R=B, G=G*0.5, B=0] ≈ orange.

        3. **ROI crop** — the left and right ``GLOVE_ROI_MARGIN`` fractions are
           blacked out to suppress monitor reflections at the frame edges.

        Args:
            image: BGR frame (modified in place and returned).

        Returns:
            The masked and recoloured image.
        """
        # Step 1: HSV colour segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, GLOVE_HSV_LOWER, GLOVE_HSV_UPPER)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Step 2: Channel mix — make cyan look orange to MediaPipe
        result[:, :, 1] = (result[:, :, 1] * 0.5).astype(np.uint8)  # halve green
        result[:, :, 2] = 0                                           # zero red

        # Step 3: Blank edge columns to exclude mirror/monitor reflections
        h, w = result.shape[:2]
        margin = int(w * GLOVE_ROI_MARGIN)
        result[:, :margin] = 0
        result[:, w - margin:] = 0

        return result

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def process_custom_metrics(self, detection_result, timestamp_ms: int,
                               w: int, h: int, depth_frame=None) -> None:
        """Update all motion metrics from the latest detection result.

        Called once per frame during tracking (skipped while playback is paused).

        Computes:
        - Hand centroid position in pixel space and appends to ``all_hand_coords``
        - 3-D world position via RealSense deprojection (if depth available)
        - Filtered position using One Euro Filter (Casiez et al. 2012)
        - Incremental path length, idle time, hover time, and jerk

        Landmark indices used for the centroid: 0 (wrist), 5, 9, 13, 17
        (MCP joints of each finger).  This provides a stable centroid that
        roughly corresponds to the centre of the hand palm.

        Args:
            detection_result: MediaPipe ``HandLandmarkerResult``.
            timestamp_ms:     Frame timestamp in milliseconds (used as time base
                              for derivative calculations).
            w, h:             Frame dimensions in pixels.
            depth_frame:      RealSense depth frame, or None if unavailable.
        """
        current_time = timestamp_ms / 1000.0

        # --- Hand loss ---
        if not detection_result.hand_landmarks:
            if self.is_tracking and self.prev_position_3d is not None:
                if self.path_points and self.path_points[-1] is not None:
                    self.path_points.append(None)  # Marks a gap in the path
                self.prev_position_3d = None
            return

        # --- Compute pixel centroid from key landmarks ---
        # Landmarks 0, 5, 9, 13, 17 = wrist + 4 MCP joints
        landmarks = detection_result.hand_landmarks[0]
        centroid_indices = [0, 5, 9, 13, 17]
        x_sum = sum(int(landmarks[i].x * w) for i in centroid_indices)
        y_sum = sum(int(landmarks[i].y * h) for i in centroid_indices)
        n = len(centroid_indices)
        cx = max(0, min(w - 1, x_sum // n))
        cy = max(0, min(h - 1, y_sum // n))

        if not self.is_tracking:
            return

        self.all_hand_coords.append((cx, cy))

        # --- 3-D position from RealSense depth ---
        if depth_frame is None:
            return

        d_meters = depth_frame.get_distance(cx, cy)
        if d_meters <= 0 or d_meters >= MAX_DEPTH_M:
            return

        # RealSense deprojection: converts pixel (cx, cy) + depth (metres)
        # into a 3-D point [X, Y, Z] in the camera coordinate frame (metres).
        # Uses the colour stream's intrinsic parameters (focal length, principal point).
        raw_point = rs.rs2_deproject_pixel_to_point(
            self.color_intrinsics, [cx, cy], d_meters
        )

        # One Euro Filter — Casiez et al. 2012 — applied independently to each axis
        if self.filter_x:
            px = self.filter_x.filter(current_time, raw_point[0])
            py = self.filter_y.filter(current_time, raw_point[1])
            pz = self.filter_z.filter(current_time, raw_point[2])
            pos = np.array([px, py, pz])
        else:
            pos = np.array(raw_point)

        # --- Incremental metrics ---
        if self.prev_position_3d is not None and self.prev_metric_time is not None:
            dt = current_time - self.prev_metric_time
            if dt > 0:
                dist = np.linalg.norm(pos - self.prev_position_3d)

                # Discard artefact jumps (tracking loss, re-detection at new position)
                if dist < MAX_DIST_JUMP_M:
                    self.total_distance += dist

                    vel = (pos - self.prev_position_3d) / dt
                    speed = np.linalg.norm(vel)

                    if speed < IDLE_SPEED_THRESHOLD:
                        self.idle_time += dt

                    if d_meters < HOVER_DEPTH_THRESHOLD and speed >= IDLE_SPEED_THRESHOLD:
                        self.hover_time += dt

                    # Jerk = d³position/dt³ (rate of change of acceleration)
                    if self.prev_velocity is not None:
                        accel = (vel - self.prev_velocity) / dt
                        if self.prev_accel is not None:
                            jerk = (accel - self.prev_accel) / dt
                            self.cumulative_jerk += np.linalg.norm(jerk)
                            self.jerk_count += 1
                        self.prev_accel = accel
                    self.prev_velocity = vel

                    self.prev_position_3d = pos
                    self.prev_metric_time = current_time
                    self.path_points.append((cx, cy))
        else:
            # First detection in this session — seed history
            self.prev_position_3d = pos
            self.prev_metric_time = current_time
            self.path_points.append((cx, cy))

    # ------------------------------------------------------------------
    # UI panels
    # ------------------------------------------------------------------

    def draw_live_stats(self, img: np.ndarray) -> None:
        """Render the real-time metrics panel on the right-side overlay.

        Draws a semi-transparent dark panel over the right ``SIDE_MARGIN_RATIO``
        of the frame and overlays large TIME and DIST values plus a compact table
        of REST, HOVER, and JERK metrics.

        Args:
            img: Display frame (modified in place).
        """
        h, w = img.shape[:2]
        margin = int(w * SIDE_MARGIN_RATIO)
        panel_x = w - margin
        cx = panel_x + margin // 2  # Horizontal centre of the panel

        font = cv2.FONT_HERSHEY_PLAIN
        scale_big = 2.2
        scale_sm = 1.2
        thick_big = 3
        thick_sm = 2

        def _stat(y: int, label: str, value: str) -> None:
            """Draw a label + large value centred in the panel."""
            (lw, _), _ = cv2.getTextSize(label, font, scale_sm, thick_sm)
            cv2.putText(img, label, (cx - lw // 2, y - 38), font, scale_sm, UI_TEXT_DIM, thick_sm)
            (vw, _), _ = cv2.getTextSize(value, font, scale_big, thick_big)
            cv2.putText(img, value, (cx - vw // 2 + 2, y + 2), font, scale_big, UI_BTN_SHADOW, thick_big)
            cv2.putText(img, value, (cx - vw // 2, y), font, scale_big, UI_TEXT, thick_big)

        def _row(y: int, label: str, value: str) -> None:
            """Draw a compact left-label / right-value row."""
            cv2.putText(img, label, (panel_x + 18, y), font, scale_sm, UI_TEXT_DIM, thick_sm)
            (vw, _), _ = cv2.getTextSize(value, font, scale_sm, thick_sm)
            cv2.putText(img, value, (w - 18 - vw, y), font, scale_sm, UI_TEXT, thick_sm)

        y = 115
        _stat(y, "TIME", time.strftime('%M:%S', time.gmtime(self.elapsed_time)))
        y += 80
        _stat(y, "DIST", f"{self.total_distance:.2f} m")
        y += 80

        rr = (self.idle_time / self.elapsed_time) if self.elapsed_time > 0 else 0.0
        _row(y, "REST",  f"{rr:.1%}");       y += 38
        _row(y, "HOVER", f"{self.hover_time:.0f}s"); y += 38
        avg_j = (self.cumulative_jerk / self.jerk_count) if self.jerk_count > 0 else 0.0
        _row(y, "JERK",  f"{avg_j:.1f}")

    # ------------------------------------------------------------------
    # Window icon
    # ------------------------------------------------------------------

    def set_window_icon(self) -> None:
        """Set the taskbar/title-bar icon for the OpenCV window (Windows only).

        Looks for ``app_icon.ico`` next to the script or executable.  Silently
        skips on non-Windows platforms or if the icon file is absent.
        """
        try:
            import ctypes
            icon_dir = (os.path.dirname(sys.executable)
                        if getattr(sys, 'frozen', False)
                        else os.path.dirname(os.path.abspath(__file__)))
            icon_path = os.path.join(icon_dir, "app_icon.ico")
            if not os.path.exists(icon_path):
                return
            hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)
            if hwnd:
                hicon = ctypes.windll.user32.LoadImageW(
                    None, icon_path, 1, 0, 0, 0x00000010)
                if hicon:
                    ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 0, hicon)
                    ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 1, hicon)
        except Exception as exc:
            print(f"[WARN] Could not set window icon: {exc}")

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset_session_state(self) -> None:
        """Reset all per-session metrics and tracking state to zero/empty.

        Called at the start of each new recording session (live or batch) so that
        metrics from the previous session do not carry over.  Also reinitialises
        the One Euro filters with fresh timestamps.
        """
        self.is_tracking = False
        self.total_distance = 0.0
        self.elapsed_time = 0.0
        self.start_time = 0.0
        self.last_frame_time = time.time()
        self.prev_position_3d = None
        self.path_points.clear()
        self.all_hand_coords.clear()
        self.first_frame = None
        self.timestamp_ms = 0

        now = time.time()
        self.filter_x = OneEuroFilter(now, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA)
        self.filter_y = OneEuroFilter(now, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA)
        self.filter_z = OneEuroFilter(now, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA)

        self.idle_time = 0.0
        self.hover_time = 0.0
        self.cumulative_jerk = 0.0
        self.jerk_count = 0
        self.prev_velocity = None
        self.prev_accel = None
        self.prev_metric_time = None

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Create the OpenCV window and enter the main event loop.

        Delegates each frame to the appropriate state handler.  Exits cleanly
        on 'q' keypress or when ``self.running`` is set to False by a button
        handler.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        try:
            import ctypes
            hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
        except Exception:
            pass
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.set_window_icon()

        while self.running:
            if self.state == AppState.CONNECTING:
                self.run_connecting()
            elif self.state == AppState.TRACKING:
                self.run_tracking()
            elif self.state == AppState.FINISHED:
                self.run_finished()
            elif self.state == AppState.BATCH_PROCESSING:
                self.run_batch_processing()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Graceful shutdown
        if self.is_tracking:
            self.log_session()
            self.stop_recording()
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("[INFO] Application closed.")

    # ------------------------------------------------------------------
    # State: CONNECTING (main menu)
    # ------------------------------------------------------------------

    def run_connecting(self) -> None:
        """Render the main menu screen and handle source-selection buttons.

        Buttons:
        - **Start Live Camera** — opens the connected RealSense camera.
        - **Load Recording**   — opens a .bag file for playback.
        - **Batch Process**    — switches to the batch queue screen.
        - **X**                — closes the application.

        Source initialisation is performed on a background thread to keep this
        screen responsive while the RealSense pipeline starts up.
        """
        h, w = STREAM_HEIGHT, STREAM_WIDTH
        img = np.full((h, w, 3), UI_BG, dtype=np.uint8)

        # Horizontal accent rule below header area
        cv2.line(img, (40, 105), (w - 40, 105), UI_BORDER, 1)

        # Title and subtitle
        self._draw_title(img, APP_TITLE, y=75)
        self._draw_subtitle(img, "Intel RealSense  |  MediaPipe Hand Landmarks", y=98)

        center_x = w // 2
        btn_w, btn_h = 240, 50
        btn_live_rect  = (center_x - btn_w // 2, 145, btn_w, btn_h)
        btn_load_rect  = (center_x - btn_w // 2, 210, btn_w, btn_h)
        btn_batch_rect = (center_x - btn_w // 2, 275, btn_w, btn_h)

        close_size = 40
        close_rect = (w - close_size - 10, 10, close_size, close_size)

        mx, my = self.mouse_pos
        hover_live  = self.is_inside_button((mx, my), btn_live_rect[:2],  btn_live_rect[2:])
        hover_load  = self.is_inside_button((mx, my), btn_load_rect[:2],  btn_load_rect[2:])
        hover_batch = self.is_inside_button((mx, my), btn_batch_rect[:2], btn_batch_rect[2:])
        hover_close = self.is_inside_button((mx, my), close_rect[:2],     close_rect[2:])

        live_label = "Retry Connection" if self.connection_attempted else "Start Live Camera"
        self.draw_button(img, btn_live_rect,  live_label,      hover_live)
        self.draw_button(img, btn_load_rect,  "Load Recording", hover_load)
        self.draw_button(img, btn_batch_rect, "Batch Process",  hover_batch)
        self.draw_button(img, close_rect,     "X",              hover_close)

        # Loading spinner text
        if self.loading_pipeline:
            self._draw_subtitle(img, "Initialising pipeline...", y=355)

        # Connection-failure notice
        if self.connection_attempted:
            font = cv2.FONT_HERSHEY_PLAIN
            msg = "Camera not found — check USB connection"
            (mw, _), _ = cv2.getTextSize(msg, font, 1.2, 1)
            cv2.putText(img, msg, ((w - mw) // 2, 345), font, 1.2, UI_WARNING, 1)

        cv2.imshow(self.window_name, img)

        if self.clicked:
            if hover_live and not self.loading_pipeline:
                self.loading_pipeline = True
                self._pending_bag_file = None
                threading.Thread(target=self._init_pipeline_async, daemon=True).start()

            elif hover_load and not self.loading_pipeline:
                file_path = open_file_dialog(
                    title="Select Recording",
                    filetypes=[("ROS Bag", "*.bag"), ("All Files", "*.*")],
                )
                if file_path:
                    # Auto-detect scan type and participant ID from filename
                    fname = os.path.basename(file_path)
                    fn_upper = fname.upper()
                    if "_M_" in fn_upper or "_M." in fn_upper:
                        self.live_scan_type = "MAN"
                    elif "AI" in fn_upper:
                        self.live_scan_type = "AI"

                    m = re.search(r'\d+', fname)
                    if m:
                        safe_id = max(0, min(MAX_PARTICIPANT_ID, int(m.group())))
                        self.id_slider_val = safe_id
                        self.user_id_input = f"{safe_id:02d}"

                    self.loading_pipeline = True
                    self._pending_bag_file = file_path
                    threading.Thread(target=self._init_pipeline_async, daemon=True).start()

            elif hover_batch:
                self.state = AppState.BATCH_PROCESSING
                self.batch_status = "Idle"

            elif hover_close:
                self.running = False

            self.clicked = False

        # Check for async pipeline completion
        if self.loading_result is not None:
            if self.loading_result:
                self.state = AppState.TRACKING
                self.connection_attempted = False
            else:
                self.connection_attempted = True
            self.loading_result = None
            self.loading_pipeline = False

    # ------------------------------------------------------------------
    # State: BATCH_PROCESSING
    # ------------------------------------------------------------------

    def run_batch_processing(self) -> None:
        """Render the batch queue screen and handle queue management interactions.

        Each file in the queue shows its filename, per-file participant ID
        (adjustable with < / > arrows), scan-type toggle (A / M), and a
        remove button (x).  A progress bar is shown while the batch runs.
        """
        h, w = STREAM_HEIGHT, STREAM_WIDTH
        img = np.full((h, w, 3), UI_BG, dtype=np.uint8)

        font = cv2.FONT_HERSHEY_PLAIN

        # Header
        self._draw_title(img, "BATCH PROCESSING QUEUE", y=48)
        cv2.line(img, (40, 58), (w - 40, 58), UI_BORDER, 1)

        # --- File list panel ---
        list_x, list_y = 80, 68
        list_w, list_h = w - 160, 230
        cv2.rectangle(img, (list_x, list_y),
                      (list_x + list_w, list_y + list_h), UI_PANEL, -1)
        cv2.rectangle(img, (list_x, list_y),
                      (list_x + list_w, list_y + list_h), UI_BORDER, 1)

        mx, my = self.mouse_pos
        self._batch_file_toggles = []
        y_off = list_y + 25
        scale = 1.2

        if not self.batch_queue:
            cv2.putText(img, "No files — click 'Add Files' to begin.",
                        (list_x + 15, y_off), font, scale, UI_TEXT_DIM, 1)
        else:
            max_vis = 5
            vis_start = self.batch_scroll_offset
            vis_end = min(vis_start + max_vis, len(self.batch_queue))

            for disp_idx, i in enumerate(range(vis_start, vis_end)):
                item = self.batch_queue[i]
                fname = os.path.basename(item['path'])
                if fname.lower().endswith('.bag'):
                    fname = fname[:-4]
                pid = item.get('participant_id', '00')
                stype = item.get('scan_type', 'AI')

                # Status badge
                res = self.batch_file_results.get(i)
                status_txt = ""
                status_col = UI_TEXT
                if res == "DONE":
                    status_txt, status_col = "[DONE]", UI_SUCCESS
                elif res == "PARTIAL":
                    status_txt, status_col = "[PARTIAL]", UI_WARNING
                elif i == self.batch_progress and self.batch_status == "Running":
                    status_txt, status_col = "[...]", UI_ACCENT

                fname_short = (fname[:18] + "..") if len(fname) > 18 else fname
                cv2.putText(img, f"{i+1}.", (list_x + 10, y_off), font, 0.9, UI_TEXT_DIM, 1)
                cv2.putText(img, fname_short, (list_x + 35, y_off), font, 0.85, UI_TEXT, 1)

                # Per-file stats for completed entries
                if i in self.batch_file_stats:
                    s = self.batch_file_stats[i]
                    stxt = f"{s['distance']:.2f}m  H:{s['hover']:.0f}s  J:{s['jerk']:.2f}"
                    cv2.putText(img, stxt, (list_x + 185, y_off), font, 0.7, UI_TEXT_DIM, 1)

                # Per-file controls (right-aligned)
                is_active = (self.batch_status == "Running" and i == self.batch_progress)
                right = list_x + list_w - 15
                cw, ch = 22, 22
                tw2, th2 = 28, 22
                iw, ih = 55, 22
                btn_y = y_off - 15

                close_x = right - cw
                ai_x    = close_x - tw2 - 10
                man_x   = ai_x - tw2 - 5
                id_x    = man_x - iw - 8

                # Dim colours for the actively-processing row
                dim = is_active
                id_bg  = (UI_PANEL if dim else (UI_BTN_HOVER if self.is_inside_button((mx, my), (id_x, btn_y), (iw, ih)) else UI_BTN_IDLE))
                id_brd = UI_BORDER

                cv2.rectangle(img, (id_x, btn_y), (id_x + iw, btn_y + ih), id_bg, -1)
                cv2.rectangle(img, (id_x, btn_y), (id_x + iw, btn_y + ih), id_brd, 1)
                txt_cy = btn_y + ih // 2 + 4
                arrow_col = UI_TEXT_DIM if dim else UI_ACCENT
                cv2.putText(img, "<",    (id_x + 3,        txt_cy), font, 0.9, arrow_col, 1)
                cv2.putText(img, f"{pid}", (id_x + 20,     txt_cy), font, 1.0, UI_TEXT if not dim else UI_TEXT_DIM, 1)
                cv2.putText(img, ">",    (id_x + iw - 10,  txt_cy), font, 0.9, arrow_col, 1)

                # MAN toggle
                is_man = stype == "MAN"
                man_rect = (man_x, btn_y, tw2, th2)
                man_bg = UI_BTN_ACTIVE if (is_man and not dim) else UI_BTN_IDLE
                cv2.rectangle(img, (man_x, btn_y), (man_x + tw2, btn_y + th2), man_bg, -1)
                cv2.rectangle(img, (man_x, btn_y), (man_x + tw2, btn_y + th2), UI_BORDER, 1)
                cv2.putText(img, "M", (man_x + 9, txt_cy), font, 1.0,
                            UI_TEXT if is_man and not dim else UI_TEXT_DIM, 1)

                # AI toggle
                is_ai = stype == "AI"
                ai_rect = (ai_x, btn_y, tw2, th2)
                ai_bg = UI_BTN_ACTIVE if (is_ai and not dim) else UI_BTN_IDLE
                cv2.rectangle(img, (ai_x, btn_y), (ai_x + tw2, btn_y + th2), ai_bg, -1)
                cv2.rectangle(img, (ai_x, btn_y), (ai_x + tw2, btn_y + th2), UI_BORDER, 1)
                cv2.putText(img, "A", (ai_x + 9, txt_cy), font, 1.0,
                            UI_TEXT if is_ai and not dim else UI_TEXT_DIM, 1)

                # Remove (x)
                close_rect_f = (close_x, btn_y, cw, ch)
                x_bg = (UI_BTN_HOVER if self.is_inside_button((mx, my), (close_x, btn_y), (cw, ch))
                        and not dim else UI_BTN_IDLE)
                cv2.rectangle(img, (close_x, btn_y), (close_x + cw, btn_y + ch), x_bg, -1)
                cv2.rectangle(img, (close_x, btn_y), (close_x + cw, btn_y + ch), UI_BORDER, 1)
                cv2.putText(img, "x" if not dim else "-",
                            (close_x + 6, txt_cy - 2), font, 1.0,
                            (80, 100, 220) if not dim else UI_TEXT_DIM, 1)

                if status_txt:
                    (sw, _), _ = cv2.getTextSize(status_txt, font, scale * 0.7, 1)
                    cv2.putText(img, status_txt, (id_x - sw - 10, y_off),
                                font, scale * 0.7, status_col, 1)

                self._batch_file_toggles.append({
                    'idx': i,
                    'id_rect': (id_x, btn_y, iw, ih),
                    'id_center_x': id_x + iw // 2,
                    'man_rect': man_rect,
                    'ai_rect': ai_rect,
                    'close_rect': close_rect_f,
                })
                y_off += 35

            if len(self.batch_queue) > max_vis:
                cv2.putText(img,
                            f"  {vis_start+1}–{vis_end} of {len(self.batch_queue)}  (scroll)",
                            (list_x + 15, y_off), font, 0.9, UI_TEXT_DIM, 1)

        # --- Progress bar ---
        bar_y = list_y + list_h + 15
        bar_h = 22
        if self.batch_status == "Running":
            total = len(self.batch_queue)
            pct = self.batch_progress / total if total > 0 else 0
            cv2.rectangle(img, (list_x, bar_y), (list_x + list_w, bar_y + bar_h), UI_PANEL, -1)
            cv2.rectangle(img, (list_x, bar_y),
                          (list_x + int(list_w * pct), bar_y + bar_h), UI_PROGRESS, -1)
            cv2.rectangle(img, (list_x, bar_y), (list_x + list_w, bar_y + bar_h), UI_BORDER, 1)
            pct_txt = (f"{int(pct*100)}%  |  File {self.batch_progress+1}/{total}"
                       f"  |  Frames: {getattr(self, 'batch_frame_count', 0)}")
            (ptw, _), _ = cv2.getTextSize(pct_txt, font, 1.1, 1)
            cv2.putText(img, pct_txt, ((w - ptw) // 2, bar_y + 16), font, 1.1, UI_TEXT, 1)

        # --- Action buttons ---
        btn_y2 = bar_y + bar_h + 18
        btn_bw, btn_bh = 135, 40
        btn_area = w // 2 - 100
        running = self.batch_status == "Running"

        btn_add_rect   = (btn_area,                btn_y2, btn_bw, btn_bh)
        btn_clear_rect = (btn_area + btn_bw + 12,  btn_y2, btn_bw, btn_bh)
        btn_start_rect = (btn_area + (btn_bw+12)*2, btn_y2, btn_bw + 15, btn_bh)
        back_rect      = (20, 20, 40, 40)
        close_rect_g   = (w - 60, 20, 40, 40)

        hov_add   = self.is_inside_button((mx, my), btn_add_rect[:2],   btn_add_rect[2:])
        hov_clear = self.is_inside_button((mx, my), btn_clear_rect[:2], btn_clear_rect[2:])
        hov_start = self.is_inside_button((mx, my), btn_start_rect[:2], btn_start_rect[2:])
        hov_back  = self.is_inside_button((mx, my), back_rect[:2],      back_rect[2:])
        hov_close = self.is_inside_button((mx, my), close_rect_g[:2],   close_rect_g[2:])

        self.draw_button(img, btn_add_rect,   "Add Files",            hov_add,   is_active=False)
        if running:
            # Visually disable Clear while running
            cv2.rectangle(img, (btn_clear_rect[0], btn_clear_rect[1]),
                          (btn_clear_rect[0]+btn_clear_rect[2], btn_clear_rect[1]+btn_clear_rect[3]),
                          UI_BTN_IDLE, -1)
            cv2.putText(img, "Clear",
                        (btn_clear_rect[0]+38, btn_clear_rect[1]+25), font, 1.2, UI_TEXT_DIM, 1)
            self.draw_button(img, btn_start_rect, "Cancel", hov_start, is_active=True)
        else:
            self.draw_button(img, btn_clear_rect, "Clear", hov_clear)
            self.draw_button(img, btn_start_rect, "Start", hov_start)
        self.draw_button(img, back_rect,    "<", hov_back)
        self.draw_button(img, close_rect_g, "X", hov_close)

        cv2.imshow(self.window_name, img)

        if self.clicked:
            self.clicked = False

            if hov_back:
                if running:
                    self.batch_cancel_requested = True
                self.reset_session_state()
                self.state = AppState.CONNECTING

            elif hov_add:
                try:
                    file_paths = open_files_dialog(
                        title="Select Bag Files",
                        filetypes=[("ROS Bag", "*.bag"), ("All Files", "*.*")],
                    )
                    if file_paths:
                        existing = [item['path'] for item in self.batch_queue]
                        for fp in file_paths:
                            fp = os.path.normpath(fp)
                            if fp in existing:
                                continue
                            fn = os.path.basename(fp)
                            fn_up = fn.upper()
                            stype = self.batch_default_scan_type
                            if "_M_" in fn_up or "_M." in fn_up:
                                stype = "MAN"
                            elif "AI" in fn_up:
                                stype = "AI"
                            pid = self.batch_default_participant_id
                            m = re.search(r'\d+', fn)
                            if m:
                                pid = f"{int(m.group()):02d}"
                            self.batch_queue.append(
                                {'path': fp, 'participant_id': pid, 'scan_type': stype}
                            )
                except Exception as exc:
                    print(f"[ERROR] File dialog: {exc}")

            elif hov_clear and not running:
                self.batch_queue.clear()
                self.batch_progress = 0
                self.batch_status = "Idle"
                self.batch_file_stats.clear()
                self.batch_file_results.clear()
                self.batch_scroll_offset = 0

            elif hov_start:
                if running:
                    self.batch_cancel_requested = True
                elif self.batch_queue:
                    self.start_batch_processing()

            elif hov_close:
                if running:
                    self.batch_cancel_requested = True
                self.running = False

            else:
                # Per-file toggle clicks
                for toggle in getattr(self, '_batch_file_toggles', []):
                    idx = toggle['idx']
                    if idx >= len(self.batch_queue):
                        continue
                    is_active_file = (running and idx == self.batch_progress)
                    if self.is_inside_button((mx, my),
                                             toggle['close_rect'][:2],
                                             toggle['close_rect'][2:]):
                        if not is_active_file:
                            self.batch_queue.pop(idx)
                            if running and idx < self.batch_progress:
                                self.batch_progress -= 1
                        break
                    if not is_active_file:
                        if self.is_inside_button((mx, my),
                                                  toggle['id_rect'][:2],
                                                  toggle['id_rect'][2:]):
                            try:
                                cur = int(self.batch_queue[idx]['participant_id'])
                                new_id = ((cur - 1) % (MAX_PARTICIPANT_ID + 1)
                                          if mx < toggle['id_center_x']
                                          else (cur + 1) % (MAX_PARTICIPANT_ID + 1))
                                self.batch_queue[idx]['participant_id'] = f"{new_id:02d}"
                            except ValueError:
                                self.batch_queue[idx]['participant_id'] = "00"
                            break
                        elif self.is_inside_button((mx, my),
                                                    toggle['man_rect'][:2],
                                                    toggle['man_rect'][2:]):
                            self.batch_queue[idx]['scan_type'] = "MAN"
                            break
                        elif self.is_inside_button((mx, my),
                                                    toggle['ai_rect'][:2],
                                                    toggle['ai_rect'][2:]):
                            self.batch_queue[idx]['scan_type'] = "AI"
                            break

    # ------------------------------------------------------------------
    # State: FINISHED (session summary)
    # ------------------------------------------------------------------

    def run_finished(self) -> None:
        """Render the post-session summary screen.

        Shows session ID, scan mode, and the full set of computed metrics in a
        card layout.  Buttons return to the main menu or close the application.
        """
        h, w = STREAM_HEIGHT, STREAM_WIDTH
        img = np.full((h, w, 3), UI_BG, dtype=np.uint8)

        self._draw_title(img, "SESSION COMPLETE", y=48)
        cv2.line(img, (40, 58), (w - 40, 58), UI_BORDER, 1)

        font = cv2.FONT_HERSHEY_PLAIN

        # Info card
        bx, by, bw, bh = 80, 68, w - 160, 285
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), UI_PANEL, -1)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), UI_BORDER, 1)

        sid = getattr(self, 'session_id', f"{SESSION_ID_PREFIX}-{self.user_id_input}")
        scan_type = getattr(self, 'live_scan_type', 'AI')
        type_label = "Manual" if scan_type == "MAN" else "AI-Assisted"

        iy = by + 35
        cv2.putText(img, f"Session ID: {sid}", (bx + 20, iy), font, 1.5, UI_TEXT, 2)
        cv2.putText(img, f"Mode: {type_label}", (bx + 360, iy), font, 1.5, UI_ACCENT, 2)
        cv2.line(img, (bx + 20, iy + 12), (bx + bw - 20, iy + 12), UI_BORDER, 1)

        # Stats grid (2 columns)
        duration = getattr(self, 'elapsed_time', 0.0)
        distance = getattr(self, 'total_distance', 0.0)
        rest_pct = (self.idle_time / duration * 100) if duration > 0 else 0.0
        hover_t  = getattr(self, 'hover_time', 0.0)
        avg_j    = (self.cumulative_jerk / self.jerk_count
                    if self.jerk_count > 0 else 0.0)
        clip_n   = getattr(self, 'clip_number', 1)
        n_pts    = len(self.all_hand_coords) if hasattr(self, 'all_hand_coords') else 0

        stats = [
            ("Duration:",    f"{duration:.1f}s"),
            ("Distance:",    f"{distance:.3f}m"),
            ("Rest Ratio:",  f"{rest_pct:.1f}%"),
            ("Hover Time:",  f"{hover_t:.1f}s"),
            ("Avg Jerk:",    f"{avg_j:.2f}"),
            ("Clip ID:",     f"{clip_n:02d}"),
            ("Path Points:", f"{n_pts}"),
        ]
        if getattr(self, 'current_bag_file', None):
            stats.append(("File:", os.path.splitext(
                os.path.basename(self.current_bag_file))[0]))

        sy = iy + 42
        for i, (lbl, val) in enumerate(stats):
            row, col = i // 2, i % 2
            x = bx + 20 + col * 310
            y = sy + row * 34
            cv2.putText(img, lbl, (x, y), font, 1.2, UI_TEXT_DIM, 1)
            vs = 1.2
            # Auto-shrink long filenames
            (vw, _), _ = cv2.getTextSize(val, font, vs, 2)
            max_w = (bx + bw - 10) - (x + 180)
            while vw > max_w and vs > 0.6:
                vs -= 0.1
                (vw, _), _ = cv2.getTextSize(val, font, vs, 2)
            cv2.putText(img, val, (x + 180, y), font, vs, UI_TEXT, 2)

        # Buttons
        btn_y = by + bh + 25
        btn_bw2, btn_bh2 = 175, 45
        gap = 40
        back_x = (w - btn_bw2 * 2 - gap) // 2
        back_rect  = (back_x,              btn_y, btn_bw2, btn_bh2)
        close_rect = (back_x + btn_bw2 + gap, btn_y, btn_bw2, btn_bh2)

        mx, my = self.mouse_pos
        hov_back  = self.is_inside_button((mx, my), back_rect[:2],  back_rect[2:])
        hov_close = self.is_inside_button((mx, my), close_rect[:2], close_rect[2:])
        self.draw_button(img, back_rect,  "Back to Menu", hov_back)
        self.draw_button(img, close_rect, "Close",        hov_close)

        if self.clicked:
            if hov_back:
                self.state = AppState.CONNECTING
                self.current_bag_file = None
            elif hov_close:
                self.running = False
            self.clicked = False

        cv2.imshow(self.window_name, img)

    # ------------------------------------------------------------------
    # Batch processing engine
    # ------------------------------------------------------------------

    def start_batch_processing(self) -> None:
        """Launch the batch processing worker thread if not already running."""
        if hasattr(self, '_batch_thread') and self._batch_thread.is_alive():
            print("[BATCH] Already running.")
            return
        self.batch_status = "Running"
        self.batch_progress = 0
        self.batch_frame_count = 0
        self._batch_thread = threading.Thread(
            target=self._batch_processing_worker, daemon=True)
        self._batch_thread.start()

    def _batch_processing_worker(self) -> None:
        """Process each file in ``self.batch_queue`` sequentially.

        For each file:
        1. Fully resets session state.
        2. Initialises the RealSense pipeline in fast (non-real-time) mode.
        3. Runs the full tracking loop, writing the annotated video.
        4. Saves the heatmap and path image, logs metrics to CSV.
        5. Waits for async save to complete before moving to the next file.

        Respects ``self.batch_cancel_requested`` between and within files.
        """
        self.batch_file_results.clear()
        self.batch_file_stats.clear()
        self.batch_progress = 0

        while self.batch_progress < len(self.batch_queue):
            i = self.batch_progress
            if self.batch_cancel_requested:
                print(f"[BATCH] Cancelled before file {i+1}.")
                self.batch_status = "Cancelled"
                self.batch_cancel_requested = False
                return

            item = self.batch_queue[i]
            fpath = item['path']
            self.user_id_input = item.get('participant_id', '00')
            self.session_id = f"{SESSION_ID_PREFIX}-{self.user_id_input}"
            self.live_scan_type = item.get('scan_type', 'AI')
            self.batch_frame_count = 0

            print(f"\n{'='*60}")
            print(f"[BATCH] File {i+1}/{len(self.batch_queue)}: {os.path.basename(fpath)}")
            print(f"        Participant: {self.user_id_input}  Scan: {self.live_scan_type}")
            print(f"{'='*60}")

            self.reset_session_state()

            # Recreate the MediaPipe landmarker to reset its internal timestamp counter
            try:
                self.landmarker = self._create_hand_landmarker()
            except Exception as exc:
                print(f"[BATCH] Could not recreate landmarker: {exc}")
                self.batch_progress += 1
                continue

            if not self.init_pipeline(bag_file=fpath):
                print(f"[BATCH] Skipping {os.path.basename(fpath)} — pipeline init failed.")
                self.batch_progress += 1
                continue

            self.is_tracking = True
            self.clip_number = self.get_next_clip_number(self.user_id_input)
            self.recording_filename = (
                os.path.splitext(os.path.basename(fpath))[0] + "_processed"
            )
            self.start_time = time.time()

            # Use a monotonic local timestamp (avoids MediaPipe monotonicity errors)
            local_ts_ms = 0

            # Processing loop
            while True:
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError:
                    break  # End of bag file

                aligned = self.align.process(frames)
                if not aligned:
                    continue
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                if self.is_rgb_input:
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

                h_f, w_f = color_image.shape[:2]

                if self.first_frame is None:
                    self.first_frame = color_image.copy()
                if self.video_writer is None:
                    self.start_recording(w_f, h_f)

                # Glove masking → MediaPipe inference
                inf_img = self.apply_strict_glove_mask(color_image.copy())
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=inf_img)
                local_ts_ms += FRAME_MS
                result = self.landmarker.detect_for_video(mp_img, local_ts_ms)

                self.elapsed_time = time.time() - self.start_time
                self.batch_frame_count += 1

                # Draw landmarks and update metrics
                color_image = self.draw_landmarks(color_image, result)
                self.process_custom_metrics(result, local_ts_ms, w_f, h_f, depth_frame)

                # XYZ overlay
                self._draw_xyz_overlay(color_image, w_f, h_f)

                # Fading trail + position dot
                self._draw_trail(color_image)

                if self.video_writer:
                    m = int(w_f * SIDE_MARGIN_RATIO)
                    self.video_writer.write(color_image[:, m:w_f - m])

                if self.batch_cancel_requested:
                    print("[BATCH] Cancel requested — saving partial results.")
                    break

            # End of file
            is_partial = bool(self.batch_cancel_requested)
            self.batch_file_results[i] = "PARTIAL" if is_partial else "DONE"
            avg_j = (self.cumulative_jerk / self.jerk_count
                     if self.jerk_count > 0 else 0.0)
            self.batch_file_stats[i] = {
                'distance': self.total_distance, 'hover': self.hover_time,
                'jerk': avg_j, 'duration': self.elapsed_time,
            }

            self.log_session(status="PARTIAL" if is_partial else "COMPLETED",
                             process_mode="Batch Processed")
            self.stop_recording()

            print(f"\n  Summary: {os.path.basename(fpath)}")
            print(f"  Distance:  {self.total_distance:.3f} m")
            print(f"  Hover:     {self.hover_time:.1f} s")
            print(f"  Avg Jerk:  {avg_j:.4f}")
            print(f"  Duration:  {self.elapsed_time:.1f} s")
            print(f"  Frames:    {self.batch_frame_count}")

            # Wait for async save before resetting (path image needs all_hand_coords)
            wait_start = time.time()
            while getattr(self, 'is_saving', False):
                time.sleep(0.1)
                if time.time() - wait_start > 30:
                    print("[WARN] Timeout waiting for save.")
                    break

            if self.pipeline:
                try:
                    self.pipeline.stop()
                    self.pipeline = None
                except Exception:
                    pass

            self.is_tracking = False
            self.reset_session_state()
            self.batch_progress += 1

            if self.batch_cancel_requested:
                self.batch_status = "Cancelled"
                self.batch_cancel_requested = False
                break

        self.batch_status = "Idle"
        self.batch_progress = len(self.batch_queue)
        self.batch_cancel_requested = False
        self.is_tracking = False
        self.pipeline = None
        self.state = AppState.BATCH_PROCESSING
        print("[BATCH] Complete.")

    # ------------------------------------------------------------------
    # State: TRACKING (live camera or bag playback)
    # ------------------------------------------------------------------

    def run_tracking(self) -> None:
        """Main per-frame handler for live and bag-file tracking sessions.

        Performs in order:
        1. Frame acquisition (or freeze if paused)
        2. Colour-space correction (RGB→BGR for bag files)
        3. Auto-pause after 1 s for bag files (so the user can review before tracking)
        4. Glove masking and MediaPipe hand detection
        5. Metric update (``process_custom_metrics``)
        6. Visualisation: landmark skeleton, fading position trail, XYZ readout
        7. Video frame write (centre-cropped)
        8. UI overlay: side panels, stats, buttons, participant ID slider, scan-type toggle
        """
        # --- Frame acquisition ---
        if self.current_bag_file and self.bag_playback_paused:
            if not hasattr(self, '_frozen_aligned_frames'):
                return  # No frozen frame yet
            aligned = self._frozen_aligned_frames
            depth_frame = self._frozen_depth_frame
            color_frame = self._frozen_color_frame
        else:
            try:
                frames = self.pipeline.wait_for_frames()
            except RuntimeError:
                print("[INFO] Stream ended or connection lost.")
                if self.is_tracking:
                    self.log_session()
                    self.stop_recording()
                    self.is_tracking = False
                self.state = (AppState.FINISHED if self.current_bag_file
                              else AppState.CONNECTING)
                return

            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                return

            if self.current_bag_file:
                self._frozen_aligned_frames = aligned
                self._frozen_depth_frame = depth_frame
                self._frozen_color_frame = color_frame

        color_image = np.asanyarray(color_frame.get_data())
        if self.is_rgb_input:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        h, w = color_image.shape[:2]

        # --- Auto-pause bag files after 1 second (review before tracking starts) ---
        if self.current_bag_file and not self.bag_auto_pause_done:
            if self.bag_playback_start_time is None:
                self.bag_playback_start_time = time.time()
            elif time.time() - self.bag_playback_start_time >= 1.0:
                self.bag_playback_paused = True
                self.bag_auto_pause_done = True

        # --- Button layout ---
        left_margin = int(w * SIDE_MARGIN_RATIO)
        btn_w, btn_h = 120, 40
        btn_x = (left_margin - btn_w) // 2
        start_rect = (btn_x, 80, btn_w, btn_h)
        reset_rect = (btn_x, 80 + btn_h + 20, btn_w, btn_h)
        close_rect = (w - 50, 10, 40, 40)
        back_rect  = (20, 10, 40, 40)
        end_btn_w, end_btn_h = 180, 45
        end_rect = (w - left_margin + (left_margin - end_btn_w) // 2,
                    h - end_btn_h - 25, end_btn_w, end_btn_h)

        # --- Glove masking + MediaPipe detection ---
        inf_img = self.apply_strict_glove_mask(color_image.copy())
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=inf_img)
        self.timestamp_ms += FRAME_MS
        result = self.landmarker.detect_for_video(mp_img, int(self.timestamp_ms))

        mx, my = self.mouse_pos
        hover_start = self.is_inside_button((mx, my), start_rect[:2], start_rect[2:])
        hover_reset = self.is_inside_button((mx, my), reset_rect[:2], reset_rect[2:])
        hover_close = self.is_inside_button((mx, my), close_rect[:2], close_rect[2:])
        hover_back  = self.is_inside_button((mx, my), back_rect[:2],  back_rect[2:])
        hover_end   = self.is_inside_button((mx, my), end_rect[:2],   end_rect[2:])

        # --- Button click handling ---
        if self.clicked:
            if hover_start:
                if self.is_tracking:
                    self.log_session()
                    self.stop_recording()
                else:
                    now = time.time()
                    self.last_frame_time = now
                    self.start_recording(w, h)
                    self.filter_x = OneEuroFilter(now, FILTER_MIN_CUTOFF, FILTER_BETA)
                    self.filter_y = OneEuroFilter(now, FILTER_MIN_CUTOFF, FILTER_BETA)
                    self.filter_z = OneEuroFilter(now, FILTER_MIN_CUTOFF, FILTER_BETA)
                    self.idle_time = self.hover_time = 0.0
                    self.cumulative_jerk = 0.0
                    self.jerk_count = 0
                    self.prev_velocity = self.prev_accel = self.prev_metric_time = None
                self.is_tracking = not self.is_tracking
                self.clicked = False

            elif hover_reset:
                if self.is_tracking:
                    self.log_session()
                    self.stop_recording()
                    self.is_tracking = False
                self.total_distance = 0.0
                self.elapsed_time = 0.0
                self.prev_position_3d = None
                self.path_points.clear()
                self.clicked = False

            elif hover_back:
                if self.is_tracking:
                    self.log_session()
                    self.stop_recording()
                    self.is_tracking = False
                if self.current_bag_file and self.pipeline:
                    try:
                        self.pipeline.stop()
                    except Exception:
                        pass
                self.state = AppState.CONNECTING
                self.clicked = False
                return

            elif hover_close:
                if self.is_tracking:
                    self.log_session()
                    self.stop_recording()
                self.running = False
                return

            elif hover_end:
                if self.is_tracking:
                    self.log_session()
                    self.stop_recording()
                    self.is_tracking = False
                self.state = AppState.FINISHED
                self.clicked = False

        # --- Timer update ---
        now_wall = time.time()
        if self.is_tracking and not self.bag_playback_paused:
            self.elapsed_time += now_wall - self.last_frame_time
            if now_wall - self.last_autosave_time > AUTOSAVE_INTERVAL_S:
                self.log_session()
                self.last_autosave_time = now_wall
        self.last_frame_time = now_wall

        # --- Annotation ---
        display = self.draw_landmarks(color_image, result)

        if self.is_tracking and self.first_frame is None:
            self.first_frame = color_image.copy()

        if self.is_tracking and not self.bag_playback_paused:
            self.process_custom_metrics(result, int(now_wall * 1000), w, h, depth_frame)

        # Fading trail + position dot
        self._draw_trail(display)

        # XYZ coordinate readout
        self._draw_xyz_overlay(display, w, h)

        # --- Video write (centre crop) ---
        if self.is_tracking:
            if not self.video_writer:
                self.start_recording(w, h)
            m = int(w * SIDE_MARGIN_RATIO)
            self.video_writer.write(display[:, m:w - m])

        # --- UI overlay ---
        ui = display.copy()

        # Semi-transparent grey overlay on left/right panels
        overlay_buf = ui.copy()
        m = int(w * SIDE_MARGIN_RATIO)
        cv2.rectangle(overlay_buf, (0, 0),    (m, h),    (60, 60, 60), -1)
        cv2.rectangle(overlay_buf, (w-m, 0),  (w, h),    (60, 60, 60), -1)
        cv2.addWeighted(overlay_buf, 0.40, ui, 0.60, 0, ui)

        # Stats panel
        self.draw_live_stats(ui)

        # Buttons
        start_lbl = "STOP" if self.is_tracking else "START"
        self.draw_button(ui, start_rect, start_lbl, hover_start, is_active=self.is_tracking)
        self.draw_button(ui, reset_rect, "RESET",   hover_reset)
        self.draw_button(ui, close_rect, "X",       hover_close)
        self.draw_button(ui, back_rect,  "<",       hover_back)
        self.draw_button(ui, end_rect,   "End Session", hover_end)

        # --- Participant ID slider ---
        is_locked = self.is_tracking
        ctrl_y = 80 + (btn_h + 20) * 2 + 18
        slider_h_px = 8
        slider_y = ctrl_y + 18

        # ID label
        id_lbl = f"{SESSION_ID_PREFIX}-{self.user_id_input}"
        font = cv2.FONT_HERSHEY_PLAIN
        (lw, _), _ = cv2.getTextSize(id_lbl, font, 1.4, 2)
        lx = btn_x + (btn_w - lw) // 2
        cv2.putText(ui, id_lbl, (lx+1, ctrl_y+11), font, 1.4, UI_BTN_SHADOW, 2)
        cv2.putText(ui, id_lbl, (lx,   ctrl_y+10), font, 1.4, UI_TEXT, 2)

        # Track
        tr_col = UI_PANEL if is_locked else UI_BTN_IDLE
        cv2.circle(ui, (btn_x, slider_y + slider_h_px//2), slider_h_px//2, tr_col, -1)
        cv2.circle(ui, (btn_x + btn_w, slider_y + slider_h_px//2), slider_h_px//2, tr_col, -1)
        cv2.rectangle(ui, (btn_x, slider_y), (btn_x + btn_w, slider_y + slider_h_px), tr_col, -1)

        # Knob
        try:
            id_val = int(self.user_id_input)
        except ValueError:
            id_val = 0
        knob_x = int(btn_x + (id_val / MAX_PARTICIPANT_ID) * btn_w)
        knob_y = slider_y + slider_h_px // 2
        knob_col = UI_ACCENT if not is_locked else UI_TEXT_DIM
        cv2.circle(ui, (knob_x+1, knob_y+1), 8, UI_BTN_SHADOW, -1)
        cv2.circle(ui, (knob_x, knob_y), 8, knob_col, -1)
        cv2.circle(ui, (knob_x, knob_y), 5, UI_TEXT, -1)

        # Lock / drag hint
        hint = "[LOCKED]" if is_locked else "DRAG TO SET"
        hint_col = UI_TEXT_DIM
        (hw, _), _ = cv2.getTextSize(hint, font, 1.1, 1)
        hx = btn_x + (btn_w - hw) // 2
        cv2.putText(ui, hint, (hx+1, slider_y+30), font, 1.1, UI_BTN_SHADOW, 1)
        cv2.putText(ui, hint, (hx,   slider_y+29), font, 1.1, hint_col, 1)

        # --- Scan-type toggle (M / A) ---
        tog_y = ctrl_y + 88
        tog_w, tog_h = 55, 30
        man_rect_l = (btn_x,           tog_y, tog_w, tog_h)
        ai_rect_l  = (btn_x + tog_w + 10, tog_y, tog_w, tog_h)
        scan = getattr(self, 'live_scan_type', 'AI')
        self.draw_button(ui, man_rect_l, "M", False, is_active=(scan == "MAN"))
        self.draw_button(ui, ai_rect_l,  "A", False, is_active=(scan == "AI"))

        # Current bag filename (below toggles)
        if self.current_bag_file:
            fn = os.path.splitext(os.path.basename(self.current_bag_file))[0]
            fs2 = 1.0
            (fw, _), _ = cv2.getTextSize(fn, font, fs2, 1)
            while fw > (btn_w + 20) and fs2 > 0.5:
                fs2 -= 0.1
                (fw, _), _ = cv2.getTextSize(fn, font, fs2, 1)
            fx = btn_x + (btn_w - fw) // 2
            fy = tog_y + tog_h + 18
            cv2.putText(ui, fn, (fx+1, fy+1), font, fs2, UI_BTN_SHADOW, 1)
            cv2.putText(ui, fn, (fx, fy),     font, fs2, UI_TEXT_DIM, 1)

        # --- Pause/play button (bag files only) ---
        if self.current_bag_file:
            pb_sz = 30
            pb_x = 150
            pb_y = h - pb_sz - 15
            pb_col = UI_BTN_HOVER if self.is_inside_button((mx, my), (pb_x, pb_y), (pb_sz, pb_sz)) else UI_BTN_IDLE
            cv2.rectangle(ui, (pb_x, pb_y), (pb_x+pb_sz, pb_y+pb_sz), pb_col, -1)
            cv2.rectangle(ui, (pb_x, pb_y), (pb_x+pb_sz, pb_y+pb_sz), UI_BORDER, 2)
            ic = UI_ACCENT
            if self.bag_playback_paused:
                pts = np.array([[pb_x+10, pb_y+7], [pb_x+10, pb_y+23], [pb_x+22, pb_y+15]], np.int32)
                cv2.fillPoly(ui, [pts], ic)
            else:
                cv2.rectangle(ui, (pb_x+9, pb_y+7),  (pb_x+13, pb_y+23), ic, -1)
                cv2.rectangle(ui, (pb_x+17, pb_y+7), (pb_x+21, pb_y+23), ic, -1)
            if self.clicked and self.is_inside_button((mx, my), (pb_x, pb_y), (pb_sz, pb_sz)):
                self.bag_playback_paused = not self.bag_playback_paused
                self.clicked = False

        # --- Slider drag (only when not tracking) ---
        if not is_locked and self.mouse_down:
            if (btn_x - 10 <= mx <= btn_x + btn_w + 10
                    and slider_y - 10 <= my <= slider_y + 20):
                ratio = max(0.0, min(1.0, (mx - btn_x) / btn_w))
                new_id = int(ratio * MAX_PARTICIPANT_ID + 0.5)
                if new_id != self.id_slider_val:
                    self.id_slider_val = new_id
                    self.user_id_input = f"{new_id:02d}"

        # --- Scan-type click (only when not tracking) ---
        if not is_locked and self.clicked:
            if self.is_inside_button((mx, my), man_rect_l[:2], man_rect_l[2:]):
                self.live_scan_type = "MAN"
                self.clicked = False
            elif self.is_inside_button((mx, my), ai_rect_l[:2], ai_rect_l[2:]):
                self.live_scan_type = "AI"
                self.clicked = False

        cv2.imshow(self.window_name, ui)
        self.clicked = False  # Consume any unhandled click

    # ------------------------------------------------------------------
    # Shared visualisation helpers
    # ------------------------------------------------------------------

    def _draw_trail(self, img: np.ndarray) -> None:
        """Draw a fading position trail and current-position dot onto *img* (in place).

        The trail fades from a dimmer colour at the oldest point to the accent
        colour at the newest.  A three-ring dot marks the most recent position.
        """
        coords = self.all_hand_coords
        if not coords:
            return

        n = len(coords)
        trail_start = max(0, n - TRAIL_LENGTH)

        if n - trail_start > 1:
            for j in range(trail_start, n - 1):
                p1, p2 = coords[j], coords[j + 1]
                alpha = (j - trail_start) / TRAIL_LENGTH
                # Interpolate: dim → accent colour as trail ages from tail to head
                color = (
                    int(50  * (1 - alpha) + 200 * alpha),
                    int(100 * (1 - alpha) + 185 * alpha),
                    int(150 * (1 - alpha) + 50  * alpha),
                )
                cv2.line(img, p1, p2, color, max(1, int(3 * alpha)))

        pt = coords[-1]
        cv2.circle(img, pt, 12, UI_ACCENT, 2)          # Outer ring
        cv2.circle(img, pt, 8,  UI_ACCENT, -1)          # Filled disc
        cv2.circle(img, pt, 3,  UI_TEXT,   -1)          # White centre highlight

    def _draw_xyz_overlay(self, img: np.ndarray, w: int, h: int) -> None:
        """Render a fixed-width XYZ coordinate readout bar at the bottom of *img*.

        Uses a fixed-width box so the layout does not shift when values change.
        Shows "–.–––" placeholders when no position is available.
        """
        box_w, box_h = 320, 18
        box_x = w // 2 - box_w // 2
        box_y = h - 22

        if self.prev_position_3d is not None:
            xv, yv, zv = self.prev_position_3d
            txt = f"X: {xv:+7.3f}  Y: {yv:+7.3f}  Z: {zv:+7.3f}"
        else:
            txt = "X:  -.---  Y:  -.---  Z:  -.---"

        cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), UI_PANEL, -1)
        cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), UI_BORDER, 1)

        font = cv2.FONT_HERSHEY_PLAIN
        (tw, _), _ = cv2.getTextSize(txt, font, 1.0, 1)
        cv2.putText(img, txt, (box_x + (box_w - tw) // 2, box_y + 14),
                    font, 1.0, UI_TEXT, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        model_file = os.path.join(RESOURCE_PATH, "hand_landmarker.task")
        if not os.path.exists(model_file):
            print(f"[ERROR] Model not found: {model_file}")
            print(f"        Place hand_landmarker.task next to this script or executable.")
            if getattr(sys, 'frozen', False):
                input("Press Enter to exit...")
            sys.exit(1)

        app = App()
        app.run()

    except Exception as exc:
        print(f"\n=== UNHANDLED EXCEPTION ===")
        print(f"{type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        if getattr(sys, 'frozen', False):
            input("\nPress Enter to exit...")
        sys.exit(1)
