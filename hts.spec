# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec file — Hand Tracking System
# =============================================
# Builds a single-file Windows executable that bundles:
#   - hts.py  (the application)
#   - hand_landmarker.task  (MediaPipe model — REQUIRED, ~7.5 MB)
#   - app_icon.ico  (optional window/taskbar icon)
#   - all mediapipe runtime data and binaries
#
# Prerequisites
# -------------
#   pip install pyinstaller
#   # Place hand_landmarker.task in the same directory as this spec file before building.
#   # Optionally place app_icon.ico there too.
#
# Build
# -----
#   pyinstaller hts.spec
#
# Output
# ------
#   dist\HTS.exe   — self-contained, no Python required on the target machine
#   Outputs (CSV, video clips, heatmaps) are written next to the .exe at runtime.

import os

# Root directory of this spec file — all relative paths resolve from here.
spec_root = os.path.dirname(os.path.abspath(SPEC))

# ---------------------------------------------------------------------------
# Required data files
# ---------------------------------------------------------------------------

datas = []

# MediaPipe model file — mandatory.
model_file = os.path.join(spec_root, 'hand_landmarker.task')
if not os.path.exists(model_file):
    raise FileNotFoundError(
        "\n\n  hand_landmarker.task not found next to the spec file.\n"
        "  Download it from:\n"
        "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task\n"
        "  and place it in:  " + spec_root + "\n"
    )
datas.append((model_file, '.'))

# App icon — optional.  Bundled only if present; the app runs fine without it.
icon_file = os.path.join(spec_root, 'app_icon.ico')
icon_arg = icon_file if os.path.exists(icon_file) else None

# ---------------------------------------------------------------------------
# Collect all mediapipe runtime data, binaries, and hidden imports
# ---------------------------------------------------------------------------

from PyInstaller.utils.hooks import collect_all

mediapipe_datas, mediapipe_binaries, mediapipe_hiddenimports = collect_all('mediapipe')

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

a = Analysis(
    [os.path.join(spec_root, 'hts.py')],
    pathex=[spec_root],
    binaries=mediapipe_binaries,
    datas=datas + mediapipe_datas,
    hiddenimports=[
        'mediapipe',
        'mediapipe.tasks',
        'mediapipe.tasks.python',
        'mediapipe.tasks.python.core',
        'mediapipe.tasks.c',
        'cv2',
        'numpy',
        'pyrealsense2',
    ] + mediapipe_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='HTS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,       # No console window — use True only for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_arg,       # None if app_icon.ico is absent — perfectly fine
)
