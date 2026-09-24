# -*- coding: utf-8 -*-
"""
LineFormer Streamlit App
Chart line data extraction using LineFormer.
Automatic axis detection using ChartDete + OCR.
Output compatible with StarryDigitizer and WebPlotDigitizer formats.
"""

import sys
import os

# Project root is parent of src/
_src_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_src_dir)

# Add ChartDete submodule to path FIRST (highest priority for custom mmdet models)
sys.path.insert(0, os.path.join(_project_root, 'submodules', 'chartdete'))

# Add lineformer submodule to path
sys.path.insert(0, os.path.join(_project_root, 'submodules', 'lineformer'))

# Import mmdet (uses ChartDete's mmdet which includes custom models like CascadeRoIHead_LGF)
import mmdet  # noqa: F401
from mmdet.models.roi_heads.cascade_roi_head_LGF import CascadeRoIHead_LGF  # noqa: F401

# Add src to path for local imports
sys.path.insert(0, _src_dir)

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import hashlib
import copy
import json
import io
import base64
import zipfile
import tarfile
from datetime import datetime

try:
    import plotly.graph_objects as go
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:  # noqa: BLE001
    PLOTLY_EVENTS_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="AutoLineDigitizer",
    page_icon="📈",
    layout="wide"
)


@st.cache_resource
def load_lineformer_model():
    """Load LineFormer model (cached)."""
    import infer

    # Model weights in models/, config in submodules/lineformer/
    CKPT = os.path.join(_project_root, "models", "iter_3000.pth")
    CONFIG = os.path.join(_project_root, "submodules", "lineformer", "lineformer_swin_t_config.py")
    DEVICE = "cpu"

    infer.load_model(CONFIG, CKPT, DEVICE)
    return infer


@st.cache_resource
def load_chartdete_model():
    """Load ChartDete model for axis detection (cached)."""
    import chartdete_infer
    chartdete_infer.load_chartdete_model(device='cpu')
    return chartdete_infer


def detect_axis_calibration(chartdete_module, img):
    """
    Detect chart elements and extract axis calibration using ChartDete + OCR.

    Returns:
        axis_config: dict with calibration data or None
        detections: raw detection results
        ocr_results: OCR results for labels
    """
    # Run ChartDete detection
    detections = chartdete_module.detect_chart_elements(img, score_thr=0.3)

    # Get axis info with OCR
    axis_info = chartdete_module.get_axis_info(detections, img=img, with_ocr=True)

    calibration = axis_info.get('calibration')
    ocr_results = axis_info.get('ocr_results', {})

    if calibration is None:
        return None, detections, ocr_results

    # Convert calibration to axis_config format
    # For x-axis: use xlabel center positions (x pixel, fixed y at label position)
    # For y-axis: use ylabel center positions (fixed x at label position, y pixel)
    axis_config = None

    has_x = 'x1_pixel' in calibration and 'x2_pixel' in calibration
    has_y = 'y1_pixel' in calibration and 'y2_pixel' in calibration

    if has_x and has_y:
        # Get plot_area to determine y position for x-axis calibration points
        plot_area = axis_info.get('plot_area')

        if plot_area:
            # x calibration: use bottom of plot area for y
            x_calib_y = plot_area[3]  # y2 of plot_area (bottom)
            # y calibration: use left of plot area for x
            y_calib_x = plot_area[0]  # x1 of plot_area (left)
        else:
            # Fallback: use image dimensions
            x_calib_y = img.shape[0] * 0.9
            y_calib_x = img.shape[1] * 0.1

        # Note: In calibration from OCR:
        #   y1_pixel/y1_value = top label (higher Y pixel, but could be higher or lower value)
        #   y2_pixel/y2_value = bottom label (lower Y pixel)
        # In StarryDigitizer/WPD format:
        #   y1 = bottom point (lower Y pixel = higher on screen)
        #   y2 = top point (higher Y pixel = lower on screen)
        # So we swap y1 and y2 from OCR calibration

        axis_config = {
            # X axis calibration points (at bottom of chart)
            "x1_px": calibration['x1_pixel'],
            "x1_py": x_calib_y,
            "x1_val": calibration['x1_value'],
            "x2_px": calibration['x2_pixel'],
            "x2_py": x_calib_y,
            "x2_val": calibration['x2_value'],
            # Y axis calibration points (at left of chart)
            # y1 = bottom (higher pixel Y), y2 = top (lower pixel Y)
            "y1_px": y_calib_x,
            "y1_py": calibration['y2_pixel'],  # bottom label
            "y1_val": calibration['y2_value'],
            "y2_px": y_calib_x,
            "y2_py": calibration['y1_pixel'],  # top label
            "y2_val": calibration['y1_value'],
            "xIsLogScale": False,
            "yIsLogScale": False,
        }

    return axis_config, detections, ocr_results


def downsample_points(points, mode, fixed_step, max_points):
    """Downsample points based on mode."""
    if len(points) <= 1:
        return points

    if mode == "none":
        return points
    elif mode == "fixed":
        return points[::fixed_step]
    elif mode == "max_points":
        if len(points) <= max_points:
            return points
        step = max(1, len(points) // max_points)
        return points[::step]

    return points


def sort_data_series(data_series, sort_mode):
    """Sort data series based on the specified mode."""
    if sort_mode == "original" or len(data_series) == 0:
        return data_series

    if sort_mode == "mean_y_desc":
        # Sort by mean Y descending (higher Y value = lower on screen in image coords)
        # For chart interpretation: lower Y pixel = higher value, so desc means high→low value
        return sorted(data_series, key=lambda s: np.mean([pt[1] for pt in s["points"]]))
    elif sort_mode == "mean_y_asc":
        # Sort by mean Y ascending
        return sorted(data_series, key=lambda s: np.mean([pt[1] for pt in s["points"]]), reverse=True)

    return data_series


def extract_lines(infer_module, img, downsample_mode, fixed_step, max_points):
    """Extract line data from image."""
    line_dataseries = infer_module.get_dataseries(img, to_clean=False)

    data_series = []
    for line in line_dataseries:
        if len(line) == 0:
            continue

        # Extract all points
        all_points = [[int(pt['x']), int(pt['y'])] for pt in line]

        # Downsample
        points = downsample_points(all_points, downsample_mode, fixed_step, max_points)

        data_series.append({"points": points})

    return data_series, line_dataseries


def _pixel_to_data(px, py, axis_config):
    """Convert image pixel coords to data-space (x, y) using the calibration."""
    if not axis_config:
        return px, py
    x1p, x2p = float(axis_config['x1_px']), float(axis_config['x2_px'])
    y1p, y2p = float(axis_config['y1_py']), float(axis_config['y2_py'])
    x1v, x2v = float(axis_config['x1_val']), float(axis_config['x2_val'])
    y1v, y2v = float(axis_config['y1_val']), float(axis_config['y2_val'])
    dx_px = (x2p - x1p) or 1.0
    dy_px = (y2p - y1p) or 1.0
    x = x1v + (float(px) - x1p) * (x2v - x1v) / dx_px
    y = y1v + (float(py) - y1p) * (y2v - y1v) / dy_px
    return x, y


def _data_to_pixel(x, y, axis_config):
    """Inverse of _pixel_to_data — data-space (x, y) → image pixel coords."""
    if not axis_config:
        return float(x), float(y)
    x1p, x2p = float(axis_config['x1_px']), float(axis_config['x2_px'])
    y1p, y2p = float(axis_config['y1_py']), float(axis_config['y2_py'])
    x1v, x2v = float(axis_config['x1_val']), float(axis_config['x2_val'])
    y1v, y2v = float(axis_config['y1_val']), float(axis_config['y2_val'])
    dx_v = (x2v - x1v) or 1.0
    dy_v = (y2v - y1v) or 1.0
    px = x1p + (float(x) - x1v) * (x2p - x1p) / dx_v
    py = y1p + (float(y) - y1v) * (y2p - y1p) / dy_v
    return px, py


def draw_points_on_image(img, data_series, axis_config=None,
                         show_calibration=True, show_calibration_values=False,
                         show_line_numbers=False, highlight_idx=None,
                         line_indices=None, total_lines=None):
    """Draw extracted points on the image.

    line_indices: optional list mapping each entry of data_series back to its
      original line number (so filtering to one line keeps its color/number).
    total_lines: original series count, used to pick colors consistently.
    highlight_idx: if set, drawn line is emphasized (larger markers).
    """
    import line_utils

    result_img = img.copy()
    n = total_lines if total_lines is not None else len(data_series)
    palette = list(line_utils.get_distinct_colors(max(1, n)))
    markers = [
        cv2.MARKER_CROSS,
        cv2.MARKER_DIAMOND,
        cv2.MARKER_SQUARE,
        cv2.MARKER_TRIANGLE_UP,
        cv2.MARKER_TRIANGLE_DOWN,
        cv2.MARKER_STAR,
    ]

    for local_idx, series in enumerate(data_series):
        orig_idx = line_indices[local_idx] if line_indices is not None else local_idx
        color = palette[orig_idx % len(palette)]
        marker = markers[orig_idx % len(markers)]
        emph = (highlight_idx is not None and orig_idx == highlight_idx)
        size = 14 if emph else 8
        thick = 3 if emph else 2

        for pt in series["points"]:
            x, y = int(pt[0]), int(pt[1])
            cv2.drawMarker(result_img, (x, y), color, marker,
                           markerSize=size, thickness=thick)

        # Number label near the first point of the line.
        if show_line_numbers and series["points"]:
            fx, fy = int(series["points"][0][0]), int(series["points"][0][1])
            label = str(orig_idx + 1)
            H, W = result_img.shape[:2]
            fscale = max(0.5, min(W, H) / 1400.0)
            (tw, th), _bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                            fscale, 2)
            lx = max(2, min(W - tw - 2, fx + 6))
            ly = max(th + 2, min(H - 2, fy - 6))
            cv2.rectangle(result_img, (lx - 2, ly - th - 2),
                          (lx + tw + 2, ly + 2), (255, 255, 255), -1)
            cv2.rectangle(result_img, (lx - 2, ly - th - 2),
                          (lx + tw + 2, ly + 2), (0, 0, 0), 1)
            cv2.putText(result_img, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, fscale, color, 2, cv2.LINE_AA)

    # Draw axis calibration points if available
    if axis_config is not None and show_calibration:
        H, W = result_img.shape[:2]
        # Scale visual weight with image size but never so thin that dashes
        # disappear at web-display resolution.
        s = max(0.5, min(W, H) / 1200.0)
        calib_color = (255, 0, 255)
        outline_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45 * s
        line_thick = max(2, int(round(2 * s)))    # axis dashes: keep visible
        text_thick = max(1, int(round(1.2 * s)))  # text remains lighter
        marker_r_outer = max(7, int(round(9 * s)))
        marker_r_inner = max(5, int(round(7 * s)))
        cross_arm = max(3, int(round(4 * s)))
        label_pad = max(2, int(round(3 * s)))

        def draw_text_with_bg(img, text, pos, color):
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thick)
            x, y = pos
            x = max(label_pad, min(W - text_w - label_pad, x))
            y = max(text_h + label_pad, min(H - baseline - label_pad, y))
            cv2.rectangle(img, (x - label_pad, y - text_h - label_pad),
                          (x + text_w + label_pad, y + baseline + label_pad),
                          (255, 255, 255), -1)
            cv2.rectangle(img, (x - label_pad, y - text_h - label_pad),
                          (x + text_w + label_pad, y + baseline + label_pad),
                          outline_color, 1)
            cv2.putText(img, text, (x, y), font, font_scale, color, text_thick, cv2.LINE_AA)

        def draw_calib_point(img, x, y, color, label, side):
            # Marker only — the value labels obscure the printed tick numbers
            # in the corners of typical scientific plots. Turn on
            # show_calibration_values if you need them burned into the image.
            cv2.circle(img, (x, y), marker_r_outer, outline_color, 2)
            cv2.circle(img, (x, y), marker_r_inner, color, -1)
            cv2.line(img, (x - cross_arm, y), (x + cross_arm, y), outline_color, 2)
            cv2.line(img, (x, y - cross_arm), (x, y + cross_arm), outline_color, 2)
            if not show_calibration_values:
                return
            (tw, th), _bl = cv2.getTextSize(label, font, font_scale, text_thick)
            gap = max(6, int(round(8 * s)))
            if side == "left":
                lx = max(label_pad, x - marker_r_outer - gap - tw)
                ly = y + th // 2
            elif side == "right":
                lx = min(W - tw - label_pad, x + marker_r_outer + gap)
                ly = y + th // 2
            elif side == "above":
                lx = x - tw // 2
                ly = max(th + label_pad, y - marker_r_outer - gap)
            else:  # below
                lx = x - tw // 2
                ly = min(H - _bl - label_pad, y + marker_r_outer + gap + th)
            draw_text_with_bg(img, label, (lx, ly), color)

        x1_x, x1_y = int(axis_config['x1_px']), int(axis_config['x1_py'])
        x2_x, x2_y = int(axis_config['x2_px']), int(axis_config['x2_py'])
        y1_x, y1_y = int(axis_config['y1_px']), int(axis_config['y1_py'])
        y2_x, y2_y = int(axis_config['y2_px']), int(axis_config['y2_py'])

        # Dashed calibration axes — kept visible at any scale.
        dash_length = max(8, int(round(10 * s)))
        gap_length = max(4, int(round(5 * s)))
        for (ax1, ay1, ax2, ay2) in [(x1_x, x1_y, x2_x, x2_y),
                                     (y1_x, y1_y, y2_x, y2_y)]:
            dx, dy = ax2 - ax1, ay2 - ay1
            dist = max(1, int(np.sqrt(dx * dx + dy * dy)))
            for i in range(0, dist, dash_length + gap_length):
                sx = int(ax1 + dx * i / dist)
                sy = int(ay1 + dy * i / dist)
                ei = min(i + dash_length, dist)
                ex = int(ax1 + dx * ei / dist)
                ey = int(ay1 + dy * ei / dist)
                cv2.line(result_img, (sx, sy), (ex, ey), calib_color, line_thick)

        # Compact labels (drop the "X1="/"Y1=" prefix — the position tells you which)
        def _fmt(v):
            try:
                fv = float(v)
                return f"{fv:g}"
            except (TypeError, ValueError):
                return str(v)
        draw_calib_point(result_img, x1_x, x1_y, calib_color,
                         _fmt(axis_config['x1_val']), "below")
        draw_calib_point(result_img, x2_x, x2_y, calib_color,
                         _fmt(axis_config['x2_val']), "below")
        draw_calib_point(result_img, y1_x, y1_y, calib_color,
                         _fmt(axis_config['y1_val']), "left")
        draw_calib_point(result_img, y2_x, y2_y, calib_color,
                         _fmt(axis_config['y2_val']), "left")

    return result_img


def convert_to_starry_digitizer_format(data_series, img_shape, axis_config=None):
    """
    Convert LineFormer output to StarryDigitizer project.json format.

    axis_config: dict with x1, x2, y1, y2 pixel coordinates and values
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # Default axis set (user needs to calibrate in StarryDigitizer)
    if axis_config is None:
        # Use image corners as default
        axis_set = {
            "id": 1,
            "name": "XY Axes 1",
            "x1": {
                "name": "x1",
                "value": 0,
                "coord": {"xPx": 0, "yPx": float(img_shape[0])}
            },
            "x2": {
                "name": "x2",
                "value": 100,
                "coord": {"xPx": float(img_shape[1]), "yPx": float(img_shape[0])}
            },
            "y1": {
                "name": "y1",
                "value": 0,
                "coord": {"xPx": 0, "yPx": float(img_shape[0])}
            },
            "y2": {
                "name": "y2",
                "value": 100,
                "coord": {"xPx": 0, "yPx": 0}
            },
            "xIsLogScale": False,
            "yIsLogScale": False,
            "considerGraphTilt": False,
            "pointMode": 0,
            "isVisible": True
        }
    else:
        axis_set = {
            "id": 1,
            "name": "XY Axes 1",
            "x1": {
                "name": "x1",
                "value": axis_config["x1_val"],
                "coord": {"xPx": axis_config["x1_px"], "yPx": axis_config["x1_py"]}
            },
            "x2": {
                "name": "x2",
                "value": axis_config["x2_val"],
                "coord": {"xPx": axis_config["x2_px"], "yPx": axis_config["x2_py"]}
            },
            "y1": {
                "name": "y1",
                "value": axis_config["y1_val"],
                "coord": {"xPx": axis_config["y1_px"], "yPx": axis_config["y1_py"]}
            },
            "y2": {
                "name": "y2",
                "value": axis_config["y2_val"],
                "coord": {"xPx": axis_config["y2_px"], "yPx": axis_config["y2_py"]}
            },
            "xIsLogScale": axis_config.get("xIsLogScale", False),
            "yIsLogScale": axis_config.get("yIsLogScale", False),
            "considerGraphTilt": False,
            "pointMode": 0,
            "isVisible": True
        }

    # Convert datasets
    datasets = []

    # Add empty dataset 1 (StarryDigitizer convention)
    datasets.append({
        "id": 1,
        "name": "dataset 1",
        "axisSetId": 1,
        "points": [],
        "visiblePointIds": [],
        "manuallyAddedPointIds": []
    })

    # Add extracted lines
    for idx, series in enumerate(data_series):
        points = []
        visible_ids = []
        for pt_idx, pt in enumerate(series["points"]):
            pt_id = pt_idx + 1
            points.append({
                "id": pt_id,
                "xPx": float(pt[0]),
                "yPx": float(pt[1])
            })
            visible_ids.append(pt_id)

        datasets.append({
            "id": idx + 2,  # Start from 2 (1 is empty dataset)
            "name": f"Line {idx + 1}",
            "axisSetId": 1,
            "points": points,
            "visiblePointIds": visible_ids,
            "manuallyAddedPointIds": []
        })

    project = {
        "version": "1.11.2",
        "timestamp": timestamp,
        "axisSets": [axis_set],
        "activeAxisSetId": 1,
        "datasets": datasets,
        "activeDatasetId": len(datasets),
        "canvasHandler": {
            "scale": 1.0,
            "manualMode": 0
        }
    }

    return project


def create_starry_digitizer_zip(img, project_json):
    """
    Create a ZIP file containing image.png and project.json
    for StarryDigitizer import.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add image.png
        _, img_encoded = cv2.imencode('.png', img)
        zf.writestr('image.png', img_encoded.tobytes())

        # Add project.json
        json_str = json.dumps(project_json, indent=2, ensure_ascii=False)
        zf.writestr('project.json', json_str.encode('utf-8'))

    zip_buffer.seek(0)
    return zip_buffer


def convert_to_wpd_format(data_series, img_shape, axis_config=None):
    """
    Convert LineFormer output to WebPlotDigitizer JSON format.

    This format can be loaded in WebPlotDigitizer after loading the image.
    """
    # Default calibration points (user needs to recalibrate in WPD)
    if axis_config is None:
        calibration_points = [
            {"px": 0.0, "py": float(img_shape[0]), "dx": "0", "dy": "0", "dz": None},
            {"px": float(img_shape[1]), "py": float(img_shape[0]), "dx": "100", "dy": "0", "dz": None},
            {"px": 0.0, "py": float(img_shape[0]), "dx": "0", "dy": "0", "dz": None},
            {"px": 0.0, "py": 0.0, "dx": "0", "dy": "100", "dz": None}
        ]
        is_log_x = False
        is_log_y = False
    else:
        calibration_points = [
            {"px": axis_config["x1_px"], "py": axis_config["x1_py"],
             "dx": str(axis_config["x1_val"]), "dy": str(axis_config["y1_val"]), "dz": None},
            {"px": axis_config["x2_px"], "py": axis_config["x2_py"],
             "dx": str(axis_config["x2_val"]), "dy": str(axis_config["y1_val"]), "dz": None},
            {"px": axis_config["y1_px"], "py": axis_config["y1_py"],
             "dx": str(axis_config["x1_val"]), "dy": str(axis_config["y1_val"]), "dz": None},
            {"px": axis_config["y2_px"], "py": axis_config["y2_py"],
             "dx": str(axis_config["x1_val"]), "dy": str(axis_config["y2_val"]), "dz": None}
        ]
        is_log_x = axis_config.get("xIsLogScale", False)
        is_log_y = axis_config.get("yIsLogScale", False)

    axes_coll = [{
        "name": "XY",
        "type": "XYAxes",
        "isLogX": is_log_x,
        "isLogY": is_log_y,
        "noRotation": False,
        "calibrationPoints": calibration_points
    }]

    # Convert datasets
    dataset_coll = []

    # Add empty default dataset (WPD convention)
    dataset_coll.append({
        "name": "Default Dataset",
        "axesName": "XY",
        "colorRGB": [200, 0, 0, 255],
        "metadataKeys": [],
        "data": [],
        "autoDetectionData": None
    })

    # Add extracted lines
    for idx, series in enumerate(data_series):
        data_points = []
        for pt in series["points"]:
            # WPD format: x, y are pixel coords, value is [realX, realY] (null if not calibrated)
            data_points.append({
                "x": float(pt[0]),
                "y": float(pt[1]),
                "value": None  # Will be calculated by WPD after calibration
            })

        dataset_coll.append({
            "name": f"Dataset {idx + 1}",
            "axesName": "XY",
            "colorRGB": [200, 0, 0, 255],
            "metadataKeys": [],
            "data": data_points,
            "autoDetectionData": None
        })

    wpd_json = {
        "version": [4, 2],
        "axesColl": axes_coll,
        "datasetColl": dataset_coll,
        "measurementColl": []
    }

    return wpd_json


def create_wpd_tar(img, wpd_json, project_name="project"):
    """
    Create a TAR file for WebPlotDigitizer import.

    WPD expects TAR structure:
      projectName/
      projectName/info.json
      projectName/wpd.json
      projectName/image.png
    """
    import time
    tar_buffer = io.BytesIO()
    mtime = time.time()

    with tarfile.open(fileobj=tar_buffer, mode='w') as tf:
        # Add project folder
        folder_info = tarfile.TarInfo(name=f'{project_name}/')
        folder_info.type = tarfile.DIRTYPE
        folder_info.mtime = mtime
        tf.addfile(folder_info)

        # Add info.json
        info_json = {
            "version": [4, 0],
            "json": "wpd.json",
            "images": ["image.png"]
        }
        info_bytes = json.dumps(info_json, ensure_ascii=False).encode('utf-8')
        info_tarinfo = tarfile.TarInfo(name=f'{project_name}/info.json')
        info_tarinfo.size = len(info_bytes)
        info_tarinfo.mtime = mtime
        tf.addfile(info_tarinfo, io.BytesIO(info_bytes))

        # Add wpd.json
        wpd_bytes = json.dumps(wpd_json, ensure_ascii=False).encode('utf-8')
        wpd_tarinfo = tarfile.TarInfo(name=f'{project_name}/wpd.json')
        wpd_tarinfo.size = len(wpd_bytes)
        wpd_tarinfo.mtime = mtime
        tf.addfile(wpd_tarinfo, io.BytesIO(wpd_bytes))

        # Add image.png
        _, img_encoded = cv2.imencode('.png', img)
        img_bytes = img_encoded.tobytes()
        img_tarinfo = tarfile.TarInfo(name=f'{project_name}/image.png')
        img_tarinfo.size = len(img_bytes)
        img_tarinfo.mtime = mtime
        tf.addfile(img_tarinfo, io.BytesIO(img_bytes))

    tar_buffer.seek(0)
    return tar_buffer


def _render_sidebar():
    """Render sidebar settings shared across all tabs; return a config dict."""
    st.sidebar.header("Settings")
    show_visualization = st.sidebar.checkbox("Show visualization", value=True)

    st.sidebar.subheader("Axis Detection")
    auto_axis = st.sidebar.checkbox(
        "Auto-detect axis (ChartDete + OCR)", value=True,
        help="Automatically detect axis labels and calibration",
    )
    show_calibration = st.sidebar.checkbox(
        "Show calibration overlay on chart", value=True,
        help="Draws the four calibration points + dashed axes on the result "
             "image so you can see where the auto-detection landed.",
    )
    show_calibration_values = st.sidebar.checkbox(
        "Burn calibration values into image", value=False,
        help="Also writes the detected values (e.g. '3.0', '1200') next to "
             "each marker. Off by default because the values are already "
             "printed on the axis — turn ON to include them in the exported "
             "visualization.",
        disabled=not show_calibration,
    )

    st.sidebar.subheader("Line Sorting")
    sort_mode = st.sidebar.selectbox(
        "Sort by",
        options=["original", "mean_y_desc", "mean_y_asc"],
        format_func=lambda x: {
            "original": "Original (Detection Order)",
            "mean_y_desc": "Mean Y (High → Low)",
            "mean_y_asc": "Mean Y (Low → High)",
        }[x],
    )

    st.sidebar.subheader("Downsampling")
    downsample_mode = st.sidebar.selectbox(
        "Mode", options=["max_points", "fixed", "none"], index=0,
        help="max_points: Limit points per line, fixed: Every N points, none: All points",
    )
    if downsample_mode == "fixed":
        fixed_step = st.sidebar.slider("Fixed step (every N points)", 1, 50, 10)
        max_points = 50
    elif downsample_mode == "max_points":
        max_points = st.sidebar.slider("Max points per line", 10, 200, 50)
        fixed_step = 10
    else:
        fixed_step, max_points = 10, 50

    return dict(
        show_visualization=show_visualization,
        auto_axis=auto_axis,
        show_calibration=show_calibration,
        show_calibration_values=show_calibration_values,
        sort_mode=sort_mode,
        downsample_mode=downsample_mode,
        fixed_step=fixed_step,
        max_points=max_points,
    )


def _load_models(config):
    """Load LineFormer (required) + ChartDete (optional); return both."""
    with st.spinner("Loading LineFormer model..."):
        try:
            infer_module = load_lineformer_model()
            st.sidebar.success("LineFormer loaded!")
        except Exception as e:
            st.error(f"Failed to load LineFormer: {e}")
            st.stop()

    chartdete_module = None
    if config["auto_axis"]:
        with st.spinner("Loading ChartDete model..."):
            try:
                chartdete_module = load_chartdete_model()
                st.sidebar.success("ChartDete loaded!")
            except Exception as e:
                st.warning(f"ChartDete not available: {e}")
                config["auto_axis"] = False
    return infer_module, chartdete_module


def _get_input_image(key_prefix="single"):
    """Render file uploader + paste button; return (bgr ndarray, name) or (None, None)."""
    upload_col, paste_col = st.columns([3, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a chart image (or paste from clipboard →)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key=f"{key_prefix}_uploader",
        )
    with paste_col:
        st.write("")
        try:
            from streamlit_paste_button import paste_image_button
            paste_result = paste_image_button(
                label="📋 Paste (Cmd+V)",
                key=f"{key_prefix}_paste",
                errors="ignore",
            )
        except Exception as _paste_err:  # noqa: BLE001
            paste_result = None
            st.caption(f"Paste unavailable: {_paste_err}")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img, uploaded_file.name
    if paste_result is not None and paste_result.image_data is not None:
        pil_img = paste_result.image_data
        img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        return img, f"clipboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    return None, None


def _render_visual_editor(img, edited_series, axis_config, selected_idx,
                          ss_key, img_key, palette):
    """Interactive point editor: chart image as plotly background, click to
    add/delete points on the selected line. Requires streamlit-plotly-events."""
    if not PLOTLY_EVENTS_AVAILABLE:
        st.info("Visual editor requires `plotly` + `streamlit-plotly-events` "
                "(both installed in this env — reload if you don't see it).")
        return

    H, W = img.shape[:2]

    st.markdown("**Visual editor** — click the chart to add/delete points on Line "
                f"**{selected_idx + 1}**.")
    mode_col, undo_col = st.columns([3, 1])
    with mode_col:
        mode = st.radio(
            "Click mode",
            ["👁 View only", "➕ Add point", "❌ Delete nearest point"],
            horizontal=True, key=f"vismode_{img_key}_{selected_idx}",
        )
    with undo_col:
        st.write("")
        if st.button("↩︎ Undo last visual edit",
                     key=f"vis_undo_{img_key}_{selected_idx}",
                     disabled=f"vis_undo_stack_{img_key}_{selected_idx}"
                              not in st.session_state):
            stack_key = f"vis_undo_stack_{img_key}_{selected_idx}"
            if st.session_state.get(stack_key):
                prev = st.session_state[stack_key].pop()
                st.session_state[ss_key][selected_idx]["points"] = prev
                st.rerun()

    # Encode image as data URL for plotly background.
    _ok, buf = cv2.imencode(".png", img)
    if not _ok:
        st.error("Could not encode image for the plotly editor.")
        return
    img_b64 = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

    pts = edited_series[selected_idx]["points"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    line_color = "rgb({},{},{})".format(*palette[selected_idx % len(palette)][::-1])

    fig = go.Figure()
    fig.add_layout_image(
        source=img_b64, xref="x", yref="y",
        x=0, y=0, sizex=W, sizey=H,
        sizing="stretch", opacity=1.0, layer="below",
    )
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+lines",
        marker=dict(size=10, color=line_color,
                    line=dict(color="black", width=1)),
        line=dict(color=line_color, width=1.5),
        hovertemplate="pixel (%{x:.0f}, %{y:.0f})<extra></extra>",
        name=f"Line {selected_idx+1}",
    ))
    fig.update_xaxes(range=[0, W], visible=False, constrain="domain")
    fig.update_yaxes(range=[H, 0], visible=False, scaleanchor="x", scaleratio=1)
    # Compact layout — height matches other charts (55vh ≈ 500px).
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        showlegend=False,
        dragmode=False,
    )

    click_event = mode != "👁 View only"
    events = plotly_events(
        fig,
        click_event=click_event,
        override_height=520,
        key=f"vis_editor_{img_key}_{selected_idx}",
    )

    if events and click_event:
        cx = float(events[0]["x"])
        cy = float(events[0]["y"])
        stack_key = f"vis_undo_stack_{img_key}_{selected_idx}"
        st.session_state.setdefault(stack_key, [])
        st.session_state[stack_key].append(copy.deepcopy(pts))
        if len(st.session_state[stack_key]) > 20:
            st.session_state[stack_key] = st.session_state[stack_key][-20:]

        if mode.startswith("➕"):
            # Insert in x-sorted position so lines stay ordered.
            new_pts = list(pts) + [[int(round(cx)), int(round(cy))]]
            new_pts.sort(key=lambda p: p[0])
            st.session_state[ss_key][selected_idx]["points"] = new_pts
            st.rerun()
        elif mode.startswith("❌") and pts:
            # Find nearest point in pixel space and remove it.
            arr = np.array(pts, dtype=float)
            d2 = (arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2
            i = int(np.argmin(d2))
            new_pts = [p for j, p in enumerate(pts) if j != i]
            st.session_state[ss_key][selected_idx]["points"] = new_pts
            st.rerun()


def _render_single_image_pipeline(img, name, infer_module, chartdete_module, config):
    """Existing single-image workflow, refactored to accept a preloaded image."""
    show_visualization = config["show_visualization"]
    auto_axis = config["auto_axis"]
    sort_mode = config["sort_mode"]
    downsample_mode = config["downsample_mode"]
    fixed_step = config["fixed_step"]
    max_points = config["max_points"]

    status_placeholder = st.empty()
    if name.startswith("clipboard_"):
        status_placeholder.success(f"Pasted from clipboard ({img.shape[1]}x{img.shape[0]})")

    if True:

        # Initialize axis detection variables
        axis_config = None
        detections = None
        ocr_results = None

        # Display columns - show input image immediately
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Size: {img.shape[1]} x {img.shape[0]} pixels")

        with col2:
            # Create placeholders for dynamic updates
            st.subheader("Extraction Result")
            viz_placeholder = st.empty()
            summary_placeholder = st.empty()
            caption_placeholder = st.empty()

        # Placeholder for axis calibration results (outside columns)
        axis_placeholder = st.empty()

        # Step 1: Extract lines (faster) - with spinner
        with status_placeholder.container():
            with st.spinner("⏳ Extracting lines (LineFormer)..."):
                data_series, raw_lines = extract_lines(
                    infer_module, img, downsample_mode, fixed_step, max_points
                )

                # Sort lines
                data_series = sort_data_series(data_series, sort_mode)

        # ---- Session-state per image: preserve user edits across reruns ----
        img_bytes_key = hashlib.md5(img.tobytes()).hexdigest()[:12]
        ss_key = f"series_{img_bytes_key}"
        ax_key = f"axis_{img_bytes_key}"
        if ss_key not in st.session_state:
            st.session_state[ss_key] = copy.deepcopy(data_series)
        edited_series = st.session_state[ss_key]

        # Show initial result (without axis calibration) immediately
        if show_visualization:
            result_img = draw_points_on_image(
                img, edited_series, None,
                show_calibration=config.get("show_calibration", True),
                show_calibration_values=config.get("show_calibration_values", False),
                show_line_numbers=True,
            )
            viz_placeholder.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Show line summary
        total_points = sum(len(s['points']) for s in edited_series)
        line_pts = [len(s['points']) for s in edited_series]
        summary_placeholder.success(f"**{len(edited_series)} lines** detected ({total_points} points total)")
        caption_placeholder.caption(f"Points per line: {', '.join(map(str, line_pts))}")

        # Step 2: Run axis detection (slower, OCR-heavy) - with spinner
        if auto_axis and chartdete_module is not None:
            if ax_key not in st.session_state:
                with status_placeholder.container():
                    with st.spinner("🔍 Detecting axis labels (ChartDete + OCR)..."):
                        axis_config, detections, ocr_results = detect_axis_calibration(
                            chartdete_module, img
                        )
                st.session_state[ax_key] = (axis_config, detections, ocr_results)
            axis_config, detections, ocr_results = st.session_state[ax_key]
            status_placeholder.empty()
        else:
            status_placeholder.empty()

        # Show axis calibration results
        if axis_config is not None:
            with axis_placeholder.container():
                with st.expander("Axis Calibration (Auto-detected)", expanded=False):
                    col_x, col_y = st.columns(2)
                    with col_x:
                        st.markdown("**X-axis:**")
                        st.write(f"  {axis_config['x1_val']} → {axis_config['x2_val']}")
                    with col_y:
                        st.markdown("**Y-axis:**")
                        st.write(f"  {axis_config['y1_val']} → {axis_config['y2_val']}")

                    # Show OCR details
                    if ocr_results:
                        st.markdown("**Detected Labels:**")
                        ocr_text = []
                        if 'xlabels' in ocr_results:
                            x_vals = [f"{l['value']}" for l in ocr_results['xlabels'] if l['value'] is not None]
                            ocr_text.append(f"X: [{', '.join(x_vals)}]")
                        if 'ylabels' in ocr_results:
                            y_vals = [f"{l['value']}" for l in ocr_results['ylabels'] if l['value'] is not None]
                            ocr_text.append(f"Y: [{', '.join(y_vals)}]")
                        st.caption(' | '.join(ocr_text))
        elif auto_axis:
            axis_placeholder.warning("Could not auto-detect axis calibration. Manual calibration needed in WPD/StarryDigitizer.")

        # ---- ✦ Claude assistance ----
        api_key = (st.session_state.get("vlm_api_key", "")
                   or os.environ.get("ANTHROPIC_API_KEY", "")).strip()
        with st.expander("✦ Claude assistance (axis names, legend labels)",
                         expanded=False):
            if not api_key:
                st.caption("Set ANTHROPIC_API_KEY in the Claude + KMDS tab "
                           "(or as an env var) to unlock these buttons.")
            axis_props_key = f"axis_props_{img_bytes_key}"
            legend_key = f"legend_names_{img_bytes_key}"
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✦ Read axis names + units",
                             disabled=not api_key,
                             key=f"vlm_axes_{img_bytes_key}"):
                    with st.spinner("Claude reading axis properties…"):
                        try:
                            from vlm_verifier import VLMVerifier
                            v = VLMVerifier(api_key=api_key)
                            props = v.read_axis_properties(img)
                            st.session_state[axis_props_key] = props
                        except Exception as e:
                            st.error(f"Claude axis read failed: {e}")
                if axis_props_key in st.session_state:
                    st.write(st.session_state[axis_props_key])
            with c2:
                if st.button("✦ Label curves from legend",
                             disabled=not api_key,
                             key=f"vlm_legend_{img_bytes_key}"):
                    with st.spinner("Claude matching curves to legend entries…"):
                        try:
                            from vlm_verifier import VLMVerifier
                            v = VLMVerifier(api_key=api_key)
                            names = v.label_lines_by_legend(img, edited_series)
                            st.session_state[legend_key] = names
                        except Exception as e:
                            st.error(f"Claude legend labeling failed: {e}")
                if legend_key in st.session_state:
                    for i, nm in enumerate(st.session_state[legend_key]):
                        st.write(f"Line {i+1}: **{nm}**")

        # ---- Per-curve inspection & editing ----
        st.subheader("Curves")
        legend_names = st.session_state.get(f"legend_names_{img_bytes_key}", [])
        curve_labels = ["All curves"]
        for i, s in enumerate(edited_series):
            nm = (legend_names[i] if i < len(legend_names) and legend_names[i]
                  else None)
            label = f"Line {i+1}" + (f" — {nm}" if nm else "") + f" ({len(s['points'])} pts)"
            curve_labels.append(label)
        sel = st.selectbox(
            "Show", curve_labels, index=0,
            key=f"curve_sel_{img_bytes_key}",
            help="Pick a single line to isolate it in the visualization and "
                 "edit its X/Y points below.",
        )

        if sel == "All curves":
            viz_data = edited_series
            viz_indices = list(range(len(edited_series)))
            highlight = None
        else:
            idx = curve_labels.index(sel) - 1
            viz_data = [edited_series[idx]]
            viz_indices = [idx]
            highlight = idx

        # Re-render viz with selection + numbers baked in.
        if show_visualization:
            result_img = draw_points_on_image(
                img, viz_data, axis_config,
                show_calibration=config.get("show_calibration", True),
                show_calibration_values=config.get("show_calibration_values", False),
                show_line_numbers=True,
                highlight_idx=highlight,
                line_indices=viz_indices,
                total_lines=len(edited_series),
            )
            viz_placeholder.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                                  use_container_width=True)

        # Visual editor + XY table for the selected single curve.
        if sel != "All curves":
            idx = curve_labels.index(sel) - 1

            # Rebuild the color palette so the visual editor matches the
            # numbered chart above.
            import line_utils
            palette = list(line_utils.get_distinct_colors(max(1, len(edited_series))))

            with st.expander("🎯 Visual editor (click to add / delete points)",
                             expanded=True):
                _render_visual_editor(
                    img, edited_series, axis_config, idx,
                    ss_key, img_bytes_key, palette,
                )

            pts_px = edited_series[idx]["points"]
            if axis_config is not None:
                rows = [_pixel_to_data(p[0], p[1], axis_config) for p in pts_px]
                cols = ("X", "Y")
                cal_note = " (data-space, using detected calibration)"
            else:
                rows = [(float(p[0]), float(p[1])) for p in pts_px]
                cols = ("X_px", "Y_px")
                cal_note = " (pixel coords — no calibration detected)"
            df = pd.DataFrame(rows, columns=cols)
            st.caption(f"Line {idx+1} — {len(rows)} points{cal_note}. "
                       "Edit any cell, add rows at the bottom, or use the row "
                       "checkbox + Delete key to remove points.")
            edited_df = st.data_editor(
                df, num_rows="dynamic", use_container_width=True,
                key=f"editor_{img_bytes_key}_{idx}",
                column_config={c: st.column_config.NumberColumn(c, format="%.6g")
                               for c in cols},
            )

            new_pts = []
            for row in edited_df.itertuples(index=False):
                try:
                    x, y = float(row[0]), float(row[1])
                except (TypeError, ValueError):
                    continue
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                if axis_config is not None:
                    px, py = _data_to_pixel(x, y, axis_config)
                else:
                    px, py = x, y
                new_pts.append([int(round(px)), int(round(py))])

            btn_col1, btn_col2, _ = st.columns([1, 1, 3])
            with btn_col1:
                if st.button("💾 Apply edits", key=f"apply_{img_bytes_key}_{idx}",
                             type="primary"):
                    st.session_state[ss_key][idx]["points"] = new_pts
                    st.rerun()
            with btn_col2:
                if st.button("↩︎ Reset this line",
                             key=f"reset_{img_bytes_key}_{idx}"):
                    st.session_state[ss_key][idx]["points"] = copy.deepcopy(
                        data_series[idx]["points"])
                    st.rerun()

            if len(new_pts) != len(pts_px):
                st.info(f"Pending: {len(new_pts)} points "
                        f"(was {len(pts_px)}). Press **Apply edits** to save.")

        else:
            reset_col1, _ = st.columns([1, 4])
            with reset_col1:
                if st.button("↩︎ Reset all lines to auto-detected",
                             key=f"reset_all_{img_bytes_key}"):
                    st.session_state[ss_key] = copy.deepcopy(data_series)
                    st.rerun()

        # Downloads/exports use the *edited* series so user changes reach WPD/SD.
        # Build StarryDigitizer project
        project_json = convert_to_starry_digitizer_format(
            edited_series, img.shape, axis_config
        )

        # Build WebPlotDigitizer project
        wpd_json = convert_to_wpd_format(
            edited_series, img.shape, axis_config
        )

        # Create ZIP file for StarryDigitizer
        zip_buffer = create_starry_digitizer_zip(img, project_json)

        # Create TAR file for WebPlotDigitizer
        base_name = os.path.splitext(name)[0]
        tar_buffer = create_wpd_tar(img, wpd_json, project_name=base_name)

        # Generate filenames
        timestamp_str = datetime.now().strftime('%Y%m%d-%H%M%S')
        zip_filename = f"sd-{timestamp_str}.zip"
        tar_filename = f"wpd-{timestamp_str}.tar"

        # Download buttons
        st.subheader("Download")
        col_dl1, col_dl2, col_dl3 = st.columns(3)

        with col_dl1:
            st.download_button(
                label="📦 StarryDigitizer (.zip)",
                data=zip_buffer.getvalue(),
                file_name=zip_filename,
                mime="application/zip",
                help="ZIP containing image.png + project.json"
            )

        with col_dl2:
            st.download_button(
                label="📊 WebPlotDigitizer (.tar)",
                data=tar_buffer.getvalue(),
                file_name=tar_filename,
                mime="application/x-tar",
                help="TAR with project folder structure"
            )

        with col_dl3:
            if show_visualization:
                _, buffer = cv2.imencode('.png', result_img)
                st.download_button(
                    label="🖼️ Visualization (.png)",
                    data=buffer.tobytes(),
                    file_name=f"{base_name}_result.png",
                    mime="image/png"
                )

        # ---- CSV export of the edited curves ----
        csv_lines = ["line,x,y"]
        for li, s in enumerate(edited_series, 1):
            for p in s["points"]:
                if axis_config is not None:
                    x, y = _pixel_to_data(p[0], p[1], axis_config)
                else:
                    x, y = float(p[0]), float(p[1])
                csv_lines.append(f"{li},{x:.6g},{y:.6g}")
        csv_bytes = "\n".join(csv_lines).encode("utf-8")
        st.download_button(
            label=f"⬇️ Combined CSV (all {len(edited_series)} lines, "
                  f"{'data' if axis_config else 'pixel'} coords)",
            data=csv_bytes,
            file_name=f"{base_name}_curves.csv",
            mime="text/csv",
            key=f"csv_dl_{img_bytes_key}",
        )

        # Show JSON previews
        with st.expander("Preview StarryDigitizer project.json"):
            json_str = json.dumps(project_json, indent=2, ensure_ascii=False)
            if len(json_str) > 5000:
                st.code(json_str[:5000] + "\n... (truncated)", language="json")
            else:
                st.code(json_str, language="json")

        with st.expander("Preview WebPlotDigitizer wpd.json"):
            wpd_preview = json.dumps(wpd_json, indent=2, ensure_ascii=False)
            if len(wpd_preview) > 5000:
                st.code(wpd_preview[:5000] + "\n... (truncated)", language="json")
            else:
                st.code(wpd_preview, language="json")

        # Instructions
        st.info("""
        **StarryDigitizer:** Download ZIP → Open [StarryDigitizer](https://starrydigitizer.vercel.app/) → Load Project

        **WebPlotDigitizer:** Download TAR → Open [WPD](https://apps.automeris.io/wpd4/) → File → Load Project (.tar)
        """)


def single_image_tab(infer_module, chartdete_module, config):
    """Tab 1: single chart image → line extraction (the original workflow)."""
    st.markdown("Upload a chart image (or paste from the clipboard) to extract line data.")
    img, name = _get_input_image(key_prefix="single")
    if img is not None:
        _render_single_image_pipeline(img, name, infer_module, chartdete_module, config)


def pdf_gallery_tab(infer_module, chartdete_module, config):
    """Tab 2: upload a PDF, gallery-select figures, digitize per figure."""
    st.markdown("Upload a paper PDF — every chart figure is detected and shown as a gallery.")
    try:
        import pdf_figures  # noqa: F401
    except Exception as e:
        st.error(f"PDF support unavailable: {e}")
        return

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")
    if pdf_file is None:
        st.info("Waiting for a PDF…")
        return

    if ("pdf_bytes" not in st.session_state
            or st.session_state.get("pdf_name") != pdf_file.name):
        st.session_state["pdf_bytes"] = pdf_file.getvalue()
        st.session_state["pdf_name"] = pdf_file.name
        st.session_state.pop("pdf_figures_cache", None)

    if "pdf_figures_cache" not in st.session_state:
        with st.spinner("Extracting figures from PDF…"):
            import tempfile, pdf_figures as pf
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(st.session_state["pdf_bytes"])
                pdf_path = tmp.name
            try:
                figs = list(pf.extract_figures(pdf_path))
                st.session_state["pdf_figures_cache"] = figs
            except Exception as e:
                st.error(f"Failed to extract figures: {e}")
                return

    figs = st.session_state["pdf_figures_cache"]
    if not figs:
        st.warning("No chart figures were detected in this PDF.")
        return

    st.success(f"Detected {len(figs)} figure(s).")
    cols = st.columns(4)
    for i, (fig_bgr, meta) in enumerate(figs):
        with cols[i % 4]:
            st.image(cv2.cvtColor(fig_bgr, cv2.COLOR_BGR2RGB),
                     caption=f"#{i+1} p.{meta.get('page','?')}",
                     use_container_width=True)
            if st.button(f"Digitize #{i+1}", key=f"pdf_fig_btn_{i}"):
                st.session_state["pdf_selected_idx"] = i

    sel = st.session_state.get("pdf_selected_idx")
    if sel is not None and 0 <= sel < len(figs):
        st.divider()
        st.subheader(f"Figure #{sel+1}")
        fig_bgr, fig_meta = figs[sel]
        base = f"{os.path.splitext(st.session_state['pdf_name'])[0]}_fig{sel+1}.png"
        _render_single_image_pipeline(fig_bgr, base, infer_module, chartdete_module, config)


def scatter_tab(config):
    """Tab 3: scatter chart — detect markers directly (no line tracing)."""
    st.markdown("Upload a scatter chart — markers are detected directly (LineFormer is skipped).")
    try:
        import marker_extractor  # noqa: F401
    except Exception as e:
        st.error(f"Marker extractor unavailable: {e}")
        return

    img, name = _get_input_image(key_prefix="scatter")
    if img is None:
        return

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Input: {name}",
             use_container_width=True)
    with st.spinner("Detecting scatter markers…"):
        try:
            from marker_extractor import MarkerExtractor
            extractor = MarkerExtractor(img)
            result = extractor.extract()
        except Exception as e:
            st.error(f"Marker detection failed: {e}")
            return

    # MarkerExtractor.extract() returns a dict; adapt to the {points:[]} shape
    # that draw_points_on_image expects.
    series = []
    if isinstance(result, dict) and "series" in result:
        for s in result["series"]:
            pts = s.get("points") or s.get("markers") or []
            series.append({"points": [[int(p[0]), int(p[1])] for p in pts]})
    elif isinstance(result, list):
        series = [{"points": [[int(p[0]), int(p[1])] for p in s]} for s in result if s]

    if not series:
        st.warning("No markers detected. (MarkerExtractor returned an empty/unknown shape — "
                   f"type={type(result).__name__})")
        with st.expander("Raw result"):
            st.write(result)
        return
    total = sum(len(s["points"]) for s in series)
    st.success(f"Detected {len(series)} series, {total} points total.")
    if config["show_visualization"]:
        result_img = draw_points_on_image(img, series, None)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)


def starrydata_tab():
    """Tab 4: upload approved digitizations to Starrydata2/3."""
    st.markdown("Push digitizations to **Starrydata2** (NIMS staging/production) or **Starrydata3** (local KMDS).")
    with st.expander("Starrydata2 (NIMS internal API)"):
        st.write("Set `SD2_TOKEN` env var or `~/.sd2_token` file. NIMS network only.")
        base = st.text_input("Base URL", value="https://starrydata-stg.nims.go.jp",
                             key="sd2_base")
        export_json = st.file_uploader("Upload export.json (from tools/build_export_from_kmds)",
                                       type=["json"], key="sd2_export")
        commit = st.checkbox("Commit (uncheck for DRY-RUN)", value=False, key="sd2_commit")
        if st.button("Push to Starrydata2", key="sd2_push"):
            if export_json is None:
                st.error("Upload an export.json first.")
            else:
                try:
                    import tempfile, subprocess, os as _os
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                        tmp.write(export_json.read())
                        export_path = tmp.name
                    args = [
                        sys.executable,
                        _os.path.join(_project_root, "tools", "starrydata_upload.py"),
                        export_path, "--base", base,
                    ]
                    if commit:
                        args.append("--commit")
                    with st.spinner("Uploading…"):
                        r = subprocess.run(args, capture_output=True, text=True, timeout=120)
                    st.code(r.stdout + "\n---STDERR---\n" + r.stderr, language="text")
                    if r.returncode == 0:
                        st.success("Upload finished.")
                    else:
                        st.error(f"Upload script exited with code {r.returncode}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    with st.expander("Starrydata3 (local KMDS)"):
        try:
            import starrydata3_client  # noqa: F401
            st.write("Client module available.")
        except Exception as e:
            st.warning(f"Starrydata3 client not available: {e}")
        url = st.text_input("Starrydata3 URL", value="", key="sd3_url")
        key = st.text_input("API key", value="", type="password", key="sd3_key")
        st.caption("Full Starrydata3 upload flow will be wired here — coming in a follow-up.")


def vlm_kmds_tab():
    """Tab 5: Claude API key management + KMDS vocabulary lookup + record viewer."""
    st.markdown("Claude-assisted curation and KMDS canonical-term matching.")

    # --- Anthropic API key ---
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    st.text_input(
        "ANTHROPIC_API_KEY (also read from env var)",
        value=st.session_state.get("vlm_api_key", "") or env_key,
        type="password", key="vlm_api_key",
        help="Session-only. Env var wins if both are set. "
             "The Single Image tab's ✦ Claude buttons read this value.",
    )
    active_key = (st.session_state.get("vlm_api_key") or env_key).strip()
    st.caption(("🟢 Key is active — Claude buttons in other tabs are enabled."
                if active_key else
                "⚪ No key set — Claude buttons remain disabled."))

    st.divider()

    # --- KMDS vocabulary matcher ---
    st.subheader("KMDS vocabulary lookup")
    try:
        import kmds_vocab
        vocab_ok = kmds_vocab.available()
    except Exception as e:
        st.error(f"kmds_vocab unavailable: {e}")
        vocab_ok = False

    if vocab_ok:
        q = st.text_input("Look up a property name",
                          placeholder="e.g. Seebeck coefficient, "
                                      "Discharge capacity, Voltage",
                          key="kmds_vocab_query")
        if q.strip():
            match = kmds_vocab.match(q.strip())
            if match:
                is_ext = kmds_vocab.is_extension(match)
                unit = kmds_vocab.unit_of(match) or "—"
                icon = "🔵 extension" if is_ext else "✅ official"
                st.success(f"**{match}** ({icon}) · default unit: `{unit}`")
            else:
                st.warning(f"'{q}' is not in the KMDS vocabulary. "
                           "It would be created as an extension term on first use.")
    else:
        st.info("KMDS vocabulary not loaded (kmds_v15.2.4_nullable.json missing?).")

    st.divider()

    # --- KMDS record viewer ---
    st.subheader("KMDS record viewer / editor")
    st.caption("Upload a `<paper>_kmds/paper.json` (or any KMDS record JSON) "
               "to inspect and edit as a flat table. Save the edited JSON back "
               "with the download button.")
    rec_file = st.file_uploader("Upload paper.json", type=["json"],
                                key="kmds_rec_upload")
    if rec_file is not None:
        try:
            record = json.loads(rec_file.read().decode("utf-8"))
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")
            return
        try:
            from kmds_editor import flatten_record, apply_text_edits
            rows = flatten_record(record)
            df = pd.DataFrame(
                [{"path": ".".join(str(x) for x in r["path"]),
                  "value": r.get("text_value", "")}
                 for r in rows]
            )
            edited = st.data_editor(
                df, use_container_width=True, num_rows="fixed",
                key="kmds_rec_editor",
                column_config={"path": st.column_config.TextColumn(disabled=True)},
            )
            if st.button("Apply edits → Download updated JSON",
                         key="kmds_apply"):
                try:
                    new_record = copy.deepcopy(record)
                    edit_rows = []
                    for r, (_, row) in zip(rows, edited.iterrows()):
                        edit_rows.append({**r, "text_value": row["value"]})
                    apply_text_edits(new_record, edit_rows)
                    st.download_button(
                        "⬇️ Download updated paper.json",
                        data=json.dumps(new_record, indent=2,
                                        ensure_ascii=False).encode("utf-8"),
                        file_name=f"edited_{rec_file.name}",
                        mime="application/json",
                        key="kmds_dl",
                    )
                    st.success("Edits applied. Click the download button above.")
                except Exception as e:
                    st.error(f"apply_text_edits failed: {e}")
        except Exception as e:
            st.error(f"kmds_editor failed: {e}")

    st.divider()

    # --- Extract KMDS from PDF (heavy — uses Claude Sonnet) ---
    with st.expander("Extract a fresh KMDS record from a PDF (heavy — uses Claude)"):
        st.caption("Runs kmds_parallel.extract_kmds_parallel() on the uploaded "
                   "PDF: bibliography, samples, measurements, all parallel Claude "
                   "calls. Requires ANTHROPIC_API_KEY and a few minutes.")
        pdf_up = st.file_uploader("PDF", type=["pdf"], key="kmds_extract_pdf")
        if pdf_up is not None and active_key:
            if st.button("Extract KMDS (this will take 1-3 min)",
                         key="kmds_run"):
                try:
                    import tempfile, kmds_parallel
                    with tempfile.NamedTemporaryFile(suffix=".pdf",
                                                    delete=False) as tmp:
                        tmp.write(pdf_up.getvalue()); pdf_path = tmp.name
                    with tempfile.TemporaryDirectory() as out_dir:
                        with st.spinner("Claude parallel extraction…"):
                            os.environ["ANTHROPIC_API_KEY"] = active_key
                            rec = kmds_parallel.extract_kmds_parallel(
                                pdf_path, output_dir=out_dir,
                            )
                        st.success("Extraction done.")
                        st.download_button(
                            "⬇️ Download paper.json",
                            data=json.dumps(rec, indent=2,
                                            ensure_ascii=False).encode("utf-8"),
                            file_name=f"{os.path.splitext(pdf_up.name)[0]}_kmds.json",
                            mime="application/json",
                        )
                        with st.expander("Preview record"):
                            st.json(rec)
                except Exception as e:
                    st.error(f"KMDS extraction failed: {e}")

    with st.expander("Backend module status"):
        for m in ("vlm_verifier", "vlm_extract", "vlm_screener",
                  "kmds_parallel", "kmds_editor", "kmds_vocab", "legend_mapper"):
            try:
                __import__(m); st.write(f"✅ `{m}`")
            except Exception as e:
                st.write(f"❌ `{m}`: {e}")


def main():
    # Full-width layout + zero horizontal scroll. Streamlit's default
    # .block-container has a max-width of ~46rem even with layout="wide"
    # in some builds, and various child elements (data_editor, wide code
    # blocks, oversized images) still push past the viewport unless we
    # clamp them explicitly.
    st.markdown("""
    <style>
      /* Hide Streamlit's built-in header (Deploy button, 3-dot menu, "Running"
         badge). Their fixed slot on the right was the reason the page could
         still slide sideways even after overflow-x: hidden — Streamlit
         reserves horizontal space for them regardless of layout=wide. */
      header[data-testid="stHeader"] {display: none !important;}
      [data-testid="stToolbar"]      {display: none !important;}
      [data-testid="stDecoration"]   {display: none !important;}
      [data-testid="stStatusWidget"] {display: none !important;}
      #MainMenu {visibility: hidden !important;}
      footer   {visibility: hidden !important;}

      html, body {overflow-x: hidden !important; width: 100% !important;}
      * {max-width: 100%;}
      [data-testid="stAppViewContainer"],
      [data-testid="stMain"] {
        overflow-x: hidden !important;
        max-width: 100vw !important;
      }
      .main .block-container,
      section.main > div.block-container {
        max-width: 100% !important;
        width: 100% !important;
        padding: 0.6rem 1rem !important;
      }
      /* Full width for the main content area next to the sidebar. */
      section[data-testid="stSidebar"] + section {width: 100% !important;}
      [data-testid="stSidebar"] {min-width: 240px; max-width: 280px;}

      /* Image: fit inside its column both ways. */
      /* Image: obey BOTH max-width (column) AND max-height (viewport) so
         landscape charts shrink to fit without ever being clipped. Do NOT
         force width:100% — that overrides the natural aspect and cuts the
         axis ticks off along one edge. */
      [data-testid="stImage"] img,
      [data-testid="stImage"] > img,
      div[data-testid="stImage"] img {
        max-width: 100% !important;
        max-height: 55vh !important;
        width: auto !important;
        height: auto !important;
        margin: 0 auto !important;
        display: block !important;
        object-fit: contain !important;
      }
      /* Column must be allowed to shrink below its content (default
         min-width:auto blocks that) but NOT clip children — clipping
         was what removed the calibration markers near the image edge. */
      [data-testid="stColumn"], [data-testid="column"] {
        min-width: 0 !important;
        max-width: 100% !important;
      }
      [data-testid="stHorizontalBlock"] {
        max-width: 100% !important;
        flex-wrap: wrap;
      }

      /* Long JSON lines were the other source of horizontal scroll. */
      pre, code {white-space: pre-wrap !important; word-break: break-word;}

      /* Data editor / dataframe: keep inside the column width. */
      [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
        max-width: 100% !important;
        overflow-x: auto;  /* scroll INSIDE the table, not the whole page */
      }

      div[data-testid="stExpander"] summary {padding: 0.25rem 0.5rem;}
      h1 {font-size: 1.4rem !important; margin: 0.2rem 0 0.3rem 0 !important;}
      h2 {font-size: 1.15rem !important; margin: 0.3rem 0 !important;}
      h3 {font-size: 1rem !important;    margin: 0.25rem 0 !important;}
      .stMarkdown p {margin-bottom: 0.3rem;}
      [data-testid="stDownloadButton"] button {white-space: nowrap;}
    </style>
    """, unsafe_allow_html=True)

    st.title("📈 AutoLineDigitizer")
    st.caption("Chart line data extraction — output compatible with "
               "[StarryDigitizer](https://starrydigitizer.vercel.app/) and "
               "[WebPlotDigitizer](https://apps.automeris.io/wpd4/).")

    config = _render_sidebar()
    infer_module, chartdete_module = _load_models(config)

    tab_single, tab_pdf, tab_scatter, tab_sd, tab_vlm = st.tabs([
        "📈 Single Image",
        "📄 PDF Gallery",
        "⚫ Scatter",
        "☁️ Starrydata",
        "✨ Claude + KMDS",
    ])
    with tab_single:
        single_image_tab(infer_module, chartdete_module, config)
    with tab_pdf:
        pdf_gallery_tab(infer_module, chartdete_module, config)
    with tab_scatter:
        scatter_tab(config)
    with tab_sd:
        starrydata_tab()
    with tab_vlm:
        vlm_kmds_tab()


if __name__ == "__main__":
    main()
