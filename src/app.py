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
    """Load LineFormer (required) + ChartDete (optional); return both.

    Non-fatal on missing weights so the 🧠 Models tab stays reachable and the
    user can download from within the app instead of hitting a hard stop.
    """
    infer_module = None
    with st.spinner("Loading LineFormer model..."):
        try:
            infer_module = load_lineformer_model()
            st.sidebar.success("LineFormer loaded!")
        except FileNotFoundError as e:
            st.sidebar.error("LineFormer weights not found — see 🧠 Models.")
            st.sidebar.caption(str(e))
        except Exception as e:  # noqa: BLE001
            st.sidebar.error("LineFormer load failed — see 🧠 Models.")
            st.sidebar.caption(str(e))

    chartdete_module = None
    if config["auto_axis"]:
        with st.spinner("Loading ChartDete model..."):
            try:
                chartdete_module = load_chartdete_model()
                st.sidebar.success("ChartDete loaded!")
            except Exception as e:
                st.sidebar.warning(f"ChartDete not available: {e}")
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
    """Interactive point editor built on streamlit-drawable-canvas.

    Fabric.js canvas over the chart image supports clicks on empty space
    (unlike plotly_events which only fires on data points) AND lets users
    drag existing points to move them.
    """
    try:
        from streamlit_drawable_canvas import st_canvas
        from PIL import Image as PILImage
    except Exception as e:  # noqa: BLE001
        st.info(f"Visual editor needs streamlit-drawable-canvas: {e}")
        return

    H, W = img.shape[:2]
    pts = edited_series[selected_idx]["points"]
    # BGR palette -> RGB tuple -> css.
    r, g, b = palette[selected_idx % len(palette)][::-1]
    line_css = f"rgb({r},{g},{b})"

    st.markdown(
        f"**Visual editor** — Line **{selected_idx + 1}** "
        f"({len(pts)} pts). Choose a mode and click / drag on the chart."
    )
    mode_col, radius_col, undo_col = st.columns([3, 1, 1])
    with mode_col:
        mode = st.radio(
            "Mode",
            ["👁 View only",
             "➕ Add point (click empty space)",
             "🖐 Move points (drag existing)",
             "❌ Delete nearest (click near a point)"],
            horizontal=True, key=f"vismode_{img_key}_{selected_idx}",
        )
    with radius_col:
        r_pt = st.slider("Point radius", 3, 20, 8,
                         key=f"visr_{img_key}_{selected_idx}")
    with undo_col:
        st.write("")
        stack_key = f"vis_undo_stack_{img_key}_{selected_idx}"
        if st.button("↩︎ Undo",
                     key=f"vis_undo_{img_key}_{selected_idx}",
                     disabled=not st.session_state.get(stack_key)):
            if st.session_state.get(stack_key):
                prev = st.session_state[stack_key].pop()
                st.session_state[ss_key][selected_idx]["points"] = prev
                st.rerun()

    # Fit canvas to viewport width while preserving image aspect.
    canvas_h = 520
    canvas_w = int(round(canvas_h * W / H))
    scale = canvas_h / H  # canvas pixels per image pixel

    def to_canvas(px, py):
        return px * scale, py * scale

    def to_image(cx, cy):
        return cx / scale, cy / scale

    is_view = mode.startswith("👁")
    is_move = mode.startswith("🖐")
    is_add = mode.startswith("➕")
    is_delete = mode.startswith("❌")

    # ---- View mode: render a static image with points drawn on it. No
    # canvas at all so nothing can be accidentally dragged. ----
    if is_view:
        static = img.copy()
        color_bgr = tuple(int(c) for c in palette[selected_idx % len(palette)])
        for (px, py) in pts:
            cv2.circle(static, (int(px), int(py)), r_pt, color_bgr, -1)
            cv2.circle(static, (int(px), int(py)), r_pt, (0, 0, 0), 1)
        st.image(cv2.cvtColor(static, cv2.COLOR_BGR2RGB),
                 use_container_width=True)
        return

    # Pre-populate the canvas with the current points as fabric.js circles.
    # Movement is only enabled in Move mode; Add/Delete keep the initial
    # objects fully locked so a stray drag can't reshape the line.
    can_drag = is_move
    initial_objects = []
    for i, (px, py) in enumerate(pts):
        cx, cy = to_canvas(px, py)
        initial_objects.append({
            "type": "circle",
            "originX": "center", "originY": "center",
            "left": cx, "top": cy,
            "radius": r_pt,
            "fill": line_css,
            "stroke": "#000000",
            "strokeWidth": 1,
            "selectable": can_drag,
            "evented": can_drag,
            "hoverCursor": "move" if can_drag else "default",
            "hasControls": False,
            "hasBorders": can_drag,
            "lockRotation": True,
            "lockScalingX": True, "lockScalingY": True,
            "lockMovementX": not can_drag,
            "lockMovementY": not can_drag,
        })

    if is_add or is_delete:
        drawing_mode = "point"
    else:  # move
        drawing_mode = "transform"

    pil_bg = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "RGB")
    # update_streamlit=True everywhere: with False the canvas keeps its
    # dragged state client-side but never reports it to Streamlit, so
    # Save moves saw the stale initial positions and reported "no change".
    # The flicker that motivated False before was caused by our own
    # unconditional st.rerun() — with the auto-save-without-rerun path
    # below, drags now stick without a full re-mount.
    canvas_key = f"vis_canvas_{img_key}_{selected_idx}_{mode[:2]}"
    result = st_canvas(
        fill_color=line_css,
        stroke_color=line_css,
        stroke_width=1,
        background_image=pil_bg,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode=drawing_mode,
        initial_drawing={"version": "4.4.0", "objects": initial_objects},
        display_toolbar=False,
        point_display_radius=r_pt,
        key=canvas_key,
    )

    if is_move:
        st.caption("🖐 Drag any point to reposition it. Changes commit "
                   "automatically after each drag and reach the XY table below.")

    if not result or not result.json_data:
        return
    objects = result.json_data.get("objects", []) or []

    # Extract circle centres in image coords.
    canvas_pts = []
    for obj in objects:
        t = obj.get("type")
        left = float(obj.get("left", 0))
        top = float(obj.get("top", 0))
        radius = float(obj.get("radius", r_pt))
        origin_x = obj.get("originX", "left")
        origin_y = obj.get("originY", "top")
        # Fabric "point" produces a circle with originX/Y = "center".
        if origin_x != "center":
            cx = left + radius
        else:
            cx = left
        if origin_y != "center":
            cy = top + radius
        else:
            cy = top
        if t in ("circle",):
            px, py = to_image(cx, cy)
            canvas_pts.append([px, py])

    def push_undo():
        st.session_state.setdefault(stack_key, [])
        st.session_state[stack_key].append(copy.deepcopy(pts))
        if len(st.session_state[stack_key]) > 20:
            st.session_state[stack_key] = st.session_state[stack_key][-20:]

    changed = False
    new_pts = None

    if mode.startswith("➕"):
        # Every extra circle beyond the initial set is a newly-added point.
        if len(canvas_pts) > len(pts):
            new_ones = canvas_pts[len(pts):]
            merged = list(pts) + [[int(round(p[0])), int(round(p[1]))]
                                  for p in new_ones]
            merged.sort(key=lambda p: p[0])
            new_pts = merged
            changed = True
    elif mode.startswith("❌"):
        # New circles are "delete cursors": for each new click, drop the
        # nearest existing point.
        if len(canvas_pts) > len(pts) and pts:
            new_ones = canvas_pts[len(pts):]
            surviving = [list(p) for p in pts]
            for click in new_ones:
                if not surviving:
                    break
                arr = np.array(surviving, dtype=float)
                d2 = (arr[:, 0] - click[0]) ** 2 + (arr[:, 1] - click[1]) ** 2
                idx = int(np.argmin(d2))
                del surviving[idx]
            new_pts = [[int(round(p[0])), int(round(p[1]))] for p in surviving]
            changed = True
    elif is_move:
        # Every drag release triggers a Streamlit rerun (update_streamlit=True);
        # if the returned canvas positions differ from what session_state
        # remembers, commit them. No st.rerun() call — Streamlit already
        # re-rendered from the canvas event, so mutating session_state now is
        # enough for the XY table below to see the new points on this render.
        if len(canvas_pts) == len(pts):
            moved = [[int(round(p[0])), int(round(p[1]))] for p in canvas_pts]
            moved.sort(key=lambda p: p[0])
            if moved != [list(x) for x in pts]:
                new_pts = moved
                changed = True

    if changed and new_pts is not None:
        push_undo()
        st.session_state[ss_key][selected_idx]["points"] = new_pts
        # Bump the rev so the data_editor below re-instantiates and picks up
        # the new points instead of its cached deltas.
        rev_key = f"rev_{img_key}"
        st.session_state[rev_key] = st.session_state.get(rev_key, 0) + 1
        # For Add/Delete we still st.rerun() so the canvas re-mounts with the
        # updated object count. For Move the count is stable, and rerunning
        # would tear down the fabric.js state mid-interaction.
        if not is_move:
            st.rerun()


def _render_axis_picker(img, axis_config, ax_key, img_key, detections, ocr_results):
    """Visual calibration picker: drag the four X1/X2/Y1/Y2 markers on the
    input image to their true tick positions, then click Save.

    Keeps X1/X2 y-values coupled (both live on the X-axis line at the bottom of
    the plot) and Y1/Y2 x-values coupled (both live on the Y-axis line at the
    left), matching how detect_axis_calibration builds the config.
    """
    try:
        from streamlit_drawable_canvas import st_canvas
        from PIL import Image as PILImage
    except Exception as e:  # noqa: BLE001
        st.info(f"Visual axis picker needs streamlit-drawable-canvas: {e}")
        return

    H, W = img.shape[:2]
    canvas_h = 460
    canvas_w = int(round(canvas_h * W / H))
    scale = canvas_h / H

    def to_canvas(px, py):
        return px * scale, py * scale

    def to_image(cx, cy):
        return cx / scale, cy / scale

    # Order in initial_objects is preserved by fabric.js — we rely on that to
    # map indices back to X1, X2, Y1, Y2.
    calib_points = [
        ("X1", axis_config["x1_px"], axis_config["x1_py"], "#e74c3c"),
        ("X2", axis_config["x2_px"], axis_config["x2_py"], "#e74c3c"),
        ("Y1", axis_config["y1_px"], axis_config["y1_py"], "#3498db"),
        ("Y2", axis_config["y2_px"], axis_config["y2_py"], "#3498db"),
    ]
    initial_objects = []
    for label, px, py, color in calib_points:
        cx, cy = to_canvas(float(px), float(py))
        initial_objects.append({
            "type": "circle",
            "originX": "center", "originY": "center",
            "left": cx, "top": cy,
            "radius": 12,
            "fill": color,
            "stroke": "#000000", "strokeWidth": 2,
            "selectable": True,
            "hasControls": False,
            "hasBorders": True,
            "lockRotation": True, "lockScalingX": True, "lockScalingY": True,
        })

    # Explicit RGB mode — some Pillow builds return a mode-less array wrapper
    # that fabric.js chokes on silently (renders a blank canvas with no
    # background). Reopening the RGB view is cheap and forces a known mode.
    pil_bg = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "RGB")

    canvas_key = f"axis_pick_{img_key}"
    try:
        result = st_canvas(
            fill_color="#e74c3c",
            stroke_color="#000000",
            stroke_width=2,
            background_image=pil_bg,
            update_streamlit=True,
            height=canvas_h, width=canvas_w,
            drawing_mode="transform",
            initial_drawing={"version": "4.4.0", "objects": initial_objects},
            display_toolbar=False,
            key=canvas_key,
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Canvas failed to render: {e}")
        result = None

    st.caption(
        "Red = X-axis calibration points (drag to the left tick then the "
        "right tick). Blue = Y-axis calibration points (drag to the bottom "
        "tick then the top tick). "
        "Enter the DATA VALUE each dragged marker represents, then Save."
    )

    # ---- Live position readout ----
    # After each drag, the canvas reports new object positions on the next
    # Streamlit rerun. Show them here so users can see where each marker
    # actually is (pixel coords + provisional data-space using existing
    # calibration) without having to guess.
    if result and result.json_data:
        objs = result.json_data.get("objects", []) or []
        if len(objs) >= 4:
            cur_positions = []
            for i in range(4):
                obj = objs[i]
                left = float(obj.get("left", 0))
                top = float(obj.get("top", 0))
                ox = obj.get("originX", "left")
                oy = obj.get("originY", "top")
                radius = float(obj.get("radius", 0))
                cx = left if ox == "center" else (left + radius)
                cy = top if oy == "center" else (top + radius)
                ix, iy = to_image(cx, cy)
                data_x, data_y = _pixel_to_data(ix, iy, axis_config)
                cur_positions.append((ix, iy, data_x, data_y))
            _label = ["🔴 X1", "🔴 X2", "🔵 Y1", "🔵 Y2"]
            with st.expander("Current marker positions (updates as you drag)",
                             expanded=False):
                for i, (ix, iy, dx, dy) in enumerate(cur_positions):
                    st.write(
                        f"{_label[i]}: pixel `({ix:.0f}, {iy:.0f})` — "
                        f"current calibration says this pixel is "
                        f"X≈`{dx:.3g}`, Y≈`{dy:.3g}`"
                    )
                st.caption(
                    "Use these as a reference for what value to type below "
                    "(e.g. drag the Y2 dot to the 250 tick, then type 250)."
                )

    # Value inputs for the 4 markers — pre-filled with current calibration.
    st.markdown("**Data values at each dragged marker (type manually):**")
    vc1, vc2 = st.columns(2)
    with vc1:
        v_x1 = st.number_input(
            "🔴 X1 value (left tick on X axis)",
            value=float(axis_config["x1_val"]),
            key=f"axpick_x1v_{img_key}", format="%.6g",
        )
        v_x2 = st.number_input(
            "🔴 X2 value (right tick on X axis)",
            value=float(axis_config["x2_val"]),
            key=f"axpick_x2v_{img_key}", format="%.6g",
        )
    with vc2:
        v_y1 = st.number_input(
            "🔵 Y1 value (bottom tick on Y axis)",
            value=float(axis_config["y1_val"]),
            key=f"axpick_y1v_{img_key}", format="%.6g",
        )
        v_y2 = st.number_input(
            "🔵 Y2 value (top tick on Y axis)",
            value=float(axis_config["y2_val"]),
            key=f"axpick_y2v_{img_key}", format="%.6g",
        )

    save_col, _ = st.columns([1, 4])
    with save_col:
        if st.button("💾 Save positions + values",
                     key=f"axpick_save_{img_key}", type="primary"):
            if result and result.json_data:
                objs = result.json_data.get("objects", []) or []
                # Extract centres in image coords.
                new_px = []
                for obj in objs:
                    left = float(obj.get("left", 0))
                    top = float(obj.get("top", 0))
                    ox = obj.get("originX", "left")
                    oy = obj.get("originY", "top")
                    radius = float(obj.get("radius", 0))
                    cx = left if ox == "center" else (left + radius)
                    cy = top if oy == "center" else (top + radius)
                    ix, iy = to_image(cx, cy)
                    new_px.append((ix, iy))
                if len(new_px) >= 4:
                    x1_px, x1_py = new_px[0]
                    x2_px, x2_py = new_px[1]
                    y1_px, y1_py = new_px[2]
                    y2_px, y2_py = new_px[3]
                    new_axis = dict(axis_config)
                    new_axis.update({
                        # X calibration line runs along the bottom — its
                        # y-values should share the y position of Y1
                        # (the bottom-left corner).
                        "x1_px": x1_px, "x2_px": x2_px,
                        "x1_py": y1_py, "x2_py": y1_py,
                        # Y calibration line runs along the left — both
                        # endpoints share the x of X1 (bottom-left).
                        "y1_px": x1_px, "y2_px": x1_px,
                        "y1_py": y1_py, "y2_py": y2_py,
                        # Values from the four typed fields — no more
                        # "dragged to 250 but still labeled 200" mismatch.
                        "x1_val": v_x1, "x2_val": v_x2,
                        "y1_val": v_y1, "y2_val": v_y2,
                    })
                    st.session_state[ax_key] = (new_axis, detections, ocr_results)
                    st.session_state["_show_values_after_apply"] = True
                    # Bump rev so downstream widgets refresh.
                    for k in list(st.session_state):
                        if k.startswith(f"rev_{img_key}"):
                            st.session_state[k] += 1
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

        # ---- Session-state per image: preserve extraction + user edits ----
        img_bytes_key = hashlib.md5(img.tobytes()).hexdigest()[:12]
        ss_key = f"series_{img_bytes_key}"
        ax_key = f"axis_{img_bytes_key}"
        rev_key = f"rev_{img_bytes_key}"
        # Raw extraction cache keyed by (image, extraction params). This is the
        # important fix — LineFormer inference used to fire on every rerun,
        # even when nothing had changed, so every canvas click / Apply / mode
        # switch re-ran a 5-15 s pipeline and the Extraction Result appeared to
        # "blink" as the spinner and result alternated.
        raw_key = (f"raw_{img_bytes_key}_{downsample_mode}_"
                   f"{fixed_step}_{max_points}_{sort_mode}")

        if raw_key not in st.session_state:
            # First time we've seen this image (or a params change) — run the
            # pipeline once and cache the sorted series.
            with status_placeholder.container():
                with st.spinner("⏳ Extracting lines (LineFormer)…"):
                    data_series, _raw_lines = extract_lines(
                        infer_module, img, downsample_mode, fixed_step, max_points
                    )
                    data_series = sort_data_series(data_series, sort_mode)
            st.session_state[raw_key] = data_series
        data_series = st.session_state[raw_key]

        # A monotonically increasing revision the visual editor / axis editor
        # bump whenever they write to session_state — so widgets whose data is
        # driven from outside (data_editor especially, which caches per-key
        # user edits) can be forced to re-instantiate with fresh data.
        if ss_key not in st.session_state:
            st.session_state[ss_key] = copy.deepcopy(data_series)
        if rev_key not in st.session_state:
            st.session_state[rev_key] = 0
        edited_series = st.session_state[ss_key]
        rev = st.session_state[rev_key]

        # NOTE: no initial viz render here — we wait until after axis detection
        # so viz_placeholder is written exactly once per rerun, with the final
        # axis_config baked in. The double-render was overwriting Applied
        # calibration edits back to the auto-detected (or None) state.

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

        # ---- Axis calibration: editable ----
        H_img, W_img = img.shape[:2]
        # Seed a manual calibration if auto-detection failed / was disabled.
        if axis_config is None and ax_key not in st.session_state:
            axis_config = {
                "x1_px": W_img * 0.10, "x1_py": H_img * 0.90, "x1_val": 0.0,
                "x2_px": W_img * 0.90, "x2_py": H_img * 0.90, "x2_val": 1.0,
                "y1_px": W_img * 0.10, "y1_py": H_img * 0.90, "y1_val": 0.0,
                "y2_px": W_img * 0.10, "y2_py": H_img * 0.10, "y2_val": 1.0,
                "xIsLogScale": False, "yIsLogScale": False,
            }
            st.session_state[ax_key] = (axis_config, None, None)

        if axis_config is not None:
            with axis_placeholder.container():
                title = ("Axis Calibration (Auto-detected — click to edit)"
                         if detections is not None
                         else "Axis Calibration (Manual — auto-detection unavailable)")
                # Drag picker rendered outside any expander/tab. Nested layout
                # (expander -> canvas, expander -> tab -> canvas) was leaving
                # the fabric.js iframe with size 0 or unmounted, so the
                # background never showed. Rendering at top level fixes that.
                st.markdown("### 🎯 Drag calibration markers on chart")
                st.caption(
                    "Drag each dot to the tick it represents, enter the "
                    "corresponding data value, then press **Save**. "
                    "Red = X axis, Blue = Y axis."
                )
                _render_axis_picker(img, axis_config, ax_key,
                                    img_bytes_key, detections, ocr_results)

                with st.expander(title, expanded=(detections is None)):
                    st.info(
                        "🔢 **Numeric edit** — declares what each marker "
                        "REPRESENTS (data-space values) and lets you type "
                        "exact pixel positions. ⚠️ Value edits alone do NOT "
                        "visually move the marker on the chart — use the "
                        "**🎯 Drag calibration markers on chart** expander "
                        "above for that.",
                        icon="ℹ️",
                    )
                    if True:
                        # Prefill from current axis_config values.
                        st.caption("Change the tick values (data-space) or the "
                                   "pixel positions of the four calibration points. "
                                   "Press **Apply calibration** to redraw and "
                                   "re-export.")
                        xc1, xc2 = st.columns(2)
                        with xc1:
                            st.markdown("**X-axis**")
                            nx1v = st.number_input(
                                "X1 value (left tick)",
                                value=float(axis_config["x1_val"]),
                                key=f"ax_x1v_{img_bytes_key}", format="%.6g",
                            )
                            nx2v = st.number_input(
                                "X2 value (right tick)",
                                value=float(axis_config["x2_val"]),
                                key=f"ax_x2v_{img_bytes_key}", format="%.6g",
                            )
                            nx1p = st.number_input(
                                "X1 pixel (from left)", min_value=0, max_value=W_img,
                                value=int(round(float(axis_config["x1_px"]))),
                                key=f"ax_x1p_{img_bytes_key}",
                            )
                            nx2p = st.number_input(
                                "X2 pixel (from left)", min_value=0, max_value=W_img,
                                value=int(round(float(axis_config["x2_px"]))),
                                key=f"ax_x2p_{img_bytes_key}",
                            )
                            nxlog = st.checkbox(
                                "X-axis is log scale",
                                value=bool(axis_config.get("xIsLogScale", False)),
                                key=f"ax_xlog_{img_bytes_key}",
                            )
                        with xc2:
                            st.markdown("**Y-axis**")
                            ny1v = st.number_input(
                                "Y1 value (bottom tick)",
                                value=float(axis_config["y1_val"]),
                                key=f"ax_y1v_{img_bytes_key}", format="%.6g",
                            )
                            ny2v = st.number_input(
                                "Y2 value (top tick)",
                                value=float(axis_config["y2_val"]),
                                key=f"ax_y2v_{img_bytes_key}", format="%.6g",
                            )
                            # y1_py is the bottom (higher pixel), y2_py the top (lower pixel)
                            ny1p = st.number_input(
                                "Y1 pixel (from top, larger = lower on chart)",
                                min_value=0, max_value=H_img,
                                value=int(round(float(axis_config["y1_py"]))),
                                key=f"ax_y1p_{img_bytes_key}",
                            )
                            ny2p = st.number_input(
                                "Y2 pixel (from top, smaller = higher on chart)",
                                min_value=0, max_value=H_img,
                                value=int(round(float(axis_config["y2_py"]))),
                                key=f"ax_y2p_{img_bytes_key}",
                            )
                            nylog = st.checkbox(
                                "Y-axis is log scale",
                                value=bool(axis_config.get("yIsLogScale", False)),
                                key=f"ax_ylog_{img_bytes_key}",
                            )

                        # Live readout so the user can confirm the values that will
                        # be applied — changing only the value (not the pixel)
                        # doesn't visually move the marker on the chart, so this
                        # panel is the primary "did my edit take?" feedback.
                        st.markdown(
                            "**Pending calibration (will apply on click):**\n\n"
                            f"- X: `{nx1v:g}` @ px `{nx1p}` → "
                            f"`{nx2v:g}` @ px `{nx2p}`\n"
                            f"- Y: `{ny1v:g}` @ py `{ny1p}` → "
                            f"`{ny2v:g}` @ py `{ny2p}`\n"
                            f"- Log scales: X={'on' if nxlog else 'off'}, "
                            f"Y={'on' if nylog else 'off'}"
                        )

                        apply_col, reset_col, _ = st.columns([1, 1, 3])
                        with apply_col:
                            if st.button("💾 Apply calibration",
                                         key=f"ax_apply_{img_bytes_key}",
                                         type="primary"):
                                new_axis = dict(axis_config)
                                new_axis.update({
                                    "x1_val": nx1v, "x2_val": nx2v,
                                    "y1_val": ny1v, "y2_val": ny2v,
                                    "x1_px": float(nx1p), "x2_px": float(nx2p),
                                    # X calibration line runs along the bottom of
                                    # the plot; keep both endpoints at the same y.
                                    "x1_py": float(ny1p), "x2_py": float(ny1p),
                                    # Y calibration runs along the left of the plot;
                                    # keep both endpoints at the same x.
                                    "y1_px": float(nx1p), "y2_px": float(nx1p),
                                    "y1_py": float(ny1p), "y2_py": float(ny2p),
                                    "xIsLogScale": bool(nxlog),
                                    "yIsLogScale": bool(nylog),
                                })
                                st.session_state[ax_key] = (new_axis, detections, ocr_results)
                                # Auto-enable value labels on the chart so the user
                                # can visually confirm that a value-only edit (e.g.
                                # Y2: 200 → 250 with pixel unchanged) actually
                                # landed — otherwise the marker sits in the same
                                # pixel and looks like nothing happened.
                                st.session_state["_show_values_after_apply"] = True
                                # Bump rev so the XY table (which converts pixels via
                                # this calibration) is forced to refresh from source.
                                st.session_state[rev_key] = rev + 1
                                st.rerun()
                        with reset_col:
                            if st.button("↩︎ Re-detect", key=f"ax_redetect_{img_bytes_key}",
                                         disabled=chartdete_module is None):
                                st.session_state.pop(ax_key, None)
                                st.rerun()

                        if ocr_results:
                            st.markdown("**OCR-detected tick labels (reference):**")
                            ocr_text = []
                            if 'xlabels' in ocr_results:
                                x_vals = [f"{l['value']}" for l in ocr_results['xlabels'] if l['value'] is not None]
                                ocr_text.append(f"X: [{', '.join(x_vals)}]")
                            if 'ylabels' in ocr_results:
                                y_vals = [f"{l['value']}" for l in ocr_results['ylabels'] if l['value'] is not None]
                                ocr_text.append(f"Y: [{', '.join(y_vals)}]")
                            st.caption(' | '.join(ocr_text))
        elif auto_axis:
            axis_placeholder.warning(
                "Could not auto-detect axis calibration. Enable manual "
                "calibration below or turn OFF **Auto-detect axis** in the "
                "sidebar to enter it by hand.")

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
        # Options are stable identifiers ("all" or int line index) so the
        # selection survives when a line's point count changes mid-session
        # (e.g. Delete mode removed a point). Point count is shown only via
        # format_func, not baked into the option value itself.
        options = ["all"] + list(range(len(edited_series)))

        def _curve_label(opt):
            if opt == "all":
                return "All curves"
            i = opt
            nm = (legend_names[i] if i < len(legend_names) and legend_names[i]
                  else None)
            base = f"Line {i+1}" + (f" — {nm}" if nm else "")
            return f"{base}  ({len(edited_series[i]['points'])} pts)"

        sel = st.selectbox(
            "Show", options, index=0,
            format_func=_curve_label,
            key=f"curve_sel_{img_bytes_key}",
            help="Pick a single line to isolate it in the visualization and "
                 "edit its X/Y points below.",
        )

        if sel == "all":
            viz_data = edited_series
            viz_indices = list(range(len(edited_series)))
            highlight = None
        else:
            idx = int(sel)
            viz_data = [edited_series[idx]]
            viz_indices = [idx]
            highlight = idx

        # Re-render viz with selection + numbers baked in.
        # Sticky "burn values after Apply" flag: once the user edits
        # calibration, keep values visible so subsequent value-only edits
        # remain visible; sidebar toggle still overrides when explicitly set.
        show_vals = (config.get("show_calibration_values", False)
                     or st.session_state.get("_show_values_after_apply", False))
        if show_visualization:
            result_img = draw_points_on_image(
                img, viz_data, axis_config,
                show_calibration=config.get("show_calibration", True),
                show_calibration_values=show_vals,
                show_line_numbers=True,
                highlight_idx=highlight,
                line_indices=viz_indices,
                total_lines=len(edited_series),
            )
            viz_placeholder.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                                  use_container_width=True)

        # Visual editor + XY table for the selected single curve.
        if sel != "all":
            idx = int(sel)

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
            # rev is bumped by the visual editor / axis calibration Apply so
            # the data_editor's cached user-edit deltas can't shadow fresh
            # points that came from a canvas drag or a recalibrated axis.
            edited_df = st.data_editor(
                df, num_rows="dynamic", use_container_width=True,
                # Read rev fresh from session_state so a mid-rerun bump (Move
                # mode saves without st.rerun) is reflected in the widget key.
                key=(f"editor_{img_bytes_key}_{idx}_r"
                     f"{st.session_state.get(rev_key, rev)}"),
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

            btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 2])
            with btn_col1:
                if st.button("💾 Apply edits", key=f"apply_{img_bytes_key}_{idx}",
                             type="primary"):
                    st.session_state[ss_key][idx]["points"] = new_pts
                    st.rerun()
            with btn_col2:
                if st.button("↩︎ Reset this line",
                             key=f"reset_{img_bytes_key}_{idx}",
                             help="Restore this line's auto-detected points."):
                    st.session_state[ss_key][idx]["points"] = copy.deepcopy(
                        data_series[idx]["points"])
                    st.rerun()
            with btn_col3:
                # Two-step confirm so an accidental click doesn't wipe a whole
                # line. Session flag remembers the arm state per line/image.
                arm_key = f"delline_arm_{img_bytes_key}_{idx}"
                if st.session_state.get(arm_key):
                    if st.button(f"⚠️ Confirm delete Line {idx+1}",
                                 key=f"del_confirm_{img_bytes_key}_{idx}",
                                 type="secondary",
                                 help="This drops the entire line from the "
                                      "session — Reset all lines can bring "
                                      "it back."):
                        # Drop the line entirely.
                        del st.session_state[ss_key][idx]
                        st.session_state.pop(arm_key, None)
                        # Force the Curves selectbox off this now-missing
                        # index so we don't crash on the next render.
                        sel_key = f"curve_sel_{img_bytes_key}"
                        st.session_state[sel_key] = "all"
                        st.session_state[rev_key] = rev + 1
                        st.rerun()
                    if st.button("Cancel",
                                 key=f"del_cancel_{img_bytes_key}_{idx}"):
                        st.session_state.pop(arm_key, None)
                        st.rerun()
                else:
                    if st.button(f"🗑 Delete entire Line {idx+1}",
                                 key=f"del_line_{img_bytes_key}_{idx}",
                                 help="Remove this line from the extraction. "
                                      "You'll be asked to confirm."):
                        st.session_state[arm_key] = True
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
                    # Reset any partially-armed delete buttons.
                    for k in list(st.session_state):
                        if k.startswith(f"delline_arm_{img_bytes_key}"):
                            st.session_state.pop(k, None)
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
    if infer_module is None:
        st.warning("⚠️ LineFormer weights are missing. Open the 🧠 Models tab "
                   "and download them first.")
        return
    st.markdown("Upload a chart image (or paste from the clipboard) to extract line data.")
    img, name = _get_input_image(key_prefix="single")
    if img is not None:
        _render_single_image_pipeline(img, name, infer_module, chartdete_module, config)


def pdf_gallery_tab(infer_module, chartdete_module, config):
    """Tab 2: upload a PDF, gallery-select figures, digitize per figure."""
    if infer_module is None:
        st.warning("⚠️ LineFormer weights are missing. Open the 🧠 Models tab "
                   "and download them first.")
        return
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


# -----------------------------------------------------------------------------
# Model registry + GitHub Release download / update check
# -----------------------------------------------------------------------------
MODELS_DIR = os.path.join(_project_root, "models")
# Canonical cache is the macOS "Application Support" folder — the models/
# directory in the repo holds symlinks into it, so downloads land in one place
# even when the same weights are reused by other tools (e.g. the packaged .app).
MODELS_CACHE_DIR = os.path.expanduser(
    "~/Library/Application Support/AutoLineDigitizer/models"
)
MODEL_REGISTRY = {
    "lineformer_general": {
        "name": "LineFormer (general)",
        "role": "Line extraction",
        "filename": "iter_3000.pth",
        "required": True,
        "github_repo": "adityaaa-IIT-BHU/AutoLineDigitizer",
        "release_tag": "models",
    },
    "chartdete": {
        "name": "ChartDete",
        "role": "Axis detection",
        "filename": "checkpoint.pth",
        "required": True,
        "github_repo": "adityaaa-IIT-BHU/AutoLineDigitizer",
        "release_tag": "models",
    },
    "lineformer_battery_finetuned": {
        "name": "LineFormer (battery finetuned)",
        "role": "Line extraction — optimised for battery charge curves",
        "filename": "lineformer_battery_finetuned.pth",
        "required": False,
        "github_repo": "adityaaa-IIT-BHU/AutoLineDigitizer",
        "release_tag": "models",
    },
    "lineformer_battery_realistic": {
        "name": "LineFormer (battery realistic)",
        "role": "Line extraction — battery variant",
        "filename": "lineformer_battery_realistic.pth",
        "required": False,
        "github_repo": "adityaaa-IIT-BHU/AutoLineDigitizer",
        "release_tag": "models",
    },
    "lineformer_general_alt": {
        "name": "LineFormer (general, alt)",
        "role": "Line extraction — general v2 variant",
        "filename": "lineformer_general.pth",
        "required": False,
        "github_repo": "adityaaa-IIT-BHU/AutoLineDigitizer",
        "release_tag": "models",
    },
    "lineformer_200k": {
        "name": "LineFormer (200k iterations)",
        "role": "Line extraction — 200k iter checkpoint",
        "filename": "lf_200k_iter9500.pth",
        "required": False,
        "github_repo": "adityaaa-IIT-BHU/AutoLineDigitizer",
        "release_tag": "models",
    },
}


def _model_local_path(model_key):
    m = MODEL_REGISTRY[model_key]
    return os.path.join(MODELS_CACHE_DIR, m["filename"])


def _model_local_info(model_key):
    p = _model_local_path(model_key)
    if not os.path.exists(p):
        return {"present": False}
    st_ = os.stat(p)
    return {
        "present": True,
        "path": p,
        "size": st_.st_size,
        "mtime": datetime.fromtimestamp(st_.st_mtime),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_release_info(repo, tag):
    """Query GitHub Releases API and cache for 30 min.

    Unauthenticated calls are rate-limited to 60/hour per IP — we cache to
    stay well under that. Returns (info, error_str)."""
    import urllib.request
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    req = urllib.request.Request(
        api,
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "AutoLineDigitizer-webapp"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode("utf-8"))
        assets = {a["name"]: a for a in data.get("assets", [])}
        return {
            "name": data.get("name", tag),
            "tag": data.get("tag_name", tag),
            "published_at": data.get("published_at"),
            "html_url": data.get("html_url", ""),
            "assets": assets,
        }, None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def _fmt_mb(n_bytes):
    return f"{n_bytes / (1024*1024):.1f} MB"


def _download_with_progress(url, dest_path, progress_bar, status_text):
    """Streaming download with periodic st.progress updates."""
    import urllib.request
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"
    req = urllib.request.Request(
        url, headers={"User-Agent": "AutoLineDigitizer-webapp"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        chunk = 1024 * 512  # 512 KB
        with open(tmp_path, "wb") as f:
            while True:
                data = r.read(chunk)
                if not data:
                    break
                f.write(data)
                got += len(data)
                if total:
                    progress_bar.progress(min(1.0, got / total))
                    status_text.text(
                        f"Downloading {_fmt_mb(got)} / {_fmt_mb(total)} "
                        f"({100 * got / total:.1f}%)"
                    )
                else:
                    status_text.text(f"Downloading {_fmt_mb(got)}…")
    os.replace(tmp_path, dest_path)
    # Keep a symlink inside <repo>/models/ so app.py finds it via the fixed path.
    os.makedirs(MODELS_DIR, exist_ok=True)
    link_path = os.path.join(MODELS_DIR, os.path.basename(dest_path))
    if os.path.islink(link_path) or os.path.exists(link_path):
        try:
            os.remove(link_path)
        except IsADirectoryError:
            pass
    try:
        os.symlink(dest_path, link_path)
    except OSError:
        pass  # link creation is best-effort; app.py still finds the cache path


def _iter_missing_required_models():
    for key, m in MODEL_REGISTRY.items():
        if m["required"] and not _model_local_info(key)["present"]:
            yield key, m


def _iter_updates_available():
    """Yield model_key, local_info, remote_asset for models where sizes differ
    (proxy for 'a newer weights file has been published upstream')."""
    seen_releases = {}
    for key, m in MODEL_REGISTRY.items():
        local = _model_local_info(key)
        if not local["present"]:
            continue
        cache_key = (m["github_repo"], m["release_tag"])
        if cache_key not in seen_releases:
            info, _err = _fetch_release_info(*cache_key)
            seen_releases[cache_key] = info
        info = seen_releases[cache_key]
        if not info:
            continue
        asset = info["assets"].get(m["filename"])
        if not asset:
            continue
        remote_size = int(asset.get("size", 0))
        if remote_size and remote_size != local["size"]:
            yield key, local, asset, info


def models_tab():
    """Tab 6: model registry — show local vs upstream, notify updates, download."""
    st.markdown("Manage the ML weights AutoLineDigitizer uses. Weights are "
                "cached under `~/Library/Application Support/AutoLineDigitizer/"
                "models/` and symlinked into the repo's `models/` folder.")

    if st.button("↻ Refresh upstream info", key="mdl_refresh"):
        _fetch_release_info.clear()
        st.rerun()

    # Group by upstream release so we only call the API once per release.
    seen_releases = {}
    for key, m in MODEL_REGISTRY.items():
        rel_key = (m["github_repo"], m["release_tag"])
        seen_releases.setdefault(rel_key, None)
    for rel_key in list(seen_releases):
        info, err = _fetch_release_info(*rel_key)
        seen_releases[rel_key] = (info, err)

    for key, m in MODEL_REGISTRY.items():
        local = _model_local_info(key)
        info, err = seen_releases[(m["github_repo"], m["release_tag"])]
        asset = info["assets"].get(m["filename"]) if info else None

        with st.container(border=True):
            hdr_col, action_col = st.columns([4, 1])
            with hdr_col:
                badge = "🔴 required" if m["required"] else "⚪ optional"
                st.markdown(f"### {m['name']} · {badge}")
                st.caption(f"{m['role']} — `{m['filename']}`")

            # Local status
            if local["present"]:
                st.markdown(
                    f"**Local:** ✅ present · {_fmt_mb(local['size'])} · "
                    f"downloaded {local['mtime'].strftime('%Y-%m-%d %H:%M')}"
                )
            else:
                st.markdown("**Local:** ❌ not installed")

            # Upstream status
            if err:
                st.warning(f"Upstream check failed: {err}")
            elif not info:
                st.caption("Upstream info unavailable.")
            elif not asset:
                st.caption(
                    f"Upstream release '{info['name']}' has no asset called "
                    f"`{m['filename']}` — you may need a manual URL."
                )
            else:
                pub = asset.get("updated_at", info.get("published_at", ""))
                pub_short = (pub or "")[:10]
                st.markdown(
                    f"**Upstream:** [{info['name']}]({info['html_url']}) · "
                    f"{_fmt_mb(int(asset.get('size', 0)))} · "
                    f"published {pub_short}"
                )

                # Update / install decision
                if not local["present"]:
                    status = "❌ Not installed — click Download."
                elif int(asset.get("size", 0)) == local["size"]:
                    status = "✅ Up to date."
                else:
                    status = ("⚠️ **Newer version available upstream** "
                              f"(remote {_fmt_mb(int(asset.get('size', 0)))}, "
                              f"local {_fmt_mb(local['size'])}).")
                st.markdown(status)

            with action_col:
                label = ("⬇ Re-download" if local["present"]
                         else "⬇ Download")
                if st.button(label, key=f"mdl_dl_{key}",
                             disabled=(asset is None),
                             type="primary" if not local["present"] else "secondary"):
                    dest = _model_local_path(key)
                    prog = st.progress(0.0)
                    stat = st.empty()
                    try:
                        _download_with_progress(asset["browser_download_url"],
                                                dest, prog, stat)
                        stat.success(f"Downloaded {m['filename']}.")
                        # Invalidate cached model loaders so a fresh weight is picked up.
                        try:
                            load_lineformer_model.clear()
                            load_chartdete_model.clear()
                        except Exception:
                            pass
                        st.rerun()
                    except Exception as e:
                        stat.error(f"Download failed: {e}")

    st.divider()
    st.caption(
        "**Cache size:** " + _fmt_mb(sum(
            _model_local_info(k)["size"]
            for k in MODEL_REGISTRY if _model_local_info(k)["present"]
        )) + f" total across {sum(1 for k in MODEL_REGISTRY if _model_local_info(k)['present'])} file(s)."
    )
    if st.button("🗑 Purge all cached models (frees disk)", key="mdl_purge"):
        removed = []
        for k, m in MODEL_REGISTRY.items():
            p = _model_local_path(k)
            if os.path.exists(p):
                try:
                    os.remove(p)
                    removed.append(m["filename"])
                except Exception as e:
                    st.error(f"Failed to remove {p}: {e}")
        # Also remove dangling symlinks in repo models/.
        for name in os.listdir(MODELS_DIR):
            if name.endswith(".pth"):
                try:
                    os.remove(os.path.join(MODELS_DIR, name))
                except Exception:
                    pass
        try:
            load_lineformer_model.clear()
            load_chartdete_model.clear()
        except Exception:
            pass
        if removed:
            st.success(f"Removed: {', '.join(removed)}")
            st.rerun()


def _render_update_banner():
    """Small header banner: notify if required models are missing or upstream
    has newer weights. Runs on every rerun but the GitHub call is cached."""
    missing = list(_iter_missing_required_models())
    if missing:
        names = ", ".join(m["name"] for _, m in missing)
        st.error(
            f"⚠️ Missing required model(s): **{names}**. "
            "Open the 🧠 Models tab to download."
        )
        return

    try:
        updates = list(_iter_updates_available())
    except Exception:
        updates = []
    if updates:
        names = ", ".join(MODEL_REGISTRY[k]["name"] for k, *_ in updates)
        st.info(
            f"🆕 A newer version is available for: **{names}**. "
            "Open the 🧠 Models tab to update."
        )


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

    # Header-level notice for missing or outdated weights — checked before the
    # sidebar loads models so users see the ⚠️ before Streamlit tries to import
    # a checkpoint that doesn't exist yet.
    _render_update_banner()

    config = _render_sidebar()
    infer_module, chartdete_module = _load_models(config)

    (tab_single, tab_pdf, tab_scatter, tab_sd, tab_vlm,
     tab_models) = st.tabs([
        "📈 Single Image",
        "📄 PDF Gallery",
        "⚫ Scatter",
        "☁️ Starrydata",
        "✨ Claude + KMDS",
        "🧠 Models",
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
    with tab_models:
        models_tab()


if __name__ == "__main__":
    main()
