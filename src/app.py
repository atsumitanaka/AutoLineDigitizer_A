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
import json
import io
import zipfile
import tarfile
from datetime import datetime

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


def draw_points_on_image(img, data_series, axis_config=None):
    """Draw extracted points as symbols on image, with optional axis calibration markers."""
    import line_utils

    result_img = img.copy()
    num_lines = len(data_series)
    colors = list(line_utils.get_distinct_colors(num_lines))

    # Marker symbols (using different shapes)
    markers = [
        cv2.MARKER_CROSS,
        cv2.MARKER_DIAMOND,
        cv2.MARKER_SQUARE,
        cv2.MARKER_TRIANGLE_UP,
        cv2.MARKER_TRIANGLE_DOWN,
        cv2.MARKER_STAR,
    ]

    for line_idx, series in enumerate(data_series):
        color = colors[line_idx]
        marker = markers[line_idx % len(markers)]

        for pt in series["points"]:
            x, y = int(pt[0]), int(pt[1])
            cv2.drawMarker(result_img, (x, y), color, marker, markerSize=8, thickness=2)

    # Draw axis calibration points if available
    if axis_config is not None:
        # Single color for all calibration points (Magenta - visible on white backgrounds)
        calib_color = (255, 0, 255)   # Magenta (BGR)
        outline_color = (0, 0, 0)  # Black outline for contrast
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Helper function to draw text with background
        def draw_text_with_bg(img, text, pos, color):
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = pos
            # Draw background rectangle
            padding = 3
            cv2.rectangle(img, (x - padding, y - text_h - padding),
                         (x + text_w + padding, y + baseline + padding),
                         (255, 255, 255), -1)  # White background
            cv2.rectangle(img, (x - padding, y - text_h - padding),
                         (x + text_w + padding, y + baseline + padding),
                         outline_color, 1)  # Black border
            # Draw text
            cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

        # Helper function to draw calibration point with marker
        def draw_calib_point(img, x, y, color, label, label_offset):
            # Draw filled circle with outline
            cv2.circle(img, (x, y), 12, outline_color, 3)  # Black outline
            cv2.circle(img, (x, y), 10, color, -1)  # Filled circle
            cv2.circle(img, (x, y), 10, outline_color, 2)  # Inner outline
            # Draw crosshair inside circle
            cv2.line(img, (x - 6, y), (x + 6, y), outline_color, 2)
            cv2.line(img, (x, y - 6), (x, y + 6), outline_color, 2)
            # Draw label with background
            label_x = x + label_offset[0]
            label_y = y + label_offset[1]
            draw_text_with_bg(img, label, (label_x, label_y), color)

        # Get calibration points
        x1_x, x1_y = int(axis_config['x1_px']), int(axis_config['x1_py'])
        x2_x, x2_y = int(axis_config['x2_px']), int(axis_config['x2_py'])
        y1_x, y1_y = int(axis_config['y1_px']), int(axis_config['y1_py'])
        y2_x, y2_y = int(axis_config['y2_px']), int(axis_config['y2_py'])

        # Draw dashed lines connecting calibration points
        # X-axis line (X1 to X2)
        dash_length = 10
        gap_length = 5
        # Draw dashed line for X-axis
        dx = x2_x - x1_x
        dy = x2_y - x1_y
        dist = max(1, int(np.sqrt(dx*dx + dy*dy)))
        for i in range(0, dist, dash_length + gap_length):
            start_x = int(x1_x + dx * i / dist)
            start_y = int(x1_y + dy * i / dist)
            end_i = min(i + dash_length, dist)
            end_x = int(x1_x + dx * end_i / dist)
            end_y = int(x1_y + dy * end_i / dist)
            cv2.line(result_img, (start_x, start_y), (end_x, end_y), calib_color, 2)

        # Draw dashed line for Y-axis
        dx = y2_x - y1_x
        dy = y2_y - y1_y
        dist = max(1, int(np.sqrt(dx*dx + dy*dy)))
        for i in range(0, dist, dash_length + gap_length):
            start_x = int(y1_x + dx * i / dist)
            start_y = int(y1_y + dy * i / dist)
            end_i = min(i + dash_length, dist)
            end_x = int(y1_x + dx * end_i / dist)
            end_y = int(y1_y + dy * end_i / dist)
            cv2.line(result_img, (start_x, start_y), (end_x, end_y), calib_color, 2)

        # Draw calibration points with labels
        # X1 point (left on X axis)
        draw_calib_point(result_img, x1_x, x1_y, calib_color,
                        f"X1={axis_config['x1_val']}", (15, -5))

        # X2 point (right on X axis) - label on left side to avoid edge
        draw_calib_point(result_img, x2_x, x2_y, calib_color,
                        f"X2={axis_config['x2_val']}", (-100, -5))

        # Y1 point (bottom on Y axis)
        draw_calib_point(result_img, y1_x, y1_y, calib_color,
                        f"Y1={axis_config['y1_val']}", (15, 20))

        # Y2 point (top on Y axis)
        draw_calib_point(result_img, y2_x, y2_y, calib_color,
                        f"Y2={axis_config['y2_val']}", (15, -5))

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

        # Show initial result (without axis calibration) immediately
        if show_visualization:
            result_img = draw_points_on_image(img, data_series, None)
            viz_placeholder.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Show line summary
        total_points = sum(len(s['points']) for s in data_series)
        line_pts = [len(s['points']) for s in data_series]
        summary_placeholder.success(f"**{len(data_series)} lines** detected ({total_points} points total)")
        caption_placeholder.caption(f"Points per line: {', '.join(map(str, line_pts))}")

        # Step 2: Run axis detection (slower, OCR-heavy) - with spinner
        if auto_axis and chartdete_module is not None:
            with status_placeholder.container():
                with st.spinner("🔍 Detecting axis labels (ChartDete + OCR)..."):
                    axis_config, detections, ocr_results = detect_axis_calibration(
                        chartdete_module, img
                    )

            # Update visualization with axis calibration
            if show_visualization:
                result_img = draw_points_on_image(img, data_series, axis_config)
                viz_placeholder.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Clear the status
            status_placeholder.empty()
        else:
            # Clear status if no axis detection
            status_placeholder.empty()

        # Show axis calibration results
        if axis_config is not None:
            with axis_placeholder.container():
                with st.expander("Axis Calibration (Auto-detected)", expanded=True):
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

        # Build StarryDigitizer project
        project_json = convert_to_starry_digitizer_format(
            data_series, img.shape, axis_config
        )

        # Build WebPlotDigitizer project
        wpd_json = convert_to_wpd_format(
            data_series, img.shape, axis_config
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
    """Tab 5: Claude curation + KMDS record editing (skeleton)."""
    st.markdown("Claude-assisted axis reading, legend naming, and KMDS record editing.")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    st.text_input(
        "ANTHROPIC_API_KEY (set via env or paste here for this session)",
        value=api_key, type="password", key="vlm_api_key",
        help="Not stored — env var wins if both set.",
    )
    st.info("VLM + KMDS wiring is in progress — the backend modules "
            "(vlm_verifier, vlm_extract, kmds_parallel, kmds_editor, kmds_vocab) "
            "all import cleanly, so hooking them into this tab is the next step.")
    with st.expander("Backend module status"):
        for m in ("vlm_verifier", "vlm_extract", "vlm_screener",
                  "kmds_parallel", "kmds_editor", "kmds_vocab", "legend_mapper"):
            try:
                __import__(m); st.write(f"✅ `{m}`")
            except Exception as e:
                st.write(f"❌ `{m}`: {e}")


def main():
    st.title("📈 AutoLineDigitizer")
    st.markdown("""
    Extract chart line data from images or full PDFs.
    Output is compatible with **[StarryDigitizer](https://starrydigitizer.vercel.app/)** and **[WebPlotDigitizer](https://apps.automeris.io/wpd4/)**.

    **[LineFormer Paper (ICDAR 2023)](https://arxiv.org/abs/2305.01837)** |
    **[ChartDete Paper (ICDAR 2023)](https://arxiv.org/abs/2305.04151)**
    """)

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
