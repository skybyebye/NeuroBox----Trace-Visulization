from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import tifffile

import matplotlib
matplotlib.use('QtAgg')
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QColorDialog,
)
from PyQt6.QtWidgets import QListWidget as ROIListWidget
from PyQt6.QtWidgets import QListWidgetItem as ROIListWidgetItem
from PyQt6.QtWidgets import QAbstractItemView

from cal_params import get_params
from mask_weight import load_mean_image_from_tif, normalize_image
import cal_waveform
from cal_wavelet import denoise_trace as pca_wavelet_trace
import plot_wavelet_pca as wavelet_pca_plot
from trace_process import (
    detect_spikes,
    detect_trace_event_intervals,
    compute_volpy_snr,
    extract_weighted_roi_traces,
    F_trace,
    generate_firingRate,
    highpass_filt_trace,
    lowpass_filt_trace,
    pix_exp,
    pix_max,
    pix_overmean,
    weighted_trace,
    wavelet_trace,
    snr_trace,
)
from scipy.ndimage import convolve, gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import savgol_filter
import util
import normcorre as normcorre_backend
import pmd_denoise as pmd_backend


AVG_PANEL_MODES = {
    'event': 'average event response',
    'waveform': 'average waveform',
    'firing_rate': 'average firing rate',
}

COMBINE_MODES = {
    'individual': 'individual colors',
    'mean': 'mean',
    'sum': 'sum',
}

ROI_COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

TRACE_SOURCE_DIRECT_OPTIONS = [
    ('raw(suite2p)', 'suite2p_raw'),
    ('weighted(pix_max)', 'weighted_pix_max'),
    ('weighted(pix_exp)', 'weighted_pix_exp'),
    ('weighted(pix_overmean)', 'weighted_pix_overmean'),
]

TRACE_SOURCE_VOLPY_KEYS = ['ts', 't', 'dFF', 't_sub', 't_rec']

IMAGE_LAYER_MODES = [
    ('video', 'video'),
    ('z_average', 'z-average'),
    ('max_projection', 'max projection'),
    ('correlation', 'correlation image'),
]

IMAGE_LAYER_MODELS = ['Raw', 'VolPy', 'NoRMCorre', 'PMD', 'Local']

MASK_SOURCE_OPTIONS = [
    ('suite2p', 'suite2p'),
    ('weighted_pix_exp', 'pix-exp'),
    ('weighted_pix_overmean', 'pix-overmean'),
    ('weighted_pix_max', 'pix-max'),
    ('volpy', 'VolPy'),
]

PMD_PIXEL_WEIGHT_SOURCE_OPTIONS = [
    ('none', 'none'),
    ('suite2p', 'suite2p'),
    ('volpy', 'VolPy'),
    ('correlation', 'correlation'),
    ('z_average', 'z-average'),
]

MASK_COLOR_OPTIONS = [
    ('roi', 'different ROIs'),
    ('type', 'ROI types'),
]

ROI_TYPE_COLORS = {
    'stim_cell': (1.0, 0.15, 0.10),
    'unstim_cell': (0.10, 0.45, 1.0),
    'stim_non_cell': (1.0, 0.55, 0.10),
    'unstim_non_cell': (0.65, 0.65, 0.65),
    'roi': (1.0, 0.85, 0.10),
}

IMAGE_INFO_USER_FIELDS = [
    'Experiment date',
    'Animal ID',
    'Brain area',
    'Imaging source',
    'Indicator',
    'Notes',
]


@dataclass
class GUIState:
    source_type: Optional[str] = None  # folder | table | volpy_folder
    source_path: str = ''
    frame_rate: float = 0.0
    traces: List[np.ndarray] = field(default_factory=list)
    trace_names: List[str] = field(default_factory=list)
    firing_rate: List[np.ndarray] = field(default_factory=list)
    spike_times: List[List[np.ndarray]] = field(default_factory=list)
    thresholds: List[Optional[np.ndarray]] = field(default_factory=list)
    cells: Optional[np.ndarray] = None
    stim_cells: Optional[np.ndarray] = None
    onset_times_trial: Optional[np.ndarray] = None
    offset_times_trial: Optional[np.ndarray] = None
    trial_duration: Optional[float] = None
    raw_image: Optional[np.ndarray] = None
    raw_movie: Optional[np.ndarray] = None
    tif_path: Optional[str] = None
    weight_alpha: Optional[np.ndarray] = None
    weight_alpha_maps: dict[str, np.ndarray] = field(default_factory=dict)
    suite2p_alpha: Optional[np.ndarray] = None
    stat: Optional[np.ndarray] = None
    ops: Optional[dict] = None
    vpy: Optional[dict] = None
    volpy_suite2p_indices: Optional[np.ndarray] = None
    negative_mode: bool = False
    intensity_max: Optional[float] = None
    trace_reverse_max: Optional[np.ndarray] = None
    image_info_path: Optional[str] = None
    image_user_info: dict[str, str] = field(default_factory=dict)

    @property
    def n_rois(self) -> int:
        if not self.traces:
            return 0
        return int(self.traces[0].shape[0])


@dataclass
class TraceControlRow:
    widget: QGroupBox
    content_widget: QWidget
    fold_button: QPushButton
    visible_checkbox: QCheckBox
    source_combo: QComboBox
    lowpass_checkbox: QCheckBox
    lowpass_edit: QLineEdit
    highpass_checkbox: QCheckBox
    highpass_edit: QLineEdit
    wavelet_checkbox: QCheckBox
    wavelet_name_edit: QLineEdit
    wavelet_level_edit: QLineEdit
    wavelet_scale_edit: QLineEdit
    wavelet_mode_combo: QComboBox
    pca_wavelet_checkbox: QCheckBox
    pca_wavelet_fmin_edit: QLineEdit
    pca_wavelet_fmax_edit: QLineEdit
    pca_wavelet_fn_edit: QLineEdit
    pca_wavelet_param_button: QPushButton
    pca_wavelet_cfg: dict[str, Any]
    snr_checkbox: QCheckBox
    snr_window_edit: QLineEdit
    volpy_checkbox: QCheckBox
    volpy_combo: QComboBox
    baseline_mode_combo: QComboBox
    baseline_lowpass_checkbox: QCheckBox
    baseline_lowpass_edit: QLineEdit
    baseline_rolling_checkbox: QCheckBox
    baseline_rolling_mode_combo: QComboBox
    baseline_rolling_window_edit: QLineEdit
    baseline_polyfit_checkbox: QCheckBox
    baseline_poly_order_edit: QLineEdit
    baseline_savgol_checkbox: QCheckBox
    baseline_savgol_window_edit: QLineEdit
    baseline_savgol_order_edit: QLineEdit
    spike_checkbox: QCheckBox
    spike_method_combo: QComboBox
    spike_k_edit: QLineEdit
    threshold_checkbox: QCheckBox
    waveform_checkbox: QCheckBox
    waveform_mode_combo: QComboBox
    waveform_export_button: QPushButton
    remove_button: QPushButton


@dataclass
class ImageLayerControlRow:
    layer_id: str
    widget: QFrame
    model_combo: Optional[QComboBox]
    visible_checkbox: QCheckBox
    mode_combo: QComboBox
    mask_source_combo: QComboBox
    nframes_label: QLabel
    summary_label: QLabel
    remove_button: Optional[QPushButton] = None
    export_button: Optional[QPushButton] = None
    image_layer_params: dict[str, Any] = field(default_factory=dict)


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, figure: Figure, parent: Optional[QWidget] = None):
        super().__init__(figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class InteractiveFigureWidget(QWidget):
    """Matplotlib canvas with mouse zoom, drag, and double-click reset."""

    def __init__(self, title: str, mode: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.mode = mode
        self.group = QGroupBox(title)
        self.figure = Figure(figsize=(8, 3), dpi=100)
        self.canvas = PlotCanvas(self.figure, self)
        self.toolbar = None

        self.control_label = QLabel()
        self.control_slider = QSlider(Qt.Orientation.Horizontal)
        self.control_slider.setTracking(True)
        self.control_slider.setMinimum(0)
        self.control_slider.setMaximum(1000)
        self.control_slider.setValue(0)
        self.control_slider.valueChanged.connect(self._on_slider_changed)

        self.reset_button = QPushButton('Home')
        self.reset_button.clicked.connect(self.reset_view)

        self._x_total = 1.0
        self._x_window = 1.0
        self._x_start = 0.0
        self._x_origin = 0.0
        self._x_locked = False
        self._x_axes = []

        self._image_full_xlim: Optional[tuple[float, float]] = None
        self._image_full_ylim: Optional[tuple[float, float]] = None
        self._image_home_limits: list[tuple[Any, tuple[float, float], tuple[float, float]]] = []
        self._image_zoom_factor: float = 1.0
        self._drag_state: Optional[dict[str, Any]] = None
        self._independent_x_axes: bool = False
        self._x_axis_home_limits: list[tuple[Any, tuple[float, float]]] = []
        self.on_xwindow_changed = None

        self.group_layout = QVBoxLayout(self.group)
        self.group_layout.setContentsMargins(2, 2, 2, 2)
        self.group_layout.setSpacing(2)
        self.top_controls_layout = QVBoxLayout()
        self.top_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.top_controls_layout.setSpacing(2)
        self.group_layout.addLayout(self.top_controls_layout)
        self.canvas_row = QHBoxLayout()
        self.canvas_row.setContentsMargins(0, 0, 0, 0)
        self.canvas_row.setSpacing(4)
        self.canvas_row.addWidget(self.canvas, stretch=1)
        self.group_layout.addLayout(self.canvas_row, stretch=1)

        self.extra_controls_layout = QVBoxLayout()
        self.extra_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.extra_controls_layout.setSpacing(2)
        self.group_layout.addLayout(self.extra_controls_layout)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.group)

        if self.mode == 'image':
            self.control_label.setText('Zoom')
            self.control_slider.setMinimum(100)
            self.control_slider.setMaximum(800)
            self.control_slider.setValue(100)
        else:
            self.control_label.setText('Time')
            self.control_slider.setMinimum(0)
            self.control_slider.setMaximum(1000)
            self.control_slider.setValue(0)

        self.canvas.mpl_connect('scroll_event', self._on_scroll_zoom)
        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)

    def set_side_panel(self, widget: QWidget, width: int = 260):
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        side_scroll.setMinimumWidth(250)
        side_scroll.setMaximumWidth(width)
        side_scroll.setWidget(widget)
        self.canvas_row.addWidget(side_scroll)

    def _axes(self):
        return self.figure.axes[0] if self.figure.axes else None

    def set_x_axes(self, axes):
        self._independent_x_axes = False
        self._x_axis_home_limits = []
        if axes is None:
            self._x_axes = []
        elif isinstance(axes, (list, tuple, np.ndarray)):
            self._x_axes = list(axes)
        else:
            self._x_axes = [axes]
        if self.mode == 'xwindow':
            self.control_slider.setEnabled(True)

    def set_independent_x_axes(self, axes):
        if axes is None:
            self._x_axes = []
        elif isinstance(axes, (list, tuple, np.ndarray)):
            self._x_axes = list(axes)
        else:
            self._x_axes = [axes]
        self._independent_x_axes = True
        self._x_axis_home_limits = [(ax, tuple(ax.get_xlim())) for ax in self._x_axes]
        if self.mode == 'xwindow':
            self.control_slider.setEnabled(False)

    def set_placeholder(self, text: str):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha='center', va='center')
        ax.axis('off')
        self.canvas.draw_idle()

    def set_x_window(
        self,
        total_width: float,
        window_width: float,
        x_start: Optional[float] = None,
        origin: float = 0.0,
    ):
        self._x_origin = float(origin)
        self._x_total = max(float(total_width), 1e-6)
        self._x_window = max(min(float(window_width), self._x_total), 1e-6)
        if x_start is not None:
            self._x_start = float(np.clip(x_start, 0.0, max(0.0, self._x_total - self._x_window)))
        else:
            self._x_start = float(np.clip(self._x_start, 0.0, max(0.0, self._x_total - self._x_window)))
        self._sync_slider_from_x_window()

    def get_x_start(self) -> float:
        return self._x_start

    def capture_image_home(self):
        axes = list(self.figure.axes)
        if not axes:
            return
        self._image_home_limits = [(ax, tuple(ax.get_xlim()), tuple(ax.get_ylim())) for ax in axes]
        self._image_full_xlim = self._image_home_limits[0][1]
        self._image_full_ylim = self._image_home_limits[0][2]
        self._image_zoom_factor = 1.0
        self._sync_image_slider_from_zoom()

    def reset_view(self):
        if self.mode == 'image' and self._image_home_limits:
            for ax, xlim, ylim in self._image_home_limits:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
            self._image_zoom_factor = 1.0
            self._sync_image_slider_from_zoom()
            self.canvas.draw_idle()
            return

        if self.mode == 'xwindow':
            if self._independent_x_axes:
                for ax, xlim in self._x_axis_home_limits:
                    ax.set_xlim(*xlim)
                self.canvas.draw_idle()
                return
            self._x_start = 0.0
            self._sync_slider_from_x_window()
            self._apply_x_limits(0.0, min(self._x_window, self._x_total))
            self._notify_xwindow_changed()
            self.canvas.draw_idle()

    def _sync_slider_from_x_window(self):
        if self.mode != 'xwindow':
            return
        travel = max(self._x_total - self._x_window, 0.0)
        self._x_locked = True
        try:
            if travel <= 1e-9:
                self.control_slider.setEnabled(False)
                self.control_slider.setValue(0)
            else:
                self.control_slider.setEnabled(True)
                value = int(round(1000 * self._x_start / travel))
                self.control_slider.setValue(max(0, min(1000, value)))
        finally:
            self._x_locked = False

    def _sync_image_slider_from_zoom(self):
        if self.mode != 'image':
            return
        self._x_locked = True
        try:
            self.control_slider.setValue(int(round(self._image_zoom_factor * 100)))
        finally:
            self._x_locked = False

    def _on_slider_changed(self, value: int):
        if self._x_locked:
            return
        ax = self._axes()
        if ax is None:
            return

        if self.mode == 'xwindow':
            travel = max(self._x_total - self._x_window, 0.0)
            self._x_start = 0.0 if travel <= 1e-9 else (value / 1000.0) * travel
            self._apply_x_limits(self._x_start, self._x_start + self._x_window)
            self._notify_xwindow_changed()
            self.canvas.draw_idle()
            return

        if self.mode == 'image':
            factor = max(1.0, value / 100.0)
            self.apply_image_zoom(factor)

    def apply_image_zoom(self, factor: float, center_x: Optional[float] = None, center_y: Optional[float] = None):
        if not self._image_home_limits:
            return

        factor = max(1.0, min(float(factor), 8.0))
        for ax, home_xlim, home_ylim in self._image_home_limits:
            x0, x1 = home_xlim
            y0, y1 = home_ylim
            full_w = abs(x1 - x0)
            full_h = abs(y1 - y0)
            view_w = full_w / factor
            view_h = full_h / factor

            if center_x is None or center_y is None:
                cur_x0, cur_x1 = ax.get_xlim()
                cur_y0, cur_y1 = ax.get_ylim()
                cx = 0.5 * (cur_x0 + cur_x1)
                cy = 0.5 * (cur_y0 + cur_y1)
            else:
                cx = center_x
                cy = center_y

            left = float(np.clip(cx - view_w / 2.0, min(x0, x1), max(x0, x1) - view_w))
            bottom = float(np.clip(cy - view_h / 2.0, min(y0, y1), max(y0, y1) - view_h))
            right = left + view_w
            top = bottom + view_h

            if x1 >= x0:
                ax.set_xlim(left, right)
            else:
                ax.set_xlim(right, left)
            if y1 >= y0:
                ax.set_ylim(bottom, top)
            else:
                ax.set_ylim(top, bottom)

        self._image_zoom_factor = factor
        self._sync_image_slider_from_zoom()
        self.canvas.draw_idle()

    def _on_scroll_zoom(self, event):
        ax = event.inaxes or self._axis_at_event(event)
        if ax is None:
            return

        if event.button == 'up':
            zoom_step = 1.2
        elif event.button == 'down':
            zoom_step = 1 / 1.2
        else:
            return

        if self.mode == 'image':
            next_factor = self._image_zoom_factor * zoom_step
            self.apply_image_zoom(next_factor, center_x=event.xdata, center_y=event.ydata)
            return

        if self.mode == 'xwindow':
            axis_region = self._axis_region(event, ax)
            if axis_region == 'y':
                self._zoom_y_limits(ax, zoom_step, center_y=event.ydata)
                self.canvas.draw_idle()
                return

            if self._independent_x_axes:
                x0, x1 = ax.get_xlim()
                width = x1 - x0
                if abs(width) < 1e-12:
                    return
                new_width = width / zoom_step
                center = event.xdata if event.xdata is not None else 0.5 * (x0 + x1)
                ax.set_xlim(float(center) - new_width / 2.0, float(center) + new_width / 2.0)
                self.canvas.draw_idle()
                return

            ref_ax = self._x_axes[0] if self._x_axes else ax
            current_width = ref_ax.get_xlim()[1] - ref_ax.get_xlim()[0]
            total = self._x_total
            new_width = float(np.clip(current_width / zoom_step, min(total, 0.05), total))
            center_abs = event.xdata if event.xdata is not None else (ref_ax.get_xlim()[0] + ref_ax.get_xlim()[1]) / 2.0
            center_rel = float(center_abs) - self._x_origin
            new_start = float(np.clip(center_rel - new_width / 2.0, 0.0, max(0.0, total - new_width)))
            self._x_window = new_width
            self._x_start = new_start
            self._apply_x_limits(new_start, new_start + new_width)
            self._sync_slider_from_x_window()
            self._notify_xwindow_changed()
            self.canvas.draw_idle()

    def _axis_at_event(self, event):
        for ax in self.figure.axes:
            bbox = ax.bbox
            x_near = bbox.x0 - 48 <= event.x <= bbox.x1 + 8
            y_near = bbox.y0 - 36 <= event.y <= bbox.y1 + 8
            if x_near and y_near:
                return ax
        return None

    def _axis_region(self, event, ax) -> str:
        bbox = ax.bbox
        if bbox.x0 - 48 <= event.x < bbox.x0 and bbox.y0 <= event.y <= bbox.y1:
            return 'y'
        return 'x'

    def _zoom_y_limits(self, ax, zoom_step: float, center_y: Optional[float] = None):
        y0, y1 = ax.get_ylim()
        height = y1 - y0
        if abs(height) < 1e-12:
            return
        new_height = height / zoom_step
        if center_y is None:
            center_y = 0.5 * (y0 + y1)
        ax.set_ylim(center_y - new_height / 2.0, center_y + new_height / 2.0)

    def _apply_x_limits(self, left: float, right: float):
        axes = self._x_axes if self._x_axes else list(self.figure.axes)
        left_abs = self._x_origin + float(left)
        right_abs = self._x_origin + float(right)
        for ax in axes:
            ax.set_xlim(left_abs, right_abs)

    def _notify_xwindow_changed(self):
        if self.on_xwindow_changed is not None:
            self.on_xwindow_changed()

    def _toolbar_active(self) -> bool:
        return False

    def _on_button_press(self, event):
        if self._toolbar_active() or event.inaxes is None or event.button != 1:
            return
        if event.dblclick:
            self.reset_view()
            return

        if self.mode == 'xwindow':
            if self._independent_x_axes:
                self._drag_state = {
                    'mode': 'axis_xwindow',
                    'ax': event.inaxes,
                    'press_event_x': event.x,
                    'press_axis_width': max(float(event.inaxes.bbox.width), 1.0),
                    'xlim': tuple(event.inaxes.get_xlim()),
                }
                return
            self._drag_state = {
                'mode': 'xwindow',
                'press_event_x': event.x,
                'press_axis_width': max(float(event.inaxes.bbox.width), 1.0),
                'x_start': self._x_start,
                'x_window': self._x_window,
            }
            return

        if self.mode == 'image':
            self._drag_state = {
                'mode': 'image',
                'ax': event.inaxes,
                'press_xy': event.inaxes.transData.inverted().transform((event.x, event.y)),
                'limits': [(ax, tuple(ax.get_xlim()), tuple(ax.get_ylim())) for ax in self.figure.axes],
            }

    def _on_button_release(self, event):
        if self._drag_state is not None:
            self._on_motion(event)
        self._drag_state = None

    def _on_motion(self, event):
        if self._toolbar_active() or self._drag_state is None:
            return

        if self._drag_state['mode'] == 'xwindow':
            if event.x is None:
                return
            dx = float(event.x - self._drag_state['press_event_x'])
            x_delta = dx / self._drag_state['press_axis_width'] * self._drag_state['x_window']
            travel = max(self._x_total - self._x_window, 0.0)
            self._x_start = float(np.clip(self._drag_state['x_start'] - x_delta, 0.0, travel))
            self._apply_x_limits(self._x_start, self._x_start + self._x_window)
            self._sync_slider_from_x_window()
            self._notify_xwindow_changed()
            self.canvas.draw_idle()
            return

        if self._drag_state['mode'] == 'axis_xwindow':
            if event.x is None:
                return
            ax = self._drag_state['ax']
            x0, x1 = self._drag_state['xlim']
            width = x1 - x0
            dx = float(event.x - self._drag_state['press_event_x'])
            x_delta = dx / self._drag_state['press_axis_width'] * width
            ax.set_xlim(x0 - x_delta, x1 - x_delta)
            self.canvas.draw_idle()
            return

        if self._drag_state['mode'] == 'image' and event.inaxes is not None:
            ax0 = self._drag_state['ax']
            cur_xy = ax0.transData.inverted().transform((event.x, event.y))
            dx = float(cur_xy[0] - self._drag_state['press_xy'][0])
            dy = float(cur_xy[1] - self._drag_state['press_xy'][1])
            for ax, xlim, ylim in self._drag_state['limits']:
                ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
                ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.canvas.draw_idle()


class BackgroundTask(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, callback, with_progress: bool = False):
        super().__init__()
        self.callback = callback
        self.with_progress = with_progress

    def run(self):
        try:
            if self.with_progress:
                result = self.callback(self.progress.emit)
            else:
                result = self.callback()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class NeuroBoxGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NeuroBox --trace visualize')
        self.resize(1280, 760)
        self.setMinimumSize(900, 620)

        self.state = GUIState()
        self.vpy: Optional[dict] = None
        self._roi_labels: List[str] = []
        self._combine_radio: dict[str, QRadioButton] = {}
        self._avg_radio: dict[str, QCheckBox] = {}
        self.trace_rows: List[TraceControlRow] = []
        self.image_layers: List[ImageLayerControlRow] = []
        self._image_layer_counter = 0
        self._last_image_arrays: dict[str, np.ndarray] = {}
        self._image_axis_mask_sources: dict[Any, str] = {}
        self._image_canvas_target: Optional[tuple[int, int, tuple[tuple[int, int], ...]]] = None
        self._image_projection_cache: dict[tuple[Any, ...], tuple[np.ndarray, int]] = {}
        self._direct_trace_cache: dict[str, np.ndarray] = {}
        self._computed_trace_cache: dict[tuple[Any, ...], tuple[np.ndarray, Optional[np.ndarray]]] = {}
        self._trace_result_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self._updating_event_trace_panel = False
        self._ensuring_volpy_mapping = False
        self._active_thread: Optional[QThread] = None
        self._active_worker: Optional[BackgroundTask] = None
        self._pending_load_label = ''
        self._pending_load_done_message = ''
        self._pending_job_done_message = ''
        self._pending_job_success = None
        self._loading_pipeline = False
        self._pipeline_denoise_queue: list[str] = []
        self._headless_pipeline_path = 'none'
        self._avg_subplot_axes: list[Any] = []
        self._avg_subplot_export_names: list[str] = []
        self._render_reset_trace_view = False
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_all_now)
        self._image_play_timer = QTimer(self)
        self._image_play_timer.timeout.connect(self._advance_image_frame)
        self._trace_play_timer = QTimer(self)
        self._trace_play_timer.timeout.connect(self._advance_trace_window)
        self.selected_roi_indices: List[int] = []

        self.combine_mode = 'individual'
        self.avg_mode = 'event'
        self.avg_modes: set[str] = {'event'}

        self._build_menu()
        self._build_layout()
        self._update_image_info_table()
        self._set_status('GUI initialized.')

    # ------------------------------ UI scaffold ------------------------------
    def _build_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')

        load_folder_action = QAction('Load Bruker folder...', self)
        load_folder_action.triggered.connect(self.load_folder_callback)
        file_menu.addAction(load_folder_action)

        load_table_action = QAction('Load Femtonics xlsx file...', self)
        load_table_action.triggered.connect(self.load_femtonics_xlsx_callback)
        file_menu.addAction(load_table_action)

        load_volpy_action = QAction('Load VolPy folder...', self)
        load_volpy_action.triggered.connect(self.load_volpy_folder_callback)
        file_menu.addAction(load_volpy_action)

        file_menu.addSeparator()

        export_image_action = QAction('Export image panel...', self)
        export_image_action.triggered.connect(lambda: self.export_panel_callback('image'))
        file_menu.addAction(export_image_action)

        export_trace_action = QAction('Export trace panel...', self)
        export_trace_action.triggered.connect(lambda: self.export_panel_callback('trace'))
        file_menu.addAction(export_trace_action)

        export_avg_action = QAction('Export average panel...', self)
        export_avg_action.triggered.connect(lambda: self.export_panel_callback('average'))
        file_menu.addAction(export_avg_action)

        file_menu.addSeparator()
        quit_action = QAction('Quit', self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = menu_bar.addMenu('Help')
        about_action = QAction('About NeuroBox', self)
        about_action.triggered.connect(self.about_callback)
        help_menu.addAction(about_action)

        help_action = QAction('Shortcuts / Help', self)
        help_action.triggered.connect(lambda: self.developing('HelpDialog'))
        help_menu.addAction(help_action)

    def _build_layout(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 0, 4)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self.main_splitter, stretch=1)

        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFixedWidth(430)
        self.left_scroll.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.left_frame = QWidget()
        self.left_frame.setMinimumWidth(390)
        self.left_frame.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.left_layout = QVBoxLayout(self.left_frame)
        self.left_layout.setContentsMargins(2, 2, 2, 2)
        self.left_layout.setSpacing(4)
        self.left_scroll.setWidget(self.left_frame)

        self.plot_scroll = QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.plot_frame = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_frame)
        self.plot_layout.setContentsMargins(0, 0, 5, 0)
        self.plot_layout.setSpacing(2)
        self.plot_scroll.setWidget(self.plot_frame)

        self.main_splitter.addWidget(self.left_scroll)
        self.main_splitter.addWidget(self.plot_scroll)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([430, 850])

        self._build_controls()
        self._build_plot_area()

        status = QStatusBar()
        self.setStatusBar(status)

    def _build_controls(self):
        data_box = QGroupBox('Data')
        data_layout = QVBoxLayout(data_box)
        self.source_label = QLabel('No Data Loaded')
        self.source_label.setWordWrap(True)
        data_layout.addWidget(QLabel('Source'))
        data_layout.addWidget(self.source_label)

        btn_row = QHBoxLayout()
        load_btn = QPushButton('Load')
        load_menu = QMenu(load_btn)
        load_folder_action = load_menu.addAction('Bruker folder')
        load_folder_action.triggered.connect(self.load_folder_callback)
        load_table_action = load_menu.addAction('Femtonics xlsx file')
        load_table_action.triggered.connect(self.load_femtonics_xlsx_callback)
        load_volpy_action = load_menu.addAction('VolPy folder')
        load_volpy_action.triggered.connect(self.load_volpy_folder_callback)
        load_btn.setMenu(load_menu)
        source_more_btn = QPushButton('...')
        source_more_btn.setToolTip('Show image and source information')
        source_more_btn.clicked.connect(self.show_image_info_dialog)
        btn_row.addWidget(load_btn, stretch=1)
        btn_row.addWidget(source_more_btn)
        data_layout.addLayout(btn_row)

        pipeline_row = QHBoxLayout()
        save_pipeline_btn = QPushButton('Save pipeline')
        apply_pipeline_btn = QPushButton('Apply pipeline')
        save_pipeline_btn.clicked.connect(self.save_pipeline_callback)
        apply_pipeline_btn.clicked.connect(self.apply_pipeline_callback)
        pipeline_row.addWidget(save_pipeline_btn)
        pipeline_row.addWidget(apply_pipeline_btn)
        data_layout.addLayout(pipeline_row)

        polarity_row = QHBoxLayout()
        self.data_polarity_button = QPushButton('pos')
        self.data_polarity_button.setCheckable(True)
        self.data_polarity_button.setToolTip('Use positive or negative data polarity for loading and extraction.')
        self.data_polarity_button.toggled.connect(self._on_data_polarity_toggled)
        self.data_raw_checkbox = QCheckBox('raw')
        self.data_raw_checkbox.setToolTip('When negative polarity is loaded, show and extract traces from the unreversed data without recalculating pixel weights.')
        self.data_raw_checkbox.stateChanged.connect(lambda _state: self._on_data_raw_toggled())
        polarity_row.addWidget(self.data_polarity_button)
        polarity_row.addWidget(self.data_raw_checkbox)
        polarity_row.addStretch(1)
        data_layout.addLayout(polarity_row)
        self.left_layout.addWidget(data_box)

        image_box = QGroupBox('Image panel')
        image_layout = QVBoxLayout(image_box)
        self.image_layer_layout = QVBoxLayout()
        image_layout.addLayout(self.image_layer_layout)
        self._build_raw_image_layer_row()

        image_layer_label = QLabel('-- image layer --')
        image_layer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(image_layer_label)
        self.extra_image_layer_layout = QVBoxLayout()
        image_layout.addLayout(self.extra_image_layer_layout)
        add_image_btn = QPushButton('+ Add image layer')
        add_image_btn.clicked.connect(self.add_image_layer)
        image_layout.addWidget(add_image_btn)
        self.left_layout.addWidget(image_box)

        self._build_event_panel()

        trace_box = QGroupBox('Trace panel')
        trace_layout = QVBoxLayout(trace_box)
        win_row = QHBoxLayout()
        self.trace_window_label = QLabel('Window(s)')
        win_row.addWidget(self.trace_window_label)
        self.trace_window_edit = QLineEdit('5')
        self.trace_window_edit.setMaximumWidth(56)
        self.trace_window_edit.editingFinished.connect(self.render_all)
        win_row.addWidget(self.trace_window_edit)
        trace_add_btn = QPushButton('+ Trace')
        trace_add_btn.clicked.connect(self.add_trace_row)
        win_row.addWidget(trace_add_btn)
        self.trace_fold_all_button = QPushButton('Fold all')
        self.trace_fold_all_button.clicked.connect(self.toggle_all_trace_rows)
        win_row.addWidget(self.trace_fold_all_button)
        trace_layout.addLayout(win_row)

        self.trace_rows_layout = QVBoxLayout()
        trace_layout.addLayout(self.trace_rows_layout)
        self.add_trace_row()
        self.left_layout.addWidget(trace_box)

        avg_box = QGroupBox('Average panel')
        avg_layout = QVBoxLayout(avg_box)
        for value, text in AVG_PANEL_MODES.items():
            checkbox = QCheckBox(text)
            checkbox.setChecked(value in self.avg_modes)
            checkbox.stateChanged.connect(lambda state, v=value: self._on_avg_mode_changed(v, state == Qt.CheckState.Checked.value))
            avg_layout.addWidget(checkbox)
            self._avg_radio[value] = checkbox

        form = QGridLayout()
        form.addWidget(QLabel('Stim pre/post(s)'), 0, 0)
        self.avg_pre_edit = QLineEdit('-0.2')
        self.avg_post_edit = QLineEdit('0.5')
        self.avg_pre_edit.setMaximumWidth(64)
        self.avg_post_edit.setMaximumWidth(64)
        self.avg_pre_edit.editingFinished.connect(self.render_all)
        self.avg_post_edit.editingFinished.connect(self.render_all)
        form.addWidget(self.avg_pre_edit, 0, 1)
        form.addWidget(self.avg_post_edit, 0, 2)

        form.addWidget(QLabel('Wave pre/post(s)'), 1, 0)
        self.waveform_pre_edit = QLineEdit('-0.025')
        self.waveform_post_edit = QLineEdit('0.025')
        self.waveform_pre_edit.setMaximumWidth(64)
        self.waveform_post_edit.setMaximumWidth(64)
        self.waveform_pre_edit.editingFinished.connect(self.render_all)
        self.waveform_post_edit.editingFinished.connect(self.render_all)
        form.addWidget(self.waveform_pre_edit, 1, 1)
        form.addWidget(self.waveform_post_edit, 1, 2)
        avg_layout.addLayout(form)
        avg_layout.addWidget(QLabel('Use traces'))
        self.avg_trace_list = QListWidget()
        self.avg_trace_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.avg_trace_list.setMinimumHeight(70)
        self.avg_trace_list.setMaximumHeight(140)
        self.avg_trace_list.itemSelectionChanged.connect(lambda: self._render_average_panel())
        avg_layout.addWidget(self.avg_trace_list)
        avg_layout.addWidget(QLabel('Use events'))
        self.avg_event_list = QListWidget()
        self.avg_event_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.avg_event_list.setMinimumHeight(50)
        self.avg_event_list.setMaximumHeight(110)
        self.avg_event_list.itemSelectionChanged.connect(lambda: self._render_average_panel())
        avg_layout.addWidget(self.avg_event_list)
        self.left_layout.addWidget(avg_box)

        roi_box = QGroupBox('ROI')
        roi_layout = QVBoxLayout(roi_box)
        roi_toggle_row = QHBoxLayout()
        self.only_cells_checkbox = QCheckBox('Only cells')
        self.only_cells_checkbox.setChecked(False)
        self.only_cells_checkbox.stateChanged.connect(self._refresh_roi_list)
        self.show_masks_checkbox = QCheckBox('Mask')
        self.show_masks_checkbox.setChecked(True)
        self.show_masks_checkbox.stateChanged.connect(lambda _state: self._render_image_panel())
        roi_toggle_row.addWidget(self.only_cells_checkbox)
        roi_toggle_row.addWidget(self.show_masks_checkbox)
        roi_toggle_row.addStretch(1)
        roi_layout.addLayout(roi_toggle_row)
        self.roi_list = ROIListWidget()
        self.roi_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.roi_list.setMinimumHeight(160)
        self.roi_list.itemSelectionChanged.connect(self._on_roi_list_selection_changed)
        self.active_roi_label = QLabel('Selected ROI: none')
        roi_layout.addWidget(self.active_roi_label)
        roi_layout.addWidget(self.roi_list, stretch=1)

        roi_layout.addWidget(QLabel('Multi-ROI display'))
        for value, label in COMBINE_MODES.items():
            radio = QRadioButton(label)
            radio.setChecked(value == self.combine_mode)
            radio.toggled.connect(lambda checked, v=value: self._on_combine_mode_changed(v, checked))
            roi_layout.addWidget(radio)
            self._combine_radio[value] = radio

        roi_layout.addWidget(QLabel('Mask color'))
        self.mask_color_combo = NoWheelComboBox()
        for value, label in MASK_COLOR_OPTIONS:
            self.mask_color_combo.addItem(label, value)
        self.mask_color_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.mask_color_combo.setMinimumContentsLength(12)
        self.mask_color_combo.currentIndexChanged.connect(lambda _idx: self.render_all())
        roi_layout.addWidget(self.mask_color_combo)

        action_row = QHBoxLayout()
        refresh_btn = QPushButton('Refresh view')
        refresh_btn.clicked.connect(self.render_all)
        developing_btn = QPushButton('Developing')
        developing_btn.clicked.connect(lambda: self.developing('Reserved ROI tool'))
        action_row.addWidget(refresh_btn)
        action_row.addWidget(developing_btn)
        roi_layout.addLayout(action_row)

        self.roi_box = roi_box

    def _build_event_panel(self):
        event_box = QGroupBox('Event panel')
        event_layout = QVBoxLayout(event_box)
        self.event_table = QTableWidget(0, 3)
        self.event_table.setHorizontalHeaderLabels(['Source', 'Label', 'Color'])
        self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.event_table.verticalHeader().setVisible(False)
        self.event_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.event_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.event_table.setToolTip('Events are stored internally as frame ranges; start/end frames are not shown in this table.')
        self.event_table.itemChanged.connect(self._on_event_table_item_changed)
        self.event_table.itemSelectionChanged.connect(self._on_event_table_selection_changed)
        self.event_table.itemDoubleClicked.connect(self._on_event_table_item_double_clicked)
        self.event_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        event_layout.addWidget(self.event_table, stretch=1)

        event_btn_row = QHBoxLayout()
        self.event_add_source_combo = NoWheelComboBox()
        for source in ('npy file', 'trace', '...'):
            self.event_add_source_combo.addItem(source)
        self.event_add_source_combo.setToolTip('.npy files may contain a 2 x n seconds array or a saved event dictionary.')
        add_event_btn = QPushButton('Add')
        remove_event_btn = QPushButton('Remove')
        save_event_btn = QPushButton('Save')
        add_event_btn.clicked.connect(self.add_event_callback)
        remove_event_btn.clicked.connect(self.remove_selected_event_callback)
        save_event_btn.clicked.connect(self.save_events_callback)
        event_btn_row.addWidget(self.event_add_source_combo, stretch=1)
        event_btn_row.addWidget(add_event_btn)
        event_btn_row.addWidget(remove_event_btn)
        event_btn_row.addWidget(save_event_btn)
        event_layout.addLayout(event_btn_row)

        self.trace_event_box = QGroupBox('Trace event extractor')
        trace_event_layout = QGridLayout(self.trace_event_box)
        trace_event_layout.setContentsMargins(4, 4, 4, 4)
        trace_event_layout.setHorizontalSpacing(4)
        trace_event_layout.setVerticalSpacing(3)
        self.event_trace_source_combo = NoWheelComboBox()
        self.event_trace_source_combo.setToolTip('Detect events from raw trace or one current trace row for the first selected ROI.')
        self.event_trace_action_combo = NoWheelComboBox()
        self.event_trace_action_combo.addItems(['raw', 'discard', 'replace'])
        self.event_trace_action_combo.setToolTip('raw detects and labels events without changing traces; discard removes event frames; replace holds previous values.')
        self.event_trace_direction_combo = NoWheelComboBox()
        self.event_trace_direction_combo.addItems(['above', 'below', 'between'])
        self.event_trace_direction_combo.currentIndexChanged.connect(lambda _idx: self._update_trace_event_mode_controls())
        self.event_trace_value_edit = QLineEdit('0')
        self.event_trace_interval_edit = QLineEdit('1')
        self.event_trace_adjacent_edit = QLineEdit('0')
        self.event_trace_merge_edit = QLineEdit('3')
        self.event_trace_diff_edit = QLineEdit('0')
        self.event_trace_from_edit = QLineEdit('0')
        self.event_trace_to_edit = QLineEdit('0')
        self.event_trace_value_edit.setToolTip('intensity')
        self.event_trace_interval_edit.setToolTip('frame')
        self.event_trace_adjacent_edit.setToolTip('frame')
        self.event_trace_merge_edit.setToolTip('frame')
        self.event_trace_diff_edit.setToolTip('intensity')
        self.event_trace_from_edit.setToolTip('frame')
        self.event_trace_to_edit.setToolTip('frame')
        self.event_trace_apply_button = QPushButton('Apply')
        self.event_trace_apply_button.clicked.connect(self._on_trace_event_params_changed)
        for edit in (
            self.event_trace_value_edit,
            self.event_trace_interval_edit,
            self.event_trace_adjacent_edit,
            self.event_trace_merge_edit,
            self.event_trace_diff_edit,
            self.event_trace_from_edit,
            self.event_trace_to_edit,
        ):
            edit.setMaximumWidth(58)
        trace_event_layout.addWidget(QLabel('Source'), 0, 0)
        trace_event_layout.addWidget(self.event_trace_source_combo, 0, 1, 1, 5)
        trace_event_layout.addWidget(self.event_trace_action_combo, 1, 0)
        trace_event_layout.addWidget(self.event_trace_direction_combo, 1, 1)
        trace_event_layout.addWidget(QLabel('value'), 1, 2)
        trace_event_layout.addWidget(self.event_trace_value_edit, 1, 3)
        trace_event_layout.addWidget(QLabel('interval'), 1, 4)
        trace_event_layout.addWidget(self.event_trace_interval_edit, 1, 5)
        trace_event_layout.addWidget(QLabel('adjacent'), 2, 0)
        trace_event_layout.addWidget(self.event_trace_adjacent_edit, 2, 1)
        trace_event_layout.addWidget(QLabel('merge'), 2, 2)
        trace_event_layout.addWidget(self.event_trace_merge_edit, 2, 3)
        trace_event_layout.addWidget(QLabel('diff'), 2, 4)
        trace_event_layout.addWidget(self.event_trace_diff_edit, 2, 5)
        trace_event_layout.addWidget(QLabel('from'), 3, 0)
        trace_event_layout.addWidget(self.event_trace_from_edit, 3, 1)
        trace_event_layout.addWidget(QLabel('to'), 3, 2)
        trace_event_layout.addWidget(self.event_trace_to_edit, 3, 3)
        trace_event_layout.addWidget(self.event_trace_apply_button, 4, 0, 1, 6)
        event_layout.addWidget(self.trace_event_box)
        self._set_trace_event_controls_enabled(False)
        self._update_trace_event_mode_controls()
        self.left_layout.addWidget(event_box)

    def _build_raw_image_layer_row(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QGridLayout(frame)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(3)
        layout.setColumnStretch(1, 1)
        model = NoWheelComboBox()
        model.addItems(IMAGE_LAYER_MODELS)
        model.setCurrentIndex(self._combo_index_for_text(model, 'Raw'))
        model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        model.setMinimumContentsLength(7)
        layout.addWidget(model, 0, 0)

        visible = QCheckBox('visible')
        visible.setChecked(True)
        visible.stateChanged.connect(lambda _state: self._on_image_layer_trace_source_changed())
        layout.addWidget(visible, 0, 1)
        export = QPushButton('Export')
        export.setToolTip('Export this image layer as TIFF.')
        layout.addWidget(export, 0, 2)

        mode = NoWheelComboBox()
        for value, label in IMAGE_LAYER_MODES:
            mode.addItem(label, value)
        mode.setCurrentIndex(0)
        mode.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        mode.setMinimumContentsLength(7)
        mode.currentIndexChanged.connect(lambda _idx: self.render_all())
        layout.addWidget(QLabel('Mode'), 1, 0)
        layout.addWidget(mode, 1, 1, 1, 2)

        mask_source = NoWheelComboBox()
        for value, label in MASK_SOURCE_OPTIONS:
            mask_source.addItem(label, value)
        mask_source.setCurrentIndex(self._combo_index_for_data(mask_source, 'suite2p'))
        mask_source.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        mask_source.setMinimumContentsLength(7)
        mask_source.currentIndexChanged.connect(lambda _idx: self._on_image_layer_trace_source_changed())
        layout.addWidget(QLabel('Mask'), 2, 0)
        layout.addWidget(mask_source, 2, 1, 1, 2)

        nframes = QLabel('0 frames')
        layout.addWidget(nframes, 3, 0, 1, 3)
        summary = QLabel('Frames: 0 | Size: - | Rate: -')
        summary.setWordWrap(True)
        layout.addWidget(summary, 4, 0, 1, 3)

        row = ImageLayerControlRow(
            layer_id='raw',
            widget=frame,
            model_combo=model,
            visible_checkbox=visible,
            mode_combo=mode,
            mask_source_combo=mask_source,
            nframes_label=nframes,
            summary_label=summary,
            remove_button=None,
            export_button=export,
        )
        self.image_layers.append(row)
        self.image_layer_layout.addWidget(frame)
        model.currentIndexChanged.connect(lambda _idx, r=row: self._on_image_layer_model_changed(r))
        export.clicked.connect(lambda _checked=False, r=row: self.export_image_layer_callback(r))
        self._refresh_mask_target_list()

    def add_image_layer(self):
        self._image_layer_counter += 1
        layer_id = f'image_{self._image_layer_counter}'
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QGridLayout(frame)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(3)
        layout.setColumnStretch(1, 1)

        model = NoWheelComboBox()
        model.addItems(IMAGE_LAYER_MODELS)
        model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        model.setMinimumContentsLength(7)
        layout.addWidget(model, 0, 0)

        visible = QCheckBox('visible')
        visible.setChecked(True)
        layout.addWidget(visible, 0, 1)

        mode = NoWheelComboBox()
        for value, label in IMAGE_LAYER_MODES:
            mode.addItem(label, value)
        mode.setCurrentIndex(0)
        mode.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        mode.setMinimumContentsLength(7)
        layout.addWidget(QLabel('Mode'), 1, 0)
        layout.addWidget(mode, 1, 1, 1, 2)

        mask_source = NoWheelComboBox()
        for value, label in MASK_SOURCE_OPTIONS:
            mask_source.addItem(label, value)
        mask_source.setCurrentIndex(self._combo_index_for_data(mask_source, 'suite2p'))
        mask_source.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        mask_source.setMinimumContentsLength(7)
        layout.addWidget(QLabel('Mask'), 2, 0)
        layout.addWidget(mask_source, 2, 1, 1, 2)

        remove = QPushButton('Remove')
        export = QPushButton('Export')
        export.setToolTip('Export this image layer as TIFF.')
        layout.addWidget(export, 0, 2)
        layout.addWidget(remove, 0, 3)

        nframes = QLabel('0 frames')
        layout.addWidget(nframes, 3, 0, 1, 3)
        summary = QLabel('Frames: 0 | Size: - | Rate: -')
        summary.setWordWrap(True)
        layout.addWidget(summary, 4, 0, 1, 3)

        row = ImageLayerControlRow(
            layer_id=layer_id,
            widget=frame,
            model_combo=model,
            visible_checkbox=visible,
            mode_combo=mode,
            mask_source_combo=mask_source,
            nframes_label=nframes,
            summary_label=summary,
            remove_button=remove,
            export_button=export,
        )
        self.image_layers.append(row)
        self.extra_image_layer_layout.addWidget(frame)

        model.currentIndexChanged.connect(lambda _idx, r=row: self._on_image_layer_model_changed(r))
        visible.stateChanged.connect(lambda _state: self._on_image_layer_trace_source_changed())
        mode.currentIndexChanged.connect(lambda _idx: self.render_all())
        mask_source.currentIndexChanged.connect(lambda _idx: self._on_image_layer_trace_source_changed())
        export.clicked.connect(lambda _checked=False, r=row: self.export_image_layer_callback(r))
        remove.clicked.connect(lambda _checked=False, r=row: self.remove_image_layer(r))

        self._refresh_mask_target_list()
        self._on_image_layer_model_changed(row)

    def remove_image_layer(self, row: ImageLayerControlRow):
        if row not in self.image_layers or row.layer_id == 'raw':
            return
        self.image_layers.remove(row)
        row.widget.setParent(None)
        row.widget.deleteLater()
        self._refresh_mask_target_list()
        self.render_all()

    def _on_image_layer_trace_source_changed(self):
        self._computed_trace_cache.clear()
        self._invalidate_trace_cache()
        self._refresh_trace_source_options()
        self._refresh_event_trace_source_options()
        self.render_all()

    def add_trace_row(self):
        box = QGroupBox(f'Trace {len(self.trace_rows) + 1}')
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        top = QGridLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setHorizontalSpacing(4)
        top.setVerticalSpacing(3)
        top.setColumnStretch(1, 1)
        visible = QCheckBox('visible')
        visible.setChecked(True)
        top.addWidget(visible, 0, 0)
        source = NoWheelComboBox()
        source.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        source.setMinimumContentsLength(12)
        remove = QPushButton('Remove')
        top.addWidget(remove, 0, 1, 1, 2)
        fold = QPushButton('Fold')
        fold.setCheckable(True)
        top.addWidget(fold, 0, 3)
        top.addWidget(QLabel('Source'), 1, 0)
        top.addWidget(source, 1, 1, 1, 3)
        layout.addLayout(top)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(4)
        pipeline = QGridLayout()
        pipeline.setContentsMargins(0, 0, 0, 0)
        pipeline.setHorizontalSpacing(4)
        pipeline.setVerticalSpacing(3)
        lowpass = QCheckBox('low-pass')
        lowpass_edit = QLineEdit('1.0')
        lowpass_edit.setMaximumWidth(58)
        pipeline.addWidget(lowpass, 0, 0)
        pipeline.addWidget(QLabel('Hz'), 0, 1)
        pipeline.addWidget(lowpass_edit, 0, 2)

        highpass = QCheckBox('high-pass')
        highpass_edit = QLineEdit('10.0')
        highpass_edit.setMaximumWidth(58)
        pipeline.addWidget(highpass, 1, 0)
        pipeline.addWidget(QLabel('Hz'), 1, 1)
        pipeline.addWidget(highpass_edit, 1, 2)

        wavelet = QCheckBox('wavelet')
        wavelet_name = QLineEdit('sym4')
        wavelet_level = QLineEdit('4')
        wavelet_scale = QLineEdit('1.2')
        wavelet_name.setMaximumWidth(58)
        wavelet_level.setMaximumWidth(42)
        wavelet_scale.setMaximumWidth(58)
        wavelet_mode = NoWheelComboBox()
        wavelet_mode.addItems(['hard', 'soft'])
        wavelet_mode.setMaximumWidth(74)
        pipeline.addWidget(wavelet, 2, 0)
        pipeline.addWidget(QLabel('name'), 2, 1)
        pipeline.addWidget(wavelet_name, 2, 2)
        pipeline.addWidget(QLabel('level'), 2, 3)
        pipeline.addWidget(wavelet_level, 2, 4)
        pipeline.addWidget(QLabel('scale'), 3, 1)
        pipeline.addWidget(wavelet_scale, 3, 2)
        pipeline.addWidget(QLabel('mode'), 3, 3)
        pipeline.addWidget(wavelet_mode, 3, 4)

        pca_wavelet = QCheckBox('pca_wavelet')
        pca_wavelet.setToolTip('Use cal_wavelet.py')
        pca_wavelet_fmin = QLineEdit('1.0')
        pca_wavelet_fmax = QLineEdit(self._default_pca_wavelet_fmax_text())
        pca_wavelet_fmax.setProperty('autoFrameRate', True)
        pca_wavelet_fn = QLineEdit('100')
        pca_wavelet_params = QPushButton('...')
        pca_wavelet_params.setToolTip('Edit pca_wavelet parameters')
        pca_wavelet_fmin.setMaximumWidth(58)
        pca_wavelet_fmax.setMaximumWidth(76)
        pca_wavelet_fn.setMaximumWidth(58)
        pca_wavelet_params.setMaximumWidth(34)
        pipeline.addWidget(pca_wavelet, 4, 0)
        pipeline.addWidget(QLabel('f_min'), 4, 1)
        pipeline.addWidget(pca_wavelet_fmin, 4, 2)
        pipeline.addWidget(QLabel('f_max'), 4, 3)
        pipeline.addWidget(pca_wavelet_fmax, 4, 4)
        pipeline.addWidget(QLabel('f_n'), 5, 0)
        pipeline.addWidget(pca_wavelet_fn, 5, 1)
        pipeline.addWidget(pca_wavelet_params, 5, 2)
        pca_wavelet_params.setToolTip('Edit pca_wavelet parameters and show result figures for the first selected ROI.')

        snr = QCheckBox('snr')
        snr_window = QLineEdit('0.0125')
        snr_window.setMaximumWidth(72)
        pipeline.addWidget(snr, 6, 0)
        pipeline.addWidget(QLabel('window'), 6, 1)
        pipeline.addWidget(snr_window, 6, 2)

        volpy = QCheckBox('VolPy')
        volpy_combo = NoWheelComboBox()
        volpy_combo.addItems(TRACE_SOURCE_VOLPY_KEYS)
        volpy_combo.setMaximumWidth(86)
        pipeline.addWidget(volpy, 7, 0)
        pipeline.addWidget(volpy_combo, 7, 1, 1, 2)
        content_layout.addLayout(pipeline)

        baseline_box = QGroupBox('Baseline')
        baseline_layout = QGridLayout(baseline_box)
        baseline_layout.setContentsMargins(4, 4, 4, 4)
        baseline_layout.setHorizontalSpacing(4)
        baseline_layout.setVerticalSpacing(3)
        baseline_mode = NoWheelComboBox()
        baseline_mode.addItem('dF/F', 'dff')
        baseline_mode.addItem('subtract', 'subtract')
        baseline_mode.addItem('add', 'add')
        baseline_mode.addItem('raw', 'raw')
        baseline_mode.setMaximumWidth(96)
        baseline_layout.addWidget(QLabel('Mode'), 0, 0)
        baseline_layout.addWidget(baseline_mode, 0, 1, 1, 2)

        baseline_lowpass = QCheckBox('low-pass')
        baseline_lowpass_freq = QLineEdit('1.0')
        baseline_lowpass_freq.setMaximumWidth(58)
        baseline_layout.addWidget(baseline_lowpass, 1, 0)
        baseline_layout.addWidget(QLabel('Hz'), 1, 1)
        baseline_layout.addWidget(baseline_lowpass_freq, 1, 2)

        baseline_rolling = QCheckBox('rolling-base')
        baseline_rolling_mode = NoWheelComboBox()
        baseline_rolling_mode.addItems(['mean', 'median'])
        baseline_rolling_window = QLineEdit('4')
        baseline_rolling_mode.setMaximumWidth(78)
        baseline_rolling_window.setMaximumWidth(58)
        baseline_layout.addWidget(baseline_rolling, 2, 0)
        baseline_layout.addWidget(baseline_rolling_mode, 2, 1)
        baseline_layout.addWidget(QLabel('window'), 2, 2)
        baseline_layout.addWidget(baseline_rolling_window, 2, 3)

        baseline_polyfit = QCheckBox('polyfit')
        baseline_poly_order = QLineEdit('3')
        baseline_poly_order.setMaximumWidth(46)
        baseline_layout.addWidget(baseline_polyfit, 3, 0)
        baseline_layout.addWidget(QLabel('order'), 3, 1)
        baseline_layout.addWidget(baseline_poly_order, 3, 2)

        baseline_savgol = QCheckBox('savgol fit')
        baseline_savgol_window = QLineEdit('1.0')
        baseline_savgol_order = QLineEdit('3')
        baseline_savgol_window.setMaximumWidth(58)
        baseline_savgol_order.setMaximumWidth(46)
        baseline_layout.addWidget(baseline_savgol, 4, 0)
        baseline_layout.addWidget(QLabel('window'), 4, 1)
        baseline_layout.addWidget(baseline_savgol_window, 4, 2)
        baseline_layout.addWidget(QLabel('order'), 4, 3)
        baseline_layout.addWidget(baseline_savgol_order, 4, 4)
        content_layout.addWidget(baseline_box)

        spike_box = QGroupBox('Spike detection')
        spike_layout = QGridLayout(spike_box)
        spike_layout.setContentsMargins(4, 4, 4, 4)
        spike_layout.setHorizontalSpacing(4)
        spike_layout.setVerticalSpacing(3)
        spike = QCheckBox('detect spikes')
        spike_layout.addWidget(spike, 0, 0)
        spike_method = NoWheelComboBox()
        for method in ['std', 'mad', 'snr', 't_res']:
            spike_method.addItem(method)
        spike_method.setMaximumWidth(78)
        spike_layout.addWidget(spike_method, 0, 1)
        spike_layout.addWidget(QLabel('k'), 0, 2)
        spike_k = QLineEdit('5.0')
        spike_k.setMaximumWidth(58)
        spike_layout.addWidget(spike_k, 0, 3)
        threshold = QCheckBox('thres')
        spike_layout.addWidget(threshold, 1, 0)
        waveform = QCheckBox('waveform')
        waveform_mode = NoWheelComboBox()
        waveform_mode.addItems(['raw', 'current', 'both'])
        waveform_mode.setMaximumWidth(78)
        export_waveform = QPushButton('Export')
        export_waveform.setToolTip('Export spike-centered waveform windows for this trace row.')
        spike_layout.addWidget(waveform, 1, 1)
        spike_layout.addWidget(waveform_mode, 1, 2)
        spike_layout.addWidget(export_waveform, 1, 3)
        content_layout.addWidget(spike_box)
        layout.addWidget(content)

        row = TraceControlRow(
            widget=box,
            content_widget=content,
            fold_button=fold,
            visible_checkbox=visible,
            source_combo=source,
            lowpass_checkbox=lowpass,
            lowpass_edit=lowpass_edit,
            highpass_checkbox=highpass,
            highpass_edit=highpass_edit,
            wavelet_checkbox=wavelet,
            wavelet_name_edit=wavelet_name,
            wavelet_level_edit=wavelet_level,
            wavelet_scale_edit=wavelet_scale,
            wavelet_mode_combo=wavelet_mode,
            pca_wavelet_checkbox=pca_wavelet,
            pca_wavelet_fmin_edit=pca_wavelet_fmin,
            pca_wavelet_fmax_edit=pca_wavelet_fmax,
            pca_wavelet_fn_edit=pca_wavelet_fn,
            pca_wavelet_param_button=pca_wavelet_params,
            pca_wavelet_cfg=self._default_pca_wavelet_cfg_dict(
                f_min=1.0,
                f_max=self._safe_float(pca_wavelet_fmax.text(), default=max(self.state.frame_rate, 1.0)),
                f_n=100,
            ),
            snr_checkbox=snr,
            snr_window_edit=snr_window,
            volpy_checkbox=volpy,
            volpy_combo=volpy_combo,
            baseline_mode_combo=baseline_mode,
            baseline_lowpass_checkbox=baseline_lowpass,
            baseline_lowpass_edit=baseline_lowpass_freq,
            baseline_rolling_checkbox=baseline_rolling,
            baseline_rolling_mode_combo=baseline_rolling_mode,
            baseline_rolling_window_edit=baseline_rolling_window,
            baseline_polyfit_checkbox=baseline_polyfit,
            baseline_poly_order_edit=baseline_poly_order,
            baseline_savgol_checkbox=baseline_savgol,
            baseline_savgol_window_edit=baseline_savgol_window,
            baseline_savgol_order_edit=baseline_savgol_order,
            spike_checkbox=spike,
            spike_method_combo=spike_method,
            spike_k_edit=spike_k,
            threshold_checkbox=threshold,
            waveform_checkbox=waveform,
            waveform_mode_combo=waveform_mode,
            waveform_export_button=export_waveform,
            remove_button=remove,
        )
        self.trace_rows.append(row)
        self.trace_rows_layout.addWidget(box)
        self._renumber_trace_rows()

        widgets = [
            visible, lowpass, highpass, wavelet, pca_wavelet, snr, volpy,
            baseline_lowpass, baseline_rolling, baseline_polyfit, baseline_savgol,
            threshold, waveform,
            source, wavelet_mode, volpy_combo, baseline_mode, baseline_rolling_mode, spike_method, waveform_mode,
            lowpass_edit, highpass_edit, wavelet_name, wavelet_level, wavelet_scale,
            pca_wavelet_fmin, pca_wavelet_fmax, pca_wavelet_fn, snr_window, baseline_lowpass_freq,
            baseline_rolling_window, baseline_poly_order, baseline_savgol_window,
            baseline_savgol_order, spike_k,
        ]
        for widget in widgets:
            if isinstance(widget, QCheckBox):
                widget.stateChanged.connect(lambda _state: self._on_trace_control_changed())
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(lambda _idx, r=row: self._on_trace_row_combo_changed(r))
            elif isinstance(widget, QLineEdit):
                widget.editingFinished.connect(self._on_trace_control_changed)
        pca_wavelet_fmax.editingFinished.connect(lambda e=pca_wavelet_fmax: e.setProperty('autoFrameRate', False))
        spike.stateChanged.connect(lambda state, r=row: self._on_spike_detect_toggled(r, state == Qt.CheckState.Checked.value))
        remove.clicked.connect(lambda _checked=False, r=row: self.remove_trace_row(r))
        pca_wavelet_params.clicked.connect(lambda _checked=False, r=row: self.show_pca_wavelet_param_dialog(r))
        export_waveform.clicked.connect(lambda _checked=False, r=row: self.export_waveform_callback(r))
        fold.toggled.connect(lambda checked, r=row: self._set_trace_row_folded(r, checked))

        self._refresh_trace_source_options()
        self._update_trace_remove_buttons()
        self._update_trace_fold_all_button()
        self._refresh_avg_trace_list()
        if hasattr(self, 'trace_widget'):
            self.render_all()

    def remove_trace_row(self, row: TraceControlRow):
        if row not in self.trace_rows:
            return
        if len(self.trace_rows) == 1:
            return
        self.trace_rows.remove(row)
        row.widget.setParent(None)
        row.widget.deleteLater()
        self._renumber_trace_rows()
        self._invalidate_trace_cache()
        self._update_trace_remove_buttons()
        self._update_trace_fold_all_button()
        self._refresh_avg_trace_list()
        self.render_all()

    def _update_trace_remove_buttons(self):
        removable = len(self.trace_rows) > 1
        for row in self.trace_rows:
            row.remove_button.setEnabled(removable)

    def _renumber_trace_rows(self):
        for idx, row in enumerate(self.trace_rows, start=1):
            row.widget.setTitle(f'Trace {idx}')

    def _set_trace_row_folded(self, row: TraceControlRow, folded: bool):
        row.content_widget.setVisible(not folded)
        row.fold_button.setText('Unfold' if folded else 'Fold')
        self._update_trace_fold_all_button()

    def toggle_all_trace_rows(self):
        if not self.trace_rows:
            return
        fold_all = any(not row.fold_button.isChecked() for row in self.trace_rows)
        for row in self.trace_rows:
            row.fold_button.setChecked(fold_all)
            self._set_trace_row_folded(row, fold_all)

    def _update_trace_fold_all_button(self):
        if not hasattr(self, 'trace_fold_all_button'):
            return
        all_folded = bool(self.trace_rows) and all(row.fold_button.isChecked() for row in self.trace_rows)
        self.trace_fold_all_button.setText('Unfold all' if all_folded else 'Fold all')

    def _invalidate_trace_cache(self):
        self._trace_result_cache.clear()

    def _on_trace_control_changed(self):
        self._invalidate_trace_cache()
        self.render_all()

    def _data_negative_enabled(self) -> bool:
        return bool(hasattr(self, 'data_polarity_button') and self.data_polarity_button.isChecked())

    def _data_raw_view_enabled(self) -> bool:
        return bool(
            self.state.negative_mode
            and hasattr(self, 'data_raw_checkbox')
            and self.data_raw_checkbox.isChecked()
        )

    def _negative_view_requested(self) -> bool:
        return bool(
            hasattr(self, 'data_polarity_button')
            and self.data_polarity_button.isChecked()
            and not (hasattr(self, 'data_raw_checkbox') and self.data_raw_checkbox.isChecked())
        )

    def _data_view_needs_reverse(self) -> bool:
        return bool(self.state.negative_mode != self._negative_view_requested())

    def _on_data_polarity_toggled(self, checked: bool):
        self.data_polarity_button.setText('neg' if checked else 'pos')
        self._direct_trace_cache.clear()
        self._computed_trace_cache.clear()
        self._invalidate_trace_cache()
        self.render_all()

    def _on_data_raw_toggled(self):
        self._direct_trace_cache.clear()
        self._computed_trace_cache.clear()
        self._invalidate_trace_cache()
        self.render_all(reset_trace_view=True)

    def _on_spike_detect_toggled(self, row: TraceControlRow, checked: bool):
        if checked:
            row.threshold_checkbox.blockSignals(True)
            row.waveform_checkbox.blockSignals(True)
            row.threshold_checkbox.setChecked(True)
            row.waveform_checkbox.setChecked(True)
            row.threshold_checkbox.blockSignals(False)
            row.waveform_checkbox.blockSignals(False)
        self._on_trace_control_changed()

    def _refresh_trace_source_options(self):
        options: list[tuple[str, Optional[str]]] = []
        for row in self.image_layers:
            if not row.visible_checkbox.isChecked():
                continue
            mask_source = row.mask_source_combo.currentData()
            options.append((f'{self._image_layer_label(row)} + {row.mask_source_combo.currentText()}', f'image:{row.layer_id}:{mask_source}'))
        if self.state.source_type == 'table' and self.state.traces:
            options.append(('xlsx raw trace', 'state:0'))
        if not options:
            options.append(('No visible image layer', None))

        for row in self.trace_rows:
            current = row.source_combo.currentData()
            row.source_combo.blockSignals(True)
            row.source_combo.clear()
            for label, value in options:
                row.source_combo.addItem(label, value)
            if current is not None:
                for idx in range(row.source_combo.count()):
                    if row.source_combo.itemData(idx) == current:
                        row.source_combo.setCurrentIndex(idx)
                        break
            row.source_combo.blockSignals(False)
            self._sync_trace_row_volpy_source_state(row, show_warning=False)
        self._refresh_event_trace_source_options()

    def _refresh_event_trace_source_options(self):
        if not hasattr(self, 'event_trace_source_combo'):
            return
        current = self.event_trace_source_combo.currentData()
        self.event_trace_source_combo.blockSignals(True)
        self.event_trace_source_combo.clear()
        if self.trace_rows:
            self.event_trace_source_combo.addItem('raw', 'event:raw')
        for idx, _row in enumerate(self.trace_rows):
            self.event_trace_source_combo.addItem(f'Trace {idx + 1}', f'event:trace:{idx}')
        if current is not None:
            for idx in range(self.event_trace_source_combo.count()):
                if self.event_trace_source_combo.itemData(idx) == current:
                    self.event_trace_source_combo.setCurrentIndex(idx)
                    break
        self.event_trace_source_combo.blockSignals(False)

    def _refresh_avg_trace_list(self, results: Optional[list[dict[str, Any]]] = None):
        if not hasattr(self, 'avg_trace_list'):
            return
        if results is None:
            results = self._get_active_trace_results()
        previous = {item.data(Qt.ItemDataRole.UserRole) for item in self.avg_trace_list.selectedItems()}
        self.avg_trace_list.blockSignals(True)
        self.avg_trace_list.clear()
        for idx, result in enumerate(results):
            item = QListWidgetItem(f'Trace {idx + 1}')
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.avg_trace_list.addItem(item)
            if not previous or idx in previous:
                item.setSelected(True)
        self.avg_trace_list.blockSignals(False)

    def get_selected_average_trace_indices(self, count: int) -> set[int]:
        if not hasattr(self, 'avg_trace_list') or self.avg_trace_list.count() == 0:
            return set(range(count))
        selected = {int(item.data(Qt.ItemDataRole.UserRole)) for item in self.avg_trace_list.selectedItems()}
        return selected if selected else set(range(count))

    def _combo_index_for_data(self, combo: QComboBox, value: str) -> int:
        for idx in range(combo.count()):
            if combo.itemData(idx) == value:
                return idx
        return 0

    def _combo_index_for_text(self, combo: QComboBox, value: str) -> int:
        for idx in range(combo.count()):
            if combo.itemText(idx) == value:
                return idx
        return 0

    def _set_combo_data(self, combo: QComboBox, value: Any) -> bool:
        for idx in range(combo.count()):
            if combo.itemData(idx) == value:
                combo.setCurrentIndex(idx)
                return True
        return False

    def _set_combo_text(self, combo: QComboBox, value: str) -> bool:
        for idx in range(combo.count()):
            if combo.itemText(idx) == value:
                combo.setCurrentIndex(idx)
                return True
        return False

    def _refresh_mask_target_list(self):
        if not hasattr(self, 'mask_target_list'):
            return
        previous = {item.data(Qt.ItemDataRole.UserRole) for item in self.mask_target_list.selectedItems()}
        if not previous:
            previous = {'raw'}
        self.mask_target_list.blockSignals(True)
        self.mask_target_list.clear()
        for row in self.image_layers:
            item = QListWidgetItem(self._image_layer_label(row))
            item.setData(Qt.ItemDataRole.UserRole, row.layer_id)
            self.mask_target_list.addItem(item)
            if row.layer_id in previous:
                item.setSelected(True)
        self.mask_target_list.blockSignals(False)

    def _image_layer_label(self, row: ImageLayerControlRow) -> str:
        model = self._image_layer_model(row)
        return f'{row.layer_id}: {model}'

    def _image_layer_model(self, row: ImageLayerControlRow) -> str:
        return row.model_combo.currentText() if row.model_combo is not None else 'Raw'

    def _on_image_layer_model_changed(self, row: ImageLayerControlRow):
        model = self._image_layer_model(row)
        if row.model_combo is not None and model == 'VolPy':
            if row.mask_source_combo.currentData() == 'volpy':
                row.mask_source_combo.setCurrentIndex(self._combo_index_for_data(row.mask_source_combo, 'suite2p'))
        elif row.model_combo is not None:
            row.mask_source_combo.setCurrentIndex(self._combo_index_for_data(row.mask_source_combo, 'suite2p'))
        if model in {'NoRMCorre', 'PMD'} and not self._loading_pipeline:
            self._show_image_layer_parameter_dialog(row, model)
        elif model == 'Local' and not self._loading_pipeline:
            self._select_local_image_layer_path(row)
        self._computed_trace_cache.clear()
        self._invalidate_trace_cache()
        self._refresh_mask_target_list()
        self._refresh_trace_source_options()
        self._refresh_event_trace_source_options()
        self.render_all()

    def _show_image_layer_parameter_dialog(self, row: ImageLayerControlRow, model: str):
        params = self._image_layer_params_for_row(row, model)
        sources = self._image_layer_input_sources(row)
        dialog = QDialog(self)
        dialog.setWindowTitle(f'{model} parameters')
        layout = QVBoxLayout(dialog)

        table = QTableWidget(len(params) + 1, 2)
        table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        input_item = QTableWidgetItem('Input video')
        input_item.setFlags(input_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(0, 0, input_item)
        input_combo = NoWheelComboBox()
        for label, value in sources:
            input_combo.addItem(label, value)
        previous_input = str(row.image_layer_params.get('input_source') or params.get('input_source') or '')
        if previous_input:
            input_combo.setCurrentIndex(self._combo_index_for_data(input_combo, previous_input))
        table.setCellWidget(0, 1, input_combo)

        param_keys = list(params.keys())
        param_widgets: dict[str, QWidget] = {}
        for row_idx, key in enumerate(param_keys, start=1):
            key_item = QTableWidgetItem(str(key))
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row_idx, 0, key_item)
            if model == 'PMD' and key == 'pixel_weight_source':
                weight_source_combo = NoWheelComboBox()
                for value, label in PMD_PIXEL_WEIGHT_SOURCE_OPTIONS:
                    weight_source_combo.addItem(label, value)
                weight_source_combo.setCurrentIndex(
                    self._combo_index_for_data(weight_source_combo, str(params.get(key) or 'none'))
                )
                table.setCellWidget(row_idx, 1, weight_source_combo)
                param_widgets[key] = weight_source_combo
            else:
                table.setItem(row_idx, 1, QTableWidgetItem(self._format_image_layer_param(params[key])))
        layout.addWidget(table)

        buttons = QHBoxLayout()
        apply_btn = QPushButton('Apply')
        close_btn = QPushButton('Close')
        buttons.addStretch(1)
        buttons.addWidget(apply_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        def apply_params():
            updated = dict(params)
            for row_idx, key in enumerate(param_keys, start=1):
                widget = param_widgets.get(key)
                if isinstance(widget, QComboBox):
                    updated[key] = widget.currentData()
                    continue
                item = table.item(row_idx, 1)
                text = item.text() if item is not None else ''
                updated[key] = self._parse_image_layer_param(text, params.get(key))
            updated['input_source'] = input_combo.currentData()
            updated['_model'] = model
            row.image_layer_params = updated
            if model == 'PMD':
                if self._apply_pmd_from_dialog(row, updated):
                    dialog.accept()
                return
            if self._apply_normcorre_from_dialog(row, updated):
                dialog.accept()

        apply_btn.clicked.connect(apply_params)
        close_btn.clicked.connect(dialog.reject)
        dialog.resize(560, 520)
        dialog.exec()

    def _image_layer_params_for_row(self, row: ImageLayerControlRow, model: str) -> dict[str, Any]:
        if row.image_layer_params.get('_model') == model:
            stored = {key: value for key, value in row.image_layer_params.items() if key not in {'_model', 'input_source'}}
            if model == 'NoRMCorre':
                params = dict(normcorre_backend.DEFAULT_PARAMS)
                params['fr'] = float(self.state.frame_rate) if self.state.frame_rate else params.get('fr', 1.0)
                params['output_name'] = self._image_layer_default_output_name(model)
                params.update(stored)
                return params
            if model == 'PMD':
                params = dict(pmd_backend.DEFAULT_PARAMS)
                params['output_name'] = self._image_layer_default_output_name(model)
                params.update(stored)
                return params
            return stored
        if model == 'NoRMCorre':
            params = dict(normcorre_backend.DEFAULT_PARAMS)
            params['fr'] = float(self.state.frame_rate) if self.state.frame_rate else params.get('fr', 1.0)
            params['output_name'] = self._image_layer_default_output_name(model)
        elif model == 'PMD':
            params = dict(pmd_backend.DEFAULT_PARAMS)
            params['output_name'] = self._image_layer_default_output_name(model)
        else:
            params = {}
        return params

    def _image_layer_default_output_name(self, model: str) -> str:
        model_name = 'normcorre' if model == 'NoRMCorre' else model.lower()
        return f'{model_name}_{self._database_name()}.tiff'

    def _image_layer_input_sources(self, current_row: ImageLayerControlRow) -> list[tuple[str, str]]:
        sources: list[tuple[str, str]] = []
        if self.state.tif_path and Path(self.state.tif_path).exists():
            sources.append(('Raw TIFF', 'raw'))
        sources.append(('Local TIFF/mmap...', 'local'))
        if self._find_volpy_tif_path() is not None:
            sources.append(('VolPy TIFF', 'volpy'))
        for row in self.image_layers:
            if row is current_row or row.model_combo is None:
                continue
            movie = self._get_image_layer_movie(row)
            if movie is not None:
                sources.append((self._image_layer_label(row), f'layer:{row.layer_id}'))
        if not sources:
            sources.append(('No input video available', ''))
        return sources

    def _format_image_layer_param(self, value: Any) -> str:
        if value is None:
            return 'None'
        return str(value)

    def _parse_image_layer_param(self, text: str, current: Any) -> Any:
        value = text.strip()
        if value.lower() in {'none', 'null'}:
            return None
        if isinstance(current, bool):
            return value.lower() in {'true', '1', 'yes', 'y'}
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value
        if isinstance(current, tuple) and isinstance(parsed, list):
            return tuple(parsed)
        if isinstance(current, int) and not isinstance(current, bool):
            return int(parsed)
        if isinstance(current, float):
            return float(parsed)
        return parsed

    def _select_local_image_layer_path(self, row: ImageLayerControlRow):
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Select local image layer',
            '',
            'Image or movie (*.tif *.tiff *.mmap);;All files (*.*)',
        )
        if not path:
            return
        row.image_layer_params = {'_model': 'Local', 'local_path': path}

    def _apply_normcorre_from_dialog(self, row: ImageLayerControlRow, params: dict[str, Any], prompt_local: bool = True) -> bool:
        row.image_layer_params = dict(params)
        input_path = self._resolve_image_layer_input_path(str(params.get('input_source') or ''), row, params, prompt_local=prompt_local)
        if input_path is None:
            self._set_status('No valid input video is available for NoRMCorre.')
            return False
        output_path = self._image_layer_output_path(row, params, default_name=self._image_layer_default_output_name('NoRMCorre'))
        reload = bool(params.get('Reload'))
        self._image_projection_cache.clear()
        self._computed_trace_cache.clear()
        self._invalidate_trace_cache()

        def on_success(_path):
            self._image_projection_cache.clear()
            self._computed_trace_cache.clear()
            self._invalidate_trace_cache()
            self._refresh_trace_source_options()
            self.render_all(reset_trace_view=True)

        return self._start_background_job(
            lambda: normcorre_backend.run_normcorre(
                input_path=input_path,
                output_path=output_path,
                frame_rate=float(params.get('fr') or self.state.frame_rate or 1.0),
                params=params,
                reload=reload,
            ),
            'Running NoRMCorre...',
            'NoRMCorre saved: {result}',
            on_success=on_success,
        )

    def _pmd_input_image_shape(self, input_path: Path, params: dict[str, Any]) -> Optional[tuple[int, int]]:
        with tifffile.TiffFile(input_path) as tif:
            shape = tuple(int(v) for v in tif.series[0].shape if int(v) != 1)
        if len(shape) < 2:
            return None
        axis_order = str(params.get('input_axis_order') or 'TYX').upper()
        if axis_order == 'YXT' and len(shape) >= 3:
            return int(shape[-3]), int(shape[-2])
        return int(shape[-2]), int(shape[-1])

    def _pmd_pixel_weighting_from_params(
        self,
        params: dict[str, Any],
        image_shape: tuple[int, int],
        input_path: Path,
    ) -> tuple[bool, Optional[np.ndarray]]:
        source = str(params.get('pixel_weight_source') or 'none').strip().lower()
        if source in {'', 'none', 'null'}:
            return True, None
        valid_sources = {value for value, _label in PMD_PIXEL_WEIGHT_SOURCE_OPTIONS}
        if source not in valid_sources:
            self._set_status(f'Unsupported PMD pixel_weight_source: {source}')
            return False, None
        boost = float(params.get('pixel_weight_boost') or 1.0)
        if boost <= 0:
            self._set_status('PMD pixel_weight_boost must be greater than 0.')
            return False, None
        if source in {'correlation', 'z_average'}:
            weight_map = self._pmd_image_weight_map(input_path, params, source, image_shape)
            if weight_map is None:
                return False, None
            weights = np.ones(tuple(image_shape), dtype=np.float32)
            weights += (boost - 1.0) * weight_map.astype(np.float32, copy=False)
            return True, weights
        masks, _mapping = self._mask_stack_for_source(source, image_shape)
        if masks is None or masks.size == 0:
            self._set_status(f'PMD pixel weighting requires available {source} masks.')
            return False, None
        masks = np.asarray(masks, dtype=np.float32)
        if masks.ndim != 3 or tuple(masks.shape[1:3]) != tuple(image_shape):
            self._set_status(
                f'PMD {source} masks have shape {masks.shape}; expected ROI stack with image shape {image_shape}.'
            )
            return False, None
        roi_map = np.nan_to_num(np.max(np.abs(masks), axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        roi_max = float(np.max(roi_map))
        if roi_max <= 0:
            self._set_status(f'PMD {source} masks do not contain positive ROI weights.')
            return False, None
        roi_map = roi_map / roi_max
        weights = np.ones(tuple(image_shape), dtype=np.float32)
        weights += (boost - 1.0) * roi_map.astype(np.float32, copy=False)
        return True, weights

    def _pmd_image_weight_map(
        self,
        input_path: Path,
        params: dict[str, Any],
        source: str,
        image_shape: tuple[int, int],
    ) -> Optional[np.ndarray]:
        try:
            movie, _original_dtype = pmd_backend.load_tiff_as_tyx(
                input_path,
                str(params.get('input_axis_order') or 'TYX'),
            )
        except Exception as exc:
            self._set_status(f'Could not load PMD weight input: {exc}')
            return None
        if source == 'correlation':
            image = self._correlation_image(movie)
        else:
            image = np.asarray(np.nanmean(movie, axis=0), dtype=float)
        image = np.asarray(image, dtype=np.float32)
        if tuple(image.shape[:2]) != tuple(image_shape):
            self._set_status(
                f'PMD {source} weight image has shape {image.shape}; expected {image_shape}.'
            )
            return None
        return self._normalize_weight_image(image, source)

    def _normalize_weight_image(self, image: np.ndarray, label: str) -> Optional[np.ndarray]:
        arr = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            self._set_status(f'PMD {label} weight image has no finite values.')
            return None
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmax <= vmin:
            self._set_status(f'PMD {label} weight image has no intensity range.')
            return None
        return ((arr - vmin) / (vmax - vmin)).astype(np.float32, copy=False)

    def _apply_pmd_from_dialog(self, row: ImageLayerControlRow, params: dict[str, Any], prompt_local: bool = True) -> bool:
        row.image_layer_params = dict(params)
        input_path = self._resolve_image_layer_input_path(str(params.get('input_source') or ''), row, params, prompt_local=prompt_local)
        if input_path is None:
            self._set_status('No valid input video is available for PMD.')
            return False
        input_path = self._ensure_pmd_tiff_input(input_path, params, row)
        if input_path is None:
            self._set_status('No valid TIFF input video is available for PMD.')
            return False
        image_shape = self._pmd_input_image_shape(input_path, params)
        if image_shape is None:
            self._set_status('Could not determine the PMD input image shape.')
            return False
        pixel_weight_ok, pixel_weighting = self._pmd_pixel_weighting_from_params(params, image_shape, input_path)
        if not pixel_weight_ok:
            return False
        run_params = dict(params)
        if pixel_weighting is None:
            run_params.pop('pixel_weighting', None)
        else:
            run_params['pixel_weighting'] = pixel_weighting
        output_path = self._image_layer_output_path(row, params, default_name=self._image_layer_default_output_name('PMD'))
        reload = bool(params.get('Reload'))
        self._image_projection_cache.clear()
        self._computed_trace_cache.clear()
        self._invalidate_trace_cache()

        def on_success(_path):
            self._image_projection_cache.clear()
            self._computed_trace_cache.clear()
            self._invalidate_trace_cache()
            self._refresh_trace_source_options()
            self.render_all(reset_trace_view=True)

        return self._start_background_job(
            lambda progress: pmd_backend.run_pmd(
                input_path=input_path,
                output_path=output_path,
                frame_rate=float(self.state.frame_rate or 1.0),
                params=run_params,
                reload=reload,
                progress_callback=progress,
            ),
            'Running PMD...',
            'PMD saved: {result}',
            on_success=on_success,
            with_progress=True,
        )

    def _resolve_image_layer_input_path(
        self,
        source: str,
        current_row: ImageLayerControlRow,
        params: dict[str, Any],
        prompt_local: bool = True,
    ) -> Optional[Path]:
        if source == 'raw':
            if self.state.tif_path and Path(self.state.tif_path).exists():
                return Path(self.state.tif_path)
            return None
        if source == 'local':
            path_text = str(params.get('local_input_path') or '').strip()
            if not path_text:
                if not prompt_local:
                    return None
                path_text, _ = QFileDialog.getOpenFileName(
                    self,
                    'Select local input video',
                    '',
                    'Image or movie (*.tif *.tiff *.mmap);;All files (*.*)',
                )
                if not path_text:
                    return None
                params['local_input_path'] = path_text
                current_row.image_layer_params['local_input_path'] = path_text
            path = Path(path_text)
            return path if path.exists() else None
        if source == 'volpy':
            return self._find_volpy_tif_path()
        if source.startswith('layer:'):
            layer_id = source.split(':', 1)[1]
            source_row = next((row for row in self.image_layers if row.layer_id == layer_id), None)
            if source_row is None:
                return None
            movie = self._get_image_layer_movie(source_row)
            if movie is None:
                return None
            temp_dir = self._image_layer_output_dir() / '.neurobox_temp'
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f'{current_row.layer_id}_from_{layer_id}.tiff'
            self._write_movie_tiff(temp_path, movie)
            return temp_path
        return None

    def _ensure_pmd_tiff_input(self, input_path: Path, params: dict[str, Any], row: ImageLayerControlRow) -> Optional[Path]:
        if input_path.suffix.lower() in {'.tif', '.tiff'}:
            return input_path
        if input_path.suffix.lower() != '.mmap':
            return None
        movie = self._load_local_mmap_movie(input_path)
        if movie is None:
            return None
        temp_dir = self._image_layer_output_dir() / '.neurobox_temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f'{row.layer_id}_pmd_input.tiff'
        self._write_movie_tiff(temp_path, movie)
        params['local_input_tiff'] = str(temp_path)
        return temp_path

    def _image_layer_output_dir(self) -> Path:
        if self.state.source_path:
            path = Path(self.state.source_path)
            return path if path.is_dir() else path.parent
        return Path.cwd()

    def _image_layer_output_path(self, row: ImageLayerControlRow, params: dict[str, Any], default_name: str) -> Path:
        output_name = str(params.get('output_name') or default_name).strip()
        output_path = Path(output_name)
        if not output_path.is_absolute():
            output_path = self._image_layer_output_dir() / output_path
        return output_path

    def _write_movie_tiff(self, path: Path, movie: Any):
        arr = np.asarray(movie)
        if arr.ndim < 2:
            raise ValueError('Input video must have at least two dimensions.')
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, arr.astype(np.float32, copy=False), bigtiff=True)

    def _on_trace_row_combo_changed(self, row: TraceControlRow):
        self._sync_trace_row_volpy_source_state(row, show_warning=True)
        self._invalidate_trace_cache()
        self.render_all()

    def _sync_trace_row_volpy_source_state(self, row: TraceControlRow, show_warning: bool):
        data = row.source_combo.currentData()
        use_volpy_trace = self._is_volpy_image_trace_source(data)
        if use_volpy_trace:
            was_checked = row.volpy_checkbox.isChecked()
            current_key = row.volpy_combo.currentText().strip()
            vpy_data = self.ensure_volpy_loaded(show_warning=show_warning)
            target_key = current_key if was_checked and current_key else 'ts'
            if isinstance(vpy_data, dict):
                available = []
                for key in TRACE_SOURCE_VOLPY_KEYS:
                    if key in vpy_data and self._coerce_trace_matrix(vpy_data[key]) is not None:
                        available.append(key)
                if target_key not in available:
                    target_key = 'ts' if 'ts' in available else (available[0] if available else target_key)
            row.volpy_checkbox.blockSignals(True)
            row.volpy_combo.blockSignals(True)
            row.volpy_checkbox.setChecked(True)
            row.volpy_combo.setCurrentIndex(self._combo_index_for_text(row.volpy_combo, target_key))
            row.volpy_combo.blockSignals(False)
            row.volpy_checkbox.blockSignals(False)
        elif row.volpy_checkbox.isChecked():
            row.volpy_checkbox.blockSignals(True)
            row.volpy_checkbox.setChecked(False)
            row.volpy_checkbox.blockSignals(False)

    def _is_volpy_image_trace_source(self, data: Any) -> bool:
        if not isinstance(data, str) or not data.startswith('image:'):
            return False
        parts = data.split(':', 2)
        if len(parts) != 3 or parts[2] != 'volpy':
            return False
        layer = next((item for item in self.image_layers if item.layer_id == parts[1]), None)
        return bool(layer is not None and self._image_layer_model(layer) == 'VolPy')

    def _build_plot_area(self):
        self.image_widget = InteractiveFigureWidget('Image panel', mode='image')
        self.image_widget.set_side_panel(self.roi_box, width=260)
        self.trace_widget = InteractiveFigureWidget('Trace panel', mode='xwindow')
        self.avg_widget = InteractiveFigureWidget('Average panel', mode='xwindow')
        self.trace_widget.on_xwindow_changed = self._sync_trace_frame_slider

        roi_row = QHBoxLayout()
        roi_row.addWidget(QLabel('ROI index'))
        self.roi_index_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_index_slider.setMinimum(0)
        self.roi_index_slider.setMaximum(0)
        self.roi_index_slider.setValue(0)
        self.roi_index_slider.valueChanged.connect(self._on_roi_index_slider_changed)
        self.roi_index_label = QLabel('0')
        roi_row.addWidget(self.roi_index_slider, stretch=1)
        roi_row.addWidget(self.roi_index_label)
        self.image_widget.top_controls_layout.addLayout(roi_row)

        image_play_row = QHBoxLayout()
        self.image_play_widget = QWidget()
        image_play_inner = QHBoxLayout(self.image_play_widget)
        image_play_inner.setContentsMargins(0, 0, 0, 0)
        self.image_play_button = QPushButton('Play')
        self.image_play_button.clicked.connect(self._toggle_image_play)
        self.image_frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.image_frame_slider.setMinimum(0)
        self.image_frame_slider.setMaximum(0)
        self.image_frame_slider.valueChanged.connect(lambda _value: self._render_image_panel())
        self.image_frame_label = QLabel('0 / 0')
        self.image_labels_checkbox = QCheckBox('labels')
        self.image_labels_checkbox.setChecked(True)
        self.image_labels_checkbox.stateChanged.connect(lambda _state: self._render_image_panel())
        image_play_inner.addWidget(self.image_play_button)
        image_play_inner.addWidget(self.image_frame_slider, stretch=1)
        image_play_inner.addWidget(self.image_frame_label)
        image_play_row.addWidget(self.image_play_widget, stretch=1)
        image_play_row.addWidget(self.image_labels_checkbox)
        self.image_widget.extra_controls_layout.addLayout(image_play_row)
        self.image_play_widget.setVisible(False)

        trace_play_row = QHBoxLayout()
        self.trace_play_button = QPushButton('Play')
        self.trace_play_button.clicked.connect(self._toggle_trace_play)
        self.trace_frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.trace_frame_slider.setMinimum(0)
        self.trace_frame_slider.setMaximum(0)
        self.trace_frame_slider.valueChanged.connect(self._on_trace_frame_slider_changed)
        self.trace_frame_label = QLabel('0 / 0')
        self.trace_x_unit_combo = NoWheelComboBox()
        self.trace_x_unit_combo.addItem('seconds', 'seconds')
        self.trace_x_unit_combo.addItem('frames', 'frames')
        self.trace_x_unit_combo.currentIndexChanged.connect(self._on_trace_x_unit_changed)
        self.trace_labels_checkbox = QCheckBox('labels')
        self.trace_labels_checkbox.setChecked(True)
        self.trace_labels_checkbox.stateChanged.connect(lambda _state: self._render_trace_panel())
        trace_play_row.addWidget(self.trace_play_button)
        trace_play_row.addWidget(self.trace_frame_slider, stretch=1)
        trace_play_row.addWidget(self.trace_frame_label)
        trace_play_row.addWidget(self.trace_x_unit_combo)
        trace_play_row.addWidget(self.trace_labels_checkbox)
        self.trace_widget.extra_controls_layout.addLayout(trace_play_row)

        avg_row = QHBoxLayout()
        self.avg_labels_checkbox = QCheckBox('labels')
        self.avg_labels_checkbox.setChecked(True)
        self.avg_labels_checkbox.stateChanged.connect(lambda _state: self._render_average_panel())
        self.avg_individual_button = QPushButton('Hide individual traces')
        self.avg_individual_button.setCheckable(True)
        self.avg_individual_button.setChecked(True)
        self.avg_individual_button.toggled.connect(self._on_avg_individual_toggled)
        avg_row.addStretch(1)
        avg_row.addWidget(self.avg_individual_button)
        avg_row.addWidget(self.avg_labels_checkbox)
        self.avg_widget.extra_controls_layout.addLayout(avg_row)
        self.avg_export_layout = QHBoxLayout()
        self.avg_export_layout.addStretch(1)
        self.avg_widget.extra_controls_layout.addLayout(self.avg_export_layout)

        self.image_widget.canvas.mpl_connect('button_press_event', self._on_image_click_select_roi)

        self.plot_layout.addWidget(self.image_widget)
        self.plot_layout.addWidget(self.trace_widget)
        self.plot_layout.addWidget(self.avg_widget)

        self.image_widget.set_placeholder('Load data.')
        self.trace_widget.set_placeholder('Trace panel')
        self.avg_widget.set_placeholder('Average panel')

    # ------------------------------ callbacks ------------------------------
    def _on_combine_mode_changed(self, value: str, checked: bool):
        if checked:
            self.combine_mode = value
            self.render_all()

    def _on_avg_mode_changed(self, value: str, checked: bool):
        if checked:
            self.avg_modes.add(value)
            self.avg_mode = value
        else:
            self.avg_modes.discard(value)
            if not self.avg_modes:
                self.avg_modes.add(value)
                checkbox = self._avg_radio.get(value)
                if checkbox is not None:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(True)
                    checkbox.blockSignals(False)
            self.avg_mode = self._active_avg_modes()[0]
        self.render_all()

    def _active_avg_modes(self) -> list[str]:
        modes = [mode for mode in AVG_PANEL_MODES if mode in getattr(self, 'avg_modes', {self.avg_mode})]
        return modes if modes else ['event']

    def _on_roi_index_slider_changed(self, value: int):
        if hasattr(self, 'roi_index_label'):
            self.roi_index_label.setText(str(value))
        if self.state.cells is not None and len(self.state.cells) > 0:
            self._select_roi(int(np.clip(value, 0, len(self.state.cells) - 1)), update_slider=False)
            return
        self.render_all()

    def _on_trace_x_unit_changed(self, _idx: int):
        if hasattr(self, 'trace_window_label'):
            self.trace_window_label.setText('Window(frames)' if self._trace_x_unit() == 'frames' else 'Window(s)')
        self._render_trace_panel(reset_view=True)

    def _trace_x_unit(self) -> str:
        if hasattr(self, 'trace_x_unit_combo'):
            value = self.trace_x_unit_combo.currentData()
            if value in {'seconds', 'frames'}:
                return value
        return 'seconds'

    def _trace_x_label(self) -> str:
        return 'Frame' if self._trace_x_unit() == 'frames' else 'Time (s)'

    def _trace_x_values(self, n_frames: int) -> np.ndarray:
        frames = np.arange(n_frames, dtype=float)
        if self._trace_x_unit() == 'frames':
            return frames
        return frames / max(self.state.frame_rate, 1e-8)

    def _trace_plot_stride(self, n_frames: int, max_points: int = 12000) -> int:
        return max(1, int(np.ceil(max(n_frames, 1) / max(max_points, 1))))

    def _trace_x_total(self, n_frames: int) -> float:
        if self._trace_x_unit() == 'frames':
            return float(max(n_frames, 1))
        return float(max(n_frames, 1)) / max(self.state.frame_rate, 1e-8)

    def _trace_window_width(self, value: float) -> float:
        if self._trace_x_unit() == 'frames':
            return max(value, 1.0)
        return max(value, 0.1)

    def _trace_frame_to_x(self, frame_idx: int) -> float:
        if self._trace_x_unit() == 'frames':
            return float(frame_idx)
        return float(frame_idx) / max(self.state.frame_rate, 1.0)

    def _trace_x_to_frame(self, x_value: float) -> int:
        if self._trace_x_unit() == 'frames':
            return int(round(x_value))
        return int(round(x_value * max(self.state.frame_rate, 1.0)))

    def _trace_time_to_x(self, seconds):
        seconds = np.asarray(seconds, dtype=float)
        if self._trace_x_unit() == 'frames':
            seconds = seconds * max(self.state.frame_rate, 1.0)
        if seconds.ndim == 0:
            return float(seconds)
        return seconds

    def _on_avg_individual_toggled(self, checked: bool):
        self.avg_individual_button.setText('Hide individual traces' if checked else 'Show individual traces')
        self._render_average_panel()

    def _avg_show_individual_traces(self) -> bool:
        return getattr(self, 'avg_individual_button', None) is None or self.avg_individual_button.isChecked()

    def _init_events_from_state(self):
        self.events = []
        if self.state.onset_times_trial is None or self.state.offset_times_trial is None:
            return
        fr = max(float(self.state.frame_rate), 1.0)
        starts = np.rint(np.asarray(self.state.onset_times_trial, dtype=float) * fr).astype(int)
        ends = np.rint(np.asarray(self.state.offset_times_trial, dtype=float) * fr).astype(int)
        self.events.append(
            {
                'source': 'opt_sti',
                'label': 'stimulate',
                'color': '#ff0000',
                'start_frames': starts,
                'end_frames': ends,
                'mode': 'label',
            }
        )

    def _refresh_event_table(self):
        if not hasattr(self, 'event_table'):
            return
        self.event_table.blockSignals(True)
        self.event_table.setRowCount(len(self.events))
        for row_idx, event in enumerate(self.events):
            for col_idx, key in enumerate(('source', 'label', 'color')):
                item = QTableWidgetItem('' if key == 'color' else str(event.get(key, '')))
                flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                if key == 'label':
                    flags |= Qt.ItemFlag.ItemIsEditable
                item.setFlags(flags)
                if key == 'color':
                    color = self._event_color_hex(event)
                    item.setBackground(QColor(color))
                    item.setToolTip(f'Double-click to edit color: {color}')
                    item.setData(Qt.ItemDataRole.UserRole, color)
                self.event_table.setItem(row_idx, col_idx, item)
        self.event_table.blockSignals(False)
        self._resize_event_table()
        self._on_event_table_selection_changed()
        self._refresh_avg_event_list()

    def _resize_event_table(self):
        rows = max(1, self.event_table.rowCount())
        header = self.event_table.horizontalHeader().height()
        row_height = self.event_table.verticalHeader().defaultSectionSize()
        height = int(header + rows * row_height + 6)
        self.event_table.setFixedHeight(height)

    def _event_color_hex(self, event: dict[str, Any]) -> str:
        color = str(event.get('color') or '#ff0000')
        return color if QColor(color).isValid() else '#ff0000'

    def _refresh_avg_event_list(self):
        if not hasattr(self, 'avg_event_list'):
            return
        previous = {item.data(Qt.ItemDataRole.UserRole) for item in self.avg_event_list.selectedItems()}
        self.avg_event_list.blockSignals(True)
        self.avg_event_list.clear()
        for idx, event in enumerate(self.events):
            label = str(event.get('label') or event.get('source') or f'event {idx + 1}')
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.avg_event_list.addItem(item)
            if not previous or idx in previous:
                item.setSelected(True)
        self.avg_event_list.blockSignals(False)

    def _on_event_table_item_changed(self, item: QTableWidgetItem):
        row = item.row()
        if row < 0 or row >= len(self.events):
            return
        key = ('source', 'label', 'color')[item.column()]
        if key in {'source', 'color'}:
            return
        self.events[row][key] = item.text().strip()
        self._invalidate_trace_cache()
        self._refresh_avg_event_list()
        self.render_all()

    def _on_event_table_item_double_clicked(self, item: QTableWidgetItem):
        if item.column() != 2:
            return
        row = item.row()
        if row < 0 or row >= len(self.events):
            return
        current = QColor(self._event_color_hex(self.events[row]))
        color = QColorDialog.getColor(current, self, 'Select event color')
        if not color.isValid():
            return
        self.events[row]['color'] = color.name()
        self._invalidate_trace_cache()
        self._refresh_event_table()
        self.render_all()

    def _selected_event_row(self) -> Optional[int]:
        if not hasattr(self, 'event_table'):
            return None
        selected = self.event_table.selectionModel().selectedRows() if self.event_table.selectionModel() is not None else []
        if not selected:
            return None
        row = int(selected[0].row())
        return row if 0 <= row < len(self.events) else None

    def _on_event_table_selection_changed(self):
        row = self._selected_event_row()
        enabled = row is not None and self.events[row].get('source') == 'trace'
        self._set_trace_event_controls_enabled(enabled)
        if enabled:
            self._load_trace_event_params_to_panel(self.events[row])

    def _trace_event_control_widgets(self):
        return (
            self.event_trace_source_combo,
            self.event_trace_action_combo,
            self.event_trace_direction_combo,
            self.event_trace_value_edit,
            self.event_trace_interval_edit,
            self.event_trace_adjacent_edit,
            self.event_trace_merge_edit,
            self.event_trace_diff_edit,
            self.event_trace_from_edit,
            self.event_trace_to_edit,
            self.event_trace_apply_button,
        )

    def _set_trace_event_controls_enabled(self, enabled: bool):
        if not hasattr(self, 'event_trace_source_combo'):
            return
        for widget in self._trace_event_control_widgets():
            widget.setEnabled(enabled)
        if enabled:
            self._update_trace_event_mode_controls()

    def _update_trace_event_mode_controls(self):
        if not hasattr(self, 'event_trace_direction_combo'):
            return
        controls_enabled = self.event_trace_apply_button.isEnabled()
        between = self.event_trace_direction_combo.currentText() == 'between'
        threshold_widgets = (
            self.event_trace_value_edit,
            self.event_trace_interval_edit,
            self.event_trace_adjacent_edit,
            self.event_trace_merge_edit,
            self.event_trace_diff_edit,
        )
        for widget in threshold_widgets:
            widget.setEnabled(controls_enabled and not between)
        self.event_trace_from_edit.setEnabled(controls_enabled and between)
        self.event_trace_to_edit.setEnabled(controls_enabled and between)

    def _load_trace_event_params_to_panel(self, event: dict[str, Any]):
        params = self._trace_event_params_from_event(event)
        self._updating_event_trace_panel = True
        try:
            self.event_trace_source_combo.setCurrentIndex(self._combo_index_for_data(self.event_trace_source_combo, params.get('source_data')))
            self.event_trace_action_combo.setCurrentIndex(self._combo_index_for_text(self.event_trace_action_combo, params.get('mode', 'discard')))
            self.event_trace_direction_combo.setCurrentIndex(self._combo_index_for_text(self.event_trace_direction_combo, params.get('direction', 'above')))
            self.event_trace_value_edit.setText(str(params.get('value', 0.0)))
            self.event_trace_interval_edit.setText(str(params.get('interval', 1.0)))
            self.event_trace_adjacent_edit.setText(str(params.get('adjacent', 0.0)))
            self.event_trace_merge_edit.setText(str(params.get('merge', 0.0)))
            self.event_trace_diff_edit.setText(str(params.get('diff', 0.0)))
            self.event_trace_from_edit.setText(str(params.get('from', 0)))
            self.event_trace_to_edit.setText(str(params.get('to', 0)))
        finally:
            self._updating_event_trace_panel = False
        self._update_trace_event_mode_controls()

    def _trace_event_params_from_event(self, event: dict[str, Any]) -> dict[str, Any]:
        params = dict(event.get('trace_params') or {})
        params.setdefault('source_data', None)
        params.setdefault('source_name', event.get('source_name', ''))
        params.setdefault('mode', event.get('mode', 'discard'))
        params.setdefault('direction', 'above')
        params.setdefault('value', 0.0)
        params.setdefault('interval', 1.0)
        params.setdefault('adjacent', 0.0)
        params.setdefault('merge', 0.0)
        params.setdefault('diff', event.get('diff', 0.0))
        params.setdefault('from', 0)
        params.setdefault('to', 0)
        return params

    def _trace_event_params_from_panel(self) -> dict[str, Any]:
        data = self.event_trace_source_combo.currentData()
        return {
            'source_data': data,
            'source_name': self.event_trace_source_combo.currentText(),
            'mode': self.event_trace_action_combo.currentText(),
            'direction': self.event_trace_direction_combo.currentText(),
            'value': self._safe_float(self.event_trace_value_edit.text(), default=0.0),
            'interval': max(0.0, self._safe_float(self.event_trace_interval_edit.text(), default=1.0)),
            'adjacent': max(0.0, self._safe_float(self.event_trace_adjacent_edit.text(), default=0.0)),
            'merge': max(0.0, self._safe_float(self.event_trace_merge_edit.text(), default=0.0)),
            'diff': self._safe_float(self.event_trace_diff_edit.text(), default=0.0),
            'from': max(0, int(round(self._safe_float(self.event_trace_from_edit.text(), default=0.0)))),
            'to': max(0, int(round(self._safe_float(self.event_trace_to_edit.text(), default=0.0)))),
        }

    def _apply_trace_event_detection_to_event(self, event: dict[str, Any]) -> bool:
        params = self._trace_event_params_from_event(event)
        if params.get('source_data') is None and hasattr(self, 'event_trace_source_combo'):
            params['source_data'] = self.event_trace_source_combo.currentData()
            params['source_name'] = self.event_trace_source_combo.currentText()
        event['trace_params'] = params
        event['mode'] = params['mode']
        event['diff'] = params['diff']
        source = self._get_trace_event_source_matrix(params.get('source_data'))
        if source is None:
            event['start_frames'] = np.asarray([], dtype=int)
            event['end_frames'] = np.asarray([], dtype=int)
            return False
        trace_mat, source_name, mapping = source
        params['source_name'] = source_name
        selected_trace, roi_idx, row_idx = self._single_trace_for_event_detection(
            np.asarray(trace_mat, dtype=float),
            mapping,
        )
        intervals = self._detect_trace_event_intervals(selected_trace, params)
        params['roi_index'] = roi_idx
        params['source_row'] = row_idx
        event['trace_params'] = params
        event['source_name'] = source_name
        event['start_frames'] = np.asarray([start for start, _stop in intervals], dtype=int)
        event['end_frames'] = np.asarray([stop for _start, stop in intervals], dtype=int)
        return True

    def _get_trace_event_source_matrix(self, data: Any) -> Optional[tuple[np.ndarray, str, Optional[np.ndarray]]]:
        if data == 'event:raw':
            if not self.trace_rows:
                return None
            source = self._get_trace_source_matrix(self.trace_rows[0])
            if source is None:
                return None
            trace_mat, source_name, mapping = source
            return np.asarray(trace_mat, dtype=float), f'raw | {source_name}', mapping
        if isinstance(data, str) and data.startswith('event:trace:'):
            try:
                row_idx = int(data.rsplit(':', 1)[1])
            except ValueError:
                return None
            if row_idx < 0 or row_idx >= len(self.trace_rows):
                return None
            result = self._build_trace_result(self.trace_rows[row_idx])
            if result is None:
                return None
            return np.asarray(result['data'], dtype=float), f'Trace {row_idx + 1}', result.get('roi_indices')
        return self._get_source_matrix_by_data(data)

    def add_event_callback(self):
        source = self.event_add_source_combo.currentText()
        if source == 'npy file':
            self._add_npy_event()
        elif source == 'trace':
            self._add_trace_event()
        else:
            path, _ = QFileDialog.getOpenFileName(self, 'Select event source file', '', 'All files (*.*)')
            if path:
                self.developing(f'Event source: {path}')

    def remove_selected_event_callback(self):
        rows = sorted({idx.row() for idx in self.event_table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self.events):
                del self.events[row]
        self._invalidate_trace_cache()
        self._refresh_event_table()
        self.render_all()

    def _add_npy_event(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select event seconds file', '', 'NumPy (*.npy);;All files (*.*)')
        if not path:
            return
        loaded = np.load(path, allow_pickle=True)
        if hasattr(loaded, 'shape') and loaded.shape == ():
            loaded = loaded.item()
        if isinstance(loaded, dict):
            events = self._events_from_saved_payload(loaded)
            if not events:
                QMessageBox.critical(self, 'Invalid event file', 'Saved event file must contain start_time, end_time, and label lists.')
                return
            self.events.extend(events)
            self._invalidate_trace_cache()
            self._refresh_event_table()
            self.render_all()
            return
        arr = np.asarray(loaded, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != 2:
            QMessageBox.critical(self, 'Invalid event file', 'Event .npy file must contain a 2 x n array of start/end times in seconds or a saved event dictionary.')
            return

        fr = max(float(self.state.frame_rate), 1.0)
        starts = np.rint(arr[0] * fr).astype(int)
        ends = np.rint(arr[1] * fr).astype(int)
        self.events.append(
            {
                'source': 'npy file',
                'label': Path(path).stem,
                'color': '#ff6666',
                'start_frames': starts,
                'end_frames': ends,
                'mode': 'label',
            }
        )
        self._refresh_event_table()
        self.render_all()

    def _events_from_saved_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if 'params' in payload:
            return self._events_from_params_payload(payload)

        start_list = list(payload.get('start_time', []))
        end_list = list(payload.get('end_time', []))
        labels = list(payload.get('label', []))
        sources = list(payload.get('source', ['npy file'] * len(labels)))
        colors = list(payload.get('color', ['#ff6666'] * len(labels)))
        trace_extract = list(payload.get('trace_extract', [{} for _ in labels]))
        count = min(len(start_list), len(end_list), len(labels))
        fr = max(float(self.state.frame_rate), 1.0)
        events = []
        for idx in range(count):
            starts_sec = np.asarray(start_list[idx], dtype=float).reshape(-1)
            ends_sec = np.asarray(end_list[idx], dtype=float).reshape(-1)
            valid = np.isfinite(starts_sec) & np.isfinite(ends_sec)
            source = str(sources[idx]) if idx < len(sources) else 'npy file'
            trace_params = dict(trace_extract[idx]) if idx < len(trace_extract) and isinstance(trace_extract[idx], dict) else {}
            mode = str(trace_params.get('mode', 'label')) if source == 'trace' else 'label'
            events.append(
                {
                    'source': source,
                    'label': str(labels[idx]),
                    'color': str(colors[idx]) if idx < len(colors) else '#ff6666',
                    'start_frames': np.rint(starts_sec[valid] * fr).astype(int),
                    'end_frames': np.rint(ends_sec[valid] * fr).astype(int),
                    'mode': mode,
                    'diff': float(trace_params.get('diff', 0.0)) if trace_params else 0.0,
                    'trace_params': trace_params,
                    'source_name': str(trace_params.get('source_name', '')) if trace_params else '',
                }
            )
        return events

    def _events_from_params_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        params_payload = payload.get('params')
        if isinstance(params_payload, dict):
            params_list = [params_payload]
        else:
            params_list = list(params_payload or [])
        labels = list(payload.get('label', []))
        sources = list(payload.get('source', ['trace'] * len(params_list)))
        colors = list(payload.get('color', ['#ff9900'] * len(params_list)))
        start_list = list(payload.get('start_time', []))
        end_list = list(payload.get('end_time', []))
        fr = max(float(self.state.frame_rate), 1.0)
        events = []
        for idx, params in enumerate(params_list):
            source = str(sources[idx]) if idx < len(sources) else 'trace'
            label = str(labels[idx]) if idx < len(labels) else f'trace event {idx + 1}'
            color = str(colors[idx]) if idx < len(colors) else '#ff9900'
            if source == 'trace' and isinstance(params, dict) and params:
                event = {
                    'source': 'trace',
                    'label': label,
                    'color': color,
                    'start_frames': np.asarray([], dtype=int),
                    'end_frames': np.asarray([], dtype=int),
                    'mode': str(params.get('mode', 'discard')),
                    'diff': float(params.get('diff', 0.0)),
                    'source_name': str(params.get('source_name', '')),
                    'trace_params': dict(params),
                }
                self._apply_trace_event_detection_to_event(event)
                events.append(event)
                continue

            if idx < len(start_list) and idx < len(end_list):
                starts_sec = np.asarray(start_list[idx], dtype=float).reshape(-1)
                ends_sec = np.asarray(end_list[idx], dtype=float).reshape(-1)
                valid = np.isfinite(starts_sec) & np.isfinite(ends_sec)
                events.append(
                    {
                        'source': source or 'npy file',
                        'label': label,
                        'color': color,
                        'start_frames': np.rint(starts_sec[valid] * fr).astype(int),
                        'end_frames': np.rint(ends_sec[valid] * fr).astype(int),
                        'mode': 'label',
                    }
                )
        return events

    def _add_trace_event(self):
        params = self._trace_event_params_from_panel()
        source_name = str(params.get('source_name') or self.event_trace_source_combo.currentText())
        params['source_name'] = source_name
        row_idx = len(self.events)
        self.events.append(
            {
                'source': 'trace',
                'label': f'trace event {row_idx + 1}',
                'color': '#ff9900',
                'start_frames': np.asarray([], dtype=int),
                'end_frames': np.asarray([], dtype=int),
                'mode': params['mode'],
                'diff': params['diff'],
                'source_name': source_name,
                'trace_params': params,
            }
        )
        self._invalidate_trace_cache()
        self._refresh_event_table()
        self.event_table.selectRow(row_idx)
        self._set_status('Added an empty trace event. Edit extractor parameters and click Apply.')
        self.render_all()

    def _detect_trace_event_intervals(self, trace_mat: np.ndarray, params: dict[str, Any]) -> list[tuple[int, int]]:
        if str(params.get('direction', 'above')) == 'between':
            n_frames = int(np.asarray(trace_mat).shape[-1]) if np.asarray(trace_mat).ndim else 0
            start = int(np.clip(int(params.get('from', 0)), 0, max(n_frames, 0)))
            stop = int(np.clip(int(params.get('to', 0)), 0, max(n_frames, 0)))
            if stop < start:
                start, stop = stop, start
            return [(start, stop)] if stop > start else []
        return detect_trace_event_intervals(
            trace_mat,
            self.state.frame_rate,
            value=float(params.get('value', 0.0)),
            direction=str(params.get('direction', 'above')),
            interval=float(params.get('interval', 0.0)),
            adjacent=float(params.get('adjacent', 0.0)),
            merge=float(params.get('merge', 0.0)),
        )

    def _single_trace_for_event_detection(
        self,
        trace_mat: np.ndarray,
        mapping: Optional[np.ndarray],
    ) -> tuple[np.ndarray, Optional[int], int]:
        trace_mat = np.asarray(trace_mat, dtype=float)
        if trace_mat.ndim == 0:
            return np.zeros((0, 0), dtype=float), None, 0
        if trace_mat.ndim == 1:
            trace_mat = trace_mat.reshape(1, -1)
        if trace_mat.ndim != 2 or trace_mat.shape[0] == 0:
            return np.zeros((0, 0), dtype=float), None, 0

        selected = self.get_selected_roi_indices()
        pairs = self._result_roi_row_pairs({'data': trace_mat, 'roi_indices': mapping}, selected)
        if pairs:
            roi_idx, row_idx = pairs[0]
            return trace_mat[row_idx:row_idx + 1], int(roi_idx), int(row_idx)
        return trace_mat[0:1], None, 0

    def _on_trace_event_params_changed(self):
        if self._updating_event_trace_panel:
            return
        row = self._selected_event_row()
        if row is None or row >= len(self.events) or self.events[row].get('source') != 'trace':
            return
        params = self._trace_event_params_from_panel()
        self.events[row]['trace_params'] = params
        self._apply_trace_event_detection_to_event(self.events[row])
        self._invalidate_trace_cache()
        self._refresh_event_table()
        self.event_table.selectRow(row)
        self.render_all()

    def save_events_callback(self):
        if not self.events:
            self._set_status('No events to save.')
            return
        out, _ = QFileDialog.getSaveFileName(self, 'Save events', '', 'NumPy (*.npy)')
        if not out:
            return
        fr = max(float(self.state.frame_rate), 1.0)
        payload = {
            'start_time': [],
            'end_time': [],
            'label': [str(event.get('label') or '') for event in self.events],
            'source': [str(event.get('source') or '') for event in self.events],
            'color': [self._event_color_hex(event) for event in self.events],
            'params': [],
        }
        for event in self.events:
            if event.get('source') == 'trace':
                payload['start_time'].append(np.asarray([], dtype=float))
                payload['end_time'].append(np.asarray([], dtype=float))
                payload['params'].append(dict(event.get('trace_params') or {}))
            else:
                payload['start_time'].append(np.asarray(event.get('start_frames', []), dtype=float).reshape(-1) / fr)
                payload['end_time'].append(np.asarray(event.get('end_frames', []), dtype=float).reshape(-1) / fr)
                payload['params'].append({})
        np.save(out, payload, allow_pickle=True)
        self._set_status(f'Saved events: {out}')

    def _active_event_indices(self) -> list[int]:
        if not hasattr(self, 'avg_event_list'):
            return list(range(len(self.events)))
        selected = [int(item.data(Qt.ItemDataRole.UserRole)) for item in self.avg_event_list.selectedItems()]
        return selected if selected else list(range(len(self.events)))

    def _event_frame_ranges(self, event: dict[str, Any]) -> list[tuple[int, int]]:
        starts = np.asarray(event.get('start_frames', []), dtype=int).reshape(-1)
        ends = np.asarray(event.get('end_frames', []), dtype=int).reshape(-1)
        count = min(starts.size, ends.size)
        ranges = []
        for start, end in zip(starts[:count], ends[:count]):
            if end < start:
                start, end = end, start
            ranges.append((max(0, int(start)), max(0, int(end))))
        return ranges

    def _apply_event_trace_actions(self, trace_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_frames = trace_mat.shape[1]
        frame_indices = np.arange(n_frames, dtype=int)
        keep = np.ones(n_frames, dtype=bool)
        trace_mat = np.asarray(trace_mat, dtype=float).copy()
        for event in self.events:
            if event.get('source') != 'trace':
                continue
            ranges = self._event_frame_ranges(event)
            params = dict(event.get('trace_params') or {})
            mode = str(params.get('mode', event.get('mode', 'label')))
            if mode == 'raw':
                continue
            if mode == 'replace':
                diff = float(params.get('diff', event.get('diff', 0.0)))
                for start, stop in ranges:
                    for frame in range(max(start, 1), min(stop, n_frames)):
                        delta = np.abs(trace_mat[:, frame] - trace_mat[:, frame - 1])
                        replace = delta > diff
                        trace_mat[replace, frame] = trace_mat[replace, frame - 1]
            elif mode == 'discard':
                for start, stop in ranges:
                    keep[max(0, start):min(n_frames, stop)] = False
        return trace_mat[:, keep], frame_indices[keep]


    def _toggle_image_play(self):
        if self._image_play_timer.isActive():
            self._image_play_timer.stop()
            self.image_play_button.setText('Play')
            return
        if self.image_frame_slider.maximum() <= 0:
            return
        self._image_play_timer.start(max(10, int(1000 / max(self.state.frame_rate, 1.0))))
        self.image_play_button.setText('Pause')

    def _advance_image_frame(self):
        slider = self.image_frame_slider
        if slider.maximum() <= 0:
            self._image_play_timer.stop()
            self.image_play_button.setText('Play')
            return
        next_value = slider.value() + 1
        if next_value > slider.maximum():
            next_value = 0
        slider.setValue(next_value)

    def _on_trace_frame_slider_changed(self, value: int):
        if not hasattr(self, 'trace_widget') or getattr(self.trace_widget, '_x_locked', False):
            return
        x_start = self._trace_frame_to_x(value)
        travel = max(self.trace_widget._x_total - self.trace_widget._x_window, 0.0)
        self.trace_widget._x_start = float(np.clip(x_start, 0.0, travel))
        self.trace_widget._apply_x_limits(self.trace_widget._x_start, self.trace_widget._x_start + self.trace_widget._x_window)
        self.trace_widget.canvas.draw_idle()
        self._sync_trace_frame_slider()

    def _sync_trace_frame_slider(self):
        if not hasattr(self, 'trace_frame_slider'):
            return
        max_frame = max(0, self._trace_x_to_frame(max(self.trace_widget._x_total - self.trace_widget._x_window, 0.0)))
        current = max(0, self._trace_x_to_frame(self.trace_widget._x_start))
        self.trace_widget._x_locked = True
        try:
            self.trace_frame_slider.setMaximum(max_frame)
            self.trace_frame_slider.setEnabled(max_frame > 0)
            self.trace_frame_slider.setValue(min(current, max_frame))
        finally:
            self.trace_widget._x_locked = False
        total_frames = max(1, self._trace_x_to_frame(self.trace_widget._x_total))
        self.trace_frame_label.setText(f'{min(current + 1, total_frames)} / {total_frames}')

    def _toggle_trace_play(self):
        if self._trace_play_timer.isActive():
            self._trace_play_timer.stop()
            self.trace_play_button.setText('Play')
            return
        if self.trace_widget._x_total <= self.trace_widget._x_window + 1e-9:
            return
        self._trace_play_timer.start(80)
        self.trace_play_button.setText('Pause')

    def _advance_trace_window(self):
        widget = self.trace_widget
        travel = max(widget._x_total - widget._x_window, 0.0)
        if travel <= 1e-9:
            self._trace_play_timer.stop()
            self.trace_play_button.setText('Play')
            return
        min_step = 1.0 if self._trace_x_unit() == 'frames' else 1.0 / max(self.state.frame_rate, 1.0)
        step = max(widget._x_window / 20.0, min_step)
        widget._x_start = 0.0 if widget._x_start + step > travel else widget._x_start + step
        widget._sync_slider_from_x_window()
        widget._apply_x_limits(widget._x_start, widget._x_start + widget._x_window)
        self._sync_trace_frame_slider()
        widget.canvas.draw_idle()

    def _set_status(self, text: str):
        self.statusBar().showMessage(text)

    def _start_background_load(self, loader, label: str, done_message: str):
        if self._active_thread is not None:
            self._set_status('A background load is already running.')
            return
        self._set_controls_enabled(False)
        self._set_status('Loading data in background...')
        thread = QThread(self)
        worker = BackgroundTask(loader)
        worker.moveToThread(thread)
        self._pending_load_label = label
        self._pending_load_done_message = done_message
        thread.started.connect(worker.run)
        worker.finished.connect(self._finish_background_load)
        worker.failed.connect(self._fail_background_load)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._active_thread = thread
        self._active_worker = worker
        thread.start()

    def _start_background_job(self, job, running_message: str, done_message: str, on_success=None, with_progress: bool = False):
        if self._active_thread is not None:
            self._set_status('A background task is already running.')
            return False
        self._set_controls_enabled(False)
        self._set_status(running_message)
        thread = QThread(self)
        worker = BackgroundTask(job, with_progress=with_progress)
        worker.moveToThread(thread)
        self._pending_job_done_message = done_message
        self._pending_job_success = on_success
        thread.started.connect(worker.run)
        worker.progress.connect(self._set_status)
        worker.finished.connect(self._finish_background_job)
        worker.failed.connect(self._fail_background_job)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._active_thread = thread
        self._active_worker = worker
        thread.start()
        return True

    def _finish_background_job(self, result: Any):
        callback = self._pending_job_success
        if callback is not None:
            callback(result)
        message = self._pending_job_done_message
        self._pending_job_done_message = ''
        self._pending_job_success = None
        self._set_status(message.format(result=result) if message else 'Background task finished.')
        self._close_background_thread()
        self._set_controls_enabled(True)
        self._start_next_pipeline_denoise_job()

    def _fail_background_job(self, message: str):
        QMessageBox.critical(self, 'Processing failed', message)
        self._pipeline_denoise_queue.clear()
        self._pending_job_done_message = ''
        self._pending_job_success = None
        self._set_status('Processing failed.')
        self._close_background_thread()
        self._set_controls_enabled(True)

    def _finish_background_load(self, state: GUIState):
        self.state = state
        self.vpy = self.state.vpy
        globals()['vpy'] = self.vpy
        self.source_label.setText(self._pending_load_label)
        self._image_projection_cache.clear()
        self._direct_trace_cache.clear()
        self._computed_trace_cache.clear()
        self._trace_result_cache.clear()
        self._init_events_from_state()
        pipeline_loaded = self._auto_load_pipeline_for_state()
        if not pipeline_loaded:
            self._apply_no_pipeline_defaults()
        self._load_image_info_from_disk()
        self._update_image_info_table()
        self._update_process_controls()
        self._refresh_event_table()
        self._refresh_avg_event_list()
        self._refresh_roi_list(select_first=True)
        self.render_all(reset_trace_view=True)
        suffix = ' Pipeline loaded.' if pipeline_loaded else ''
        self._set_status(f'{self._pending_load_done_message}{suffix}')
        self._close_background_thread()
        self._set_controls_enabled(True)
        self._start_next_pipeline_denoise_job()

    def _fail_background_load(self, message: str):
        QMessageBox.critical(self, 'Load failed', message)
        self._set_status('Load failed.')
        self._close_background_thread()
        self._set_controls_enabled(True)

    def _apply_no_pipeline_defaults(self):
        if hasattr(self, 'data_polarity_button'):
            self.data_polarity_button.blockSignals(True)
            self.data_polarity_button.setChecked(False)
            self.data_polarity_button.setText('pos')
            self.data_polarity_button.blockSignals(False)
        if hasattr(self, 'data_raw_checkbox'):
            self.data_raw_checkbox.blockSignals(True)
            self.data_raw_checkbox.setChecked(False)
            self.data_raw_checkbox.blockSignals(False)
        if hasattr(self, 'show_masks_checkbox'):
            self.show_masks_checkbox.setChecked(True)
        if hasattr(self, 'image_labels_checkbox'):
            self.image_labels_checkbox.setChecked(True)
        if hasattr(self, 'trace_labels_checkbox'):
            self.trace_labels_checkbox.setChecked(True)
        self._reset_image_layers_for_default()
        row = self._reset_trace_rows_for_default()
        self._refresh_trace_source_options()
        if self.state.source_type == 'table':
            self._set_combo_data(row.source_combo, 'state:0')
        elif self.state.source_type == 'folder':
            self._set_combo_data(row.source_combo, 'image:raw:suite2p')
        self._computed_trace_cache.clear()
        self._direct_trace_cache.clear()
        self._invalidate_trace_cache()

    def _reset_image_layers_for_default(self):
        while len(self.image_layers) > 1:
            self.remove_image_layer(self.image_layers[-1])
        if not self.image_layers:
            return
        row = self.image_layers[0]
        if row.model_combo is not None:
            self._set_combo_text(row.model_combo, 'Raw')
        row.image_layer_params = {}
        row.visible_checkbox.setChecked(self.state.source_type == 'folder')
        self._set_combo_data(row.mode_combo, 'video')
        self._set_combo_data(row.mask_source_combo, 'suite2p')

    def _reset_trace_rows_for_default(self) -> TraceControlRow:
        while len(self.trace_rows) > 1:
            self.remove_trace_row(self.trace_rows[-1])
        if not self.trace_rows:
            self.add_trace_row()
        row = self.trace_rows[0]
        row.visible_checkbox.setChecked(True)
        row.fold_button.setChecked(False)
        self._set_trace_row_folded(row, False)
        for checkbox in (
            row.lowpass_checkbox,
            row.highpass_checkbox,
            row.wavelet_checkbox,
            row.pca_wavelet_checkbox,
            row.snr_checkbox,
            row.volpy_checkbox,
            row.baseline_lowpass_checkbox,
            row.baseline_rolling_checkbox,
            row.baseline_polyfit_checkbox,
            row.baseline_savgol_checkbox,
            row.spike_checkbox,
            row.threshold_checkbox,
            row.waveform_checkbox,
        ):
            checkbox.setChecked(False)
        row.pca_wavelet_cfg = self._default_pca_wavelet_cfg_dict(
            f_min=self._safe_float(row.pca_wavelet_fmin_edit.text(), default=1.0),
            f_max=self._safe_float(row.pca_wavelet_fmax_edit.text(), default=max(self.state.frame_rate, 1.0)),
            f_n=int(self._safe_float(row.pca_wavelet_fn_edit.text(), default=100)),
        )
        return row

    def _close_background_thread(self):
        if self._active_thread is None:
            self._active_worker = None
            return
        thread = self._active_thread
        self._active_thread = None
        self._active_worker = None
        self._pending_load_label = ''
        self._pending_load_done_message = ''
        self._pending_job_done_message = ''
        self._pending_job_success = None
        thread.quit()
        thread.wait()

    def _set_controls_enabled(self, enabled: bool):
        if hasattr(self, 'left_scroll'):
            self.left_scroll.setEnabled(enabled)
        if hasattr(self, 'plot_scroll'):
            self.plot_scroll.setEnabled(enabled)

    def load_folder_callback(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Bruker data folder')
        if not folder:
            return
        negative = self._data_negative_enabled()
        self._start_background_load(lambda: self._load_folder_state(folder, negative=negative), folder, 'Bruker folder loaded.')

    def load_femtonics_xlsx_callback(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Select Femtonics xlsx file',
            '',
            'Excel workbook (*.xlsx);;All files (*.*)',
        )
        if not path:
            return
        negative = self._data_negative_enabled()
        self._start_background_load(lambda: self._load_table_state(path, negative=negative), path, 'Femtonics xlsx file loaded.')

    def load_volpy_folder_callback(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select VolPy folder')
        if not folder:
            return
        self._start_background_load(lambda: self._load_volpy_folder_state(folder), folder, 'VolPy folder loaded.')

    def about_callback(self):
        QMessageBox.information(
            self,
            'About NeuroBox',
            'NeuroBox-style 2P ROI GUI\n\n'
            'Left: data, image, trace, and average controls\n'
            'Right: image with ROI controls, trace panel, and average panel\n\n'
            'PyQt6 version with Qt-native matplotlib embedding, wheel zoom, '\
            'and shared plot scrolling.\n\n'
            'Functions not fully specified are routed to developing().'
        )

    def export_panel_callback(self, panel_name: str):
        fig_map = {
            'image': self.image_widget.figure,
            'trace': self.trace_widget.figure,
            'average': self.avg_widget.figure,
        }
        fig = fig_map[panel_name]
        out, _ = QFileDialog.getSaveFileName(
            self,
            f'Export {panel_name} panel',
            '',
            'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)',
        )
        if not out:
            return
        try:
            fig.savefig(out, dpi=300, bbox_inches='tight')
            self._set_status(f'Exported: {out}')
        except Exception as exc:
            QMessageBox.critical(self, 'Export failed', str(exc))
            self._set_status('Export failed.')

    def export_image_layer_callback(self, row: ImageLayerControlRow):
        image, _nframes, _title = self._get_image_layer_display(row)
        if image is None:
            self._set_status('No image is available for export.')
            return
        default_path = self._image_layer_export_default_path(row)
        out, _ = QFileDialog.getSaveFileName(
            self,
            'Export image layer',
            str(default_path),
            'TIFF (*.tiff *.tif);;All files (*.*)',
        )
        if not out:
            return
        arr = np.asarray(image, dtype=float)
        if hasattr(self, 'show_masks_checkbox') and self.show_masks_checkbox.isChecked():
            rgb = np.repeat(normalize_image(arr)[..., None], 3, axis=2)
            rgba = self._build_mask_rgba(arr.shape, source=row.mask_source_combo.currentData())
            if rgba is not None:
                alpha = np.clip(rgba[..., 3:4], 0.0, 1.0)
                rgb = rgb * (1.0 - alpha) + rgba[..., :3] * alpha
            tifffile.imwrite(out, np.clip(rgb * 255.0, 0, 255).astype(np.uint8), photometric='rgb')
        else:
            tifffile.imwrite(out, arr.astype(np.float32, copy=False))
        self._set_status(f'Exported image layer: {out}')

    def _image_layer_export_default_path(self, row: ImageLayerControlRow) -> Path:
        folder = self._pipeline_data_folder() or Path.cwd()
        mode = str(row.mode_combo.currentData() or 'image')
        return folder / f'image_{row.layer_id}_{mode}_{self._database_name()}.tiff'

    def export_waveform_callback(self, row: TraceControlRow):
        result = self._build_trace_result(row)
        if result is None:
            self._set_status('No trace result is available for waveform export.')
            return
        spike_times = result.get('spike_times')
        if spike_times is None:
            self._set_status('Enable spike detection before exporting waveforms.')
            return
        mode = row.waveform_mode_combo.currentText()
        modes = ['raw', 'current'] if mode == 'both' else [mode]
        out_paths = []
        for source_mode in modes:
            payload = self._waveform_export_payload(row, result, source_mode)
            out_path = self._waveform_export_path(row, source_mode)
            np.save(out_path, payload, allow_pickle=True)
            out_paths.append(str(out_path))
        self._set_status(f'Exported waveforms: {", ".join(out_paths)}')

    def _waveform_export_path(self, row: TraceControlRow, source_mode: str, output_dir: Optional[Path] = None) -> Path:
        trace_idx = self.trace_rows.index(row) + 1 if row in self.trace_rows else 1
        prefix = 'waveform_raw' if source_mode == 'raw' else 'waveform'
        folder = Path(output_dir) if output_dir is not None else (self._pipeline_data_folder() or Path.cwd())
        return folder / f'{prefix}_{trace_idx}_{self._database_name()}.npy'

    def _waveform_export_payload(
        self,
        row: TraceControlRow,
        result: dict[str, Any],
        source_mode: str,
        pipeline_path: str = 'none',
    ) -> dict[str, Any]:
        spike_times = result.get('spike_times') or []
        fr = max(float(self.state.frame_rate), 1e-8)
        pre = self._safe_float(self.waveform_pre_edit.text(), default=-0.025)
        post = self._safe_float(self.waveform_post_edit.text(), default=0.025)
        start_offset = int(np.floor(pre * fr))
        stop_offset = int(np.ceil(post * fr))
        offsets = np.arange(start_offset, stop_offset + 1, dtype=int)
        peak = int(-start_offset)
        trace_key = 'raw_data' if source_mode == 'raw' else 'data'
        trace_mat = np.asarray(result.get(trace_key, result['data']), dtype=float)
        waveforms = []
        spike_time_list = []
        waveform_features = []
        counts = np.zeros(trace_mat.shape[0], dtype=int)
        snr_values = np.full(trace_mat.shape[0], np.nan, dtype=float)
        result_snr = result.get('snr_values')
        if result_snr is not None:
            for row_idx in range(min(trace_mat.shape[0], len(result_snr))):
                value = result_snr[row_idx]
                if value is not None and np.isfinite(float(value)):
                    snr_values[row_idx] = float(value)
        for row_idx in range(trace_mat.shape[0]):
            events = np.asarray(spike_times[row_idx], dtype=float) if row_idx < len(spike_times) else np.asarray([], dtype=float)
            event_frames = np.rint(events * fr).astype(int)
            if event_frames.size == 0 or offsets.size == 0:
                empty = np.zeros((0, offsets.size), dtype=float)
                waveforms.append(empty)
                spike_time_list.append(np.asarray([], dtype=float))
                waveform_features.append(cal_waveform.empty_average_peak_features(row_idx))
                continue
            indices = event_frames[:, None] + offsets[None, :]
            valid = np.all((indices >= 0) & (indices < trace_mat.shape[1]), axis=1)
            if not np.any(valid):
                empty = np.zeros((0, offsets.size), dtype=float)
                waveforms.append(empty)
                spike_time_list.append(np.asarray([], dtype=float))
                waveform_features.append(cal_waveform.empty_average_peak_features(row_idx))
                continue
            extracted = trace_mat[row_idx, indices[valid]]
            waveforms.append(np.asarray(extracted, dtype=float))
            spike_time_list.append(events[valid])
            counts[row_idx] = extracted.shape[0]
            summary = cal_waveform.quantify_average_peak_waveform(
                extracted,
                fr,
                peak_index=peak,
                spike_index=row_idx,
            )
            waveform_features.append(summary['features'])
        method = row.spike_method_combo.currentText().lower()
        source = 'volpy' if method == 't_res' else method
        full_duration = trace_mat.shape[1] / fr if trace_mat.shape[1] else np.nan
        average_fr = counts.astype(float) / full_duration if np.isfinite(full_duration) and full_duration > 0 else np.full(counts.shape, np.nan)
        spikes = {
            'nROI': int(trace_mat.shape[0]),
            'nSpikes': counts,
            'waveform': waveforms,
            'average_fr': average_fr,
            'snr': snr_values,
            'window': offsets.astype(float) / fr,
            'spike_time': spike_time_list,
            'waveform_features': waveform_features,
            'framerate': fr,
            'peak': peak,
            'source': source,
            'waveform_source': source_mode,
            'k': None if source == 'volpy' else self._safe_float(row.spike_k_edit.text(), default=5.0),
            'pipeline': pipeline_path or 'none',
        }
        return {'spikes': spikes}

    # ------------------------------ pipeline state ------------------------------
    def _database_name(self) -> str:
        if self.state.tif_path:
            return Path(self.state.tif_path).stem
        if self.state.source_path:
            path = Path(self.state.source_path)
            return path.stem if path.is_file() else path.name
        return 'neurobox'

    def _pipeline_data_folder(self) -> Optional[Path]:
        if self.state.tif_path:
            return Path(self.state.tif_path).parent
        if self.state.source_path:
            path = Path(self.state.source_path)
            return path if path.is_dir() else path.parent
        return None

    def _cache_dir(self) -> Path:
        folder = self._pipeline_data_folder() or Path.cwd()
        return folder / '.neurobox_cache'

    def _json_ready(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): self._json_ready(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (list, tuple)):
            return [self._json_ready(item) for item in value]
        return value

    def _cache_token(self, payload: dict[str, Any]) -> str:
        text = json.dumps(self._json_ready(payload), sort_keys=True, separators=(',', ':'))
        return hashlib.sha1(text.encode('utf-8')).hexdigest()[:20]

    def _file_signature(self, path: Optional[Path]) -> dict[str, Any]:
        if path is None:
            return {'path': '', 'exists': False}
        path = Path(path)
        if not path.exists():
            return {'path': str(path), 'exists': False}
        stat = path.stat()
        return {
            'path': str(path.resolve()),
            'exists': True,
            'size': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns),
        }

    def _image_layer_movie_source_path(self, row: ImageLayerControlRow, model: str) -> Optional[Path]:
        if model == 'Raw':
            return Path(self.state.tif_path) if self.state.tif_path else None
        if model == 'VolPy':
            return self._find_volpy_mmap_path() or self._find_volpy_tif_path()
        if model == 'NoRMCorre':
            params = self._image_layer_params_for_row(row, model)
            return self._image_layer_output_path(row, params, default_name=self._image_layer_default_output_name(model))
        if model == 'PMD':
            params = self._image_layer_params_for_row(row, model)
            return self._image_layer_output_path(row, params, default_name=self._image_layer_default_output_name(model))
        if model == 'Local':
            path_text = str(row.image_layer_params.get('local_path') or '').strip()
            return Path(path_text) if path_text else None
        return None

    def _image_layer_movie_cache_metadata(self, row: ImageLayerControlRow) -> dict[str, Any]:
        model = self._image_layer_model(row)
        return {
            'cache': 'image_layer_movie',
            'version': 1,
            'database_name': self._database_name(),
            'source_type': self.state.source_type,
            'source_path': self.state.source_path,
            'layer_id': row.layer_id,
            'model': model,
            'source_file': self._file_signature(self._image_layer_movie_source_path(row, model)),
        }

    def _image_layer_movie_cache_path(self, row: ImageLayerControlRow) -> Path:
        meta = self._image_layer_movie_cache_metadata(row)
        return self._cache_dir() / f"video_{self._cache_token(meta)}.npy"

    def _load_image_layer_movie_cache(self, row: ImageLayerControlRow) -> Optional[np.ndarray]:
        path = self._image_layer_movie_cache_path(row)
        if not path.exists():
            return None
        arr = np.load(path, mmap_mode='r', allow_pickle=False)
        return np.asarray(arr) if arr.ndim >= 3 else None

    def _save_image_layer_movie_cache(self, row: ImageLayerControlRow, movie: Any) -> np.ndarray:
        arr = np.asarray(movie)
        if arr.ndim < 3:
            return arr
        path = self._image_layer_movie_cache_path(row)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, arr, allow_pickle=False)
        return arr

    def _mask_signature(self, masks: np.ndarray) -> dict[str, Any]:
        arr = np.ascontiguousarray(np.asarray(masks, dtype=np.float32))
        digest = hashlib.sha1(arr.view(np.uint8)).hexdigest()
        return {'shape': tuple(int(v) for v in arr.shape), 'dtype': str(arr.dtype), 'sha1': digest}

    def _trace_cache_metadata(self, layer: ImageLayerControlRow, mask_source: str, masks: np.ndarray) -> dict[str, Any]:
        return {
            'cache': 'roi_trace',
            'version': 1,
            'movie': self._image_layer_movie_cache_metadata(layer),
            'mask_source': str(mask_source),
            'mask': self._mask_signature(masks),
            'negative_view': bool(self._negative_view_requested()),
            'negative_mode': bool(self.state.negative_mode),
            'intensity_max': None if self.state.intensity_max is None else float(self.state.intensity_max),
        }

    def _trace_cache_path(self, layer: ImageLayerControlRow, mask_source: str, masks: np.ndarray) -> Path:
        meta = self._trace_cache_metadata(layer, mask_source, masks)
        return self._cache_dir() / f"trace_{self._cache_token(meta)}.npy"

    def _load_roi_trace_cache(self, layer: ImageLayerControlRow, mask_source: str, masks: np.ndarray) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
        path = self._trace_cache_path(layer, mask_source, masks)
        if not path.exists():
            return None
        payload = np.load(path, allow_pickle=True).item()
        if not isinstance(payload, dict) or 'trace_mat' not in payload:
            return None
        mapping = payload.get('mapping')
        if mapping is not None:
            mapping = np.asarray(mapping, dtype=int)
        return np.asarray(payload['trace_mat'], dtype=float), mapping

    def _save_roi_trace_cache(
        self,
        layer: ImageLayerControlRow,
        mask_source: str,
        masks: np.ndarray,
        trace_mat: np.ndarray,
        mapping: Optional[np.ndarray],
    ):
        path = self._trace_cache_path(layer, mask_source, masks)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'metadata': self._trace_cache_metadata(layer, mask_source, masks),
            'trace_mat': np.asarray(trace_mat, dtype=np.float32),
            'mapping': None if mapping is None else np.asarray(mapping, dtype=int),
        }
        np.save(path, payload, allow_pickle=True)

    def _pipeline_file_path(self) -> Optional[Path]:
        folder = self._pipeline_data_folder()
        if folder is None:
            return None
        return folder / f'pipeline_{self._database_name()}.npy'

    def save_pipeline_callback(self):
        path = self._pipeline_file_path()
        if path is None:
            self._set_status('Load data before saving a pipeline.')
            return
        payload = self._collect_pipeline_payload()
        if path.exists():
            path.unlink()
        np.save(path, payload, allow_pickle=True)
        self._set_status(f'Saved pipeline: {path}')

    def apply_pipeline_callback(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Apply pipeline', '', 'NumPy (*.npy);;All files (*.*)')
        if not path:
            return
        loaded = np.load(path, allow_pickle=True)
        payload = loaded.item() if hasattr(loaded, 'shape') and loaded.shape == () else loaded
        if not isinstance(payload, dict) or payload.get('schema') != 'neurobox_pipeline':
            QMessageBox.critical(self, 'Invalid pipeline', 'Pipeline file must contain a NeuroBox pipeline dictionary.')
            return
        queued = self._apply_pipeline_payload(payload)
        suffix = f' Queued {queued} image denoising job(s).' if queued else ''
        self._set_status(f'Applied pipeline: {path}{suffix}')
        self._start_next_pipeline_denoise_job()

    def _auto_load_pipeline_for_state(self) -> bool:
        path = self._pipeline_file_path()
        if path is None or not path.exists():
            return False
        loaded = np.load(path, allow_pickle=True)
        payload = loaded.item() if hasattr(loaded, 'shape') and loaded.shape == () else loaded
        if not isinstance(payload, dict) or payload.get('schema') != 'neurobox_pipeline':
            return False
        self._apply_pipeline_payload(payload)
        return True

    def _queue_pipeline_denoise_rows(self, rows: list[ImageLayerControlRow]) -> int:
        denoise_ids = []
        for row in rows:
            model = self._image_layer_model(row)
            if model not in {'NoRMCorre', 'PMD'}:
                continue
            params = self._image_layer_run_params(row, model)
            if not self._denoise_output_available(row, model, params):
                denoise_ids.append(row.layer_id)
        if denoise_ids:
            self._pipeline_denoise_queue.extend(denoise_ids)
        return len(denoise_ids)

    def _start_next_pipeline_denoise_job(self):
        if self._active_thread is not None:
            return
        while self._pipeline_denoise_queue:
            layer_id = self._pipeline_denoise_queue.pop(0)
            row = next((item for item in self.image_layers if item.layer_id == layer_id), None)
            if row is None:
                continue
            model = self._image_layer_model(row)
            if model == 'NoRMCorre':
                params = self._image_layer_run_params(row, model)
                if self._denoise_output_available(row, model, params):
                    continue
                if self._apply_normcorre_from_dialog(row, params, prompt_local=False):
                    return
                continue
            if model == 'PMD':
                params = self._image_layer_run_params(row, model)
                if self._denoise_output_available(row, model, params):
                    continue
                if self._apply_pmd_from_dialog(row, params, prompt_local=False):
                    return
                continue

    def _image_layer_run_params(self, row: ImageLayerControlRow, model: str) -> dict[str, Any]:
        params = self._image_layer_params_for_row(row, model)
        for key in ('input_source', 'local_input_path', 'local_input_tiff'):
            if key in row.image_layer_params:
                params[key] = row.image_layer_params[key]
        params['_model'] = model
        return params

    def _denoise_output_available(self, row: ImageLayerControlRow, model: str, params: dict[str, Any]) -> bool:
        if bool(params.get('Reload')):
            return False
        output_path = self._image_layer_output_path(row, params, default_name=self._image_layer_default_output_name(model))
        return output_path.exists()

    def _run_pipeline_denoise_queue_sync(self):
        while self._pipeline_denoise_queue:
            layer_id = self._pipeline_denoise_queue.pop(0)
            row = next((item for item in self.image_layers if item.layer_id == layer_id), None)
            if row is None:
                continue
            model = self._image_layer_model(row)
            if model not in {'NoRMCorre', 'PMD'}:
                continue
            params = self._image_layer_run_params(row, model)
            if self._denoise_output_available(row, model, params):
                continue
            self._run_image_denoise_sync(row, model, params)

    def _run_image_denoise_sync(self, row: ImageLayerControlRow, model: str, params: dict[str, Any]) -> Path:
        row.image_layer_params = dict(params)
        if model == 'NoRMCorre':
            input_path = self._resolve_image_layer_input_path(
                str(params.get('input_source') or ''),
                row,
                params,
                prompt_local=False,
            )
            if input_path is None:
                raise ValueError('No valid input video is available for NoRMCorre.')
            output_path = self._image_layer_output_path(row, params, default_name=self._image_layer_default_output_name(model))
            result = normcorre_backend.run_normcorre(
                input_path=input_path,
                output_path=output_path,
                frame_rate=float(params.get('fr') or self.state.frame_rate or 1.0),
                params=params,
                reload=bool(params.get('Reload')),
            )
        elif model == 'PMD':
            input_path = self._resolve_image_layer_input_path(
                str(params.get('input_source') or ''),
                row,
                params,
                prompt_local=False,
            )
            if input_path is None:
                raise ValueError('No valid input video is available for PMD.')
            input_path = self._ensure_pmd_tiff_input(input_path, params, row)
            if input_path is None:
                raise ValueError('No valid TIFF input video is available for PMD.')
            image_shape = self._pmd_input_image_shape(input_path, params)
            if image_shape is None:
                raise ValueError('Could not determine the PMD input image shape.')
            pixel_weight_ok, pixel_weighting = self._pmd_pixel_weighting_from_params(params, image_shape, input_path)
            if not pixel_weight_ok:
                raise ValueError('PMD pixel weighting could not be prepared.')
            run_params = dict(params)
            if pixel_weighting is None:
                run_params.pop('pixel_weighting', None)
            else:
                run_params['pixel_weighting'] = pixel_weighting
            output_path = self._image_layer_output_path(row, params, default_name=self._image_layer_default_output_name(model))
            result = pmd_backend.run_pmd(
                input_path=input_path,
                output_path=output_path,
                frame_rate=float(self.state.frame_rate or 1.0),
                params=run_params,
                reload=bool(params.get('Reload')),
            )
        else:
            raise ValueError(f'Unsupported denoise model: {model}')
        self._image_projection_cache.clear()
        self._computed_trace_cache.clear()
        self._invalidate_trace_cache()
        return Path(result)

    def _collect_pipeline_payload(self) -> dict[str, Any]:
        return {
            'schema': 'neurobox_pipeline',
            'version': 2,
            'database_name': self._database_name(),
            'source_type': self.state.source_type,
            'state': {
                'data': self._collect_data_control_state(),
                'image_layers': [self._collect_image_layer_state(row) for row in self.image_layers],
                'events': [self._serialize_event(event) for event in self.events],
                'traces': [self._collect_trace_row_state(row) for row in self.trace_rows],
                'average': self._collect_average_state(),
                'roi': self._collect_roi_state(),
                'panels': self._collect_panel_state(),
            },
        }

    def _collect_data_control_state(self) -> dict[str, Any]:
        return {
            'polarity_negative': self.data_polarity_button.isChecked() if hasattr(self, 'data_polarity_button') else False,
            'raw_view': self.data_raw_checkbox.isChecked() if hasattr(self, 'data_raw_checkbox') else False,
        }

    def _collect_image_layer_state(self, row: ImageLayerControlRow) -> dict[str, Any]:
        return {
            'layer_id': row.layer_id,
            'model': self._image_layer_model(row),
            'visible': row.visible_checkbox.isChecked(),
            'mode': row.mode_combo.currentData(),
            'mask_source': row.mask_source_combo.currentData(),
            'params': dict(row.image_layer_params),
            'denoise': self._collect_image_denoise_state(row),
        }

    def _collect_image_denoise_state(self, row: ImageLayerControlRow) -> dict[str, Any]:
        model = self._image_layer_model(row)
        if model not in {'NoRMCorre', 'PMD', 'Local'}:
            return {}
        params = dict(row.image_layer_params)
        payload = {
            'model': model,
            'params': params,
            'input_source': params.get('input_source', ''),
            'output_name': params.get('output_name', ''),
            'default_output_name': False,
        }
        if model in {'NoRMCorre', 'PMD'}:
            default_name = self._image_layer_default_output_name(model)
            output_name = str(params.get('output_name') or default_name)
            output_path = Path(output_name)
            payload['output_name'] = output_name
            payload['default_output_name'] = (not output_path.is_absolute() and output_path.name == default_name)
        return payload

    def _collect_trace_row_state(self, row: TraceControlRow) -> dict[str, Any]:
        return {
            'visible': row.visible_checkbox.isChecked(),
            'folded': row.fold_button.isChecked(),
            'source_data': row.source_combo.currentData(),
            'lowpass': row.lowpass_checkbox.isChecked(),
            'lowpass_hz': row.lowpass_edit.text(),
            'highpass': row.highpass_checkbox.isChecked(),
            'highpass_hz': row.highpass_edit.text(),
            'wavelet': row.wavelet_checkbox.isChecked(),
            'wavelet_name': row.wavelet_name_edit.text(),
            'wavelet_level': row.wavelet_level_edit.text(),
            'wavelet_scale': row.wavelet_scale_edit.text(),
            'wavelet_mode': row.wavelet_mode_combo.currentText(),
            'pca_wavelet': row.pca_wavelet_checkbox.isChecked(),
            'pca_wavelet_fmin': row.pca_wavelet_fmin_edit.text(),
            'pca_wavelet_fmax': row.pca_wavelet_fmax_edit.text(),
            'pca_wavelet_fn': row.pca_wavelet_fn_edit.text(),
            'pca_wavelet_cfg': dict(row.pca_wavelet_cfg),
            'snr': row.snr_checkbox.isChecked(),
            'snr_window': row.snr_window_edit.text(),
            'volpy': row.volpy_checkbox.isChecked(),
            'volpy_key': row.volpy_combo.currentText(),
            'baseline_mode': row.baseline_mode_combo.currentData(),
            'baseline_lowpass': row.baseline_lowpass_checkbox.isChecked(),
            'baseline_lowpass_hz': row.baseline_lowpass_edit.text(),
            'baseline_rolling': row.baseline_rolling_checkbox.isChecked(),
            'baseline_rolling_mode': row.baseline_rolling_mode_combo.currentText(),
            'baseline_rolling_window': row.baseline_rolling_window_edit.text(),
            'baseline_polyfit': row.baseline_polyfit_checkbox.isChecked(),
            'baseline_poly_order': row.baseline_poly_order_edit.text(),
            'baseline_savgol': row.baseline_savgol_checkbox.isChecked(),
            'baseline_savgol_window': row.baseline_savgol_window_edit.text(),
            'baseline_savgol_order': row.baseline_savgol_order_edit.text(),
            'spike': row.spike_checkbox.isChecked(),
            'spike_method': row.spike_method_combo.currentText(),
            'spike_k': row.spike_k_edit.text(),
            'threshold': row.threshold_checkbox.isChecked(),
            'waveform': row.waveform_checkbox.isChecked(),
            'waveform_mode': row.waveform_mode_combo.currentText(),
        }

    def _serialize_event(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = dict(event)
        payload['start_frames'] = np.asarray(event.get('start_frames', []), dtype=int)
        payload['end_frames'] = np.asarray(event.get('end_frames', []), dtype=int)
        payload['trace_params'] = dict(event.get('trace_params') or {})
        return payload

    def _collect_average_state(self) -> dict[str, Any]:
        return {
            'modes': sorted(self.avg_modes),
            'avg_pre': self.avg_pre_edit.text(),
            'avg_post': self.avg_post_edit.text(),
            'waveform_pre': self.waveform_pre_edit.text(),
            'waveform_post': self.waveform_post_edit.text(),
            'show_individual': self.avg_individual_button.isChecked() if hasattr(self, 'avg_individual_button') else True,
            'selected_traces': [int(item.data(Qt.ItemDataRole.UserRole)) for item in self.avg_trace_list.selectedItems()] if hasattr(self, 'avg_trace_list') else [],
            'selected_events': [int(item.data(Qt.ItemDataRole.UserRole)) for item in self.avg_event_list.selectedItems()] if hasattr(self, 'avg_event_list') else [],
        }

    def _collect_roi_state(self) -> dict[str, Any]:
        return {
            'only_cells': self.only_cells_checkbox.isChecked() if hasattr(self, 'only_cells_checkbox') else False,
            'show_masks': self.show_masks_checkbox.isChecked() if hasattr(self, 'show_masks_checkbox') else True,
            'combine_mode': self.combine_mode,
            'mask_color': self.mask_color_combo.currentData() if hasattr(self, 'mask_color_combo') else 'roi',
            'selected_rois': list(self.selected_roi_indices),
        }

    def _collect_panel_state(self) -> dict[str, Any]:
        return {
            'trace_window': self.trace_window_edit.text() if hasattr(self, 'trace_window_edit') else '5',
            'trace_x_unit': self.trace_x_unit_combo.currentData() if hasattr(self, 'trace_x_unit_combo') else 'seconds',
            'image_labels': self.image_labels_checkbox.isChecked() if hasattr(self, 'image_labels_checkbox') else True,
            'trace_labels': self.trace_labels_checkbox.isChecked() if hasattr(self, 'trace_labels_checkbox') else True,
            'average_labels': self.avg_labels_checkbox.isChecked() if hasattr(self, 'avg_labels_checkbox') else True,
        }

    def _apply_pipeline_payload(self, payload: dict[str, Any]) -> int:
        state = payload.get('state', {})
        layer_id_map: dict[str, str] = {}
        self._loading_pipeline = True
        try:
            self._apply_data_control_state(state.get('data', {}))
            layer_id_map = self._apply_image_layer_states(state.get('image_layers', []), payload)
            self.events = [self._deserialize_event(item) for item in state.get('events', []) if isinstance(item, dict)]
            self._apply_trace_row_states(state.get('traces', []), layer_id_map=layer_id_map)
            self._apply_average_state(state.get('average', {}))
            self._apply_roi_state(state.get('roi', {}))
            self._apply_panel_state(state.get('panels', {}))
        finally:
            self._loading_pipeline = False
        self._computed_trace_cache.clear()
        self._direct_trace_cache.clear()
        self._invalidate_trace_cache()
        self._refresh_trace_source_options()
        self._refresh_event_trace_source_options()
        self._refresh_event_table()
        self._refresh_avg_event_list()
        self._refresh_avg_trace_list()
        self._restore_average_selections()
        self.render_all(reset_trace_view=True)
        return self._queue_pipeline_denoise_rows(self.image_layers)

    def _apply_data_control_state(self, data: dict[str, Any]):
        if hasattr(self, 'data_polarity_button'):
            self.data_polarity_button.setChecked(bool(data.get('polarity_negative', self.data_polarity_button.isChecked())))
        if hasattr(self, 'data_raw_checkbox'):
            self.data_raw_checkbox.setChecked(bool(data.get('raw_view', self.data_raw_checkbox.isChecked())))

    def _apply_image_layer_states(self, states: list[dict[str, Any]], payload: Optional[dict[str, Any]] = None) -> dict[str, str]:
        states = list(states or [])
        if not states:
            return {}
        while len(self.image_layers) > 1:
            self.remove_image_layer(self.image_layers[-1])
        while len(self.image_layers) < len(states):
            self.add_image_layer()
        layer_id_map = self._assign_pipeline_layer_ids(states)
        for row, row_state in zip(self.image_layers, states):
            model = str(row_state.get('model', 'Raw'))
            params = self._pipeline_image_layer_params(row_state, model)
            params = self._remap_image_layer_params(params, layer_id_map)
            params = self._rebase_pipeline_output_name(params, model, payload or {}, row_state)
            row.image_layer_params = params
            self._set_combo_text(row.model_combo, model)
            row.visible_checkbox.setChecked(bool(row_state.get('visible', True)))
            self._set_combo_data(row.mode_combo, row_state.get('mode', 'video'))
            self._set_combo_data(row.mask_source_combo, row_state.get('mask_source', 'suite2p'))
        self._refresh_mask_target_list()
        self._refresh_trace_source_options()
        return layer_id_map

    def _assign_pipeline_layer_ids(self, states: list[dict[str, Any]]) -> dict[str, str]:
        used: set[str] = set()
        layer_id_map: dict[str, str] = {}
        for idx, (row, row_state) in enumerate(zip(self.image_layers, states)):
            saved_id = str(row_state.get('layer_id') or '').strip()
            if idx == 0:
                new_id = 'raw'
            elif saved_id and ':' not in saved_id and saved_id not in used and saved_id != 'raw':
                new_id = saved_id
            else:
                new_id = self._next_available_image_layer_id(used)
            old_id = saved_id or row.layer_id
            layer_id_map[old_id] = new_id
            row.layer_id = new_id
            used.add(new_id)
        self._sync_image_layer_counter_from_ids()
        return layer_id_map

    def _next_available_image_layer_id(self, used: set[str]) -> str:
        idx = max(int(self._image_layer_counter), 0) + 1
        while f'image_{idx}' in used:
            idx += 1
        self._image_layer_counter = idx
        return f'image_{idx}'

    def _sync_image_layer_counter_from_ids(self):
        max_id = int(self._image_layer_counter)
        for row in self.image_layers:
            if row.layer_id.startswith('image_'):
                suffix = row.layer_id.split('_', 1)[1]
                if suffix.isdigit():
                    max_id = max(max_id, int(suffix))
        self._image_layer_counter = max_id

    def _pipeline_image_layer_params(self, row_state: dict[str, Any], model: str) -> dict[str, Any]:
        params = dict(row_state.get('params') or {})
        denoise = row_state.get('denoise') if isinstance(row_state.get('denoise'), dict) else {}
        if denoise and str(denoise.get('model') or model) == model:
            params.update(dict(denoise.get('params') or {}))
            if denoise.get('input_source') and 'input_source' not in params:
                params['input_source'] = denoise.get('input_source')
            if denoise.get('output_name') and 'output_name' not in params:
                params['output_name'] = denoise.get('output_name')
        if model in {'NoRMCorre', 'PMD', 'Local'}:
            params['_model'] = model
        return params

    def _remap_image_layer_params(self, params: dict[str, Any], layer_id_map: dict[str, str]) -> dict[str, Any]:
        updated = dict(params)
        source = updated.get('input_source')
        if isinstance(source, str) and source.startswith('layer:'):
            old_id = source.split(':', 1)[1]
            updated['input_source'] = f"layer:{layer_id_map.get(old_id, old_id)}"
        return updated

    def _rebase_pipeline_output_name(
        self,
        params: dict[str, Any],
        model: str,
        payload: dict[str, Any],
        row_state: dict[str, Any],
    ) -> dict[str, Any]:
        if model not in {'NoRMCorre', 'PMD'}:
            return params
        updated = dict(params)
        default_name = self._image_layer_default_output_name(model)
        output_name = str(updated.get('output_name') or '').strip()
        denoise = row_state.get('denoise') if isinstance(row_state.get('denoise'), dict) else {}
        old_database = str(payload.get('database_name') or '').strip()
        old_default = self._default_image_layer_output_name_for_database(model, old_database)
        should_rebase = bool(denoise.get('default_output_name'))
        if not output_name:
            should_rebase = True
        else:
            output_path = Path(output_name)
            if not output_path.is_absolute() and old_default and output_path.name == old_default:
                should_rebase = True
        if should_rebase:
            updated['output_name'] = default_name
        return updated

    def _default_image_layer_output_name_for_database(self, model: str, database_name: str) -> str:
        if not database_name:
            return ''
        model_name = 'normcorre' if model == 'NoRMCorre' else model.lower()
        return f'{model_name}_{database_name}.tiff'

    def _remap_trace_source_data(self, source: Any, layer_id_map: dict[str, str]) -> Any:
        if not isinstance(source, str) or not source.startswith('image:'):
            return source
        parts = source.split(':', 2)
        if len(parts) != 3:
            return source
        return f'image:{layer_id_map.get(parts[1], parts[1])}:{parts[2]}'

    def _apply_trace_row_states(self, states: list[dict[str, Any]], layer_id_map: Optional[dict[str, str]] = None):
        states = list(states or [])
        if not states:
            return
        while len(self.trace_rows) < len(states):
            self.add_trace_row()
        while len(self.trace_rows) > len(states) and len(self.trace_rows) > 1:
            self.remove_trace_row(self.trace_rows[-1])
        self._refresh_trace_source_options()
        for row, row_state in zip(self.trace_rows, states):
            row.visible_checkbox.setChecked(bool(row_state.get('visible', True)))
            row.fold_button.setChecked(bool(row_state.get('folded', False)))
            self._set_trace_row_folded(row, row.fold_button.isChecked())
            source_data = self._remap_trace_source_data(row_state.get('source_data'), layer_id_map or {})
            self._set_combo_data(row.source_combo, source_data)
            row.lowpass_checkbox.setChecked(bool(row_state.get('lowpass', False)))
            row.lowpass_edit.setText(str(row_state.get('lowpass_hz', '1.0')))
            row.highpass_checkbox.setChecked(bool(row_state.get('highpass', False)))
            row.highpass_edit.setText(str(row_state.get('highpass_hz', '10.0')))
            row.wavelet_checkbox.setChecked(bool(row_state.get('wavelet', False)))
            row.wavelet_name_edit.setText(str(row_state.get('wavelet_name', 'sym4')))
            row.wavelet_level_edit.setText(str(row_state.get('wavelet_level', '4')))
            row.wavelet_scale_edit.setText(str(row_state.get('wavelet_scale', '1.2')))
            self._set_combo_text(row.wavelet_mode_combo, str(row_state.get('wavelet_mode', 'hard')))
            row.pca_wavelet_checkbox.setChecked(bool(row_state.get('pca_wavelet', False)))
            row.pca_wavelet_fmin_edit.setText(str(row_state.get('pca_wavelet_fmin', '1.0')))
            row.pca_wavelet_fmax_edit.setText(str(row_state.get('pca_wavelet_fmax', self._default_pca_wavelet_fmax_text())))
            row.pca_wavelet_fn_edit.setText(str(row_state.get('pca_wavelet_fn', '100')))
            row.pca_wavelet_cfg = dict(row_state.get('pca_wavelet_cfg') or row.pca_wavelet_cfg)
            row.snr_checkbox.setChecked(bool(row_state.get('snr', False)))
            row.snr_window_edit.setText(str(row_state.get('snr_window', '0.0125')))
            row.volpy_checkbox.setChecked(bool(row_state.get('volpy', False)))
            self._set_combo_text(row.volpy_combo, str(row_state.get('volpy_key', 'ts')))
            self._set_combo_data(row.baseline_mode_combo, row_state.get('baseline_mode', 'dff'))
            row.baseline_lowpass_checkbox.setChecked(bool(row_state.get('baseline_lowpass', False)))
            row.baseline_lowpass_edit.setText(str(row_state.get('baseline_lowpass_hz', '1.0')))
            row.baseline_rolling_checkbox.setChecked(bool(row_state.get('baseline_rolling', False)))
            self._set_combo_text(row.baseline_rolling_mode_combo, str(row_state.get('baseline_rolling_mode', 'mean')))
            row.baseline_rolling_window_edit.setText(str(row_state.get('baseline_rolling_window', '4')))
            row.baseline_polyfit_checkbox.setChecked(bool(row_state.get('baseline_polyfit', False)))
            row.baseline_poly_order_edit.setText(str(row_state.get('baseline_poly_order', '3')))
            row.baseline_savgol_checkbox.setChecked(bool(row_state.get('baseline_savgol', False)))
            row.baseline_savgol_window_edit.setText(str(row_state.get('baseline_savgol_window', '1.0')))
            row.baseline_savgol_order_edit.setText(str(row_state.get('baseline_savgol_order', '3')))
            row.spike_checkbox.setChecked(bool(row_state.get('spike', False)))
            self._set_combo_text(row.spike_method_combo, str(row_state.get('spike_method', 'std')))
            row.spike_k_edit.setText(str(row_state.get('spike_k', '5.0')))
            row.threshold_checkbox.setChecked(bool(row_state.get('threshold', False)))
            row.waveform_checkbox.setChecked(bool(row_state.get('waveform', False)))
            self._set_combo_text(row.waveform_mode_combo, str(row_state.get('waveform_mode', 'raw')))

    def _deserialize_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        event = dict(payload)
        event['start_frames'] = np.asarray(payload.get('start_frames', []), dtype=int)
        event['end_frames'] = np.asarray(payload.get('end_frames', []), dtype=int)
        event['trace_params'] = dict(payload.get('trace_params') or {})
        return event

    def _apply_average_state(self, state: dict[str, Any]):
        modes = set(state.get('modes', [])) or {'event'}
        self.avg_modes = {mode for mode in modes if mode in AVG_PANEL_MODES} or {'event'}
        for mode, checkbox in self._avg_radio.items():
            checkbox.setChecked(mode in self.avg_modes)
        self.avg_pre_edit.setText(str(state.get('avg_pre', self.avg_pre_edit.text())))
        self.avg_post_edit.setText(str(state.get('avg_post', self.avg_post_edit.text())))
        self.waveform_pre_edit.setText(str(state.get('waveform_pre', self.waveform_pre_edit.text())))
        self.waveform_post_edit.setText(str(state.get('waveform_post', self.waveform_post_edit.text())))
        if hasattr(self, 'avg_individual_button'):
            self.avg_individual_button.setChecked(bool(state.get('show_individual', self.avg_individual_button.isChecked())))
        self._pending_avg_selected_traces = set(int(v) for v in state.get('selected_traces', []))
        self._pending_avg_selected_events = set(int(v) for v in state.get('selected_events', []))

    def _restore_average_selections(self):
        if hasattr(self, 'avg_trace_list') and hasattr(self, '_pending_avg_selected_traces'):
            for idx in range(self.avg_trace_list.count()):
                item = self.avg_trace_list.item(idx)
                item.setSelected(int(item.data(Qt.ItemDataRole.UserRole)) in self._pending_avg_selected_traces)
        if hasattr(self, 'avg_event_list') and hasattr(self, '_pending_avg_selected_events'):
            for idx in range(self.avg_event_list.count()):
                item = self.avg_event_list.item(idx)
                item.setSelected(int(item.data(Qt.ItemDataRole.UserRole)) in self._pending_avg_selected_events)

    def _apply_roi_state(self, state: dict[str, Any]):
        if hasattr(self, 'only_cells_checkbox'):
            self.only_cells_checkbox.setChecked(bool(state.get('only_cells', self.only_cells_checkbox.isChecked())))
        if hasattr(self, 'show_masks_checkbox'):
            self.show_masks_checkbox.setChecked(bool(state.get('show_masks', self.show_masks_checkbox.isChecked())))
        self.combine_mode = str(state.get('combine_mode', self.combine_mode))
        for value, radio in self._combine_radio.items():
            radio.setChecked(value == self.combine_mode)
        if hasattr(self, 'mask_color_combo'):
            self._set_combo_data(self.mask_color_combo, state.get('mask_color', 'roi'))
        selected = [int(idx) for idx in state.get('selected_rois', []) if int(idx) >= 0]
        if selected:
            self.selected_roi_indices = selected

    def _apply_panel_state(self, state: dict[str, Any]):
        if hasattr(self, 'trace_window_edit'):
            self.trace_window_edit.setText(str(state.get('trace_window', self.trace_window_edit.text())))
        if hasattr(self, 'trace_x_unit_combo'):
            self._set_combo_data(self.trace_x_unit_combo, state.get('trace_x_unit', 'seconds'))
        if hasattr(self, 'image_labels_checkbox'):
            self.image_labels_checkbox.setChecked(bool(state.get('image_labels', self.image_labels_checkbox.isChecked())))
        if hasattr(self, 'trace_labels_checkbox'):
            self.trace_labels_checkbox.setChecked(bool(state.get('trace_labels', self.trace_labels_checkbox.isChecked())))
        if hasattr(self, 'avg_labels_checkbox'):
            self.avg_labels_checkbox.setChecked(bool(state.get('average_labels', self.avg_labels_checkbox.isChecked())))

    # ------------------------------ load state ------------------------------
    @staticmethod
    def _finite_max_value(arr: Any) -> Optional[float]:
        if arr is None:
            return None
        values = np.asarray(arr)
        if values.size == 0:
            return None
        if np.issubdtype(values.dtype, np.integer):
            return float(np.max(values))
        finite_max = float(np.nanmax(values))
        return finite_max if np.isfinite(finite_max) else None

    @staticmethod
    def _reverse_with_max(arr: Any, intensity_max: Optional[float]) -> Any:
        if arr is None or intensity_max is None:
            return arr
        return float(intensity_max) - np.asarray(arr, dtype=float)

    def _load_folder_state(self, folder: str, negative: bool = False) -> GUIState:
        tif_path = self._find_tif_path(folder)
        raw_movie = self._load_raw_movie(tif_path) if tif_path is not None else None
        raw_image = self._load_raw_image(folder, None, tif_path=tif_path)
        intensity_max = self._finite_max_value(raw_movie)
        if intensity_max is None:
            intensity_max = self._finite_max_value(raw_image)

        frame_rate, traces, cells, stim_cells, onset_times_trial, offset_times_trial, trial_duration, spike_times, firing_rate, thresholds = get_params(
            folder,
            negative=negative,
            intensity_max=intensity_max,
        )
        state = GUIState(
            source_type='folder',
            source_path=folder,
            frame_rate=float(frame_rate),
            traces=[np.asarray(x, dtype=float) for x in traces],
            trace_names=[f'get_params trace {idx + 1}' for idx in range(len(traces))],
            firing_rate=[np.asarray(x, dtype=float) for x in firing_rate],
            spike_times=spike_times,
            thresholds=thresholds,
            cells=np.asarray(cells, dtype=int),
            stim_cells=np.asarray(stim_cells, dtype=int),
            onset_times_trial=np.asarray(onset_times_trial, dtype=float) if onset_times_trial is not None else None,
            offset_times_trial=np.asarray(offset_times_trial, dtype=float) if offset_times_trial is not None else None,
            trial_duration=float(trial_duration) if trial_duration is not None else None,
            negative_mode=bool(negative),
            intensity_max=intensity_max,
        )

        plane_dir = Path(folder) / 'suite2p' / 'plane0'
        if (plane_dir / 'ops.npy').exists():
            state.ops = np.load(plane_dir / 'ops.npy', allow_pickle=True).item()
        if (plane_dir / 'stat.npy').exists():
            state.stat = np.load(plane_dir / 'stat.npy', allow_pickle=True)
        state.volpy_suite2p_indices = self._suite2p_cell_indices_for_state(state)

        state.tif_path = str(tif_path) if tif_path is not None else None
        state.image_info_path = str(self._image_info_path_for_state(state))
        state.raw_movie = raw_movie
        state.raw_image = raw_image
        if state.raw_image is None and state.ops is not None and 'meanImg' in state.ops:
            raw_image_from_ops = np.asarray(state.ops['meanImg'], dtype=float)
            state.raw_image = raw_image_from_ops
        if state.raw_image is not None and state.stat is not None:
            state.weight_alpha_maps = self._build_mask_alpha_maps(state.raw_image, state.stat)
            state.weight_alpha = state.weight_alpha_maps.get('weighted_pix_exp')
            state.suite2p_alpha = state.weight_alpha_maps.get('suite2p')
        return state

    def _load_table_state(self, table_path: str, negative: bool = False) -> GUIState:
        import pandas as pd

        suffix = Path(table_path).suffix.lower()
        if suffix != '.xlsx':
            raise ValueError('Femtonics table must be an .xlsx file.')
        df = pd.read_excel(table_path, engine='openpyxl')
        if df.shape[1] < 2:
            raise ValueError('Table must contain time column + at least one trace column.')

        time = df.iloc[:, 0].to_numpy(dtype=float)
        dt = float(np.nanmean(np.diff(time)))
        if dt <= 0:
            raise ValueError('Invalid time axis in table.')
        frame_rate = 1000.0 / dt if dt > 0.1 else 1.0 / dt

        traces = df.iloc[:, 1:].to_numpy(dtype=float).T
        trace_reverse_max = np.nanmax(traces, axis=1, keepdims=True)
        if negative:
            traces = trace_reverse_max - traces
        spike_times, firing_rate, thresholds, _snr_values = detect_spikes(traces, frame_rate, thr=5.0, mode='mad')
        cells = np.ones(traces.shape[0], dtype=int)
        stim_cells = np.zeros(traces.shape[0], dtype=int)

        state = GUIState(
            source_type='table',
            source_path=table_path,
            frame_rate=frame_rate,
            traces=[traces],
            trace_names=['table trace'],
            firing_rate=[firing_rate],
            spike_times=[spike_times],
            thresholds=[thresholds],
            cells=cells,
            stim_cells=stim_cells,
            onset_times_trial=None,
            offset_times_trial=None,
            trial_duration=None,
            raw_image=None,
            raw_movie=None,
            tif_path=None,
            weight_alpha=None,
            weight_alpha_maps={},
            suite2p_alpha=None,
            stat=None,
            ops=None,
            volpy_suite2p_indices=None,
            negative_mode=bool(negative),
            trace_reverse_max=trace_reverse_max,
        )
        state.image_info_path = str(self._image_info_path_for_state(state))
        return state

    def _load_volpy_folder_state(self, folder: str) -> GUIState:
        search_root = Path(folder)
        if not search_root.exists() or not search_root.is_dir():
            raise ValueError(f'VolPy folder was not found: {folder}')

        vpy_data: dict[str, Any] = {}
        candidates = self._find_volpy_files(search_root)
        vpy_path: Optional[Path] = None
        for candidate in candidates:
            loaded = np.load(candidate, allow_pickle=True)
            if hasattr(loaded, 'shape') and loaded.shape == ():
                loaded = loaded.item()
            loaded = self._normalize_volpy_payload(loaded)
            if isinstance(loaded, dict):
                vpy_data = loaded
                vpy_path = candidate
                break

        raw_movie = self._load_volpy_folder_movie(search_root, excluded_npy={vpy_path} if vpy_path is not None else set())
        raw_image = self._mean_image_from_movie(raw_movie)
        image_shape = tuple(int(v) for v in raw_image.shape[:2]) if raw_image is not None and raw_image.ndim >= 2 else None

        trace_mat, trace_key = self._first_volpy_trace_matrix(vpy_data)
        n_rows = trace_mat.shape[0] if trace_mat is not None else (self._volpy_row_count(vpy_data) or 0)
        frame_rate = self._infer_volpy_frame_rate(vpy_data, trace_mat)

        traces: list[np.ndarray] = []
        trace_names: list[str] = []
        spike_times: list[list[np.ndarray]] = []
        firing_rate: list[np.ndarray] = []
        thresholds: list[Optional[np.ndarray]] = []
        if trace_mat is not None and trace_mat.size:
            traces = [trace_mat]
            trace_names = [f'VolPy {trace_key}']
            spikes, rates, thres, _snr_values = detect_spikes(trace_mat, frame_rate, thr=5.0, mode='mad')
            spike_times = [spikes]
            firing_rate = [rates]
            thresholds = [thres]

        if raw_image is None and 'weights' in vpy_data:
            weights = self._coerce_roi_stack(np.asarray(vpy_data['weights'], dtype=float), image_shape=image_shape)
            if weights is not None:
                raw_image = np.nanmax(np.asarray(weights, dtype=float), axis=0)

        state = GUIState(
            source_type='volpy_folder',
            source_path=folder,
            frame_rate=frame_rate,
            traces=traces,
            trace_names=trace_names,
            firing_rate=firing_rate,
            spike_times=spike_times,
            thresholds=thresholds,
            cells=np.ones(n_rows, dtype=int) if n_rows else None,
            stim_cells=np.zeros(n_rows, dtype=int) if n_rows else None,
            raw_image=raw_image,
            raw_movie=raw_movie,
            vpy=vpy_data if vpy_data else None,
            volpy_suite2p_indices=np.arange(n_rows, dtype=int) if n_rows else None,
        )
        state.image_info_path = str(self._image_info_path_for_state(state))
        return state

    def _first_volpy_trace_matrix(self, vpy_data: dict[str, Any]) -> tuple[Optional[np.ndarray], str]:
        for key in TRACE_SOURCE_VOLPY_KEYS:
            if key not in vpy_data:
                continue
            mat = self._coerce_trace_matrix(vpy_data[key])
            if mat is not None and mat.size:
                return np.asarray(mat, dtype=float), key
        return None, ''

    def _infer_volpy_frame_rate(self, vpy_data: dict[str, Any], trace_mat: Optional[np.ndarray]) -> float:
        for key in ('fr', 'framerate', 'frame_rate', 'fs', 'sampling_rate'):
            value = vpy_data.get(key)
            if np.isscalar(value):
                rate = float(value)
                if np.isfinite(rate) and rate > 0:
                    return rate
        params = vpy_data.get('params')
        if isinstance(params, dict):
            for key in ('fr', 'framerate', 'frame_rate', 'fs', 'sampling_rate'):
                value = params.get(key)
                if np.isscalar(value):
                    rate = float(value)
                    if np.isfinite(rate) and rate > 0:
                        return rate
        n_frames = int(trace_mat.shape[1]) if trace_mat is not None and trace_mat.ndim == 2 else 0
        for key in ('time', 'times', 'timestamps', 't_sec'):
            if key not in vpy_data:
                continue
            time = np.asarray(vpy_data[key], dtype=float).reshape(-1)
            if n_frames and time.size != n_frames:
                continue
            dt = float(np.nanmedian(np.diff(time))) if time.size > 1 else 0.0
            if np.isfinite(dt) and dt > 0:
                return 1.0 / dt
        return 1.0

    def _load_volpy_folder_movie(self, search_root: Path, excluded_npy: set[Path]) -> Optional[np.ndarray]:
        mmap_path = self._find_volpy_mmap_file(search_root)
        if mmap_path is not None:
            movie = self._load_caiman_mmap_movie(mmap_path)
            if movie is not None:
                return movie

        tif_path = self._find_first_file(search_root, ['volpy*.tif', '*volpy*.tif', 'volpy*.tiff', '*volpy*.tiff'])
        if tif_path is not None:
            try:
                return np.asarray(tifffile.memmap(tif_path))
            except Exception:
                return np.asarray(tifffile.imread(tif_path))

        for pattern in ('*movie*.npy', '*mmap*.npy', '*volpy*.npy'):
            for path in sorted(search_root.rglob(pattern)):
                if path in excluded_npy:
                    continue
                loaded = np.load(path, allow_pickle=True)
                arr = np.asarray(loaded)
                if arr.ndim in (2, 3):
                    return arr
        return None

    @staticmethod
    def _mean_image_from_movie(movie: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if movie is None:
            return None
        arr = np.asarray(movie)
        if arr.ndim == 2:
            return np.asarray(arr, dtype=float)
        if arr.ndim >= 3:
            return np.asarray(np.nanmean(arr, axis=0), dtype=float)
        return None

    def _image_info_path_for_state(self, state: GUIState) -> Path:
        if state.tif_path:
            return Path(state.tif_path).parent / 'image_info.npy'
        if state.source_path:
            path = Path(state.source_path)
            base_dir = path if path.is_dir() else path.parent
            return base_dir / 'image_info.npy'
        return Path.cwd() / 'image_info.npy'

    def _load_image_info_from_disk(self):
        self.state.image_user_info = {}
        if not self.state.image_info_path:
            return
        path = Path(self.state.image_info_path)
        if not path.exists():
            return
        data = np.load(path, allow_pickle=True).item()
        if isinstance(data, dict):
            user_info = data.get('user_info') if isinstance(data.get('user_info'), dict) else data
            self.state.image_user_info = {str(key): str(value) for key, value in user_info.items()}

    def _save_image_info_to_disk(self):
        if not self.state.image_info_path:
            return
        path = Path(self.state.image_info_path)
        if not path.parent.exists():
            return
        np.save(path, self._image_info_payload(), allow_pickle=True)

    def _detected_image_info(self) -> dict[str, str]:
        image_size = ''
        frame_count = '0'
        if self.state.raw_movie is not None:
            arr = np.asarray(self.state.raw_movie)
            frame_count = str(arr.shape[0]) if arr.ndim >= 3 else '1'
            if arr.ndim >= 3:
                image_size = f'{arr.shape[-1]} x {arr.shape[-2]}'
        elif self.state.raw_image is not None:
            arr = np.asarray(self.state.raw_image)
            frame_count = '1'
            image_size = f'{arr.shape[1]} x {arr.shape[0]}' if arr.ndim >= 2 else ''
        elif self.state.traces:
            frame_count = str(self.state.traces[0].shape[1])
        return {
            'Frame number': frame_count,
            'Frame rate': f'{self.state.frame_rate:g} Hz' if self.state.frame_rate else '',
            'Image size': image_size,
        }

    def _update_image_info_table(self):
        return

    def _on_image_info_changed(self, item: QTableWidgetItem):
        if item.column() != 1:
            return
        table = item.tableWidget()
        if table is None:
            return
        field_item = table.item(item.row(), 0)
        if field_item is None:
            return
        field = field_item.text()
        if field not in IMAGE_INFO_USER_FIELDS:
            return

        value = item.text()
        if value:
            self.state.image_user_info[field] = value
        else:
            self.state.image_user_info.pop(field, None)
        self._save_image_info_to_disk()

    def show_image_info_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Image information')
        layout = QVBoxLayout(dialog)
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(['Field', 'Value'])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        rows = list(self._source_info_rows())
        table.setRowCount(len(rows))
        for idx, (field, value) in enumerate(rows):
            key_item = QTableWidgetItem(str(field))
            value_item = QTableWidgetItem(str(value))
            key_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            value_flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            if field in IMAGE_INFO_USER_FIELDS:
                value_flags |= Qt.ItemFlag.ItemIsEditable
            value_item.setFlags(value_flags)
            table.setItem(idx, 0, key_item)
            table.setItem(idx, 1, value_item)
        table.resizeRowsToContents()
        layout.addWidget(table)

        button_row = QHBoxLayout()
        save_btn = QPushButton('Save')
        close_btn = QPushButton('Close')
        button_row.addStretch(1)
        button_row.addWidget(save_btn)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        def save_info():
            user_info: dict[str, str] = {}
            for idx in range(table.rowCount()):
                field_item = table.item(idx, 0)
                value_item = table.item(idx, 1)
                if field_item is None or value_item is None:
                    continue
                field = field_item.text()
                if field not in IMAGE_INFO_USER_FIELDS:
                    continue
                value = value_item.text()
                if value:
                    user_info[field] = value
            self.state.image_user_info = user_info
            self._save_image_info_to_disk()
            self._set_status(f'Saved image information: {self.state.image_info_path}')
            dialog.accept()

        save_btn.clicked.connect(save_info)
        close_btn.clicked.connect(dialog.reject)
        dialog.resize(720, 420)
        dialog.exec()

    def show_pca_wavelet_param_dialog(self, row: TraceControlRow):
        dialog = QDialog(self)
        dialog.setWindowTitle(f'{row.widget.title()} pca_wavelet parameters')
        layout = QVBoxLayout(dialog)

        current_cfg = self._pca_wavelet_cfg_from_row(row)
        cfg_fields = list(fields(wavelet_pca_plot.cw.wavelet_cfg()))
        tabs = QTabWidget()
        tabs.setToolTip('Result figures use the first selected ROI.')
        param_tab = QWidget()
        param_layout = QVBoxLayout(param_tab)
        table = QTableWidget(len(cfg_fields), 2)
        table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        param_widgets: dict[str, QWidget] = {}
        for idx, cfg_field in enumerate(cfg_fields):
            key_item = QTableWidgetItem(cfg_field.name)
            key_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            value = current_cfg.get(cfg_field.name)
            table.setItem(idx, 0, key_item)
            if cfg_field.name == 'event_pca':
                mode_combo = NoWheelComboBox()
                for label, data in (('None', 'none'), ('pca', 'pca'), ('svd', 'svd')):
                    mode_combo.addItem(label, data)
                mode_combo.setCurrentIndex(
                    self._combo_index_for_data(
                        mode_combo,
                        wavelet_pca_plot.cw.normalize_event_denoise_mode(value),
                    )
                )
                table.setCellWidget(idx, 1, mode_combo)
                param_widgets[cfg_field.name] = mode_combo
            else:
                value_item = QTableWidgetItem('' if value is None else str(value))
                value_item.setFlags(
                    Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsEditable
                )
                table.setItem(idx, 1, value_item)
        table.resizeRowsToContents()
        param_layout.addWidget(table)
        tabs.addTab(param_tab, 'Parameters')
        layout.addWidget(tabs, stretch=1)

        preview_figures: list[Figure] = []

        def clear_preview_tabs():
            while tabs.count() > 1:
                tab = tabs.widget(1)
                tabs.removeTab(1)
                tab.deleteLater()
            while preview_figures:
                plt.close(preview_figures.pop())

        def add_preview_message(message: str):
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            label = QLabel(message)
            label.setWordWrap(True)
            tab_layout.addWidget(label)
            tabs.addTab(tab, 'Result figures')

        def refresh_preview_tabs():
            clear_preview_tabs()
            try:
                figures, roi_idx = self._build_pca_wavelet_preview_figures(row, current_cfg)
            except Exception as exc:
                add_preview_message(str(exc))
                return
            if not figures:
                add_preview_message('No pca_wavelet result figures are available.')
                return
            roi_text = self._roi_label_text(roi_idx)
            for title, fig in figures:
                tab = QWidget()
                tab.setToolTip(f'Result figure for the first selected ROI: {roi_text}')
                tab_layout = QVBoxLayout(tab)
                tab_layout.setContentsMargins(10, 10, 10, 10)
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setContentsMargins(10, 10, 10, 10)
                canvas = PlotCanvas(fig, container)
                width = int(max(760, min(1040, fig.get_figwidth() * fig.dpi)))
                height = int(max(420, min(680, fig.get_figheight() * fig.dpi)))
                canvas.setMinimumSize(width, height)
                container_layout.addWidget(canvas)
                scroll.setWidget(container)
                tab_layout.addWidget(scroll)
                tabs.addTab(tab, title)
                preview_figures.append(fig)
                canvas.draw_idle()

        button_row = QHBoxLayout()
        apply_btn = QPushButton('Apply')
        close_btn = QPushButton('Close')
        button_row.addStretch(1)
        button_row.addWidget(apply_btn)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        def apply_params():
            nonlocal current_cfg
            new_cfg: dict[str, Any] = {}
            try:
                for idx, cfg_field in enumerate(cfg_fields):
                    widget = param_widgets.get(cfg_field.name)
                    if isinstance(widget, QComboBox):
                        new_cfg[cfg_field.name] = widget.currentData()
                    else:
                        value_item = table.item(idx, 1)
                        text = value_item.text() if value_item is not None else ''
                        new_cfg[cfg_field.name] = self._parse_pca_wavelet_cfg_value(
                            text,
                            current_cfg.get(cfg_field.name),
                        )
                row.pca_wavelet_cfg = new_cfg
                current_cfg = dict(new_cfg)
                self._sync_pca_wavelet_edits_from_cfg(row)
                self._invalidate_trace_cache()
                self.render_all()
                refresh_preview_tabs()
            except Exception as exc:
                QMessageBox.critical(dialog, 'Invalid pca_wavelet parameter', str(exc))
                return

        apply_btn.clicked.connect(apply_params)
        close_btn.clicked.connect(dialog.reject)
        refresh_preview_tabs()
        dialog.resize(1100, 760)
        dialog.exec()
        clear_preview_tabs()

    def _build_pca_wavelet_preview_figures(self, row: TraceControlRow, cfg_dict: dict[str, Any]) -> tuple[list[tuple[str, Figure]], int]:
        source = self._get_trace_source_matrix(row)
        if source is None:
            raise ValueError('No trace source is available for pca_wavelet result figures.')
        trace_mat, _source_name, roi_indices = source
        trace_mat = np.asarray(trace_mat, dtype=float).copy()
        if trace_mat.size == 0:
            raise ValueError('The selected trace source is empty.')
        trace_mat, _frame_indices = self._apply_event_trace_actions(trace_mat)
        if trace_mat.shape[1] == 0:
            raise ValueError('All frames were discarded before pca_wavelet preview.')

        selected_rois = self.get_selected_roi_indices()
        if not selected_rois:
            if roi_indices is not None and len(np.asarray(roi_indices).reshape(-1)):
                selected_rois = [int(np.asarray(roi_indices, dtype=int).reshape(-1)[0])]
            else:
                selected_rois = [0]
        pairs = self._result_roi_row_pairs({'data': trace_mat, 'roi_indices': roi_indices}, selected_rois)
        if pairs:
            roi_idx, row_idx = pairs[0]
        elif roi_indices is not None and len(np.asarray(roi_indices).reshape(-1)) and trace_mat.shape[0]:
            roi_idx = int(np.asarray(roi_indices, dtype=int).reshape(-1)[0])
            row_idx = 0
        elif trace_mat.shape[0]:
            roi_idx, row_idx = 0, 0
        else:
            raise ValueError('The selected trace source has no ROI rows.')
        trace = np.asarray(trace_mat[row_idx], dtype=float)
        if not np.any(np.isfinite(trace)):
            raise ValueError('The first selected ROI has no finite trace values.')

        framerate = max(float(self.state.frame_rate), 1e-8)
        cfg_values = dict(cfg_dict)
        processed_input = self._pca_wavelet_internal_input(trace_mat, framerate, cfg_values)
        trace_output, denoise_info = pca_wavelet_trace(
            trace_mat,
            framerate,
            cfg=cfg_values,
        )
        trace_output = np.asarray(trace_output, dtype=float)
        if row_idx >= processed_input.shape[0] or not np.any(np.isfinite(processed_input[row_idx])):
            raise ValueError('The first selected ROI has no finite pca_wavelet input after preprocessing.')
        details = {
            'source_input': trace_mat,
            'input': processed_input,
            'output': trace_output,
            'denoise_info': denoise_info,
            'framerate': framerate,
            'cfg': cfg_values,
            'f_min': float(cfg_values.get('f_min') or 1.0),
            'f_max': float(cfg_values.get('f_max') or max(framerate, 1.0)),
            'f_n': int(cfg_values.get('f_n') or 100),
        }
        return self._build_pca_wavelet_detail_figures(processed_input[row_idx], details, roi_idx, row_idx=row_idx), int(roi_idx)

    def _pca_wavelet_internal_input(self, trace_mat: np.ndarray, framerate: float, cfg_dict: dict[str, Any]) -> np.ndarray:
        cfg = wavelet_pca_plot.cw.wavelet_cfg(**dict(cfg_dict))
        arr = np.asarray(trace_mat, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if bool(getattr(cfg, 'dff', True)):
            return np.asarray(arr, dtype=float)
        return np.asarray(
            wavelet_pca_plot.cw.rolling_base_trace(
                arr,
                max(float(framerate), 1e-8),
                window=cfg.base_window,
                mode=cfg.base_mode,
            ),
            dtype=float,
        )

    def _default_pca_wavelet_cfg_dict(self, f_min: float = 1.0, f_max: float = 500.0, f_n: int = 100) -> dict[str, Any]:
        cfg = wavelet_pca_plot.cw.wavelet_cfg(f_min=float(f_min), f_max=float(f_max), f_n=int(f_n))
        return {cfg_field.name: getattr(cfg, cfg_field.name) for cfg_field in fields(cfg)}

    def _pca_wavelet_cfg_from_row(self, row: TraceControlRow) -> dict[str, Any]:
        cfg = dict(row.pca_wavelet_cfg) if row.pca_wavelet_cfg else self._default_pca_wavelet_cfg_dict()
        fr = max(self.state.frame_rate, 1.0)
        cfg['f_min'] = self._safe_float(row.pca_wavelet_fmin_edit.text(), default=float(cfg.get('f_min') or 1.0))
        cfg['f_max'] = self._safe_float(row.pca_wavelet_fmax_edit.text(), default=float(cfg.get('f_max') or fr))
        cfg['f_n'] = int(self._safe_float(row.pca_wavelet_fn_edit.text(), default=float(cfg.get('f_n') or 100)))
        cfg['event_pca'] = wavelet_pca_plot.cw.normalize_event_denoise_mode(cfg.get('event_pca', 'svd'))
        return cfg

    def _sync_pca_wavelet_edits_from_cfg(self, row: TraceControlRow):
        cfg = row.pca_wavelet_cfg
        for edit, key in (
            (row.pca_wavelet_fmin_edit, 'f_min'),
            (row.pca_wavelet_fmax_edit, 'f_max'),
            (row.pca_wavelet_fn_edit, 'f_n'),
        ):
            edit.blockSignals(True)
            edit.setText(str(cfg.get(key, edit.text())))
            edit.blockSignals(False)
        row.pca_wavelet_fmax_edit.setProperty('autoFrameRate', False)

    @staticmethod
    def _parse_pca_wavelet_cfg_value(text: str, current: Any) -> Any:
        value = text.strip()
        if value.lower() in {'', 'none', 'null'}:
            return None
        if isinstance(current, bool):
            if value.lower() in {'true', '1', 'yes', 'y'}:
                return True
            if value.lower() in {'false', '0', 'no', 'n'}:
                return False
            raise ValueError(f'Expected a boolean value, got {text!r}.')
        if isinstance(current, int) and not isinstance(current, bool):
            return int(float(value))
        if isinstance(current, float):
            return float(value)
        if current is None:
            lower = value.lower()
            if lower in {'true', 'false'}:
                return lower == 'true'
            try:
                numeric = float(value)
            except ValueError:
                return value
            return int(numeric) if numeric.is_integer() else numeric
        return value

    def show_trace_result_dialog(self, row: TraceControlRow):
        if not row.pca_wavelet_checkbox.isChecked():
            self._show_trace_detail_message(row, 'The detail view is available for pca_wavelet traces.')
            return

        try:
            result = self._build_trace_result(row)
        except Exception as exc:
            QMessageBox.critical(self, 'Trace details failed', str(exc))
            return
        if result is None:
            self._show_trace_detail_message(row, 'No trace result is available for this row.')
            return

        details = result.get('details', {}).get('pca_wavelet')
        if not details:
            self._show_trace_detail_message(row, 'No pca_wavelet intermediate trace was captured.')
            return

        selected_rois = self.get_selected_roi_indices()
        roi_row_pairs = self._result_roi_row_pairs(result, selected_rois)
        if not roi_row_pairs:
            self._show_trace_detail_message(row, 'Selected ROI is outside this trace source.')
            return
        roi_idx, row_idx = roi_row_pairs[0]

        trace_input = np.asarray(details['input'], dtype=float)
        if row_idx >= trace_input.shape[0] or not np.any(np.isfinite(trace_input[row_idx])):
            self._show_trace_detail_message(row, 'Selected ROI has no finite pca_wavelet input trace.')
            return

        try:
            figures = self._build_pca_wavelet_detail_figures(trace_input[row_idx], details, roi_idx, row_idx=row_idx)
        except Exception as exc:
            QMessageBox.critical(self, 'pca_wavelet details failed', str(exc))
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f'{row.widget.title()} pca_wavelet details - {self._roi_label_text(roi_idx)}')
        layout = QVBoxLayout(dialog)
        tabs = QTabWidget()
        for title, fig in figures:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.setContentsMargins(10, 10, 10, 10)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(10, 10, 10, 10)
            canvas = PlotCanvas(fig, container)
            width = int(max(760, min(1040, fig.get_figwidth() * fig.dpi)))
            height = int(max(420, min(680, fig.get_figheight() * fig.dpi)))
            canvas.setMinimumSize(width, height)
            container_layout.addWidget(canvas)
            scroll.setWidget(container)
            tab_layout.addWidget(scroll)
            tabs.addTab(tab, title)
            canvas.draw_idle()
        layout.addWidget(tabs, stretch=1)
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        dialog.resize(1100, 760)
        dialog.exec()
        for _title, fig in figures:
            plt.close(fig)

    def _show_trace_detail_message(self, row: TraceControlRow, message: str):
        dialog = QDialog(self)
        dialog.setWindowTitle(f'{row.widget.title()} details')
        layout = QVBoxLayout(dialog)
        label = QLabel(message)
        label.setWordWrap(True)
        layout.addWidget(label)
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        dialog.resize(420, 160)
        dialog.exec()

    def _build_pca_wavelet_detail_figures(
        self,
        trace: np.ndarray,
        details: dict[str, Any],
        roi_idx: int,
        row_idx: Optional[int] = None,
    ) -> list[tuple[str, Figure]]:
        framerate = float(details['framerate'])
        trace = np.asarray(trace, dtype=float)
        trace = trace - np.nanmean(trace)
        trace = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)
        cfg_values = dict(details.get('cfg') or {})
        if not cfg_values:
            cfg_values = {
                'f_min': float(details['f_min']),
                'f_max': float(details['f_max']),
                'f_n': int(details['f_n']),
            }
        cfg = wavelet_pca_plot.cw.wavelet_cfg(**cfg_values)
        coeffs, coeffs_norm, freqs, scales = wavelet_pca_plot.cw.morlet_cwt(trace, framerate, cfg)
        freq_features, pca, opt_k = wavelet_pca_plot.cw.pca_feature(
            coeffs_norm,
            explained_variance=cfg.pca_explained_variance,
        )
        all_domains, _labels, _best_k, _domain_source = wavelet_pca_plot.get_all_domains_for_plot(
            freq_features,
            opt_k,
            cfg.ward_n_clusters,
        )
        domain_results = [
            wavelet_pca_plot.reconstruct_domain_for_plot(domain, coeffs, freqs, scales, framerate, cfg)
            for domain in all_domains
        ]
        denoised_trace = None
        output = details.get('output')
        if row_idx is not None and output is not None:
            output_arr = np.asarray(output, dtype=float)
            if output_arr.ndim == 2 and row_idx < output_arr.shape[0] and output_arr.shape[1] == trace.shape[0]:
                denoised_trace = output_arr[row_idx]
        if denoised_trace is None:
            denoised_trace = wavelet_pca_plot.make_denoised_trace_for_plot(trace, domain_results)
        window_sec = self._pca_wavelet_detail_window_sec(framerate)
        figures = [
            (
                'Spectrogram',
                wavelet_pca_plot.plot_spectrogram(
                    coeffs_norm,
                    freqs,
                    framerate,
                    title=f'CWT spectrogram: ROI {roi_idx:03d}',
                    window_sec=window_sec,
                ),
            ),
            ('First PCA', wavelet_pca_plot.plot_first_pca(freq_features, freqs, opt_k, pca=pca)),
            (
                'Domain traces',
                wavelet_pca_plot.plot_domain_traces(
                    domain_results,
                    framerate,
                    window_sec,
                    original_trace=trace,
                    denoised_trace=denoised_trace,
                ),
            ),
        ]
        if wavelet_pca_plot.cw.event_denoise_mode(cfg) != 'none':
            figures.extend(
                [
                    ('Event waveforms', wavelet_pca_plot.plot_event_waveforms(domain_results)),
                    ('Event PC1 histograms', wavelet_pca_plot.plot_event_pc1_histograms(domain_results)),
                ]
            )
        return figures

    def _pca_wavelet_detail_window_sec(self, framerate: float) -> float:
        value = self._safe_float(self.trace_window_edit.text(), default=5.0)
        if self._trace_x_unit() == 'frames':
            return max(value / max(framerate, 1.0), 0.1)
        return max(value, 0.1)

    def _image_info_payload(self) -> dict[str, Any]:
        return {
            'source': dict(self._source_info_rows()),
            'user_info': dict(self.state.image_user_info),
        }

    def _source_info_rows(self):
        detected = self._detected_image_info()
        rows = [
            ('Source path', self.state.source_path),
            ('Source type', self.state.source_type or ''),
            ('Frame number', detected.get('Frame number', '')),
            ('Image size', detected.get('Image size', '')),
            ('Frame rate', detected.get('Frame rate', '')),
            ('Metadata file', self.state.image_info_path or ''),
        ]
        rows.extend((field, self.state.image_user_info.get(field, '')) for field in IMAGE_INFO_USER_FIELDS)
        return rows

    def _image_layer_info_rows(self, row: ImageLayerControlRow):
        image, nframes, title = self._get_image_layer_display(row)
        arr = np.asarray(image) if image is not None else None
        size = f'{arr.shape[-1]} x {arr.shape[-2]}' if arr is not None and arr.ndim >= 2 else ''
        return [
            ('Layer', self._image_layer_label(row)),
            ('Layer source', title),
            ('Layer frames', nframes),
            ('Layer size', size),
            ('Layer frame rate', f'{self.state.frame_rate:g} Hz' if self.state.frame_rate else ''),
            ('Layer mask source', row.mask_source_combo.currentText()),
        ]

    def _find_tif_path(self, folder: str) -> Optional[Path]:
        path = Path(folder)
        basename = path.name
        tif_guess = path / f'{basename}_Cycle00001_Ch2_000001.ome.tif'
        if tif_guess.exists():
            return tif_guess
        tif_files = sorted(path.glob('*.tif')) + sorted(path.glob('*.tiff'))
        return tif_files[0] if tif_files else None

    def _load_raw_movie(self, tif_path: Optional[Path]) -> Optional[np.ndarray]:
        if tif_path is None:
            return None
        try:
            arr = tifffile.memmap(tif_path)
        except Exception:
            try:
                arr = tifffile.imread(tif_path)
            except Exception:
                return None
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return None
        if arr.ndim == 3:
            return arr
        if arr.ndim == 4:
            if arr.shape[-1] in (3, 4):
                return arr[..., :3].mean(axis=-1)
            if arr.shape[1] in (3, 4):
                return arr[:, :3].mean(axis=1)
        return None

    def _load_raw_image(self, folder: str, ops: Optional[dict], tif_path: Optional[Path] = None) -> Optional[np.ndarray]:
        if tif_path is not None and tif_path.exists():
            return np.asarray(load_mean_image_from_tif(tif_path), dtype=float)
        if ops is not None and 'meanImg' in ops:
            return np.asarray(ops['meanImg'], dtype=float)
        return None

    def _build_mask_alpha_maps(self, raw_image: np.ndarray, stat: np.ndarray):
        raw_image = np.asarray(raw_image, dtype=float)
        ly, lx = raw_image.shape[:2]
        maps = {
            'weighted_pix_exp': np.zeros((ly, lx), dtype=float),
            'weighted_pix_overmean': np.zeros((ly, lx), dtype=float),
            'weighted_pix_max': np.zeros((ly, lx), dtype=float),
            'suite2p': np.zeros((ly, lx), dtype=float),
        }
        for roi in stat:
            ypix = np.asarray(roi['ypix'], dtype=int)
            xpix = np.asarray(roi['xpix'], dtype=int)
            valid = (ypix >= 0) & (ypix < ly) & (xpix >= 0) & (xpix < lx)
            ypix = ypix[valid]
            xpix = xpix[valid]
            if ypix.size == 0:
                continue
            roi_values = raw_image[ypix, xpix]
            weights_by_mode = {
                'weighted_pix_exp': pix_exp(roi_values),
                'weighted_pix_overmean': pix_overmean(roi_values),
                'weighted_pix_max': pix_max(roi_values),
            }
            lam = np.asarray(roi.get('lam', np.ones_like(ypix, dtype=float)), dtype=float)
            lam = lam[:len(ypix)] if lam.size >= len(ypix) else np.pad(lam, (0, len(ypix) - lam.size), constant_values=1.0)
            lam = lam / (np.max(lam) + 1e-8)
            for name, roi_weight in weights_by_mode.items():
                roi_weight = roi_weight / (np.max(roi_weight) + 1e-8)
                maps[name][ypix, xpix] = np.maximum(maps[name][ypix, xpix], roi_weight)
            maps['suite2p'][ypix, xpix] = np.maximum(maps['suite2p'][ypix, xpix], lam)
        return {name: np.clip(alpha, 0.0, 1.0) for name, alpha in maps.items()}

    def ensure_volpy_loaded(self, show_warning: bool = True) -> Optional[dict]:
        if self.vpy is not None:
            self._ensure_volpy_suite2p_mapping(self.vpy)
            globals()['vpy'] = self.vpy
            self.state.vpy = self.vpy
            return self.vpy

        if self.state.vpy is not None:
            self.vpy = self.state.vpy
            self._ensure_volpy_suite2p_mapping(self.vpy)
            globals()['vpy'] = self.vpy
            return self.vpy

        if not self.state.source_path:
            if show_warning:
                self._set_status('VolPy data requested, but no source folder is loaded.')
            return None

        source_path = Path(self.state.source_path)
        search_root = source_path if source_path.is_dir() else source_path.parent
        candidates = self._find_volpy_files(search_root)
        if not candidates:
            if show_warning:
                self._set_status('VolPy data requested, but no volpy*.npy file was found.')
            return None

        loaded = None
        loaded_path = None
        for candidate in candidates:
            candidate_data = np.load(candidate, allow_pickle=True)
            if hasattr(candidate_data, 'shape') and candidate_data.shape == ():
                candidate_data = candidate_data.item()
            candidate_data = self._normalize_volpy_payload(candidate_data)
            if isinstance(candidate_data, dict):
                loaded = candidate_data
                loaded_path = candidate
                break
        if not isinstance(loaded, dict):
            if show_warning:
                self._set_status('No VolPy npy dictionary was found.')
            return None
        self.vpy = loaded
        self.state.vpy = loaded
        globals()['vpy'] = loaded
        self._ensure_volpy_suite2p_mapping(loaded)
        self._set_status(f'Loaded VolPy data: {loaded_path}')
        return loaded

    def _current_volpy_payload(self) -> Optional[dict]:
        if isinstance(self.vpy, dict):
            return self.vpy
        if isinstance(self.state.vpy, dict):
            return self.state.vpy
        return None

    def _find_volpy_files(self, search_root: Path) -> list[Path]:
        patterns = ['volpy*.npy', '*volpy*.npy', 'vpy*.npy', '*vpy*.npy']
        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend(sorted(search_root.glob(pattern)))
        if candidates:
            return list(dict.fromkeys(candidates))
        for pattern in patterns:
            candidates.extend(sorted(search_root.rglob(pattern)))
        return list(dict.fromkeys(candidates))

    def _coerce_roi_stack(self, arr: np.ndarray, image_shape: Optional[tuple[int, int]] = None) -> Optional[np.ndarray]:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return arr.reshape(1, arr.shape[0], arr.shape[1])
        if arr.ndim != 3:
            return None
        image_shape = image_shape or self._raw_image_shape()
        if image_shape is not None:
            if arr.shape[1:3] == image_shape:
                return arr
            if arr.shape[:2] == image_shape:
                return np.moveaxis(arr, -1, 0)
        if arr.shape[-1] < arr.shape[0] and arr.shape[-1] < arr.shape[1]:
            return np.moveaxis(arr, -1, 0)
        return arr

    def _raw_image_shape(self) -> Optional[tuple[int, int]]:
        if self.state.raw_image is not None:
            arr = np.asarray(self.state.raw_image)
            if arr.ndim >= 2:
                return int(arr.shape[-2]), int(arr.shape[-1])
        if self.state.raw_movie is not None:
            arr = np.asarray(self.state.raw_movie)
            if arr.ndim >= 3:
                return int(arr.shape[-2]), int(arr.shape[-1])
        if self.state.tif_path and Path(self.state.tif_path).exists():
            arr = tifffile.memmap(self.state.tif_path)
            if arr.ndim >= 2:
                return int(arr.shape[-2]), int(arr.shape[-1])
        return None

    def _normalize_volpy_payload(self, loaded: Any) -> Any:
        if isinstance(loaded, dict):
            return loaded
        if hasattr(loaded, 'estimates'):
            return dict(getattr(loaded, 'estimates'))
        return loaded

    def _suite2p_cell_indices_for_state(self, state: GUIState) -> Optional[np.ndarray]:
        _all_ids, cell_ids = self._suite2p_roi_index_sets_for_state(state)
        return cell_ids.astype(int) if cell_ids.size else None

    def _suite2p_roi_index_sets_for_state(self, state: GUIState) -> tuple[Optional[np.ndarray], np.ndarray]:
        cells = None
        if state.cells is not None:
            cells = np.asarray(state.cells, dtype=int).reshape(-1)
        elif state.source_path:
            iscell_path = Path(state.source_path) / 'suite2p' / 'plane0' / 'iscell.npy'
            if iscell_path.exists():
                iscell = np.asarray(np.load(iscell_path, allow_pickle=True))
                if iscell.ndim >= 2:
                    cells = np.asarray(iscell[:, 0], dtype=int).reshape(-1)
        if cells is None:
            if state.stat is not None:
                return np.arange(len(state.stat), dtype=int), np.asarray([], dtype=int)
            return None, np.asarray([], dtype=int)
        all_ids = np.arange(cells.shape[0], dtype=int)
        cell_ids = np.where(cells == 1)[0].astype(int)
        return all_ids, cell_ids

    def _volpy_row_count(self, vpy_data: Optional[dict]) -> Optional[int]:
        if not vpy_data:
            return None
        if 'weights' in vpy_data:
            masks = self._coerce_roi_stack(np.asarray(vpy_data['weights']), image_shape=self._raw_image_shape())
            if masks is not None and masks.ndim == 3 and masks.shape[0] > 0:
                return int(masks.shape[0])
        for key in ['weights', *TRACE_SOURCE_VOLPY_KEYS]:
            if key not in vpy_data:
                continue
            arr = np.asarray(vpy_data[key])
            if arr.ndim >= 1 and arr.shape[0] > 0:
                return int(arr.shape[0])
        return None

    def _ensure_volpy_suite2p_mapping(self, vpy_data: Optional[dict], row_count: Optional[int] = None) -> Optional[np.ndarray]:
        if self._ensuring_volpy_mapping:
            return self.state.volpy_suite2p_indices
        n_rows = int(row_count) if row_count is not None else self._volpy_row_count(vpy_data)
        if n_rows is None or n_rows <= 0:
            return self.state.volpy_suite2p_indices

        self._ensuring_volpy_mapping = True
        try:
            all_ids, cell_ids = self._suite2p_roi_index_sets_for_state(self.state)
            if len(cell_ids) == n_rows:
                self.state.volpy_suite2p_indices = np.asarray(cell_ids, dtype=int)
                return self.state.volpy_suite2p_indices
            if all_ids is not None and len(all_ids) == n_rows:
                self.state.volpy_suite2p_indices = np.asarray(all_ids, dtype=int)
                return self.state.volpy_suite2p_indices
        finally:
            self._ensuring_volpy_mapping = False

        self.state.volpy_suite2p_indices = None
        if all_ids is None:
            self._set_status(f'Failed to align VolPy and Suite2p ROIs: VolPy has {n_rows} ROI(s), but Suite2p ROI metadata is unavailable.')
        else:
            self._set_status(
                f'Failed to align VolPy and Suite2p ROIs: VolPy has {n_rows} ROI(s), '
                f'Suite2p has {len(all_ids)} total ROI(s) and {len(cell_ids)} cell ROI(s).'
            )
        return None

    def _volpy_row_for_suite2p_roi(self, roi_idx: int) -> Optional[int]:
        mapping = self.state.volpy_suite2p_indices
        if mapping is None:
            return None
        hits = np.where(np.asarray(mapping, dtype=int) == int(roi_idx))[0]
        return int(hits[0]) if hits.size else None

    # ------------------------------ render helpers ------------------------------
    def render_all(self, reset_trace_view: bool = False):
        self._render_reset_trace_view = self._render_reset_trace_view or reset_trace_view
        self._render_timer.start(90)

    def _render_all_now(self):
        reset_trace_view = self._render_reset_trace_view
        self._render_reset_trace_view = False
        self._render_image_panel()
        self._render_trace_panel(reset_view=reset_trace_view)
        self._render_average_panel(reset_view=reset_trace_view)

    def _apply_axis_labels(self, ax, show: bool, xlabel: str = '', ylabel: str = '', title: str = ''):
        ax.tick_params(labelbottom=True, labelleft=True)
        if show:
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            ax.set_title('')
            ax.set_xlabel('')
            ax.set_ylabel('')

    def _render_image_panel(self):
        if not hasattr(self, 'image_widget'):
            return
        fig = self.image_widget.figure
        fig.clear()
        self._image_axis_mask_sources.clear()
        visible_layers = [row for row in self.image_layers if row.visible_checkbox.isChecked()]
        if not visible_layers:
            self._sync_image_frame_slider(0)
            self.image_widget.set_placeholder('No visible image layers.')
            return

        layer_payloads = []
        video_frames = 0
        for row in visible_layers:
            image, nframes, title = self._get_image_layer_display(row)
            if row.mode_combo.currentData() == 'video' and nframes > 1:
                video_frames = max(video_frames, nframes)
            row.nframes_label.setText(f'{nframes} frame{"s" if nframes != 1 else ""}')
            self._update_image_layer_summary(row, image, nframes)
            layer_payloads.append((row, image, title))

        self._sync_image_frame_slider(video_frames)
        axes = fig.subplots(len(layer_payloads), 1, squeeze=False)
        axes = [axrow[0] for axrow in axes]
        active_roi = self.get_active_roi_index()

        for ax, (row, image, title) in zip(axes, layer_payloads):
            mask_source = row.mask_source_combo.currentData()
            self._image_axis_mask_sources[ax] = mask_source
            if image is None:
                ax.text(0.5, 0.5, self._image_layer_missing_text(row), ha='center', va='center', wrap=True)
                ax.axis('off')
                continue

            base = normalize_image(image)
            ax.imshow(base, cmap='gray', interpolation='nearest')
            if not hasattr(self, 'show_masks_checkbox') or self.show_masks_checkbox.isChecked():
                self._draw_mask_overlay(
                    ax,
                    base.shape,
                    active_roi=active_roi,
                    source=mask_source,
                )
            ax.set_aspect('equal')
            self._apply_axis_labels(ax, self.image_labels_checkbox.isChecked(), 'X (pixel)', 'Y (pixel)', title)

        self._resize_image_canvas_to_payloads([image for _row, image, _title in layer_payloads])
        fig.subplots_adjust(left=0.04, right=0.995, top=0.995, bottom=0.04, hspace=0.0, wspace=0.0)
        self.image_widget.capture_image_home()
        self.image_widget.canvas.draw_idle()

    def _image_layer_missing_text(self, row: ImageLayerControlRow) -> str:
        model = self._image_layer_model(row)
        if model == 'NoRMCorre':
            params = self._image_layer_params_for_row(row, model)
            path = self._image_layer_output_path(row, params, default_name=self._image_layer_default_output_name(model))
            return f'NoRMCorre output not found: {path.name}'
        if model == 'PMD':
            params = self._image_layer_params_for_row(row, model)
            path = self._image_layer_output_path(row, params, default_name=self._image_layer_default_output_name(model))
            return f'PMD output not found: {path.name}'
        if model == 'Local':
            path_text = str(row.image_layer_params.get('local_path') or '').strip()
            return f'Local image layer not found: {Path(path_text).name}' if path_text else 'Local image layer has no selected file.'
        if model == 'VolPy':
            return 'VolPy image file not found.'
        return 'No image data available.'

    def _resize_image_canvas_to_payloads(self, images: list[Optional[np.ndarray]]):
        shapes = []
        for image in images:
            if image is None:
                continue
            arr = np.asarray(image)
            if arr.ndim >= 2:
                shapes.append(arr.shape[-2:])
        if not shapes:
            self.image_widget.canvas.setMinimumSize(320, 280)
            return
        shape_key = tuple((int(shape[0]), int(shape[1])) for shape in shapes)
        available_width = self.plot_scroll.viewport().width() - 10 if hasattr(self, 'plot_scroll') else 850
        available_height = self.plot_scroll.viewport().height() if hasattr(self, 'plot_scroll') else 760
        width = int(np.clip(available_width, 520, 1800))
        layer_heights = []
        for height, layer_width in shape_key:
            aspect_height = int(round(width * height / max(layer_width, 1)))
            min_height = 160 if len(shape_key) > 1 else 240
            max_height = max(min_height, int(max(available_height * 0.72, 360)))
            layer_heights.append(int(np.clip(aspect_height, min_height, max_height)))
        height = int(np.clip(sum(layer_heights), 240, 6000))
        if self._image_canvas_target == (width, height, shape_key):
            return
        self._image_canvas_target = (width, height, shape_key)
        self.image_widget.canvas.setMinimumSize(320, height)
        self.image_widget.canvas.resize(width, height)
        self.image_widget.figure.set_size_inches(
            width / self.image_widget.figure.dpi,
            height / self.image_widget.figure.dpi,
            forward=True,
        )

    def _sync_image_frame_slider(self, max_frames: int):
        if not hasattr(self, 'image_frame_slider'):
            return
        max_index = max(0, int(max_frames) - 1)
        is_video = max_index > 0
        self.image_play_widget.setVisible(is_video)
        self.image_frame_slider.blockSignals(True)
        try:
            self.image_frame_slider.setMaximum(max_index)
            if self.image_frame_slider.value() > max_index:
                self.image_frame_slider.setValue(max_index)
            self.image_frame_slider.setEnabled(max_index > 0)
        finally:
            self.image_frame_slider.blockSignals(False)
        current = self.image_frame_slider.value() if max_index > 0 else 0
        self.image_frame_label.setText(f'{current + 1 if max_frames else 0} / {max_frames}')
        if max_index <= 0 and self._image_play_timer.isActive():
            self._image_play_timer.stop()
            self.image_play_button.setText('Play')

    def _get_image_layer_display(self, row: ImageLayerControlRow) -> tuple[Optional[np.ndarray], int, str]:
        mode = row.mode_combo.currentData()
        frame_idx = self.image_frame_slider.value() if hasattr(self, 'image_frame_slider') else 0
        model = self._image_layer_model(row)
        image_data = self._resolve_image_layer_data(row)
        image, nframes = self._project_image_data(image_data, None, mode, frame_idx)
        return image, nframes, f'{model} | {mode}'

    def _display_raw_movie(self) -> Optional[np.ndarray]:
        movie = self.state.raw_movie
        if movie is None:
            return None
        return np.asarray(movie)

    def _display_raw_image(self) -> Optional[np.ndarray]:
        image = self.state.raw_image
        if image is None:
            return None
        return np.asarray(image)

    def _update_image_layer_summary(self, row: ImageLayerControlRow, image: Optional[np.ndarray], nframes: int):
        arr = np.asarray(image) if image is not None else None
        size = f'{arr.shape[-1]} x {arr.shape[-2]}' if arr is not None and arr.ndim >= 2 else '-'
        rate = f'{self.state.frame_rate:g} Hz' if self.state.frame_rate else '-'
        row.summary_label.setText(f'Frames: {nframes} | Size: {size} | Rate: {rate}')

    def _project_image_data(
        self,
        movie: Optional[np.ndarray],
        image: Optional[np.ndarray],
        mode: str,
        frame_idx: int,
    ) -> tuple[Optional[np.ndarray], int]:
        if movie is not None:
            arr = np.asarray(movie)
            if arr.ndim > 3:
                arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
            if arr.ndim == 2:
                return arr.astype(float), 1
            if arr.ndim >= 3:
                nframes = int(arr.shape[0])
                if mode == 'video':
                    idx = int(np.clip(frame_idx, 0, nframes - 1))
                    return np.asarray(arr[idx], dtype=float), nframes
                cache_key = (id(movie), mode, tuple(arr.shape), str(arr.dtype))
                cached = self._image_projection_cache.get(cache_key)
                if cached is not None:
                    return cached
                if mode == 'z_average':
                    result = (np.asarray(np.nanmean(arr, axis=0), dtype=float), nframes)
                    self._image_projection_cache[cache_key] = result
                    return result
                if mode == 'max_projection':
                    result = (np.asarray(np.nanmax(arr, axis=0), dtype=float), nframes)
                    self._image_projection_cache[cache_key] = result
                    return result
                if mode == 'correlation':
                    result = (self._correlation_image(arr), nframes)
                    self._image_projection_cache[cache_key] = result
                    return result
        if image is not None:
            return np.asarray(image, dtype=float), 1
        return None, 0

    def _resolve_image_layer_data(self, row: ImageLayerControlRow) -> Optional[np.ndarray]:
        model = self._image_layer_model(row)
        if model == 'Raw':
            movie = self._get_image_layer_movie(row)
            if movie is not None:
                return movie
            return self._display_raw_image()
        if model == 'VolPy':
            return self._get_image_layer_movie(row)
        if model == 'NoRMCorre':
            return self._get_image_layer_movie(row)
        if model == 'PMD':
            return self._get_image_layer_movie(row)
        if model == 'Local':
            movie = self._get_image_layer_movie(row)
            return movie if movie is not None else self._resolve_local_image_movie(row)
        self._set_status(f'{model} image layer loading is reserved for later development.')
        return None

    def _resolve_normcorre_image_movie(self, row: ImageLayerControlRow) -> Optional[np.ndarray]:
        params = self._image_layer_params_for_row(row, 'NoRMCorre')
        output_path = self._image_layer_output_path(row, params, default_name=self._image_layer_default_output_name('NoRMCorre'))
        if not output_path.exists():
            self._set_status(f'NoRMCorre output was not found: {output_path.name}')
            return None
        try:
            arr = tifffile.memmap(output_path)
        except Exception as exc:
            self._set_status(f'NoRMCorre TIFF load failed: {exc}')
            return None
        return np.asarray(arr)

    def _resolve_pmd_image_movie(self, row: ImageLayerControlRow) -> Optional[np.ndarray]:
        params = self._image_layer_params_for_row(row, 'PMD')
        output_path = self._image_layer_output_path(row, params, default_name=self._image_layer_default_output_name('PMD'))
        if not output_path.exists():
            self._set_status(f'PMD output was not found: {output_path.name}')
            return None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*shaped series axes do not match shape.*")
                arr = tifffile.memmap(output_path)
        except Exception as exc:
            self._set_status(f'PMD TIFF load failed: {exc}')
            return None
        arr = np.asarray(arr)
        if arr.ndim > 3:
            arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
        return arr

    def _resolve_local_image_movie(self, row: ImageLayerControlRow) -> Optional[np.ndarray]:
        path_text = str(row.image_layer_params.get('local_path') or '').strip()
        if not path_text:
            self._set_status('Local image layer has no selected file.')
            return None
        path = Path(path_text)
        if not path.exists():
            self._set_status(f'Local image layer file was not found: {path}')
            return None
        suffix = path.suffix.lower()
        if suffix in {'.tif', '.tiff'}:
            try:
                return np.asarray(tifffile.memmap(path))
            except Exception:
                try:
                    return np.asarray(tifffile.imread(path))
                except Exception as exc:
                    self._set_status(f'Local TIFF load failed: {exc}')
                    return None
        if suffix == '.mmap':
            return self._load_local_mmap_movie(path)
        self._set_status(f'Unsupported local image layer file: {path.name}')
        return None

    def _resolve_volpy_image_movie(self) -> Optional[np.ndarray]:
        mmap_path = self._find_volpy_mmap_path()
        if mmap_path is not None:
            loaded = self._load_volpy_mmap_movie(mmap_path)
            if loaded is not None:
                return loaded

        tif_path = self._find_volpy_tif_path()
        if tif_path is not None:
            try:
                arr = tifffile.memmap(tif_path)
            except Exception as exc:
                self._set_status(f'VolPy TIFF load failed: {exc}')
                return None
            return np.asarray(arr)

        self._set_status('VolPy image layer requires volpy*.tif or volpy*.mmap beside the raw image file.')
        return None

    def _find_volpy_tif_path(self) -> Optional[Path]:
        search_dir = self._volpy_image_search_dir()
        if search_dir is None:
            return None
        return self._find_first_file(search_dir, ['volpy*.tif', '*volpy*.tif', 'volpy*.tiff', '*volpy*.tiff'])

    def _find_volpy_mmap_path(self) -> Optional[Path]:
        search_dir = self._volpy_image_search_dir()
        if search_dir is None:
            return None
        return self._find_volpy_mmap_file(search_dir)

    def _find_volpy_mmap_file(self, search_dir: Path) -> Optional[Path]:
        preferred_patterns = ['volpy*.mmap', '*volpy*.mmap', 'memmap*.mmap', '*memmap*.mmap']
        preferred = self._collect_candidate_files(search_dir, preferred_patterns)
        selected = self._select_caiman_mmap_candidate(preferred, decoded_only=True)
        if selected is not None:
            return selected

        all_mmaps = self._collect_candidate_files(search_dir, ['*.mmap'])
        selected = self._select_caiman_mmap_candidate(all_mmaps, decoded_only=True)
        if selected is not None:
            return selected

        selected = self._select_caiman_mmap_candidate(preferred, decoded_only=False)
        if selected is not None:
            return selected
        return self._select_caiman_mmap_candidate(all_mmaps, decoded_only=False)

    def _collect_candidate_files(self, search_dir: Path, patterns: list[str]) -> list[Path]:
        candidates: list[Path] = []
        seen: set[Path] = set()
        for recursive in (False, True):
            for pattern in patterns:
                found = sorted(search_dir.rglob(pattern) if recursive else search_dir.glob(pattern))
                for path in found:
                    if path in seen or not path.is_file():
                        continue
                    seen.add(path)
                    candidates.append(path)
            if candidates:
                return candidates
        return candidates

    def _select_caiman_mmap_candidate(self, candidates: list[Path], decoded_only: bool) -> Optional[Path]:
        if not candidates:
            return None

        scored: list[tuple[tuple[int, int, int, int, int], Path]] = []
        fallback: Optional[Path] = None
        for pos, path in enumerate(candidates):
            decoded = self._decode_caiman_mmap_filename(path)
            if decoded is None:
                if fallback is None:
                    fallback = path
                continue
            size_ok = self._caiman_mmap_size_matches(path, decoded)
            if decoded_only and not size_ok:
                continue
            order_score = 1 if str(decoded.get('order')) == 'C' else 0
            frames = int(decoded.get('T', 0))
            name = path.name.lower()
            name_score = int('volpy' in name or 'memmap' in name)
            scored.append(((int(size_ok), order_score, frames, name_score, -pos), path))

        if scored:
            return max(scored, key=lambda item: item[0])[1]
        if decoded_only:
            return None
        return fallback

    def _find_first_file(self, search_dir: Path, patterns: list[str]) -> Optional[Path]:
        for pattern in patterns:
            candidates = sorted(search_dir.glob(pattern))
            if candidates:
                return candidates[0]
        for pattern in patterns:
            candidates = sorted(search_dir.rglob(pattern))
            if candidates:
                return candidates[0]
        return None

    def _volpy_image_search_dir(self) -> Optional[Path]:
        if self.state.tif_path:
            raw_path = Path(self.state.tif_path)
            if raw_path.exists():
                return raw_path.parent
        if self.state.source_path:
            source_path = Path(self.state.source_path)
            search_dir = source_path if source_path.is_dir() else source_path.parent
            if search_dir.exists():
                return search_dir
        return None

    def _load_volpy_mmap_movie(self, path: Path) -> Optional[np.ndarray]:
        caiman_movie = self._load_caiman_mmap_movie(path)
        if caiman_movie is not None:
            return caiman_movie

        shape = self._raw_movie_shape()
        if shape is None:
            self._set_status('VolPy mmap loading needs raw TIFF movie shape.')
            return None
        if not path.exists():
            self._set_status(f'VolPy mmap file was not found: {path}')
            return None
        frames = int(shape[0])
        dims = tuple(int(v) for v in shape[1:])
        n_pixels = int(np.prod(dims))
        expected_size = frames * n_pixels * np.dtype(np.float32).itemsize
        if path.stat().st_size != expected_size:
            self._set_status('VolPy mmap size does not match raw TIFF movie shape.')
            return None
        try:
            raw = np.memmap(path, dtype=np.float32, mode='r', shape=(n_pixels, frames), order='F')
        except Exception as exc:
            self._set_status(f'VolPy mmap load failed: {exc}')
            return None
        self._set_status(f'Loaded VolPy mmap using raw TIFF dimensions: {path}')
        return np.reshape(raw.T, (frames,) + dims, order='F')

    def _load_local_mmap_movie(self, path: Path) -> Optional[np.ndarray]:
        caiman_movie = self._load_caiman_mmap_movie(path)
        if caiman_movie is not None:
            return caiman_movie

        shape = self._raw_movie_shape()
        if shape is None:
            self._set_status('Local mmap loading needs raw TIFF movie shape or CaImAn dimensions in the filename.')
            return None
        frames = int(shape[0])
        dims = tuple(int(v) for v in shape[1:])
        n_pixels = int(np.prod(dims))
        expected_size = frames * n_pixels * np.dtype(np.float32).itemsize
        if path.stat().st_size != expected_size:
            self._set_status('Local mmap size does not match raw TIFF movie shape.')
            return None
        try:
            raw = np.memmap(path, dtype=np.float32, mode='r', shape=(n_pixels, frames), order='F')
        except Exception as exc:
            self._set_status(f'Local mmap load failed: {exc}')
            return None
        self._set_status(f'Loaded local mmap using raw TIFF dimensions: {path}')
        return np.reshape(raw.T, (frames,) + dims, order='F')

    def _load_caiman_mmap_movie(self, path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            self._set_status(f'VolPy mmap file was not found: {path}')
            return None
        decoded = self._decode_caiman_mmap_filename(path)
        if decoded is None:
            return None
        d1 = int(decoded['d1'])
        d2 = int(decoded['d2'])
        d3 = int(decoded.get('d3', 1))
        frames = int(decoded['T'])
        order = str(decoded['order'])
        dims = (d1, d2) if d3 == 1 else (d1, d2, d3)
        n_pixels = int(np.prod(dims))
        if not self._caiman_mmap_size_matches(path, decoded):
            self._set_status('VolPy mmap size does not match the shape encoded in the filename.')
            return None
        try:
            raw = np.memmap(path, dtype=np.float32, mode='r', shape=(n_pixels, frames), order=order)
        except Exception as exc:
            self._set_status(f'VolPy mmap load failed: {exc}')
            return None
        self._set_status(f'Loaded CaImAn/VolPy mmap: {path}')
        return np.reshape(raw.T, (frames,) + dims, order='F')

    def _decode_caiman_mmap_filename(self, path: Path) -> Optional[dict[str, Any]]:
        parts = [part for part in path.stem.split('_') if part]
        if not parts:
            return None

        decoded: dict[str, Any] = {}
        for field in ('d1', 'd2', 'd3', 'order', 'frames', 'T'):
            for idx in range(len(parts) - 1, -1, -1):
                if parts[idx] != field or idx + 1 >= len(parts):
                    continue
                value = parts[idx + 1]
                if field == 'order':
                    decoded[field] = value
                else:
                    try:
                        decoded[field] = int(value)
                    except ValueError:
                        return None
                break

        if 'T' not in decoded and 'frames' in decoded:
            decoded['T'] = decoded['frames']
        if 'd3' not in decoded:
            decoded['d3'] = 1
        required = {'d1', 'd2', 'T', 'order'}
        if not required.issubset(decoded) or decoded['order'] not in {'C', 'F'}:
            return None
        return decoded

    def _caiman_mmap_size_matches(self, path: Path, decoded: dict[str, Any]) -> bool:
        try:
            d1 = int(decoded['d1'])
            d2 = int(decoded['d2'])
            d3 = int(decoded.get('d3', 1))
            frames = int(decoded['T'])
        except (KeyError, TypeError, ValueError):
            return False
        expected_size = d1 * d2 * d3 * frames * np.dtype(np.float32).itemsize
        try:
            return path.stat().st_size == expected_size
        except OSError:
            return False

    def _raw_movie_shape(self) -> Optional[tuple[int, ...]]:
        if self.state.raw_movie is not None:
            arr = np.asarray(self.state.raw_movie)
            if arr.ndim >= 3:
                return tuple(int(v) for v in arr.shape[:3])
        if self.state.tif_path and Path(self.state.tif_path).exists():
            arr = tifffile.memmap(self.state.tif_path)
            if arr.ndim >= 3:
                return tuple(int(v) for v in arr.shape[:3])
        return None

    def _correlation_image(self, movie: np.ndarray) -> np.ndarray:
        arr = np.asarray(movie, dtype=np.float32)
        if arr.ndim < 3:
            return np.asarray(arr, dtype=float)
        if arr.shape[0] > 500:
            sample_idx = np.linspace(0, arr.shape[0] - 1, 500).astype(int)
            arr = arr[sample_idx]
        if arr.ndim == 4 and arr.shape[-1] in (3, 4):
            arr = arr[..., :3].mean(axis=-1)
        if arr.ndim != 3:
            return np.asarray(np.nanmean(arr, axis=0), dtype=float)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        arr = arr - np.mean(arr, axis=0, keepdims=True)
        arr_std = np.std(arr, axis=0, keepdims=True)
        arr_std[arr_std == 0] = np.inf
        arr = arr / arr_std

        kernel = np.ones((3, 3), dtype=np.float32)
        kernel[1, 1] = 0.0
        yconv = None
        try:
            import cv2
            yconv = np.stack([cv2.filter2D(frame, -1, kernel, borderType=0) for frame in arr])
            mask = cv2.filter2D(np.ones(arr.shape[1:], dtype=np.float32), -1, kernel, borderType=0)
        except Exception:
            yconv = convolve(arr, kernel[np.newaxis, :, :], mode='constant')
            mask = convolve(np.ones(arr.shape[1:], dtype=np.float32), kernel, mode='constant')
        mask = np.asarray(mask, dtype=np.float32)
        mask[mask == 0] = np.inf
        corr = np.mean(yconv * arr, axis=0) / mask
        return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    def _draw_mask_overlay(self, ax, image_shape: tuple[int, ...], active_roi: Optional[int], source: str):
        rgba = self._build_mask_rgba(image_shape, source=source)
        if rgba is not None:
            ax.imshow(rgba, interpolation='nearest')
        outline_rois = self.get_selected_roi_indices()
        if not outline_rois and active_roi is not None:
            outline_rois = [active_roi]
        for roi_idx in outline_rois:
            self._draw_roi_highlight_overlay(ax, image_shape, roi_idx, source=source, color=self._roi_display_color(roi_idx))

    def _draw_roi_highlight_overlay(self, ax, image_shape: tuple[int, ...], roi_idx: int, source: str, color: str):
        rgba = np.zeros((int(image_shape[0]), int(image_shape[1]), 4), dtype=float)
        rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
        text_xy: Optional[tuple[float, float]] = None
        if source == 'volpy':
            vpy_data = self.ensure_volpy_loaded(show_warning=False)
            rois = self._volpy_mask_stack(image_shape=image_shape[:2])
            volpy_row = self._volpy_row_for_suite2p_roi(roi_idx)
            if not vpy_data or rois is None or volpy_row is None or volpy_row >= rois.shape[0]:
                return
            mask = np.asarray(rois[volpy_row], dtype=float)
            ypix, xpix = np.where(mask > 0)
            if xpix.size == 0:
                return
            alpha = np.abs(mask[ypix, xpix])
            max_alpha = float(np.nanmax(alpha)) if alpha.size else 0.0
            alpha = alpha / max_alpha if max_alpha > 1e-8 else np.ones_like(alpha, dtype=float)
            text_xy = (float(np.mean(xpix)), float(np.mean(ypix)))
        else:
            if self.state.stat is None or roi_idx >= len(self.state.stat):
                return
            roi = self.state.stat[roi_idx]
            xpix = np.asarray(roi['xpix'], dtype=int)
            ypix = np.asarray(roi['ypix'], dtype=int)
            valid = (ypix >= 0) & (ypix < rgba.shape[0]) & (xpix >= 0) & (xpix < rgba.shape[1])
            ypix = ypix[valid]
            xpix = xpix[valid]
            if xpix.size == 0:
                return
            alpha_values = self._roi_mask_alpha(source, roi, ypix, xpix)
            alpha = np.asarray(alpha_values, dtype=float) if alpha_values is not None else np.ones_like(ypix, dtype=float)
            text_xy = (float(np.mean(xpix)), float(np.mean(ypix)))
        rgba[ypix, xpix, :3] = rgb
        rgba[ypix, xpix, 3] = np.clip(0.2 + 0.75 * alpha, 0.0, 0.95)
        ax.imshow(rgba, interpolation='nearest', zorder=8)
        if text_xy is not None:
            ax.text(
                text_xy[0],
                text_xy[1],
                str(roi_idx),
                color=color,
                fontsize=8,
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, alpha=0.85),
                zorder=9,
            )

    def _build_mask_rgba(self, image_shape: tuple[int, ...], source: str) -> Optional[np.ndarray]:
        if source == 'volpy':
            alpha = self._build_volpy_roi_alpha_map(image_shape)
            if alpha is None:
                return None
            rgba = np.zeros((*alpha.shape[:2], 4), dtype=float)
            rgba[..., :3] = ROI_TYPE_COLORS['roi']
            rgba[..., 3] = np.clip(alpha * 0.75, 0.0, 0.85)
            return rgba

        if self.state.stat is None:
            alpha = self._get_current_mask_alpha(image_shape, source=source)
            if alpha is None:
                return None
            rgba = np.zeros((*alpha.shape[:2], 4), dtype=float)
            rgba[..., :3] = ROI_TYPE_COLORS['roi']
            rgba[..., 3] = np.clip(alpha * 0.75, 0.0, 0.85)
            return rgba

        ly, lx = image_shape[:2]
        rgba = np.zeros((ly, lx, 4), dtype=float)
        for roi_idx, roi in enumerate(self.state.stat):
            if self.only_cells_checkbox.isChecked() and self.state.cells is not None and roi_idx < len(self.state.cells) and self.state.cells[roi_idx] != 1:
                continue
            ypix = np.asarray(roi['ypix'], dtype=int)
            xpix = np.asarray(roi['xpix'], dtype=int)
            valid = (ypix >= 0) & (ypix < ly) & (xpix >= 0) & (xpix < lx)
            ypix = ypix[valid]
            xpix = xpix[valid]
            if ypix.size == 0:
                continue
            alpha = self._roi_mask_alpha(source, roi, ypix, xpix)
            if alpha is None:
                continue
            color = self._roi_mask_color(roi_idx)
            alpha = np.clip(alpha * 0.85, 0.0, 0.85)
            replace = alpha > rgba[ypix, xpix, 3]
            if np.any(replace):
                rgba[ypix[replace], xpix[replace], :3] = color
                rgba[ypix[replace], xpix[replace], 3] = alpha[replace]
        if np.nanmax(rgba[..., 3]) <= 0:
            return None
        return rgba

    def _roi_mask_alpha(self, source: str, roi: dict[str, Any], ypix: np.ndarray, xpix: np.ndarray) -> Optional[np.ndarray]:
        if source == 'suite2p':
            alpha = np.asarray(roi.get('lam', np.ones_like(ypix, dtype=float)), dtype=float)
            if alpha.size < ypix.size:
                alpha = np.pad(alpha, (0, ypix.size - alpha.size), constant_values=1.0)
            alpha = alpha[:ypix.size]
        elif source in {'weighted_pix_exp', 'weighted_pix_overmean', 'weighted_pix_max'}:
            if self.state.raw_image is None:
                return None
            values = np.asarray(self.state.raw_image[ypix, xpix], dtype=float)
            if source == 'weighted_pix_exp':
                alpha = pix_exp(values)
            elif source == 'weighted_pix_overmean':
                alpha = pix_overmean(values)
            else:
                alpha = pix_max(values)
        else:
            return None
        max_alpha = np.nanmax(alpha) if alpha.size else 0.0
        if max_alpha <= 1e-8:
            return np.zeros_like(alpha, dtype=float)
        return np.asarray(alpha, dtype=float) / max_alpha

    def _roi_mask_color(self, roi_idx: int) -> tuple[float, float, float]:
        mode = self.mask_color_combo.currentData() if hasattr(self, 'mask_color_combo') else 'roi'
        if mode == 'type':
            cell_flag = self._roi_is_cell(roi_idx)
            stim_flag = self._roi_is_stim(roi_idx)
            if cell_flag is None:
                return ROI_TYPE_COLORS['roi']
            if cell_flag and stim_flag:
                return ROI_TYPE_COLORS['stim_cell']
            if cell_flag:
                return ROI_TYPE_COLORS['unstim_cell']
            if stim_flag:
                return ROI_TYPE_COLORS['stim_non_cell']
            return ROI_TYPE_COLORS['unstim_non_cell']
        return self._roi_color_rgb(roi_idx)

    def _roi_display_color(self, roi_idx: int) -> str:
        return ROI_COLORS[int(roi_idx) % len(ROI_COLORS)]

    def _roi_color_rgb(self, roi_idx: int) -> tuple[float, float, float]:
        return tuple(float(value) for value in mcolors.to_rgb(self._roi_display_color(roi_idx)))

    def _roi_is_cell(self, roi_idx: int) -> Optional[bool]:
        if self.state.cells is None or roi_idx >= len(self.state.cells):
            return None
        return int(self.state.cells[roi_idx]) == 1

    def _roi_is_stim(self, roi_idx: int) -> bool:
        if self.state.stim_cells is None or roi_idx >= len(self.state.stim_cells):
            return False
        return int(self.state.stim_cells[roi_idx]) == 1

    def _get_current_mask_alpha(self, image_shape: tuple[int, ...], source: str) -> Optional[np.ndarray]:
        alpha = None
        if source == 'suite2p':
            alpha = self.state.suite2p_alpha
        elif source in self.state.weight_alpha_maps:
            alpha = self.state.weight_alpha_maps[source]
        elif source == 'volpy':
            alpha = self._build_volpy_roi_alpha_map(image_shape)
        if alpha is None:
            return None
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape[:2] != image_shape[:2]:
            return None
        return alpha

    def _build_volpy_weight_alpha_map(self, image_shape: Optional[tuple[int, ...]] = None) -> Optional[np.ndarray]:
        vpy_data = self.ensure_volpy_loaded(show_warning=False)
        if not vpy_data or 'weights' not in vpy_data:
            return None
        weights = np.asarray(vpy_data['weights'], dtype=float)
        if weights.size == 0:
            return None
        if weights.ndim == 2:
            alpha = normalize_image(np.abs(weights))
            return alpha if image_shape is None or alpha.shape[:2] == image_shape[:2] else None
        if weights.ndim == 3:
            if weights.shape[0] == 0:
                return None
            alpha = normalize_image(np.nanmax(np.abs(weights), axis=0))
            return alpha if image_shape is None or alpha.shape[:2] == image_shape[:2] else None
        return None

    def _build_volpy_roi_alpha_map(self, image_shape: tuple[int, ...]) -> Optional[np.ndarray]:
        if self._current_volpy_payload() is None:
            self.ensure_volpy_loaded(show_warning=False)
        masks = self._volpy_mask_stack(image_shape=image_shape[:2])
        if masks is None or masks.size == 0 or masks.shape[0] == 0:
            return None
        if masks.ndim != 3 or masks.shape[1:3] != image_shape[:2]:
            return None
        mapping = self._ensure_volpy_suite2p_mapping(self._current_volpy_payload(), masks.shape[0])
        if mapping is None:
            return None
        if self.only_cells_checkbox.isChecked():
            if self.state.cells is not None:
                keep = [idx for idx, roi_idx in enumerate(mapping) if roi_idx < len(self.state.cells) and self.state.cells[int(roi_idx)] == 1]
                masks = masks[np.asarray(keep, dtype=int)] if keep else masks[:0]
        if masks.size == 0:
            return None
        return normalize_image(np.nanmax(np.abs(masks), axis=0))

    def _volpy_mask_stack(self, image_shape: Optional[tuple[int, int]] = None) -> Optional[np.ndarray]:
        vpy_data = self._current_volpy_payload()
        if not vpy_data or 'weights' not in vpy_data:
            return None
        masks = self._coerce_roi_stack(np.asarray(vpy_data['weights'], dtype=float), image_shape=image_shape)
        if masks is None or masks.ndim != 3:
            return None
        return np.asarray(masks, dtype=np.float32)

    def get_mask_target_layer_ids(self) -> set[str]:
        if not hasattr(self, 'mask_target_list'):
            return {'raw'}
        return {item.data(Qt.ItemDataRole.UserRole) for item in self.mask_target_list.selectedItems()}

    def get_active_roi_index(self) -> Optional[int]:
        if self.state.cells is None or not hasattr(self, 'roi_index_slider'):
            return None
        if len(self.state.cells) == 0:
            return None
        return int(np.clip(self.roi_index_slider.value(), 0, len(self.state.cells) - 1))

    def _update_roi_slider(self):
        if not hasattr(self, 'roi_index_slider'):
            return
        n_rois = len(self.state.cells) if self.state.cells is not None else 0
        self.roi_index_slider.blockSignals(True)
        try:
            self.roi_index_slider.setMaximum(max(0, n_rois - 1))
            self.roi_index_slider.setEnabled(n_rois > 0)
            if self.roi_index_slider.value() >= n_rois and n_rois > 0:
                self.roi_index_slider.setValue(n_rois - 1)
            elif n_rois == 0:
                self.roi_index_slider.setValue(0)
        finally:
            self.roi_index_slider.blockSignals(False)
        self.roi_index_label.setText(str(self.roi_index_slider.value() if n_rois else 0))
        if n_rois and not self.selected_roi_indices:
            self.selected_roi_indices = [self.roi_index_slider.value()]
        if hasattr(self, 'active_roi_label'):
            self.active_roi_label.setText(self._selected_roi_label_text(self.selected_roi_indices))

    def _update_image_layer_labels(self):
        for row in self.image_layers:
            if row.layer_id == 'raw':
                nframes = int(self.state.raw_movie.shape[0]) if self.state.raw_movie is not None and self.state.raw_movie.ndim >= 3 else (1 if self.state.raw_image is not None else 0)
            else:
                data = self._get_image_layer_movie(row)
                nframes = int(np.asarray(data).shape[0]) if data is not None and np.asarray(data).ndim >= 3 else (1 if data is not None else 0)
            row.nframes_label.setText(f'{nframes} frame{"s" if nframes != 1 else ""}')

    def _plot_roi_outline(self, ax, roi_idx: int, source: str = 'suite2p', color: str = 'C0'):
        if source == 'volpy':
            vpy_data = self.ensure_volpy_loaded(show_warning=False)
            rois = self._volpy_mask_stack()
            if not vpy_data or rois is None:
                return
            volpy_row = self._volpy_row_for_suite2p_roi(roi_idx)
            if rois.ndim < 3 or volpy_row is None or volpy_row >= rois.shape[0]:
                return
            ypix, xpix = np.where(np.asarray(rois[volpy_row]) > 0)
            if xpix.size == 0:
                return
            ax.scatter(xpix, ypix, s=2, c=color, alpha=0.9)
            ax.text(float(np.mean(xpix)), float(np.mean(ypix)), str(roi_idx), color=color, fontsize=8, ha='center', va='center')
            return

        if self.state.stat is None or roi_idx >= len(self.state.stat):
            return
        roi = self.state.stat[roi_idx]
        xpix = np.asarray(roi['xpix'], dtype=int)
        ypix = np.asarray(roi['ypix'], dtype=int)
        if xpix.size == 0:
            return
        ax.scatter(xpix, ypix, s=2, c=color, alpha=0.9)
        ax.text(
            float(np.mean(xpix)),
            float(np.mean(ypix)),
            str(roi_idx),
            color=color,
            fontsize=8,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, alpha=0.85),
        )

    def _on_image_click_select_roi(self, event):
        if event.button != 1 or event.dblclick or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        source = self._image_axis_mask_sources.get(event.inaxes)
        roi_idx = self._roi_at_pixel(int(round(event.xdata)), int(round(event.ydata)), source=source)
        if roi_idx is not None:
            self._select_roi(roi_idx)

    def _roi_at_pixel(self, x: int, y: int, source: Optional[str] = None) -> Optional[int]:
        if source == 'volpy':
            vpy_data = self.ensure_volpy_loaded(show_warning=False)
            rois = self._volpy_mask_stack()
            if vpy_data and rois is not None:
                if rois.ndim >= 3 and 0 <= y < rois.shape[-2] and 0 <= x < rois.shape[-1]:
                    hits = np.where(rois[:, y, x] > 0)[0]
                    if hits.size:
                        mapping = self._ensure_volpy_suite2p_mapping(vpy_data, rois.shape[0])
                        if mapping is None:
                            return None
                        for hit in hits:
                            if hit < len(mapping):
                                roi_idx = int(mapping[hit])
                                if self.only_cells_checkbox.isChecked() and self.state.cells is not None and roi_idx < len(self.state.cells) and self.state.cells[roi_idx] != 1:
                                    continue
                                return roi_idx
            return None

        if self.state.stat is not None:
            for roi_idx, roi in enumerate(self.state.stat):
                if self.only_cells_checkbox.isChecked() and self.state.cells is not None and self.state.cells[roi_idx] != 1:
                    continue
                xpix = np.asarray(roi['xpix'], dtype=int)
                ypix = np.asarray(roi['ypix'], dtype=int)
                if np.any((xpix == x) & (ypix == y)):
                    return int(roi_idx)

        if source is None:
            vpy_data = self.ensure_volpy_loaded(show_warning=False)
        else:
            vpy_data = None
        rois = self._volpy_mask_stack() if vpy_data else None
        if vpy_data and rois is not None:
            if rois.ndim >= 3 and 0 <= y < rois.shape[-2] and 0 <= x < rois.shape[-1]:
                hits = np.where(rois[:, y, x] > 0)[0]
                if hits.size:
                    mapping = self._ensure_volpy_suite2p_mapping(vpy_data, rois.shape[0])
                    if mapping is None:
                        return None
                    for hit in hits:
                        if hit < len(mapping):
                            roi_idx = int(mapping[hit])
                            if self.only_cells_checkbox.isChecked() and self.state.cells is not None and roi_idx < len(self.state.cells) and self.state.cells[roi_idx] != 1:
                                continue
                            return roi_idx
        return None

    def _select_roi(self, roi_idx: int, update_slider: bool = True):
        self.selected_roi_indices = [int(roi_idx)]
        if hasattr(self, 'roi_list'):
            self.roi_list.blockSignals(True)
            for idx in range(self.roi_list.count()):
                item = self.roi_list.item(idx)
                item.setSelected(int(item.data(Qt.ItemDataRole.UserRole)) == int(roi_idx))
            self.roi_list.blockSignals(False)
        if hasattr(self, 'active_roi_label'):
            self.active_roi_label.setText(self._selected_roi_label_text(self.selected_roi_indices))
        if update_slider and hasattr(self, 'roi_index_slider'):
            self.roi_index_slider.blockSignals(True)
            self.roi_index_slider.setValue(int(roi_idx))
            self.roi_index_slider.blockSignals(False)
            self.roi_index_label.setText(str(roi_idx))
        self.render_all()

    def _get_active_trace_results(self) -> list[dict[str, Any]]:
        results = []
        for row in self.trace_rows:
            if not row.visible_checkbox.isChecked():
                continue
            result = self._build_trace_result(row)
            if result is not None:
                results.append(result)
        return results

    def _build_trace_result(self, row: TraceControlRow) -> Optional[dict[str, Any]]:
        cache_key = self._trace_cache_key(row)
        cached = self._trace_result_cache.get(cache_key)
        if cached is not None:
            return cached
        source = self._get_trace_source_matrix(row)
        if source is None:
            return None
        trace_mat, source_name, roi_indices = source
        if trace_mat.size == 0:
            return None

        trace_mat = np.asarray(trace_mat, dtype=float).copy()
        trace_mat, frame_indices = self._apply_event_trace_actions(trace_mat)
        raw_trace_mat = trace_mat.copy()
        spike_times = None
        firing_rate = None
        thresholds = None
        snr_values = None
        spike_label = ''
        trace_mat, process_parts, detail_payload = self._apply_trace_pipeline(trace_mat, row)
        if row.spike_checkbox.isChecked():
            trace_mat, spike_times, firing_rate, thresholds, snr_values, spike_label = self._detect_trace_spikes(
                trace_mat,
                row,
                frame_indices=frame_indices,
            )

        title = source_name
        if process_parts:
            title = f'{title} | ' + ' + '.join(process_parts)
        if spike_label:
            title = f'{title} | spikes:{spike_label}'

        result = {
            'row': row,
            'name': title,
            'data': trace_mat,
            'raw_data': raw_trace_mat,
            'frame_indices': frame_indices,
            'roi_indices': roi_indices,
            'spike_times': spike_times,
            'firing_rate': firing_rate,
            'thresholds': thresholds,
            'snr_values': snr_values,
            'waveform': row.spike_checkbox.isChecked() and row.waveform_checkbox.isChecked(),
            'baseline': self._trace_has_baseline(row),
            'baseline_mode': row.baseline_mode_combo.currentData() if self._trace_has_baseline(row) else '',
            'details': detail_payload,
        }
        self._trace_result_cache[cache_key] = result
        return result

    def _trace_cache_key(self, row: TraceControlRow) -> tuple[Any, ...]:
        return (
            self.state.source_path,
            row.source_combo.currentData(),
            row.volpy_checkbox.isChecked(), row.volpy_combo.currentText(),
            self.state.negative_mode,
            self._negative_view_requested(),
            hasattr(self, 'data_raw_checkbox') and self.data_raw_checkbox.isChecked(),
            row.lowpass_checkbox.isChecked(), row.lowpass_edit.text(),
            row.highpass_checkbox.isChecked(), row.highpass_edit.text(),
            row.wavelet_checkbox.isChecked(), row.wavelet_name_edit.text(), row.wavelet_level_edit.text(),
            row.wavelet_scale_edit.text(), row.wavelet_mode_combo.currentText(),
            row.pca_wavelet_checkbox.isChecked(),
            row.pca_wavelet_fmin_edit.text(), row.pca_wavelet_fmax_edit.text(), row.pca_wavelet_fn_edit.text(),
            tuple(sorted(row.pca_wavelet_cfg.items())),
            row.snr_checkbox.isChecked(), row.snr_window_edit.text(),
            row.baseline_mode_combo.currentData(),
            row.baseline_lowpass_checkbox.isChecked(), row.baseline_lowpass_edit.text(),
            row.baseline_rolling_checkbox.isChecked(), row.baseline_rolling_mode_combo.currentText(),
            row.baseline_rolling_window_edit.text(),
            row.baseline_polyfit_checkbox.isChecked(), row.baseline_poly_order_edit.text(),
            row.baseline_savgol_checkbox.isChecked(), row.baseline_savgol_window_edit.text(),
            row.baseline_savgol_order_edit.text(),
            row.spike_checkbox.isChecked(), row.spike_method_combo.currentText(), row.spike_k_edit.text(),
            row.waveform_checkbox.isChecked(), row.waveform_mode_combo.currentText(),
            tuple(
                (
                    event.get('source'),
                    event.get('label'),
                    event.get('color'),
                    event.get('mode'),
                    tuple(np.asarray(event.get('start_frames', []), dtype=int).tolist()),
                    tuple(np.asarray(event.get('end_frames', []), dtype=int).tolist()),
                    event.get('diff'),
                )
                for event in self.events
            ),
        )

    def _get_trace_source_matrix(self, row: TraceControlRow) -> Optional[tuple[np.ndarray, str, Optional[np.ndarray]]]:
        if row.volpy_checkbox.isChecked():
            key = row.volpy_combo.currentText()
            vpy_data = self.ensure_volpy_loaded(show_warning=True)
            if vpy_data and key in vpy_data:
                mat = self._coerce_trace_matrix(vpy_data[key])
                if mat is not None:
                    mapping = self._ensure_volpy_suite2p_mapping(vpy_data, mat.shape[0])
                    return mat, f'VolPy {key}', mapping if mapping is not None else np.asarray([], dtype=int)

        data = row.source_combo.currentData()
        if isinstance(data, str) and data.startswith('image:'):
            return self._get_image_trace_source_matrix(data, row)
        return self._get_source_matrix_by_data(data, row)

    def _get_source_matrix_by_data(self, data: Any, row: Optional[TraceControlRow] = None) -> Optional[tuple[np.ndarray, str, Optional[np.ndarray]]]:
        if isinstance(data, str) and data.startswith('image:'):
            return self._get_image_trace_source_matrix(data, row)
        if isinstance(data, str) and data.startswith('direct:'):
            direct = self._get_direct_trace_source_matrix(data.split(':', 1)[1])
            if direct is None:
                return None
            mat, label = direct
            return mat, label, None
        if isinstance(data, str) and data.startswith('state:'):
            idx = int(data.split(':', 1)[1])
            if idx < len(self.state.traces):
                return self._state_trace_matrix(idx), self.state.trace_names[idx], None
            return None
        if isinstance(data, str) and data.startswith('volpy:'):
            key = data.split(':', 1)[1]
            vpy_data = self.ensure_volpy_loaded(show_warning=True)
            if not vpy_data or key not in vpy_data:
                return None
            mat = self._coerce_trace_matrix(vpy_data[key])
            if mat is None:
                return None
            mapping = self._ensure_volpy_suite2p_mapping(vpy_data, mat.shape[0])
            return mat, f'VolPy {key}', mapping if mapping is not None else np.asarray([], dtype=int)
        return None

    def _state_trace_matrix(self, idx: int) -> np.ndarray:
        mat = np.asarray(self.state.traces[idx], dtype=float)
        if self._data_view_needs_reverse() and self.state.trace_reverse_max is not None:
            maxima = np.asarray(self.state.trace_reverse_max, dtype=float)
            if maxima.ndim == 1:
                maxima = maxima.reshape(-1, 1)
            if maxima.shape[0] == mat.shape[0]:
                return maxima - mat
        return mat

    def _apply_recomputed_trace_polarity(self, trace_mat: np.ndarray, movie: Any, masks: np.ndarray) -> np.ndarray:
        if not self._negative_view_requested():
            return trace_mat
        intensity_max = self.state.intensity_max
        if intensity_max is None:
            intensity_max = self._finite_max_value(movie)
        if intensity_max is None:
            return trace_mat
        mask_sum = np.asarray(masks, dtype=float).reshape(masks.shape[0], -1).sum(axis=1, keepdims=True)
        return float(intensity_max) * mask_sum - np.asarray(trace_mat, dtype=float)

    def _get_image_trace_source_matrix(self, data: str, row: Optional[TraceControlRow]) -> Optional[tuple[np.ndarray, str, Optional[np.ndarray]]]:
        parts = data.split(':', 2)
        if len(parts) != 3:
            return None
        _prefix, layer_id, mask_source = parts
        layer = next((item for item in self.image_layers if item.layer_id == layer_id), None)
        if layer is None or not layer.visible_checkbox.isChecked():
            return None

        layer_model = self._image_layer_model(layer)
        if mask_source == 'volpy' and layer_model == 'VolPy':
            key = row.volpy_combo.currentText() if row is not None else 'ts'
            if not key:
                key = 'ts'
            vpy_data = self.ensure_volpy_loaded(show_warning=True)
            if vpy_data and key in vpy_data:
                mat = self._coerce_trace_matrix(vpy_data[key])
                if mat is not None:
                    mapping = self._ensure_volpy_suite2p_mapping(vpy_data, mat.shape[0])
                    return mat, f'{self._image_layer_label(layer)} + VolPy {key}', mapping if mapping is not None else np.asarray([], dtype=int)

        cache_key = ('image', layer.layer_id, mask_source, self.state.source_path, self.state.negative_mode, self._negative_view_requested())
        cached = self._computed_trace_cache.get(cache_key)
        if cached is not None:
            mat, mapping = cached
            return mat, f'{self._image_layer_label(layer)} + {mask_source}', mapping

        movie = self._get_image_layer_movie(layer)
        if movie is None:
            return None
        masks, mapping = self._mask_stack_for_source(mask_source, tuple(np.asarray(movie).shape[1:3]))
        if masks is None:
            return None
        cached_trace = self._load_roi_trace_cache(layer, mask_source, masks)
        if cached_trace is not None:
            mat, mapping = cached_trace
            self._computed_trace_cache[cache_key] = (mat, mapping)
            return mat, f'{self._image_layer_label(layer)} + {mask_source}', mapping
        mat = np.asarray(extract_weighted_roi_traces(movie, masks), dtype=float)
        mat = self._apply_recomputed_trace_polarity(mat, movie, masks)
        self._save_roi_trace_cache(layer, mask_source, masks, mat, mapping)
        self._computed_trace_cache[cache_key] = (mat, mapping)
        return mat, f'{self._image_layer_label(layer)} + {mask_source}', mapping

    def _get_image_layer_movie(self, row: ImageLayerControlRow) -> Optional[np.ndarray]:
        cached = self._load_image_layer_movie_cache(row)
        if cached is not None:
            return cached
        model = self._image_layer_model(row)
        arr = None
        if model == 'Raw':
            if self.state.raw_movie is not None:
                arr = np.asarray(self._display_raw_movie())
            elif self.state.tif_path and Path(self.state.tif_path).exists():
                arr = tifffile.memmap(self.state.tif_path)
                arr = np.asarray(arr)
        elif model == 'VolPy':
            arr = self._resolve_volpy_image_movie()
        elif model == 'NoRMCorre':
            arr = self._resolve_normcorre_image_movie(row)
        elif model == 'PMD':
            arr = self._resolve_pmd_image_movie(row)
        elif model == 'Local':
            arr = self._resolve_local_image_movie(row)
        if arr is not None and np.asarray(arr).ndim >= 3:
            return self._save_image_layer_movie_cache(row, arr)
        return None

    def _mask_stack_for_source(self, source: str, image_shape: tuple[int, int]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if source == 'volpy':
            if self._current_volpy_payload() is None:
                self.ensure_volpy_loaded(show_warning=False)
            masks = self._volpy_mask_stack(image_shape=image_shape)
            if masks is None:
                return None, None
            mapping = self._ensure_volpy_suite2p_mapping(self._current_volpy_payload(), masks.shape[0])
            if mapping is None:
                return None, None
            return masks, mapping

        if self.state.stat is None:
            return None, None
        ly, lx = int(image_shape[0]), int(image_shape[1])
        masks = []
        for roi in self.state.stat:
            mask = np.zeros((ly, lx), dtype=np.float32)
            ypix = np.asarray(roi['ypix'], dtype=int)
            xpix = np.asarray(roi['xpix'], dtype=int)
            valid = (ypix >= 0) & (ypix < ly) & (xpix >= 0) & (xpix < lx)
            ypix = ypix[valid]
            xpix = xpix[valid]
            if ypix.size:
                if source == 'suite2p':
                    values = np.asarray(roi.get('lam', np.ones_like(ypix, dtype=float)), dtype=np.float32)
                    values = values[:ypix.size] if values.size >= ypix.size else np.pad(values, (0, ypix.size - values.size), constant_values=1.0)
                elif source in {'weighted_pix_exp', 'weighted_pix_overmean', 'weighted_pix_max'}:
                    if self.state.raw_image is None:
                        values = np.ones_like(ypix, dtype=np.float32)
                    else:
                        roi_values = np.asarray(self.state.raw_image[ypix, xpix], dtype=float)
                        if source == 'weighted_pix_exp':
                            values = pix_exp(roi_values)
                        elif source == 'weighted_pix_overmean':
                            values = pix_overmean(roi_values)
                        else:
                            values = pix_max(roi_values)
                else:
                    return None, None
                mask[ypix, xpix] = np.asarray(values, dtype=np.float32)
            masks.append(mask)
        return np.asarray(masks, dtype=np.float32), None

    def _get_direct_trace_source_matrix(self, key: str) -> Optional[tuple[np.ndarray, str]]:
        folder = self.state.source_path if self.state.source_type == 'folder' else ''
        if not folder:
            return None
        plane_dir = Path(folder) / 'suite2p' / 'plane0'
        trace_negative = bool(self._negative_view_requested())
        weight_negative = bool(self.state.negative_mode)
        cache_suffix = (trace_negative, weight_negative, self.state.intensity_max)

        if key == 'suite2p_raw':
            cache_key = (key, trace_negative, self.state.intensity_max)
            if cache_key in self._direct_trace_cache:
                return self._direct_trace_cache[cache_key], 'raw(suite2p)'
            if not (plane_dir / 'F.npy').exists() or not (plane_dir / 'Fneu.npy').exists():
                self._set_status('raw(suite2p) requires suite2p/plane0/F.npy and Fneu.npy.')
                return None
            trace = np.asarray(F_trace(folder, alpha=0, negative=trace_negative, intensity_max=self.state.intensity_max), dtype=float)
            self._direct_trace_cache[cache_key] = trace
            return trace, 'raw(suite2p)'

        weighted_modes = {
            'weighted_pix_max': ('max', 'weighted(pix_max)'),
            'weighted_pix_exp': ('exp', 'weighted(pix_exp)'),
            'weighted_pix_overmean': ('overmean', 'weighted(pix_overmean)'),
        }
        if key in weighted_modes:
            cache_key = (key, *cache_suffix)
            if cache_key in self._direct_trace_cache:
                return self._direct_trace_cache[cache_key], weighted_modes[key][1]
            required = ['ops.npy', 'stat.npy', 'data.bin']
            missing = [name for name in required if not (plane_dir / name).exists()]
            if missing:
                self._set_status(f'{weighted_modes[key][1]} requires {", ".join(missing)}.')
                return None
            mode, label = weighted_modes[key]
            trace = np.asarray(
                weighted_trace(
                    folder,
                    mode=mode,
                    trace_negative=trace_negative,
                    weight_negative=weight_negative,
                    intensity_max=self.state.intensity_max,
                ),
                dtype=float,
            )
            self._direct_trace_cache[cache_key] = trace
            return trace, label
        self.developing(f'Trace source: {key}')
        return None

    def _coerce_trace_matrix(self, value: Any) -> Optional[np.ndarray]:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        if arr.ndim > 2:
            return arr.reshape(arr.shape[0], -1)
        return None

    def _apply_trace_pipeline(self, trace_mat: np.ndarray, row: TraceControlRow) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        parts: list[str] = []
        detail_payload: dict[str, Any] = {}
        fr = max(self.state.frame_rate, 1e-8)
        trace_raw = trace_mat.copy()
        if row.lowpass_checkbox.isChecked():
            f_high = self._safe_float(row.lowpass_edit.text(), default=1.0)
            trace_mat = lowpass_filt_trace(trace_mat, fr, f_high=min(max(f_high, 1e-6), fr * 0.45))
            parts.append(f'low-pass {f_high:g}Hz')

        if row.highpass_checkbox.isChecked():
            f_low = self._safe_float(row.highpass_edit.text(), default=10.0)
            trace_mat = highpass_filt_trace(trace_mat, fr, f_low=min(max(f_low, 1e-6), fr * 0.45))
            parts.append(f'high-pass {f_low:g}Hz')

        if row.wavelet_checkbox.isChecked():
            trace_mat = wavelet_trace(
                trace_mat,
                wavelet=row.wavelet_name_edit.text().strip() or 'sym4',
                min_level=int(self._safe_float(row.wavelet_level_edit.text(), default=4)),
                max_scale=self._safe_float(row.wavelet_scale_edit.text(), default=1.2),
                mode=row.wavelet_mode_combo.currentText(),
            )
            parts.append('wavelet')

        if row.pca_wavelet_checkbox.isChecked():
            cfg_dict = self._pca_wavelet_cfg_from_row(row)
            pca_input = trace_mat.copy()
            processed_input = self._pca_wavelet_internal_input(pca_input, fr, cfg_dict)
            trace_mat, denoise_info = pca_wavelet_trace(
                pca_input,
                fr,
                cfg=cfg_dict,
            )
            trace_mat = np.asarray(trace_mat, dtype=float)
            detail_payload['pca_wavelet'] = {
                'source_input': pca_input,
                'input': processed_input,
                'output': trace_mat.copy(),
                'denoise_info': denoise_info,
                'framerate': fr,
                'cfg': dict(cfg_dict),
                'f_min': float(cfg_dict['f_min']),
                'f_max': float(cfg_dict['f_max']),
                'f_n': int(cfg_dict['f_n']),
            }
            parts.append('pca_wavelet')

        if row.snr_checkbox.isChecked():
            trace_mat = snr_trace(
                trace_mat,
                fr,
                window=self._safe_float(row.snr_window_edit.text(), default=0.0125),
            )
            trace_mat = np.asarray(trace_mat, dtype=float)
            parts.append('snr')

        if self._trace_has_baseline(row):
            trace_mat = self._apply_baseline(trace_raw, trace_mat, row)
            parts.append('baseline')
        return trace_mat, parts, detail_payload

    def _trace_has_baseline(self, row: TraceControlRow) -> bool:
        return (
            row.baseline_lowpass_checkbox.isChecked()
            or row.baseline_rolling_checkbox.isChecked()
            or row.baseline_polyfit_checkbox.isChecked()
            or row.baseline_savgol_checkbox.isChecked()
        )

    def _apply_baseline(self, trace_raw: np.ndarray, trace_mat: np.ndarray, row: TraceControlRow) -> np.ndarray:
        fr = max(self.state.frame_rate, 1e-8)
        mode = row.baseline_mode_combo.currentData()
        if mode in {'dff', 'subtract'}:
            baseline = trace_mat.copy()
        else:
            baseline = trace_raw.copy()
        if row.baseline_lowpass_checkbox.isChecked():
            freq = self._safe_float(row.baseline_lowpass_edit.text(), default=1.0)
            baseline = lowpass_filt_trace(baseline, fr, f_high=min(max(freq, 1e-6), fr * 0.45))

        if row.baseline_rolling_checkbox.isChecked():
            window = self._window_to_frames(self._safe_float(row.baseline_rolling_window_edit.text(), default=4.0))
            if row.baseline_rolling_mode_combo.currentText() == 'median':
                baseline = median_filter(baseline, size=(1, max(window, 1)), mode='nearest')
            else:
                baseline = uniform_filter1d(baseline, size=max(window, 1), axis=1, mode='nearest')

        if row.baseline_polyfit_checkbox.isChecked():
            order = int(self._safe_float(row.baseline_poly_order_edit.text(), default=3))
            x = np.linspace(-1.0, 1.0, trace_mat.shape[1])
            poly_baseline = np.zeros_like(trace_mat, dtype=float)
            for idx in range(trace_mat.shape[0]):
                coeff = np.polyfit(x, baseline[idx], order)
                poly_baseline[idx] = np.polyval(coeff, x)
            baseline = poly_baseline
        if row.baseline_savgol_checkbox.isChecked():
            window = self._window_to_frames(self._safe_float(row.baseline_savgol_window_edit.text(), default=1.0))
            if window % 2 == 0:
                window += 1
            order = int(self._safe_float(row.baseline_savgol_order_edit.text(), default=3))
            order = min(order, max(1, window - 1))
            baseline = savgol_filter(baseline, window_length=max(window, 3), polyorder=order, axis=1, mode='nearest')

        denom = np.where(np.abs(baseline) < 1e-8, 1e-8, baseline)
        
        if mode == 'raw':
            return baseline
        if mode == 'subtract':
            return trace_mat - baseline
        if mode == 'add':
            return trace_mat + baseline
        return (trace_mat - baseline) / denom * 100.0

    def _detect_trace_spikes(
        self,
        trace_mat: np.ndarray,
        row: TraceControlRow,
        frame_indices: Optional[np.ndarray] = None,
    ):
        method = row.spike_method_combo.currentText()
        k = self._safe_float(row.spike_k_edit.text(), default=5.0)
        fr = max(self.state.frame_rate, 1e-8)
        mode = method.lower()
        if mode == 't_res':
            return self._detect_volpy_t_res_spikes(trace_mat, row, frame_indices)
        spike_times, firing_rate, thresholds, snr_values = detect_spikes(trace_mat, fr, thr=k, mode=mode)
        return trace_mat, spike_times, firing_rate, thresholds, snr_values, f'{method} k={k:g}'

    def _trace_row_volpy_key(self, row: TraceControlRow) -> Optional[str]:
        data = row.source_combo.currentData()
        if isinstance(data, str) and data.startswith('volpy:'):
            key = data.split(':', 1)[1]
            return key if key in TRACE_SOURCE_VOLPY_KEYS else None
        if row.volpy_checkbox.isChecked() or self._is_volpy_image_trace_source(data):
            key = row.volpy_combo.currentText().strip() or 'ts'
            return key if key in TRACE_SOURCE_VOLPY_KEYS else None
        return None

    def _volpy_row_values(self, value: Any, n_rows: int, default: float = np.nan) -> np.ndarray:
        out = np.full(n_rows, default, dtype=float)
        if value is None:
            return out
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            out[:] = float(arr)
            return out
        arr = arr.reshape(-1)
        count = min(n_rows, arr.size)
        if count:
            out[:count] = arr[:count]
        return out

    def _compressed_spike_frames(self, frames: Any, frame_indices: Optional[np.ndarray], n_frames: int) -> np.ndarray:
        frames_float = np.asarray(frames, dtype=float).reshape(-1)
        frames_float = frames_float[np.isfinite(frames_float)]
        frames = np.rint(frames_float).astype(int)
        if frames.size == 0:
            return np.asarray([], dtype=int)
        if frame_indices is None:
            return np.unique(frames[(frames >= 0) & (frames < n_frames)]).astype(int)
        original = np.asarray(frame_indices, dtype=int).reshape(-1)
        if original.size == 0:
            return np.asarray([], dtype=int)
        frames = np.unique(frames[(frames >= int(np.min(original))) & (frames <= int(np.max(original)))])
        if frames.size == 0:
            return np.asarray([], dtype=int)
        pos = np.searchsorted(original, frames)
        in_bounds = (pos >= 0) & (pos < original.size)
        valid = np.zeros(pos.shape, dtype=bool)
        valid[in_bounds] = original[pos[in_bounds]] == frames[in_bounds]
        return pos[valid].astype(int)

    def _empty_spike_detection_result(self, trace_mat: np.ndarray, label: str):
        n_rows, n_frames = trace_mat.shape
        spike_times = [np.asarray([], dtype=float) for _ in range(n_rows)]
        firing_rate = np.zeros((n_rows, n_frames), dtype=float)
        thresholds = np.full(n_rows, np.nan, dtype=float)
        snr_values = [None for _ in range(n_rows)]
        return trace_mat, spike_times, firing_rate, thresholds, snr_values, label

    def _detect_volpy_t_res_spikes(
        self,
        trace_mat: np.ndarray,
        row: TraceControlRow,
        frame_indices: Optional[np.ndarray],
    ):
        trace_mat = np.asarray(trace_mat, dtype=float)
        key = self._trace_row_volpy_key(row)
        if key is None:
            self._set_status('t_res spike detection requires a VolPy trace source.')
            return self._empty_spike_detection_result(trace_mat, 't_res unavailable')
        vpy_data = self.ensure_volpy_loaded(show_warning=True)
        if not isinstance(vpy_data, dict) or 'spikes' not in vpy_data:
            self._set_status("t_res spike detection requires vpy['spikes'].")
            return self._empty_spike_detection_result(trace_mat, 't_res unavailable')

        spikes_raw = vpy_data.get('spikes')
        spikes_source = list(spikes_raw) if spikes_raw is not None else []
        n_rows, n_frames = trace_mat.shape
        fr = max(self.state.frame_rate, 1e-8)
        spike_matrix = np.zeros((n_rows, n_frames), dtype=float)
        spike_times: list[np.ndarray] = []
        snr_from_vpy = self._volpy_row_values(vpy_data.get('snr'), n_rows, default=np.nan) if 'snr' in vpy_data else None
        snr_values: list[Optional[float]] = []

        for row_idx in range(n_rows):
            frames = spikes_source[row_idx] if row_idx < len(spikes_source) else []
            compressed = self._compressed_spike_frames(frames, frame_indices, n_frames)
            if compressed.size:
                spike_matrix[row_idx, compressed] = 1.0
            spike_times.append(compressed.astype(float) / fr)
            if snr_from_vpy is not None and row_idx < snr_from_vpy.size and np.isfinite(snr_from_vpy[row_idx]):
                snr_values.append(float(snr_from_vpy[row_idx]))
            else:
                value = compute_volpy_snr(trace_mat[row_idx], compressed)
                snr_values.append(float(value) if np.isfinite(value) else None)

        firing_rate = generate_firingRate(spike_matrix, fr)
        thresholds = np.full(n_rows, np.nan, dtype=float)
        if key == 'ts':
            thresholds = self._volpy_row_values(vpy_data.get('thresh'), n_rows, default=np.nan)
        return trace_mat, spike_times, firing_rate, thresholds, snr_values, 't_res VolPy'

    def _parse_params(self, text: str) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for part in text.split(','):
            if '=' not in part:
                continue
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            try:
                if value.lower() in {'true', 'false'}:
                    params[key] = value.lower() == 'true'
                elif any(ch in value for ch in ['.', 'e', 'E']):
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except Exception:
                params[key] = value
        return params

    def _window_to_frames(self, value: float) -> int:
        if value <= 0:
            return 1
        if value <= 20:
            return max(1, int(round(value * max(self.state.frame_rate, 1.0))))
        return max(1, int(round(value)))

    def _offset_step_for_lines(self, lines: List[np.ndarray]) -> float:
        if not lines:
            return 0.0
        std_values = [float(np.nanstd(np.asarray(line, dtype=float))) for line in lines]
        finite = [value for value in std_values if np.isfinite(value) and value > 0]
        if not finite:
            return 0.0
        return 2.0 * float(np.mean(finite))

    def _offset_step_for_rois(self, trace_mat: np.ndarray, roi_indices: List[int]) -> float:
        lines = [np.asarray(trace_mat[idx], dtype=float) for idx in roi_indices if idx < trace_mat.shape[0]]
        return self._offset_step_for_lines(lines)

    def _result_roi_row_pairs(self, result: dict[str, Any], roi_indices: List[int]) -> list[tuple[int, int]]:
        trace_mat = result['data']
        n_rows = trace_mat.shape[0]
        mapping = result.get('roi_indices')
        pairs: list[tuple[int, int]] = []
        if mapping is None:
            for roi_idx in roi_indices:
                if 0 <= roi_idx < n_rows and np.any(np.isfinite(trace_mat[roi_idx])):
                    pairs.append((int(roi_idx), int(roi_idx)))
            return pairs

        mapping = np.asarray(mapping, dtype=int)
        for roi_idx in roi_indices:
            hits = np.where(mapping == int(roi_idx))[0]
            if hits.size == 0:
                continue
            row_idx = int(hits[0])
            if 0 <= row_idx < n_rows and np.any(np.isfinite(trace_mat[row_idx])):
                pairs.append((int(roi_idx), row_idx))
        return pairs

    def _offset_step_for_pairs(self, trace_mat: np.ndarray, pairs: list[tuple[int, int]]) -> float:
        lines = [np.asarray(trace_mat[row_idx], dtype=float) for _roi_idx, row_idx in pairs if row_idx < trace_mat.shape[0]]
        return self._offset_step_for_lines(lines)

    def _render_trace_panel(self, reset_view: bool = False):
        fig = self.trace_widget.figure
        fig.clear()
        if not hasattr(self, 'trace_widget'):
            return
        results = self._get_active_trace_results()
        if not results:
            self.trace_widget.set_placeholder('Trace panel')
            return

        selected_rois = self.get_selected_roi_indices()
        if not selected_rois:
            self.trace_widget.set_placeholder('Select at least one ROI.')
            return

        nrows = len(results)
        show_waveform_column = any(result['waveform'] for result in results)
        waveform_columns = 2 if any(result['waveform'] and result['row'].waveform_mode_combo.currentText() == 'both' for result in results) else 1
        if show_waveform_column:
            grid = fig.add_gridspec(nrows, 1 + waveform_columns, width_ratios=[5] + [1] * waveform_columns, hspace=0.18, wspace=0.04)
        else:
            grid = fig.add_gridspec(nrows, 1, hspace=0.18)
        trace_axes = []
        waveform_axes = []
        for idx in range(nrows):
            if show_waveform_column:
                ax = fig.add_subplot(grid[idx, 0], sharex=trace_axes[0] if trace_axes else None)
                wax = [fig.add_subplot(grid[idx, col + 1]) for col in range(waveform_columns)]
            else:
                ax = fig.add_subplot(grid[idx, 0], sharex=trace_axes[0] if trace_axes else None)
                wax = []
            trace_axes.append(ax)
            waveform_axes.append(wax)

        window = self._safe_float(self.trace_window_edit.text(), default=5.0)
        max_frames = max(result['data'].shape[1] for result in results)
        x_total = self._trace_x_total(max_frames)
        x_window = min(self._trace_window_width(window), x_total)
        self.trace_widget.set_x_window(x_total, x_window, x_start=0.0 if reset_view else None)
        self.trace_widget.set_x_axes(trace_axes)
        x_left = self.trace_widget.get_x_start()
        x_right = min(x_left + x_window, x_total)

        for idx, (ax, wax_list, result) in enumerate(zip(trace_axes, waveform_axes, results)):
            self._plot_trace_result(ax, result, selected_rois)
            self._draw_event_spans(ax, result.get('frame_indices'))
            ax.set_xlim(x_left, x_right)
            ylabel = self._trace_result_y_label(result)
            xlabel = self._trace_x_label() if idx == len(trace_axes) - 1 else ''
            self._apply_axis_labels(ax, self.trace_labels_checkbox.isChecked(), xlabel, ylabel, result['name'])
            if idx < len(trace_axes) - 1:
                ax.tick_params(labelbottom=False)
            if result['waveform'] and wax_list:
                mode = result['row'].waveform_mode_combo.currentText()
                modes = ['raw', 'current'] if mode == 'both' else [mode]
                for wax, wave_mode in zip(wax_list, modes):
                    self._plot_inline_waveform(wax, result, selected_rois, source_mode=wave_mode)
                for wax in wax_list[len(modes):]:
                    wax.axis('off')
            else:
                for wax in wax_list:
                    wax.axis('off')
        self.trace_widget.canvas.setMinimumHeight(max(300, 180 * len(results)))
        self._sync_trace_frame_slider()
        fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.08, hspace=0.18, wspace=0.04)
        self.trace_widget.canvas.draw_idle()

    def _trace_result_y_label(self, result: dict[str, Any]) -> str:
        if not result.get('baseline'):
            return 'Intensity'
        mode = result.get('baseline_mode', 'dff')
        if mode == 'raw':
            return 'Baseline'
        if mode == 'subtract':
            return 'Intensity - baseline'
        if mode == 'add':
            return 'Intensity + baseline'
        return 'dF/F'

    def _draw_trace_metric_values(self, ax, result: dict[str, Any], roi_row_pairs: list[tuple[int, int]]):
        if not self.trace_labels_checkbox.isChecked():
            return
        snr_values = result.get('snr_values')
        spike_times = result.get('spike_times')
        duration = np.asarray(result.get('data'), dtype=float).shape[1] / max(self.state.frame_rate, 1e-8)
        for line_idx, (roi_idx, row_idx) in enumerate(roi_row_pairs):
            parts = [f'ROI {roi_idx:03d}']
            if snr_values is not None and row_idx < len(snr_values):
                value = snr_values[row_idx]
                if value is None or not np.isfinite(float(value)):
                    parts.append('snr=None')
                else:
                    parts.append(f'snr={float(value):.2f}')
            if spike_times is not None and row_idx < len(spike_times) and duration > 0:
                n_spikes = np.asarray(spike_times[row_idx], dtype=float).size
                parts.append(f'fr={n_spikes / duration:.2f} Hz')
            text = ' '.join(parts)
            ax.text(
                0.01,
                0.98 - line_idx * 0.075,
                text,
                color=self._roi_display_color(roi_idx),
                fontsize=8,
                ha='left',
                va='top',
                transform=ax.transAxes,
                zorder=10,
            )

    def _plot_trace_result(self, ax, result: dict[str, Any], roi_indices: List[int]):
        trace_mat = result['data']
        t = self._trace_x_values(trace_mat.shape[1])
        stride = self._trace_plot_stride(trace_mat.shape[1])
        plot_idx = slice(None, None, stride)
        t_plot = t[plot_idx]
        combine_mode = self.combine_mode
        roi_row_pairs = self._result_roi_row_pairs(result, roi_indices)
        if not roi_row_pairs:
            ax.text(0.5, 0.5, 'Selected ROI is outside this trace source.', ha='center', va='center', transform=ax.transAxes)
            return

        if combine_mode == 'individual':
            offset_step = self._offset_step_for_pairs(trace_mat, roi_row_pairs) if len(roi_row_pairs) > 1 else 0.0
            offsets = {roi_idx: offset_idx * offset_step for offset_idx, (roi_idx, _row_idx) in enumerate(roi_row_pairs)}
            spike_marker_payload = []
            for offset_idx, (roi_idx, row_idx) in enumerate(roi_row_pairs):
                color = self._roi_display_color(roi_idx)
                offset = offsets[roi_idx]
                y = np.asarray(trace_mat[row_idx], dtype=float)
                y_plot = y + offset
                ax.plot(t_plot, y_plot[plot_idx], color=color, lw=1.0, label=self._roi_label_text(roi_idx))
                thresholds = result.get('thresholds')
                show_threshold = result['row'].threshold_checkbox.isChecked()
                if show_threshold and thresholds is not None and row_idx < len(thresholds):
                    thr = thresholds[row_idx]
                    ax.plot(t_plot, np.full(t_plot.shape, thr + offset), color=color, lw=0.8, ls='--', alpha=0.8)
                spike_times = result.get('spike_times')
                if spike_times is not None and row_idx < len(spike_times):
                    st = np.atleast_1d(self._trace_time_to_x(spike_times[row_idx]))
                    if st.size:
                        spike_marker_payload.append((roi_idx, st, color))
            if spike_marker_payload:
                marker_levels, marker_top = self._spike_marker_levels_for_pairs(trace_mat, roi_row_pairs, offsets)
                for roi_idx, st, color in spike_marker_payload:
                    marker_y = marker_levels.get(roi_idx)
                    if marker_y is not None:
                        ax.scatter(st, np.full(st.shape, marker_y), marker='|', s=46, color=color, linewidths=1.2)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, max(ymax, marker_top))
            if len(roi_row_pairs) > 1:
                ax.legend(loc='upper right', fontsize=8, ncol=min(3, len(roi_row_pairs)))
            self._draw_trace_metric_values(ax, result, roi_row_pairs)
        else:
            data = np.vstack([trace_mat[row_idx] for _roi_idx, row_idx in roi_row_pairs])
            y = data.mean(axis=0) if combine_mode == 'mean' else data.sum(axis=0)
            label = 'Average of selected ROIs' if combine_mode == 'mean' else 'Sum of selected ROIs'
            color = 'C0'
            ax.plot(t_plot, y[plot_idx], color=color, lw=1.4, label=label)
            spike_times = result.get('spike_times')
            if spike_times is not None:
                combined_spikes = []
                for _roi_idx, row_idx in roi_row_pairs:
                    if row_idx < len(spike_times):
                        combined_spikes.extend(np.asarray(spike_times[row_idx], dtype=float).tolist())
                if combined_spikes:
                    st = np.asarray(sorted(set(combined_spikes)), dtype=float)
                    sx = np.atleast_1d(self._trace_time_to_x(st))
                    marker_y, marker_top = self._spike_marker_level_for_line(y)
                    ax.scatter(sx, np.full(sx.shape, marker_y), marker='|', s=46, color=color, linewidths=1.2)
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim(ymin, max(ymax, marker_top))
            ax.legend(loc='upper right', fontsize=8)
            self._draw_trace_metric_values(ax, result, roi_row_pairs)

    def _average_trace_std(self, trace_mat: np.ndarray, roi_indices: List[int]) -> float:
        values = [float(np.nanstd(trace_mat[idx])) for idx in roi_indices if idx < trace_mat.shape[0]]
        finite = [value for value in values if np.isfinite(value) and value > 0]
        return float(np.mean(finite)) if finite else 0.0

    def _spike_marker_levels(self, trace_mat: np.ndarray, roi_indices: List[int], offsets: dict[int, float]):
        tops = []
        bottoms = []
        for roi_idx in roi_indices:
            if roi_idx >= trace_mat.shape[0]:
                continue
            y = np.asarray(trace_mat[roi_idx], dtype=float) + offsets.get(roi_idx, 0.0)
            finite = y[np.isfinite(y)]
            if finite.size:
                tops.append(float(np.max(finite)))
                bottoms.append(float(np.min(finite)))
        if not tops:
            return {roi_idx: float(i + 1) for i, roi_idx in enumerate(roi_indices)}, float(len(roi_indices) + 1)
        y_top = max(tops)
        span = max(y_top - min(bottoms), self._average_trace_std(trace_mat, roi_indices), 1e-6)
        step = max(0.06 * span, 0.5 * self._average_trace_std(trace_mat, roi_indices), 1e-6)
        levels = {roi_idx: y_top + (idx + 1) * step for idx, roi_idx in enumerate(roi_indices)}
        return levels, y_top + (len(roi_indices) + 1) * step

    def _spike_marker_levels_for_pairs(self, trace_mat: np.ndarray, roi_row_pairs: list[tuple[int, int]], offsets: dict[int, float]):
        tops = []
        bottoms = []
        for roi_idx, row_idx in roi_row_pairs:
            if row_idx >= trace_mat.shape[0]:
                continue
            y = np.asarray(trace_mat[row_idx], dtype=float) + offsets.get(roi_idx, 0.0)
            finite = y[np.isfinite(y)]
            if finite.size:
                tops.append(float(np.max(finite)))
                bottoms.append(float(np.min(finite)))
        if not tops:
            return {roi_idx: float(i + 1) for i, (roi_idx, _row_idx) in enumerate(roi_row_pairs)}, float(len(roi_row_pairs) + 1)
        row_indices = [row_idx for _roi_idx, row_idx in roi_row_pairs]
        y_top = max(tops)
        span = max(y_top - min(bottoms), self._average_trace_std(trace_mat, row_indices), 1e-6)
        step = max(0.06 * span, 0.5 * self._average_trace_std(trace_mat, row_indices), 1e-6)
        levels = {roi_idx: y_top + (idx + 1) * step for idx, (roi_idx, _row_idx) in enumerate(roi_row_pairs)}
        return levels, y_top + (len(roi_row_pairs) + 1) * step

    def _spike_marker_level_for_line(self, line: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(line, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 1.0, 2.0
        y_top = float(np.max(finite))
        span = max(y_top - float(np.min(finite)), float(np.nanstd(finite)), 1e-6)
        step = max(0.06 * span, 0.5 * float(np.nanstd(finite)), 1e-6)
        return y_top + step, y_top + 2.0 * step

    def _plot_inline_waveform(self, ax, result: dict[str, Any], roi_indices: List[int], source_mode: str = 'current'):
        spike_times = result.get('spike_times')
        if spike_times is None:
            ax.axis('off')
            return
        pre = self._safe_float(self.waveform_pre_edit.text(), default=-0.025)
        post = self._safe_float(self.waveform_post_edit.text(), default=0.025)
        trace_key = 'raw_data' if source_mode == 'raw' else 'data'
        plotted = self._plot_event_waveforms_gray_red(
            ax,
            trace_mat=result.get(trace_key, result['data']),
            roi_indices=roi_indices,
            onset_list=spike_times,
            duration=(pre, post),
            roi_row_pairs=self._result_roi_row_pairs(result, roi_indices),
            show_peak_features=self.trace_labels_checkbox.isChecked(),
        )
        if not plotted:
            ax.text(0.5, 0.5, 'no spikes', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        ax.axvline(0, color='red', alpha=0.4, lw=0.8)
        self._apply_axis_labels(ax, self.trace_labels_checkbox.isChecked(), '', '', source_mode)
        ax.tick_params(labelsize=8)

    def _plot_event_waveforms_gray_red(
        self,
        ax,
        trace_mat,
        roi_indices,
        onset_list,
        duration,
        show_individual: bool = True,
        roi_row_pairs: Optional[list[tuple[int, int]]] = None,
        show_peak_features: bool = False,
    ) -> bool:
        groups = []
        x_axis = None
        if roi_row_pairs is None:
            roi_row_pairs = [(int(idx), int(idx)) for idx in roi_indices]
        for roi_idx, row_idx in roi_row_pairs:
            if row_idx >= trace_mat.shape[0] or row_idx >= len(onset_list):
                continue
            if not np.any(np.isfinite(trace_mat[row_idx])):
                continue
            events = np.asarray(onset_list[row_idx], dtype=float)
            if events.size == 0:
                continue
            trial_mat, x_axis = self._extract_event_matrix(trace_mat[row_idx], events, duration)
            if trial_mat is not None and trial_mat.size:
                groups.append((roi_idx, trial_mat))
        if not groups or x_axis is None:
            return False
        mean_lines = [np.nanmean(group, axis=0) for _roi_idx, group in groups]
        combine_mode = self.combine_mode if self.combine_mode in COMBINE_MODES else 'individual'
        if combine_mode != 'individual':
            if show_individual:
                for feature_idx, ((roi_idx, group), line) in enumerate(zip(groups, mean_lines)):
                    ax.plot(x_axis, line, color=self._roi_display_color(roi_idx), lw=0.9, alpha=0.35)
                    if show_peak_features:
                        self._draw_waveform_main_peak_features(ax, x_axis, group, 0.0, roi_idx, feature_idx)
            stacked = np.vstack(mean_lines)
            if combine_mode == 'sum':
                y = np.nansum(stacked, axis=0)
                label = 'Sum of selected ROI waveforms'
            else:
                y = np.nanmean(stacked, axis=0)
                label = 'Mean of selected ROI waveforms'
            ax.plot(x_axis, y, color='red', lw=2.8, label=label)
            ax.legend(loc='upper right', fontsize=8)
            return True
        offset_step = self._offset_step_for_lines(mean_lines) if len(groups) > 1 else 0.0
        for offset_idx, (roi_idx, group) in enumerate(groups):
            color = self._roi_display_color(roi_idx)
            offset = offset_idx * offset_step
            if show_individual:
                for row in group:
                    ax.plot(x_axis, row + offset, color=color, lw=0.7, alpha=0.28)
            ax.plot(x_axis, np.nanmean(group, axis=0) + offset, color=color, lw=2.6, label=self._roi_label_text(roi_idx))
            if show_peak_features:
                self._draw_waveform_main_peak_features(ax, x_axis, group, offset, roi_idx, offset_idx)
        if len(groups) > 1:
            ax.legend(loc='upper right', fontsize=8)
        return True

    def _draw_waveform_main_peak_features(
        self,
        ax,
        x_axis: np.ndarray,
        spike_windows: np.ndarray,
        offset: float,
        roi_idx: int,
        label_idx: int,
    ):
        if x_axis is None or len(x_axis) == 0:
            return
        peak_index = int(np.argmin(np.abs(np.asarray(x_axis, dtype=float))))
        try:
            summary = cal_waveform.quantify_average_peak_waveform(
                spike_windows,
                max(self.state.frame_rate, 1e-8),
                peak_index=peak_index,
                spike_index=roi_idx,
            )
        except Exception:
            return
        peak_time = summary.get('peak_time_ms', np.nan)
        peak_value = summary.get('peak_value', np.nan)
        if np.isfinite(peak_time) and np.isfinite(peak_value):
            ax.scatter(
                [float(peak_time) / 1000.0],
                [float(peak_value) + offset],
                s=24,
                color='red',
                edgecolors='none',
                zorder=12,
            )
        fit_t = np.asarray(summary.get('fit_t_ms', []), dtype=float) / 1000.0
        fit_y = np.asarray(summary.get('fit_waveform', []), dtype=float)
        features = summary.get('features', {})
        if bool(features.get('fit_success')) and fit_t.size and fit_y.size:
            ax.plot(fit_t, fit_y + offset, color='black', lw=2.4, zorder=11)
        fwhm = features.get('fwhm_ms', np.nan)
        try:
            fwhm_value = float(fwhm)
        except Exception:
            fwhm_value = np.nan
        text = f'ROI {roi_idx:03d} fwhm={fwhm_value:.2f} ms' if np.isfinite(fwhm_value) else f'ROI {roi_idx:03d} fwhm=nan'
        ax.text(
            0.02,
            0.96 - label_idx * 0.10,
            text,
            color=self._roi_display_color(roi_idx),
            fontsize=8,
            ha='left',
            va='top',
            transform=ax.transAxes,
            zorder=12,
        )

    def _render_average_panel(self, reset_view: bool = False):
        if not hasattr(self, 'avg_widget'):
            return
        fig = self.avg_widget.figure
        fig.clear()
        self._clear_average_export_buttons()
        results = self._get_active_trace_results()
        if not results:
            self.avg_widget.set_placeholder('Average panel')
            return
        self._refresh_avg_trace_list(results)
        selected_result_indices = self.get_selected_average_trace_indices(len(results))
        results = [result for idx, result in enumerate(results) if idx in selected_result_indices]
        if not results:
            self.avg_widget.set_placeholder('Select at least one average trace.')
            return

        selected_rois = self.get_selected_roi_indices()
        if not selected_rois:
            self.avg_widget.set_placeholder('Select at least one ROI.')
            return

        modes = self._active_avg_modes()
        plot_jobs: list[tuple[str, dict[str, Any], Optional[int]]] = []
        for mode in modes:
            if mode == 'event':
                event_indices = self._active_event_indices()
                plot_jobs.extend(
                    (mode, result, event_idx)
                    for result in results
                    for event_idx in event_indices
                    if 0 <= event_idx < len(self.events)
                )
            else:
                plot_jobs.extend((mode, result, None) for result in results)
        if not plot_jobs:
            self.avg_widget.set_placeholder('Select at least one event or average mode.')
            return
        nrows = len(plot_jobs)
        grid = fig.add_gridspec(nrows, 1, hspace=0.0)
        axes = [fig.add_subplot(grid[idx, 0]) for idx in range(nrows)]

        x_bounds: List[tuple[float, float]] = []
        for ax, (mode, result, event_idx) in zip(axes, plot_jobs):
            if mode == 'event':
                bounds = self._plot_average_event_result(ax, result, selected_rois, int(event_idx or 0))
            elif mode == 'waveform':
                bounds = self._plot_average_waveform_result(ax, result, selected_rois)
            elif mode == 'firing_rate':
                bounds = self._plot_average_firing_rate_result(ax, result, selected_rois)
            else:
                ax.text(0.5, 0.5, 'Unknown average mode', ha='center', va='center')
                ax.axis('off')
                bounds = None
            if bounds is not None:
                x_bounds.append(bounds)

        if x_bounds:
            self.avg_widget.set_independent_x_axes(axes)
        self._avg_subplot_axes = axes
        self._avg_subplot_export_names = [
            self._average_subplot_export_name(idx, mode, result, event_idx)
            for idx, (mode, result, event_idx) in enumerate(plot_jobs)
        ]
        self._build_average_export_buttons()
        self.avg_widget.canvas.setMinimumHeight(max(260, 250 * len(plot_jobs)))
        fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.08, hspace=0.22, wspace=0.0)
        self.avg_widget.canvas.draw_idle()

    def _clear_average_export_buttons(self):
        if not hasattr(self, 'avg_export_layout'):
            return
        while self.avg_export_layout.count() > 1:
            item = self.avg_export_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_average_export_buttons(self):
        if not hasattr(self, 'avg_export_layout'):
            return
        self._clear_average_export_buttons()
        for idx, _axis in enumerate(self._avg_subplot_axes):
            button = QPushButton(f'Export {idx + 1}')
            button.setToolTip('Export this average subplot as TIFF.')
            button.clicked.connect(lambda _checked=False, i=idx: self.export_average_subplot_callback(i))
            self.avg_export_layout.insertWidget(idx, button)

    def _average_subplot_export_name(self, idx: int, mode: str, result: dict[str, Any], event_idx: Optional[int]) -> str:
        label = mode
        if mode == 'event' and event_idx is not None and 0 <= event_idx < len(self.events):
            label = f"event_{self.events[event_idx].get('label', event_idx + 1)}"
        clean = ''.join(ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in str(label))
        return f'average_{idx + 1}_{clean}_{self._database_name()}.tiff'

    def export_average_subplot_callback(self, index: int):
        if index < 0 or index >= len(self._avg_subplot_axes):
            self._set_status('No average subplot is available for export.')
            return
        folder = self._pipeline_data_folder() or Path.cwd()
        default_name = self._avg_subplot_export_names[index] if index < len(self._avg_subplot_export_names) else f'average_{index + 1}_{self._database_name()}.tiff'
        out, _ = QFileDialog.getSaveFileName(
            self,
            'Export average subplot',
            str(folder / default_name),
            'TIFF (*.tiff *.tif);;All files (*.*)',
        )
        if not out:
            return
        fig = self.avg_widget.figure
        ax = self._avg_subplot_axes[index]
        fig.canvas.draw()
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.04, 1.12)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out, dpi=300, bbox_inches=bbox_inches)
        self._set_status(f'Exported average subplot: {out}')

    def _plot_average_event_result(self, ax, result: dict[str, Any], roi_indices: List[int], event_idx: int) -> Optional[tuple[float, float]]:
        if event_idx < 0 or event_idx >= len(self.events):
            ax.text(0.5, 0.5, 'No event selected.', ha='center', va='center')
            ax.axis('off')
            return None
        event = self.events[event_idx]
        ranges = self._event_frame_ranges(event)
        if not ranges:
            ax.text(0.5, 0.5, 'Selected event has no frame ranges.', ha='center', va='center')
            ax.axis('off')
            return None
        fr = max(float(self.state.frame_rate), 1.0)
        frame_indices = result.get('frame_indices')
        if frame_indices is not None:
            compressed = []
            frame_indices = np.asarray(frame_indices, dtype=int)
            for start, stop in ranges:
                hits = np.where((frame_indices >= start) & (frame_indices < stop))[0]
                if hits.size:
                    compressed.append((int(hits[0]), int(hits[-1]) + 1))
            ranges_for_trace = compressed
        else:
            ranges_for_trace = ranges
        if not ranges_for_trace:
            ax.text(0.5, 0.5, 'Selected event frames were discarded.', ha='center', va='center')
            ax.axis('off')
            return None
        onset_times = np.asarray([start / fr for start, _stop in ranges_for_trace], dtype=float)
        offset_times = np.asarray([stop / fr for _start, stop in ranges_for_trace], dtype=float)
        pre = self._safe_float(self.avg_pre_edit.text(), default=-0.2)
        post = self._safe_float(self.avg_post_edit.text(), default=0.5)
        x_axis, trial_groups = self._collect_trial_trace_groups_from_matrix(
            trace_mat=result['data'],
            roi_indices=roi_indices,
            onset_times=onset_times,
            offset_times=offset_times,
            duration=(pre, post),
            roi_row_pairs=self._result_roi_row_pairs(result, roi_indices),
        )
        if x_axis is None or not trial_groups:
            ax.text(0.5, 0.5, 'Cannot extract event windows.', ha='center', va='center')
            ax.axis('off')
            return None
        self._draw_average_groups(ax, x_axis, trial_groups)
        event_duration = float(np.median(offset_times - onset_times))
        color = str(event.get('color') or '#ff0000')
        ax.axvspan(0, event_duration, color=color, alpha=0.25, linewidth=0, zorder=0)
        if self.avg_labels_checkbox.isChecked() and event_duration > 0:
            ax.text(0, 0.98, str(event.get('label') or 'event'), color=color, fontsize=8, ha='left', va='top', transform=ax.get_xaxis_transform())
        title = f"{result['name']} | {event.get('label', 'event')}"
        self._apply_axis_labels(ax, self.avg_labels_checkbox.isChecked(), 'Time (s)', 'Intensity', title)
        return float(x_axis[0]), float(x_axis[-1])

    def _plot_average_waveform_result(self, ax, result: dict[str, Any], roi_indices: List[int]) -> Optional[tuple[float, float]]:
        spike_times = result.get('spike_times')
        if spike_times is None:
            ax.text(0.5, 0.5, 'Enable spike detection for waveform mode.', ha='center', va='center')
            ax.axis('off')
            return None
        pre = self._safe_float(self.waveform_pre_edit.text(), default=-0.025)
        post = self._safe_float(self.waveform_post_edit.text(), default=0.025)
        plotted = self._plot_event_waveforms_gray_red(
            ax,
            trace_mat=result['data'],
            roi_indices=roi_indices,
            onset_list=spike_times,
            duration=(pre, post),
            show_individual=self._avg_show_individual_traces(),
            roi_row_pairs=self._result_roi_row_pairs(result, roi_indices),
        )
        if not plotted:
            ax.text(0.5, 0.5, 'No spikes available for waveform averaging.', ha='center', va='center')
            ax.axis('off')
            return None
        ax.axvline(0, color='red', alpha=0.4, lw=1.0)
        self._apply_axis_labels(ax, self.avg_labels_checkbox.isChecked(), 'Time (s)', 'Intensity', result['name'])
        x0, x1 = ax.get_xlim()
        return float(x0), float(x1)

    def _plot_average_firing_rate_result(self, ax, result: dict[str, Any], roi_indices: List[int]) -> Optional[tuple[float, float]]:
        if self.state.onset_times_trial is None or self.state.offset_times_trial is None:
            ax.text(0.5, 0.5, 'No stimulus timing in current source.', ha='center', va='center')
            ax.axis('off')
            return None
        firing_rate = result.get('firing_rate')
        if firing_rate is None:
            ax.text(0.5, 0.5, 'Enable spike detection for firing-rate mode.', ha='center', va='center')
            ax.axis('off')
            return None
        pre = self._safe_float(self.avg_pre_edit.text(), default=-0.2)
        post = self._safe_float(self.avg_post_edit.text(), default=0.5)
        x_axis, trial_groups = self._collect_trial_trace_groups_from_matrix(
            trace_mat=firing_rate,
            roi_indices=roi_indices,
            onset_times=self.state.onset_times_trial,
            offset_times=self.state.offset_times_trial,
            duration=(pre, post),
            roi_row_pairs=self._result_roi_row_pairs(result, roi_indices),
        )
        if x_axis is None or not trial_groups:
            ax.text(0.5, 0.5, 'Cannot extract firing-rate windows.', ha='center', va='center')
            ax.axis('off')
            return None
        self._draw_average_groups(ax, x_axis, trial_groups)
        trial_duration = float(np.median(self.state.offset_times_trial - self.state.onset_times_trial))
        ax.axvspan(0, trial_duration, color='red', alpha=0.25, linewidth=0)
        self._apply_axis_labels(ax, self.avg_labels_checkbox.isChecked(), 'Time (s)', 'Rate (Hz)', result['name'])
        return float(x_axis[0]), float(x_axis[-1])

    def _plot_average_stimulus(self, ax, pidx: int, roi_indices: List[int]) -> Optional[tuple[float, float]]:
        if self.state.onset_times_trial is None or self.state.offset_times_trial is None:
            ax.text(0.5, 0.5, 'No stimulus timing in current source.', ha='center', va='center')
            ax.axis('off')
            return None

        pre = self._safe_float(self.avg_pre_edit.text(), default=-0.2)
        post = self._safe_float(self.avg_post_edit.text(), default=0.5)
        x_axis, roi_trial_means = self._collect_trial_means(
            signal_list=self.state.traces,
            pidx=pidx,
            roi_indices=roi_indices,
            onset_times=self.state.onset_times_trial,
            offset_times=self.state.offset_times_trial,
            duration=(pre, post),
        )
        if x_axis is None or not roi_trial_means:
            ax.text(0.5, 0.5, 'Cannot extract stimulus windows.', ha='center', va='center')
            ax.axis('off')
            return None

        self._draw_average_lines(ax, x_axis, roi_trial_means)
        trial_duration = float(np.median(self.state.offset_times_trial - self.state.onset_times_trial))
        ax.axvspan(0, trial_duration, color='red', alpha=0.25, linewidth=0)
        ax.set_title('')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Intensity')
        return float(x_axis[0]), float(x_axis[-1])

    def _plot_average_waveform(self, ax, pidx: int, roi_indices: List[int]) -> Optional[tuple[float, float]]:
        if pidx >= len(self.state.spike_times):
            ax.text(0.5, 0.5, 'No spike_times for waveform mode.', ha='center', va='center')
            ax.axis('off')
            return None

        pre = self._safe_float(self.waveform_pre_edit.text(), default=-0.025)
        post = self._safe_float(self.waveform_post_edit.text(), default=0.025)
        onset_list = self.state.spike_times[pidx]
        offset_list = self.state.spike_times[pidx]
        trial_dur = max(post - pre, 1e-3)
        x_axis, roi_trial_means = self._collect_event_means(
            trace_mat=self.state.traces[pidx],
            roi_indices=roi_indices,
            onset_list=onset_list,
            offset_list=offset_list,
            event_duration=trial_dur,
            duration=(pre, post),
        )
        if x_axis is None or not roi_trial_means:
            ax.text(0.5, 0.5, 'No spikes available for waveform averaging.', ha='center', va='center')
            ax.axis('off')
            return None

        self._draw_average_lines(ax, x_axis, roi_trial_means)
        ax.axvline(0, color='red', alpha=0.4, lw=1.0)
        ax.set_title('')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Intensity')
        return float(x_axis[0]), float(x_axis[-1])

    def _plot_average_firing_rate(self, ax, pidx: int, roi_indices: List[int]) -> Optional[tuple[float, float]]:
        if self.state.onset_times_trial is None or self.state.offset_times_trial is None:
            ax.text(0.5, 0.5, 'No stimulus timing in current source.', ha='center', va='center')
            ax.axis('off')
            return None
        if pidx >= len(self.state.firing_rate):
            ax.text(0.5, 0.5, 'No firing rate for selected process.', ha='center', va='center')
            ax.axis('off')
            return None

        pre = self._safe_float(self.avg_pre_edit.text(), default=-0.2)
        post = self._safe_float(self.avg_post_edit.text(), default=0.5)
        x_axis, roi_trial_means = self._collect_trial_means(
            signal_list=self.state.firing_rate,
            pidx=pidx,
            roi_indices=roi_indices,
            onset_times=self.state.onset_times_trial,
            offset_times=self.state.offset_times_trial,
            duration=(pre, post),
        )
        if x_axis is None or not roi_trial_means:
            ax.text(0.5, 0.5, 'Cannot extract firing-rate windows.', ha='center', va='center')
            ax.axis('off')
            return None

        self._draw_average_lines(ax, x_axis, roi_trial_means)
        trial_duration = float(np.median(self.state.offset_times_trial - self.state.onset_times_trial))
        ax.axvspan(0, trial_duration, color='red', alpha=0.25, linewidth=0)
        ax.set_title('')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rate (Hz)')
        return float(x_axis[0]), float(x_axis[-1])

    def _draw_average_lines(self, ax, x_axis: np.ndarray, traces: List[np.ndarray]):
        if not traces:
            return
        stacked = np.vstack([np.asarray(line, dtype=float) for line in traces])
        if self._avg_show_individual_traces():
            for line in stacked:
                ax.plot(x_axis, line, color='0.78', lw=0.7, alpha=0.75)
        ax.plot(x_axis, np.nanmean(stacked, axis=0), color='red', lw=1.8)

    def _draw_average_groups(self, ax, x_axis: np.ndarray, groups: List[tuple[int, np.ndarray]]):
        if not groups:
            return
        mean_lines = [np.nanmean(group, axis=0) for _roi_idx, group in groups]
        combine_mode = self.combine_mode if self.combine_mode in COMBINE_MODES else 'individual'
        if combine_mode != 'individual':
            if self._avg_show_individual_traces():
                for (roi_idx, _group), line in zip(groups, mean_lines):
                    ax.plot(x_axis, line, color=self._roi_display_color(roi_idx), lw=0.9, alpha=0.35)
            stacked = np.vstack(mean_lines)
            if combine_mode == 'sum':
                y = np.nansum(stacked, axis=0)
                label = 'Sum of selected ROI averages'
            else:
                y = np.nanmean(stacked, axis=0)
                label = 'Mean of selected ROI averages'
            ax.plot(x_axis, y, color='red', lw=1.8, label=label)
            ax.legend(loc='upper right', fontsize=8)
            return
        offset_step = self._offset_step_for_lines(mean_lines) if len(groups) > 1 else 0.0
        for offset_idx, (roi_idx, group) in enumerate(groups):
            color = self._roi_display_color(roi_idx)
            offset = offset_idx * offset_step
            if self._avg_show_individual_traces():
                for line in group:
                    ax.plot(x_axis, line + offset, color=color, lw=0.7, alpha=0.28)
            ax.plot(x_axis, np.nanmean(group, axis=0) + offset, color=color, lw=1.8, label=self._roi_label_text(roi_idx))
        if len(groups) > 1:
            ax.legend(loc='upper right', fontsize=8)

    def _collect_trial_means(self, signal_list, pidx, roi_indices, onset_times, offset_times, duration):
        trace_mat = signal_list[pidx]
        return self._collect_trial_traces_from_matrix(trace_mat, roi_indices, onset_times, offset_times, duration)

    def _collect_trial_means_from_matrix(self, trace_mat, roi_indices, onset_times, offset_times, duration):
        return self._collect_trial_traces_from_matrix(trace_mat, roi_indices, onset_times, offset_times, duration)

    def _collect_trial_traces_from_matrix(self, trace_mat, roi_indices, onset_times, offset_times, duration):
        trial_traces = []
        x_axis = None
        for roi_idx in roi_indices:
            if roi_idx >= trace_mat.shape[0]:
                continue
            if not np.any(np.isfinite(trace_mat[roi_idx])):
                continue
            trial_mat, x_axis = self._extract_trial_matrix(trace_mat[roi_idx], onset_times, offset_times, duration)
            if trial_mat is not None and trial_mat.size:
                trial_traces.extend([row for row in trial_mat])
        return x_axis, trial_traces

    def _collect_trial_trace_groups_from_matrix(
        self,
        trace_mat,
        roi_indices,
        onset_times,
        offset_times,
        duration,
        roi_row_pairs: Optional[list[tuple[int, int]]] = None,
    ):
        trial_groups = []
        x_axis = None
        if roi_row_pairs is None:
            roi_row_pairs = [(int(idx), int(idx)) for idx in roi_indices]
        for roi_idx, row_idx in roi_row_pairs:
            if row_idx >= trace_mat.shape[0]:
                continue
            if not np.any(np.isfinite(trace_mat[row_idx])):
                continue
            trial_mat, x_axis = self._extract_trial_matrix(trace_mat[row_idx], onset_times, offset_times, duration)
            if trial_mat is not None and trial_mat.size:
                trial_groups.append((roi_idx, trial_mat))
        return x_axis, trial_groups

    def _collect_event_means(self, trace_mat, roi_indices, onset_list, offset_list, event_duration, duration):
        roi_means = []
        x_axis = None
        for roi_idx in roi_indices:
            if roi_idx >= trace_mat.shape[0] or roi_idx >= len(onset_list):
                continue
            if not np.any(np.isfinite(trace_mat[roi_idx])):
                continue
            events = np.asarray(onset_list[roi_idx], dtype=float)
            if events.size == 0:
                continue
            trial_mat, x_axis = self._extract_event_matrix(trace_mat[roi_idx], events, duration)
            if trial_mat is not None and trial_mat.size:
                roi_means.append(np.mean(trial_mat, axis=0))
        return x_axis, roi_means

    def _extract_trial_matrix(self, signal_1d, onset_times, offset_times, duration):
        onset_times = np.asarray(onset_times, dtype=float)
        offset_times = np.asarray(offset_times, dtype=float)
        if onset_times.size == 0 or offset_times.size == 0:
            return None, None
        indices = util.get_trace_indices(onset_times, offset_times, self.state.frame_rate, duration)
        n_trials = len(onset_times)
        if indices.size == 0 or n_trials == 0:
            return None, None
        if indices.shape[0] % n_trials != 0:
            usable = (indices.shape[0] // n_trials) * n_trials
            indices = indices[:usable]
        if indices.size == 0:
            return None, None

        min_idx = int(indices.min())
        max_idx = int(indices.max())
        left_pad = max(0, -min_idx)
        right_pad = max(0, max_idx - (len(signal_1d) - 1))
        padded = np.pad(np.asarray(signal_1d, dtype=float), (left_pad, right_pad), mode='constant')
        shifted = indices + left_pad

        trial_len = indices.shape[0] // n_trials
        trial_mat = padded[shifted].reshape(n_trials, trial_len)
        x_axis = np.arange(trial_len, dtype=float) / self.state.frame_rate + duration[0]
        return trial_mat, x_axis

    def _extract_event_matrix(self, signal_1d, event_times, duration):
        fr = max(self.state.frame_rate, 1e-8)
        signal = np.asarray(signal_1d, dtype=float)
        events = np.asarray(event_times, dtype=float)
        if signal.size == 0 or events.size == 0:
            return None, None
        start_offset = int(np.floor(float(duration[0]) * fr))
        stop_offset = int(np.ceil(float(duration[1]) * fr))
        if start_offset > 0 or stop_offset < 0 or stop_offset < start_offset:
            return None, None
        offsets = np.arange(start_offset, stop_offset + 1, dtype=int)
        event_frames = np.rint(events * fr).astype(int)
        indices = event_frames[:, None] + offsets[None, :]
        valid = np.all((indices >= 0) & (indices < signal.size), axis=1)
        if not np.any(valid):
            return None, None
        x_axis = offsets.astype(float) / fr
        return signal[indices[valid]], x_axis

    def _draw_stim_spans(self, ax):
        if self.state.onset_times_trial is None or self.state.offset_times_trial is None:
            return
        for t_on, t_off in zip(self.state.onset_times_trial, self.state.offset_times_trial):
            ax.axvspan(self._trace_time_to_x(t_on), self._trace_time_to_x(t_off), color='red', alpha=0.22, linewidth=0)

    def _draw_event_spans(self, ax, frame_indices: Optional[np.ndarray] = None, x_axis: Optional[np.ndarray] = None):
        if not self.events:
            return
        label_payload: list[tuple[str, str]] = []
        for event in self.events:
            color = str(event.get('color') or '#ff0000')
            label = str(event.get('label') or event.get('source') or 'event')
            drew_event = False
            for start, stop in self._event_frame_ranges(event):
                spans = self._event_span_x_ranges(start, stop, frame_indices=frame_indices, x_axis=x_axis)
                for left, right in spans:
                    ax.axvspan(left, right, color=color, alpha=0.18, linewidth=0)
                    drew_event = drew_event or right > left
            if drew_event:
                label_payload.append((label, color))
        if self.trace_labels_checkbox.isChecked():
            seen = set()
            unique_labels = []
            for label, color in label_payload:
                key = (label, color)
                if key in seen:
                    continue
                seen.add(key)
                unique_labels.append((label, color))
            for idx, (label, color) in enumerate(unique_labels):
                ax.text(
                    0.99,
                    0.98 - idx * 0.065,
                    label,
                    color=color,
                    fontsize=8,
                    ha='right',
                    va='top',
                    transform=ax.transAxes,
                    zorder=11,
                )

    def _event_span_x_ranges(
        self,
        start: int,
        stop: int,
        frame_indices: Optional[np.ndarray] = None,
        x_axis: Optional[np.ndarray] = None,
    ) -> list[tuple[float, float]]:
        if frame_indices is None:
            left = self._trace_frame_to_x(start)
            right = self._trace_frame_to_x(stop)
            return [(left, right)] if right > left else []
        frame_indices = np.asarray(frame_indices, dtype=int)
        hits = np.where((frame_indices >= start) & (frame_indices < stop))[0]
        if hits.size == 0:
            return []
        splits = np.flatnonzero(np.diff(hits) > 1) + 1
        groups = np.split(hits, splits)
        ranges = []
        for group in groups:
            if group.size == 0:
                continue
            left_idx = int(group[0])
            right_idx = int(group[-1]) + 1
            if x_axis is not None and len(x_axis) > right_idx:
                left = float(x_axis[left_idx])
                right = float(x_axis[right_idx])
            else:
                left = self._trace_frame_to_x(left_idx)
                right = self._trace_frame_to_x(right_idx)
            ranges.append((left, right))
        return ranges

    # ------------------------------ ROI utilities ------------------------------
    def _refresh_roi_list(self, _state: Optional[int] = None, select_first: bool = False):
        if self.state.cells is None:
            self.selected_roi_indices = []
            if hasattr(self, 'roi_list'):
                self.roi_list.clear()
            if hasattr(self, 'active_roi_label'):
                self.active_roi_label.setText('Selected ROI: none')
            self.render_all()
            return

        candidates = list(range(len(self.state.cells)))
        if self.only_cells_checkbox.isChecked():
            candidates = [i for i in candidates if self.state.cells[i] == 1]
        previous = [idx for idx in self.selected_roi_indices if idx in candidates]
        if select_first or not previous:
            previous = candidates[:1]
        self.selected_roi_indices = previous
        if hasattr(self, 'roi_list'):
            self.roi_list.blockSignals(True)
            self.roi_list.clear()
            for roi_idx in candidates:
                item = ROIListWidgetItem(self._roi_label_text(roi_idx))
                item.setData(Qt.ItemDataRole.UserRole, roi_idx)
                self.roi_list.addItem(item)
                item.setSelected(roi_idx in self.selected_roi_indices)
            self.roi_list.blockSignals(False)
        if hasattr(self, 'active_roi_label'):
            if self.selected_roi_indices:
                self.active_roi_label.setText(self._selected_roi_label_text(self.selected_roi_indices))
            else:
                self.active_roi_label.setText('Selected ROI: none')
        self.render_all()

    def _on_roi_list_selection_changed(self):
        selected = []
        for item in self.roi_list.selectedItems():
            selected.append(int(item.data(Qt.ItemDataRole.UserRole)))
        self.selected_roi_indices = selected
        if selected and hasattr(self, 'roi_index_slider'):
            self.roi_index_slider.blockSignals(True)
            self.roi_index_slider.setValue(selected[0])
            self.roi_index_slider.blockSignals(False)
            self.roi_index_label.setText(str(selected[0]))
        if hasattr(self, 'active_roi_label'):
            self.active_roi_label.setText(self._selected_roi_label_text(selected))
        self.render_all()

    def _selected_roi_label_text(self, roi_indices: List[int]) -> str:
        if not roi_indices:
            return 'Selected ROI: none'
        if len(roi_indices) == 1:
            return f'Selected ROI: {self._roi_label_text(roi_indices[0])}'
        shown = ', '.join(str(idx) for idx in roi_indices[:6])
        suffix = '...' if len(roi_indices) > 6 else ''
        return f'Selected ROIs ({len(roi_indices)}): {shown}{suffix}'

    def _roi_label_text(self, roi_idx: int) -> str:
        is_cell = self._roi_is_cell(roi_idx)
        is_stim = self._roi_is_stim(roi_idx)
        if is_cell is None:
            tag = 'roi'
        elif is_cell and is_stim:
            tag = 'stim cell'
        elif is_cell:
            tag = 'unstim cell'
        elif is_stim:
            tag = 'stim non-cell'
        else:
            tag = 'unstim non-cell'
        return f'ROI {roi_idx:03d} [{tag}]'

    def get_selected_roi_indices(self) -> List[int]:
        if self.state.cells is None:
            return []
        if self.selected_roi_indices:
            return [idx for idx in self.selected_roi_indices if idx < len(self.state.cells)]
        return [0] if len(self.state.cells) else []

    def get_selected_process_indices(self) -> List[int]:
        indices = []
        for row_idx, row in enumerate(self.trace_rows):
            if row.visible_checkbox.isChecked():
                indices.append(row_idx)
        return indices

    def _default_pca_wavelet_fmax_text(self) -> str:
        if self.state.frame_rate > 0:
            return f'{self.state.frame_rate:g}'
        return '500'

    def _update_pca_wavelet_fmax_defaults(self):
        text = self._default_pca_wavelet_fmax_text()
        for row in self.trace_rows:
            if bool(row.pca_wavelet_fmax_edit.property('autoFrameRate')):
                row.pca_wavelet_fmax_edit.blockSignals(True)
                row.pca_wavelet_fmax_edit.setText(text)
                row.pca_wavelet_fmax_edit.blockSignals(False)
                row.pca_wavelet_cfg['f_max'] = self._safe_float(text, default=500.0)

    def _update_process_controls(self):
        self._refresh_trace_source_options()
        self._update_pca_wavelet_fmax_defaults()
        self._update_trace_remove_buttons()
        self._update_roi_slider()
        self._update_image_layer_labels()
        self._refresh_mask_target_list()
        self._refresh_avg_trace_list()

    def _load_state_for_path(self, data_path: Path) -> GUIState:
        path = Path(data_path)
        if path.is_file() and path.suffix.lower() == '.xlsx':
            return self._load_table_state(str(path), negative=False)
        if path.is_dir():
            return self._load_folder_state(str(path), negative=False)
        raise ValueError(f'Data must be a Bruker folder or .xlsx file: {path}')

    def _load_pipeline_payload_from_path(self, pipeline_path: Path) -> dict[str, Any]:
        loaded = np.load(pipeline_path, allow_pickle=True)
        payload = loaded.item() if hasattr(loaded, 'shape') and loaded.shape == () else loaded
        if not isinstance(payload, dict) or payload.get('schema') != 'neurobox_pipeline':
            raise ValueError('Pipeline file must contain a NeuroBox pipeline dictionary.')
        return payload

    def _apply_pipeline_file(self, pipeline_path: Path) -> int:
        payload = self._load_pipeline_payload_from_path(pipeline_path)
        return self._apply_pipeline_payload(payload)

    def _apply_loaded_state_sync(
        self,
        state: GUIState,
        label: str,
        pipeline_path: Optional[Path] = None,
        auto_pipeline: bool = False,
    ) -> bool:
        self.state = state
        self.vpy = self.state.vpy
        globals()['vpy'] = self.vpy
        self.source_label.setText(label)
        self._image_projection_cache.clear()
        self._direct_trace_cache.clear()
        self._computed_trace_cache.clear()
        self._trace_result_cache.clear()
        self._pipeline_denoise_queue.clear()
        self._init_events_from_state()
        pipeline_loaded = False
        if pipeline_path is not None:
            self._apply_pipeline_file(pipeline_path)
            pipeline_loaded = True
        elif auto_pipeline:
            pipeline_loaded = self._auto_load_pipeline_for_state()
        if not pipeline_loaded:
            self._apply_no_pipeline_defaults()
        self._load_image_info_from_disk()
        self._update_image_info_table()
        self._update_process_controls()
        self._refresh_event_table()
        self._refresh_avg_event_list()
        self._refresh_roi_list(select_first=True)
        self._run_pipeline_denoise_queue_sync()
        self._refresh_trace_source_options()
        return pipeline_loaded

    def _is_bruker_data_folder(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        plane_dir = path / 'suite2p' / 'plane0'
        if plane_dir.exists():
            return True
        return self._find_tif_path(str(path)) is not None

    def _discover_batch_inputs(self, parent: Path) -> list[Path]:
        if not parent.exists() or not parent.is_dir():
            raise ValueError(f'Batch data path must be a parent folder: {parent}')
        items = []
        for child in sorted(parent.iterdir()):
            if child.is_file() and child.suffix.lower() == '.xlsx':
                items.append(child)
            elif child.is_dir() and self._is_bruker_data_folder(child):
                items.append(child)
        return items

    def _loaded_spike_times_for_trace_row(self, row: TraceControlRow) -> Optional[list[np.ndarray]]:
        if not self.state.spike_times:
            return None
        data = row.source_combo.currentData()
        idx = 0
        if isinstance(data, str) and data.startswith('state:'):
            idx = int(data.split(':', 1)[1])
        if idx >= len(self.state.spike_times):
            return None
        return self.state.spike_times[idx]

    def _headless_dataset_output_dir(self, output_root: Path) -> Path:
        folder = output_root / self._database_name()
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _export_headless_waveforms(self, output_dir: Path, pipeline_path: str) -> list[str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_paths: list[str] = []
        for row in self.trace_rows:
            if not row.visible_checkbox.isChecked():
                continue
            result = self._build_trace_result(row)
            if result is None:
                continue
            if result.get('spike_times') is None:
                loaded_spikes = self._loaded_spike_times_for_trace_row(row)
                if loaded_spikes is None:
                    continue
                result = dict(result)
                result['spike_times'] = loaded_spikes
            mode = row.waveform_mode_combo.currentText()
            modes = ['raw', 'current'] if mode == 'both' else [mode]
            for source_mode in modes:
                payload = self._waveform_export_payload(
                    row,
                    result,
                    source_mode,
                    pipeline_path=pipeline_path,
                )
                out_path = self._waveform_export_path(row, source_mode, output_dir=output_dir)
                np.save(out_path, payload, allow_pickle=True)
                out_paths.append(str(out_path))
        return out_paths

    def process_headless_dataset(self, data_path: Path, pipeline_path: Optional[Path], output_root: Path) -> dict[str, Any]:
        state = self._load_state_for_path(data_path)
        pipeline_text = str(pipeline_path.resolve()) if pipeline_path is not None else 'none'
        self._headless_pipeline_path = pipeline_text
        self._apply_loaded_state_sync(
            state,
            str(data_path),
            pipeline_path=pipeline_path,
            auto_pipeline=False,
        )
        output_dir = self._headless_dataset_output_dir(output_root)
        waveform_paths = self._export_headless_waveforms(output_dir, pipeline_text)
        summary = {
            'data': str(Path(data_path).resolve()),
            'database_name': self._database_name(),
            'pipeline': pipeline_text,
            'waveforms': waveform_paths,
        }
        np.save(output_dir / f"summary_{self._database_name()}.npy", summary, allow_pickle=True)
        return summary

    # ------------------------------ misc helpers ------------------------------
    def developing(self, func_name: str = 'Unknown function'):
        QMessageBox.information(self, 'Developing', f'Sorry, I am still developing this function:\n{func_name}')
        self._set_status(f'Developing warning: {func_name}')

    @staticmethod
    def _safe_float(value: str, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default


def _parse_cli_bool(value: str, name: str) -> bool:
    text = str(value).strip().strip('"\'').lower()
    if text in {'true', '1', 'yes', 'y'}:
        return True
    if text in {'false', '0', 'no', 'n'}:
        return False
    raise ValueError(f'{name} must be True or False.')


def _parse_neurobox_cli_args(argv: list[str]) -> dict[str, Any]:
    args: dict[str, Any] = {
        'showGUI': True,
        'batch': False,
        'pipeline': 'none',
        'data': None,
    }
    key_map = {'showgui': 'showGUI', 'batch': 'batch', 'pipeline': 'pipeline', 'data': 'data'}
    for item in argv:
        if '=' not in item:
            raise ValueError(f'Arguments must use key=value syntax: {item}')
        key, value = item.split('=', 1)
        key_norm = key.strip().lower()
        if key_norm not in key_map:
            raise ValueError(f'Unsupported argument: {key}')
        target = key_map[key_norm]
        value = value.strip().strip('"\'')
        if target in {'showGUI', 'batch'}:
            args[target] = _parse_cli_bool(value, target)
        elif target == 'pipeline':
            args[target] = 'none' if value.lower() in {'', 'none', 'null'} else value
        else:
            args[target] = None if value.lower() in {'', 'none', 'null'} else value
    if args['batch'] and args['showGUI']:
        raise ValueError('batch=True is only available when showGUI=False.')
    if not args['showGUI'] and args['data'] is None:
        raise ValueError('showGUI=False requires data=<Bruker folder, xlsx file, or parent folder>.')
    if args['showGUI'] and args['data'] is None and args['pipeline'] != 'none':
        raise ValueError('pipeline requires data when showGUI=True.')
    return args


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        cli = _parse_neurobox_cli_args(argv)
    except Exception as exc:
        print(f'NeuroBox argument error: {exc}', file=sys.stderr)
        return 2

    app = QApplication(sys.argv[:1])
    window = NeuroBoxGUI()
    pipeline_path = None if cli['pipeline'] == 'none' else Path(str(cli['pipeline']))
    if pipeline_path is not None and not pipeline_path.exists():
        print(f'NeuroBox argument error: pipeline file was not found: {pipeline_path}', file=sys.stderr)
        return 2

    if cli['showGUI']:
        if cli['data'] is not None:
            try:
                state = window._load_state_for_path(Path(str(cli['data'])))
                window._apply_loaded_state_sync(
                    state,
                    str(cli['data']),
                    pipeline_path=pipeline_path,
                    auto_pipeline=False,
                )
            except Exception as exc:
                print(f'NeuroBox load error: {exc}', file=sys.stderr)
                return 1
        window.show()
        return app.exec()

    data_path = Path(str(cli['data']))
    output_root = (data_path if cli['batch'] else (data_path.parent if data_path.is_file() else data_path.parent)) / 'results'
    try:
        if cli['batch']:
            inputs = window._discover_batch_inputs(data_path)
            summaries = [
                window.process_headless_dataset(item, pipeline_path, output_root)
                for item in inputs
            ]
            output_root.mkdir(parents=True, exist_ok=True)
            np.save(output_root / 'batch_summary.npy', {'items': summaries}, allow_pickle=True)
        else:
            summary = window.process_headless_dataset(data_path, pipeline_path, output_root)
            output_root.mkdir(parents=True, exist_ok=True)
            np.save(output_root / 'summary.npy', summary, allow_pickle=True)
    except Exception as exc:
        print(f'NeuroBox processing error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
