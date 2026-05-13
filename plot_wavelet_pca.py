"""
Plot intermediate results from the CWT denoising pipeline in cal_wavelet.py.

Figures generated:
1. CWT spectrogram: x = time in seconds, y = CWT returned frequency.
   A matplotlib Slider controls the displayed time window.
2. First PCA frequency features: x = PCA component, y = frequency,
   value = PCA coefficient/score of each frequency row.
3. CWT-domain activity traces for all clustered frequency domains, including
   kept and discarded domains. These traces are before ICWT and are plotted in
   the same CWT activity units as the threshold from recon_domain_traces().
4. Extracted event waveforms for each kept domain. Events are extracted from
   the ICWT reconstructed / threshold-filtered domain trace. Every event trace
   is gray; the mean waveform is colored.
5. Histograms of second event-PCA PC1 scores for events in each kept domain,
   with a vertical dotted line at x = 0.

Example:
    python plot_cwt_denoise_results.py --input traces.npy --framerate 400 --cell 0 --save-dir cwt_figures

For VolPy-like dictionaries saved with np.save:
    python plot_cwt_denoise_results.py --input volpy_result.npy --trace-key dFF --framerate 400 --cell 0

Put this file in the same folder as cal_wavelet.py, or run it from a directory
where cal_wavelet.py is importable.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.widgets import Slider
from numpy import argmax
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import cal_wavelet as cw


# Keep Slider objects alive; otherwise matplotlib may garbage-collect them.
_SLIDERS: List[Slider] = []


def parse_optional_float(value: str | None) -> Optional[float]:
    """Parse a float argument where 'none' means Python None."""
    if value is None:
        return None
    if str(value).lower() in {"none", "null", ""}:
        return None
    return float(value)


def load_trace(path: Path, trace_key: Optional[str], cell: int) -> np.ndarray:
    """Load one trace from .npy/.npz/.txt/.csv.

    Supported layouts:
      - 1D array: one trace, shape (n_frames,)
      - 2D array: traces, shape (n_cells, n_frames); choose row by --cell
      - dict saved by np.save: choose array by --trace-key, then apply above rules
      - npz: choose array by --trace-key, or the first key if omitted
    """
    suffix = path.suffix.lower()

    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if trace_key is None:
            if len(data.files) == 0:
                raise ValueError(f"No arrays found in {path}")
            trace_key = data.files[0]
            print(f"No --trace-key provided; using first npz key: {trace_key!r}")
        if trace_key not in data.files:
            raise KeyError(f"trace_key {trace_key!r} not found. Available keys: {data.files}")
        arr = data[trace_key]
    elif suffix == ".npy":
        loaded = np.load(path, allow_pickle=True)
        # np.save(dict) loads as a 0-D object array.
        if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
            obj = loaded.item()
            if not isinstance(obj, dict):
                raise TypeError(f"Expected dict object in {path}, got {type(obj)}")
            if trace_key is None:
                raise ValueError(
                    "This .npy contains a dictionary. Please pass --trace-key. "
                    f"Available keys: {list(obj.keys())}"
                )
            if trace_key not in obj:
                raise KeyError(f"trace_key {trace_key!r} not found. Available keys: {list(obj.keys())}")
            arr = obj[trace_key]
        else:
            arr = loaded
    elif suffix in {".txt", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        arr = np.loadtxt(path, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported input suffix {suffix!r}. Use .npy, .npz, .txt, or .csv")

    arr = np.asarray(arr, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        trace = arr
    elif arr.ndim == 2:
        if not (0 <= cell < arr.shape[0]):
            raise IndexError(f"--cell {cell} is out of range for array shape {arr.shape}")
        trace = arr[cell]
    else:
        raise ValueError(
            f"Expected a 1D or 2D trace array after loading, got shape {arr.shape}. "
            "Please select/reshape the trace before plotting."
        )

    trace = np.asarray(trace, dtype=float)
    if trace.size < 3:
        raise ValueError("Trace is too short to analyze.")
    return trace


def make_demo_trace(framerate: float, n_seconds: float = 6.0) -> np.ndarray:
    """Small synthetic trace for testing the plotting script without data."""
    n = int(round(framerate * n_seconds))
    t = np.arange(n) / framerate
    rng = np.random.default_rng(4)
    trace = 0.05 * rng.normal(size=n) + 0.02 * np.sin(2 * np.pi * 7 * t)
    spike_times = np.array([0.8, 1.35, 2.1, 3.2, 3.85, 5.0])
    width = max(2, int(round(0.008 * framerate)))
    kernel_t = np.arange(-5 * width, 8 * width + 1)
    kernel = -np.exp(-np.maximum(kernel_t, 0) / (2.0 * width)) * np.exp(-(kernel_t / (2.5 * width)) ** 2)
    for st in spike_times:
        center = int(round(st * framerate))
        start = max(0, center - 5 * width)
        stop = min(n, center + 8 * width + 1)
        k0 = start - (center - 5 * width)
        k1 = k0 + (stop - start)
        trace[start:stop] += kernel[k0:k1]
    return trace


def build_cfg(args: argparse.Namespace, framerate: float) -> cw.wavelet_cfg:
    f_max = args.f_max if args.f_max is not None else framerate
    return cw.wavelet_cfg(
        wavelet=args.wavelet,
        f_min=args.f_min,
        f_max=f_max,
        f_n=args.f_n,
        cwt_method=args.cwt_method,
        pca_explained_variance=args.pca_explained_variance,
        ward_n_clusters=args.ward_n_clusters,
        window_round=args.window_round,
        thres_std=args.thres_std,
        thres_mask_min=args.thres_mask_min,
        thres_mask_max=args.thres_mask_max,
        event_pca=args.event_pca or ("none" if args.disable_event_pca else "svd"),
        event_window_ms=args.event_window_ms,
        event_merge_gap_ms=args.event_merge_gap_ms,
        event_min_count=args.event_min_count,
        event_pc1_noise_max=args.event_pc1_noise_max,
        event_attenuation_min=args.event_attenuation_min,
        event_attenuation_max=args.event_attenuation_max,
    )


def extract_domains_from_clusters_all(
    features: np.ndarray,
    opt_k: int,
    n_clusters: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray, int]:
    """Cluster frequency rows for plotting when cal_wavelet extraction fails."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a 2D frequency-by-feature matrix.")
    if features.shape[0] == 0 or features.shape[1] == 0:
        return [], np.zeros(0, dtype=int), 0

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    opt_k = int(np.clip(int(opt_k), 1, features.shape[1]))
    X = np.asarray(features[:, :opt_k], dtype=float)
    n_samples = X.shape[0]

    max_valid_k = min(int(n_clusters), n_samples - 1)
    if max_valid_k < 2 or float(np.nanstd(X)) <= 1e-12:
        labels = np.zeros(n_samples, dtype=int)
        best_k = 1
    else:
        scores: List[float] = []
        all_labels: Dict[int, np.ndarray] = {}
        for k in range(2, max_valid_k + 1):
            labels_k = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
            score = silhouette_score(X, labels_k)
            scores.append(float(score))
            all_labels[k] = labels_k
        best_k = int(argmax(scores) + 2)
        labels = all_labels[best_k]

    domains: List[Dict[str, Any]] = []
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        pc1_mean = float(np.mean(X[idx, 0])) if idx.size else float("nan")
        domains.append(
            {
                "domain_id": int(cid),
                "freq_idx": idx,
                "pc1_mean": pc1_mean,
                "kept": True,
            }
        )

    # Low frequency to high frequency is easier to read in subplot order.
    domains.sort(key=lambda d: float(np.mean(d["freq_idx"])) if len(d["freq_idx"]) else -1.0)
    return domains, labels, best_k


def _domain_list_like(value: Any) -> bool:
    """Return True if value looks like a list/tuple of domain dictionaries."""
    if not isinstance(value, (list, tuple)):
        return False
    return len(value) == 0 or all(isinstance(item, dict) for item in value)


def _standardize_domains(
    domains: Iterable[Dict[str, Any]],
    kept: Optional[bool],
    freq_features: np.ndarray,
) -> List[Dict[str, Any]]:
    """Make domain dictionaries consistent for plotting.

    This accepts domains returned by either the original cal_wavelet.py function
    or a modified version that returns kept and discarded domains.
    """
    standardized: List[Dict[str, Any]] = []
    for fallback_id, domain in enumerate(domains):
        d = dict(domain)
        idx = np.asarray(d.get("freq_idx", []), dtype=int).reshape(-1)
        if idx.size == 0:
            continue

        pc1_mean = d.get("pc1_mean", None)
        if pc1_mean is None:
            pc1_mean = float(np.mean(freq_features[idx, 0]))
        else:
            pc1_mean = float(pc1_mean)

        if kept is None:
            if "kept" in d:
                is_kept = bool(d["kept"])
            elif "is_kept" in d:
                is_kept = bool(d["is_kept"])
            elif "status" in d:
                is_kept = str(d["status"]).lower() != "discarded"
            else:
                is_kept = True
        else:
            is_kept = bool(kept)

        d.update(
            {
                "domain_id": int(d.get("domain_id", fallback_id)),
                "freq_idx": idx,
                "pc1_mean": pc1_mean,
                "kept": is_kept,
            }
        )
        standardized.append(d)

    return standardized


def get_all_domains_for_plot(
    freq_features: np.ndarray,
    opt_k: int,
    n_clusters: int,
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], Optional[int], str]:
    """Get both kept and discarded domains.

    Preferred path: use cal_wavelet.extract_domains_from_clusters(). This supports
    a modified version that returns (kept_domains, discarded_domains), a dict with
    those keys, or one list where each domain has a kept/status flag.

    Fallback path is used only if cal_wavelet extraction fails or returns no
    domains; fallback domains are treated as kept because plotting must not infer
    discarded status from PC1 sign.
    """
    source = "cal_wavelet.extract_domains_from_clusters"
    try:
        returned = cw.extract_domains_from_clusters(freq_features, opt_k, n_clusters)
    except Exception as exc:
        print(f"Could not use cal_wavelet.extract_domains_from_clusters ({exc}); using local fallback.")
        domains, labels, best_k = extract_domains_from_clusters_all(freq_features, opt_k, n_clusters)
        return domains, labels, best_k, "local fallback"

    domains: List[Dict[str, Any]] = []
    labels: Optional[np.ndarray] = None
    best_k: Optional[int] = None

    if isinstance(returned, dict):
        kept_part = returned.get("kept_domains", returned.get("kept", []))
        discarded_part = returned.get("discarded_domains", returned.get("discarded", []))
        if _domain_list_like(kept_part) and _domain_list_like(discarded_part):
            domains = _standardize_domains(kept_part, True, freq_features)
            domains += _standardize_domains(discarded_part, False, freq_features)
            labels = returned.get("labels")
            best_k = returned.get("best_k")
    elif isinstance(returned, tuple) and len(returned) >= 2 and _domain_list_like(returned[0]) and _domain_list_like(returned[1]):
        # Expected modified signature: return kept_domains, discarded_domains
        domains = _standardize_domains(returned[0], True, freq_features)
        domains += _standardize_domains(returned[1], False, freq_features)
        if len(returned) >= 3 and returned[2] is not None:
            if isinstance(returned[2], np.ndarray):
                labels = returned[2]
            elif np.isscalar(returned[2]):
                best_k = int(returned[2])
        if len(returned) >= 4 and returned[3] is not None:
            if isinstance(returned[3], np.ndarray):
                labels = returned[3]
            elif np.isscalar(returned[3]):
                best_k = int(returned[3])
    elif _domain_list_like(returned):
        # Works if the modified function returns one combined list containing both
        # kept and discarded domains, each marked with kept/is_kept/status or pc1_mean.
        domains = _standardize_domains(returned, None, freq_features)

    if not domains:
        domains, labels, best_k = extract_domains_from_clusters_all(freq_features, opt_k, n_clusters)
        source = "local fallback because cal_wavelet returned no domains"

    domains.sort(key=lambda d: float(np.mean(d["freq_idx"])) if len(d["freq_idx"]) else -1.0)
    if best_k is None:
        best_k = len({int(d["domain_id"]) for d in domains}) if domains else 0
    return domains, labels, best_k, source


def reconstruct_domain_for_plot(
    domain: Dict[str, Any],
    coeffs: np.ndarray,
    freqs: np.ndarray,
    scales: np.ndarray,
    framerate: float,
    cfg: cw.wavelet_cfg,
) -> Dict[str, Any]:
    """Reproduce recon_domain_traces() internals and expose plotting variables."""
    idx = np.asarray(domain["freq_idx"], dtype=int).reshape(-1)
    domain_coeffs = coeffs[idx, :]
    domain_freqs = freqs[idx]
    domain_scales = scales[idx]

    activity = np.abs(domain_coeffs).mean(axis=0)
    max_freq = max(float(np.max(domain_freqs)), 1e-12)
    # This matches the window expression in recon_domain_traces().
    window = 2 * (round(framerate / max_freq) * cfg.window_round // 2) + 1
    window = max(1, int(window))
    baseline = np.convolve(activity, np.ones(window, dtype=float) / window, mode="same")
    thres = baseline + cfg.thres_std * np.std(activity - baseline)

    delta = np.maximum(-(activity - baseline), 0)
    delta_max = float(np.max(delta)) if delta.size else 0.0
    if delta_max <= 1e-12:
        below_mask = np.zeros_like(delta, dtype=float)
    else:
        below_mask = np.clip(delta / delta_max, cfg.thres_mask_min, cfg.thres_mask_max)

    events = np.where(activity > thres)[0]
    mask_thres = np.where(activity > thres, 1.0, 1.0 - below_mask)
    event_regions, event_peaks = cw.candidate_event_regions(activity, thres, framerate, cfg)

    raw_trace = cw.git_icwt_1(domain_coeffs, domain_scales, cfg)
    threshold_filtered_trace = cw.git_icwt_1(domain_coeffs, domain_scales, cfg, mask=mask_thres)

    event_pca_info = None
    final_filtered_trace = threshold_filtered_trace.copy()
    if bool(domain["kept"]) and cw.event_denoise_mode(cfg) != "none":
        final_filtered_trace, event_pca_info = cw.event_pca_attenuate_domain_trace(
            threshold_filtered_trace,
            event_regions,
            event_peaks,
            framerate,
            cfg,
            events=events,
        )

    event_windows = np.zeros((0, 0), dtype=float)
    event_windows_for_plot = np.zeros((0, 0), dtype=float)
    event_time_ms = np.zeros(0, dtype=float)
    valid_event_mask = np.zeros(0, dtype=bool)
    accepted_event_mask = np.zeros(0, dtype=bool)
    if bool(domain["kept"]) and len(event_peaks) > 0:
        if event_pca_info is not None and "event_window_half_width" in event_pca_info:
            half_window = int(event_pca_info["event_window_half_width"])
        else:
            half_window = cw.event_pca_half_window(events, event_regions, framerate, cfg)
        event_windows = cw.extract_event_windows(threshold_filtered_trace, event_peaks, half_window)
        event_time_ms = (np.arange(event_windows.shape[1]) - half_window) / framerate * 1000.0
        centered = event_windows - np.mean(event_windows, axis=1, keepdims=True)
        scale = np.std(centered, axis=1)
        valid_event_mask = scale > 1e-8
        event_windows_for_plot = np.zeros_like(centered)
        if np.any(valid_event_mask):
            # This is the normalized event waveform used by event_pca_attenuate_domain_trace().
            event_windows_for_plot[valid_event_mask] = centered[valid_event_mask] / scale[valid_event_mask, None]
        if event_pca_info is not None:
            pc1_scores = np.asarray(event_pca_info.get("pc1_scores", []), dtype=float)
            noise_events = np.asarray(event_pca_info.get("noise_events", []), dtype=bool)
            if pc1_scores.shape == valid_event_mask.shape:
                accepted_event_mask = valid_event_mask & np.isfinite(pc1_scores) & (pc1_scores > 0)
                if noise_events.shape == valid_event_mask.shape:
                    accepted_event_mask &= ~noise_events

    return {
        "domain_id": int(domain["domain_id"]),
        "freq_idx": idx,
        "freqs": domain_freqs,
        "scales": domain_scales,
        "pc1_mean": float(domain["pc1_mean"]),
        "kept": bool(domain["kept"]),
        "activity": activity,
        "baseline": baseline,
        "thres": thres,
        "mask_thres": mask_thres,
        "events": events,
        "event_regions": event_regions,
        "event_peaks": np.asarray(event_peaks, dtype=int),
        "raw_trace": np.asarray(raw_trace, dtype=float),
        "threshold_filtered_trace": np.asarray(threshold_filtered_trace, dtype=float),
        "final_filtered_trace": np.asarray(final_filtered_trace, dtype=float),
        "event_pca": event_pca_info,
        "event_windows": event_windows,
        "event_windows_for_plot": event_windows_for_plot,
        "event_time_ms": event_time_ms,
        "valid_event_mask": valid_event_mask,
        "accepted_event_mask": accepted_event_mask,
    }


def centers_to_edges(values: np.ndarray, log: bool = False) -> np.ndarray:
    """Convert center coordinates into pcolormesh-style edges."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([0.0, 1.0])
    if values.size == 1:
        width = 0.5 * values[0] if log and values[0] > 0 else 0.5
        return np.array([values[0] - width, values[0] + width])

    if log and np.all(values > 0):
        edges = np.empty(values.size + 1, dtype=float)
        edges[1:-1] = np.sqrt(values[:-1] * values[1:])
        first_ratio = values[1] / values[0]
        last_ratio = values[-1] / values[-2]
        edges[0] = values[0] / math.sqrt(first_ratio)
        edges[-1] = values[-1] * math.sqrt(last_ratio)
        return edges

    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def format_plain_float(value: float, precision: int = 3) -> str:
    if not np.isfinite(value):
        return str(value)
    text = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def format_freq_range(freqs: np.ndarray) -> str:
    if len(freqs) == 0:
        return "empty"
    return f"{format_plain_float(float(np.min(freqs)), 2)}-{format_plain_float(float(np.max(freqs)), 2)} Hz"


def apply_plain_number_format(*axes: plt.Axes) -> None:
    for ax in axes:
        formatter_x = ScalarFormatter(useOffset=False)
        formatter_x.set_scientific(False)
        formatter_y = ScalarFormatter(useOffset=False)
        formatter_y.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter_x)
        ax.yaxis.set_major_formatter(formatter_y)


def apply_plain_colorbar_format(cbar: Any) -> None:
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    cbar.formatter = formatter
    cbar.update_ticks()


def pca_component_ticks(n_pc: int) -> np.ndarray:
    if n_pc <= 0:
        return np.asarray([], dtype=int)
    ticks = np.arange(10, n_pc + 1, 10, dtype=int)
    if ticks.size == 0 or ticks[-1] != n_pc:
        ticks = np.append(ticks, n_pc)
    return np.unique(ticks)


def finalize_figure(fig: plt.Figure, rect: Optional[Tuple[float, float, float, float]] = None) -> None:
    if rect is None:
        fig.tight_layout(pad=1.0)
    else:
        fig.tight_layout(rect=rect, pad=1.0)


def add_time_window_slider(fig: plt.Figure, axes: Iterable[plt.Axes], total_seconds: float, window_sec: float) -> None:
    """Add one Slider plus mouse scroll, drag, and double-click reset controls."""
    axes = list(axes)
    if total_seconds <= 0 or window_sec <= 0 or window_sec >= total_seconds:
        return

    fig.subplots_adjust(bottom=0.18)
    state = {"window": min(float(window_sec), float(total_seconds)), "start": 0.0, "drag": None}
    slider_ax = fig.add_axes([0.18, 0.05, 0.65, 0.03])
    slider = Slider(
        ax=slider_ax,
        label="window start (s)",
        valmin=0.0,
        valmax=max(0.0, total_seconds - state["window"]),
        valinit=0.0,
        valstep=None,
    )

    def set_slider_bounds() -> None:
        slider.valmax = max(0.0, float(total_seconds) - state["window"])
        slider.ax.set_xlim(slider.valmin, slider.valmax)

    def apply_window(start: float, sync_slider: bool = True) -> None:
        state["start"] = float(np.clip(start, 0.0, max(0.0, float(total_seconds) - state["window"])))
        stop = state["start"] + state["window"]
        for ax in axes:
            ax.set_xlim(state["start"], stop)
        if sync_slider and abs(slider.val - state["start"]) > 1e-9:
            slider.set_val(state["start"])
        fig.canvas.draw_idle()

    def update(start: float) -> None:
        apply_window(float(start), sync_slider=False)

    def on_scroll(event: Any) -> None:
        if event.inaxes not in axes:
            return
        center = float(event.xdata) if event.xdata is not None else state["start"] + state["window"] * 0.5
        factor = 0.8 if event.button == "up" else 1.25
        new_window = float(np.clip(state["window"] * factor, min(0.05, total_seconds), total_seconds))
        rel = (center - state["start"]) / max(state["window"], 1e-9)
        state["window"] = new_window
        set_slider_bounds()
        apply_window(center - rel * state["window"])

    def on_press(event: Any) -> None:
        if event.inaxes not in axes:
            return
        if getattr(event, "dblclick", False):
            state["window"] = min(float(window_sec), float(total_seconds))
            set_slider_bounds()
            apply_window(0.0)
            return
        if event.button == 1 and event.xdata is not None:
            state["drag"] = (float(event.xdata), state["start"])

    def on_motion(event: Any) -> None:
        drag = state.get("drag")
        if drag is None or event.xdata is None:
            return
        x0, start0 = drag
        apply_window(start0 - (float(event.xdata) - x0))

    def on_release(_event: Any) -> None:
        state["drag"] = None

    slider.on_changed(update)
    set_slider_bounds()
    apply_window(0.0)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    _SLIDERS.append(slider)
    setattr(fig, "_time_window_slider", slider)


def plot_spectrogram(
    coeffs_norm: np.ndarray,
    freqs: np.ndarray,
    framerate: float,
    title: str,
    window_sec: float,
) -> plt.Figure:
    n_frames = coeffs_norm.shape[1]
    total_seconds = n_frames / framerate
    time_edges = np.arange(n_frames + 1) / framerate
    freq_edges = centers_to_edges(freqs, log=True)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    mesh = ax.pcolormesh(time_edges, freq_edges, coeffs_norm, shading="auto", cmap="jet")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("normalized |CWT coefficient|")
    apply_plain_colorbar_format(cbar)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency returned by pywt.cwt (Hz)")
    ax.set_yscale("log")
    ax.set_title(title)
    apply_plain_number_format(ax)
    finalize_figure(fig, rect=[0.0, 0.12, 1.0, 1.0])
    add_time_window_slider(fig, [ax], total_seconds, window_sec)
    return fig


def plot_first_pca(
    freq_features: np.ndarray,
    freqs: np.ndarray,
    opt_k: int,
    pca: Optional[Any] = None,
) -> plt.Figure:
    n_pc = freq_features.shape[1]
    display_n_pc = min(n_pc, max(1, int(math.ceil(2.5 * max(int(opt_k), 1)))))
    displayed_features = freq_features[:, :display_n_pc]
    pc_edges = np.arange(display_n_pc + 1) + 0.5
    freq_edges = centers_to_edges(freqs, log=True)
    abs_limit = max(10.0, float(np.nanmax(np.abs(displayed_features))) if displayed_features.size else 10.0)
    xticks = pca_component_ticks(display_n_pc)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10.5, 8.0), gridspec_kw={"height_ratios": [3.2, 1.3]})
    ax = axes[0]
    var_ax = axes[1]
    mesh = ax.pcolormesh(
        pc_edges,
        freq_edges,
        displayed_features,
        shading="auto",
        cmap="jet",
        vmin=-abs_limit,
        vmax=abs_limit,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("PCA coefficient / score")
    apply_plain_colorbar_format(cbar)
    ax.set_ylabel("frequency returned by pywt.cwt (Hz)")
    ax.set_yscale("log")
    ax.set_title(f"First PCA across frequency rows; elbow/used components = {opt_k}; showing {display_n_pc}/{n_pc}")
    ax.set_xticks(xticks)
    ax.set_xlim(0.5, max(float(display_n_pc) + 0.5, 1.5))
    if 1 <= opt_k <= display_n_pc:
        ax.axvline(opt_k + 0.5, linestyle=":", linewidth=1.5)

    explained = np.asarray(getattr(pca, "explained_variance_ratio_", []), dtype=float)
    if explained.size:
        explained_display = explained[:display_n_pc]
        component_idx = np.arange(1, explained_display.size + 1)
        var_ax.plot(component_idx, explained_display, marker="o", linewidth=1.2, markersize=3)
        var_ax.set_xlim(0.5, max(float(display_n_pc) + 0.5, 1.5))
    else:
        var_ax.text(0.5, 0.5, "Explained variance ratio unavailable", ha="center", va="center", transform=var_ax.transAxes)
    var_ax.set_xticks(xticks)
    var_ax.set_xlabel("number of PCA components")
    var_ax.set_ylabel("explained variance ratio")
    var_ax.set_title("PCA explained variance ratio")
    apply_plain_number_format(ax, var_ax)
    finalize_figure(fig, rect=[0, 0, 1, 0.98])
    fig.subplots_adjust(hspace=0.28)
    return fig


def make_denoised_trace_for_plot(original_trace: np.ndarray, domain_results: List[Dict[str, Any]]) -> np.ndarray:
    kept = kept_results(domain_results)
    original_trace = np.asarray(original_trace, dtype=float)
    if not kept:
        return np.zeros_like(original_trace)
    summed = np.sum([np.asarray(result["final_filtered_trace"], dtype=float) for result in kept], axis=0)
    if np.vdot(summed, summed).real <= 1e-12:
        return np.zeros_like(original_trace)
    return np.asarray(cw.rebuild_amp(original_trace, summed), dtype=float)


def domain_trace_colors(domain_results: List[Dict[str, Any]]) -> List[Any]:
    if not domain_results:
        return []
    cmap = plt.get_cmap("turbo")
    if len(domain_results) == 1:
        return [cmap(0.15)]
    positions = np.linspace(0.05, 0.95, len(domain_results))
    return [cmap(float(pos)) for pos in positions]


def plot_domain_frequency_bins(ax: plt.Axes, domain_results: List[Dict[str, Any]], colors: List[Any]) -> None:
    freq_arrays = []
    for result in domain_results:
        freqs = np.asarray(result.get("freqs", []), dtype=float).reshape(-1)
        freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
        if freqs.size:
            freq_arrays.append(freqs)

    if not freq_arrays:
        ax.text(0.5, 0.5, "No domain frequencies", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    all_freqs = np.unique(np.concatenate(freq_arrays))
    all_freqs.sort()
    freq_edges = centers_to_edges(all_freqs, log=True)

    for result, color in zip(domain_results, colors):
        domain_freqs = np.asarray(result.get("freqs", []), dtype=float).reshape(-1)
        domain_freqs = domain_freqs[np.isfinite(domain_freqs) & (domain_freqs > 0)]
        for freq in domain_freqs:
            pos = int(np.searchsorted(all_freqs, freq))
            if pos >= all_freqs.size:
                continue
            if not np.isclose(all_freqs[pos], freq, rtol=1e-7, atol=1e-12):
                continue
            ax.axvspan(freq_edges[pos], freq_edges[pos + 1], 0.0, 1.0, color=color, alpha=0.9, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlim(float(freq_edges[0]), float(freq_edges[-1]))
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("frequency (Hz)")
    ax.set_title("domain frequencies", loc="left", fontsize=9, pad=6)
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.minorticks_off()


def plot_domain_traces(
    domain_results: List[Dict[str, Any]],
    framerate: float,
    window_sec: float,
    original_trace: Optional[np.ndarray] = None,
    denoised_trace: Optional[np.ndarray] = None,
) -> plt.Figure:
    n_domains = len(domain_results)
    if n_domains == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No domains found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    n_frames = len(domain_results[0]["raw_trace"])
    t = np.arange(n_frames) / framerate
    total_seconds = n_frames / framerate

    show_summary_traces = original_trace is not None and denoised_trace is not None
    n_trace_axes = n_domains * 2 + (2 if show_summary_traces else 0)
    fig_height = min(max(2.25 * n_trace_axes + 0.8, 7.5), 46.0)
    fig = plt.figure(figsize=(11.8, fig_height))
    gs = fig.add_gridspec(n_trace_axes + 1, 1, height_ratios=[1.0] * n_trace_axes + [0.38])
    trace_axes: List[plt.Axes] = []
    for axis_idx in range(n_trace_axes):
        share_axis = trace_axes[0] if trace_axes else None
        trace_axes.append(fig.add_subplot(gs[axis_idx, 0], sharex=share_axis))
    freq_ax = fig.add_subplot(gs[-1, 0])
    axis_offset = 0
    colors = domain_trace_colors(domain_results)

    if show_summary_traces:
        original_ax = trace_axes[0]
        denoised_ax = trace_axes[1]
        original_ax.plot(t, np.asarray(original_trace, dtype=float), linewidth=0.95, color="0.18")
        original_ax.set_title("original trace", loc="left", fontsize=9, pad=6)
        original_ax.set_ylabel("")
        denoised_ax.plot(t, np.asarray(denoised_trace, dtype=float), linewidth=0.95, color="0.35")
        denoised_ax.set_title("denoised trace", loc="left", fontsize=9, pad=6)
        denoised_ax.set_ylabel("")
        axis_offset = 2

    for domain_idx, result in enumerate(domain_results):
        raw_ax = trace_axes[axis_offset + domain_idx * 2]
        filtered_ax = trace_axes[axis_offset + domain_idx * 2 + 1]
        kept = result["kept"]
        status = "kept" if kept else "discarded"
        color = colors[domain_idx]
        domain_label = domain_idx + 1
        raw_ax.plot(t, result["raw_trace"], linewidth=0.9, color=color)
        raw_ax.set_ylabel("")
        raw_ax.set_title(f"domain {domain_label}: {status}(non-filtered)", loc="left", fontsize=9, pad=6)

        filtered_ax.plot(t, result["final_filtered_trace"], linewidth=0.9, color=color)
        filtered_ax.set_ylabel("")
        filtered_ax.set_title(f"domain {domain_label}: {status}(filtered)", loc="left", fontsize=9, pad=6)

    trace_axes[-1].set_xlabel("time (s)")
    plot_domain_frequency_bins(freq_ax, domain_results, colors)
    apply_plain_number_format(*trace_axes)
    finalize_figure(fig, rect=[0.0, 0.11, 1.0, 0.995])
    fig.subplots_adjust(hspace=0.54)
    add_time_window_slider(fig, trace_axes, total_seconds, window_sec)
    return fig


def kept_results(domain_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in domain_results if r["kept"]]


def plot_event_waveforms(domain_results: List[Dict[str, Any]]) -> plt.Figure:
    kept = kept_results(domain_results)
    n = len(kept)
    if n == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No kept domains", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 4.6 * nrows), squeeze=False)

    for ax, result in zip(axes.flat, kept):
        windows = result["event_windows"]
        valid = result.get("valid_event_mask", np.zeros(windows.shape[0], dtype=bool))
        event_time_ms = result["event_time_ms"]
        if windows.size == 0 or not np.any(valid):
            ax.text(0.5, 0.5, "No valid events", ha="center", va="center", transform=ax.transAxes)
        else:
            valid_windows = windows[valid]
            for w in valid_windows:
                ax.plot(event_time_ms, w, color="0.75", linewidth=0.7)
            ax.plot(event_time_ms, np.mean(valid_windows, axis=0), color="C0", linewidth=2.0, label="average")
            ax.legend(loc="best", fontsize=8)
        ax.set_title(f"domain {result['domain_id']} events, {format_freq_range(result['freqs'])}")
        ax.set_xlabel("time from event peak (ms)")
        ax.set_ylabel("event trace")
        apply_plain_number_format(ax)

    for ax in axes.flat[n:]:
        ax.set_axis_off()

    fig.suptitle("Average event traces for each kept frequency domain", y=0.995)
    finalize_figure(fig, rect=[0, 0, 1, 0.965])
    fig.subplots_adjust(hspace=0.56, wspace=0.28)
    return fig


def plot_event_pc1_histograms(domain_results: List[Dict[str, Any]]) -> plt.Figure:
    kept = kept_results(domain_results)
    n = len(kept)
    if n == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No kept domains", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 4.35 * nrows), squeeze=False)

    for ax, result in zip(axes.flat, kept):
        info = result.get("event_pca")
        if info is None:
            scores = np.asarray([], dtype=float)
        else:
            scores = np.asarray(info.get("pc1_scores", []), dtype=float)
            scores = scores[np.isfinite(scores)]

        if scores.size == 0:
            ax.text(0.5, 0.5, "No valid second-PCA scores", ha="center", va="center", transform=ax.transAxes)
        else:
            bins = min(25, max(5, int(np.sqrt(scores.size))))
            ax.hist(scores, bins=bins)
        ax.axvline(0.0, linestyle=":", linewidth=1.5)
        ax.set_title(f"domain {result['domain_id']} kept, {format_freq_range(result['freqs'])}")
        ax.set_xlabel("event score")
        ax.set_ylabel("event count")
        apply_plain_number_format(ax)

    for ax in axes.flat[n:]:
        ax.set_axis_off()

    fig.suptitle("Event score histograms", y=0.995)
    finalize_figure(fig, rect=[0, 0, 1, 0.965])
    fig.subplots_adjust(hspace=0.56, wspace=0.28)
    return fig


def save_figures(figures: Dict[str, plt.Figure], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        out = save_dir / f"{name}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Saved {out}")


def run(args: argparse.Namespace) -> Dict[str, plt.Figure]:
    framerate = float(args.framerate)
    if args.input is None:
        print("No --input provided; using a synthetic demo trace. Pass --input for real data.")
        trace = make_demo_trace(framerate)
    else:
        trace = load_trace(Path(args.input), args.trace_key, args.cell)

    trace = np.asarray(trace, dtype=float)
    trace = trace - np.nanmean(trace)
    trace = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)

    cfg = build_cfg(args, framerate)
    coeffs, coeffs_norm, freqs, scales = cw.morlet_cwt(trace, framerate, cfg)
    freq_features, pca, opt_k = cw.pca_feature(coeffs_norm, explained_variance=cfg.pca_explained_variance)
    all_domains, labels, best_k, domain_source = get_all_domains_for_plot(freq_features, opt_k, cfg.ward_n_clusters)
    domain_results = [reconstruct_domain_for_plot(d, coeffs, freqs, scales, framerate, cfg) for d in all_domains]
    denoised_trace = make_denoised_trace_for_plot(trace, domain_results)

    print(f"Trace frames: {trace.size}; duration: {trace.size / framerate:.3f} s")
    print(f"CWT frequencies: {freqs.size}; range: {np.min(freqs):.3g} to {np.max(freqs):.3g} Hz")
    print(f"First PCA components returned: {freq_features.shape[1]}; elbow/used components: {opt_k}")
    print(f"Domain source: {domain_source}")
    print(f"Ward clusters selected by silhouette: {best_k}")
    print(f"Domains: {len(all_domains)} total, {sum(d['kept'] for d in all_domains)} kept, {sum(not d['kept'] for d in all_domains)} discarded")

    stem = Path(args.input).stem if args.input else "demo"
    figures = {
        "01_spectrogram": plot_spectrogram(
            coeffs_norm,
            freqs,
            framerate,
            title=f"CWT spectrogram: {stem}",
            window_sec=args.window_sec,
        ),
        "02_first_pca_frequency_coefficients": plot_first_pca(freq_features, freqs, opt_k, pca=pca),
        "03_all_frequency_domain_traces": plot_domain_traces(
            domain_results,
            framerate,
            args.window_sec,
            original_trace=trace,
            denoised_trace=denoised_trace,
        ),
    }
    if cw.event_denoise_mode(cfg) != "none":
        figures.update(
            {
                "04_event_waveforms_kept_domains": plot_event_waveforms(domain_results),
                "05_event_pc1_histograms_kept_domains": plot_event_pc1_histograms(domain_results),
            }
        )

    if args.save_dir is not None:
        save_figures(figures, Path(args.save_dir))

    if not args.no_show:
        plt.show()

    return figures


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot CWT denoising intermediates from cal_wavelet.py")
    parser.add_argument("--input", type=str, default=None, help="Input trace file: .npy, .npz, .txt, or .csv")
    parser.add_argument("--trace-key", type=str, default=None, help="Key for dict/npz input, e.g. dFF")
    parser.add_argument("--cell", type=int, default=0, help="Cell/row index for 2D trace arrays")
    parser.add_argument("--framerate", type=float, default=400.0, help="Sampling rate in frames/second")
    parser.add_argument("--save-dir", type=str, default=None, help="Optional folder to save PNG figures")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive matplotlib windows")
    parser.add_argument("--window-sec", type=float, default=5.0, help="Time-window width controlled by Slider, in seconds")

    # Main wavelet_cfg parameters.
    parser.add_argument("--wavelet", type=str, default="cmor1.5-1.0")
    parser.add_argument("--f-min", type=float, default=1.0)
    parser.add_argument("--f-max", type=float, default=None, help="Default: framerate, matching denoise_trace()")
    parser.add_argument("--f-n", type=int, default=100)
    parser.add_argument("--cwt-method", type=str, default="fft", choices=["fft", "conv"])
    parser.add_argument("--pca-explained-variance", type=parse_optional_float, default=None)
    parser.add_argument("--ward-n-clusters", type=int, default=20)
    parser.add_argument("--window-round", type=int, default=2)
    parser.add_argument("--thres-std", type=float, default=2.0)
    parser.add_argument("--thres-mask-min", type=float, default=0.5)
    parser.add_argument("--thres-mask-max", type=float, default=1.0)

    # Event-level denoise parameters.
    parser.add_argument("--event-pca", type=str, default=None, choices=["none", "pca", "svd"])
    parser.add_argument("--disable-event-pca", action="store_true", help="Legacy alias for --event-pca none")
    parser.add_argument("--event-window-ms", type=float, default=50.0)
    parser.add_argument("--event-merge-gap-ms", type=float, default=10.0)
    parser.add_argument("--event-min-count", type=int, default=3)
    parser.add_argument("--event-pc1-noise-max", type=float, default=0.0)
    parser.add_argument("--event-attenuation-min", type=float, default=0.5)
    parser.add_argument("--event-attenuation-max", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    run(make_argparser().parse_args())
