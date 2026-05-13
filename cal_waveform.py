import numpy as np
from scipy.optimize import curve_fit


def _crossing_time(t, y, level, i1, i2):
    """
    Linear interpolation crossing time between index i1 and i2.
    """
    y1, y2 = y[i1], y[i2]
    t1, t2 = t[i1], t[i2]

    if y2 == y1:
        return np.nan

    return t1 + (level - y1) * (t2 - t1) / (y2 - y1)


def _fwhm(t, y, peak_idx):
    """
    Calculate FWHM around the aligned positive peak.
    y should be baseline-corrected and oriented so the main peak is positive.
    """
    peak_amp = y[peak_idx]

    if not np.isfinite(peak_amp) or peak_amp <= 0:
        return np.nan

    half = 0.5 * peak_amp

    # Left half-maximum crossing
    left_cross = np.nan
    for i in range(peak_idx - 1, -1, -1):
        if y[i] <= half <= y[i + 1]:
            left_cross = _crossing_time(t, y, half, i, i + 1)
            break

    # Right half-maximum crossing
    right_cross = np.nan
    for i in range(peak_idx, len(y) - 1):
        if y[i] >= half >= y[i + 1]:
            right_cross = _crossing_time(t, y, half, i, i + 1)
            break

    if np.isnan(left_cross) or np.isnan(right_cross):
        return np.nan

    return right_cross - left_cross


def _biexp_peak_aligned(t, baseline, scale, tau_r, tau_gap):
    """
    Peak-aligned bi-exponential.

    Standard form:
        y = baseline + scale * (exp(-x/tau_d) - exp(-x/tau_r))

    Here the onset time is automatically chosen so that the model peak occurs at t = 0.

    tau_d = tau_r + tau_gap, so tau_d is always larger than tau_r.
    """
    tau_d = tau_r + tau_gap

    # Time from onset to peak for difference-of-exponentials
    t_peak_from_onset = (
        tau_r * tau_d / (tau_d - tau_r) * np.log(tau_d / tau_r)
    )

    onset = -t_peak_from_onset
    x = t - onset

    y = np.full_like(t, baseline, dtype=float)
    mask = x >= 0

    y[mask] = baseline + scale * (
        np.exp(-x[mask] / tau_d) - np.exp(-x[mask] / tau_r)
    )

    return y


def _fit_biexp_peak_aligned(t_ms, y_norm, peak_idx, fit_main_lobe_only=True):
    """
    Fit bi-exponential to one normalized waveform.

    y_norm should be:
        baseline-corrected
        polarity-oriented
        peak-normalized, so y_norm[peak_idx] is about 1
    """
    dt = np.median(np.diff(t_ms))
    total_window = t_ms[-1] - t_ms[0]

    # Optional: fit only the main positive lobe.
    # This avoids forcing the bi-exponential model past the return to baseline.
    if fit_main_lobe_only:
        stop_idx = len(y_norm)

        # First baseline crossing after the peak
        after = np.where(y_norm[peak_idx + 1:] <= 0)[0]
        if len(after) > 0:
            stop_idx = peak_idx + 1 + after[0] + 1

        fit_mask = np.zeros_like(y_norm, dtype=bool)
        fit_mask[:stop_idx] = True
    else:
        fit_mask = np.ones_like(y_norm, dtype=bool)

    t_fit = t_ms[fit_mask]
    y_fit = y_norm[fit_mask]

    # Need enough points for stable fitting
    if len(t_fit) < 6:
        result = {
            "tau_r_ms": np.nan,
            "tau_d_ms": np.nan,
            "fit_r2": np.nan,
            "fit_success": False,
        }
        result["fit_t_ms"] = np.asarray([], dtype=float)
        result["fit_y"] = np.asarray([], dtype=float)
        return result

    # Initial guesses
    fwhm_guess = _fwhm(t_ms, y_norm, peak_idx)
    if not np.isfinite(fwhm_guess):
        fwhm_guess = max(3 * dt, total_window / 10)

    tau_r0 = max(dt / 2, fwhm_guess / 5)
    tau_d0 = max(2 * dt, fwhm_guess * 2)
    tau_gap0 = max(dt / 2, tau_d0 - tau_r0)

    p0 = [
        0.0,       # baseline after normalization
        1.0,       # scale, not biologically interpreted
        tau_r0,
        tau_gap0,
    ]

    lower = [
        -0.5,      # baseline
        0.001,     # scale
        dt / 20,   # tau_r
        dt / 20,   # tau_gap
    ]

    upper = [
        0.5,              # baseline
        100.0,            # scale
        total_window * 5, # tau_r
        total_window * 10 # tau_gap
    ]

    try:
        popt, _ = curve_fit(
            _biexp_peak_aligned,
            t_fit,
            y_fit,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )

        baseline, scale, tau_r, tau_gap = popt
        tau_d = tau_r + tau_gap

        y_pred = _biexp_peak_aligned(t_fit, *popt)

        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        fit_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {
            "tau_r_ms": tau_r,
            "tau_d_ms": tau_d,
            "fit_r2": fit_r2,
            "fit_success": True,
            "fit_t_ms": t_fit,
            "fit_y": y_pred,
        }

    except Exception:
        return {
            "tau_r_ms": np.nan,
            "tau_d_ms": np.nan,
            "fit_r2": np.nan,
            "fit_success": False,
            "fit_t_ms": np.asarray([], dtype=float),
            "fit_y": np.asarray([], dtype=float),
        }


def empty_average_peak_features(spike_index=0):
    return {
        "spike_index": spike_index,
        "peak_trough_ratio": np.nan,
        "fwhm_ms": np.nan,
        "tau_r_ms": np.nan,
        "tau_d_ms": np.nan,
        "fit_r2": np.nan,
        "fit_success": False,
    }


def quantify_average_peak_waveform(
    spike_windows,
    fs,
    peak_index=None,
    spike_index=0,
    polarity="auto",
    baseline_slice=None,
    fit_main_lobe_only=True,
):
    """
    Calculate main-peak features from the average spike waveform.

    This helper intentionally leaves non-peak metrics as NaN. It is used
    by NeuroBox when only the main aligned peak should be quantified and shown.
    """
    spike_windows = np.asarray(spike_windows, dtype=float)
    if spike_windows.ndim != 2:
        raise ValueError("spike_windows must be a 2D array")

    n_spikes, window_len = spike_windows.shape
    if peak_index is None:
        peak_index = window_len // 2
    peak_index = int(np.clip(peak_index, 0, max(window_len - 1, 0)))

    dt_ms = 1000.0 / float(fs)
    t_ms = (np.arange(window_len) - peak_index) * dt_ms
    empty = {
        "features": empty_average_peak_features(spike_index),
        "avg_waveform": np.full(window_len, np.nan, dtype=float),
        "t_ms": t_ms,
        "fit_t_ms": np.asarray([], dtype=float),
        "fit_waveform": np.asarray([], dtype=float),
        "peak_time_ms": np.nan,
        "peak_value": np.nan,
    }
    if n_spikes == 0 or window_len == 0:
        return empty

    avg_waveform = np.nanmean(spike_windows, axis=0)
    if not np.any(np.isfinite(avg_waveform)):
        empty["avg_waveform"] = avg_waveform
        return empty

    if baseline_slice is None:
        pre_end = max(1, int(peak_index * 0.5))
        baseline_slice = slice(0, pre_end)
    baseline = np.nanmedian(avg_waveform[baseline_slice])
    peak_raw = avg_waveform[peak_index] - baseline

    if polarity == "positive":
        sign = 1.0
    elif polarity == "negative":
        sign = -1.0
    elif polarity == "auto":
        sign = 1.0 if peak_raw >= 0 else -1.0
    else:
        raise ValueError("polarity must be 'auto', 'positive', or 'negative'")

    oriented = sign * (avg_waveform - baseline)
    peak_amp = oriented[peak_index]
    if not np.isfinite(peak_amp) or peak_amp <= 0:
        empty["avg_waveform"] = avg_waveform
        empty["peak_time_ms"] = float(t_ms[peak_index])
        empty["peak_value"] = float(avg_waveform[peak_index])
        return empty

    y_norm = oriented / peak_amp
    fwhm_ms = _fwhm(t_ms, y_norm, peak_index)
    fit_result = _fit_biexp_peak_aligned(
        t_ms,
        y_norm,
        peak_index,
        fit_main_lobe_only=fit_main_lobe_only,
    )
    features = {
        "spike_index": spike_index,
        "peak_trough_ratio": np.nan,
        "fwhm_ms": fwhm_ms,
        "tau_r_ms": fit_result["tau_r_ms"],
        "tau_d_ms": fit_result["tau_d_ms"],
        "fit_r2": fit_result["fit_r2"],
        "fit_success": fit_result["fit_success"],
    }

    fit_y = np.asarray(fit_result.get("fit_y", []), dtype=float)
    fit_waveform = sign * fit_y * peak_amp + baseline if fit_y.size else fit_y
    return {
        "features": features,
        "avg_waveform": avg_waveform,
        "t_ms": t_ms,
        "fit_t_ms": np.asarray(fit_result.get("fit_t_ms", []), dtype=float),
        "fit_waveform": fit_waveform,
        "peak_time_ms": float(t_ms[peak_index]),
        "peak_value": float(avg_waveform[peak_index]),
    }


def quantify_spike_waveforms(
    spike_windows,
    fs,
    peak_index=None,
    polarity="auto",
    baseline_slice=None,
    fit_main_lobe_only=True,
):
    """
    Calculate waveform-shape features for 2p voltage imaging spike snippets.

    Parameters
    ----------
    spike_windows : array, shape = (n_spikes, window_len)
        Each row is one spike waveform, already aligned by peak.

    fs : float
        Sampling rate in Hz.

    peak_index : int or None
        Index of aligned peak. If None, use window_len // 2.

    polarity : {"auto", "positive", "negative"}
        "positive": main spike peak is positive-going.
        "negative": main spike peak is negative-going; waveform will be inverted.
        "auto": polarity decided for each waveform from value at peak_index.

    baseline_slice : slice or None
        Samples used for baseline estimation.
        If None, use the first 25% of the pre-peak region.

    fit_main_lobe_only : bool
        If True, fit bi-exponential only until the waveform first returns to baseline
        after the peak.

    Returns
    -------
    features : pandas.DataFrame
        One row per spike.

    avg_features : dict
        Same features calculated on the average normalized waveform.

    avg_norm_waveform : array
        Average baseline-corrected, polarity-oriented, peak-normalized waveform.

    t_ms : array
        Time axis in ms, with peak at t = 0.
    """
    spike_windows = np.asarray(spike_windows, dtype=float)

    if spike_windows.ndim != 2:
        raise ValueError("spike_windows must be a 2D array: n_spikes × window_len")

    n_spikes, window_len = spike_windows.shape

    if peak_index is None:
        peak_index = window_len // 2

    dt_ms = 1000.0 / fs
    t_ms = (np.arange(window_len) - peak_index) * dt_ms

    if baseline_slice is None:
        # Use early pre-peak samples as baseline
        pre_end = max(1, int(peak_index * 0.5))
        baseline_slice = slice(0, pre_end)

    rows = []
    norm_waveforms = []

    for i, wf in enumerate(spike_windows):
        baseline = np.nanmedian(wf[baseline_slice])

        peak_raw = wf[peak_index] - baseline

        if polarity == "positive":
            sign = 1.0
        elif polarity == "negative":
            sign = -1.0
        elif polarity == "auto":
            sign = 1.0 if peak_raw >= 0 else -1.0
        else:
            raise ValueError("polarity must be 'auto', 'positive', or 'negative'")

        # Oriented waveform: main spike peak becomes positive
        y = sign * (wf - baseline)

        peak_amp = y[peak_index]

        if not np.isfinite(peak_amp) or peak_amp <= 0:
            rows.append({
                "spike_index": i,
                "peak_trough_ratio": np.nan,
                "fwhm_ms": np.nan,
                "tau_r_ms": np.nan,
                "tau_d_ms": np.nan,
                "fit_r2": np.nan,
                "fit_success": False,
            })
            norm_waveforms.append(np.full(window_len, np.nan))
            continue

        # FWHM
        fwhm_ms = _fwhm(t_ms, y, peak_index)

        # Normalize shape for fitting
        y_norm = y / peak_amp
        norm_waveforms.append(y_norm)

        fit_result = _fit_biexp_peak_aligned(
            t_ms,
            y_norm,
            peak_index,
            fit_main_lobe_only=fit_main_lobe_only,
        )

        rows.append({
            "spike_index": i,
            "peak_trough_ratio": np.nan,
            "fwhm_ms": fwhm_ms,
            "tau_r_ms": fit_result["tau_r_ms"],
            "tau_d_ms": fit_result["tau_d_ms"],
            "fit_r2": fit_result["fit_r2"],
            "fit_success": fit_result["fit_success"],
        })

    import pandas as pd
    features = pd.DataFrame(rows)

    # Average normalized waveform
    norm_waveforms = np.asarray(norm_waveforms)
    avg_norm_waveform = np.nanmean(norm_waveforms, axis=0)

    # Features on average waveform
    avg_features = {
        "peak_trough_ratio": np.nan,
        "fwhm_ms": _fwhm(t_ms, avg_norm_waveform, peak_index),
    }

    avg_fit = _fit_biexp_peak_aligned(
        t_ms,
        avg_norm_waveform,
        peak_index,
        fit_main_lobe_only=fit_main_lobe_only,
    )

    avg_features.update({
        "tau_r_ms": avg_fit["tau_r_ms"],
        "tau_d_ms": avg_fit["tau_d_ms"],
        "fit_r2": avg_fit["fit_r2"],
        "fit_success": avg_fit["fit_success"],
    })

    return features, avg_features, avg_norm_waveform, t_ms
