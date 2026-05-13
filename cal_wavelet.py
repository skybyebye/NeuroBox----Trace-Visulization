import numpy as np
import pywt
import warnings
from dataclasses import dataclass
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.signal import fftconvolve
from scipy.ndimage import median_filter, uniform_filter1d
from numpy import argmax
from typing import Optional

@dataclass
class wavelet_cfg:
    wavelet: str = "cmor1.5-1.0"
    f_min: float = 1
    f_max: float = 500.0
    f_n: int = 500
    dff: bool = False
    base_window: float = 4.0
    base_mode: str = 'median'
    cwt_method: str = "fft"
    pca_explained_variance: Optional[float] = None
    ward_n_clusters: int = 4
    window_round: int = 2
    thres_std: float = 2.0
    thres_mask_min: float = 0.5
    thres_mask_max: float = 1.0
    event_pca: str = "svd"
    event_window_ms: float = 50.0
    event_merge_gap_ms: float = 10.0
    event_min_count: int = 3
    event_pc1_noise_max: float = 0.0
    event_attenuation_min: float = 0.5
    event_attenuation_max: float = 1.0


def normalize_event_denoise_mode(value) -> str:
    if isinstance(value, bool):
        return "pca" if value else "none"
    text = str(value).strip().lower()
    if text in {"", "none", "no", "false", "0", "off"}:
        return "none"
    if text in {"pca", "true", "1", "yes", "on"}:
        return "pca"
    if text == "svd":
        return "svd"
    raise ValueError("event_pca must be one of: none, pca, svd.")


def event_denoise_mode(cfg) -> str:
    return normalize_event_denoise_mode(getattr(cfg, "event_pca", "svd"))
    
def rolling_base_trace(trace,framerate,window=0.02,mode='mean'):
    window = int(round(window*framerate))
    if mode == 'mean':
        baseline = uniform_filter1d(trace, size=window, axis=1, mode='nearest')
    elif mode == 'median':
        baseline = median_filter(trace, size=(1,window), mode='nearest')
    dff = (trace - baseline)/(baseline+1e-8)*100
    return dff

def morlet_cwt(trace, framerate, cfg):
    # complex morlet wavelet transform
    dt = 1.0 / framerate
    freqs = np.geomspace(cfg.f_min, cfg.f_max, cfg.f_n)
    freqs = freqs / framerate
    scales = pywt.frequency2scale(cfg.wavelet, freqs)
    # coeffs: f_n * nframes
    coeffs, returned_freqs = pywt.cwt(data=trace, scales=scales, wavelet=cfg.wavelet, sampling_period=dt, method=cfg.cwt_method)
    coeffs_norm = (np.abs(coeffs) - np.abs(coeffs).mean(axis=1, keepdims=True)) / np.maximum(np.abs(coeffs).std(axis=1, keepdims=True), 1e-8)
    return coeffs, coeffs_norm, np.asarray(returned_freqs, dtype=float), np.asarray(scales, dtype=float)
    
def pca_feature(data, explained_variance=0.9):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a 2D frequency-by-frame matrix.")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=explained_variance, svd_solver='full')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
        freq_features = pca.fit_transform(data)
    n_pc = int(freq_features.shape[1])
    if n_pc <= 1:
        return freq_features, pca, max(n_pc, 1)

    pc_idx = np.arange(1, n_pc + 1)
    variance = np.nan_to_num(pca.explained_variance_, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.nanmax(variance) - np.nanmin(variance)) <= 1e-12:
        opt_k = n_pc
    else:
        kneedle = KneeLocator(pc_idx, variance, curve="convex", direction="decreasing")
        opt_k = int(kneedle.elbow) if kneedle.elbow is not None else n_pc
    opt_k = int(np.clip(opt_k, 2, n_pc))
    return freq_features, pca, opt_k


def _single_domain(n_samples, X):
    return {
        "domain_id": 0,
        "freq_idx": np.arange(n_samples, dtype=int),
        "pc1_mean": float(np.mean(X[:, 0])) if n_samples and X.shape[1] else 0.0,
    }


def _is_degenerate_feature_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.size == 0 or not np.any(np.isfinite(X)):
        return True
    finite = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.nanstd(finite)) <= 1e-12

def extract_domains_from_clusters(features, opt_k, n_clusters):
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a 2D frequency-by-feature matrix.")
    if features.shape[0] == 0 or features.shape[1] == 0:
        return [], []

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    opt_k = int(np.clip(int(opt_k), 1, features.shape[1]))
    X = features[:, :opt_k]
    scores = []
    all_labels = {}

    # Limit k to be less than n_samples (silhouette_score requires k < n_samples)
    n_samples = X.shape[0]
    if _is_degenerate_feature_matrix(X):
        return [_single_domain(n_samples, X)], []
    max_valid_k = min(n_clusters, n_samples - 1)
    if max_valid_k < 2:
        return [_single_domain(n_samples, X)], []
    
    for k in range(2, max_valid_k + 1):
        # ward clustering
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        # silhouette analysis
        s = silhouette_score(X, labels)
        scores.append(s)
        all_labels[k] = labels
    if not scores:
        return [_single_domain(n_samples, X)], []
    best_k = argmax(scores) + 2
    final_labels = all_labels[best_k]

    kept_domains = []
    discarded_domains = []

    for cid in np.unique(final_labels):
        idx = np.where(final_labels == cid)[0]
        pc1_mean = X[idx, 0].mean()

        domain = {
            "domain_id": int(cid),
            "freq_idx": idx,
            "pc1_mean": float(pc1_mean),
        }

        # if pc1_mean < 0:
        #     discarded_domains.append(domain)
        # else:
        kept_domains.append(domain)
    return kept_domains, discarded_domains

# def detect_event(coeffs, scales, freq_feature, framerate):

def cal_icwt(coeffs, scales, mask=None):
    if mask is not None:
        coeffs = coeffs * mask
    log_scales = np.log(scales)
    dlog = np.gradient(log_scales)
    weights = dlog / np.sqrt(scales)

    icwt_trace = np.real(np.sum(coeffs * weights[:, None], axis=0))
    return icwt_trace

def git_icwt_0(coeffs, scales,cfg, mask=None):
    if mask is not None:
        coeffs = coeffs * mask
    mwf = pywt.ContinuousWavelet(cfg.wavelet).wavefun()
    y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]
    r_sum = np.transpose(np.sum(np.transpose(coeffs)/ scales ** 0.5, axis=-1))
    reconstructed = r_sum * (1 / y_0)
    return reconstructed

def git_icwt_1(coeffs, scales,cfg, mask=None):
    if mask is not None:
        coeffs = coeffs * mask
    mwf = pywt.ContinuousWavelet(cfg.wavelet).wavefun()
    psi, t = mwf[0], mwf[1]
    psi0 = psi[np.argmin(np.abs(t))]
    r_sum = np.sum(np.real(coeffs) / np.sqrt(scales)[:, None], axis=0)
    reconstructed = np.real(r_sum * (1 / psi0))
    return reconstructed

def rebuild_amp(trace, rebuilt_trace):
    gain = np.vdot(rebuilt_trace, trace).real / np.vdot(rebuilt_trace, rebuilt_trace).real
    rebuilt_trace = gain * rebuilt_trace+ trace.mean()
    return rebuilt_trace

def git_icwt_2(coeffs, scales, freqs, cfg, mask=None):
    mwf = pywt.ContinuousWavelet(cfg.wavelet).wavefun()
    y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]

    def W_delta(s, n, wf, freqs):
        psi_hat_star = np.conjugate(np.fft.fft(mwf[0]))
        return np.sum([psi_hat_star[np.argmin(
            np.abs(mwf[0] - s * pywt.frequency2scale(cfg.wavelet,freq=freqs[k]))
            )] for k in range(n)]) / n

    def dj(s_arr):
        return ((np.log(s_arr[1]) - np.log(s_arr[0])) / np.log(2))

    kwargs = dict(
        n=len(scales),
        wf=mwf,
        freqs=freqs,
    )

    d_j = dj(scales)
    C_d = d_j / y_0 * np.sum([np.real(W_delta(s, **kwargs))/np.sqrt(s) for s in scales])

    if mask is not None:
        coeffs = coeffs * mask
    r_sum = np.transpose(np.sum(np.transpose(coeffs)/ scales ** 0.5, axis=-1))
    reconstructed = r_sum * (d_j / (C_d * y_0))

    # finally add the mean of the data back on ('reconstructed' is an anomaly time series)
    # reconstructed += my_signal.mean()
    return reconstructed


def candidate_event_regions(activity, threshold, framerate, cfg):
    activity = np.asarray(activity, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    if activity.size == 0:
        return [], np.asarray([], dtype=int)

    above = activity > threshold
    if not np.any(above):
        return [], np.asarray([], dtype=int)

    starts = np.flatnonzero(above & np.concatenate(([True], ~above[:-1])))
    stops = np.flatnonzero(above & np.concatenate((~above[1:], [True]))) + 1
    merge_gap = max(0, int(round(cfg.event_merge_gap_ms * framerate / 1000.0)))

    regions = []
    cur_start = int(starts[0])
    cur_stop = int(stops[0])
    for start, stop in zip(starts[1:], stops[1:]):
        start = int(start)
        stop = int(stop)
        if start - cur_stop <= merge_gap:
            cur_stop = stop
        else:
            regions.append((cur_start, cur_stop))
            cur_start = start
            cur_stop = stop
    regions.append((cur_start, cur_stop))

    peaks = []
    for start, stop in regions:
        peaks.append(start + int(np.argmax(activity[start:stop])))
    return regions, np.asarray(peaks, dtype=int)


def extract_event_windows(trace, peaks, half_window):
    trace = np.asarray(trace, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    if trace.size == 0 or peaks.size == 0:
        return np.zeros((0, 0), dtype=float)

    half_window = max(0, int(half_window))
    width = 2 * half_window + 1
    pad_mode = "reflect" if trace.size > 1 else "edge"
    padded = np.pad(trace, (half_window, half_window), mode=pad_mode)
    windows = np.zeros((peaks.size, width), dtype=float)
    for i, peak in enumerate(peaks):
        start = int(peak)
        windows[i] = padded[start:start + width]
    return windows


def event_regions_from_indices(events):
    events = np.asarray(events, dtype=int).reshape(-1)
    if events.size == 0:
        return []
    events = np.unique(events)
    splits = np.flatnonzero(np.diff(events) > 1) + 1
    groups = np.split(events, splits)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups if group.size]


def event_pca_half_window(events, event_regions, framerate, cfg):
    event_spans = list(event_regions)
    if not event_spans:
        event_spans = event_regions_from_indices(events)
    lengths = np.asarray([stop - start for start, stop in event_spans if stop > start], dtype=float)
    if lengths.size:
        return max(1, int(round(float(np.mean(lengths)))))
    return max(1, int(round(cfg.event_window_ms * framerate / 1000.0 / 2.0)))


def event_average_span_frames(events, event_regions):
    event_spans = list(event_regions)
    if not event_spans:
        event_spans = event_regions_from_indices(events)
    lengths = np.asarray([stop - start for start, stop in event_spans if stop > start], dtype=float)
    return float(np.mean(lengths)) if lengths.size else 0.0


def event_noise_gains(scores, cfg, threshold=None, inclusive: bool = False):
    scores = np.asarray(scores, dtype=float)
    if threshold is None:
        threshold = cfg.event_pc1_noise_max
    threshold = float(threshold)
    if inclusive:
        noise = np.isfinite(scores) & (scores <= threshold)
    else:
        noise = np.isfinite(scores) & (scores < threshold)
    gains = np.ones(scores.shape, dtype=float)
    if not np.any(noise):
        return noise, gains

    distance = threshold - scores[noise]
    denom = float(np.max(distance))
    if denom <= 1e-12:
        severity = np.ones_like(distance)
    else:
        severity = distance / denom

    attenuation_min = float(np.clip(cfg.event_attenuation_min, 0.0, 1.0))
    attenuation_max = float(np.clip(cfg.event_attenuation_max, attenuation_min, 1.0))
    attenuation = attenuation_min + severity * (attenuation_max - attenuation_min)
    gains[noise] = 1.0 - attenuation
    return noise, gains


def _event_pca_scores(normalized):
    pca = PCA(n_components=1, svd_solver="full")
    scores = pca.fit_transform(normalized).reshape(-1)
    template = pca.components_[0]
    mean_event = np.mean(normalized, axis=0)
    if np.dot(template, mean_event) < 0:
        template = -template
        scores = -scores
    return scores, template


def _event_svd_scores(normalized):
    _u, _s, vt = np.linalg.svd(normalized, full_matrices=False)
    template = vt[0]
    mean_event = np.mean(normalized, axis=0)
    if np.dot(template, mean_event) < 0:
        template = -template
    template = template / (np.linalg.norm(template) + 1e-12)
    scores = normalized @ template
    scores = scores / (np.linalg.norm(normalized, axis=1) + 1e-12)
    return scores, template


def event_pca_attenuate_domain_trace(domain_trace, event_regions, event_peaks, framerate, cfg, events=None):
    domain_trace = np.asarray(domain_trace, dtype=float)
    event_peaks = np.asarray(event_peaks, dtype=int)
    events = np.asarray([] if events is None else events, dtype=int)
    mode = event_denoise_mode(cfg)
    mask = np.ones(domain_trace.shape, dtype=float)
    half_window = event_pca_half_window(events, event_regions, framerate, cfg)
    average_span = event_average_span_frames(events, event_regions)
    event_info = {
        "events": events,
        "regions": event_regions,
        "peaks": event_peaks,
        "event_average_span_frames": float(average_span),
        "event_average_span_seconds": float(average_span) / max(float(framerate), 1e-12),
        "event_window_half_width": int(half_window),
        "event_window_length": int(2 * half_window + 1),
        "pc1_scores": np.asarray([], dtype=float),
        "noise_events": np.asarray([], dtype=bool),
        "attenuation_mask": mask.copy(),
        "pca_template": np.asarray([], dtype=float),
        "event_method": mode,
    }
    if mode == "none" or event_peaks.size < cfg.event_min_count:
        return domain_trace, event_info

    windows = extract_event_windows(domain_trace, event_peaks, half_window)
    win_mean = np.mean(windows, axis=1, keepdims=True)
    win_std = np.std(windows, axis=1, keepdims=True)
    valid = win_std[:, 0] > 1e-8
    if np.count_nonzero(valid) < cfg.event_min_count:
        return domain_trace, event_info

    normalized = (windows[valid] - win_mean[valid]) / win_std[valid]
    if mode == "svd":
        scores_valid, template = _event_svd_scores(normalized)
        noise_threshold = 0.0
        inclusive = True
    else:
        scores_valid, template = _event_pca_scores(normalized)
        noise_threshold = cfg.event_pc1_noise_max
        inclusive = False

    pc1_scores = np.full(event_peaks.shape, np.nan, dtype=float)
    pc1_scores[valid] = scores_valid
    noise_events, gains = event_noise_gains(pc1_scores, cfg, threshold=noise_threshold, inclusive=inclusive)
    for region, is_noise, gain in zip(event_regions, noise_events, gains):
        if is_noise:
            start, stop = region
            mask[start:stop] = np.minimum(mask[start:stop], gain)

    event_info.update(
        {
            "pc1_scores": pc1_scores,
            "noise_events": noise_events,
            "attenuation_mask": mask,
            "pca_template": template,
        }
    )
    return domain_trace * mask, event_info


def recon_domain_traces(domains, coeffs,freq_feature, freqs, scales, framerate, cfg):
    raw_traces = np.zeros((len(domains), coeffs.shape[1]), dtype=float)
    filtered_traces = np.zeros((len(domains), coeffs.shape[1]), dtype=float)
    mask_thres = np.zeros((len(domains), coeffs.shape[1]), dtype=float)
    domain_info = []
    
    for eid in range(len(domains)):
        domain = domains[eid]
        idx = domain.get('freq_idx', domain.get('freq_idx', None))
        idx = np.asarray(idx, dtype=int).reshape(-1)
        domain_coeffs = coeffs[idx,:]
        domain_freqs = freqs[idx]
        domain_scales = scales[idx]
        
        # detect event
        # raw_traces[eid,:] = cal_icwt(domain_coeffs, domain_scales)
        #events, domain_coeffs = detect_event(domain_coeffs, domain_scales, domain_freq_feature, framerate)
        
        activity = np.abs(domain_coeffs).mean(axis=0)
        max_freq = float(np.max(domain_freqs))
        window = 2*(round(framerate / max_freq)*cfg.window_round // 2)+1
        baseline = np.convolve(activity, np.ones(window, dtype=float)/window, mode='same')
        thres = baseline + cfg.thres_std*np.std(activity-baseline)
        # soft thresholding mask: above thres->1, below thres->0~0.5scaled by realtive delta
        delta = np.maximum(-(activity - baseline), 0)
        delta_max = float(np.max(delta)) if delta.size else 0.0
        if delta_max <= 1e-12:
            below_mask = np.zeros_like(delta, dtype=float)
        else:
            below_mask = np.clip(delta/delta_max, cfg.thres_mask_min, cfg.thres_mask_max)
        #detect event
        events = np.where(activity > thres)[0]
        event_regions, event_peaks = candidate_event_regions(activity, thres, framerate, cfg)
        mask_thres[eid,:] = np.where(activity > thres, 1.0, 1-below_mask)
        
        
        raw_trace = git_icwt_1(domain_coeffs, domain_scales, cfg)
        threshold_filtered_trace = git_icwt_1(domain_coeffs, domain_scales, cfg, mask=mask_thres[eid,:])
        raw_traces[eid,:] = raw_trace
        filtered_trace = threshold_filtered_trace.copy()
        event_pca_info = None
        if event_denoise_mode(cfg) != "none":
            filtered_trace, event_pca_info = event_pca_attenuate_domain_trace(
                threshold_filtered_trace,
                event_regions,
                event_peaks,
                framerate,
                cfg,
                events=events,
            )
        filtered_traces[eid,:] = filtered_trace
        #raw_traces[eid,:] = git_icwt_2(domain_coeffs, domain_scales, domain_freqs, cfg)
        #filtered_traces[eid,:] = git_icwt_2(domain_coeffs, domain_scales, domain_freqs, cfg, mask=mask_thres[eid,:])

        domain_info.append(
            {
                'eid': int(eid),
                'freq_idx': idx,
                'freqs': domain_freqs,
                'scales': domain_scales,
                'events': events,
                'event_regions': event_regions,
                'event_peaks': event_peaks,
                'event_pca': event_pca_info,
                'pc1_mean': float(domain.get('pc1_mean', 0.0)),
                'kept': bool(domain.get('kept', True)),
                'raw_trace': np.asarray(raw_trace, dtype=float).copy(),
                'threshold_filtered_trace': np.asarray(threshold_filtered_trace, dtype=float).copy(),
                'final_filtered_trace': np.asarray(filtered_trace, dtype=float).copy(),
                'mask_thres': np.asarray(mask_thres[eid,:], dtype=float).copy(),
            }
        )
    return raw_traces, filtered_traces, domain_info

def denoise_trace(traces, framerate, f_min=None, f_max=None, f_n=None, cfg=None):
    if cfg is None:
        if not f_min:
            f_min = 1.0
        if not f_max:
            f_max = framerate
        if not f_n:
            f_n = 100
        cfg = wavelet_cfg(f_min=f_min, f_max=f_max, f_n=f_n)
    elif isinstance(cfg, dict):
        cfg = wavelet_cfg(**cfg)
    else:
        cfg = wavelet_cfg(**cfg.__dict__) if hasattr(cfg, '__dict__') else cfg
    if not cfg.f_min:
        cfg.f_min = 1.0
    if not cfg.f_max:
        cfg.f_max = framerate
    if not cfg.f_n:
        cfg.f_n = 100
    denoised = np.zeros(traces.shape,dtype=float)
    denoised_info = []
    traces = np.asarray(traces, dtype=float)
    if not cfg.dff:
        traces = rolling_base_trace(traces, framerate, window=cfg.base_window, mode=cfg.base_mode)
    for i in range(traces.shape[0]):
        # complex morlet wavelet transform
        coeffs, coeffs_norm, freqs, scales = morlet_cwt(traces[i],framerate,cfg)
        # pca
        freq_feature, pca, opt_k = pca_feature(coeffs_norm, explained_variance=cfg.pca_explained_variance)
        
        # ward clustering & silhouette analysis -> domains
        f_domains, discarded_domains = extract_domains_from_clusters(freq_feature, opt_k, cfg.ward_n_clusters)
        # domain trace
        raw_domain_traces, filtered_domain_traces, domain_info = recon_domain_traces(f_domains, coeffs, freq_feature, freqs, scales, framerate, cfg)
        # indigrate domain traces
        denoised[i] = np.sum(filtered_domain_traces, axis=0)
        denoised[i] = rebuild_amp(traces[i], denoised[i])
        denoised_info.append(
            {'cell_id': int(i),
             'n_events': len(f_domains),
             'freqs': [domain['freqs'] for domain in domain_info],
             'scales': [domain['scales'] for domain in domain_info],
             'pc_1': [domain['pc1_mean'] for domain in domain_info],
             'pca_freq_feature': freq_feature,
             'elbow': opt_k,
             'domains': domain_info,
             }
        )
    denoised
    return denoised, denoised_info
    

if __name__ == "__main__":
    n_cells = 3
    n_frames = 1000
    framerate = 400
    traces = np.zeros((n_cells,n_frames))
    
    
    traces = denoise_trace(traces, framerate)
