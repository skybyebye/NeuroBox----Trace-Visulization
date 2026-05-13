import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import zscore
from scipy.signal import find_peaks,savgol_filter
from scipy.ndimage import gaussian_filter1d,uniform_filter1d, median_filter,maximum_filter1d, minimum_filter1d
from cal_wavelet import denoise_trace as pca_wavelet_trace

def extract_trace(datafolder, baseline_indices, stim_indices, framerate, alpha=0, negative=False, intensity_max=None):
    traces,spike_times,firing_rate,thresholds = [],[],[],[]
    iscell = np.load(rf'{datafolder}/suite2p/plane0/iscell.npy')[:,0].astype(bool)
    # suite2p F.npy trace = F-Fneu*alpha
    trace_f = F_trace(datafolder, alpha, negative=negative, intensity_max=intensity_max)
    
    # weighted trace
    #trace_w = weighted_trace(,mode='exp')
    
    # uniform baseline trace
    trace_uni = rolling_base_trace(trace_f,framerate,window=2.5,mode='mean')
    
    # savgol baseline trace
    #trace_sg = savgol_base_trace(trace_w,framerate)
    
    # PCA-wavelet denoised trace
    trace_w = weighted_trace(
        datafolder,
        mode='overmean',
        trace_negative=negative,
        weight_negative=negative,
        intensity_max=intensity_max,
    )
    dff = rolling_base_trace(trace_w, framerate, window=4,mode='median')
    trace_pca_wavelet,denoise_info = pca_wavelet_trace(dff,framerate)
    lowpass_trace = lowpass_filt_trace(dff, framerate, f_high=1.0)
    trace_pca_wavelet_t = trace_pca_wavelet + lowpass_trace
    
    # dFF pre_stim baseline trace
    # trace_pre = pre_sti_trace(trace, baseline_indices)
    
    # polyfit baseline trace
    # trace_poly = polyfit_trace(trace, stim_indices,pol_time=2)
    
    # highpass_filtered trace
    #trace_filt = highpass_filt_trace(trace,framerate)
    
    # normal wavelet denoised trace
    #trace_wavelet = wavelet_trace(trace_filt)
    
    #traces = [trace_f, trace_w, trace_uni, trace_sg, trace_AL, trace_pre, trace_poly, trace_filt, trace_wavelet]
    traces = [trace_w, dff, trace_pca_wavelet, lowpass_trace, trace_pca_wavelet_t]
    # 0: std/mad thres  1: snr thres
    spike_mode = [0,0,0,0,0]
    for i in range(len(traces)):
        if spike_mode[i] == 0:
            spike_times_i, firing_rate_i,thresholds_i, _snr_i = detect_spikes(traces[i],framerate,thr=7.0,mode='mad')
        elif spike_mode[i] == 1:
            snr_trace, spike_times_i, firing_rate_i,thresholds_i = detect_snr_spikes(traces[i],framerate,thr=4.0)
            traces[i] = snr_trace
        spike_times.append(spike_times_i)
        firing_rate.append(firing_rate_i)
        thresholds.append(thresholds_i)
    
    return traces,spike_times,firing_rate,thresholds,iscell


def pre_sti_trace(F, baseline_indices):
    """ calculate df/f """
    F0 = np.mean(F[:, baseline_indices], axis=1, keepdims=True)
    #F0 = np.mean(F,axis=1, keepdims=True)
    dff = (F - F0) / F0 * 100
    return dff

def highpass_filt_trace(trace,framerate, f_low=10):
    sos = butter(2, f_low, btype='highpass', fs=framerate, output='sos')
    trace = sosfiltfilt(sos, trace)
    return trace

def lowpass_filt_trace(trace, framerate, f_high=1):
    sos = butter(2, f_high, btype='lowpass', fs=framerate, output='sos')
    trace = sosfiltfilt(sos, trace)
    return trace

def polyfit_trace(F, stim_indices,pol_time=5):
    mask = np.zeros(F.shape[1], dtype=bool)
    mask[stim_indices] = True
    F_stim = F[:, mask]
    F_base = F[:, ~mask]
    F_stim_fit, F_base_fit = np.zeros(F_stim.shape), np.zeros(F_base.shape)
    x = np.linspace(-5,5,F_stim.shape[1])
    for i in range(F_stim.shape[0]):
        params = np.polyfit(x,F_stim[i],pol_time)
        F_stim_fit[i] = np.polyval(params,x)
    x = np.linspace(-1,1,F_base.shape[1])
    for i in range(F_base.shape[0]):
        params = np.polyfit(x,F_base[i],pol_time)
        F_base_fit[i] = np.polyval(params,x)
    F_fit = np.empty_like(F)
    F_fit[:, mask] = F_stim_fit
    F_fit[:, ~mask] = F_base_fit
    dff = (F - F_fit)/F_fit*100
    return dff

def wavelet_trace(trace, wavelet='sym4', min_level=4, max_scale=1.2,mode='hard'):
    import pywt
    max_level = pywt.dwt_max_level(trace.shape[1], pywt.Wavelet(wavelet).dec_len)
    level = max(min_level, max_level)
    coeffs = pywt.wavedec(trace, wavelet, level=level,axis=1)
    coeffs_thresh = [coeffs[0]]
    n_detail = len(coeffs) - 1
    
    sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1],axis=1,keepdims=True)),axis=1,keepdims=True) / 0.6745
    sigma = np.maximum(sigma, 1e-8)
    for j, c in enumerate(coeffs[1:], start=1):
        scale_j = np.exp(j-n_detail+1)*max_scale
        #scale_j = 1.2

        thr_j = scale_j * sigma
        coeffs_thresh.append(pywt.threshold(c, thr_j, mode=mode))

    y = pywt.waverec(coeffs_thresh, wavelet)
    return y[:, :trace.shape[1]]

def pix_exp(values,scale=1):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    values = zscore(values)
    z = (values-values.max())*scale
    w = np.exp(z)
    s = w.sum()
    if s < 1e-8:
        return np.full_like(values, 1.0 / len(values), dtype=np.float32)
    weights = w/s
    weights = np.clip(weights, 0.0, 1.0) 
    return weights

def pix_overmean(values):
    values = np.asarray(values,dtype=np.float32)
    weights = np.zeros_like(values)
    mean_val = np.mean(values)
    weights[values > mean_val] = 1.0
    return weights

def pix_max(values):
    values = np.asarray(values,dtype=np.float32)
    weights = np.zeros_like(values)
    mean_val = np.max(values)
    weights[values == mean_val] = 1.0
    return weights

def pix_std(values,framerate,window=10,scale=1):
    values = np.asarray(values, dtype=np.float32)
    weights = np.ones_like(values)
    if values.size == 0:
        return values
    std = np.std(values)
    window = int(window * framerate)
    baseline = gaussian_filter1d(maximum_filter1d(minimum_filter1d(values, size=window), size=window), sigma=window)
    delta = values - baseline
    std = np.std(delta(delta > 0))
    weights[delta < -scale*std] = 0.0
    return weights

def weighted_trace(datafolder, mode='exp', trace_negative=False, weight_negative=False, intensity_max=None):
    ops = np.load(rf'{datafolder}/suite2p/plane0/ops.npy', allow_pickle=True).item()
    xpix,ypix = ops['Lx'],ops['Ly']
    data = np.memmap(rf'{datafolder}/suite2p/plane0/data.bin', dtype='int16', mode='r')
    data = data.reshape(-1, ypix,xpix)
    x = float(intensity_max) if intensity_max is not None and np.isfinite(float(intensity_max)) else float(np.max(data))
    data_mean = np.mean(data, axis=0)
    weight_image = x - data_mean if weight_negative else data_mean
    roi = np.load(rf'{datafolder}/suite2p/plane0/stat.npy', allow_pickle=True)
    intensity = [weight_image[roi[i]['ypix'], roi[i]['xpix']] for i in range(len(roi))]
    if mode == 'exp':
        weights = [pix_exp(intensity[i]) for i in range(len(intensity))]
    elif mode == 'overmean':
        weights = [pix_overmean(intensity[i]) for i in range(len(intensity))]
    elif mode == 'max':
        weights = [pix_max(intensity[i]) for i in range(len(intensity))]
    elif mode == 'std':
        weights = [pix_std(intensity[i],framerate=30,window=10,scale=1) for i in range(len(intensity))]
    if trace_negative:
        F = np.array([np.sum((x - data[:, roi[i]['ypix'], roi[i]['xpix']]) * weights[i], axis=1) for i in range(len(roi))])
    else:
        F = np.array([np.sum(data[:, roi[i]['ypix'], roi[i]['xpix']] * weights[i], axis=1) for i in range(len(roi))])
    return F

def extract_weighted_roi_traces(movie, masks, chunk_size=256):
    movie = np.asarray(movie)
    masks = np.asarray(masks, dtype=np.float32)
    if movie.ndim != 3:
        raise ValueError('movie must have shape frames x height x width')
    if masks.ndim != 3:
        raise ValueError('masks must have shape rois x height x width')
    if masks.shape[1:3] != movie.shape[1:3]:
        raise ValueError(f'mask shape {masks.shape[1:3]} does not match movie shape {movie.shape[1:3]}')

    flat_masks = masks.reshape(masks.shape[0], -1).astype(np.float32, copy=False)
    traces = np.zeros((masks.shape[0], movie.shape[0]), dtype=np.float32)
    for start in range(0, movie.shape[0], int(max(chunk_size, 1))):
        stop = min(movie.shape[0], start + int(max(chunk_size, 1)))
        chunk = np.asarray(movie[start:stop], dtype=np.float32).reshape(stop - start, -1)
        traces[:, start:stop] = flat_masks @ chunk.T
    return traces

def detect_trace_event_intervals(trace_mat, framerate, value, direction='above', interval=0.0, adjacent=0.0, merge=0.0):
    trace_mat = np.asarray(trace_mat, dtype=float)
    if trace_mat.ndim == 1:
        trace_mat = trace_mat.reshape(1, -1)
    if trace_mat.ndim != 2 or trace_mat.shape[1] == 0:
        return []
    if direction == 'between':
        start = int(max(0, round(float(value))))
        stop = int(min(trace_mat.shape[1], round(float(interval))))
        if stop < start:
            start, stop = stop, start
        return [(start, stop)] if stop > start else []

    interval_frames = max(1, int(round(max(float(interval), 0.0))))
    adjacent_frames = max(0, int(round(max(float(adjacent), 0.0))))
    merge_frames = max(0, int(round(max(float(merge), 0.0))))
    hits_by_roi = trace_mat > float(value) if direction == 'above' else trace_mat < float(value)
    hits = np.any(hits_by_roi, axis=0)
    starts = np.flatnonzero(hits & np.concatenate(([True], ~hits[:-1])))
    stops = np.flatnonzero(hits & np.concatenate((~hits[1:], [True]))) + 1
    intervals = [
        (max(0, int(start) - adjacent_frames), min(trace_mat.shape[1], int(stop) + adjacent_frames))
        for start, stop in zip(starts, stops)
        if int(stop) - int(start) >= interval_frames
    ]
    if not intervals:
        return []

    merged = [intervals[0]]
    for start, stop in intervals[1:]:
        last_start, last_stop = merged[-1]
        if start - last_stop < merge_frames:
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged

def F_trace(datafolder, alpha, negative=False, intensity_max=None):
    F = np.load(rf'{datafolder}/suite2p/plane0/F.npy')
    F_neu = np.load(rf'{datafolder}/suite2p/plane0/Fneu.npy')
    trace = F - F_neu*alpha
    if negative:
        x = float(intensity_max) if intensity_max is not None and np.isfinite(float(intensity_max)) else float(np.nanmax(F))
        trace = x - trace
    return trace
    
def rolling_base_trace(trace,framerate,window=0.02,mode='mean'):
    window = int(round(window*framerate))
    if mode == 'mean':
        baseline = uniform_filter1d(trace, size=window, axis=1, mode='nearest')
    elif mode == 'median':
        baseline = median_filter(trace, size=(1,window), mode='nearest')
    dff = (trace - baseline)/(baseline+1e-8)*100
    return dff

def savgol_base_trace(trace,framerate,window=10,order=3):
    window = int(round(window*framerate))
    baseline = savgol_filter(trace, window_length=window, polyorder=order, axis=1, mode='nearest')
    dff = (trace - baseline)/(baseline+1e-8)*100
    return dff

def snr_trace(trace,framerate,window=0.0125):
    snr_trace = np.zeros_like(trace)
    spike_delta = max(1, int(round(window * framerate)))

    for i in range(trace.shape[0]):
        trace_i = trace[i, :]
        
        noise_sigma = max(np.mean(np.abs(np.diff(trace_i))), 1e-8)

        # SNR trace
        putative = np.zeros(trace.shape[1], dtype=np.float64)
        putative[spike_delta:] = trace_i[spike_delta:] - trace_i[:-spike_delta]
        snr = putative / noise_sigma
        snr_trace[i] = snr
    return snr_trace

def compute_volpy_snr(t, spikes):
    t = np.asarray(t, dtype=np.float64).copy()
    spikes = np.asarray(spikes, dtype=int).reshape(-1)
    if t.size == 0 or spikes.size == 0:
        return 0.0
    spikes = spikes[(spikes >= 0) & (spikes < t.size)]
    if spikes.size == 0:
        return 0.0

    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return np.nan
    t = np.nan_to_num(t - float(np.median(finite)), nan=0.0, posinf=0.0, neginf=0.0)
    signal = float(np.mean(t[spikes]))
    negative_part = -t[t < 0]
    if negative_part.size == 0:
        return np.nan

    scale = float(np.max(np.abs(negative_part)))
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    noise = float(scale * np.sqrt(np.mean((negative_part / scale) ** 2)))
    if not np.isfinite(noise) or noise <= 0:
        return np.nan
    return signal / noise

def detect_spikes(trace,framerate,thr=5.0,prom=3.0,refractory=0.002,width=[0.001,0.1],mode='std'):

    trace = np.asarray(trace, dtype=np.float64)
    if trace.ndim == 1:
        trace = trace.reshape(1, -1)
    trace = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)
    spike_matrix = np.zeros(trace.shape)
    spike_times = []
    snr_values = []

    def finite_median(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else 0.0

    def finite_mean(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0
        scale = float(np.max(np.abs(values)))
        if not np.isfinite(scale) or scale <= 0:
            return 0.0
        return float(scale * np.mean(values / scale))

    def finite_std(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0
        scale = float(np.max(np.abs(values)))
        if not np.isfinite(scale) or scale <= 0:
            return 0.0
        normalized = values / scale
        centered = normalized - float(np.mean(normalized))
        return float(scale * np.sqrt(np.mean(centered ** 2)))

    def finite_mad(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0
        scale = float(np.max(np.abs(values)))
        if not np.isfinite(scale) or scale <= 0:
            return 0.0
        normalized = values / scale
        med = float(np.median(normalized))
        return float(scale * np.median(np.abs(normalized - med)) / 0.6745)

    def spike_snr(trace_i, peaks, noise_level):
        peaks = np.asarray(peaks, dtype=int)
        if peaks.size == 0:
            return None
        value = compute_volpy_snr(trace_i, peaks)
        if not np.isfinite(value):
            return None
        return value

    if mode == 'snr':
        refractory_frames = max(1, int(round(refractory * framerate)))
        peak_window = max(width)
        peak_window_frames = max(1, int(round(peak_window * framerate)))
        thresholds = np.zeros(trace.shape[0])

        for i in range(trace.shape[0]):
            trace_i = trace[i, :]
            above = trace_i > thr
            crossings = np.flatnonzero(above & np.concatenate(([True], ~above[:-1])))
            peak_indices = []
            last_peak = -refractory_frames
            for crossing in crossings:
                start = int(crossing)
                stop = min(trace_i.size, start + peak_window_frames)
                if stop <= start:
                    continue
                peak = start + int(np.argmax(trace_i[start:stop]))
                if peak - last_peak < refractory_frames:
                    if peak_indices and trace_i[peak] > trace_i[peak_indices[-1]]:
                        peak_indices[-1] = peak
                        last_peak = peak
                    continue
                peak_indices.append(peak)
                last_peak = peak
            peaks = np.asarray(peak_indices, dtype=int)
            spike_matrix[i, peaks] = 1
            spike_times.append(peaks / framerate)
            thresholds[i] = thr
            snr_values.append(spike_snr(trace_i, peaks, 1.0))
    
    else:
        if mode == 'mad':
            medians = np.asarray([finite_median(row) for row in trace], dtype=np.float64)
            mask = trace < medians[:, None]
            mask = [np.where(row)[0] for row in mask]
            trace_mask = [trace[i,mask[i]] for i in range(trace.shape[0])]
            mad = np.array([finite_mad(trace_mask[i]) for i in range(len(trace_mask))], dtype=np.float64)
            #mad = np.median(abs(trace_mask-np.median(trace_mask,axis=1,keepdims=True)),axis=1)/0.6745
            with np.errstate(over='ignore', invalid='ignore'):
                thresholds = medians+thr*mad
                prom = prom*mad
            noise_values = mad
        elif mode == 'std':
            means = np.asarray([finite_mean(row) for row in trace], dtype=np.float64)
            mask = trace < means[:, None]
            mask = [np.where(row)[0] for row in mask]
            trace_mask = [trace[i,mask[i]] for i in range(trace.shape[0])]
            std = np.array([finite_std(trace_mask[i]) for i in range(len(trace_mask))], dtype=np.float64)
            with np.errstate(over='ignore', invalid='ignore'):
                thresholds = means+thr*std
                prom = prom*std
            noise_values = std
        else:
            thresholds = np.zeros(trace.shape[0], dtype=np.float64)
            prom = np.zeros(trace.shape[0], dtype=np.float64)
            noise_values = np.zeros(trace.shape[0], dtype=np.float64)
        thresholds = np.nan_to_num(thresholds, nan=np.inf, posinf=np.inf, neginf=-np.inf)
        prom = np.nan_to_num(prom, nan=0.0, posinf=np.inf, neginf=0.0)
        
        refractory_frames = max(1,int(round(refractory*framerate)))
        min_width = max(1,int(round(width[0]*framerate)))
        max_width = max(1,int(round(width[1]*framerate)))
        # prom = prom*mad
        
        
        for i in range(trace.shape[0]):
            peaks, _ = find_peaks(trace[i,:],height=thresholds[i],prominence=prom[i],distance=refractory_frames,width=(min_width,max_width))
            spike_matrix[i,peaks] = 1
            spike_times.append(peaks/framerate)
            snr_values.append(spike_snr(trace[i,:], peaks, noise_values[i]))
        
    firing_rate = generate_firingRate(spike_matrix,framerate)
    
    return spike_times,firing_rate,thresholds,snr_values

def generate_firingRate(spike_matrix, framerate, sigma=0.1):
    spike_matrix = np.asarray(spike_matrix, dtype=float)
    sigma_frames = sigma * framerate
    rate = np.zeros_like(spike_matrix)
    for i in range(spike_matrix.shape[0]):
        rate[i,:] = gaussian_filter1d(spike_matrix[i,:], sigma=sigma_frames, mode='nearest')
    rate = rate * framerate
    return rate
