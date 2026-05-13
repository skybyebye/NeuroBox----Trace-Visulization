from pathlib import Path
import numpy as np
import pandas as pd
import os
import glob
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import tifffile
from scipy.signal import butter, sosfiltfilt
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from scipy.signal import find_peaks


def extract_galvo_positions(gpl_file):
    """Extract galvo positions from GPL file."""
    tree = ET.parse(gpl_file)
    root = tree.getroot()

    galvo_positions = []
    for point in root.findall('.//PVGalvoPoint'):
        x = float(point.get('X'))
        y = float(point.get('Y'))
        galvo_positions.append((x, y))

    return np.array(galvo_positions)

def _find_tif_for_metadata(metadata_file):
    path = Path(metadata_file)
    folder = path.parent
    basename = folder.name
    preferred = folder / f'{basename}_Cycle00001_Ch2_000001.ome.tif'
    if preferred.exists():
        return preferred
    tif_files = sorted(folder.glob('*.tif')) + sorted(folder.glob('*.tiff'))
    return tif_files[0] if tif_files else None

def _frame_shape_from_tif(tif_path):
    with tifffile.TiffFile(tif_path) as tif:
        shape = tuple(int(v) for v in tif.series[0].shape)
    if len(shape) < 2:
        return None
    if len(shape) >= 3 and shape[-1] in (3, 4):
        return int(shape[-3]), int(shape[-2])
    return int(shape[-2]), int(shape[-1])

def extract_converter(datafolder):
    converter_file = Path(datafolder).parent / 'corners_for_converter.gpl'
    if not converter_file.exists():
        return None
    tree = ET.parse(converter_file)
    root = tree.getroot()
    xs = []
    ys = []
    for point in root.findall('.//PVGalvoPoint'):
        x = point.get('X')
        y = point.get('Y')
        if x is not None and y is not None:
            xs.append(float(x))
            ys.append(float(y))
    if not xs or not ys:
        return None
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    neg_x = xs[xs < 0]
    pos_x = xs[xs > 0]
    neg_y = ys[ys < 0]
    pos_y = ys[ys > 0]
    if neg_x.size == 0 or pos_x.size == 0 or neg_y.size == 0 or pos_y.size == 0:
        return None
    return {
        'x_min': float(np.mean(neg_x)),
        'x_max': float(np.mean(pos_x)),
        'y_min': float(np.mean(neg_y)),
        'y_max': float(np.mean(pos_y)),
    }

# convert the galvo position to pixel position
def galvo_to_pixel(x_galvo, y_galvo, datafolder=None, frame_width=512, frame_height=512):
    """Convert galvo coordinates to pixel coordinates."""
    # Calibration: linear mapping from galvo to pixel coordinates
    # X: -2.279 V → 0 px, 1.924 V → 512 px
    # Y: -2.115 V → 0 px, 2.088 V → 512 px
    # !!! Note: this is for resonant 2X zoom in. 
    converter = extract_converter(datafolder) if datafolder is not None else None
    if converter is None:
        x_min, x_max = -2.27914416925, 1.92441355925
        y_min, y_max = -2.11542234925, 2.08402632966202
    else:
        x_min = converter['x_min']
        x_max = converter['x_max']
        y_min = converter['y_min']
        y_max = converter['y_max']
    
    xpix = (x_galvo - x_min) / (x_max - x_min) * float(frame_width)
    ypix = (y_galvo - y_min) / (y_max - y_min) * float(frame_height)
    return int(xpix), int(ypix)

def extract_roi_pixel_position(roi_file_path):
    """
    .roi file defines the imaging scanning region. 
    This function extracts the 4 corner coordinates of the scanning region,
    convert them to pixel positions, and return as a dict.
    """
    tree = ET.parse(roi_file_path)
    root = tree.getroot()

    # Find the PVROI element with mode="ResonantGalvo"
    pvroi = None
    for elem in root.iter('PVROI'):
        if elem.get('mode') == 'ResonantGalvo':
            pvroi = elem
            break

    if pvroi is None:
        raise ValueError("No PVROI with mode='ResonantGalvo' found in the .roi file.")

    # Extract the 4 corner points
    corners = {}
    for corner_tag in ['UpperLeft', 'UpperRight', 'LowerLeft', 'LowerRight']:
        point_elem = pvroi.find(corner_tag)
        if point_elem is None:
            raise ValueError(f"Missing {corner_tag} in PVROI.")
        x_galvo = float(point_elem.get('X'))
        y_galvo = float(point_elem.get('Y'))
        xpix, ypix = galvo_to_pixel(x_galvo, y_galvo)
        corners[corner_tag] = (xpix, ypix)

    return corners

def convert_stim_coords_to_roi(xpix, ypix, corners):
    """
    Convert stimulation pixel coordinates from full 512×512 space to ROI-relative coordinates.

    Parameters
    ----------
    xpix, ypix : float
        Stimulation coordinates in the full 512×512 pixel space.
    corners : dict
        Dictionary with keys 'UpperLeft', 'UpperRight', 'LowerLeft', 'LowerRight'
        containing (x, y) tuples of the ROI corners in pixel space.

    Returns
    -------
    xrel, yrel : float
        Stimulation coordinates relative to the ROI region.
    """
    # Extract bounding box of the ROI
    x_min = min(corners['UpperLeft'][0], corners['LowerLeft'][0])
    x_max = max(corners['UpperRight'][0], corners['LowerRight'][0])
    y_min = min(corners['UpperLeft'][1], corners['UpperRight'][1])
    y_max = max(corners['LowerLeft'][1], corners['LowerRight'][1])

    # Shift stimulation coordinates to ROI-relative
    xrel = xpix - x_min
    yrel = ypix - y_min    

    return int(xrel), int(yrel)


def extract_imaging_parameters(metadata_file):
    """Extract frame rate and frame dimensions from XML metadata file."""
    datafolder = Path(metadata_file).parent
    tif_path = _find_tif_for_metadata(metadata_file)
    tif_shape = None
    if tif_path is not None:
        try:
            tif_shape = _frame_shape_from_tif(tif_path)
        except Exception:
            tif_shape = None
    tree = ET.parse(metadata_file)
    root = tree.getroot()
    p_tree = {c:p for p in root.iter() for c in p}
    
    frame_rate = None
    frame_width = int(tif_shape[1]) if tif_shape is not None else 512
    frame_height = int(tif_shape[0]) if tif_shape is not None else 41
    corners = {}
    
    for state_value in root.findall('.//PVStateValue'):
        key = state_value.get('key')
        value = state_value.get('value')
        if key == 'framerate':
            frame_rate = float(value)
        elif key == 'linesPerFrame' and int(value) == frame_height:
            fov = p_tree[p_tree[state_value]]
            for corner_tag in ['UpperLeft', 'UpperRight', 'LowerLeft', 'LowerRight']:
                point_elem = fov.find(corner_tag)
                if point_elem is None:
                    raise ValueError(f"Missing {corner_tag} in PVROI.")
                x_galvo = float(point_elem.get('X'))
                y_galvo = float(point_elem.get('Y'))
                xpix, ypix = galvo_to_pixel(
                    x_galvo,
                    y_galvo,
                    datafolder=datafolder,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                corners[corner_tag] = (xpix, ypix)
            
            

    # print(f"Frame size: {frame_width} x {frame_height} pixels")
    # print(f"Frame rate: {frame_rate:.2f} Hz")
    
    return frame_rate, frame_width, frame_height, corners



def extract_stim_times(stim_xml, ITI = 10,threshold = 50): 
    # ITI is the inter-trial interval in seconds, threshold is the voltage threshold for burst detection
    
    tree = ET.parse(stim_xml)
    root = tree.getroot()

    # Extract metadata
    rate = float(root.find('.//Rate').text)
    data_filename = root.find('.//DataFile').text
    file_path = Path(stim_xml).parent / data_filename
    raw_data = np.fromfile(file_path, dtype='int16')
    voltage_signal = raw_data * -0.5
    above = -voltage_signal > threshold

    crossings = np.where(np.diff(above.astype(int)) == 1)[0] # rising edge
    onset_times = crossings / rate

    crossings = np.where(np.diff(above.astype(int)) == -1)[0] # falling edge
    offset_times = crossings / rate

    duration = offset_times - onset_times

    return onset_times, offset_times, duration, rate

def get_trial_times_ori(onset_times, offset_times, ITI):
    filtered_onset_times = []
    last_onset_time = -np.inf
    for t_on in onset_times:
        if t_on - last_onset_time >= ITI:
            filtered_onset_times.append(t_on)
            last_onset_time = t_on
    onset_times_trial = np.array(filtered_onset_times)

    filtered_offset_times = []
    last_offset_time = np.inf
    for t_off in reversed(offset_times):
        if last_offset_time - t_off >= ITI:
            filtered_offset_times.append(t_off)
            last_offset_time = t_off
    offset_times_trial = np.array(filtered_offset_times)[::-1]
    duration = offset_times_trial - onset_times_trial
    
    return onset_times_trial, offset_times_trial, duration

def get_trial_times(onset_times, offset_times, ITI=2):
    onset_times_trial = [onset_times[0]]
    offset_times_trial = []
    onset_times_trial.extend(onset_times[1:][(np.diff(onset_times) > ITI)])
    offset_times_trial.extend(offset_times[0:-1][np.diff(offset_times) > ITI])
    offset_times_trial = np.append(offset_times_trial, offset_times[-1])
    duration = offset_times_trial - onset_times_trial
    return np.array(onset_times_trial), np.array(offset_times_trial), np.array(duration)

def get_trace_indices(onset_times, offset_times, frame_rate, duration):
    # duration: baseline = onset_time-duration : onset_time
    trace_indices = []
    for i in range(onset_times.shape[0]):
        t_on = onset_times[i]
        t_off = offset_times[i]
        indices = np.arange(int((t_on+duration[0])*frame_rate), int((t_off+duration[1])*frame_rate))
        indices = indices[0:int(np.floor((t_off-t_on+duration[1]-duration[0])*frame_rate))]
        trace_indices.extend(indices)
    return np.array(trace_indices)

def cal_sti_cell(suite2pfolder,cells,stim_pos_rel):
    # stim_cells: 1 -> stimulated ROI, 0 -> unstimulated ROI. Do not change cells.
    stat = np.load(rf'{suite2pfolder}/stat.npy', allow_pickle=True)
    x_roi = ([stat[i]['xpix'] for i in range(len(stat))])
    y_roi = ([stat[i]['ypix'] for i in range(len(stat))])
    stim_cells = np.zeros(len(stat), dtype=int)
    for _i,(x,y) in enumerate(stim_pos_rel):
        cell_i = [i for i,x_r in enumerate(x_roi) if np.any(x_r == x)]
        cell_j = [j for j,y_r in enumerate(y_roi) if np.any(y_r == y)]
        cell_ij = list(set(cell_i) & set(cell_j))
        if len(cell_ij) == 1:
            stim_cells[cell_ij[0]] = 1
    return stim_cells

def cal_stim_cell(suite2pfolder,cells,stim_pos_rel):
    return cal_sti_cell(suite2pfolder,cells,stim_pos_rel)
