# frame_rate,trace,cells,stim_cells,onset_times_trial,offset_times_trial,trial_duration,firing_rate = get_params(datafolder)

import util
import trace_process
import numpy as np
from pathlib import Path


def get_params(datafolder, negative=False, intensity_max=None):
    path = Path(datafolder)
    basename = path.name
    frame_rate, frame_width, frame_height, corners = util.extract_imaging_parameters(rf'{datafolder}/{basename}.env')
    
    
    # stimulus time
    stim_xml = Path(datafolder) / f'{basename}_Cycle00001_VoltageRecording_001.xml'
    onset_times_trial = None
    offset_times_trial = None
    trial_duration = None
    if stim_xml.exists():
        onset_times, offset_times, stim_duration, rate = util.extract_stim_times(stim_xml)
        if len(onset_times) and len(offset_times):
            onset_times_trial, offset_times_trial, trial_duration_values = util.get_trial_times(onset_times, offset_times, ITI=2)
            if np.allclose(trial_duration_values, trial_duration_values[0]):
                trial_duration = trial_duration_values[0]
            else:
                #raise ValueError("Trial durations are not consistent across trials.")
                trial_duration = min(trial_duration_values)
    if onset_times_trial is None or offset_times_trial is None:
        baseline_indices = np.asarray([], dtype=int)
        stim_indices = np.asarray([], dtype=int)
    else:
        baseline_indices = util.get_trace_indices(onset_times_trial, onset_times_trial, frame_rate, duration = (-2,0))
        stim_indices = util.get_trace_indices(onset_times_trial, offset_times_trial, frame_rate, duration = (0, 0))

    # traces
    trace,spike_times,firing_rate,thresholds,cells = trace_process.extract_trace(
        datafolder,
        baseline_indices,
        stim_indices,
        frame_rate,
        negative=negative,
        intensity_max=intensity_max,
    )
    cells = np.asarray(cells, dtype=int)

    # stimulus position
    group_path = Path(datafolder) / 'group1.gpl'
    if group_path.exists() and corners:
        stim_pos = util.extract_galvo_positions(group_path)
        stim_pos_rel = [] # stimulus position in ROI-relative pixel
        for (x,y) in stim_pos:
            x,y = util.galvo_to_pixel(x, y, datafolder=datafolder, frame_width=frame_width, frame_height=frame_height)
            xrel,yrel = util.convert_stim_coords_to_roi(x,y,corners)
            stim_pos_rel.append((xrel,yrel))
        stim_pos_rel = np.array(stim_pos_rel)
        stim_cells = util.cal_sti_cell(rf'{datafolder}/suite2p/plane0',cells,stim_pos_rel)
    else:
        stim_cells = np.zeros(len(cells), dtype=int)
    

    return frame_rate,trace,cells,stim_cells,onset_times_trial,offset_times_trial, trial_duration,spike_times,firing_rate,thresholds
