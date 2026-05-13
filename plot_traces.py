import util
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import os



slider_refs = []

def plot_roi_window(datafolder,frame_rate,trace,cells,onset_times_trial,offset_times_trial, trial_duration,plot_duration=(-0.2,0.5),saveFig='',stim_cells=None):
    if saveFig == 'waveforms':
        plot_indices = [[] for _ in range(len(trace))]
        for j in range(len(trace)):
            plot_indices[j] = [[] for _ in range(cells.shape[0])]
            for i in range(cells.shape[0]):
                spike_t = onset_times_trial[j][i]
                plot_indices[j][i] = util.get_trace_indices(spike_t-trial_duration/2, spike_t+trial_duration/2, frame_rate, plot_duration)
                nTrials = onset_times_trial[j][i].shape[0]
                if nTrials == 0:
                    continue
                if plot_indices[j][i].shape[0]%nTrials != 0:
                    raise ValueError("The number of plot indices is not divisible by the number of trials.")
                plot_indices[j][i] = plot_indices[j][i].reshape(nTrials,-1)
                x_axis = np.arange(plot_indices[j][i].shape[1])/frame_rate + plot_duration[0]
        
    else:
        plot_indices = util.get_trace_indices(onset_times_trial, offset_times_trial, frame_rate, plot_duration)
        nTrials = len(onset_times_trial)
        if plot_indices.shape[0]%nTrials != 0:
            raise ValueError("The number of plot indices is not divisible by the number of trials.")
        plot_indices = plot_indices.reshape(len(onset_times_trial),-1)
        x_axis = np.arange(plot_indices.shape[1])/frame_rate + plot_duration[0]

    cells = np.asarray(cells, dtype=int)
    stim_cells = np.zeros(cells.shape, dtype=int) if stim_cells is None else np.asarray(stim_cells, dtype=int)
    for i in range(cells.shape[0]):
        if cells[i] != 1:
            continue
        else:
            #trace_name = ['F','weighted trace', 'baseline dff', 'polynomial fit dff', 'wavelet denoised trace']
            fig,axes = plt.subplots(len(trace),1,figsize=(10,1.5*len(trace)),sharex=True)
            if len(trace) == 1:
                axes = [axes]
            for f in range(len(trace)):
                ax = axes[f]
                if saveFig == 'waveforms':
                    trace_i = trace[f][i,:]
                    trace_i = np.concatenate([trace_i, np.zeros(x_axis.shape[0])], axis=0)
                    if plot_indices[f][i].shape[0] == 0:
                        trace_i = np.zeros((x_axis.shape[0],1))
                    else:
                        trace_i = trace_i[plot_indices[f][i]].T
                else:
                    trace_i = trace[f][i,:]
                    trace_i = np.concatenate([trace_i, np.zeros(x_axis.shape[0])], axis=0)
                    trace_i = trace_i[plot_indices].T
                ax.plot(x_axis,trace_i, color='gray', alpha=0.5)
                if saveFig == 'firing_rate':
                    ax.plot(x_axis, np.mean(trace_i, axis=1), color='k', linewidth=2)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Intensity')
                #ax.set_title(trace_name[f])
                # no stimulation
                ax.axvspan(0, trial_duration, facecolor='red', alpha=0.3, edgecolor='none', linewidth=0)
                #ax.scatter(spike_times[f][i], np.ones_like(spike_times[f][i])*max(trace_i)*1.1,s=10, color='blue', linewidth=0)
                if i < len(stim_cells) and stim_cells[i] == 1:
                    ax.hlines(max(trace_i.flatten())*1.1, 0, trial_duration, color='red', linewidth=2, label='Stimulation')
            fig.tight_layout()
           
            if saveFig:
                outpath = rf"{datafolder}\Results"
                os.makedirs(outpath,exist_ok=True)  
                plt.savefig(os.path.join(outpath, f'cell_{i}_{saveFig}.tiff'), dpi=300)
    #plt.show()
               
def plot_roi_trace(datafolder,frame_rate,trace,cells=None,onset_times_trial=None,offset_times_trial=None,spike_times=None,thresholds=None, window=5,saveFig='',stim_cells=None):
    
    for i in range(trace[0].shape[0]):
        if cells is not None and i < len(cells) and int(cells[i]) != 1:
            continue
        trace_name = ['weighted raw trace', 'baseline dff', 'pca_wavelet', '~1 Hz', 'pca_wavelet + ~1 Hz']
        fig, axs = plt.subplots(len(trace), 1, figsize=(10, 1.5 * len(trace)), sharex=True)
        if len(trace) == 1:
            axs = [axs]
        for f in range(len(trace)):
            ax = axs[f]
            trace_i = trace[f][i,:]
            t = np.arange(trace_i.shape[0])/frame_rate
            ax.plot(t, trace_i, color='k')
            if onset_times_trial is not None and offset_times_trial is not None:
                for j in range(len(onset_times_trial)):
                    ax.axvspan(onset_times_trial[j], offset_times_trial[j], facecolor='red', alpha=0.3, edgecolor='none', linewidth=0)
            if spike_times:
                ax.scatter(spike_times[f][i], np.ones_like(spike_times[f][i])*max(trace_i)*1.1,s=10, color='blue', linewidth=0)
            ax.set_xlim(0,window)
            if thresholds and thresholds[f] is not None:
                ax.plot(t, np.ones_like(trace_i)*thresholds[f][i], color='blue', linestyle='--', label='Threshold')
            ax.set_title(trace_name[f])
            
        fig.tight_layout()
            
        ax_slider = fig.add_axes([0.1, 0.03, 0.8, 0.03])
        slider = Slider(ax=ax_slider,label='Time(s)',valmin=0,valmax=max(0, trace_i.shape[0]/frame_rate - window),)

        def update(val, axs=axs, slider=slider, window=window, fig=fig):
            start = slider.val
            for ax in axs:
                ax.set_xlim(start, start + window)
            fig.canvas.draw_idle()

        slider.on_changed(update)
        slider_refs.append(slider) 
        
        if saveFig:
            outpath = rf"{datafolder}\Results"
            os.makedirs(outpath,exist_ok=True) 
            #mpld3.save_html(fig, rf"{datafolder}\Results\cell_{i}_{saveFig}.html")
    
    #plt.show()
    
def plot_all_traces(datafolder,trace,cells,frame_rate,onset_times_trial,offset_times_trial,window=10,offset=500,saveFig='',stim_cells=None):

    cells = np.asarray(cells, dtype=int)
    stim_cells = np.zeros(cells.shape, dtype=int) if stim_cells is None else np.asarray(stim_cells, dtype=int)
    index = np.where(cells == 1)[0]
    cells = cells[index]
    stim_cells = stim_cells[index]
    x = np.arange(trace.shape[1])/frame_rate
    fig,ax = plt.subplots()
    index_unsti = np.where(stim_cells == 0)[0]
    index_sti = np.where(stim_cells == 1)[0]
    
    ax.plot(x,trace[index_unsti,:].T+index_unsti*offset,linewidth=0.5,alpha=0.3)
    ax.plot(x,trace[index_sti,:].T+index_sti*offset,linewidth=2)
    #ax.plot(x,trace[index,:].T+np.arange(index.size)*offset,color='red')
    ax.set_xlim(0,window)
    for j in range(len(onset_times_trial)):
        ax.axvspan(onset_times_trial[j], offset_times_trial[j], facecolor='red', alpha=0.3, edgecolor='none', linewidth=0)
        
    sliderFig = fig.add_axes([0.1,0.03,0.8,0.03])
    slider = Slider(ax=sliderFig,label='Time(s)',valmin=0,valmax=max(0, trace.shape[1]/frame_rate- window))

        
    def update(val, ax=ax, fig=fig, slider=slider, window=window):
        start = slider.val
        ax.set_xlim(start, start + window)
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    
    if saveFig:
        outpath = rf"{datafolder}\Results"
        os.makedirs(outpath,exist_ok=True) 
        #mpld3.save_html(fig, rf"{datafolder}\Results\all_traces_{saveFig}.html")
        
    plt.show()
