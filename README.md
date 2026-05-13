# NeuroBox

NeuroBox is a PyQt6-based desktop GUI and batch-processing tool for visualizing and analyzing two-photon voltage imaging data. It can load Bruker/Suite2p folders, Femtonics Excel trace tables, and VolPy output folders; display image layers and ROI masks; extract and process ROI traces; run denoising/motion-correction backends; detect spikes; and export spike-centered waveform summaries.

> This repository is intended for research data exploration and pipeline prototyping. Please validate every processing step with your own control data before using exported results for quantitative analysis.

## Main features

- Load imaging datasets from Bruker-style folders with Suite2p outputs.
- Load Femtonics `.xlsx` trace tables.
- Load VolPy folders and VolPy trace keys such as `ts`, `t`, `dFF`, `t_sub`, and `t_rec`.
- Display raw, VolPy, NoRMCorre, PMD, and local TIFF/mmap image layers.
- Display video frames, z-average images, max projections, and correlation images.
- Overlay ROI masks from Suite2p, VolPy, and pixel-weighted ROI maps.
- Build multiple trace rows with optional low-pass/high-pass filtering, wavelet denoising, PCA-wavelet denoising, SNR calculation, baseline correction, and spike detection.
- Save and reload NeuroBox pipeline settings as `pipeline_<dataset>.npy`.
- Export image, trace, and average panels as figures.
- Export spike-centered waveform windows and waveform features.
- Run non-interactive single-dataset or batch processing from the command line.

## Repository layout

A typical repository should look like this:

```text
NeuroBox/
├── NeuroBox.py              # main GUI and CLI entry point
├── README.md
├── requirements.txt         # optional pip package list
├── environment.yml          # recommended conda environment file
├── cal_params.py            # local data-loading / parameter functions
├── mask_weight.py           # local ROI mask / weight functions
├── trace_process.py         # local trace processing and spike detection functions
├── cal_wavelet.py           # local PCA-wavelet denoising functions
├── plot_wavelet_pca.py      # local PCA-wavelet preview plotting functions
├── cal_waveform.py          # local waveform feature extraction functions
├── normcorre.py             # local NoRMCorre backend wrapper
├── pmd_denoise.py           # local PMD backend wrapper
└── util.py                  # local utilities
```

`NeuroBox.py` imports these local modules directly, so they must be placed in the same folder or otherwise available on `PYTHONPATH`.

## Installation

### Option A: create a conda environment from scratch

```bash
conda create -n neurobox python=3.11 -y
conda activate neurobox

# Core packages used directly by NeuroBox.py
conda install -c conda-forge numpy scipy pandas openpyxl tifffile matplotlib pyqt -y

# Optional but recommended for image/correlation utilities
pip install opencv-python
```
MAY BE SOME LACKAGE

If your local `normcorre.py` or `pmd_denoise.py` depends on additional packages such as CaImAn, masknmf, PyTorch, or other local toolboxes, install those separately in the same environment.

### Option B: install from exported files

If this repository includes both files:

```bash
conda env create -f environment.yml
conda activate neurobox
pip install -r requirements.txt
```

If the environment file already contains a `pip:` section, the second command may not be necessary.

## Exporting your current environment

Activate the environment that can already run NeuroBox:

```bash
conda activate neurobox
```

For GitHub sharing, the most useful setup is usually:

```bash
# Conda packages explicitly installed by you; more portable across machines
conda env export --from-history > environment.yml

# Pip-installed packages
python -m pip freeze > requirements.txt
```

For an exact backup of the current machine environment:

```bash
conda env export > environment-full.yml
```

For a conda-compatible requirements text file:

```bash
conda export --format=requirements --file conda-requirements.txt
```

Notes:

- `requirements.txt` created by `pip freeze` is for `pip install -r requirements.txt`.
- `conda-requirements.txt` created by `conda export --format=requirements` is for conda-style package specs.
- `environment-full.yml` may contain platform-specific packages and a local `prefix:` path. Remove the `prefix:` line before publishing if you want others to recreate the environment more easily.

## Quick start

Run the GUI:

```bash
python NeuroBox.py
```

Run the GUI and immediately load one dataset:

```bash
python NeuroBox.py showGUI=True data="/path/to/bruker_or_suite2p_folder"
```

Run the GUI and apply a saved pipeline:

```bash
python NeuroBox.py showGUI=True data="/path/to/dataset" pipeline="/path/to/pipeline_dataset.npy"
```

## Command-line usage

`NeuroBox.py` accepts arguments in `key=value` format:

```bash
python NeuroBox.py showGUI=True/False batch=True/False pipeline="none" data="/path/to/data"
```

Supported keys:

| Argument | Default | Meaning |
|---|---:|---|
| `showGUI` | `True` | Start the PyQt6 GUI. Use `False` for headless processing. |
| `batch` | `False` | Process all valid child datasets in a parent folder. Only valid when `showGUI=False`. |
| `pipeline` | `none` | Path to a saved NeuroBox pipeline `.npy` file. Use `none`, `None`, or `null` for no pipeline. |
| `data` | `None` | Dataset folder, `.xlsx` file, or parent folder for batch mode. Required when `showGUI=False`. |

Examples:

```bash
# GUI only
python NeuroBox.py

# GUI with one Bruker/Suite2p folder
python NeuroBox.py showGUI=True data="D:/data/example_folder"

# GUI with one Excel trace table
python NeuroBox.py showGUI=True data="D:/data/traces.xlsx"

# Headless processing of one dataset
python NeuroBox.py showGUI=False batch=False data="D:/data/example_folder" pipeline="D:/data/pipeline_example.npy"

# Headless batch processing of all child Bruker folders and .xlsx files
python NeuroBox.py showGUI=False batch=True data="D:/data/parent_folder" pipeline="D:/data/pipeline_example.npy"
```

Important behavior:

- Arguments must use `key=value` syntax.
- `batch=True` is only allowed when `showGUI=False`.
- `showGUI=False` requires `data=<path>`.
- `pipeline=<path>` must point to an existing file unless it is `none`.
- In batch mode, NeuroBox scans the parent folder for child `.xlsx` files and Bruker/Suite2p-style data folders.

## Input data

### Bruker/Suite2p folder

A folder is treated as an imaging dataset when it contains a TIFF movie and/or Suite2p outputs. NeuroBox looks for Suite2p-style files such as:

```text
<dataset>/suite2p/plane0/ops.npy
<dataset>/suite2p/plane0/stat.npy
```

It also searches for TIFF files and related VolPy/NoRMCorre/PMD outputs near the dataset folder.

### Femtonics Excel table

An `.xlsx` file should contain:

- first column: time axis
- remaining columns: ROI traces

NeuroBox reads the file with `pandas.read_excel(..., engine='openpyxl')`.

### VolPy folder

In GUI mode, a VolPy folder can be loaded directly. NeuroBox searches for VolPy result `.npy` files and associated TIFF/mmap movies. For trace display, it can use VolPy keys including `ts`, `t`, `dFF`, `t_sub`, and `t_rec`.

## GUI workflow

1. Click **Load** and choose a Bruker folder, Femtonics `.xlsx` file, or VolPy folder.
2. Use the **Image panel** to choose image layers such as Raw, VolPy, NoRMCorre, PMD, or Local.
3. Choose image display mode: video, z-average, max projection, or correlation image.
4. Select ROI mask sources and ROI display mode.
5. Add one or more trace rows in the **Trace panel**.
6. Configure filtering, baseline correction, denoising, SNR calculation, spike detection, and waveform display.
7. Use the **Average panel** to view average event responses, spike waveforms, or firing rates.
8. Save a reusable processing setup with **Save pipeline**.
9. Export figures or waveform `.npy` files as needed.

## Saved pipelines

When you click **Save pipeline**, NeuroBox saves:

```text
pipeline_<dataset_name>.npy
```

The pipeline stores GUI state such as data polarity, image layers, events, trace processing settings, average panel settings, ROI selections, and panel display options. A saved pipeline can be loaded later in the GUI or passed to the CLI for headless processing.

## Output files

### Figure exports

The GUI can export image, trace, and average panels as:

```text
.png
.pdf
.svg
```

### Image layer exports

Image layers can be exported as TIFF files:

```text
image_<layer_id>_<mode>_<dataset_name>.tiff
```

### Waveform exports

Spike-centered waveform files are saved as:

```text
waveform_<trace_index>_<dataset_name>.npy
waveform_raw_<trace_index>_<dataset_name>.npy
```

Each waveform file contains a dictionary with a `spikes` entry:

```python
{
    "spikes": {
        "nROI": int,
        "nSpikes": np.ndarray,
        "waveform": list[np.ndarray],
        "average_fr": np.ndarray,
        "snr": np.ndarray,
        "window": np.ndarray,
        "spike_time": list[np.ndarray],
        "waveform_features": list[dict],
        "framerate": float,
        "peak": int,
        "source": str,
        "waveform_source": str,
        "k": float | None,
        "pipeline": str,
    }
}
```

`waveform_features` contains fields such as:

```python
{
    "spike_index": int,
    "peak_trough_ratio": float,
    "fwhm_ms": float,
    "tau_r_ms": float,
    "tau_d_ms": float,
    "fit_r2": float,
    "fit_success": bool,
}
```

### Headless results

Headless processing writes results to a `results/` folder near the input data.

Single-dataset mode writes:

```text
results/
├── summary.npy
└── <dataset_name>/
    ├── summary_<dataset_name>.npy
    └── waveform_*.npy
```

Batch mode writes:

```text
results/
├── batch_summary.npy
├── <dataset_1>/
│   ├── summary_<dataset_1>.npy
│   └── waveform_*.npy
└── <dataset_2>/
    ├── summary_<dataset_2>.npy
    └── waveform_*.npy
```

## Cache and generated files

NeuroBox may generate cache and intermediate files such as:

```text
.neurobox_cache/
.neurobox_temp/
pipeline_<dataset>.npy
image_info.npy
normcorre_<dataset>.tiff
pmd_<dataset>.tiff
waveform_*.npy
results/
```

Do not commit large raw data, processed movies, caches, or batch results unless you intentionally want them in the repository.

A useful `.gitignore` section is:

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/

# local data and generated outputs
*.tif
*.tiff
*.mmap
*.npy
*.npz
*.h5
*.hdf5
*.avi
*.mp4
.neurobox_cache/
.neurobox_temp/
results/

# keep small config examples if needed
!example_pipeline.npy
```

## Development notes

- The GUI uses PyQt6 with Matplotlib QtAgg embedding.
- Long-running loading and denoising tasks are routed through `QThread` workers to keep the GUI responsive.
- NoRMCorre and PMD are called through local backend modules: `normcorre.py` and `pmd_denoise.py`.
- VolPy mmap loading supports CaImAn/VolPy-style filenames with encoded dimensions when available.
- Current headless loading supports Bruker/Suite2p folders and `.xlsx` trace tables. Direct VolPy-folder loading is available in GUI mode.

## Troubleshooting

### `ModuleNotFoundError: No module named ...`

Make sure all local NeuroBox helper files are in the same folder as `NeuroBox.py`, or add their folder to `PYTHONPATH`.

### PyQt6 or Matplotlib backend error

Check that PyQt6 is installed in the active environment and that you are not running the GUI from a headless terminal without display support.

### `.xlsx` loading fails

Install `openpyxl`:

```bash
pip install openpyxl
```

### PMD or NoRMCorre fails

Check the dependencies and default parameters inside `pmd_denoise.py` or `normcorre.py`. These backend modules may require additional packages beyond the core GUI dependencies.

## License

Add a license before making the repository public. For academic/research code, common choices include MIT, BSD-3-Clause, GPL-3.0, or a custom lab license.


## REFERENCE

 - For VolPy process (SpikePersuit algorithm):

 - For PMD analysis:
  