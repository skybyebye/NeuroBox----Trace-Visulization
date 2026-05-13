from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tifffile


WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


MOTION_PARAMS: dict[str, Any] = {
    "pw_rigid": False,
    "gSig_filt": (3, 3),
    "max_shifts": (5, 5),
    "strides": (48, 48),
    "overlaps": (24, 24),
    "max_deviation_rigid": 3,
    "border_nan": "copy",
    "num_frames_split": 80,
    "niter_rig": 1,
    "splits_rig": 14,
    "splits_els": 14,
    "upsample_factor_grid": 4,
    "shifts_opencv": True,
    "nonneg_movie": True,
    "use_cuda": False,
}

DEFAULT_PARAMS: dict[str, Any] = {
    "fr": 1.0,
    "fnames": "",
    "index": None,
    "ROIs": None,
    "weights": None,
    "Reload": False,
    "output_name": "normc.tiff",
    **MOTION_PARAMS,
}


def find_normcorre_tiff(data_folder: str | Path, output_name: str = "normc.tiff") -> Optional[Path]:
    output_path = Path(data_folder) / output_name
    return output_path if output_path.exists() else None


def _is_readable_tiff(path: Path) -> bool:
    try:
        if path.stat().st_size <= 0:
            return False
        with tifffile.TiffFile(path) as tif:
            return bool(tif.series and len(tif.series[0].shape) >= 2)
    except Exception:
        return False


def _temporary_output_path(output_path: Path) -> Path:
    temp_file = tempfile.NamedTemporaryFile(
        prefix=".nb_nc_",
        suffix=output_path.suffix,
        dir=output_path.parent,
        delete=False,
    )
    temp_file.close()
    return Path(temp_file.name)


def run_normcorre(
    input_path: str | Path,
    output_path: str | Path,
    frame_rate: float,
    params: Optional[dict[str, Any]] = None,
    reload: bool = False,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not reload and _is_readable_tiff(output_path):
        return output_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input movie was not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = dict(DEFAULT_PARAMS)
    if params:
        cfg.update(params)
    cfg["fnames"] = str(input_path)
    cfg["fr"] = float(frame_rate) if frame_rate else float(cfg.get("fr") or 1.0)

    import cormcorre as cm
    from cormcorre.motion_correction import MotionCorrect

    motion_params = {
        key: cfg[key]
        for key in (
            "pw_rigid",
            "max_shifts",
            "gSig_filt",
            "strides",
            "overlaps",
            "max_deviation_rigid",
            "border_nan",
            "num_frames_split",
            "niter_rig",
            "splits_rig",
            "splits_els",
            "upsample_factor_grid",
            "shifts_opencv",
            "nonneg_movie",
            "use_cuda",
        )
        if key in cfg
    }
    previous_temp = os.environ.get("cormcorre_TEMP")
    temp_output_path = _temporary_output_path(output_path)
    with tempfile.TemporaryDirectory(prefix="nb_nc_") as temp_dir:
        os.environ["cormcorre_TEMP"] = temp_dir
        try:
            mc = MotionCorrect(str(input_path), dview=None, **motion_params)
            mc.motion_correct(save_movie=True)
            mmap_files = mc.mmap_file if isinstance(mc.mmap_file, list) else [mc.mmap_file]
            border_to_0 = 0 if getattr(mc, "border_nan", None) == "copy" else getattr(mc, "border_to_0", 0)
            _write_mmap_files_to_tiff(cm, mmap_files, temp_output_path, add_to_movie=border_to_0)
            os.replace(temp_output_path, output_path)
        finally:
            if previous_temp is None:
                os.environ.pop("cormcorre_TEMP", None)
            else:
                os.environ["cormcorre_TEMP"] = previous_temp
            if temp_output_path.exists():
                try:
                    temp_output_path.unlink()
                except OSError:
                    pass
    return output_path


def _write_mmap_files_to_tiff(cm_module, mmap_files: list[str], output_path: Path, add_to_movie: float = 0.0) -> None:
    with tifffile.TiffWriter(output_path, bigtiff=True) as writer:
        for mmap_file in mmap_files:
            yr, dims, frame_count = cm_module.load_memmap(str(mmap_file))
            movie = np.reshape(yr.T, [int(frame_count), *list(dims)], order="F")
            arr = np.asarray(movie, dtype=np.float32)
            if add_to_movie:
                arr = arr + add_to_movie
            writer.write(arr, contiguous=True, metadata={"axes": "TYX"})
            del arr, movie, yr
