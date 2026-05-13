from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tifffile
import torch

from pmd.decomposition import pmd_decomposition


DEFAULT_PARAMS: dict[str, Any] = {
    "Reload": False,
    "output_name": "pmd.tiff",
    "input_axis_order": "TYX",
    "save_dtype": "same_as_input",
    "save_residual": False,
    "reconstruction_batch_size": 500,
    "pixel_weight_source": "none",
    "pixel_weight_boost": 2.0,
    "block_sizes": (32, 32),
    "frame_range": 2000,
    "max_components": 20,
    "sim_conf": 5,
    "frame_batch_size": 10000,
    "max_consecutive_failures": 1,
    "spatial_avg_factor": 1,
    "temporal_avg_factor": 1,
    "compute_normalizer": True,
    "device": "auto",
    "use_temporal_cnn_denoiser": False,
    "denoiser_num_epochs": 10,
    "noise_variance_quantile": 0.3,
}


class _StatusWriter:
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += str(text).replace("\r", "\n")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.callback(line)
        return len(text)

    def flush(self) -> None:
        line = self._buffer.strip()
        if line:
            self.callback(line)
        self._buffer = ""


def _emit(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def load_tiff_as_tyx(path: Path, axis_order: str) -> tuple[np.ndarray, np.dtype]:
    movie = np.asarray(tifffile.imread(path))
    original_dtype = movie.dtype
    movie = np.squeeze(movie)
    if movie.ndim != 3:
        raise ValueError(
            f"Expected a 3D TIFF after squeeze, but got shape {movie.shape}."
        )
    if axis_order == "TYX":
        pass
    elif axis_order == "YXT":
        movie = np.transpose(movie, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported input_axis_order: {axis_order}")
    return movie.astype(np.float32, copy=False), original_dtype


def convert_output_dtype(
    arr: np.ndarray,
    save_dtype: str,
    original_dtype: np.dtype,
) -> np.ndarray:
    if save_dtype == "float32":
        return arr.astype(np.float32, copy=False)
    if save_dtype == "same_as_input":
        target_dtype = original_dtype
    elif save_dtype == "uint16":
        target_dtype = np.dtype("uint16")
    elif save_dtype == "int16":
        target_dtype = np.dtype("int16")
    else:
        raise ValueError(f"Unsupported save_dtype: {save_dtype}")

    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        arr = np.clip(arr, info.min, info.max)
        arr = np.rint(arr)
    return arr.astype(target_dtype, copy=False)


def output_dtype(save_dtype: str, original_dtype: np.dtype) -> np.dtype:
    if save_dtype == "float32":
        return np.dtype("float32")
    if save_dtype == "same_as_input":
        return np.dtype(original_dtype)
    if save_dtype == "uint16":
        return np.dtype("uint16")
    if save_dtype == "int16":
        return np.dtype("int16")
    raise ValueError(f"Unsupported save_dtype: {save_dtype}")


def save_pmd_factors(pmd_arr, save_path: Path) -> None:
    u = pmd_arr.u.coalesce()
    data = {
        "shape": tuple(pmd_arr.shape),
        "u_indices": u.indices().cpu(),
        "u_values": u.values().cpu(),
        "u_size": tuple(u.size()),
        "v": pmd_arr.v.cpu(),
        "mean_img": pmd_arr.mean_img.cpu(),
        "var_img": pmd_arr.var_img.cpu(),
        "rescale": pmd_arr.rescale,
    }
    if pmd_arr.u_local_projector is not None:
        up = pmd_arr.u_local_projector.coalesce()
        data["u_local_projector_indices"] = up.indices().cpu()
        data["u_local_projector_values"] = up.values().cpu()
        data["u_local_projector_size"] = tuple(up.size())
    torch.save(data, save_path)


def save_reconstructed_tiff(
    pmd_arr,
    output_tiff: Path,
    original_dtype: np.dtype,
    save_dtype: str,
    batch_size: int,
) -> None:
    output_tiff.parent.mkdir(parents=True, exist_ok=True)
    if output_tiff.exists():
        output_tiff.unlink()
    dtype = output_dtype(save_dtype, original_dtype)
    output = tifffile.memmap(output_tiff, shape=tuple(pmd_arr.shape), dtype=dtype, bigtiff=True)
    for start in range(0, pmd_arr.shape[0], batch_size):
        end = min(start + batch_size, pmd_arr.shape[0])
        block = pmd_arr[start:end]
        if block.ndim == 2:
            block = block[None, :, :]
        output[start:end] = convert_output_dtype(block, save_dtype, original_dtype)
    output.flush()


def save_residual_tiff(
    raw_movie: np.ndarray,
    pmd_arr,
    output_tiff: Path,
    original_dtype: np.dtype,
    save_dtype: str,
    batch_size: int,
) -> None:
    output_tiff.parent.mkdir(parents=True, exist_ok=True)
    if output_tiff.exists():
        output_tiff.unlink()
    dtype = output_dtype(save_dtype, original_dtype)
    output = tifffile.memmap(output_tiff, shape=tuple(pmd_arr.shape), dtype=dtype, bigtiff=True)
    for start in range(0, pmd_arr.shape[0], batch_size):
        end = min(start + batch_size, pmd_arr.shape[0])
        recon = pmd_arr[start:end].astype(np.float32, copy=False)
        if recon.ndim == 2:
            recon = recon[None, :, :]
        raw = raw_movie[start:end].astype(np.float32, copy=False)
        output[start:end] = convert_output_dtype(raw - recon, save_dtype, original_dtype)
    output.flush()


def _run_pmd_array(movie: np.ndarray, cfg: dict[str, Any]):
    pixel_weighting = cfg.get("pixel_weighting")
    if pixel_weighting is not None:
        pixel_weighting = np.asarray(pixel_weighting, dtype=np.float32)
        expected_shape = tuple(int(v) for v in movie.shape[1:3])
        if pixel_weighting.shape != expected_shape:
            raise ValueError(
                f"PMD pixel_weighting shape {pixel_weighting.shape} does not match input image shape {expected_shape}."
            )

    if bool(cfg.get("use_temporal_cnn_denoiser")):
        from pmd.compression_strategies import CompressDenoiseStrategy

        strategy = CompressDenoiseStrategy(
            block_sizes=tuple(cfg["block_sizes"]),
            frame_range=cfg.get("frame_range"),
            max_components=int(cfg["max_components"]),
            sim_conf=int(cfg["sim_conf"]),
            frame_batch_size=int(cfg["frame_batch_size"]),
            max_consecutive_failures=int(cfg["max_consecutive_failures"]),
            spatial_avg_factor=int(cfg["spatial_avg_factor"]),
            temporal_avg_factor=int(cfg["temporal_avg_factor"]),
            compute_normalizer=bool(cfg["compute_normalizer"]),
            pixel_weighting=pixel_weighting,
            device=cfg["device"],
            noise_variance_quantile=float(cfg["noise_variance_quantile"]),
            num_epochs=int(cfg["denoiser_num_epochs"]),
        )
        return strategy.compress(movie)

    return pmd_decomposition(
        movie,
        block_sizes=tuple(cfg["block_sizes"]),
        frame_range=cfg.get("frame_range"),
        max_components=int(cfg["max_components"]),
        sim_conf=int(cfg["sim_conf"]),
        frame_batch_size=int(cfg["frame_batch_size"]),
        max_consecutive_failures=int(cfg["max_consecutive_failures"]),
        spatial_avg_factor=int(cfg["spatial_avg_factor"]),
        temporal_avg_factor=int(cfg["temporal_avg_factor"]),
        compute_normalizer=bool(cfg["compute_normalizer"]),
        pixel_weighting=pixel_weighting,
        spatial_denoiser=None,
        temporal_denoiser=None,
        device=cfg["device"],
    )


def run_pmd(
    input_path: str | Path,
    output_path: str | Path | None = None,
    frame_rate: float | None = None,
    params: dict[str, Any] | None = None,
    reload: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    _ = frame_rate
    input_path = Path(input_path)
    cfg = dict(DEFAULT_PARAMS)
    if params:
        cfg.update(params)
    if output_path is None:
        output_name = str(cfg.get("output_name") or f"{input_path.stem}_pmd.tiff").strip()
        output_path = Path(output_name)
        if not output_path.is_absolute():
            output_path = input_path.parent / output_path
    else:
        output_path = Path(output_path)
    reload = bool(reload or cfg.get("Reload"))
    if output_path.exists() and not reload:
        _emit(progress_callback, f"Using existing PMD TIFF: {output_path.name}")
        return output_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input movie was not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _emit(progress_callback, f"Loading PMD input: {input_path.name}")
    movie, original_dtype = load_tiff_as_tyx(input_path, str(cfg["input_axis_order"]))
    _emit(progress_callback, f"Running PMD decomposition on {movie.shape[0]} frames")
    if progress_callback is None:
        pmd_arr = _run_pmd_array(movie, cfg)
    else:
        with contextlib.redirect_stdout(_StatusWriter(progress_callback)), contextlib.redirect_stderr(io.StringIO()):
            pmd_arr = _run_pmd_array(movie, cfg)

    stem = output_path.stem
    _emit(progress_callback, "Saving PMD factors")
    save_pmd_factors(pmd_arr, output_path.with_name(f"{stem}_factors.pt"))
    tifffile.imwrite(
        output_path.with_name(f"{stem}_mean_img.tif"),
        pmd_arr.mean_img.cpu().numpy().astype(np.float32),
    )
    _emit(progress_callback, f"Saving PMD TIFF: {output_path.name}")
    save_reconstructed_tiff(
        pmd_arr,
        output_path,
        original_dtype,
        str(cfg["save_dtype"]),
        int(cfg["reconstruction_batch_size"]),
    )
    if bool(cfg.get("save_residual")):
        _emit(progress_callback, "Saving PMD residual TIFF")
        save_residual_tiff(
            movie,
            pmd_arr,
            output_path.with_name(f"{stem}_residual.tif"),
            original_dtype,
            "float32",
            int(cfg["reconstruction_batch_size"]),
        )
    _emit(progress_callback, f"PMD finished: {output_path.name}")
    return output_path
