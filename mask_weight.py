# mask_weight.mask(datafolder)

from pathlib import Path
import colorsys
import numpy as np
import tifffile
from trace_process import pix_exp as pix_weight
import os


def normalize_image(img, pmin=1.0, pmax=99.5):
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [pmin, pmax])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    img = (img - lo) / (hi - lo)
    return np.clip(img, 0.0, 1.0)

def load_mean_image_from_tif(tif_path):
    tif_path = Path(tif_path)

    try:
        arr = tifffile.memmap(tif_path)
        arr = np.asarray(arr)

        if arr.ndim == 2:
            return arr.astype(np.float32)

        elif arr.ndim == 3:
            return arr.astype(np.float32).mean(axis=0)

        elif arr.ndim == 4:
            if arr.shape[-1] in (3, 4):
                mean_img = arr.astype(np.float32).mean(axis=0)
                return mean_img[..., :3].mean(axis=-1)
            elif arr.shape[1] in (3, 4):
                mean_img = arr.astype(np.float32).mean(axis=0)
                return np.moveaxis(mean_img[:3], 0, -1).mean(axis=-1)
            else:
                raise ValueError(f"Unsupported TIFF shape: {arr.shape}")

        else:
            raise ValueError(f"Unsupported TIFF shape: {arr.shape}")

    except ValueError:
        with tifffile.TiffFile(tif_path) as tif:
            pages = tif.pages

            if len(pages) == 1:
                img = pages[0].asarray()

                if img.ndim == 2:
                    return img.astype(np.float32)

                elif img.ndim == 3:
                    if img.shape[-1] in (3, 4):
                        return img[..., :3].astype(np.float32).mean(axis=-1)
                    elif img.shape[0] in (3, 4):
                        return img[:3].astype(np.float32).mean(axis=0)
                    else:
                        return img.astype(np.float32).mean(axis=0)

                else:
                    raise ValueError(f"Unsupported TIFF shape: {img.shape}")

            first = pages[0].asarray()

            if first.ndim != 2:
                raise ValueError(
                    f"Expected page-wise 2D TIFF, but first page shape is {first.shape}"
                )

            acc = np.zeros(first.shape, dtype=np.float64)
            n = 0

            for page in pages:
                frame = page.asarray().astype(np.float32)
                if frame.shape != first.shape:
                    raise ValueError(
                        f"Inconsistent page shape: {frame.shape} vs {first.shape}"
                    )
                acc += frame
                n += 1

            if n == 0:
                raise ValueError("No image pages found in TIFF")

            return (acc / n).astype(np.float32)


def build_overlay(
    suite2p_dir,
    tif_path=None,
    output_path=None,
    plane="plane0",
    background="tif",
    alpha_mode="relative",
    max_alpha=0.85,
    use_iscell=False,
    save_alpha_map=True,
):
    suite2p_dir = Path(suite2p_dir)
    plane_dir = suite2p_dir / plane
    outpath = rf"{output_path}\Results"
    os.makedirs(outpath,exist_ok=True)  

    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    stat = np.load(plane_dir / "stat.npy", allow_pickle=True)

    if use_iscell and (plane_dir / "iscell.npy").exists():
        iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
        keep = iscell[:, 0].astype(bool)
        stat = stat[keep]

    Ly = int(ops["Ly"])
    Lx = int(ops["Lx"])

    if background == "tif":
        if tif_path is None:
            raise ValueError("tif_path is required when background='tif'")
        mean_img = load_mean_image_from_tif(tif_path)
    elif background == "ops":
        if "meanImg" not in ops:
            raise ValueError("ops.npy does not contain 'meanImg'")
        mean_img = np.asarray(ops["meanImg"], dtype=np.float32)
    else:
        raise ValueError("background must be 'tif' or 'ops'")

    if mean_img.shape != (Ly, Lx):
        raise ValueError(
            f"Image shape mismatch: background image is {mean_img.shape}, "
            f"but suite2p expects {(Ly, Lx)}"
        )

    base = normalize_image(mean_img)
    base_rgb = np.stack([base, base, base], axis=-1)

    alpha_acc = np.zeros((Ly, Lx), dtype=np.float32)
    color_acc = np.zeros((Ly, Lx, 3), dtype=np.float32)
    
    s_alpha_acc = np.zeros((Ly, Lx), dtype=np.float32)
    s_color_acc = np.zeros((Ly, Lx, 3), dtype=np.float32)

    n_roi = len(stat)

    for i, roi in enumerate(stat):
        ypix = np.asarray(roi["ypix"], dtype=np.int64)
        xpix = np.asarray(roi["xpix"], dtype=np.int64)
        s_weight = np.asarray(roi["lam"])

        valid = (
            (ypix >= 0) & (ypix < Ly) &
            (xpix >= 0) & (xpix < Lx)
        )
        ypix = ypix[valid]
        xpix = xpix[valid]
        s_weight = s_weight[valid]

        if ypix.size == 0:
            continue

        roi_values = mean_img[ypix, xpix]
        weights = pix_weight(roi_values)

        if alpha_mode == "exact":
            alpha = weights
            s_alpha = s_weight
        elif alpha_mode == "relative":
            alpha = weights / (weights.max() + 1e-8)
            s_alpha = s_weight / (s_weight.max() + 1e-8)
        elif alpha_mode == "sqrt":
            alpha = np.sqrt(weights / (weights.max() + 1e-8))
            s_alpha = np.sqrt(s_weight / (s_weight.max() + 1e-8))
        else:
            raise ValueError("alpha_mode must be 'exact', 'relative', or 'sqrt'")

        alpha = np.clip(alpha * max_alpha, 0.0, 1.0)
        s_alpha = np.clip(s_alpha * max_alpha, 0.0, 1.0)

        hue = (i / max(n_roi, 1)) % 1.0
        color = np.array(colorsys.hsv_to_rgb(hue, 1.0, 1.0), dtype=np.float32)

        color_acc[ypix, xpix] += alpha[:, None] * color[None, :]
        alpha_acc[ypix, xpix] += alpha
        s_color_acc[ypix, xpix] += s_alpha[:, None] * color[None, :]
        s_alpha_acc[ypix, xpix] += s_alpha
        
    alpha_acc = [alpha_acc,s_alpha_acc]
    color_acc = [color_acc,s_color_acc]
    mask_name = ['weight_mask','suite2p_mask']
    for m in range(len(alpha_acc)):
        a_acc,mask_n,c_acc = alpha_acc[m],mask_name[m],color_acc[m]

        a_acc = np.clip(a_acc, 0.0, 1.0)

        color_map = np.zeros_like(c_acc)
        mask = a_acc > 1e-8
        color_map[mask] = c_acc[mask] / a_acc[mask, None]


        overlay_rgb = base_rgb * (1.0 - a_acc[..., None]) + color_map * a_acc[..., None]
        overlay_rgb = np.clip(overlay_rgb, 0.0, 1.0)

        overlay_uint8 = (overlay_rgb * 255).astype(np.uint8)
        alpha_uint8 = (a_acc * 255).astype(np.uint8)

        output_path = Path(outpath).with_name(f"{mask_n}_overlay.tif")

        tifffile.imwrite(output_path, overlay_uint8, photometric="rgb")

        alpha_path = None
        if save_alpha_map:
            alpha_path = output_path.with_name(output_path.stem + "_alpha.tif")
            tifffile.imwrite(alpha_path, alpha_uint8)

    return output_path, alpha_path


def mask(datafolder):
    path = Path(datafolder)
    basename = path.name
    tif_path = rf"{datafolder}\{basename}_Cycle00001_Ch2_000001.ome.tif"
    suite2p_dir = rf"{datafolder}\suite2p"
    outpath = rf"{datafolder}\Results"
    os.makedirs(outpath,exist_ok=True)

    output_path, alpha_path = build_overlay(
        suite2p_dir=suite2p_dir,
        tif_path=tif_path,
        output_path=outpath,
        plane="plane0",
        background="tif",
        alpha_mode="relative",
        max_alpha=1.0,
        use_iscell=False,
        save_alpha_map=True,
    )

    print("Overlay saved to:", output_path)
    if alpha_path is not None:
        print("Alpha map saved to:", alpha_path)