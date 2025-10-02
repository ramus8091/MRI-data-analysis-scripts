#!/usr/bin/env python3
"""
Batch‐process subfolders of DICOMs:
  1. Load a reference DICOM, draw a polygonal mask manually
  2. Apply that mask to all other DICOMs in each subfolder
  3. Compute and save PNGs (raw, filtered, cropped, saturation maps, histograms)
  4. Classify each slice as Sagittal/Coronal/Axial
  5. Export quantitative results + SliceType to an Excel workbook per folder
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector
from natsort import natsorted
from scipy.ndimage import uniform_filter, median_filter

# ------------------------------------------------------------
# User‐configurable parameters
# ------------------------------------------------------------
BOTTOM_OFFSET_MM   = 4      # mm to trim from bottom of mask
PIXEL_SPACING_MM   = 1      # mm per pixel in DICOM
BACKGROUND_NOISE   = 0   # noise level for saturation calculation
PNG_DPI            = 300    # resolution for all saved PNGs
IMG_SCALE          = 0.05   # inches per pixel for figure sizing
MASK_FILTER_SIZE   = 3      # kernel size for median filtering
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch‐process DICOM subfolders (manual mask → crop → hist → Excel with slice type → satmap)."
    )
    p.add_argument(
        "-d", "--top-dir",
        type=Path,
        default=Path("/home/..."),
        help="Root directory containing subfolders with DICOM series."
    )
    return p.parse_args() # Update this root directory to the folder your working in

def list_subfolders(root: Path) -> List[Path]:
    return sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')])

def list_dicoms(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.dcm"), key=lambda p: p.name.lower())

def save_png(img: np.ndarray, out_path: Path, cmap="gray", dpi=PNG_DPI) -> None:
    h, w = img.shape
    fw, fh = max(4, w*IMG_SCALE), max(4, h*IMG_SCALE)
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=dpi)
    ax.imshow(img, cmap=cmap)
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def plot_and_save_histogram(vals: np.ndarray, title: str, out_path: Path, dpi=PNG_DPI) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    ax.hist(vals, bins=50, edgecolor="black")
    ax.set(title=title, xlabel="Voxel Intensity", ylabel="Frequency")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def save_cropped_dicom(img: np.ndarray, out_path: Path, ref_ds: pydicom.Dataset) -> None:
    ds2 = ref_ds.copy()
    ds2.PixelData = img.astype(ref_ds.pixel_array.dtype).tobytes()
    ds2.Rows, ds2.Columns = img.shape
    ds2.save_as(str(out_path))

def interactive_polygon_crop(img: np.ndarray) -> np.ndarray:
    mask = np.zeros(img.shape, dtype=bool)
    def onselect(verts: List[Tuple[float,float]]) -> None:
        path = MplPath(verts)
        ny, nx = img.shape
        yy, xx = np.mgrid[:ny, :nx]
        pts = np.vstack((xx.ravel(), yy.ravel())).T
        mask.flat[:] = path.contains_points(pts)
        plt.close()
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(img, cmap="gray")
    ax.set_title("Draw polygon on reference and press ENTER")
    selector = PolygonSelector(ax, onselect)
    fig.canvas.mpl_connect("key_press_event",
        lambda ev: (selector.disconnect_events(), plt.close()) if ev.key=="enter" else None
    )
    plt.show()
    return mask

def classify_slice_orientation(ds: pydicom.Dataset) -> str:
    """
    Use ImageOrientationPatient to compute slice normal and classify:
      axis index: 0->Sagittal, 1->Coronal, 2->Axial
    """
    iop: Optional[List[float]] = getattr(ds, "ImageOrientationPatient", None)
    if not iop or len(iop) != 6:
        return "Unknown"
    row = np.array(iop[0:3], dtype=float)
    col = np.array(iop[3:6], dtype=float)
    normal = np.cross(row, col)
    axis = np.argmax(np.abs(normal))
    return ["Sagittal","Coronal","Axial"][axis]

def compute_saturation_map(img: np.ndarray, mask: np.ndarray, median_ref: float) -> np.ndarray:
    sat = np.zeros_like(img, dtype=float)
    denom = max(median_ref - BACKGROUND_NOISE, 1e-6)
    local = (img - BACKGROUND_NOISE) / denom
    local = np.clip(local, 0.0, 1.0)
    sat[mask] = local[mask]
    return sat

def process_folder(folder: Path) -> None:
    logger.info(f"Processing folder: {folder}")
    dcm_files = list_dicoms(folder)
    if not dcm_files:
        logger.warning("  No DICOMs found; skipping.")
        return

    ref_map = {p.name.lower(): p for p in dcm_files}
    if "reference.dcm" not in ref_map:
        logger.warning("  reference.dcm missing; skipping.")
        return
    ref_path = ref_map["reference.dcm"]
    nonrefs = [p for p in dcm_files if p != ref_path]
    if not nonrefs:
        logger.warning("  No non-reference files; skipping.")
        return

    out_read   = folder / "read_pngs"
    out_crops  = folder / "cropped_pngs"
    out_dcms   = folder / "cropped_dicoms"
    out_satmap = folder / "satmap"
    for d in (out_read, out_crops, out_dcms, out_satmap):
        d.mkdir(exist_ok=True)

    # Reference
    ref_ds   = pydicom.dcmread(ref_path)
    raw_ref  = ref_ds.pixel_array.astype(float)
    img_ref  = median_filter(raw_ref, size=MASK_FILTER_SIZE)
    save_png(img_ref, out_read / f"{ref_path.stem}_median.png")
    save_png(uniform_filter(img_ref, size=MASK_FILTER_SIZE),
             out_read / f"{ref_path.stem}_uniform.png")

    mask = interactive_polygon_crop(img_ref)
    ys, xs = np.where(mask)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    offset_px = int(round(BOTTOM_OFFSET_MM/PIXEL_SPACING_MM))
    bottom_trim = max(0, ymax - offset_px)

    cropped_ref = img_ref[ymin:ymax+1, xmin:xmax+1]
    if ymax>bottom_trim:
        cropped_ref = cropped_ref[:-(ymax-bottom_trim),:]
    save_png(cropped_ref, out_crops / f"{ref_path.stem}_cropped.png")
    save_cropped_dicom(cropped_ref, out_dcms / ref_path.name, ref_ds)

    vals_ref   = img_ref[mask]
    median_ref = float(np.median(vals_ref))
    plot_and_save_histogram(vals_ref, "Reference Voxel Intensity",
                            folder / "histogram_reference.png")
    logger.info(f"  Reference median: {median_ref:.1f}")

    ds0 = pydicom.dcmread(nonrefs[0])
    metadata = {
        "EchoTime_ms":       getattr(ds0, "EchoTime", None),
        "RepetitionTime_ms": getattr(ds0, "RepetitionTime", None),
        "VoxelSpacing_mm":   getattr(ds0, "PixelSpacing", None),
        "SequenceName":      getattr(ds0, "SequenceName", None),
        "ProtocolName":      getattr(ds0, "ProtocolName", None),
        "NumberOfAverages":  getattr(ds0, "NumberOfAverages", None),
        "TotalScanTime_s":   getattr(ds0, "TotalScanTime", None),
    }
    if metadata["TotalScanTime_s"] and nonrefs:
        metadata["TimePerScan_s"] = metadata["TotalScanTime_s"]/len(nonrefs)

    results = []
    for idx, dcm_path in enumerate(nonrefs):
        ds   = pydicom.dcmread(dcm_path)
        img  = median_filter(ds.pixel_array.astype(float), size=MASK_FILTER_SIZE)
        save_png(img, out_read / f"{dcm_path.stem}_median.png")
        save_png(uniform_filter(img, size=MASK_FILTER_SIZE),
                 out_read / f"{dcm_path.stem}_uniform.png")

        vox    = img[mask]
        med_nr = float(np.median(vox))
        sat    = float(np.clip((med_nr - BACKGROUND_NOISE)/max(median_ref-BACKGROUND_NOISE,1e-6), 0, 1))
        plot_and_save_histogram(vox, f"Slice {idx} Voxel Intensity",
                                folder / f"histogram_slice{idx:03d}.png")

        cropped = img[ymin:ymax+1, xmin:xmax+1]
        if ymax>bottom_trim:
            cropped = cropped[:-(ymax-bottom_trim),:]
        save_png(cropped, out_crops / f"{dcm_path.stem}_cropped.png")
        save_cropped_dicom(cropped, out_dcms / dcm_path.name, ds)

        sat_map = compute_saturation_map(img, mask, median_ref)
        h, w = sat_map.shape
        fw, fh = max(4, w*IMG_SCALE), max(4, h*IMG_SCALE)
        fig, ax = plt.subplots(figsize=(fw,fh), dpi=PNG_DPI)
        cax = ax.imshow(sat_map, cmap="viridis", vmin=0, vmax=1)
        ax.axis("off")
        plt.tight_layout(pad=0)
        fig.colorbar(cax, fraction=0.04, pad=0.02, shrink=0.6, aspect=20,
                     label="Saturation (0–1)")
        fig.savefig(out_satmap / f"satmap_{idx:03d}.png", dpi=PNG_DPI,
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        slice_type = classify_slice_orientation(ds)
        results.append([dcm_path.name, med_nr, median_ref, sat, slice_type])
        logger.info(f"  Slice {idx:03d}: median={med_nr:.1f}, sat={sat:.2f}, type={slice_type}")

    df = pd.DataFrame(results, columns=[
        "Slice", "NonRef_Median", "Ref_Median", "Water_Saturation", "SliceType"
    ])
    excel_path = folder / "saturation_results.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name="SaturationResults", index=False)
        meta_df = pd.DataFrame.from_dict(metadata, orient="index", columns=["Value"])
        meta_df.index.name = "Parameter"
        meta_df.to_excel(writer, sheet_name="Metadata")
    logger.info(f"Wrote results to {excel_path}")

def main():
    args = parse_args()
    top_dir = args.top_dir
    if not top_dir.is_dir():
        logger.error(f"Directory not found: {top_dir}")
        return
    for sub in list_subfolders(top_dir):
        process_folder(sub)

if __name__ == "__main__":
    main()
