#!/usr/bin/env python3
"""
batch_T1_decay.py (with T1 uncertainty error bars, Sw annotations, and voxel-size metadata)

Traverse a root directory containing subfolders with variable-TR RARE DICOM series
for the same slice locations at different times. Let the user draw multiple ROIs on
the reference slice of the first (lowest-numbered) subfolder. Then apply those ROIs to every
subfolder:
  - Compute and scatter-plot filtered, normalized recovery data (signal vs. TR) per ROI
  - Fit T₁ via S(TR) = S0 * (1 – exp(–TR/T1)), capture covariance for uncertainties
  - Calculate water saturation Sw = (signal at first TR / mean signal of calibration reference)
    using provided reference.dcm in testing or root
  - Overlay ROIs on each reference slice
  - Export raw TR & signal data, T1±error, Sw, and detailed DICOM metadata (including voxel size)
    to Excel (.xlsx)
  - Generate a summary plot of cross-series T₁ values with error bars + Sw annotations
Supports 4–8 ROIs with dynamic subplot arrangement.
All outputs saved under `output/` directories at 200 dpi.

Usage:
  python batch_T1_decay.py /path/to/root_directory
  (if no path given, uses current working directory)
"""

import os
import sys
import glob
import math
import logging

import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from matplotlib.patches import Polygon
from scipy.optimize import curve_fit
from scipy.signal import medfilt

# Settings
DPI = 200
AXIS_FONT = 14
TITLE_FONT = 20
IMAGE_FILTER_KERNEL = 3
MAX_ROIS = 8

# ----------------------------------------------------------------------------
# ROI selector
# ----------------------------------------------------------------------------
class ROISelector:
    def __init__(self, ax, image):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.image = image
        self.mask = None
        self.verts = None
        self.selector = PolygonSelector(ax, self.onselect, useblit=True)
        ax.set_title("Draw ROI with clicks; close window when done", fontsize=AXIS_FONT)

    def onselect(self, verts):
        self.verts = verts
        ny, nx = self.image.shape
        pts = np.vstack((np.mgrid[:ny, :nx][::-1]).reshape(2, -1)).T
        path = Path(verts)
        self.mask = path.contains_points(pts).reshape(self.image.shape)
        self.selector.disconnect_events()
        self.canvas.draw_idle()

    def get_mask(self):
        return self.mask

    def get_verts(self):
        return self.verts

# ----------------------------------------------------------------------------
# Recovery model for T1 fitting
# ----------------------------------------------------------------------------
def recovery_model(tr, s0, t1):
    return s0 * (1 - np.exp(-tr / t1))

# ----------------------------------------------------------------------------
# Process a single folder
# ----------------------------------------------------------------------------
def process_folder(folder, masks, verts_list, reference_means, filter_kernel=IMAGE_FILTER_KERNEL):
    logging.info(f"Processing folder: {folder}")
    outdir = os.path.join(folder, 'output')
    os.makedirs(outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(folder, '*.dcm')))
    if not files:
        logging.warning("Skipping folder: no DICOMs found")
        return [], []

    # Read & filter images
    series = []
    for fn in files:
        try:
            ds = pydicom.dcmread(fn)
            tr = float(ds.RepetitionTime)
            img = ds.pixel_array.astype(float)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept
            img = medfilt(img, kernel_size=filter_kernel)
            series.append((tr, img))
        except Exception as e:
            logging.warning(f"Skipping {fn}: {e}")
    if len(series) < 2:
        logging.warning("Skipping folder: need ≥2 TRs")
        return [], []

    # Sort by TR
    series.sort(key=lambda x: x[0])
    trs, images = zip(*series)
    trs = np.array(trs)
    ref_img = images[0]

    # Paths
    scatter_png = os.path.join(outdir, 't1_recovery_scatter.png')
    overlay_png = os.path.join(outdir, 'rois_overlay.png')
    excel_xlsx = os.path.join(outdir, 'recovery_data.xlsx')

    # Containers
    raw = {'TR_ms': trs}
    t1_results = []  # list of (label, t1, t1_err)
    sw_results = []

    # Scatter & fit
    fig, ax = plt.subplots(figsize=(8,6), dpi=DPI)
    cmap = plt.get_cmap('tab10')
    for idx, mask in enumerate(masks):
        label = f"ROI {idx+1}"
        sig = np.array([im[mask].mean() for im in images])
        norm = sig / sig.max() if sig.max() else sig
        raw[f'{label}_signal'] = sig
        raw[f'{label}_norm'] = norm

        try:
            popt, pcov = curve_fit(recovery_model, trs, sig, p0=[sig.max(), trs.mean()])
            t1 = popt[1]
            t1_err = np.sqrt(np.abs(pcov[1,1]))
        except Exception:
            t1, t1_err = np.nan, np.nan
        t1_results.append((label, t1, t1_err))

        sw = sig[0] / reference_means[idx] if reference_means[idx] else np.nan
        sw_results.append(sw)

        ax.scatter(trs, norm, s=64, label=label, color=cmap(idx))

    ax.set_xlabel('Repetition Time TR (ms)', fontsize=AXIS_FONT)
    ax.set_ylabel('Normalized Signal', fontsize=AXIS_FONT)
    ax.set_title('T₁ Recovery', fontsize=TITLE_FONT)
    ax.grid(True)
    ax.legend(title='ROI', fontsize=AXIS_FONT)
    fig.savefig(scatter_png, bbox_inches='tight')
    plt.close(fig)

    # Overlay ROIs on reference slice
    fig, ax = plt.subplots(figsize=(8,6), dpi=DPI)
    ax.imshow(ref_img, cmap='gray')
    for idx, verts in enumerate(verts_list):
        ax.add_patch(Polygon(verts, edgecolor=cmap(idx), fill=False, linewidth=2))
    ax.axis('off')
    fig.savefig(overlay_png, bbox_inches='tight')
    plt.close(fig)

    # Export to Excel: include detailed metadata and Sw
    df = pd.DataFrame(raw)
    ds0 = pydicom.dcmread(files[0], stop_before_pixels=True)
    meta_keys = [
        'RepetitionTime', 'EchoTime', 'ImagingFrequency', 'NumberOfAverages',
        'PixelSpacing', 'SliceThickness', 'SpacingBetweenSlices'
    ]
    metadata = {}
    for key in meta_keys:
        metadata[key] = getattr(ds0, key, None)
    meta_df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Value'])

    sat_df = pd.DataFrame({'ROI': [f'ROI {i+1}' for i in range(len(masks))], 'Sw': sw_results})

    with pd.ExcelWriter(excel_xlsx) as writer:
        df.to_excel(writer, sheet_name='RecoveryData', index=False)
        meta_df.to_excel(writer, sheet_name='Metadata')
        sat_df.to_excel(writer, sheet_name='Saturation', index=False)

    return t1_results, sw_results

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    root = sys.argv[1] if len(sys.argv)>1 else os.getcwd()
    if not os.path.isdir(root):
        sys.exit("Invalid root directory")

    # Calibration reference lookup
    candidates = [os.path.join(root, 'testing', 'reference.dcm'), os.path.join(root, 'reference.dcm')]
    ref_path = next((p for p in candidates if os.path.isfile(p)), None)
    if ref_path is None:
        sys.exit(f"Reference not found. Checked: {candidates}")

    ds_ref = pydicom.dcmread(ref_path)
    ref_calib = ds_ref.pixel_array.astype(float)
    if hasattr(ds_ref, 'RescaleSlope') and hasattr(ds_ref, 'RescaleIntercept'):
        ref_calib = ref_calib * ds_ref.RescaleSlope + ds_ref.RescaleIntercept
    ref_calib = medfilt(ref_calib, kernel_size=IMAGE_FILTER_KERNEL)

    # Time interval
    try:
        interval = float(input("Enter time between folders (hours): "))
    except:
        interval = 48.0

    # Numeric sorting of subfolders
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    num_dirs = sorted([d for d in dirs if d.isdigit()], key=lambda x: int(x))
    vals = [os.path.join(root, d) for d in num_dirs if glob.glob(os.path.join(root, d, '*.dcm'))]
    if not vals:
        sys.exit("No DICOM subfolders found")

    # ROI drawing on first folder reference
    first = vals[0]
    ds0 = pydicom.dcmread(glob.glob(os.path.join(first, '*.dcm'))[0])
    ref_img = ds0.pixel_array.astype(float)
    if hasattr(ds0, 'RescaleSlope') and hasattr(ds0, 'RescaleIntercept'):
        ref_img = ref_img * ds0.RescaleSlope + ds0.RescaleIntercept
    ref_img = medfilt(ref_img, kernel_size=IMAGE_FILTER_KERNEL)

    print(f"Draw up to {MAX_ROIS} ROIs on: {first}")
    masks, verts = [], []
    cmap = plt.get_cmap('tab10')
    while len(masks) < MAX_ROIS:
        fig, ax = plt.subplots(figsize=(8,6), dpi=DPI)
        ax.imshow(ref_img, cmap='gray')
        sel = ROISelector(ax, ref_img)
        plt.show()
        m, v = sel.get_mask(), sel.get_verts()
        plt.close(fig)
        if m is None or v is None:
            break
        masks.append(m)
        verts.append(v)
        if input(f"ROI {len(masks)} captured. Add another? (y/n): ").lower() != 'y':
            break
    if not masks:
        sys.exit("No ROIs defined; aborting.")

    # Save ROI overlays on both first and calibration references
    outdir = os.path.join(root, 'output')
    os.makedirs(outdir, exist_ok=True)
    for name, img in [('first', ref_img), ('calibration', ref_calib)]:
        fig, ax = plt.subplots(figsize=(8,6), dpi=DPI)
        ax.imshow(img, cmap='gray')
        for idx, v in enumerate(verts):
            ax.add_patch(Polygon(v, edgecolor=cmap(idx), fill=False, linewidth=2))
        ax.axis('off')
        fig.savefig(os.path.join(outdir, f'reference_{name}_rois.png'), bbox_inches='tight')
        plt.close(fig)

    reference_means = [ref_calib[m].mean() for m in masks]

    # Process all subfolders and gather results
    results_t1 = {f'ROI {i+1}': [] for i in range(len(masks))}
    results_sw = {f'ROI {i+1}': [] for i in range(len(masks))}
    for i, folder in enumerate(vals, start=1):
        t1_list, sw_list = process_folder(folder, masks, verts, reference_means)
        for (lbl, t1, terr), sw in zip(t1_list, sw_list):
            results_t1[lbl].append((i, t1, terr))
            results_sw[lbl].append((i, sw))

    # Summary plot: T1 vs time with error bars and Sw annotations
    labels = list(results_t1.keys())
    all_t1 = [t1 for seq in results_t1.values() for _, t1, _ in seq]
    y_min, y_max = min(all_t1) - 50, max(all_t1) + 50
    cols = math.ceil(len(labels) / 2)
    fig, axes = plt.subplots(2, cols, figsize=(12,10), dpi=DPI)
    axs = axes.flatten()
    cmap = plt.get_cmap('tab10')

    for idx, lbl in enumerate(labels):
        seq = results_t1[lbl]
        sw_seq = results_sw[lbl]
        times = [(i - 1) * interval for i, _, _ in seq]
        t1_vals = [v for _, v, _ in seq]
        t1_errs = [e for _, _, e in seq]
        init_sw = sw_seq[0][1] if sw_seq else np.nan

        ax = axs[idx]
        ax.errorbar(
            times,
            t1_vals,
            yerr=t1_errs,
            fmt='o-',
            capsize=3,
            linewidth=2,
            markersize=6,
            color=cmap(idx)
        )
        for t, v in zip(times, t1_vals):
            ax.annotate(f"{v:.0f}", (t, v), textcoords='offset points', xytext=(0,5), ha='center')
        ax.text(
            0.95, 0.05,
            f"Sw={init_sw:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.7)
        )
        ax.set_title(lbl, fontsize=TITLE_FONT)
        ax.set_xlabel('Time (h)', fontsize=AXIS_FONT)
        ax.set_ylabel('T₁ (ms)', fontsize=AXIS_FONT)
        ax.set_ylim(y_min, y_max)
        ax.grid(True)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

    for ax in axs[len(labels):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 't1_summary_errorbars_sw.png'), bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
