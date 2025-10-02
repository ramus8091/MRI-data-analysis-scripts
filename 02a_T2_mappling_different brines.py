#!/usr/bin/env python3
"""
batch_t2_decay.py (with T2 uncertainty error bars and T2 annotations)

Traverse a root directory containing subfolders with multi-echo DICOM series
for the same slice locations at different times. Let the user draw multiple random ROIs
once on the reference slice of the first (lowest-numbered) subfolder. Then apply those ROIs to every
subfolder:
  - Compute and scatter-plot filtered, normalized decay data (annotated with T₂) per ROI
  - Overlay ROIs on each reference slice
  - Export filtered TE & signal data to Excel (.xlsx)
  - Generate histogram of voxel intensities for each ROI
  - Calculate water saturation Sw = (mean signal in first slice / mean signal in reference slice)
    using the provided 100% saturated reference.dcm in the "testing" folder or root
  - Save PNGs of both the first-folder and calibration reference slices with overlaid ROIs
Additionally, generate a summary plot of cross-series T₂ values for each ROI across all
subfolders in a 2-row layout, with all subplots sharing the same Y-axis limits,
include vertical error bars showing the 1σ uncertainty from the exponential fit,
and annotate each marker with its fitted T₂ value.
Supports 4–8 ROIs with dynamic subplot arrangement.
All outputs saved under `output/` directories, plots at 300 dpi.

Usage:
  python batch_t2_decay.py /path/to/root_directory
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
dpi = 300
axis_font = 12
title_font = 14
IMAGE_FILTER_KERNEL = 3

# ----------------------------------------------------------------------------
# ROI selection helper
# ----------------------------------------------------------------------------
class ROISelector:
    def __init__(self, ax, image):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.image = image
        self.mask = None
        self.verts = None
        self.selector = PolygonSelector(ax, self.onselect, useblit=True)
        ax.set_title("Draw ROI with clicks; close window when done", fontsize=title_font)

    def onselect(self, verts):
        self.verts = verts
        ny, nx = self.image.shape
        grid = np.mgrid[:ny, :nx]
        points = np.vstack((grid[1].ravel(), grid[0].ravel())).T
        path = Path(verts)
        mask_flat = path.contains_points(points)
        self.mask = mask_flat.reshape(self.image.shape)
        self.selector.disconnect_events()
        self.canvas.draw_idle()

    def get_mask(self):
        return self.mask

    def get_verts(self):
        return self.verts

# Exponential decay model
def exp_decay(te, s0, t2):
    return s0 * np.exp(-te / t2)

# ----------------------------------------------------------------------------
# Process one folder using predefined ROIs
# ----------------------------------------------------------------------------
def process_folder(folder, masks, verts_list, reference_means, filter_kernel=1):
    logging.info(f"Processing folder: {folder}")
    outdir = os.path.join(folder, 'output')
    os.makedirs(outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(folder, '*.dcm')))
    if not files:
        logging.warning("No DICOMs found; skipping")
        return [], []

    echoes = []
    for fn in files:
        try:
            ds = pydicom.dcmread(fn)
            te = float(ds.EchoTime)
            img = ds.pixel_array.astype(float)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept
            img = medfilt(img, kernel_size=filter_kernel)
            echoes.append((te, img))
        except Exception as e:
            logging.warning(f"Skipping {fn}: {e}")
    if len(echoes) < 2:
        logging.warning("Need >=2 echoes; skipping")
        return [], []

    echoes.sort(key=lambda x: x[0])
    tes, images = zip(*echoes)
    tes = np.array(tes)
    ref_img = images[0]

    # Output paths
    spath = lambda name: os.path.join(outdir, name)
    decay_png = spath('t2_decay_scatter.png')
    overlay_png = spath('rois_overlay.png')
    hist_prefix  = spath('histogram_ROI_')
    excel_xlsx   = spath('decay_data.xlsx')

    raw = {'TE_ms': tes}
    t2_results = []  # will hold (label, T2, T2_err)
    sw_results = []

    # T2 decay scatter + fit
    fig, ax = plt.subplots(figsize=(8,6), dpi=dpi)
    cmap = plt.get_cmap('tab10')
    for idx, mask in enumerate(masks):
        label = f"ROI {idx+1}"
        vals = np.array([im[mask].mean() for im in images])
        filt = medfilt(vals, kernel_size=filter_kernel)
        norm = filt / filt.max() if filt.max() else filt

        raw[f'{label}_signal']  = vals
        raw[f'{label}_filtered'] = filt
        raw[f'{label}_norm']     = norm

        # Fit exponential decay, capture covariance
        try:
            popt, pcov = curve_fit(exp_decay, tes, filt, p0=[filt.max(), 50.0])
            T2 = popt[1]
            T2_err = np.sqrt(np.abs(pcov[1,1]))
        except Exception:
            T2, T2_err = np.nan, np.nan
        t2_results.append((label, T2, T2_err))

        # Compute water saturation
        Sw = vals[0] / reference_means[idx] if reference_means[idx] else np.nan
        sw_results.append(Sw)

        ax.scatter(tes, norm, s=64, label=label, color=cmap(idx))

    ax.set_xlabel('Echo Time (ms)', fontsize=axis_font)
    ax.set_ylabel('Normalized Signal', fontsize=axis_font)
    ax.set_title('T₂ Decay', fontsize=title_font)
    ax.grid(True)
    ax.legend(title='ROI', fontsize=axis_font)
    fig.savefig(decay_png, bbox_inches='tight')
    plt.close(fig)

    # ROI overlay on reference echo
    fig, ax = plt.subplots(figsize=(8,6), dpi=dpi)
    ax.imshow(ref_img, cmap='gray')
    for idx, verts in enumerate(verts_list):
        ax.add_patch(Polygon(verts, edgecolor=cmap(idx), fill=False, linewidth=2))
    ax.axis('off')
    ax.set_title('ROIs Overlay', fontsize=title_font)
    fig.savefig(overlay_png, bbox_inches='tight')
    plt.close(fig)

    # Histograms per ROI
    for idx, mask in enumerate(masks):
        data = ref_img[mask].ravel()
        fig, ax = plt.subplots(figsize=(8,6), dpi=dpi)
        ax.hist(data, bins=50)
        ax.set_xlabel('Voxel Intensity', fontsize=axis_font)
        ax.set_ylabel('Count', fontsize=axis_font)
        ax.set_title(f'ROI {idx+1} Histogram', fontsize=title_font)
        ax.grid(True)
        fig.savefig(f"{hist_prefix}{idx+1}.png", bbox_inches='tight')
        plt.close(fig)

    # Write Excel with saturation and covariance
    df = pd.DataFrame(raw)
    df_meta = {k: getattr(pydicom.dcmread(files[0], stop_before_pixels=True), k, None)
               for k in ['EchoTime','RepetitionTime','ImagingFrequency','NumberOfAverages']}
    meta_df = pd.DataFrame.from_dict(df_meta, orient='index', columns=['Value'])

    sat_df = pd.DataFrame({'ROI': [f'ROI {i+1}' for i in range(len(masks))],
                           'Sw': sw_results})

    with pd.ExcelWriter(excel_xlsx) as writer:
        df.to_excel(writer, sheet_name='DecayData', index=False)
        meta_df.to_excel(writer, sheet_name='Metadata')
        sat_df.to_excel(writer, sheet_name='Saturation', index=False)

    return t2_results, sw_results

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    if not os.path.isdir(root):
        sys.exit("Invalid directory")

    # Locate calibration reference
    candidates = [
        os.path.join(root, 'testing', 'reference.dcm'),
        os.path.join(root, 'reference.dcm')
    ]
    ref_path = next((p for p in candidates if os.path.isfile(p)), None)
    if ref_path is None:
        sys.exit(f"Calibration reference not found. Checked: {candidates}")

    ds_ref = pydicom.dcmread(ref_path)
    ref_calib = ds_ref.pixel_array.astype(float)
    if hasattr(ds_ref, 'RescaleSlope') and hasattr(ds_ref, 'RescaleIntercept'):
        ref_calib = ref_calib * ds_ref.RescaleSlope + ds_ref.RescaleIntercept
    ref_calib = medfilt(ref_calib, kernel_size=IMAGE_FILTER_KERNEL)

    # Time between folders (hours)
    try:
        ti = float(input("Enter time between folders (hours): "))
    except:
        ti = 48.0

    # Numeric sorting of subfolders
    all_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    num_dirs = sorted([d for d in all_dirs if d.isdigit()], key=lambda x: int(x))
    vals = [os.path.join(root, d) for d in num_dirs if glob.glob(os.path.join(root, d, '*.dcm'))]
    if not vals:
        sys.exit("No DICOM subfolders")
    first = vals[0]

    # Draw ROIs on first folder
    ds0 = pydicom.dcmread(glob.glob(os.path.join(first, '*.dcm'))[0])
    ref_img = ds0.pixel_array.astype(float)
    if hasattr(ds0, 'RescaleSlope') and hasattr(ds0, 'RescaleIntercept'):
        ref_img = ref_img * ds0.RescaleSlope + ds0.RescaleIntercept
    ref_img = medfilt(ref_img, kernel_size=IMAGE_FILTER_KERNEL)

    print(f"Draw ROIs on: {first}")
    masks, verts = [], []
    cmap = plt.get_cmap('tab10')
    while True:
        fig, ax = plt.subplots(figsize=(8,6), dpi=dpi)
        ax.imshow(ref_img, cmap='gray')
        selector = ROISelector(ax, ref_img)
        plt.show()
        mask, vert = selector.get_mask(), selector.get_verts()
        plt.close(fig)
        if mask is None or vert is None:
            if input("Keep drawing? (y/n): ").lower() != 'y':
                break
            else:
                continue
        masks.append(mask)
        verts.append(vert)
        if len(masks) >= 8 or input(f"ROI {len(masks)} done. Add another? (y/n): ").lower() != 'y':
            break
    if not masks:
        sys.exit("No ROIs drawn")

    # Save reference overlays
    outdir = os.path.join(root, 'output')
    os.makedirs(outdir, exist_ok=True)

    # First-folder reference
    fig, ax = plt.subplots(figsize=(8,6), dpi=dpi)
    ax.imshow(ref_img, cmap='gray')
    for idx, v in enumerate(verts):
        ax.add_patch(Polygon(v, edgecolor=cmap(idx), fill=False, linewidth=2))
    ax.axis('off')
    ax.set_title('First-folder Reference with ROIs', fontsize=title_font)
    fig.savefig(os.path.join(outdir, 'reference_firstfolder_rois.png'), bbox_inches='tight')
    plt.close(fig)

    # Calibration reference
    fig, ax = plt.subplots(figsize=(8,6), dpi=dpi)
    ax.imshow(ref_calib, cmap='gray')
    for idx, v in enumerate(verts):
        ax.add_patch(Polygon(v, edgecolor=cmap(idx), fill=False, linewidth=2))
    ax.axis('off')
    ax.set_title('Calibration Reference with ROIs', fontsize=title_font)
    fig.savefig(os.path.join(outdir, 'reference_calibration_rois.png'), bbox_inches='tight')
    plt.close(fig)

    # Calculate reference means
    reference_means = [ref_calib[m].mean() for m in masks]

    # Process each folder and collect results
    results_t2 = {f'ROI {i+1}': [] for i in range(len(masks))}
    results_sw = {f'ROI {i+1}': [] for i in range(len(masks))}
    for i, folder in enumerate(vals, start=1):
        t2_list, sw_list = process_folder(folder, masks, verts, reference_means)
        for (roi, t2, t2_err), sw in zip(t2_list, sw_list):
            results_t2[roi].append((i, t2, t2_err))
            results_sw[roi].append((i, sw))

    # Summary plot: T₂ vs time with uniform Y-limits + error bars, Sw box, and T₂ annotations
    roi_labels = list(results_t2.keys())
    all_t2 = [t2 for seq in results_t2.values() for _, t2, _ in seq]
    if all_t2:
        y_min, y_max = (min(all_t2) - 5), (max(all_t2) + 5)
    else:
        y_min, y_max = 0, 100

    cols = math.ceil(len(roi_labels) / 2)
    fig, axes = plt.subplots(2, cols, figsize=(12,10), dpi=dpi)
    axs = axes.flatten()
    for idx, lbl in enumerate(roi_labels):
        seq = results_t2[lbl]  # list of (i, T2, T2_err)
        sw_seq = results_sw[lbl]  # list of (i, Sw)
        times_h = [(i-1) * ti for i, _, _ in seq]
        t2_vals  = [v for _, v, _ in seq]
        t2_errs  = [e for _, _, e in seq]
        initial_sw = sw_seq[0][1] if sw_seq else np.nan

        ax = axs[idx]
        # plot with error bars
        ax.errorbar(
            times_h,
            t2_vals,
            yerr=t2_errs,
            fmt='o-',
            capsize=3,
            linewidth=2,
            markersize=6,
            color=cmap(idx),
            label=lbl
        )

        # Annotate each T2 value above its marker
        for x, y in zip(times_h, t2_vals):
            ax.annotate(
                f"{y:.0f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha='center',
                va='bottom',
                fontsize=10
            )

        # Annotate initial Sw in corner box
        ax.text(
            0.95, 0.05,
            f"Sw={initial_sw:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.7)
        )
        ax.set_title(lbl, fontsize=title_font)
        ax.set_xlabel('Time (h)', fontsize=axis_font)
        ax.set_ylabel('T₂ (ms)', fontsize=axis_font)
        ax.set_ylim(y_min, y_max)
        ax.grid(True)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

    # Hide unused axes
    for ax in axs[len(roi_labels):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 't2_summary_errorbars.png'), bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
