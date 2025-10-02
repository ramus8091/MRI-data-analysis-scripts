#!/usr/bin/env python3
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from natsort import natsorted
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector
from scipy.ndimage import median_filter

# === User Parameters ===
MAIN_DIR        = "/home/..." # Update this root directory to the folder your working in
NOISE_THRESHOLD = 1500   # actual voxel intensity cutoff for REV analysis
FIGSIZE         = (10, 8)  # square figure
DPI             = 300

def list_folders(path):
    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.isdir(os.path.join(path, f))
    ]

def list_dicoms(folder):
    return natsorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".dcm")
    ])

def read_dicom_image_and_spacing(path):
    ds = pydicom.dcmread(path)
    return ds.pixel_array.astype(float), ds.PixelSpacing

def interactive_polygon_crop(img):
    mask = np.zeros(img.shape, dtype=bool)
    def onselect(verts):
        path = MplPath(verts)
        ny, nx = img.shape
        yy, xx = np.mgrid[:ny, :nx]
        pts = np.vstack((xx.ravel(), yy.ravel())).T
        mask.flat[:] = path.contains_points(pts)
        plt.close()
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.imshow(img, cmap="gray")
    ax.set_title("Draw polygon around sand-pack,\nthen press ENTER", fontsize=14)
    selector = PolygonSelector(ax, onselect)
    fig.canvas.mpl_connect(
        "key_press_event",
        lambda ev: (selector.disconnect_events(), plt.close())
                   if ev.key=="enter" else None
    )
    plt.show()
    return mask

def save_image(img, out_path, cmap="gray"):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.imshow(img, cmap=cmap)
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def rev_grid_analysis(cropped, mask, spacing, NXs, NYs, prefix):
    cell_sizes_mm   = []
    avg_intensities = []

    save_image(cropped, f"{prefix}_cropped.png", cmap='gray')

    for NX, NY in zip(NXs, NYs):
        h, w   = cropped.shape
        cy, cx = h//NY, w//NX
        ry, rx = h%NY, w%NX

        wy, wx = [0], [0]
        for _ in range(NY):
            extra = 1 if ry>0 else 0
            wy.append(wy[-1] + cy + extra)
            ry -= extra
        for _ in range(NX):
            extra = 1 if rx>0 else 0
            wx.append(wx[-1] + cx + extra)
            rx -= extra

        seg          = np.copy(cropped)
        cell_means   = []

        for j in range(NY):
            for i in range(NX):
                y0,y1 = wy[j], wy[j+1]
                x0,x1 = wx[i], wx[i+1]
                cell   = cropped[y0:y1, x0:x1]
                cmask  = mask[y0:y1, x0:x1]
                vals   = cell[cmask & (cell>=NOISE_THRESHOLD)]
                m      = np.mean(vals) if vals.size>0 else np.nan
                seg[y0:y1, x0:x1] = m
                cell_means.append(m)

        save_image(seg,
                   f"{prefix}_REV_{NX}x{NY}.png",
                   cmap='inferno')

        means = np.array(cell_means)
        avg_intensities.append(np.nanmean(means))

        ph = cy * float(spacing[0])
        pw = cx * float(spacing[1])
        cell_sizes_mm.append(np.sqrt(ph*pw))

    return cell_sizes_mm, avg_intensities

def main():
    folders = list_folders(MAIN_DIR)
    if not folders:
        print("No folders in", MAIN_DIR)
        return
    dcm_files = list_dicoms(folders[0])
    if not dcm_files:
        print("No DICOMs in", folders[0])
        return

    mid = len(dcm_files)//2
    img, spacing = read_dicom_image_and_spacing(dcm_files[mid])
    img           = median_filter(img, size=3)
    prefix        = os.path.splitext(os.path.basename(dcm_files[mid]))[0]

    # histogram of actual voxel intensities
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.hist(img.ravel(), bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(NOISE_THRESHOLD, color='red', linestyle='--',
               label=f"Noise TH = {NOISE_THRESHOLD}")
    ax.set_xlabel("Voxel Intensity", fontsize=28)
    ax.set_ylabel("Voxel Count",      fontsize=28)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=20)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(f"{prefix}_histogram.png", dpi=DPI)
    plt.close(fig)

    # manual polygon crop
    mask      = interactive_polygon_crop(img)
    ys, xs    = np.where(mask)
    ymin,ymax = ys.min(), ys.max()
    xmin,xmax = xs.min(), xs.max()
    cropped   = img[ymin:ymax+1, xmin:xmax+1]
    mask_crop = mask[ymin:ymax+1, xmin:xmax+1]
    print(f"Crop bbox: y[{ymin}:{ymax}] x[{xmin}:{xmax}], shape={cropped.shape}")

    # full NX,NY grid
    nx_list = list(range(3,51,2))
    ny_list = list(range(3,51,2))
    pairs   = [(nx,ny) for nx in nx_list for ny in ny_list]
    NXs, NYs = zip(*pairs)

    # REV analysis
    sizes, ints = rev_grid_analysis(cropped,
                                    mask_crop,
                                    spacing,
                                    NXs, NYs,
                                    prefix)

    # sort by cell size
    pts = sorted(zip(sizes, ints), key=lambda x:x[0])
    sizes_sorted, ints_sorted = map(np.array, zip(*pts))

    # final REV plot (square, 300 dpi)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(
        sizes_sorted,
        ints_sorted,
        marker='o',
        markersize=8,   # larger points
        linewidth=2,    # thicker line
        label = "REV"
        #linestyle='-'
    )
    ax.set_xticks(np.arange(0, np.max(sizes_sorted)+1, 5))
    ax.set_xlim(0, np.max(sizes_sorted))

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    ax.yaxis.offsetText.set_text(r'$\times10^4$')

    ax.set_xlabel("Geo. mean cell size (mm)", fontsize=28)
    ax.set_ylabel("Mean Intensity",    fontsize=28)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=20)
    ax.grid(True)
    #ax.set_title("REV Analysis: Intensity vs. Cell Size", fontsize=18)

    plt.tight_layout()
    fig.savefig(f"{prefix}_REV_dense_sci.png", dpi=DPI)
    plt.show()

if __name__ == "__main__":
    main()
