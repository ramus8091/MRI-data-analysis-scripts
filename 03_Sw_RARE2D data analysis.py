import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from natsort import natsorted
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from scipy.ndimage import uniform_filter, median_filter

# --- Parameters ---
main_dir = "/home/..." # Update this root directory to the folder your working in
BOTTOM_OFFSET_MM = 4
PIXEL_SPACING_MM = 1  # mm
BACKGROUND_NOISE = 0  # Assumed background noise level

# --- Utilities ---

def list_folders(path):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def list_dicoms(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith('.dcm')]
    return natsorted(files)

def save_png(img, out_path):
    """
    Save a numpy image array as a PNG at 300 DPI.
    """
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_cropped_dicom(img, out_path, ref_ds):
    ds_out = ref_ds.copy()
    ds_out.PixelData = img.astype(ref_ds.pixel_array.dtype).tobytes()
    ds_out.Rows, ds_out.Columns = img.shape
    ds_out.save_as(out_path)

def interactive_polygon_crop(img, idx):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Draw polygon around sand pack in slice {idx}. Press ENTER when done.")

    mask = np.zeros_like(img, dtype=bool)
    polygon_coords = []

    def onselect(verts):
        polygon_coords.extend(verts)
        path = Path(verts)
        y, x = np.mgrid[:img.shape[0], :img.shape[1]]
        coords = np.vstack((x.ravel(), y.ravel())).T
        mask.flat = path.contains_points(coords)
        plt.close()

    selector = PolygonSelector(ax, onselect)

    def on_key(event):
        if event.key == 'enter':
            selector.disconnect_events()
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    return mask

def plot_and_save_histogram(values, title, out_path):
    plt.figure()
    plt.hist(values, bins=50, color='gray', edgecolor='black')
    plt.title(title)
    plt.xlabel("Voxel Intensity", fontsize=24)
    plt.ylabel("Frequency", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# --- Main Script ---

def main():
    folders = list_folders(main_dir)
    if 'reference' not in folders:
        print("ERROR: No 'reference' folder found in your main directory.")
        return

    reference_folder = os.path.join(main_dir, 'reference')
    ref_dicom_files = list_dicoms(reference_folder)
    print(f"Reference folder contains {len(ref_dicom_files)} DICOM files.")

    # --- EXTRACT REFERENCE METADATA ONCE ---
    ref0 = pydicom.dcmread(os.path.join(reference_folder, ref_dicom_files[0]))
    num_slices = len(ref_dicom_files)
    total_scan_time = getattr(ref0, 'TotalScanTime', None)
    metadata_ref = {
        'EchoTime_ms': getattr(ref0, 'EchoTime', None),
        'RepetitionTime_ms': getattr(ref0, 'RepetitionTime', None),
        'VoxelSpacing_mm': getattr(ref0, 'PixelSpacing', None),
        'SequenceName': getattr(ref0, 'SequenceName', None),
        'ProtocolName': getattr(ref0, 'ProtocolName', None),
        'NumberOfAverages': getattr(ref0, 'NumberOfAverages', None),
        'TotalScanTime_s': total_scan_time,
    }
    if total_scan_time and num_slices:
        metadata_ref['TimePerSlice_s'] = total_scan_time / num_slices

    # --- Prepare reference output folders ---
    read_pngs_ref = os.path.join(reference_folder, 'read_pngs')
    cropped_pngs_ref = os.path.join(reference_folder, 'cropped_pngs')
    cropped_dicoms_ref = os.path.join(reference_folder, 'cropped_dicoms')
    os.makedirs(read_pngs_ref, exist_ok=True)
    os.makedirs(cropped_pngs_ref, exist_ok=True)
    os.makedirs(cropped_dicoms_ref, exist_ok=True)

    # --- Step 1: Process reference images ---
    masks = []
    bottom_limits = []
    ref_means = []
    slice_results_ref = []

    for idx, fname in enumerate(ref_dicom_files):
        path = os.path.join(reference_folder, fname)
        ds = pydicom.dcmread(path)
        raw_img = ds.pixel_array.astype(float)
        img = median_filter(raw_img, size=3)

        save_png(img, os.path.join(read_pngs_ref, fname.replace('.dcm', '_median3.png')))
        save_png(uniform_filter(img, size=3),
                 os.path.join(read_pngs_ref, fname.replace('.dcm', '_ma3.png')))

        mask = interactive_polygon_crop(img, idx)
        masks.append(mask)

        if not np.any(mask):
            print(f"WARNING: Empty mask in slice {idx}")
            bottom_limits.append(0)
            ref_means.append(0)
            continue

        ys, xs = np.where(mask)
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        masked_img = np.where(mask, img, 0)
        cropped = masked_img[ymin:ymax+1, xmin:xmax+1]

        offset_px = int(round(BOTTOM_OFFSET_MM / PIXEL_SPACING_MM))
        bottom_trim = ymax - offset_px
        if ymax > bottom_trim:
            cropped = cropped[:- (ymax - bottom_trim), :]

        save_png(cropped, os.path.join(cropped_pngs_ref, fname.replace('.dcm', '_cropped.png')))
        save_cropped_dicom(cropped, os.path.join(cropped_dicoms_ref, fname), ds)

        voxel_values = img[mask]
        mean_signal = np.mean(voxel_values)
        ref_means.append(mean_signal)
        bottom_limits.append(bottom_trim)

        hist_path = os.path.join(reference_folder, f"histogram_ref_slice{idx:03d}.png")
        plot_and_save_histogram(voxel_values,
                                f"Ref Slice {idx} - Voxel Intensity Histogram",
                                hist_path)

        slice_results_ref.append((fname, mean_signal))

    df_ref = pd.DataFrame(slice_results_ref, columns=['Slice', 'Mean_Signal'])
    ref_xlsx = os.path.join(reference_folder, "saturation_reference.xlsx")
    with pd.ExcelWriter(ref_xlsx) as writer:
        df_ref.to_excel(writer, sheet_name='SaturationReference', index=False)
        meta_df = pd.DataFrame.from_dict(metadata_ref, orient='index', columns=['Value'])
        meta_df.index.name = 'Parameter'
        meta_df.to_excel(writer, sheet_name='Metadata')

    print("✅ Reference folder processed and Excel saved.")

    # --- Step 2: Process non-reference folders ---
    for folder in folders:
        if folder == 'reference':
            continue

        folder_path = os.path.join(main_dir, folder)
        dicom_files = list_dicoms(folder_path)
        if len(dicom_files) != len(ref_dicom_files):
            print(f"Skipping folder '{folder}' due to mismatch in slice count.")
            continue

        ds0 = pydicom.dcmread(os.path.join(folder_path, dicom_files[0]))
        total_scan_time_nr = getattr(ds0, 'TotalScanTime', None)
        metadata_nr = {
            'EchoTime_ms': getattr(ds0, 'EchoTime', None),
            'RepetitionTime_ms': getattr(ds0, 'RepetitionTime', None),
            'VoxelSpacing_mm': getattr(ds0, 'PixelSpacing', None),
            'SequenceName': getattr(ds0, 'SequenceName', None),
            'ProtocolName': getattr(ds0, 'ProtocolName', None),
            'NumberOfAverages': getattr(ds0, 'NumberOfAverages', None),
            'TotalScanTime_s': total_scan_time_nr,
        }
        if total_scan_time_nr and len(dicom_files):
            metadata_nr['TimePerSlice_s'] = total_scan_time_nr / len(dicom_files)

        read_pngs = os.path.join(folder_path, 'read_pngs')
        cropped_pngs = os.path.join(folder_path, 'cropped_pngs')
        cropped_dicoms = os.path.join(folder_path, 'cropped_dicoms')
        os.makedirs(read_pngs, exist_ok=True)
        os.makedirs(cropped_pngs, exist_ok=True)
        os.makedirs(cropped_dicoms, exist_ok=True)

        slice_results = []

        for idx, fname in enumerate(dicom_files):
            dcm_path = os.path.join(folder_path, fname)
            ds = pydicom.dcmread(dcm_path)
            raw_img = ds.pixel_array.astype(float)
            img = median_filter(raw_img, size=3)

            save_png(img, os.path.join(read_pngs, fname.replace('.dcm', '_median3.png')))
            save_png(uniform_filter(img, size=3),
                     os.path.join(read_pngs, fname.replace('.dcm', '_ma3.png')))

            mask = masks[idx]
            if not np.any(mask):
                print(f"Skipping slice {idx} in folder '{folder}' due to empty mask.")
                continue

            voxel_values = img[mask]
            mean_nonref = np.mean(voxel_values)
            mean_ref = ref_means[idx]
            sat = 0.0
            if mean_ref > BACKGROUND_NOISE:
                sat = (mean_nonref - BACKGROUND_NOISE) / (mean_ref - BACKGROUND_NOISE)
            saturation = max(min(sat, 1.0), 0.0)

            hist_path = os.path.join(folder_path, f"histogram_slice{idx:03d}.png")
            plot_and_save_histogram(voxel_values,
                                    f"{folder} Slice {idx} - Histogram",
                                    hist_path)

            ys, xs = np.where(mask)
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()
            masked_img = np.where(mask, img, 0)
            cropped = masked_img[ymin:ymax+1, xmin:xmax+1]

            crop_bottom = ymax
            global_bottom = bottom_limits[idx]
            if crop_bottom > global_bottom:
                cropped = cropped[:-(crop_bottom - global_bottom), :]

            save_png(cropped, os.path.join(cropped_pngs, fname.replace('.dcm', '_cropped.png')))
            save_cropped_dicom(cropped, os.path.join(cropped_dicoms, fname), ds)

            slice_results.append((fname, mean_nonref, mean_ref, saturation))
            print(f"{folder} Slice {idx}: Mean_NonRef={mean_nonref:.1f}, Ref={mean_ref:.1f}, Sat={saturation:.2f}")

        df_nr = pd.DataFrame(slice_results,
                             columns=['Slice', 'NonRef_Mean', 'Ref_Mean', 'Water_Saturation'])
        xlsx_path = os.path.join(folder_path, "saturation_results.xlsx")
        with pd.ExcelWriter(xlsx_path) as writer:
            df_nr.to_excel(writer, sheet_name='SaturationResults', index=False)
            meta_df_nr = pd.DataFrame.from_dict(metadata_nr, orient='index', columns=['Value'])
            meta_df_nr.index.name = 'Parameter'
            meta_df_nr.to_excel(writer, sheet_name='Metadata')

        print(f"✅ Folder '{folder}' processed and Excel saved.")

    print("🎉 All done.")

if __name__ == "__main__":
    main()
