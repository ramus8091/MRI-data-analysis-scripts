# MRI-data-analysis-scripts
This repository contains python scripts that were generated to read, filter, and extract both qualitative and quantitative information from the MRI images of the sand packs saturated with brine and hydrogen.
It is based on sand packs measiring (32 mm x 106 mm) in diameter, horizontally oriented in a 4.7 Tesla MRI scanner. The sand packs are initailly saturated with brine and then drained with hydrogen. Hydrogen may be replaced with any other gas such as CO2 or N2. 

Key info: Before running any script, create a Python 3 environment, for example here named my_environemnt_name, and install the required packages, especially pydicom (plus numpy, pandas, matplotlib, scipy, among others and—where noted—natsort). Ensure input directories contain valid data with equal number of items (images) and no empty folders.
python3 -m venv my_environemnt_name
source my_environemnt_name/bin/activate
pip install --upgrade pip
pip install pydicom numpy pandas matplotlib scipy natsort


README - REV analysis script
Main target: Examines how measurement scale affects average signal in a single MRI slice of a horizontally oriented sand pack with 100% water-saturated pores. Input is a folder of DICOM images (.dcm). The script loads the series, selects the middle slice, reads pixel/voxel spacing from metadata to work in millimeters, and applies a light 3×3 median filter to reduce reconstruction noise.
Workflow: ROI definition is performed interactively with a polygon tool. The mask bounding box is recorded and the slice is cropped so calculations remain within the sand pack. The cropped ROI is then tiled with progressively larger grids. For each grid, cell means are computed using only pixels at or above a configurable noise threshold. Then cell means are averaged to give one value per grid size. Converting cell dimensions from voxels to millimeters allows plotting mean grid intensity versus geometric mean cell size, yielding an REV/REA curve that typically stabilizes at representative scale.
Configuration & outputs. Key settings (data folder, threshold, figure parameters) are defined near the top of the script. Outputs - saved in the working directory using the source name as prefix—include an intensity histogram (threshold marked), a cropped ROI image, grid heatmaps, and a final REV curve (intensity vs cell size in mm).
Notes. Flat results commonly indicate an ROI that excludes sand or a threshold set higher than the histogram suggests. Revising the threshold or ROI generally resolves issues.
REV/REA refers to: minimal cell size/area at which measured properties become effectively scale-invariant. And ROI refers to a region of interest selected for analysis.
 

README  - T1 analysis script
Main target. Estimates T1 in sand-pack MRI where pores may be 100% brine-saturated or contain both brine and hydrogen. Input is a root directory with systematically organized subfolders (each a time point for the same slice). A calibration reference.dcm is required and is placed in a root directory (e.g., .../testing/reference.dcm). The reference anchors intensity scaling and enables computation of water saturation (Sw) to contextualize T1 over time and saturation states.
Workflow. ROIs are drawn once on the first image of the first folder (up to eight by default; adjustable) and then applied identically to all folders and the calibration reference. A 3×3 median filter reduces noise from image reconstruction (not always necessary). For each folder, DICOM files (images) are read, TR values extracted and sorted, and mean ROI signals measured per TR. The recovery T1 model is fit by non-linear least squares, providing T1 and uncertainty. Sw per ROI is computed as the ratio of the first-TR signal to the corresponding ROI mean on the calibration reference.
Outputs. Per-folder T1 recovery plots (normalised signal vs TR), ROI overlays, and a summary figure of T1 vs time (hours) with error bars and initial Sw annotations. Each folder’s output/ contains figures (≈300 dpi) and an Excel workbook with TR–signal tables, ROI-level normalised signals, Saturation (Sw), and Metadata (TR, TE, imaging frequency, voxel size). ROI overlays for the first and reference images are also saved.

 
README — T2 analysis script
Main target. Estimates T2 from multi-echo acquisitions in the same sand-pack context (100% brine or brine and hydrogen). Input is a root directory with systematically organized time-point subfolders, each holding a multi-echo DICOM series of the same slice. A calibration reference.dcm (e.g., .../testing/reference.dcm) in a root directory provides a stable baseline and supports Sw computation across conditions.
Workflow. ROIs are defined once on the first folder’s image and propagated to all time points and the reference. After 3×3 median filtering, echo times (TE) are extracted and sorted; mean ROI signals are measured per TE. The decay T2  is fit by non-linear least squares to obtain T2 with uncertainty. Sw per ROI is calculated as the ratio of the first-echo signal to the corresponding ROI mean on the calibration reference.
Outputs. Per-folder T2 decay plots (normalised signal vs TE), ROI overlays, and a summary figure of T2 vs time with error bars and initial Sw. Each folder’s output/ contains high-resolution figures and an Excel workbook with Decay Data (TE, signals), Saturation (Sw), and Metadata (EchoTime, RepetitionTime, ImagingFrequency, NumberOfAverages, voxel size).
 

README - Saturation extraction (RARE 2D) script
Main target. Quantifies slice-wise water saturation (Sw) from RARE 2D data of a horizontally oriented sand pack. Input is a root directory with a dedicated reference/ series and additional time-point subfolders, each containing a matching-length DICOM series of the same slice.
Workflow. The reference series is read first to collect metadata and to draw one ROI per slice. These slice-wise masks are then applied to every non-reference folder. Optional bottom trimming (via BOTTOM_OFFSET_MM and PIXEL_SPACING_MM) aligns the cropped region. Noise is reduced with a 3×3 median filter; previews, overlays, and histograms are saved for QC. For each slice and folder, the mean ROI signal is measured and Sw computed using as a signal ration between target and reference slices. Cropped ROI images are exported as PNG and as cropped DICOMs (with updated headers).
Outputs. The reference folder writes saturation_reference.xlsx (slice means + metadata). Each non-reference folder writes saturation_results.xlsx with NonRef_Mean, Ref_Mean, Water_Saturation, and Metadata. PNGs/DICOMs are organized under read_pngs/, cropped_pngs/, and cropped_dicoms/.
 

README  - Dynamic image analysis (RARE 2D) script
Main target. Tracks dynamic changes in Sw for RARE 2D sand-pack data across multiple subfolders (time points). Each subfolder must include a reference.dcm, which anchors intensity scaling and Sw computation.
Workflow. The subfolder’s reference is loaded, a 3×3 median filter applied, and a single polygon ROI drawn on the reference and reused for all images in that subfolder. Optional bottom trimming standardises the cropped base. For each non-reference image, the median ROI signal is measured, Sw is computed relative to the reference slice (background-corrected), and a color-mapped saturation map is generated. Images are cropped by the ROI bounding box and exported as PNG and as cropped DICOMs. Slice orientation (Sagittal/Coronal/Axial) is inferred from ImageOrientationPatient and recorded.
Outputs. Within each subfolder: read_pngs/, cropped_pngs/, cropped_dicoms/, satmap/, and histograms. An Excel file saturation_results.xlsx contains SaturationResults (file name, NonRef_Median, Ref_Median, Water_Saturation, SliceType) and Metadata (EchoTime, RepetitionTime, PixelSpacing, Sequence/Protocol names, NumberOfAverages, TotalScanTime, derived TimePerScan).
Note (all saturation scripts). Sw denotes water saturation derived from ROI signal relative to a reference image; ROI denotes the polygon-defined analysis area reused consistently across time.

