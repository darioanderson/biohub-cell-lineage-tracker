# Local dataset locations

For the original single-recording UI, choose Upload .npy volume and use:
C:/Projects/cell-tracking/advanced/data/cshaper_demo/Sample18_t080_082.npy
This is real Sample18 microscopy, three frames (80-82).

Original synthetic UI example (NOT real microscopy):
C:/Projects/cell-tracking/advanced/data/sample/synthetic_volume.npy

Earlier real Cell Tracking Challenge CHO data:
C:/Projects/cell-tracking/data/Fluo-N3DH-CHO
This is TIFF-based data, not a Biohub Zarr root or an uploadable .npy movie.

Older empty Biohub download attempt: the local folder
C:/Projects/cell-tracking/data/biohub-stage1
contains an authentication_required status file and zero downloaded competition bytes.
Do not set this empty folder as a dataset root expecting images.

Actual local Biohub dataset root (three zebrafish training recordings):
C:/Users/Admin/Desktop/biohub_data
Train images and annotations: C:/Users/Admin/Desktop/biohub_data/train
Two converted complete 100-frame recordings for the comparison workspace:
C:/Projects/cell-tracking/advanced/data/local_biohub_pair/44b6_0113de3b_all100.npy
C:/Projects/cell-tracking/advanced/data/local_biohub_pair/44b6_0b24845f_all100.npy
Each is 838,860,928 bytes. Use Files on this PC; do not upload through the browser.
Choose Biohub original pipeline — classical detection. Their genotypes are not
established; the app labels them Recording A/B, not biological control/mutant.

Modified project: C:/Projects/cell-tracking/advanced
Untouched original baseline: C:/Projects/cell-tracking/baseline

Larger real wild-type and wee-1.1 mutant test clips are prepared under:
C:/Projects/cell-tracking/advanced/data/real_control_variant_large
Read its PROVENANCE.json before interpreting differences.

Prepared larger native-grid real clips (30 frames each):
Control: C:/Projects/cell-tracking/advanced/data/real_control_variant_large/CONTROL_WT_Emb8_t001_030.npy (612,434,048 bytes)
Variant: C:/Projects/cell-tracking/advanced/data/real_control_variant_large/VARIANT_wee-1.1_MT2_t001_030.npy (1,028,014,208 bytes)
Use Control and variant > Files on this PC > Use downloaded 30-frame recordings.
Both embryos are uncompressed. Time sampling, biological stages, and voxel grids are not aligned. Keep the alignment checkbox unticked. Exploratory comparison graphs remain available; paired differences require verified alignment.
