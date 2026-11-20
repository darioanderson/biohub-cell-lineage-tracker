# Incremental Biohub data

The downloader requests individual files only. Default stage: frames 0-7 from two samples belonging to different embryos, all their GEFF annotations, and image metadata. The selected uncompressed file sizes must total at most 512 MiB. If this cannot be satisfied, the script refuses to download and the stage must be reduced. It never falls back to downloading the whole competition.

Kaggle authentication and competition rule acceptance are required. Browser sign-in does not automatically configure the Python API. No credentials should be pasted into chat or saved in Git. Use Kaggle's official authentication setup locally.

Run from C:\Projects\cell-tracking\advanced:

```powershell
..\.venv\Scripts\python.exe scripts\download_biohub_subset.py --frames 8 --max-mib 512 --download
```

Catalog pagination is cached and resumable. Every downloaded file is checked against the catalog size and receives a SHA-256 receipt. The plan and status live in the shared data directory.

These Zarr stores are intentionally incomplete. Only read the frame numbers recorded in subset-manifest.json. Do not use a generic reader that silently converts absent chunks into zero images. GEFF coordinates and time indices remain in their original coordinate system; filter training labels to the downloaded frame indices.

Sparse Biohub annotations do not label every cell. The CHO training script is NOT a valid drop-in supervised Biohub trainer: it assumes a complete marker set for detection supervision. A sparse-label objective and embryo-disjoint evaluation must be implemented before Biohub training. Increasing sample count is conditional on measured validation progress, not an automatic claim of improvement.
