# Cell Lineage Tracker

Streamlit application for 3D time-lapse fluorescence microscopy, predicted cell tracking, lineage inspection, exports, and descriptive control/variant review.

Start with [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) for the tested college demonstration and exact real-data results. See [DATA_LOCATIONS.md](DATA_LOCATIONS.md) for local data paths.

```powershell
cd C:\Projects\cell-tracking\advanced
& "C:\Projects\cell-tracking\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
```

Use http://localhost:8501. The same app pages work for either selected recording. Compare Lineages contains descriptive graphs and a published-reference count check.

The optional experimental Cellpose mode uses official pretrained 2D nucleus weights on image projections and estimates depth from intensities. It is not full 3D instance segmentation, and it has documented misses in the real variant clip. The previous custom detector failed on this acquisition; those failures are preserved under results. No genetic classifier, complete-cell-detection claim, or official competition score is provided.

For a fresh environment, install the project with the `app`, `review`, and `test` optional dependencies. Pretrained weights and source provenance are stored locally under `models/cellpose_nuclei/`.

See LICENSE, CITATION.cff, THIRD_PARTY_NOTICES.md, and THIRD_PARTY_LICENSES for attribution and licenses. The original baseline is preserved separately; modifications belong in this advanced project.


## Portable setup and preserved evidence

Use Python 3.10. Create and activate a local virtual environment, then install with `python -m pip install -e ".[app,review,test]"` and start with `python -m streamlit run app.py --server.address 127.0.0.1`. Large microscopy files and machine environments are not ordinary Git source files. Preserve/download the private release backup and its restore instructions before removing local data. No public data redistribution, genetic diagnosis, or complete detection claim is implied. Original upstream history, MIT license and third-party attribution are retained.

Raw competition microscopy, per-cell source tables and private notebook display outputs are not redistributed in this public source repository. Aggregate experiment reports remain available; the full local recovery archive is private. Use source links and follow the original dataset access conditions.
