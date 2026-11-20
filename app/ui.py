"""Streamlit UI helpers."""

from __future__ import annotations

import io
import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st

from biohub.analysis import AnalysisResult, run_tracking_pipeline
from biohub.config import Config, PIPELINE_VERSION, SCALE
from biohub.dataset_catalog import DatasetCatalog, DatasetEntry, discover_catalog, load_zarr_preview
from biohub.export import division_events
from biohub.sample_data import ensure_sample_data, load_sample_volume, synthetic_volume


def invalidate_analysis():
    """A changed input must never inherit predictions or review edits."""
    st.session_state.result = None
    st.session_state.curation_audit = None
    for key in list(st.session_state):
        if key in ('overlay_gif', 'review_frame', 'det_t', 'det_z', 'vol_t', 'vol_z',
                   'lineage_focus_node', 'result_requested_config_hash', 'input_sha256') or key.startswith(('curate_', 'lineage_role_jump_')):
            st.session_state.pop(key, None)


def init_session() -> None:
    defaults = {
        "cfg": Config(),
        "volume_4d": None,
        "volume_source": "biohub",
        "dataset_name": "",
        "selected_entry": None,
        "catalog": None,
        "dataset_root": "",
        "result": None,
        "preview_frames": 20,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def scan_dataset_root(root_path: str) -> DatasetCatalog:
    catalog = discover_catalog(Path(root_path))
    st.session_state.catalog = catalog
    st.session_state.cfg = st.session_state.cfg.copy_with(
        data_root=catalog.root,
        train_dir=catalog.train_dir,
        test_dir=catalog.test_dir,
    )
    return catalog


def select_dataset_entry(label: str) -> DatasetEntry:
    catalog: DatasetCatalog = st.session_state.catalog
    entry = catalog.by_label(label)
    if entry is None:
        raise ValueError(f"Unknown dataset: {label}")
    if st.session_state.get('selected_entry') != entry:
        invalidate_analysis()
        st.session_state.volume_4d = None
    st.session_state.selected_entry = entry
    st.session_state.dataset_name = entry.name
    return entry


def load_selected_volume(
    entry: DatasetEntry,
    preview_frames: int,
    progress=None,
) -> np.ndarray:
    vol4d = load_zarr_preview(entry, max_frames=preview_frames, progress_callback=progress)
    invalidate_analysis()
    st.session_state.volume_4d = vol4d
    st.session_state.volume_source = "biohub"
    return vol4d


def load_demo_volume(preview_frames: int) -> Tuple[np.ndarray, str]:
    path = ensure_sample_data()
    vol4d = load_sample_volume(path)
    if preview_frames and vol4d.shape[0] > preview_frames:
        vol4d = vol4d[:preview_frames]
    invalidate_analysis()
    st.session_state.volume_4d = vol4d
    st.session_state.volume_source = "demo"
    st.session_state.selected_entry = None
    st.session_state.dataset_name = "synthetic_demo"
    return vol4d, "synthetic_demo"


def load_uploaded_npy(uploaded_bytes: bytes, preview_frames: int) -> Tuple[np.ndarray, str]:
    vol4d = np.load(io.BytesIO(uploaded_bytes), allow_pickle=False)
    if not isinstance(vol4d, np.ndarray) or vol4d.ndim != 4 or min(vol4d.shape) < 1:
        raise ValueError(f"Expected 4D array (T,Z,Y,X); got shape {vol4d.shape}.")
    if vol4d.dtype.kind not in 'uif' or not np.isfinite(vol4d).all():
        raise ValueError('The movie must contain finite numeric intensities.')
    invalidate_analysis()
    digest = hashlib.sha256(uploaded_bytes).hexdigest()
    from app.recording_metadata import known_recording, calibrated_config
    info = known_recording(digest)
    st.session_state.cfg = calibrated_config(st.session_state.cfg, info)
    st.session_state.input_sha256 = digest
    if preview_frames and vol4d.shape[0] > preview_frames:
        vol4d = vol4d[:preview_frames]
    st.session_state.volume_4d = vol4d
    st.session_state.volume_source = "upload_npy"
    st.session_state.selected_entry = None
    st.session_state.dataset_name = "uploaded_volume"
    return vol4d, "uploaded_volume"


def run_pipeline_on_session(cfg: Config, preview_frames: int) -> AnalysisResult:
    entry: Optional[DatasetEntry] = st.session_state.get("selected_entry")
    name = st.session_state.get("dataset_name", "dataset")

    if entry is not None and st.session_state.volume_source == "biohub":
        # Stream from zarr for full runs; preview mode caps frames inside the pipeline.
        max_frames = preview_frames if preview_frames < entry.shape[0] else None
        result = run_tracking_pipeline(entry.zarr_path, cfg, dataset_name=name, max_frames=max_frames)
        result.stats['input_paths'] = [str(entry.zarr_path)]
    else:
        vol4d = st.session_state.volume_4d
        if vol4d is None:
            raise ValueError("No volume loaded.")
        frames = vol4d[:preview_frames] if preview_frames and vol4d.shape[0] > preview_frames else vol4d
        result = run_tracking_pipeline(frames, cfg, dataset_name=name)
        from app.recording_metadata import known_recording
        info = known_recording(st.session_state.get('input_sha256'))
        result.stats.update(time_interval_verified=info['time_verified'], spatial_calibration_verified=info['spatial_verified'],
                            input_sha256=st.session_state.get('input_sha256'))
    from biohub.manifest import config_sha256
    st.session_state.result_requested_config_hash = config_sha256(cfg)
    return result


def sidebar_settings(cfg: Config) -> tuple[Config, str, int]:
    with st.sidebar.expander("Analysis settings", expanded=True):
        mode = st.radio("Processing mode", ["Preview", "Full sequence"], index=0)
        if mode == "Preview":
            preview_frames = st.number_input("Preview frames", 3, 200, int(cfg.preview_max_frames))
        else:
            preview_frames = 10_000
        cfg = cfg.copy_with(
            thresh_rel=st.slider("Detection threshold (rel.)", 0.18, 0.42, float(cfg.thresh_rel), 0.02),
            max_link_dist_um=st.slider("Max link distance (µm)", 8.0, 14.0, float(cfg.max_link_dist_um), 0.5),
            div_parent_dist_um=st.slider("Division parent gate (µm)", 8.0, 14.0, float(cfg.div_parent_dist_um), 0.5),
            div_sister_dist_um=st.slider("Division sister gate (µm)", 5.0, 10.0, float(cfg.div_sister_dist_um), 0.5),
            detect_divisions=st.checkbox("Detect divisions", value=cfg.detect_divisions),
            prune_isolated_nodes=st.checkbox("Prune isolated nodes", value=cfg.prune_isolated_nodes),
            use_rich_linking=st.checkbox("Rich linking cost", value=cfg.use_rich_linking),
            preview_max_frames=int(preview_frames),
        )
        st.caption(f"Pipeline v{PIPELINE_VERSION}")
        if st.session_state.get("result") is not None and st.session_state.result.stats.get("spacing_note"):
            st.caption(st.session_state.result.stats['spacing_note'])
        else:
            st.caption(f"Voxel scale (Z,Y,X): {cfg.voxel_size_zyx_um[0]}, {cfg.voxel_size_zyx_um[1]}, {cfg.voxel_size_zyx_um[2]} µm")
    return cfg, mode, int(preview_frames)


def render_dataset_sidebar() -> tuple[str, Optional[bytes]]:
    """Primary data-source controls. Returns (source_mode, uploaded_bytes)."""
    st.sidebar.header("Dataset")
    source = st.sidebar.radio(
        "Source",
        ["biohub", "demo", "upload_npy"],
        format_func=lambda x: {
            "biohub": "Local Biohub dataset",
            "demo": "Synthetic demo",
            "upload_npy": "Upload .npy volume",
        }[x],
        index=0 if st.session_state.volume_source != "demo" else 1,
    )
    if source != st.session_state.volume_source:
        invalidate_analysis()
        st.session_state.update(volume_4d=None, selected_entry=None, dataset_name='')
    st.session_state.volume_source = source
    uploaded = None

    if source == "biohub":
        default_root = st.session_state.get("dataset_root", "")
        root_input = st.sidebar.text_input(
            "Dataset root directory",
            value=default_root,
            placeholder="/path/to/biohub-cell-tracking-during-development",
            help="Folder containing train/ and/or test/ with .zarr and .geff files.",
        )
        if st.sidebar.button("Scan dataset", type="primary", use_container_width=True):
            if not root_input.strip():
                st.sidebar.error("Enter the path to your downloaded dataset.")
            else:
                try:
                    catalog = scan_dataset_root(root_input.strip())
                    st.session_state.dataset_root = root_input.strip()
                    st.sidebar.success(
                        f"Found {len(catalog.entries)} volumes "
                        f"({catalog.n_train} train, {catalog.n_test} test)."
                    )
                except Exception as exc:
                    st.sidebar.error(str(exc))

        catalog: Optional[DatasetCatalog] = st.session_state.get("catalog")
        if catalog and catalog.entries:
            split_filter = st.sidebar.selectbox(
                "Split",
                ["all", "train", "test"],
                format_func=lambda s: {"all": "All splits", "train": "Train", "test": "Test"}[s],
            )
            visible = catalog.entries if split_filter == "all" else catalog.entries_for_split(split_filter)
            labels = [e.label for e in visible]
            if labels:
                current = st.session_state.get("dataset_label")
                idx = labels.index(current) if current in labels else 0
                chosen = st.sidebar.selectbox("Volume", labels, index=idx)
                st.session_state.dataset_label = chosen
                if st.sidebar.button("Load volume", use_container_width=True):
                    try:
                        entry = select_dataset_entry(chosen)
                        progress = st.sidebar.progress(0.0, text="Loading frames…")
                        pf = int(st.session_state.get("preview_frames", 20))

                        def _prog(p):
                            progress.progress(min(p, 1.0))

                        load_selected_volume(entry, pf, progress=_prog)
                        progress.empty()
                        st.sidebar.success(f"Loaded {entry.name} ({entry.shape[0]} frames).")
                    except Exception as exc:
                        st.sidebar.error(str(exc))
            else:
                st.sidebar.info("No volumes in this split.")
        elif source == "biohub":
            st.sidebar.caption("Point to your local dataset root, then click **Scan dataset**.")

    elif source == "demo":
        if st.sidebar.button("Load synthetic demo", use_container_width=True):
            pf = int(st.session_state.get("preview_frames", 12))
            load_demo_volume(pf)
            st.sidebar.success("Synthetic demo loaded.")

    elif source == "upload_npy":
        uploaded = st.sidebar.file_uploader("Volume array (T,Z,Y,X)", type=["npy"])
        if uploaded is not None and st.sidebar.button("Load upload", use_container_width=True):
            try:
                pf = int(st.session_state.get("preview_frames", 20))
                load_uploaded_npy(uploaded.getvalue(), pf)
                st.sidebar.success("Upload loaded.")
            except Exception as exc:
                st.sidebar.error(str(exc))

    return source, uploaded
