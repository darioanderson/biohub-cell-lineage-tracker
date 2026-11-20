"""Load the verified organizer-model Kaggle output for a matching local test movie."""
import hashlib
import io
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import streamlit as st

from biohub.analysis import AnalysisResult, FrameResult
from biohub.tracking import annotate_lineage_ids
from app import kaggle_batch

CSV = Path(__file__).resolve().parents[1] / "results/kaggle_organizer_baseline_v3/submission.csv"
SHA256 = "94d82d2f7453cba94722a651d826e170a7c29ebcf261f53f650ab25588cda84e"


def load_result(dataset, shape, raw=None):
    source = "Kaggle organizer pretrained model, custom upload"
    if raw is None:
        raw = CSV.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SHA256:
            raise ValueError("The Kaggle CSV does not match the recorded run checksum.")
        source = "Kaggle organizer pretrained model, version 3"
    table = pd.read_csv(io.BytesIO(raw))
    table = table[table.dataset.eq(dataset)]
    nodes = table[table.row_type.eq("node")].copy()
    edges = table[table.row_type.eq("edge")].copy()
    axes = ["t", "z", "y", "x"]
    if nodes.empty or nodes.node_id.duplicated().any():
        raise ValueError("The Kaggle run has no valid graph for this dataset.")
    coords = nodes[axes].to_numpy()
    if ((coords < 0) | (coords >= np.asarray(shape))).any():
        raise ValueError("Prediction coordinates exceed the selected volume bounds.")
    times = nodes.set_index("node_id").t
    if not edges.source_id.map(times).lt(edges.target_id.map(times)).all():
        raise ValueError("The prediction graph has invalid edges.")
    if edges.target_id.duplicated().any() or edges.groupby("source_id").size().gt(2).any():
        raise ValueError("The prediction graph has invalid parent or daughter counts.")
    records = nodes.to_dict("records")
    annotate_lineage_ids(records, edges.to_dict("records"))
    frames = [FrameResult(t, np.empty((0, 3)), np.empty(0), np.empty(0), []) for t in range(shape[0])]
    for t, group in nodes.groupby("t"):
        missing = np.full(len(group), np.nan)
        frames[t] = FrameResult(t, group[axes[1:]].to_numpy(), missing, missing.copy(), group.node_id.tolist())
    return AnalysisResult(dataset, tuple(shape), pd.DataFrame(records), edges, frames,
                          {"source": source})


def custom_inference():
    with st.expander("Run pretrained model on an uploaded movie"):
        st.caption("Use a Biohub-format uint16 (T,64,256,256) .npy at the original voxel spacing, up to 64 MiB. Kaggle processes each upload as a private batch job.")
        movie = st.file_uploader("Test movie", type="npy", key="kaggle_movie")
        owner = st.text_input("Kaggle username", value="dushi2")
        allowed = st.checkbox("I have permission to upload this movie to my private Kaggle dataset.")
        job = st.session_state.get("kaggle_upload_job")
        if st.button("Upload privately to Kaggle", disabled=movie is None or not allowed or job is not None):
            try:
                job = kaggle_batch.start(movie.getvalue(), owner.strip())
                st.session_state.kaggle_upload_job = job
                st.session_state.kaggle_upload_bytes = movie.getvalue()
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                st.error(str(exc))
        if job is None:
            return
        st.write(job["ref"])
        if st.button("Check Kaggle job"):
            try:
                status, raw = kaggle_batch.check(job)
                st.write(status)
                if raw is not None:
                    result = load_result("uploaded_volume", job["shape"], raw)
                    state = st.session_state
                    state.volume_4d = np.load(io.BytesIO(state.kaggle_upload_bytes), allow_pickle=False)
                    state.volume_source = "upload_npy"
                    state.selected_entry = None
                    state.dataset_name = result.dataset_name
                    state.result = result
                    state.curation_audit = None
                    state.pop("overlay_gif", None)
                    st.success("Predictions ready in Detection, Lineage, and Exports.")
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                st.error(str(exc))
        if st.button("Clear local job reference"):
            st.session_state.pop("kaggle_upload_job", None)
            st.session_state.pop("kaggle_upload_bytes", None)
            st.rerun()


def kaggle_tab():
    from app.paired_movies import paired_movies_page
    paired_movies_page()


def single_recording_tools():
    st.subheader("Compare Lineages")
    example_path = CSV.parents[2] / "data/cshaper_demo/Sample18_t080_082.npy"
    if example_path.exists() and st.button("Run real Sample18 example"):
        from biohub.analysis import run_tracking_pipeline
        volume = np.load(example_path, allow_pickle=False)
        config = st.session_state.cfg.copy_with(
            detector_backend="cshaper",
            learned_detector_path=CSV.parents[2] / "results/kaggle_ce_cshaper_v2/detector.pt",
            voxel_size_zyx_um=(1.0, 1.0, 1.0),
        )
        with st.spinner("Running the trained detector on three real frames"):
            result = run_tracking_pipeline(list(volume), config, dataset_name="Sample18_t080_082")
        result.stats["source"] = "CShaper held-out Sample18, frames 80–82; Kaggle-trained center model"
        result.stats["spacing_note"] = "Tracking uses unit voxel spacing; source NIfTI header reports 0.25 per axis, with physical units not verified."
        st.session_state.update(volume_4d=volume, volume_source="upload_npy", selected_entry=None,
                                dataset_name=result.dataset_name, result=result, cfg=config,
                                curation_audit=None)
        st.session_state.pop("overlay_gif", None)
        st.success(f"Real movie analyzed: {result.n_nodes} observations and {result.n_edges} links. Open Detection.")
    with st.expander("Run the trained CShaper detector"):
        st.caption("Uses our saved Kaggle checkpoint locally. Load a CShaper-like nucleus movie as (T,Z,Y,X). Supply its voxel spacing for physical tracking measurements.")
        spacing = [st.number_input(f"{axis} voxel spacing (µm)", min_value=0.001, value=1.0, key=f"cshaper_{axis}") for axis in ("Z", "Y", "X")]
        confirmed = st.checkbox("I checked these voxel sizes against the recording metadata.")
        if st.button("Analyze with trained CShaper model", disabled=not confirmed):
            volume = st.session_state.get("volume_4d")
            if volume is None:
                st.error("Load a movie first.")
            else:
                from biohub.analysis import run_tracking_pipeline
                config = st.session_state.cfg.copy_with(
                    detector_backend="cshaper",
                    learned_detector_path=CSV.parents[2] / "results/kaggle_ce_cshaper_v2/detector.pt",
                    voxel_size_zyx_um=tuple(spacing),
                )
                try:
                    with st.spinner("Analyzing loaded frames"):
                        result = run_tracking_pipeline(list(volume), config, dataset_name=st.session_state.dataset_name)
                    result.stats["source"] = "Kaggle-trained CShaper detector with classical tracking"
                    if st.session_state.dataset_name == "Sample18_t080_082":
                        result.stats["spacing_note"] = "Sample18 physical calibration is unverified; tracking uses the supplied spacing. Source header values are 0.25 per axis."
                    st.session_state.result = result
                    st.session_state.cfg = config
                    st.session_state.curation_audit = None
                    st.session_state.pop("overlay_gif", None)
                    st.success(f"Ready: {result.n_nodes} observations. Open Detection, Lineage, or Exports.")
                except (OSError, ValueError, RuntimeError) as exc:
                    st.error(str(exc))
    current = st.session_state.get("result")
    if current is not None:
        from app.result_review import render_real_summary
        with st.expander("Real model output from the loaded recording", expanded=True):
            render_real_summary(current, key="comparison_real")
    st.caption("Research goal: compare a known variant with controls at the same developmental stage. A single recording cannot supply this comparison.")
    with st.expander("Compare annotated lineage measurements"):
        st.write("Upload one control and one variant CSV with columns cell, stage_min, parent, daughter_a, daughter_b. Use corresponding cell names and times measured from the same developmental landmark.")
        control = st.file_uploader("Control division events", type="csv", key="comparison_control")
        variant = st.file_uploader("Variant division events", type="csv", key="comparison_variant")
        if control is not None and variant is not None:
            try:
                st.caption("Optional columns: birth_min, last_seen_min, path_length_um. Use the same stage landmark and comparable observation windows.")
                timing_threshold = st.number_input(
                    "Division timing difference to flag (minutes)",
                    min_value=0.1, value=3.0, step=0.1,
                    key="comparison_timing_threshold",
                )
                comparison = compare_divisions(control, variant)
                if any(getattr(upload, "name", "").startswith("DEMO_ONLY_") for upload in (control, variant)):
                    st.warning("Synthetic demonstration data only — these graphs are not model predictions or evidence of a genetic defect.")
                render_comparison_graphs(comparison)
                flagged = comparison[
                    comparison.difference_min.abs().ge(timing_threshold)
                    | comparison.same_named_parent.eq(False).fillna(False)
                    | comparison.same_named_daughters.eq(False).fillna(False)
                ]
                st.write(f"{len(flagged)} named cells flagged for review")
                st.dataframe(flagged, hide_index=True)
                st.download_button(
                    "Download flagged cells", flagged.to_csv(index=False),
                    file_name="lineage_review_flags.csv", mime="text/csv",
                )
                st.dataframe(comparison, hide_index=True)
                st.caption("Flags identify timing or relationship differences for review. The timing threshold is a screening setting and needs calibration against repeated controls.")
                st.caption("Differences are variant minus control. Blank values indicate missing measurements. An early last observation can reflect the recording window or tracking quality.")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    st.subheader("Kaggle model results")
    st.markdown("[Organizer pretrained model run](https://www.kaggle.com/code/dushi2/biohub-organizer-pretrained-submission-check)")
    custom_inference()
    entry = st.session_state.get("selected_entry")
    if entry is None or entry.split != "test" or st.session_state.volume_4d is None:
        st.info("Load a matching Biohub test volume from the sidebar first.")
        return
    if st.button("Use saved Kaggle predictions"):
        try:
            result = load_result(entry.name, entry.shape)
        except (OSError, ValueError) as exc:
            st.error(str(exc))
            return
        st.session_state.result = result
        st.session_state.curation_audit = None
        st.session_state.pop("overlay_gif", None)
        st.success(f"Loaded {result.n_nodes:,} detections and {result.n_edges:,} links. Open Detection, Lineage, or Exports.")


def render_comparison_graphs(comparison):
    """Plot paired measurements without imputing absent cells or measurements."""
    import matplotlib.pyplot as plt

    st.markdown("**Compare part by part**")
    st.caption("Choose corresponding named cells or a lineage prefix (for example AB). Graphs use your uploaded measurements; missing values are omitted.")
    prefix = st.text_input("Lineage / part prefix", key="comparison_prefix").strip()
    subset = comparison[comparison.cell.str.startswith(prefix)] if prefix else comparison
    cells = st.multiselect("Cells to compare", subset.cell.tolist(),
                           default=subset.cell.tolist()[:20], key="comparison_cells")
    subset = subset[subset.cell.isin(cells)]
    if subset.empty:
        st.info("Select at least one named cell with measurements.")
        return

    metrics = [
        ("Division timing", "stage_min", "Minutes from stage landmark"),
        ("Cell cycle", "cell_cycle_min", "Minutes from birth to division"),
        ("Birth timing", "birth_min", "Minutes from stage landmark"),
        ("Last observation", "last_seen_min", "Minutes from stage landmark"),
        ("Movement", "path_length_um", "Path length (µm)"),
    ]
    tabs = st.tabs([item[0] for item in metrics] + ["Parent and daughters"])
    for tab, (label, field, unit) in zip(tabs, metrics):
        with tab:
            columns = [f"{field}_control", f"{field}_variant"]
            if not set(columns).issubset(subset.columns):
                st.info(f"No {label.lower()} measurements provided in the CSVs.")
                continue
            paired = subset.dropna(subset=columns).set_index("cell")[columns]
            if paired.empty:
                st.info("No cells have this measurement in both recordings.")
                continue
            st.caption(f"{len(paired)} paired cells; {len(subset) - len(paired)} omitted because one or both measurements are missing.")
            fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
            positions = np.arange(len(paired))
            axes[0].bar(positions - .2, paired.iloc[:, 0], .4, label="Control", color="#3567a5")
            axes[0].bar(positions + .2, paired.iloc[:, 1], .4, label="Variant", color="#df8b35")
            axes[0].set_ylabel(unit)
            axes[0].set_title(label)
            axes[0].legend()
            delta = paired.iloc[:, 1] - paired.iloc[:, 0]
            axes[1].bar(positions, delta, color=np.where(delta >= 0, "#df8b35", "#3567a5"))
            axes[1].axhline(0, color="gray", linewidth=1)
            axes[1].set_ylabel("Variant − control")
            if field == "stage_min":
                threshold = st.session_state.get("comparison_timing_threshold", 3.)
                for value in (-threshold, threshold):
                    axes[1].axhline(value, color="gray", linestyle="--", linewidth=.8)
            for ax in axes:
                ax.set_xticks(positions, paired.index, rotation=60, ha="right")
            st.pyplot(fig)
            plt.close(fig)
    with tabs[-1]:
        selected = st.selectbox("Inspect one cell's family", subset.cell.tolist())
        row = subset.set_index("cell").loc[selected]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for ax, group, color in zip(axes, ("control", "variant"), ("#3567a5", "#df8b35")):
            labels = [row[f"parent_{group}"], selected, row[f"daughter_a_{group}"], row[f"daughter_b_{group}"]]
            points = [(0.5, .9), (.5, .5), (.2, .1), (.8, .1)]
            for a, b in ((0, 1), (1, 2), (1, 3)):
                if pd.notna(labels[a]) and pd.notna(labels[b]) and str(labels[a]).strip() and str(labels[b]).strip():
                    ax.annotate("", xy=points[b], xytext=points[a], arrowprops={"arrowstyle": "->", "color": color})
            for name, point in zip(labels, points):
                ax.text(*point, str(name) if pd.notna(name) and str(name).strip() else "Not recorded",
                        ha="center", va="center", bbox={"boxstyle": "round", "fc": "white", "ec": color})
            ax.set_title(group.title())
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    st.caption("These are descriptive comparisons of the supplied recordings. Differences alone do not establish a genetic defect or cause; last observation is not evidence of cell death.")


def compare_divisions(control_file, variant_file):
    fields = ["cell", "stage_min", "parent", "daughter_a", "daughter_b"]
    optional = ["birth_min", "last_seen_min", "path_length_um"]
    tables = []
    for source in (control_file, variant_file):
        table = pd.read_csv(io.BytesIO(source.getvalue()), dtype={"cell": str})
        if not set(fields).issubset(table):
            raise ValueError("Each CSV must contain: " + ", ".join(fields))
        table = table.reindex(columns=fields + optional).copy()
        table["cell"] = table["cell"].str.strip()
        if table.empty or table.cell.isna().any() or table.cell.eq("").any() or table.cell.duplicated().any():
            raise ValueError("Provide one division event per named cell in each CSV.")
        table["stage_min"] = pd.to_numeric(table["stage_min"], errors="raise")
        if not np.isfinite(table.stage_min).all():
            raise ValueError("Division times must be finite numbers.")
        for field in optional:
            table[field] = pd.to_numeric(table[field], errors="raise")
            if not np.isfinite(table[field].dropna()).all():
                raise ValueError(f"{field} must contain finite numbers or blanks.")
        if (table.birth_min > table.stage_min).any() or (table.last_seen_min < table.stage_min).any() or (table.path_length_um < 0).any():
            raise ValueError("Check birth and last observation times and use nonnegative path lengths.")
        table["cell_cycle_min"] = table.stage_min - table.birth_min
        tables.append(table)
    comparison = tables[0].merge(tables[1], on="cell", how="outer", suffixes=("_control", "_variant"), indicator="record_presence")
    comparison["difference_min"] = comparison.stage_min_variant - comparison.stage_min_control
    comparison["first_observed_min"] = comparison[["stage_min_control", "stage_min_variant"]].min(axis=1)
    for field in [*optional, "cell_cycle_min"]:
        if comparison[[f"{field}_control", f"{field}_variant"]].notna().any().any():
            comparison[f"{field}_difference"] = comparison[f"{field}_variant"] - comparison[f"{field}_control"]
        else:
            comparison.drop(columns=[f"{field}_control", f"{field}_variant"], inplace=True)

    def same_daughters(row):
        names = [row[f"daughter_{letter}_{group}"] for group in ("control", "variant") for letter in ("a", "b")]
        if any(pd.isna(name) or not str(name).strip() for name in names):
            return pd.NA
        names = [str(name).strip() for name in names]
        return set(names[:2]) == set(names[2:])

    comparison["same_named_daughters"] = comparison.apply(same_daughters, axis=1).astype("boolean")
    parents = comparison[["parent_control", "parent_variant"]].astype("string").apply(lambda col: col.str.strip())
    comparison["same_named_parent"] = parents.parent_control.eq(parents.parent_variant).where(
        parents.notna().all(axis=1) & parents.ne("").all(axis=1)
    )
    return comparison.sort_values(["first_observed_min", "cell"])
