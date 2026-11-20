"""
Biohub Cell Lineage Tracker — interactive analysis application.

Run: streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

LIVE_APP_URL = "https://bioapp---cell-tracking-during-development-ezpjtah7icn2tfr8tie4.streamlit.app/"
REPO_URL = "https://github.com/Tobi-joshua/Biohub---Cell-Tracking-During-Development"

from app.ui import (  # noqa: E402
    init_session,
    render_dataset_sidebar,
    run_pipeline_on_session,
    sidebar_settings,
)
from biohub.config import MATCH_GATE_UM, PIPELINE_VERSION, SCALE  # noqa: E402
from biohub.curation import apply_curation  # noqa: E402
from biohub.data import load_geff_graph  # noqa: E402
from biohub.dataset_catalog import DatasetCatalog  # noqa: E402
from biohub.export import division_events, to_graph_json, to_lineage_csv  # noqa: E402
from biohub.lineage_explorer import (  # noqa: E402
    lineage_event_table,
    lineage_summary,
    node_relationships,
    node_role_ids,
    node_role_table,
    relationship_table,
)
from biohub.scientific import build_scientific_tables, export_scientific_bundle  # noqa: E402
from biohub.visualization import (  # noqa: E402
    build_overlay_gif,
    plot_frame_counts,
    plot_gt_overlay,
    plot_intensity_histogram,
    plot_lineage_graph,
    plot_lineage_timeline,
    plot_node_family_tree,
    plot_node_temporal_context,
    plot_selected_lineage_trajectory,
    plot_selected_lineage_tree,
    plot_slice_overlay,
    plot_temporal_montage,
    plot_track_overlay,
    plot_volume_slices,
)

st.set_page_config(
    page_title="Biohub Cell Lineage Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_CSS = """
<style>
/* Push main content below the Streamlit toolbar */
section.main > div.block-container {
    padding-top: 4.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

h1 { font-weight: 600; letter-spacing: -0.02em; }

/* Page navigation bar */
.biohub-nav-wrap {
    margin: 0 0 1.25rem 0;
    padding: 0.65rem 0 0.75rem 0;
    border-bottom: 1px solid rgba(49, 51, 63, 0.15);
    background: var(--background-color, #ffffff);
}

div[data-testid="stSegmentedControl"] {
    width: 100%;
}

div[data-testid="stSegmentedControl"] > div {
    flex-wrap: wrap;
    gap: 0.35rem;
}
</style>
"""

from app.kaggle_results import kaggle_tab

NAV_PAGES = ["Home", "Dataset", "Volume", "Pipeline", "Detection", "Lineage", "Exports", "Compare Lineages"]

st.markdown(APP_CSS, unsafe_allow_html=True)


init_session()


def home_tab() -> None:
    st.title("Biohub Cell Lineage Tracker")
    st.subheader('Follow cell division, movement and developmental change')
    st.markdown('Turn real time-lapse microscopy into **predicted cell tracks and parent–daughter trees**, then return to the original images to review the evidence.')
    st.image(str(Path(__file__).resolve().parent/'assets/lineage_workflow.svg'),width='stretch')
    st.caption('Illustrative example: observations are linked into a proposed track; a branch represents a proposed parent–daughter division. This diagram is not a microscopy image or a model result.')

    for column,title,description in zip(st.columns(3),
        ['Detect nuclei','Trace families','Compare recordings'],
        ['Locate visible nuclei and inspect their positions through time.',
         'Follow a selected observation to its parent, daughters and descendants. Review proposed divisions and uncertain track endings.',
         'Compare predicted cell counts, division proposals and track continuity, with reference checks where annotations are available.']):
        with column:
            with st.container(border=True):
                st.markdown('**'+title+'**')
                st.write(description)

    result=st.session_state.get('result')
    volume=st.session_state.get('volume_4d')
    st.markdown('### Your analysis')
    if result is not None:
        st.caption('Selected recording: '+result.dataset_name)
        c1,c2,c3=st.columns(3)
        c1.metric('Analyzed frames',len(result.frames))
        c2.metric('Predicted observations',f'{result.n_nodes:,}')
        c3.metric('Division proposals',f'{result.n_divisions:,}')
        st.caption('An observation is one predicted nucleus in one frame. Division proposals require review; these numbers are not accuracy scores.')
    elif volume is not None:
        st.info(f'Recording loaded: {len(volume)} frames. Run the pipeline to generate predictions.')
    else:
        st.info('Start in the sidebar: load one recording to explore its lineage, or choose Control and variant to compare two recordings.')

    def open_section(section):
        st.session_state.main_nav=section

    c1,c2,c3=st.columns(3)
    c1.button('Inspect recording',key='home_inspect',on_click=open_section,args=('Volume',),disabled=volume is None,width='stretch')
    c2.button('Explore parent–daughter trees',key='home_lineages',on_click=open_section,args=('Lineage',),disabled=result is None,width='stretch')
    c3.button('Compare recordings',key='home_compare',on_click=open_section,args=('Compare Lineages',),width='stretch')

    st.markdown('### Research goal: identify developmental differences')
    st.write('When and where do cell lineages in embryos with a known genetic variant first differ from control embryos? We aim to compare division timing, movement and lineage structure, and make each flagged difference inspectable in the microscopy frames.')
    st.caption('Available now: detection, proposed tracks, lineage review, descriptive comparisons and exports. Genetic-defect diagnosis is not validated. A genetic comparison requires source-labelled control/variant embryos, aligned developmental stages and repeated recordings; unlabelled Biohub samples are shown as Recording A/B.')


def data_tab() -> None:
    st.subheader("Dataset summary")
    catalog: DatasetCatalog | None = st.session_state.get("catalog")
    entry = st.session_state.get("selected_entry")
    vol4d = st.session_state.get("volume_4d")

    if catalog is not None:
        st.markdown(f"**Root:** `{catalog.root}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("Train volumes", catalog.n_train)
        c2.metric("Test volumes", catalog.n_test)
        c3.metric("Total", len(catalog.entries))
        rows = []
        for e in catalog.entries:
            rows.append(
                {
                    "split": e.split,
                    "name": e.name,
                    "shape": str(e.shape),
                    "dtype": str(e.dtype),
                    "geff": "yes" if e.has_geff else "no",
                    "est_nodes": e.estimated_nodes,
                    "gt_nodes": e.n_geff_nodes,
                    "gt_edges": e.n_geff_edges,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if entry is not None:
        st.markdown("**Selected volume**")
        st.json(
            {
                "name": entry.name,
                "split": entry.split,
                "zarr": str(entry.zarr_path),
                "geff": str(entry.geff_path) if entry.geff_path else None,
                "shape": entry.shape,
                "dtype": str(entry.dtype),
                "estimated_nodes": entry.estimated_nodes,
            }
        )

    if vol4d is None:
        st.info("Scan a local dataset and load a volume from the sidebar.")
        return

    t, z, y, x = vol4d.shape
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Loaded frames (T)", t)
    c2.metric("Z slices", z)
    c3.metric("Y size", y)
    c4.metric("X size", x)
    if entry is not None and t < entry.shape[0]:
        st.caption(f"Preview shows {t} of {entry.shape[0]} timepoints. Use **Full sequence** to process all frames.")
    st.caption(f"dtype: {vol4d.dtype} | min={vol4d.min()} max={vol4d.max()} mean={vol4d.mean():.1f}")
    st.dataframe(
        pd.DataFrame(
            {
                "axis": ["T", "Z", "Y", "X"],
                "loaded_size": [t, z, y, x],
                "full_size": [entry.shape[0] if entry else t, z, y, x],
                "spacing_um": [None, *([None]*3 if st.session_state.get('analysis_workspace') == 'Control and variant' else st.session_state.cfg.voxel_size_zyx_um)],
            }
        ),
        width="stretch",
        hide_index=True,
    )


def volume_tab() -> None:
    st.subheader("Volume viewer")
    vol4d = st.session_state.volume_4d
    entry = st.session_state.get("selected_entry")
    if vol4d is None:
        st.info("Load a volume from the sidebar.")
        return

    t_max, z_max = vol4d.shape[0] - 1, vol4d.shape[1] - 1
    c1, c2 = st.columns(2)
    t_idx = c1.slider("Time index", 0, t_max, 0, key="vol_t")
    z_idx = c2.slider("Z slice", 0, z_max, z_max // 2, key="vol_z")
    vol = vol4d[t_idx]

    if entry is not None and entry.has_geff:
        gt_nodes, _ = load_geff_graph(entry.geff_path)
        if gt_nodes is not None:
            gt_t = gt_nodes[gt_nodes["t"] == t_idx][["z", "y", "x"]].to_numpy()
            fig_gt = plot_gt_overlay(
                vol,
                gt_t,
                title=f"Ground-truth labels (sparse) | {entry.name} t={t_idx}",
            )
            st.pyplot(fig_gt, clear_figure=True)

    fig_hist = plot_intensity_histogram(vol, title=f"Intensity histogram, t={t_idx}")
    st.pyplot(fig_hist, clear_figure=True)
    fig = plot_volume_slices(vol, title=f"{st.session_state.dataset_name} | t={t_idx}", z_idx=z_idx)
    st.pyplot(fig, clear_figure=True)


def _division_highlight_coords(result, t_idx: int) -> np.ndarray | None:
    """Return division-parent coordinates for frame t_idx, or None."""
    if not result.n_divisions or result.nodes.empty:
        return None
    div_parents = set(division_events(result)["source_id"].astype(int))
    sub = result.nodes[result.nodes["node_id"].isin(div_parents) & (result.nodes["t"] == t_idx)]
    if sub.empty:
        return None
    return sub[["z", "y", "x"]].to_numpy()


def pipeline_tab(cfg) -> None:
    st.subheader("Analysis pipeline")
    if st.session_state.volume_4d is None and st.session_state.get("selected_entry") is None:
        st.info("Load a volume first.")
        return
    paired_workspace = st.session_state.get('analysis_workspace') == 'Control and variant'
    if paired_workspace:
        st.caption('Use Load model and analyze both in the sidebar to keep the two recordings on the same model and settings.')
    if not paired_workspace and st.button("Run detection and tracking", type="primary"):
        st.session_state.result = None
        st.session_state.curation_audit = None
        with st.spinner("Processing frames…"):
            try:
                result = run_pipeline_on_session(
                    cfg,
                    int(st.session_state.get("preview_frames", cfg.preview_max_frames)),
                )
                st.session_state.result = result
                st.session_state.curation_audit = None
                st.success(
                    f"Done: {result.n_nodes} nodes, {result.n_edges} edges, "
                    f"{result.n_divisions} divisions."
                )
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
    result = st.session_state.result
    if result is None:
        return
    st.markdown("**Summary**")
    st.json(result.stats)
    counts = result.nodes.groupby('t').size().reindex(range(len(result.frames)), fill_value=0).astype(int).tolist()
    fig_counts = plot_frame_counts(counts, title="Detections per frame")
    st.pyplot(fig_counts, clear_figure=True)
    if counts and len(set(counts)) == 1:
        st.caption(f'The model retained {counts[0]} predicted nuclei in each of these {len(counts)} frames, so the line is flat. The graph shows counts, not cell movement or detection accuracy.')


def detection_tab() -> None:
    st.subheader("Detection overlay")
    result = st.session_state.result
    vol4d = st.session_state.volume_4d
    if result is None or vol4d is None:
        st.info("Run the pipeline first.")
        return
    if result.stats.get("source") == "Kaggle organizer pretrained model, version 3":
        entry = st.session_state.get("selected_entry")
        if entry is None or entry.split != "test" or entry.name != result.dataset_name:
            st.info("Load the matching test volume to display these Kaggle predictions.")
            return
        st.caption(result.stats["source"])
    elif result.stats.get("source") == "Kaggle organizer pretrained model, custom upload":
        st.caption(result.stats["source"])

    t_max = min(len(result.frames), len(vol4d)) - 1
    calibrated = not bool(result.stats.get("spacing_note"))
    if not calibrated:
        st.caption(result.stats.get("source", result.dataset_name))
        st.caption(result.stats["spacing_note"] + " Learned detections; classical linking.")
    z_max = vol4d.shape[1] - 1
    review_mode = st.radio("Display", ["Interactive volume review", "Original slice overlays"], horizontal=True)
    if review_mode == "Interactive volume review":
        from app.result_review import render_detection_review
        render_detection_review(result, vol4d)
        return
    c1, c2 = st.columns(2)
    t_idx = c1.slider("Frame", 0, t_max, 0, key="det_t") if t_max else 0
    z_idx = c2.slider("Z slice", 0, z_max, z_max // 2, key="det_z")
    frame = result.frames[t_idx]
    highlight = _division_highlight_coords(result, t_idx)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = plot_slice_overlay(
            vol4d[t_idx],
            z_idx,
            frame.coords,
            title=f"Detections at t={t_idx}",
            highlight_divisions=highlight,
            show_scale_bar=calibrated,
        )
        st.pyplot(fig, clear_figure=True)
        st.caption(f"{len(frame.coords)} detections in this frame.")

    with col_b:
        if t_idx < t_max:
            next_frame = result.frames[t_idx + 1]
            curr_ids = set(frame.node_ids)
            next_ids = set(next_frame.node_ids)
            links = [
                (int(row.source_id), int(row.target_id))
                for row in result.edges.itertuples(index=False)
                if int(row.source_id) in curr_ids and int(row.target_id) in next_ids
            ]
            fig_links = plot_track_overlay(
                vol4d[t_idx],
                frame.coords,
                next_frame.coords,
                links,
                z_idx,
                vol_b=vol4d[t_idx + 1],
                title=f"Links t={t_idx} → t={t_idx + 1}",
                frame_a_ids=frame.node_ids,
                frame_b_ids=next_frame.node_ids,
                show_scale_bar=calibrated,
            )
            st.pyplot(fig_links, clear_figure=True)
        else:
            st.info("Select an earlier frame to view inter-frame links.")

    st.markdown("**Temporal montage**")
    fig_montage = plot_temporal_montage(
        vol4d,
        result.frames[:t_max + 1],
        z_idx,
        title=f"Detection montage (z={z_idx})",
    )
    st.pyplot(fig_montage, clear_figure=True)

    st.markdown("**Time-lapse animation**")
    fps = st.slider("GIF frame rate (fps)", 2, 12, 4, key="gif_fps")
    if st.button("Generate overlay animation", key="gen_gif"):
        with st.spinner("Rendering GIF…"):
            gif_bytes = build_overlay_gif(vol4d, result.frames[:t_max + 1], z_idx, fps=fps)
            st.session_state.overlay_gif = gif_bytes
    if st.session_state.get("overlay_gif"):
        st.image(st.session_state.overlay_gif, caption=f"Detection overlay, z={z_idx}")
        st.download_button(
            "Download animation (GIF)",
            data=st.session_state.overlay_gif,
            file_name=f"{st.session_state.dataset_name}_overlay_z{z_idx}.gif",
            mime="image/gif",
        )


def lineage_tab(cfg) -> None:
    st.subheader("Lineage explorer")
    result = st.session_state.result
    if result is None:
        st.info("Run the pipeline first.")
        return

    if result.n_divisions == 0:
        st.info('No parent with two predicted daughter links was found in this analyzed clip. The clip may fall between biological divisions, or the detector/tracker may have missed a division. This does not mean the embryo never divides.')

    tables = build_scientific_tables(result, cfg)
    observations = tables["observations"].copy()
    if observations.empty:
        st.info("No observations are available in this result.")
        return

    # One focus node drives every study view so graphs, tables, and image evidence stay synchronized.
    node_options = observations["node_id"].astype(int).sort_values().tolist()
    if st.session_state.get("lineage_focus_node") not in node_options:
        st.session_state["lineage_focus_node"] = node_options[0]
    selected_node = st.selectbox(
        "Focus observation",
        node_options,
        key="lineage_focus_node",
        format_func=lambda node: f"Node {node}",
        help="Search or select one observation. Every panel below follows this node and its lineage.",
    )

    selected = observations.loc[
        observations["node_id"].astype(int).eq(int(selected_node))
    ].iloc[0]
    rel = node_relationships(result, int(selected_node))
    lineage_id = int(selected["lineage_id"])
    track_id = int(selected["track_id"])
    summary = lineage_summary(result, lineage_id)

    event_labels = {
        "division_parent": "Division parent",
        "daughter": "Daughter after division",
        "terminal": "Terminal observation",
        "lineage_root": "Lineage root",
        "isolated": "Isolated observation",
        "continuation": "Track continuation",
    }
    event_label = event_labels.get(str(rel["event_type"]), str(rel["event_type"]).replace("_", " ").title())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lineage", f"L{lineage_id}")
    c2.metric("Track", f"T{track_id}")
    c3.metric("Generation", int(selected.get("generation", 0)))
    c4.metric("Frame", int(selected["t"]))
    c5.metric("Event", event_label)
    st.caption(
        "Study view: one focused lineage at a time. Green = selected node, red = division parent, "
        "orange = ancestors, purple = descendants, gray = terminal observation."
    )

    # Scientist-facing role finder: turn the legend into searchable node lists.
    with st.expander("Node Role Finder", expanded=True):
        role_scope_label = st.radio(
            "Scope",
            ["Selected lineage", "Entire dataset"],
            horizontal=True,
            key="lineage_role_scope",
            help=(
                "Ancestors and descendants are always relative to the focused node. "
                "Division parents and terminal nodes follow the selected scope."
            ),
        )
        role_scope = "selected_lineage" if role_scope_label == "Selected lineage" else "entire_dataset"
        role_ids = node_role_ids(result, int(selected_node), scope=role_scope)
        role_table = node_role_table(result, int(selected_node), scope=role_scope)

        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        rc1.metric("Selected", len(role_ids["selected"]))
        rc2.metric("Division parents", len(role_ids["division_parent"]))
        rc3.metric("Ancestors", len(role_ids["ancestor"]))
        rc4.metric("Descendants", len(role_ids["descendant"]))
        rc5.metric("Terminal", len(role_ids["terminal"]))

        role_labels = {
            "All roles": None,
            "Selected": "selected",
            "Division parents": "division_parent",
            "Ancestors of selected": "ancestor",
            "Descendants of selected": "descendant",
            "Terminal nodes": "terminal",
        }
        finder_col, jump_col = st.columns([1, 2])
        role_label = finder_col.selectbox(
            "Show role",
            list(role_labels),
            key="lineage_role_filter",
        )
        role_key = role_labels[role_label]
        if role_key is None:
            visible_ids = role_table["node_id"].astype(int).tolist() if not role_table.empty else []
            visible_table = role_table
        else:
            visible_ids = [int(x) for x in role_ids[role_key]]
            visible_table = (
                role_table[role_table["node_id"].astype(int).isin(visible_ids)].copy()
                if not role_table.empty else role_table
            )

        def _jump_to_role_node(widget_key: str) -> None:
            value = st.session_state.get(widget_key)
            if value is not None:
                st.session_state["lineage_focus_node"] = int(value)

        if visible_ids:
            jump_key = f"lineage_role_jump_{role_scope}_{role_key or 'all'}"
            default_index = visible_ids.index(int(selected_node)) if int(selected_node) in visible_ids else 0
            jump_col.selectbox(
                "Jump to node",
                visible_ids,
                index=default_index,
                key=jump_key,
                format_func=lambda nid: f"Node {nid}",
                on_change=_jump_to_role_node,
                args=(jump_key,),
                help="Selecting a node makes it the new Focus observation and refreshes every lineage panel.",
            )
        else:
            jump_col.info("No nodes match this role in the current scope.")

        if visible_table.empty:
            st.info("No role nodes are available for the current filter.")
        else:
            st.dataframe(
                visible_table,
                width="stretch",
                hide_index=True,
                column_order=[
                    "node_id", "roles", "track_id", "lineage_id", "generation",
                    "frame", "parent", "children", "qc_status",
                ],
            )

        # Fast stepping through biologically important event nodes.
        def _focus_node(target: int) -> None:
            st.session_state["lineage_focus_node"] = int(target)

        def _neighbor(ids: list[int], current: int, direction: int) -> int | None:
            if not ids:
                return None
            ids = list(ids)
            if current in ids:
                idx = ids.index(current) + direction
                if 0 <= idx < len(ids):
                    return ids[idx]
            if direction > 0:
                return next((nid for nid in ids if nid > current), ids[0])
            return next((nid for nid in reversed(ids) if nid < current), ids[-1])

        divisions = [int(x) for x in role_ids["division_parent"]]
        terminals = [int(x) for x in role_ids["terminal"]]
        prev_div = _neighbor(divisions, int(selected_node), -1)
        next_div = _neighbor(divisions, int(selected_node), 1)
        prev_term = _neighbor(terminals, int(selected_node), -1)
        next_term = _neighbor(terminals, int(selected_node), 1)

        q1, q2, q3, q4 = st.columns(4)
        q1.button(
            "Previous division",
            disabled=prev_div is None,
            on_click=_focus_node if prev_div is not None else None,
            args=(prev_div,) if prev_div is not None else (),
            key="prev_division_node",
        )
        q2.button(
            "Next division",
            disabled=next_div is None,
            on_click=_focus_node if next_div is not None else None,
            args=(next_div,) if next_div is not None else (),
            key="next_division_node",
        )
        q3.button(
            "Previous terminal",
            disabled=prev_term is None,
            on_click=_focus_node if prev_term is not None else None,
            args=(prev_term,) if prev_term is not None else (),
            key="prev_terminal_node",
        )
        q4.button(
            "Next terminal",
            disabled=next_term is None,
            on_click=_focus_node if next_term is not None else None,
            args=(next_term,) if next_term is not None else (),
            key="next_terminal_node",
        )

    selected_lineage_tab, explorer_tab, signals_tab, global_tab = st.tabs(
        ["Selected lineage", "Cell Explorer", "Biological signals", "Global overview"]
    )

    with selected_lineage_tab:
        st.markdown("### Parent → daughter lineage tree")
        st.caption(
            "This replaces the force-directed overview as the primary lineage view. "
            "Horizontal position is the actual frame, so branch points can be read directly."
        )
        fig_tree = plot_selected_lineage_tree(
            result, lineage_id=lineage_id, selected_node_id=int(selected_node)
        )
        st.pyplot(fig_tree, clear_figure=True, width="stretch")

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Observations", summary["observations"])
        sc2.metric("Tracks", summary["tracks"])
        sc3.metric("Divisions", summary["divisions"])
        sc4.metric("Terminal nodes", summary["terminals"])
        sc5.metric("QC flagged", summary["qc_flagged"])
        st.caption(
            f"Lineage L{lineage_id}: frame {summary['start_t']} → {summary['end_t']} | "
            f"root components: {summary['roots']}"
        )

        st.markdown("### Track trajectories within this lineage")
        fig_trajectory = plot_selected_lineage_trajectory(
            result, lineage_id=lineage_id, selected_track_id=track_id
        )
        st.pyplot(fig_trajectory, clear_figure=True, width="stretch")

        st.markdown("### Frame-by-frame lineage table")
        event_table = lineage_event_table(result, lineage_id)
        st.dataframe(event_table, width="stretch", hide_index=True)

        lineage_observations = observations[
            observations["lineage_id"].astype(int).eq(lineage_id)
        ].sort_values(["t", "track_id", "node_id"])
        d1, d2 = st.columns(2)
        d1.download_button(
            "Download selected lineage observations (CSV)",
            data=lineage_observations.to_csv(index=False).encode(),
            file_name=f"{result.dataset_name}_lineage_{lineage_id}_observations.csv",
            mime="text/csv",
            width="stretch",
        )
        d2.download_button(
            "Download selected lineage event table (CSV)",
            data=event_table.to_csv(index=False).encode(),
            file_name=f"{result.dataset_name}_lineage_{lineage_id}_events.csv",
            mime="text/csv",
            width="stretch",
        )

    with explorer_tab:
        st.markdown(f"### Node {int(selected_node)} — biological structure")
        ancestors = list(rel["ancestors"])
        descendants = list(rel["descendants"])
        children = list(rel["children"])
        siblings = list(rel["siblings"])

        breadcrumb = " → ".join(map(str, ancestors + [int(selected_node)]))
        st.markdown(f"**Ancestry path:** `{breadcrumb}`")
        if children:
            st.markdown("**Direct children:** " + ", ".join(f"`{node}`" for node in children))
        else:
            st.markdown("**Direct children:** none")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Parent", rel["parent_node_id"] if rel["parent_node_id"] is not None else "Root")
        r2.metric("Children", len(children))
        r3.metric("Siblings", len(siblings))
        r4.metric("Descendants", len(descendants))

        st.markdown("#### Local family tree")
        family_fig = plot_node_family_tree(
            result, int(selected_node), ancestor_depth=3, descendant_depth=3
        )
        st.pyplot(family_fig, clear_figure=True, width="stretch")

        st.markdown("#### Connectivity")
        st.dataframe(
            relationship_table(result, int(selected_node)),
            width="stretch",
            hide_index=True,
        )
        if len(descendants) > 30:
            st.caption(
                f"This node has {len(descendants)} descendants; the table reports all of them while the local tree is depth-limited for readability."
            )

        st.markdown("#### Event interpretation")
        if rel["event_type"] == "division_parent":
            st.success(
                f"Node {selected_node} is a predicted division parent with daughters: "
                + ", ".join(map(str, children))
                + ". Treat this as a predicted event until validated against annotation or manual review."
            )
        elif rel["event_type"] == "daughter":
            st.info(
                f"Node {selected_node} is a daughter observation produced by division of parent node "
                f"{rel['parent_node_id']}."
            )
        elif rel["event_type"] == "terminal":
            st.warning(
                f"Node {selected_node} has no outgoing child. This is a terminal observation, not automatically a cell death; "
                "it may reflect a real termination, the imaging boundary, or a tracking failure."
            )
        elif rel["event_type"] == "lineage_root":
            st.info(f"Node {selected_node} is the first observed node of this lineage component.")
        elif rel["event_type"] == "isolated":
            st.warning(f"Node {selected_node} has no parent or child link in the current graph.")
        else:
            st.info(f"Node {selected_node} is a continuation observation within track T{track_id}.")

        st.markdown("#### Observation measurements")
        parent_edges = (
            result.edges.loc[result.edges["target_id"].astype(int).eq(int(selected_node))]
            if not result.edges.empty else pd.DataFrame()
        )
        confidence_column = next(
            (name for name in ("link_confidence", "confidence") if name in parent_edges.columns), None
        )
        link_confidence = (
            float(parent_edges.iloc[0][confidence_column])
            if confidence_column and not parent_edges.empty and pd.notna(parent_edges.iloc[0][confidence_column])
            else "unavailable (not calibrated)"
        )
        measurement_fields = [
            "node_id", "track_id", "lineage_id", "generation", "t", "time_s",
            "z", "y", "x", "z_um", "y_um", "x_um", "raw_detection_score",
            "intensity", "qc_status", "filter_reason",
        ]
        details = {field: selected[field] for field in measurement_fields if field in selected.index}
        details.update({
            "event_type": event_label,
            "parent_node_id": rel["parent_node_id"],
            "children": ", ".join(map(str, children)) if children else "—",
            "previous_track_node": rel["previous_track_node"],
            "next_track_node": rel["next_track_node"],
            "link_confidence": link_confidence,
        })
        st.dataframe(pd.DataFrame([details]), width="stretch", hide_index=True)

        vol4d = st.session_state.get("volume_4d")
        if vol4d is not None:
            st.markdown("#### Local image evidence")
            context_fig = plot_node_temporal_context(vol4d, result, int(selected_node))
            st.pyplot(context_fig, clear_figure=True, width="stretch")
        else:
            st.caption("Load the image volume to inspect parent/current/child image evidence.")

        st.markdown("#### Manual review")
        st.caption("Every accepted correction is validated against graph invariants and recorded in the audit table.")
        action_labels = {
            "Approve node": "approve_node",
            "Reject node": "reject_node",
            "Approve edge": "approve_edge",
            "Reject edge": "reject_edge",
            "Relink target": "relink",
        }
        default_source = int(rel["parent_node_id"]) if rel["parent_node_id"] is not None else int(selected_node)
        default_target = int(selected_node)
        with st.form("curation_form", clear_on_submit=False):
            label = st.selectbox("Action", list(action_labels))
            action = action_labels[label]
            node_action = action in {"approve_node", "reject_node"}
            c1, c2, c3 = st.columns(3)
            node_id = c1.number_input(
                "Node ID", min_value=0, value=int(selected_node), step=1,
                disabled=not node_action, key=f"curate_node_{selected_node}",
            )
            source_id = c2.number_input(
                "Source ID", min_value=0, value=default_source, step=1,
                disabled=node_action, key=f"curate_source_{selected_node}",
            )
            target_id = c3.number_input(
                "Target ID", min_value=0, value=default_target, step=1,
                disabled=node_action, key=f"curate_target_{selected_node}",
            )
            reviewer = st.text_input("Reviewer", value="scientist")
            reason = st.text_input("Reason")
            submitted = st.form_submit_button("Apply reviewed correction")
        if submitted:
            payload = {"action": action, "reviewer": reviewer, "reason": reason}
            if node_action:
                payload["node_id"] = int(node_id)
            else:
                payload.update({"source_id": int(source_id), "target_id": int(target_id)})
            try:
                curated, audit = apply_curation(result, [payload])
                previous = st.session_state.get("curation_audit")
                st.session_state.curation_audit = (
                    audit if previous is None else pd.concat([previous, audit], ignore_index=True)
                )
                st.session_state.result = curated
                if int(selected_node) not in set(curated.nodes["node_id"].astype(int)):
                    st.session_state.pop("lineage_focus_node", None)
                st.success("Correction applied; graph invariants passed.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        audit = st.session_state.get("curation_audit")
        if audit is not None and not audit.empty:
            st.markdown("#### Review audit")
            st.dataframe(audit, width="stretch", hide_index=True)

    with signals_tab:
        st.markdown("### Biological signals for the focused lineage")
        st.caption(
            "These tables are measurements or candidates. Terminal tracks are not automatically apoptosis, "
            "and predicted divisions are not automatically confirmed biology."
        )
        division_table = tables["divisions"]
        if not division_table.empty and "lineage_id" in division_table.columns:
            division_table = division_table[division_table["lineage_id"].astype(int).eq(lineage_id)]
        st.markdown("#### Division events")
        if division_table.empty:
            st.info("No predicted divisions in this lineage.")
        else:
            st.dataframe(division_table, width="stretch", hide_index=True)

        phenotype_table = tables.get("division_phenotypes", pd.DataFrame())
        if not phenotype_table.empty and "lineage_id" in phenotype_table.columns:
            phenotype_table = phenotype_table[phenotype_table["lineage_id"].astype(int).eq(lineage_id)]
        if not phenotype_table.empty:
            st.markdown("#### Division phenotypes")
            st.dataframe(phenotype_table, width="stretch", hide_index=True)

        termination_table = tables.get("track_terminations", pd.DataFrame())
        if not termination_table.empty and "lineage_id" in termination_table.columns:
            termination_table = termination_table[termination_table["lineage_id"].astype(int).eq(lineage_id)]
        st.markdown("#### Track terminations")
        if termination_table.empty:
            st.info("No termination records available for this lineage.")
        else:
            st.dataframe(termination_table, width="stretch", hide_index=True)

        qc_table = tables.get("qc_flags", pd.DataFrame())
        if not qc_table.empty and "lineage_id" in qc_table.columns:
            qc_table = qc_table[qc_table["lineage_id"].astype(int).eq(lineage_id)]
        st.markdown("#### QC flags")
        if qc_table.empty:
            st.success("No non-pass QC flags in the focused lineage.")
        else:
            st.dataframe(qc_table, width="stretch", hide_index=True)

    with global_tab:
        st.markdown("### Global diagnostic overview")
        st.caption(
            "These plots summarize the whole result and can be crowded. Use the Selected lineage and Cell Explorer tabs for study decisions."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            fig = plot_lineage_graph(result)
            st.pyplot(fig, clear_figure=True, width="stretch")
        with col_b:
            fig_tl = plot_lineage_timeline(result)
            st.pyplot(fig_tl, clear_figure=True, width="stretch")
        global_divisions = division_events(result)
        st.markdown("#### All division events")
        if global_divisions.empty:
            st.write("No division events detected.")
        else:
            st.dataframe(global_divisions, width="stretch", hide_index=True)


def exports_tab(cfg) -> None:
    st.subheader("Exports")
    result = st.session_state.result
    if result is None:
        st.info("Run the pipeline first.")
        return
    csv_df = to_lineage_csv(result)
    graph = to_graph_json(result)
    from uuid import uuid4
    if 'run_id' not in result.stats:
        result.stats['run_id'] = uuid4().hex
    import hashlib
    revision=hashlib.sha256((result.nodes.to_csv(index=False)+result.edges.to_csv(index=False)).encode()).hexdigest()[:12]
    out_dir = Path(cfg.output_dir) / result.stats['run_id'] / revision
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{result.dataset_name}_lineage.csv"
    json_path = out_dir / f"{result.dataset_name}_graph.json"
    audit_path = out_dir / f"{result.dataset_name}_curation_audit.csv"
    csv_df.to_csv(csv_path)
    json_path.write_text(json.dumps(graph, indent=2))
    audit = st.session_state.get("curation_audit")
    if audit is not None and not audit.empty:
        audit.to_csv(audit_path, index=False)
    st.download_button(
        "Download lineage CSV",
        data=csv_df.to_csv().encode(),
        file_name=csv_path.name,
        mime="text/csv",
    )
    st.download_button(
        "Download graph JSON",
        data=json.dumps(graph, indent=2).encode(),
        file_name=json_path.name,
        mime="application/json",
    )
    if audit is not None and not audit.empty:
        st.download_button(
            "Download curation audit CSV",
            data=audit.to_csv(index=False).encode(),
            file_name=audit_path.name,
            mime="text/csv",
        )
    if result.stats.get('time_interval_verified') is False:
        st.info('Scientific time/speed exports are unavailable because this recording’s frame interval is unverified. The basic graph and frame-index CSV above remain available.')
        return
    bundle_dir = out_dir / f"{result.dataset_name}_scientific"
    bundle_paths = export_scientific_bundle(result, cfg, bundle_dir)
    st.markdown("**Scientific export bundle**")
    for name, path in bundle_paths.items():
        mime = "application/json" if path.suffix == ".json" else "text/csv"
        st.download_button(
            f"Download {path.name}", data=path.read_bytes(), file_name=path.name,
            mime=mime, key=f"scientific_{name}",
        )
    st.caption(f"Saved lineage files to `{out_dir}` and scientific bundle to `{bundle_dir}`.")


# Sidebar controls (dataset + pipeline settings)
with st.sidebar:
    from app.paired_movies import render_pair_sidebar, restore_single_recording
    workspace = st.radio('Analysis workspace', ['Single recording', 'Control and variant'], key='analysis_workspace')
    if workspace == 'Control and variant':
        render_pair_sidebar()
        cfg = st.session_state.cfg
    else:
        restore_single_recording(st.session_state)
        render_dataset_sidebar()
        cfg, _mode, preview_frames = sidebar_settings(st.session_state.cfg)
        st.session_state.cfg = cfg
        st.session_state.preview_frames = preview_frames
        from biohub.manifest import config_sha256
        requested_hash = st.session_state.get('result_requested_config_hash')
        if st.session_state.get('result') is not None and requested_hash and requested_hash != config_sha256(cfg):
            from app.ui import invalidate_analysis
            invalidate_analysis()
            st.info('Analysis settings changed. Run detection and tracking again for this configuration.')

if workspace == 'Control and variant' and st.session_state.get('pair_active_index') is not None:
    st.caption('Viewing: ' + st.session_state.real_pair_output.get('labels',['Control','Variant'])[st.session_state.pair_active_index] + ' · ' + st.session_state.real_pair_output['filenames'][st.session_state.pair_active_index])

# Page navigation — segmented control stays in normal document flow (no sticky overlap)
st.markdown('<div class="biohub-nav-wrap">', unsafe_allow_html=True)
page = st.segmented_control(
    "Section",
    NAV_PAGES,
    default=NAV_PAGES[0],
    label_visibility="collapsed",
    key="main_nav",
)
st.markdown("</div>", unsafe_allow_html=True)

if page == "Home":
    home_tab()
elif page == "Dataset":
    data_tab()
elif page == "Volume":
    volume_tab()
elif page == "Pipeline":
    pipeline_tab(cfg)
elif page == "Detection":
    detection_tab()
elif page == "Lineage":
    lineage_tab(cfg)
elif page == "Exports":
    exports_tab(cfg)
elif page == "Compare Lineages":
    kaggle_tab()
