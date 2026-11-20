"""Visualization helpers for analysis and publication figures."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from biohub.analysis import AnalysisResult
from biohub.config import SCALE

PLOT_STYLE = {
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.2,
}

COLORS = {
    "detection": "#1B9E77",
    "division": "#D95F02",
    "link": "#7570B3",
    "gt": "#66C2A5",
    "pred": "#FC8D62",
    "accent": "#3366AA",
}


def apply_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def mip(vol: np.ndarray, axis: int = 0) -> np.ndarray:
    return vol.max(axis=axis)


def plot_intensity_histogram(vol: np.ndarray, title: str = "Intensity distribution") -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 3))
    sample = vol.ravel()
    if sample.size > 500_000:
        rng = np.random.default_rng(0)
        sample = rng.choice(sample, size=500_000, replace=False)
    ax.hist(sample, bins=80, color="#3366AA", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("intensity")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _add_scale_bar(ax: plt.Axes, pixel_size_um: float, length_um: float = 20.0) -> None:
    """Draw a horizontal scale bar in the lower-left corner."""
    length_px = length_um / pixel_size_um
    y0, x0 = ax.get_ylim()[0] + 4, ax.get_xlim()[0] + 4
    ax.plot([x0, x0 + length_px], [y0, y0], color="white", linewidth=2.5, solid_capstyle="butt")
    ax.plot([x0, x0 + length_px], [y0, y0], color="black", linewidth=1.2, solid_capstyle="butt")
    ax.text(
        x0 + length_px / 2,
        y0 + 6,
        f"{length_um:.0f} µm",
        ha="center",
        va="bottom",
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, edgecolor="none"),
    )


def plot_slice_overlay(
    vol: np.ndarray,
    z_idx: int,
    centroids: Optional[np.ndarray] = None,
    title: str = "",
    highlight_divisions: Optional[np.ndarray] = None,
    show_scale_bar: bool = True,
) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    vmin, vmax = np.percentile(vol[z_idx], (1, 99.5))
    ax.imshow(vol[z_idx], cmap="gray", aspect="equal", vmin=vmin, vmax=vmax)
    if centroids is not None and len(centroids):
        zc, yc, xc = centroids[:, 0], centroids[:, 1], centroids[:, 2]
        on = np.abs(zc - z_idx) <= 1
        ax.scatter(
            xc[on],
            yc[on],
            s=28,
            c=COLORS["detection"],
            linewidths=0.5,
            edgecolors="white",
            alpha=0.92,
            label="detections",
        )
        if highlight_divisions is not None and len(highlight_divisions):
            hd = highlight_divisions
            ond = np.abs(hd[:, 0] - z_idx) <= 1
            ax.scatter(
                hd[ond, 2],
                hd[ond, 1],
                s=70,
                facecolors="none",
                edgecolors=COLORS["division"],
                linewidths=1.6,
                label="divisions",
            )
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(title or f"XY slice, z = {z_idx}")
    if centroids is not None and len(centroids):
        ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#cccccc")
    if show_scale_bar:
        _add_scale_bar(ax, SCALE[2])
    fig.tight_layout()
    return fig


def plot_volume_slices(
    vol: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    title: str = "",
    z_idx: Optional[int] = None,
) -> plt.Figure:
    apply_style()
    if z_idx is None:
        z_idx = vol.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    axes[0].imshow(vol[z_idx], cmap="gray", aspect="equal")
    axes[0].set_title(f"XY @ z={z_idx}")
    axes[1].imshow(mip(vol, axis=2), cmap="gray", aspect="auto")
    axes[1].set_title("ZY projection")
    axes[2].imshow(mip(vol, axis=1), cmap="gray", aspect="auto")
    axes[2].set_title("ZX projection")
    if centroids is not None and len(centroids):
        zc, yc, xc = centroids[:, 0], centroids[:, 1], centroids[:, 2]
        on_slice = np.abs(zc - z_idx) <= 1
        axes[0].scatter(xc[on_slice], yc[on_slice], s=12, c="#2E8B57", linewidths=0.5, edgecolors="white")
    if title:
        fig.suptitle(title, y=1.02)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_frame_counts(counts: Sequence[int], title: str = "Detections per frame") -> plt.Figure:
    from matplotlib.ticker import MaxNLocator
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(len(counts)), counts, color="#3366AA", linewidth=1.5, marker="o", markersize=5)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Predicted nuclei")
    ax.set_ylim(0, max(3, int(np.ceil(max(counts, default=0)*1.2))))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if len(counts) == 1:
        ax.set_xlim(-0.5, 0.5)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_gt_overlay(
    vol: np.ndarray,
    gt_xyz: np.ndarray,
    pred_xyz: Optional[np.ndarray] = None,
    t: int = 0,
    title: str = "",
) -> plt.Figure:
    """Overlay ground-truth and optional predictions on a mid-Z slice."""
    apply_style()
    z_mid = int(np.median(gt_xyz[:, 0])) if len(gt_xyz) else vol.shape[0] // 2
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(vol[z_mid], cmap="gray")
    if len(gt_xyz):
        on = np.abs(gt_xyz[:, 0] - z_mid) <= 1
        ax.scatter(gt_xyz[on, 2], gt_xyz[on, 1], s=20, c="cyan", label="GT", alpha=0.8)
    if pred_xyz is not None and len(pred_xyz):
        on = np.abs(pred_xyz[:, 0] - z_mid) <= 1
        ax.scatter(pred_xyz[on, 2], pred_xyz[on, 1], s=14, c="lime", label="pred", alpha=0.7)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(title or f"t={t}, z={z_mid}")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_track_overlay(
    vol_a: np.ndarray,
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    links: Sequence[tuple],
    z_idx: int,
    vol_b: Optional[np.ndarray] = None,
    title: str = "Track overlay",
    frame_a_ids: Optional[Sequence[int]] = None,
    frame_b_ids: Optional[Sequence[int]] = None,
    show_scale_bar: bool = True,
) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    slice_a = vol_a[z_idx].astype(np.float32)
    slice_b = vol_b[z_idx].astype(np.float32) if vol_b is not None else slice_a
    base = 0.5 * slice_a + 0.5 * slice_b
    vmin, vmax = np.percentile(base, (1, 99.5))
    ax.imshow(base, cmap="gray", vmin=vmin, vmax=vmax)
    if len(frame_a) and len(frame_b):
        za, ya, xa = frame_a[:, 0], frame_a[:, 1], frame_a[:, 2]
        zb, yb, xb = frame_b[:, 0], frame_b[:, 1], frame_b[:, 2]
        on_a = np.abs(za - z_idx) <= 1
        on_b = np.abs(zb - z_idx) <= 1
        ax.scatter(xa[on_a], ya[on_a], s=22, c=COLORS["accent"], label="t", edgecolors="white", linewidths=0.4)
        ax.scatter(xb[on_b], yb[on_b], s=22, c=COLORS["detection"], label="t+1", edgecolors="white", linewidths=0.4)
        ids_a = list(frame_a_ids) if frame_a_ids is not None else list(range(1, len(frame_a) + 1))
        ids_b = list(frame_b_ids) if frame_b_ids is not None else list(range(1, len(frame_b) + 1))
        if len(ids_a) != len(frame_a) or len(ids_b) != len(frame_b):
            raise ValueError("Frame ID count must match coordinate count")
        id_to_xy_a = {int(ids_a[i]): (xa[i], ya[i]) for i in range(len(frame_a)) if on_a[i]}
        id_to_xy_b = {int(ids_b[i]): (xb[i], yb[i]) for i in range(len(frame_b)) if on_b[i]}
        for s, t in links:
            if s in id_to_xy_a and t in id_to_xy_b:
                x1, y1 = id_to_xy_a[s]
                x2, y2 = id_to_xy_b[t]
                ax.plot([x1, x2], [y1, y2], color=COLORS["link"], linewidth=1.0, alpha=0.85)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#cccccc")
    if show_scale_bar:
        _add_scale_bar(ax, SCALE[2])
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_lineage_graph(result: AnalysisResult, max_nodes: int = 120) -> plt.Figure:
    apply_style()
    import networkx as nx

    g = nx.DiGraph()
    if not result.nodes.empty:
        sort_cols=[c for c in ('lineage_id','track_id','t','node_id') if c in result.nodes]
        sub = result.nodes.sort_values(sort_cols).head(max_nodes)
        for row in sub.itertuples(index=False):
            g.add_node(int(row.node_id), t=int(row.t))
    if not result.edges.empty:
        for row in result.edges.itertuples(index=False):
            source,target=int(row.source_id),int(row.target_id)
            if source in g and target in g:
                g.add_edge(source,target)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No graph to display", ha="center", va="center")
        ax.axis("off")
        return fig

    pos = nx.spring_layout(g, seed=42, k=0.45)
    nx.draw_networkx_nodes(g, pos, node_size=45, node_color=COLORS["accent"], alpha=0.9, ax=ax)
    nx.draw_networkx_edges(g, pos, edge_color="#6B7280", arrows=True, arrowsize=10, width=0.8, ax=ax)
    ax.set_title(f"Lineage graph preview ({g.number_of_nodes()} of {len(result.nodes)} observations)")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_lineage_timeline(result: AnalysisResult, max_tracks: int = 80) -> plt.Figure:
    """Spaghetti plot of track centroids over time (common in tracking papers)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    if result.nodes.empty:
        ax.text(0.5, 0.5, "No tracks to display", ha="center", va="center")
        ax.axis("off")
        return fig

    identity_column = "track_id" if "track_id" in result.nodes.columns else "node_id"
    nodes = result.nodes.sort_values([identity_column, "t"])
    track_ids = nodes[identity_column].unique()[:max_tracks]
    cmap = plt.cm.tab20(np.linspace(0, 1, min(len(track_ids), 20)))

    for i, tid in enumerate(track_ids):
        sub = nodes[nodes[identity_column] == tid]
        color = cmap[i % len(cmap)]
        ax.plot(sub["t"], sub["y"], color=color, alpha=0.75, linewidth=1.0)
        ax.scatter(sub["t"], sub["y"], s=10, color=color, alpha=0.85, edgecolors="none")

    ax.set_xlabel("time frame")
    ax.set_ylabel("y position (pixels)")
    ax.set_title("Track trajectories over time")
    ax.grid(True, alpha=0.2, linestyle="--")
    fig.tight_layout()
    return fig



def _tree_time_positions(result: AnalysisResult, node_ids: set[int]) -> dict[int, tuple[float, float]]:
    """Lay out a lineage left-to-right by frame with branches separated vertically."""
    if not node_ids:
        return {}
    nodes = result.nodes.set_index("node_id", drop=False)
    children: dict[int, list[int]] = {nid: [] for nid in node_ids}
    parent: dict[int, int] = {}
    if not result.edges.empty:
        for row in result.edges.itertuples(index=False):
            source, target = int(row.source_id), int(row.target_id)
            if source in node_ids and target in node_ids:
                children.setdefault(source, []).append(target)
                parent[target] = source
    for source in children:
        children[source].sort(
            key=lambda nid: (int(nodes.loc[nid, "t"]) if nid in nodes.index else 10**9, nid)
        )

    roots = sorted(
        [nid for nid in node_ids if nid not in parent],
        key=lambda nid: (int(nodes.loc[nid, "t"]) if nid in nodes.index else 10**9, nid),
    )
    y_pos: dict[int, float] = {}
    leaf_counter = [0.0]
    visiting: set[int] = set()

    def assign_y(nid: int) -> float:
        if nid in y_pos:
            return y_pos[nid]
        if nid in visiting:
            # Graph invariants should prevent this; fall back to a stable row.
            value = leaf_counter[0]
            leaf_counter[0] += 1.0
            y_pos[nid] = value
            return value
        visiting.add(nid)
        direct = [child for child in children.get(nid, []) if child in node_ids]
        if not direct:
            value = leaf_counter[0]
            leaf_counter[0] += 1.0
        else:
            child_y = [assign_y(child) for child in direct]
            value = float(np.mean(child_y))
        visiting.discard(nid)
        y_pos[nid] = value
        return value

    for root in roots:
        assign_y(root)
        # Leave a visual gap between multiple disconnected roots.
        leaf_counter[0] += 0.75
    for nid in sorted(node_ids):
        assign_y(nid)

    return {
        nid: (
            float(nodes.loc[nid, "t"]) if nid in nodes.index else 0.0,
            -float(y_pos[nid]),
        )
        for nid in node_ids
    }


def _draw_lineage_tree(
    result: AnalysisResult,
    node_ids: set[int],
    *,
    selected_node_id: Optional[int] = None,
    title: str = "Lineage tree",
    label_limit: int = 90,
) -> plt.Figure:
    """Draw a readable lineage subgraph using time as the horizontal axis."""
    apply_style()
    import networkx as nx
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(11, 5.5))
    if not node_ids:
        ax.text(0.5, 0.5, "No lineage nodes to display", ha="center", va="center")
        ax.axis("off")
        return fig

    nodes = result.nodes.set_index("node_id", drop=False)
    g = nx.DiGraph()
    for nid in sorted(node_ids):
        if nid in nodes.index:
            g.add_node(nid)
    if not result.edges.empty:
        for row in result.edges.itertuples(index=False):
            source, target = int(row.source_id), int(row.target_id)
            if source in g and target in g:
                g.add_edge(source, target)

    pos = _tree_time_positions(result, set(g.nodes))
    selected = int(selected_node_id) if selected_node_id is not None else None
    ancestors: set[int] = set()
    descendants: set[int] = set()
    if selected is not None and selected in g:
        ancestors = nx.ancestors(g, selected)
        descendants = nx.descendants(g, selected)

    division_parents = {nid for nid in g if g.out_degree(nid) >= 2}
    terminal_nodes = {nid for nid in g if g.out_degree(nid) == 0}

    colors = []
    sizes = []
    for nid in g.nodes:
        if nid == selected:
            colors.append("#16A34A")
            sizes.append(180)
        elif nid in division_parents:
            colors.append("#DC2626")
            sizes.append(125)
        elif nid in ancestors:
            colors.append("#F59E0B")
            sizes.append(95)
        elif nid in descendants:
            colors.append("#7C3AED")
            sizes.append(95)
        elif nid in terminal_nodes:
            colors.append("#64748B")
            sizes.append(75)
        else:
            colors.append(COLORS["accent"])
            sizes.append(70)

    nx.draw_networkx_edges(
        g, pos, edge_color="#94A3B8", arrows=True, arrowsize=11,
        width=1.1, alpha=0.9, ax=ax, connectionstyle="arc3,rad=0.0",
    )
    nx.draw_networkx_nodes(
        g, pos, node_size=sizes, node_color=colors, edgecolors="white",
        linewidths=0.7, alpha=0.96, ax=ax,
    )

    if len(g) <= label_limit:
        labels = {nid: str(nid) for nid in g}
    else:
        important = division_parents | terminal_nodes | ancestors | descendants
        if selected is not None:
            important.add(selected)
        labels = {nid: str(nid) for nid in important if nid in g}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=7, font_color="#0F172A", ax=ax)

    ax.set_xlabel("time frame  →")
    ax.set_yticks([])
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.12, linestyle="--")
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#16A34A", markersize=9, label="selected"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#DC2626", markersize=8, label="division parent"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#F59E0B", markersize=8, label="ancestor"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7C3AED", markersize=8, label="descendant"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#64748B", markersize=8, label="terminal"),
    ]
    ax.legend(handles=legend, loc="upper left", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_selected_lineage_tree(
    result: AnalysisResult,
    lineage_id: int,
    selected_node_id: Optional[int] = None,
) -> plt.Figure:
    """Draw one lineage only, preserving parent/daughter structure and time."""
    if "lineage_id" not in result.nodes.columns:
        return _draw_lineage_tree(result, set(), selected_node_id=selected_node_id)
    ids = set(
        result.nodes.loc[
            result.nodes["lineage_id"].astype(int).eq(int(lineage_id)), "node_id"
        ].astype(int)
    )
    return _draw_lineage_tree(
        result,
        ids,
        selected_node_id=selected_node_id,
        title=f"Lineage {int(lineage_id)} — parent/daughter tree",
    )


def plot_node_family_tree(
    result: AnalysisResult,
    node_id: int,
    ancestor_depth: int = 3,
    descendant_depth: int = 3,
) -> plt.Figure:
    """Draw a bounded family neighborhood centered on one observation."""
    from biohub.lineage_explorer import local_family_node_ids

    ids = local_family_node_ids(
        result, node_id, ancestor_depth=ancestor_depth, descendant_depth=descendant_depth,
        include_siblings=True,
    )
    return _draw_lineage_tree(
        result,
        ids,
        selected_node_id=int(node_id),
        title=f"Node {int(node_id)} — local family structure",
        label_limit=120,
    )


def plot_selected_lineage_trajectory(
    result: AnalysisResult,
    lineage_id: int,
    selected_track_id: Optional[int] = None,
) -> plt.Figure:
    """Plot physical-Y trajectories only for one lineage, emphasizing one track."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if result.nodes.empty or "lineage_id" not in result.nodes.columns:
        ax.text(0.5, 0.5, "No lineage trajectory to display", ha="center", va="center")
        ax.axis("off")
        return fig

    sub = result.nodes[result.nodes["lineage_id"].astype(int).eq(int(lineage_id))].copy()
    if sub.empty:
        ax.text(0.5, 0.5, f"Lineage {lineage_id} not found", ha="center", va="center")
        ax.axis("off")
        return fig
    identity = "track_id" if "track_id" in sub.columns else "node_id"
    scale_y = SCALE[1]
    for track_id, group in sub.sort_values([identity, "t"]).groupby(identity, sort=True):
        is_selected = selected_track_id is not None and int(track_id) == int(selected_track_id)
        color = "#16A34A" if is_selected else "#64748B"
        alpha = 0.98 if is_selected else 0.48
        width = 2.6 if is_selected else 1.0
        ax.plot(group["t"], group["y"].astype(float) * scale_y, color=color, alpha=alpha, linewidth=width)
        ax.scatter(group["t"], group["y"].astype(float) * scale_y, s=22 if is_selected else 10,
                   color=color, alpha=alpha, edgecolors="none")

    # Mark division observations for this lineage.
    ids = set(sub["node_id"].astype(int))
    if not result.edges.empty:
        counts = result.edges[result.edges["source_id"].astype(int).isin(ids)].groupby("source_id").size()
        div_ids = set(counts[counts >= 2].index.astype(int))
        div_rows = sub[sub["node_id"].astype(int).isin(div_ids)]
        if not div_rows.empty:
            ax.scatter(
                div_rows["t"], div_rows["y"].astype(float) * scale_y,
                s=70, facecolors="none", edgecolors="#DC2626", linewidths=1.5,
                label="division parent", zorder=4,
            )
            ax.legend(frameon=False, loc="best")

    ax.set_xlabel("time frame")
    ax.set_ylabel("Y position (µm)")
    ax.set_title(f"Lineage {int(lineage_id)} trajectories" + (
        f" — track {int(selected_track_id)} highlighted" if selected_track_id is not None else ""
    ))
    ax.grid(True, alpha=0.2, linestyle="--")
    fig.tight_layout()
    return fig


def plot_node_temporal_context(
    vol4d: np.ndarray,
    result: AnalysisResult,
    node_id: int,
) -> plt.Figure:
    """Show parent/current/child image context using each relation's own Z slice."""
    from biohub.lineage_explorer import node_relationships

    apply_style()
    rel = node_relationships(result, node_id)
    nodes = result.nodes.set_index("node_id", drop=False)
    related = [rel.get("parent_node_id"), int(node_id)]
    children = list(rel.get("children", []))
    related.append(children[0] if children else None)
    titles = ["Parent", "Selected", "Child" if children else "Next relation"]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, related_id, label in zip(axes, related, titles):
        if related_id is None or int(related_id) not in nodes.index:
            ax.text(0.5, 0.5, f"No {label.lower()}", ha="center", va="center")
            ax.axis("off")
            continue
        row = nodes.loc[int(related_id)]
        t = int(row["t"])
        z = int(np.clip(round(float(row["z"])), 0, vol4d.shape[1] - 1))
        if t < 0 or t >= vol4d.shape[0]:
            ax.text(0.5, 0.5, f"Frame {t} not loaded", ha="center", va="center")
            ax.axis("off")
            continue
        plane = vol4d[t, z]
        vmin, vmax = np.percentile(plane, (1, 99.5))
        ax.imshow(plane, cmap="gray", vmin=vmin, vmax=vmax)
        marker_color = "#16A34A" if int(related_id) == int(node_id) else "#F59E0B"
        ax.scatter([float(row["x"])], [float(row["y"])], s=90, facecolors="none",
                   edgecolors=marker_color, linewidths=2.0)
        ax.set_title(f"{label}: node {int(related_id)} | t={t}, z={z}")
        ax.axis("off")
    fig.suptitle("Local image evidence around the selected observation", y=1.02, fontsize=11)
    fig.tight_layout()
    return fig


def plot_temporal_montage(
    vol4d: np.ndarray,
    frames: Sequence,
    z_idx: int,
    frame_indices: Optional[Sequence[int]] = None,
    title: str = "Temporal montage",
) -> plt.Figure:
    """Multi-panel strip of detection overlays across selected frames."""
    apply_style()
    n_t = vol4d.shape[0]
    if frame_indices is None:
        frame_indices = np.linspace(0, n_t - 1, min(6, n_t), dtype=int)
    frame_indices = list(frame_indices)
    ncols = len(frame_indices)
    fig, axes = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.2))
    if ncols == 1:
        axes = [axes]

    for ax, t_idx in zip(axes, frame_indices):
        vol = vol4d[t_idx]
        vmin, vmax = np.percentile(vol[z_idx], (1, 99.5))
        ax.imshow(vol[z_idx], cmap="gray", vmin=vmin, vmax=vmax)
        coords = frames[t_idx].coords if t_idx < len(frames) else None
        if coords is not None and len(coords):
            zc, yc, xc = coords[:, 0], coords[:, 1], coords[:, 2]
            on = np.abs(zc - z_idx) <= 1
            ax.scatter(xc[on], yc[on], s=12, c=COLORS["detection"], edgecolors="white", linewidths=0.3)
        ax.set_title(f"t = {t_idx}", fontsize=9)
        ax.axis("off")

    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    return fig


def build_overlay_gif(
    vol4d: np.ndarray,
    frames: Sequence,
    z_idx: int,
    fps: int = 4,
    highlight_divisions: Optional[np.ndarray] = None,
) -> bytes:
    """Render a time-lapse GIF of detection overlays for Streamlit display or export."""
    import imageio.v3 as imageio

    images: List[np.ndarray] = []
    for t_idx in range(vol4d.shape[0]):
        fig = plot_slice_overlay(
            vol4d[t_idx],
            z_idx,
            frames[t_idx].coords if t_idx < len(frames) else None,
            title=f"t = {t_idx}",
            highlight_divisions=highlight_divisions,
            show_scale_bar=False,
        )
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        images.append(rgba[:, :, :3].copy())
        plt.close(fig)

    buf = io.BytesIO()
    imageio.imwrite(buf, images, extension=".gif", fps=fps, loop=0)
    return buf.getvalue()


def save_figure(fig: plt.Figure, path: Path, dpi: int = 200) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
