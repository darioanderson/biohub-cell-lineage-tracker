"""Zarr volume and GEFF graph loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import blosc2
except ImportError:
    blosc2 = None  # type: ignore

try:
    import zarr
except ImportError:
    zarr = None  # type: ignore


def list_datasets(split_dir: Path) -> List[str]:
    """Return sorted dataset names (without .zarr suffix)."""
    if not split_dir or not split_dir.is_dir():
        return []
    return sorted(p.name[:-5] for p in split_dir.iterdir() if p.name.endswith(".zarr"))


def read_zarr_meta(zarr_path: Path) -> Tuple[Tuple[int, ...], np.dtype]:
    """Read shape and dtype from zarr metadata."""
    meta_path = zarr_path / "0" / "zarr.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text())
    return tuple(meta["shape"]), np.dtype(meta["data_type"])


def read_physical_metadata(zarr_path: Path) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
    """Read OME-Zarr spatial voxel size and frame interval when declared."""
    meta_path = zarr_path / "zarr.json"
    if not meta_path.exists():
        return None, None
    try:
        attrs = json.loads(meta_path.read_text()).get("attributes", {})
        multiscale = attrs.get("multiscales", [])[0]
        axes = multiscale.get("axes", [])
        dataset = multiscale.get("datasets", [])[0]
        transform = next(
            item for item in dataset.get("coordinateTransformations", [])
            if item.get("type") == "scale"
        )
        values = [float(v) for v in transform["scale"]]
        names = [str(axis.get("name", "")).lower() for axis in axes]
        mapped = dict(zip(names, values))
        units = {str(axis.get('name','')).lower(): str(axis.get('unit','')).lower() for axis in axes}
        spatial_factors = {'micrometer':1., 'micrometre':1., 'um':1., 'µm':1., 'nanometer':.001, 'nanometre':.001, 'millimeter':1000., 'millimetre':1000., 'meter':1e6}
        time_factors = {'second':1., 's':1., 'millisecond':.001, 'minute':60., 'hour':3600.}
        voxel = tuple(mapped[k] * spatial_factors[units[k]] for k in ('z','y','x')) if all(units.get(k) in spatial_factors for k in ('z','y','x')) else None
        frame_interval = mapped.get('t', 0) * time_factors[units['t']] if units.get('t') in time_factors else None
        if voxel is not None and (not np.isfinite(voxel).all() or min(voxel) <= 0):
            voxel = None
        if frame_interval is not None and (not np.isfinite(frame_interval) or frame_interval <= 0):
            frame_interval = None
        return voxel, frame_interval
    except (IndexError, KeyError, StopIteration, TypeError, ValueError):
        return None, None


def load_volume(
    zarr_path: Path,
    t: int,
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    """Load one timepoint as (Z, Y, X)."""
    chunk = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
    if blosc2 is not None and chunk.exists():
        raw = blosc2.decompress(chunk.read_bytes())
        return np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
    if zarr is not None:
        arr = zarr.open_array(str(zarr_path / "0"), mode="r")
        return np.asarray(arr[t], dtype=dtype)
    raise ImportError("blosc2 or zarr required to read volumes")


def read_estimated_nodes(geff_path: Path) -> float | None:
    """Read estimated_number_of_nodes from GEFF metadata if present."""
    meta_path = geff_path / "zarr.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None

    def _find(obj):
        if isinstance(obj, dict):
            if "estimated_number_of_nodes" in obj:
                return obj["estimated_number_of_nodes"]
            for v in obj.values():
                found = _find(v)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _find(item)
                if found is not None:
                    return found
        return None

    val = _find(meta)
    return float(val) if val is not None else None


def find_geff_path(split_dir: Path, name: str) -> Optional[Path]:
    """Locate a GEFF graph for ``name`` across common competition layouts."""
    candidates = [
        split_dir / f"{name}.geff",
        split_dir / "geff" / f"{name}.geff",
        split_dir / "graphs" / f"{name}.geff",
        split_dir / "labels" / f"{name}.geff",
        split_dir.parent / "geff" / f"{name}.geff",
        split_dir.parent / "graphs" / f"{name}.geff",
        split_dir.parent / "labels" / f"{name}.geff",
    ]
    return next((p for p in candidates if p.exists()), None)


def _read_single_chunk_zarr3(array_path: Path) -> np.ndarray:
    """Read the single-chunk Zarr v3 arrays used by current GEFF labels.

    This compatibility path is needed on Python 3.10, where stable zarr 3
    releases are unavailable. It fails loudly if a future GEFF array uses
    multiple chunks or an unsupported codec.
    """
    metadata = json.loads((array_path / "zarr.json").read_text())
    shape = tuple(int(v) for v in metadata["shape"])
    chunk_shape = tuple(int(v) for v in metadata["chunk_grid"]["configuration"]["chunk_shape"])
    if chunk_shape != shape:
        raise ValueError(f"Expected a single chunk at {array_path}; shape={shape}, chunk={chunk_shape}")
    key = array_path / "c"
    for _ in shape:
        key /= "0"
    payload = key.read_bytes()
    endian = "="
    for codec in reversed(metadata.get("codecs", [])):
        name = codec["name"]
        if name == "zstd":
            from numcodecs import Zstd

            payload = Zstd().decode(payload)
        elif name == "bytes":
            endian = "<" if codec.get("configuration", {}).get("endian") == "little" else ">"
        else:
            raise ValueError(f"Unsupported Zarr v3 codec {name!r} at {array_path}")
    dtype = np.dtype(metadata["data_type"]).newbyteorder(endian)
    expected = int(np.prod(shape))
    values = np.frombuffer(payload, dtype=dtype, count=expected)
    if values.size != expected:
        raise ValueError(f"Decoded {values.size} values at {array_path}; expected {expected}")
    return values.reshape(shape).copy()


def _load_geff_graph_zarr3_compat(geff_path: Path):
    import pandas as pd

    nodes = pd.DataFrame(
        {
            "node_id": _read_single_chunk_zarr3(geff_path / "nodes" / "ids"),
            "t": _read_single_chunk_zarr3(geff_path / "nodes" / "props" / "t" / "values"),
            "z": _read_single_chunk_zarr3(geff_path / "nodes" / "props" / "z" / "values"),
            "y": _read_single_chunk_zarr3(geff_path / "nodes" / "props" / "y" / "values"),
            "x": _read_single_chunk_zarr3(geff_path / "nodes" / "props" / "x" / "values"),
        }
    )
    edge_ids = _read_single_chunk_zarr3(geff_path / "edges" / "ids")
    edges = pd.DataFrame({"source_id": edge_ids[:, 0], "target_id": edge_ids[:, 1]})
    return nodes, edges


def load_geff_graph(geff_path: Path):
    """Load node and edge tables from a GEFF directory."""
    if not geff_path.exists():
        raise FileNotFoundError(f"GEFF directory not found: {geff_path}")
    try:
        import pandas as pd

        if zarr is None:
            raise ImportError('zarr unavailable; attempting explicit GEFF compatibility reader')
        root = zarr.open_group(str(geff_path), mode="r")
        nodes_grp = root["nodes"]
        edges_grp = root["edges"]
        node_ids = np.asarray(nodes_grp["ids"])
        props = nodes_grp["props"]
        nodes = pd.DataFrame(
            {
                "node_id": node_ids,
                "t": np.asarray(props["t"]["values"]),
                "z": np.asarray(props["z"]["values"]),
                "y": np.asarray(props["y"]["values"]),
                "x": np.asarray(props["x"]["values"]),
            }
        )
        edge_ids = np.asarray(edges_grp["ids"])
        edges = pd.DataFrame({"source_id": edge_ids[:, 0], "target_id": edge_ids[:, 1]})
        return nodes, edges
    except Exception as exc:
        compat_error: Exception | None = None
        try:
            nodes, edges = _load_geff_graph_zarr3_compat(geff_path)
            logger.warning(
                "Stable GEFF reader failed for %s (%s); used explicit Zarr v3 compatibility reader",
                geff_path,
                exc,
            )
            return nodes, edges
        except Exception as compat_exc:
            compat_error = compat_exc
        contents = sorted(
            str(p.relative_to(geff_path))
            for p in geff_path.rglob("*")
            if p.is_file()
        )
        message = (
            f"Failed to load GEFF graph at {geff_path}: primary={exc}; "
            f"compatibility={compat_error}. "
            f"Found {len(contents)} files; first entries: {contents[:20]}"
        )
        logger.exception(message)
        raise RuntimeError(message) from exc
