"""Submit a private Kaggle batch job using the verified organizer notebook."""
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np

REFERENCE = Path(__file__).resolve().parents[1] / "results/kaggle_organizer_baseline_v3"


def cli(*args):
    result = subprocess.run(["kaggle", *map(str, args)], capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout


def start(raw, owner):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", owner):
        raise ValueError("Enter your Kaggle username.")
    if len(raw) > 64 * 1024 * 1024 or not raw.startswith(b"\x93NUMPY"):
        raise ValueError("Upload a .npy file up to 64 MiB.")
    volume = np.load(io.BytesIO(raw), allow_pickle=False)
    if volume.ndim != 4 or not 2 <= volume.shape[0] <= 100 or volume.shape[1:] != (64, 256, 256):
        raise ValueError("Use 2–100 frames in Biohub (T,64,256,256) format.")
    if volume.dtype != np.uint16:
        raise ValueError("Use uint16 fluorescence intensities at Biohub voxel spacing.")
    slug = f"biohub-volume-{uuid4().hex[:12]}"
    ref = f"{owner}/{slug}"
    with TemporaryDirectory() as folder:
        path = Path(folder)
        (path / "movie.npy").write_bytes(raw)
        (path / "dataset-metadata.json").write_text(json.dumps(
            {"id": ref, "title": slug, "licenses": [{"name": "other"}]}))
        cli("datasets", "create", "-p", path)
    return {"ref": ref, "slug": slug, "shape": volume.shape,
            "sha256": hashlib.sha256(raw).hexdigest(), "submitted": False}


def notebook_for(job, folder):
    ref = job["ref"]
    code = json.loads(next(REFERENCE.glob("*.ipynb")).read_text(encoding="utf8"))
    meta = json.loads((REFERENCE / "kernel-metadata.json").read_text())
    meta.pop("id_no", None)
    meta.update(id=ref, title=job["slug"], code_file="run.ipynb", is_private=True,
                competition_sources=[], dataset_sources=[*meta["dataset_sources"], ref])
    cells = [cell for cell in code["cells"] if cell["cell_type"] == "code"]
    cells[0]["source"] = "".join(cells[0]["source"]).replace(
        'TEST_DIR = f"{COMP_DIR}/test"', 'TEST_DIR = "/kaggle/working/custom-input"')
    setup = """
import hashlib, json
from pathlib import Path
import numpy as np
import zarr
inputs = list(Path("/kaggle/input").rglob("movie.npy"))
assert len(inputs) == 1, "Expected one mounted movie.npy"
assert hashlib.sha256(inputs[0].read_bytes()).hexdigest() == INPUT_SHA, "Input checksum mismatch"
movie = np.load(inputs[0], allow_pickle=False)
group = zarr.open_group(str(Path(TEST_DIR) / "uploaded_volume.zarr"), mode="w")
group.create_array("0", data=movie, chunks=(1, 16, 128, 128))
group.attrs["multiscales"] = [{"version": "0.5",
    "axes": [{"name": name, "type": "time" if name == "T" else "space"}
             for name in ("T", "Z", "Y", "X")],
    "datasets": [{"path": "0", "coordinateTransformations": [
        {"type": "scale", "scale": [1.0, 1.625, 0.40625, 0.40625]}]}]}]
"""
    cells[2]["source"] = f"INPUT_SHA = {job['sha256']!r}\n" + setup + "\n" + "".join(cells[2]["source"])
    cells[-1]["source"] = "".join(cells[-1]["source"]) + """
Path("/kaggle/working/prediction_manifest.json").write_text(json.dumps({
    "input_sha256": INPUT_SHA, "csv_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))
import shutil
shutil.rmtree(REPO_DIR)
shutil.rmtree(TEST_DIR)
"""
    code["cells"] = cells
    for cell in cells:
        cell.update(outputs=[], execution_count=None)
    (folder / "run.ipynb").write_text(json.dumps(code), encoding="utf8")
    (folder / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf8")


def check(job):
    if not job["submitted"]:
        status = cli("datasets", "status", job["ref"])
        if not re.search(r"\bready\b", status, re.I):
            return status.strip(), None
        with TemporaryDirectory() as folder:
            notebook_for(job, Path(folder))
            cli("kernels", "push", "-p", folder)
        job["submitted"] = True
        return "Private inference job submitted.", None
    status = cli("kernels", "status", job["ref"])
    if not re.search(r"\bCOMPLETE\b", status, re.I):
        return status.strip(), None
    with TemporaryDirectory() as folder:
        cli("kernels", "output", job["ref"], "-p", folder)
        folder = Path(folder)
        manifest = json.loads((folder / "prediction_manifest.json").read_text())
        raw = (folder / "submission.csv").read_bytes()
        if manifest["input_sha256"] != job["sha256"] or manifest["csv_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("Kaggle output does not match this upload.")
    return "Inference complete.", raw
