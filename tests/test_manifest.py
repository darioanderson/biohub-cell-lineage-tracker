from biohub.config import Config
from biohub.manifest import build_run_manifest, config_sha256, input_fingerprint


def test_manifest_records_honest_input_fingerprint_definition(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "zarr.json").write_text('{"shape":[1,2,3]}')
    (source / "chunk").write_bytes(b"pixels")
    cfg = Config()
    manifest = build_run_manifest(cfg, [source])
    assert manifest["config_sha256"] == config_sha256(cfg)
    assert manifest["inputs"][0]["fingerprint_sha256"] == input_fingerprint(source)
    assert "not a full byte checksum" in manifest["fingerprint_definition"]
