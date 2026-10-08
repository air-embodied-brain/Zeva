"""Inference bundle integrity and legacy weight-name compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from openpi.zeva.cte_eap_policy import canonical_cte_state_dict  # noqa: E402
from openpi.zeva.pim_release import RELEASE_SCHEMA  # noqa: E402
from openpi.zeva.pim_release import _runtime_memory_options  # noqa: E402
from openpi.zeva.pim_release import sha256  # noqa: E402
from openpi.zeva.pim_release import verify_release  # noqa: E402


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    # A tiny integrity fixture, not a usable model or a simulator result.
    root = tmp_path / "release"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"memory_options": {"compression": "uniform"}}))
    manifest = {
        "schema": RELEASE_SCHEMA,
        "policy_schema": "zeva-cte-eap-pim-v1",
        "files": {
            "config.json": {
                "sha256": sha256(root / "config.json"),
                "bytes": (root / "config.json").stat().st_size,
            }
        },
    }
    (root / "release.json").write_text(json.dumps(manifest))
    (root / "COMPLETE").write_text("complete")
    return root


def test_complete_bundle_verifies(bundle: Path) -> None:
    assert verify_release(bundle)["schema"] == RELEASE_SCHEMA


def test_hf_snapshot_cache_links_verify(bundle: Path, tmp_path: Path) -> None:
    cache = tmp_path / "models--sample--pim"
    blobs = cache / "blobs"
    blobs.mkdir(parents=True)
    snapshot = cache / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    for source in bundle.iterdir():
        target = blobs / source.name
        target.write_bytes(source.read_bytes())
        (snapshot / source.name).symlink_to(target)
    (snapshot / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    assert verify_release(snapshot)["schema"] == RELEASE_SCHEMA
    (blobs / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="verification"):
        verify_release(snapshot)


def test_hf_local_download_metadata_is_not_model_content(bundle: Path) -> None:
    metadata = bundle / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    (metadata / "config.json.metadata").write_text("download metadata")
    (bundle / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    assert verify_release(bundle)["schema"] == RELEASE_SCHEMA


def test_symlink_outside_hf_blob_cache_is_rejected(bundle: Path, tmp_path: Path) -> None:
    source = bundle / "config.json"
    external = tmp_path / "external.json"
    external.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(external)
    with pytest.raises(ValueError, match="verification"):
        verify_release(bundle)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [(None, "similarity_merge"), ({}, "similarity_merge"), ({"compression": "uniform"}, "uniform")],
)
def test_runtime_compression_default_and_explicit_override(overrides, expected):
    saved = {"compression": "uniform", "max_attempts": 4, "merge_threshold": 0.95, "phase_weight": 0.5}
    options = _runtime_memory_options(saved, overrides)
    assert options == {**saved, "compression": expected}
    assert saved["compression"] == "uniform"


@pytest.mark.parametrize("change", ["tamper", "missing", "extra", "incomplete"])
def test_bundle_rejects_incomplete_or_changed_files(bundle: Path, change: str) -> None:
    if change == "tamper":
        (bundle / "config.json").write_text("{}")
    elif change == "missing":
        (bundle / "config.json").unlink()
    elif change == "extra":
        (bundle / "training_state.pth").write_bytes(b"not-an-inference-file")
    else:
        (bundle / "COMPLETE").unlink()
    with pytest.raises(ValueError, match=r"verification|extra|incomplete"):
        verify_release(bundle)


def test_bundle_rejects_private_metadata(bundle: Path) -> None:
    path = bundle / "release.json"
    manifest = json.loads(path.read_text())
    manifest["local_path"] = "/private/checkpoint"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"Private|local reference"):
        verify_release(bundle)


def test_bundle_rejects_path_traversal(bundle: Path) -> None:
    path = bundle / "release.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../outside.json"] = manifest["files"]["config.json"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="verification"):
        verify_release(bundle)


def test_earlier_mamba_names_are_converted_without_changing_tensors():
    aliases = {
        "mamba_v.A_log": "temporal.0.A_log",
        "mamba_a.D": "temporal.1.D",
        "mamba_b.x_proj.weight": "temporal.2.x_proj.weight",
        "norm_v.weight": "temporal_norm.0.weight",
        "norm_a.bias": "temporal_norm.1.bias",
        "norm_b.weight": "temporal_norm.2.weight",
        "attn_va.in_proj_weight": "vision_from_action.in_proj_weight",
        "attn_av.out_proj.bias": "action_from_vision.out_proj.bias",
        "attn_b.in_proj_bias": "interaction_from_transition.in_proj_bias",
        "norms.2.weight": "attention_norm.2.weight",
    }
    state = {"blocks.3." + key: torch.randn(2) for key in aliases}
    state["vision_encoder.weight"] = torch.randn(2)
    converted = canonical_cte_state_dict(state)
    for before, after in aliases.items():
        assert converted["blocks.3." + after] is state["blocks.3." + before]
    assert canonical_cte_state_dict(converted) == converted
    with pytest.raises(ValueError, match="Duplicate CTE"):
        canonical_cte_state_dict({**state, **converted})
