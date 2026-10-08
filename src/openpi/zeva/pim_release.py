"""Portable, inference-only cross-attempt PIM checkpoints."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from safetensors.torch import load_file

from openpi.zeva.cte_eap_policy import canonical_cte_state_dict

RELEASE_SCHEMA = "zeva-cross-attempt-pim-release-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_json(value):
    """Remove local training/source references from saved inference configs."""
    excluded = {
        "_name_or_path",
        "name_or_path",
        "pretrained_path",
        "repo_id",
        "push_to_hub",
        "source",
        "source_path",
        "dataset_root",
        "cache_dir",
        "revision",
        "token",
        "password",
        "secret",
        "api_key",
        "access_token",
    }
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in excluded or key.startswith(("optimizer_", "scheduler_", "wandb_")):
                continue
            if isinstance(item, str) and (item.startswith(("/", "~", "file://")) or re.match(r"^[A-Za-z]:[\\/]", item)):
                if "tokenizer" in key:
                    result[key] = "tokenizer"
                elif key in {"state_file", "state_filename"}:
                    result[key] = Path(item).name
                else:
                    raise ValueError(f"Unrecognized local reference in inference config: {key}")
            else:
                result[key] = _public_json(item)
        return result
    if isinstance(value, list):
        return [_public_json(item) for item in value]
    if isinstance(value, str) and (
        value.startswith(("/", "~", "file://"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
        or re.search(r"github_pat_|gh[pousr]_[A-Za-z0-9]{20,}|PRIVATE KEY", value)
    ):
        raise ValueError("Private reference found in public metadata")
    return value


def verify_release(root: Path, *, require_complete: bool = True) -> dict:
    root = root.resolve()
    if require_complete and not (root / "COMPLETE").is_file():
        raise ValueError("Release export is incomplete")
    manifest = json.loads((root / "release.json").read_text())
    if manifest.get("schema") != RELEASE_SCHEMA or manifest.get("policy_schema") != "zeva-cte-eap-pim-v1":
        raise ValueError("Unsupported PIM release schema")
    if _public_json(manifest) != manifest:
        raise ValueError("Release manifest retains source metadata")
    files = manifest.get("files", {})
    if not files:
        raise ValueError("Release has no files")
    for name, expected in files.items():
        path = root / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not (
                path.resolve().is_relative_to(root)
                or (root.parent.name == "snapshots" and path.resolve().is_relative_to(root.parent.parent / "blobs"))
            )
            or not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"Release file failed verification: {name}")
        if path.suffix == ".json" and path.name != "tokenizer.json":
            value = json.loads(path.read_text())
            if _public_json(value) != value:
                raise ValueError(f"Release file retains source metadata: {name}")
    # HF snapshots link verified files into a sibling blob cache. Local-dir
    # downloads also carry transport metadata, which is not model content.
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).parts[:3] != (".cache", "huggingface", "download")
        and path.relative_to(root) != Path(".gitattributes")
    }
    if actual - {"release.json", "COMPLETE"} != set(files):
        raise ValueError("Release contains extra or missing files")
    if any(Path(name).suffix in {".pth", ".pt", ".log", ".mp4"} for name in files):
        raise ValueError("Training or experiment files present in inference release")
    return manifest


def _runtime_memory_options(saved: dict, overrides: dict | None) -> dict:
    """Use the code's compression default unless the caller selects a mode."""
    from openpi.zeva.pim_policy import DEFAULT_PIM_COMPRESSION  # noqa: PLC0415

    return {**saved, "compression": DEFAULT_PIM_COMPRESSION, **(overrides or {})}


def load_release(root: Path, *, device="cuda", memory_options=None):
    """Load sanitized inference files without requiring any training checkpoint."""
    from openpi.zeva.cte_eap import ZevaCTE  # noqa: PLC0415
    from openpi.zeva.cte_eap import ZevaCTEConfig  # noqa: PLC0415
    from openpi.zeva.pim_policy import ZevaCrossAttemptPIMPolicy  # noqa: PLC0415
    from openpi.zeva.robotwin_policy import RobotWinZevaPolicy  # noqa: PLC0415

    root = Path(root)
    manifest = verify_release(root)
    config = ZevaCTEConfig(**json.loads((root / "cte/config.json").read_text()))
    cte = ZevaCTE(replace(config, vision_pretrained=False))
    cte.config = config
    cte.load_state_dict(canonical_cte_state_dict(load_file(str(root / "cte/model.safetensors"))), strict=True)
    cte.to(device).requires_grad_(requires_grad=False).eval()
    loader = RobotWinZevaPolicy.from_handoff(
        root / "handoff",
        goal_embedding_checkpoint=root / "memory/goal",
        install_injection_hooks=False,
        device=device,
    )
    bank = json.loads((root / "memory/task_memory.json").read_text())
    bank.update(load_file(str(root / "memory/task_memory.safetensors")))
    retrieval = json.loads((root / "memory/retrieval.json").read_text())
    retrieval_state = load_file(str(root / "memory/retrieval.safetensors"))
    retrieval["task_prototypes"] = retrieval_state.pop("task_prototypes")
    retrieval["model_state_dict"] = retrieval_state
    options = _runtime_memory_options(manifest["memory_options"], memory_options)
    policy = ZevaCrossAttemptPIMPolicy(loader, cte, bank, retrieval, **options).to(device)
    policy.eap.load_state_dict(load_file(str(root / "eap.safetensors")), strict=True)
    policy.identity = {"schema": manifest["policy_schema"], "release_sha256": sha256(root / "release.json")}
    return policy.eval()
