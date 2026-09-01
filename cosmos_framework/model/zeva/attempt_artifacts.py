"""Validation and manifests for fixed-seed PIM attempt artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ATTEMPT_RECORD_VERSION = "zeva_pim_attempt_v1"
CHUNK_RECORD_VERSION = "zeva_pim_chunk_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_attempt_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "session_id",
        "task",
        "attempt_id",
        "environment_seed",
        "success",
        "total_steps",
        "chunk_records",
        "artifact_npz",
        "artifact_sha256",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"attempt record is missing fields: {sorted(missing)}")
    if record["schema_version"] != ATTEMPT_RECORD_VERSION:
        raise ValueError(f"unsupported attempt schema: {record['schema_version']!r}")
    if not isinstance(record["session_id"], str) or not record["session_id"]:
        raise ValueError("session_id must be a non-empty string")
    if not isinstance(record["task"], str) or not record["task"]:
        raise ValueError("task must be a non-empty string")
    if int(record["attempt_id"]) < 0 or int(record["environment_seed"]) < 0:
        raise ValueError("attempt_id and environment_seed must be non-negative")
    if not isinstance(record["success"], bool):
        raise ValueError("success must be boolean")
    if int(record["total_steps"]) < 0:
        raise ValueError("total_steps must be non-negative")
    chunks = record["chunk_records"]
    if not isinstance(chunks, list):
        raise ValueError("chunk_records must be a list")
    for index, chunk in enumerate(chunks):
        if chunk.get("schema_version") != CHUNK_RECORD_VERSION:
            raise ValueError(f"unsupported chunk schema at index {index}")
        if int(chunk.get("replan_index", -1)) != index:
            raise ValueError("chunk replan indices must be contiguous from zero")
        start, end = int(chunk.get("start_step", -1)), int(chunk.get("end_step", -1))
        if start < 0 or end < start or int(chunk.get("executed_count", -1)) != end - start:
            raise ValueError(f"invalid chunk interval at index {index}")
    artifact = str(record["artifact_npz"])
    checksum = str(record["artifact_sha256"])
    if not artifact.endswith(".npz") or len(checksum) != 64:
        raise ValueError("invalid attempt artifact reference")


def write_manifest(path: Path, records: Iterable[dict[str, Any]]) -> None:
    rows = list(records)
    for row in rows:
        validate_attempt_record(row)
    payload = {
        "schema_version": ATTEMPT_RECORD_VERSION,
        "attempts": len(rows),
        "records": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
