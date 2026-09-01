from __future__ import annotations

from cosmos_framework.model.zeva.attempt_artifacts import (
    ATTEMPT_RECORD_VERSION,
    CHUNK_RECORD_VERSION,
    validate_attempt_record,
)


def test_attempt_artifact_schema() -> None:
    record = {
        "schema_version": ATTEMPT_RECORD_VERSION,
        "session_id": "kettle:197",
        "task": "TurnOnElectricKettle",
        "attempt_id": 0,
        "environment_seed": 197,
        "success": False,
        "total_steps": 16,
        "chunk_records": [
            {
                "schema_version": CHUNK_RECORD_VERSION,
                "replan_index": 0,
                "start_step": 0,
                "end_step": 16,
                "executed_count": 16,
            }
        ],
        "artifact_npz": "attempt_000_seed_197.npz",
        "artifact_sha256": "0" * 64,
    }
    validate_attempt_record(record)
