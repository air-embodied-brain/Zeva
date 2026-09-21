import pytest
from zeva_robotwin_clean.evaluation import EvaluationSpec
from zeva_robotwin_clean.evaluation import build_jobs
from zeva_robotwin_clean.evaluation import summarize_results


def test_release_protocol_has_5000_unique_jobs() -> None:
    spec = EvaluationSpec()
    jobs = build_jobs(spec)
    assert len(jobs) == 5000
    assert len({(job["task"], job["episode"]) for job in jobs}) == 5000
    assert {job["seed"] for job in jobs} == set(range(1000, 1100))
    assert {job["replan_horizon"] for job in jobs} == {15}


def test_result_macro_average() -> None:
    spec = EvaluationSpec(episodes_per_task=2)
    rows = [
        {
            "task": task,
            "episode": episode,
            "seed": spec.first_seed + episode,
            "split": spec.split,
            "instruction_protocol": spec.instruction_protocol,
            "replan_horizon": spec.replan_horizon,
            "success": task_index == 0,
        }
        for task_index, task in enumerate(spec.tasks)
        for episode in range(2)
    ]
    summary = summarize_results(rows, spec)
    assert summary["per_task"][spec.tasks[0]]["success_rate"] == 1.0
    assert summary["macro_success_rate"] == pytest.approx(1 / 50)


def test_summary_rejects_incomplete_results() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        summarize_results([], EvaluationSpec(episodes_per_task=1))
