# SPDX-License-Identifier: OpenMDW-1.1
import numpy as np
import pytest

from .gain_memory import CHANNELS, TrialEvidence, blend_action, candidate_evidence, fit_gain_head, replay_episodes


def episode(key, donors):
    return {"episode": key, "decisions": [{"candidates": [
        {"donor": donor, "features": [i, 1.], "distance": float(i),
         "gain": dict.fromkeys(CHANNELS, .25 + .1*i), "random_gain": dict.fromkeys(CHANNELS, 0.)}
        for i, donor in enumerate(donors)
    ]}]}


def test_blend_and_score_candidate():
    baseline = np.zeros((4, 3))
    desired = np.ones_like(baseline)
    action = blend_action(baseline, desired, channels=(0, 2), horizon=2, strength=.5)
    np.testing.assert_array_equal(action[:, 1], baseline[:, 1])
    np.testing.assert_array_equal(action[-1], baseline[-1])
    np.testing.assert_allclose(action[:, 0], [.5, .5, .25, 0.])
    forecast = {"action": baseline, "effect": np.zeros(2)}
    proposal = {"action": action, "effect": np.ones(2)}
    score = candidate_evidence("a", [0., 1.], .5, baseline=forecast, proposal=proposal,
                               random_proposal=forecast, target=proposal)
    assert score["gain"]["action"] > 0
    assert score["gain"]["effect"] == 1.
    assert score["random_gain"] == dict.fromkeys(CHANNELS, 0.)


def test_chronological_replay_and_late_updates():
    heads = {name: fit_gain_head([episode("source", ["a", "b"])], name) for name in CHANNELS}
    stream = [episode("first", ["a", "b"]), episode("second", ["b", "first"]),
              episode("third", ["first", "second"])]
    records = replay_episodes(stream, heads, initial_donors=["a", "b"], capacity=2, warmup=1)
    assert records[0]["donors"] == ["a", "b"]
    assert records[0]["channels"]["action"]["mode"] == "empty"
    assert records[1]["donors"] == ["b", "first"]
    assert records[1]["channels"]["action"]["mode"] == "learned"
    assert records[2]["donors"] == ["first", "second"]
    with pytest.raises(ValueError, match="previously completed"):
        replay_episodes([episode("first", ["second"])], heads, initial_donors=["a"])


def test_evidence_rejects_harm_and_requires_control_advantage():
    evidence = TrialEvidence(margin=.1, warmup=2, evidence_limit=2)
    for _ in range(2):
        evidence.observe_completed([[.1, .2, .1, 0.]])
    assert evidence.preferred() == "learned"
    for _ in range(2):
        evidence.observe_completed([[-.3, -.2, 0., 0.]])
    assert evidence.preferred() == "empty"
