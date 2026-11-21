# SPDX-License-Identifier: OpenMDW-1.1
"""Fit a frozen-policy memory selector and validate it on completed trials."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

CHANNELS = ("action", "effect")
MODES = ("nearest", "learned")


def blend_action(baseline, candidate, *, channels, horizon, strength=0.25):
    """Blend specified action coordinates in a shared frame, then taper to baseline."""
    baseline, candidate = np.asarray(baseline), np.asarray(candidate)
    if baseline.ndim != 2 or baseline.shape != candidate.shape:
        raise ValueError("Actions must have the same [horizon, action_dim] shape")
    if not (1 <= horizon <= len(baseline) and 0 <= strength <= 1):
        raise ValueError("Invalid action blend horizon or strength")
    channels = tuple(channels)
    if not channels or len(set(channels)) != len(channels) or any(not 0 <= i < baseline.shape[1] for i in channels):
        raise ValueError("Select distinct action coordinates within action_dim")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("Action forecasts must be finite")
    result = baseline.astype(np.result_type(baseline, np.float32), copy=True)
    tail = np.linspace(1., 0., len(baseline) - horizon + 1)[1:]
    weights = strength * np.concatenate((np.ones(horizon), tail))
    result[:, channels] += weights[:, None] * (candidate[:, channels] - baseline[:, channels])
    return result


@dataclass(frozen=True)
class GainHead:
    """A small ridge regressor fitted to paired, within-query gains."""

    center: list[float]
    scale: list[float]
    intercept: float
    weight: list[float]

    def scores(self, features):
        scaled = np.clip((np.asarray(features) - self.center) / self.scale, -5., 5.)
        return self.intercept + scaled @ self.weight


def paired_gains(baseline, proposals, target, *, metric):
    """Return baseline loss minus proposal loss, with a common observed target."""
    baseline, proposals, target = (np.asarray(x, dtype=np.float64) for x in (baseline, proposals, target))
    if baseline.shape != target.shape or proposals.ndim != target.ndim + 1 or proposals.shape[1:] != target.shape:
        raise ValueError("Forecast and target shapes must agree")
    if not all(np.isfinite(x).all() for x in (baseline, proposals, target)):
        raise ValueError("Forecasts and targets must be finite")
    axes = tuple(range(1, proposals.ndim))
    if metric == "trajectory_rmse" and target.ndim == 2:
        return np.sqrt(np.mean(np.sum((baseline-target)**2, axis=-1))) - np.sqrt(
            np.mean(np.sum((proposals-target)**2, axis=-1), axis=-1)
        )
    if metric == "rmse":
        return np.sqrt(np.mean((baseline-target)**2)) - np.sqrt(np.mean((proposals-target)**2, axis=axes))
    if metric == "mse":
        return np.mean((baseline-target)**2) - np.mean((proposals-target)**2, axis=axes)
    raise ValueError(f"Unknown metric: {metric}")


def candidate_evidence(donor, features, distance, *, baseline, proposal, random_proposal, target,
                       action_metric="trajectory_rmse"):
    """Score one matched-stage proposal and its same-donor random-stage control."""
    descriptor = np.asarray(features, dtype=np.float64)
    if descriptor.ndim != 1 or not len(descriptor) or not np.isfinite(descriptor).all() or not np.isfinite(distance):
        raise ValueError("Expected a finite candidate descriptor and distance")
    gains, random_gains = {}, {}
    for name, metric in (("action", action_metric), ("effect", "mse")):
        proposals = np.stack((proposal[name], random_proposal[name]))
        gains[name], random_gains[name] = paired_gains(baseline[name], proposals, target[name], metric=metric).tolist()
    return dict(donor=donor, features=descriptor.tolist(), distance=float(distance), gain=gains,
                random_gain=random_gains)


def fit_gain_head(episodes, channel, *, ridge=0.1):
    """Fit from completed episodes, giving every episode the same total weight."""
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    episode_ids = [e["episode"] for e in episodes]
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Expected nonempty, unique training episodes")
    rows = [(episode["episode"], decision["candidates"]) for episode in episodes
            for decision in episode["decisions"] if decision["candidates"]]
    if not rows:
        raise ValueError("Expected paired queries with candidates")
    for key, items in rows:
        donors = {item["donor"] for item in items}
        if key in donors or len(donors) != len(items):
            raise ValueError("Fit candidates must come from distinct other episodes")
    count = Counter(key for key, _ in rows)
    x = np.asarray([item["features"] for _, items in rows for item in items], dtype=np.float64)
    y = np.asarray([item["gain"][channel] for _, items in rows for item in items], dtype=np.float64)
    if x.ndim != 2 or not x.shape[1] or y.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Candidate features and gains must have consistent, finite shapes")
    w = np.asarray([1. / (count[key] * len(items)) for key, items in rows for _ in items])
    w /= w.sum()
    center = w @ x
    scale = np.sqrt(w @ ((x-center)**2)).clip(1e-6)
    design = np.column_stack((np.ones(len(x)), np.clip((x-center)/scale, -5., 5.)))
    contrasts, targets = [], []
    start = 0
    for _, items in rows:
        end = start + len(items)
        part, label = design[start:end], y[start:end]
        contrasts.append(part-part.mean(axis=0))
        targets.append(label-label.mean())
        start = end
    contrast = np.concatenate(contrasts)
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.
    coef = np.linalg.solve(
        (design.T*w) @ design + (contrast.T*w) @ contrast + penalty,
        (design.T*w) @ y + (contrast.T*w) @ np.concatenate(targets),
    )
    return GainHead(center.tolist(), scale.tolist(), float(coef[0]), coef[1:].tolist())


def choose_candidate(head, candidates, mode):
    """Choose using only current candidate features and a precommitted mode."""
    if mode == "empty" or not candidates:
        return None
    if mode == "nearest":
        return min(range(len(candidates)), key=lambda i: candidates[i]["distance"])
    if mode == "learned":
        return int(head.scores([item["features"] for item in candidates]).argmax())
    raise ValueError(f"Unknown selection mode: {mode}")


def _lower_bound(values):
    rng = np.random.default_rng(607)
    index = rng.integers(len(values), size=(2500, len(values)))
    return float(np.quantile(values[index].mean(axis=1), .025))


def replay_summary(rows, channel):
    """Summarize gains and paired controls, with episodes as the resampling unit."""
    records = [row["channels"][channel] for row in rows]
    gain = np.array([row["gain"] for row in records])
    random = np.array([row["random_gain"] for row in records])
    rng = np.random.default_rng(822)
    samples = rng.integers(len(records), size=(4000, len(records)))
    activated = sum(row["activated"] for row in records)
    return {
        "gain": float(gain.mean()),
        "gain_ci95": np.quantile(gain[samples].mean(axis=1), [.025, .975]).tolist(),
        "random_gain": float(random.mean()),
        "gain_over_random_ci95": np.quantile((gain-random)[samples].mean(axis=1), [.025, .975]).tolist(),
        "coverage": float(np.mean([r["coverage"] for r in records])),
        "harm_when_used": sum(row["harmed"] for row in records)/activated if activated else None,
    }


class TrialEvidence:
    """Select a mode using completed trials only, including matched random controls."""

    def __init__(self, *, margin, warmup=8, evidence_limit=16, max_harm=0.30):
        if not 1 <= warmup <= evidence_limit or margin < 0 or not 0 <= max_harm <= 1:
            raise ValueError("Invalid evidence settings")
        self.margin = margin
        self.warmup = warmup
        self.max_harm = max_harm
        self._trials = deque(maxlen=evidence_limit)

    def preferred(self):
        if len(self._trials) < self.warmup:
            return "empty"
        means = np.asarray([trial.mean(axis=0) for trial in self._trials])
        rows = np.concatenate(self._trials)
        eligible = [
            mode for j, mode in enumerate(MODES)
            if _lower_bound(means[:, j]) > 0
            and _lower_bound(means[:, j]-means[:, j+2]) > 0
            and np.mean(rows[:, j] < -self.margin) <= self.max_harm
        ]
        if "learned" in eligible and ("nearest" not in eligible or _lower_bound(means[:, 1]-means[:, 0]) > 0):
            return "learned"
        return "nearest" if "nearest" in eligible else "empty"

    def observe_completed(self, rows):
        evidence = np.asarray(rows, dtype=np.float64)
        if evidence.ndim != 2 or evidence.shape[1] != 4 or not len(evidence) or not np.isfinite(evidence).all():
            raise ValueError("Expected finite [nearest, learned, random-nearest, random-learned] gains")
        self._trials.append(evidence.copy())


def replay_episodes(episodes, heads, *, initial_donors, capacity=2, margins=None, warmup=8,
                    evidence_limit=16, max_harm=0.30, rolling=True):
    """Commit modes before evaluating a trial and insert its evidence on completion.

    Candidate IDs must refer to previously completed donors. Adapters may record
    candidates for any previous trial; only the active bank is eligible to read.
    """
    if capacity < 1 or len(set(initial_donors)) != len(initial_donors):
        raise ValueError("Invalid initial memory bank")
    margins = margins if margins is not None else dict.fromkeys(CHANNELS, 0.)
    evidence = {name: TrialEvidence(margin=margins[name], warmup=warmup,
                                   evidence_limit=evidence_limit, max_harm=max_harm) for name in CHANNELS}
    donors = deque(initial_donors, maxlen=capacity)
    seen = set(initial_donors)
    report = []
    for episode in episodes:
        key = episode["episode"]
        if key in seen:
            raise ValueError("An episode cannot be replayed or serve as its own donor")
        modes = {name: trace.preferred() for name, trace in evidence.items()}
        trial = {name: [] for name in CHANNELS}
        for decision in episode["decisions"]:
            items = decision["candidates"]
            keys = [item["donor"] for item in items]
            if len(set(keys)) != len(keys) or not set(keys) <= seen:
                raise ValueError("Candidates must be distinct, previously completed episodes")
            active = [item for item in items if item["donor"] in donors]
            if not active:
                continue
            for name in CHANNELS:
                chosen = [active[choose_candidate(heads[name], active, mode)] for mode in MODES]
                trial[name].append([item["gain"][name] for item in chosen]
                                   + [item["random_gain"][name] for item in chosen])
        result = {"episode": key, "donors": list(donors), "channels": {}}
        for name, mode in modes.items():
            rows = trial[name]
            index = MODES.index(mode) if rows and mode != "empty" else None
            total = len(episode["decisions"])
            means = np.sum(rows, axis=0)/total if rows else np.zeros(4)
            result["channels"][name] = {
                "mode": MODES[index] if index is not None else "empty",
                "gain": float(means[index]) if index is not None else 0.,
                "random_gain": float(means[index+2]) if index is not None else 0.,
                "coverage": len(rows)/total if index is not None else 0.,
                "activated": len(rows) if index is not None else 0,
                "harmed": sum(1 for row in rows if row[index] < -margins[name]) if index is not None else 0,
            }
            if rows:
                evidence[name].observe_completed(rows)
        report.append(result)
        seen.add(key)
        if rolling:
            donors.append(key)
    return report
