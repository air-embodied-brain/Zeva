# RoboTwin cross-attempt PIM evaluation

This guide reports only the cumulative success rate after attempts 1–4. It
uses the frozen 10-task × 20-episode seen-instruction manifest in
[`configs/robotwin_multiattempt_original10x20.json`](../../configs/robotwin_multiattempt_original10x20.json)
(SHA256 `1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b`).
The manifest contains seeds and instructions, not outcome labels.

## Inputs

- RoboTwin with the same randomized task configuration, expert-valid scene
  seeds, seen instructions, and per-task step limits.
- The matching cross-attempt PIM inference package described in
  [Checkpoint package](README_CHECKPOINT.md).
- A RoboTwin/model-server adapter implementing the interface below. The
  simulator, model server, and their machine-specific configuration are
  external to this repository; this release does not yet provide a turnkey
  simulator adapter. Keep local endpoints and credentials outside Git.

The adapter factory is `package.module:function`. It receives a JSON config
with an injected `evaluation_slot` object containing `index`, `count`,
`model_rng_seed`, and `model_seed_policy`. It must return an object with:

```text
setup_scene(task, seed, instruction)
close_scene()
reset_policy(scope)
rollout(video_path) -> (success: bool, executed_steps: int, step_limit: int)
close()
```

## Run

From the repository root, give each run a fresh output directory:

```bash
python3 -m scripts.robotwin.evaluate_multiattempt \
  --seed-manifest configs/robotwin_multiattempt_original10x20.json \
  --expected-manifest-sha256 1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b \
  --adapter your_robotwin_adapter:make_adapter \
  --adapter-config /path/to/local-pim-config.json \
  --slots 8 --model-rng-seed 20260907 \
  --output-root /path/to/new-pim-evaluation
```

Tasks are assigned by sorted task index modulo eight; each slot retains its
own model RNG stream across its assigned tasks. Attempt 1 starts with empty
PIM (`reset_policy("episode")`). After failure, recreate the same scene and
instruction, then call `reset_policy("attempt")` to commit that attempt's BIT
trace. Stop on success or after attempt 4. Save every executed attempt's
video. The runner refuses an existing output directory and writes `COMPLETE`
only after the progress and video audit passes; partial runs are not silently
resumed because the model RNG state cannot be recovered from progress JSON.

To re-audit a completed run:

```bash
python3 -m scripts.robotwin.report_multiattempt \
  --run-root /path/to/new-pim-evaluation \
  --expected-manifest-sha256 1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b \
  --output /path/to/pim-cumulative-report.json
```

The reporter verifies all 200 seed/instruction identities, reset scopes,
stop-on-success behavior, and one nonempty video per executed attempt.
`--metadata-only` skips video verification and marks it as not run.

## Cumulative metric

| Attempt ≤ | Solved / 200 | Cumulative success rate |
| ---: | ---: | ---: |
| 1 | 119 / 200 | 59.5% |
| 2 | 151 / 200 | 75.5% |
| 3 | 165 / 200 | 82.5% |
| 4 | 177 / 200 | 88.5% |

The JSON output lists `attempt`, `solved`, `total`, and `rate` for these four cumulative points, plus the manifest hash and video-audit status.
