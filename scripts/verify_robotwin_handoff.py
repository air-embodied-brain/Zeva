"""Verify that a RoboTwin handoff matches Zeva's frozen PI0.5 contract."""

from __future__ import annotations

import argparse
import json

from openpi.zeva.robotwin_contract import RobotWinHandoff


def main(handoff_root: str) -> None:
    handoff = RobotWinHandoff.from_root(handoff_root)
    print(
        json.dumps(
            {
                "passed": True,
                "handoff": str(handoff.root),
                "checkpoint": str(handoff.checkpoint),
                "state": "absolute-joint14",
                "action": "chunk-start-relative-eef16",
                "action_horizon": 50,
                "normalization": "mean-std",
                "cameras": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-root", required=True)
    main(parser.parse_args().handoff_root)
