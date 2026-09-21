# Contributing to ZeVA

Contributions should preserve the public RoboTwin and real-robot scope.

## Development

1. Create a focused branch.
2. Add or update tests for behavioral changes.
3. Run formatting, lint, and the relevant test subset.
4. Keep datasets, checkpoints, logs, videos, credentials, cluster paths, and experiment reports out of commits.
5. Update method or reproduction documentation when a public contract changes.

## Design rules

- Keep causal boundaries explicit.
- Do not let CTE consume future actions or images as inputs.
- Keep PIM scope and reset semantics visible in APIs.
- Prefer one canonical implementation over compatibility wrappers.
- Do not add simulator-specific code outside the RoboTwin integration.
- Preserve third-party notices and licenses.

Please describe the motivation, affected interfaces, and validation performed in each pull request.

