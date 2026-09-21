import torch
from zeva_robotwin_clean.normalization import quantile_normalize
from zeva_robotwin_clean.normalization import quantile_unnormalize


def test_quantile_normalization_round_trip() -> None:
    statistics = {"q01": [-2.0, 1.0], "q99": [2.0, 5.0]}
    values = torch.tensor([[-2.0, 1.0], [0.0, 3.0], [2.0, 5.0]])
    normalized = quantile_normalize(values, statistics)
    torch.testing.assert_close(normalized, torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]))
    torch.testing.assert_close(quantile_unnormalize(normalized, statistics), values)
