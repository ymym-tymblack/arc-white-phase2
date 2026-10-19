"""研究用の対比較も最終層MSEだけでなく課金FLOPと0.1床を反映する。"""

import pytest

from research.torch_port.compare_dev import compare


def record(rows):
    return {"per": [{"idx": idx, "mse": mse, "flops": int(util * 2**41)}
                    for idx, mse, util in rows]}


def test_better_mse_can_have_worse_adjusted_score():
    reference = record([(0, 2.0, 0.2), (1, 2.0, 0.05)])
    candidate = record([(0, 1.0, 0.8), (1, 1.0, 0.8)])
    result = compare(candidate, reference)
    assert result["ratio"] == pytest.approx(0.5)
    assert result["head_adj"] == pytest.approx(0.3)
    assert result["cand_adj"] == pytest.approx(0.8)
    assert result["adj_ratio_of_means"] == pytest.approx(8 / 3)
    assert result["mean_per_net_adj_ratio"] == pytest.approx(3.0)
    assert result["improved"] == 2
    assert result["improved_adj"] == 0


@pytest.mark.parametrize(
    "rows",
    [[], [(1, 1.0, 0.2)], [(0, 1.0, 0.2), (0, 1.0, 0.2)]],
)
def test_mismatched_empty_or_duplicate_indices_are_not_silently_dropped(rows):
    with pytest.raises(ValueError, match="indices must match"):
        compare(record(rows), record([(0, 1.0, 0.2)]))


def test_failed_predictions_prevent_an_improvement_claim():
    candidate = dict(record([(0, 1.0, 0.2)]), failures=1)
    with pytest.raises(ValueError, match="Prediction failures"):
        compare(candidate, record([(0, 2.0, 0.2)]))
