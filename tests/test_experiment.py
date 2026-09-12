"""診斷不能因為未來才出現的商品，改變當時可見目錄的分母。"""
import duckdb
import pytest

from amazon_recsys.evaluation.experiment import catalogue_diagnostics


def test_catalogue_diagnostics_keep_cold_items_in_truth():
    with duckdb.connect() as con:
        con.execute("CREATE TABLE inter(item_idx INT, ts INT)")
        con.execute("INSERT INTO inter VALUES (1, 1), (2, 2), (3, 9)")
        result = catalogue_diagnostics(con, "inter", 5, [{1, 3}, {3}])
    assert result["n_items_before_cutoff"] == 2
    assert result["n_items_all_dates"] == 3
    assert result["cold_truth_fraction_micro"] == pytest.approx(2 / 3)
    assert result["train_catalogue_recall_ceiling_macro"] == pytest.approx(0.25)
