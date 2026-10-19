import numpy as np
import pandas as pd
import pytest

from rainview import MissingDataWarning, format_summary, raincloudplot


def test_descriptive_statistics_match_reference(frame):
    _, _, summary = raincloudplot(frame, "score", "method")
    for label, expected in frame.groupby("method").score:
        actual = summary["groups"][label]
        assert actual["n"] == len(expected)
        for field in ("mean", "median", "min", "max", "std"):
            assert actual[field] == pytest.approx(getattr(expected, field)())
        assert actual["q1"] == pytest.approx(expected.quantile(0.25))
        assert actual["q3"] == pytest.approx(expected.quantile(0.75))


def test_missing_rows_counted_once_and_input_unchanged(frame):
    frame.loc[0, "score"] = np.nan
    frame.loc[30, "score"] = np.nan
    frame.loc[31, "method"] = None
    frame.loc[32, ["score", "method"]] = [np.nan, None]
    frame["irrelevant"] = np.nan
    before = frame.copy(deep=True)
    with pytest.warns(MissingDataWarning):
        _, _, summary = raincloudplot(frame, "score", "method")
    assert summary["total_rows"] == 60
    assert summary["valid_rows"] == 56
    assert summary["dropped_rows"] == 4
    assert summary["missing_group_rows"] == 2
    assert summary["groups"]["A"]["dropped_rows"] == 1
    assert summary["groups"]["B"]["dropped_rows"] == 1
    pd.testing.assert_frame_equal(frame, before)


def test_duplicate_index_does_not_change_counts(frame):
    frame.index = [0] * len(frame)
    _, _, summary = raincloudplot(frame, "score", "method")
    assert summary["valid_rows"] == 60
    assert summary["groups"]["A"]["n"] == 30


def test_numeric_group_keys_preserved(frame):
    frame["method"] = np.repeat([10, 20], 30)
    _, _, summary = raincloudplot(frame, "score", "method", order=[20, 10])
    assert list(summary["groups"]) == [20, 10]


def test_text_report(frame):
    _, _, summary = raincloudplot(frame, "score", "method")
    report = format_summary(summary)
    assert "Всего строк: 60" in report
    assert "A: n=30" in report
    assert "std=" in report
    assert "медиана=" in report
    assert "исключено: 0" in report
