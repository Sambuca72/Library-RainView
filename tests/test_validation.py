import numpy as np
import pandas as pd
import pytest

from rainview import (
    ColumnNotFoundError,
    EmptyGroupError,
    GroupCountWarning,
    InsufficientDataError,
    InsufficientGroupsError,
    InvalidDataError,
    InvalidParameterError,
    MissingDataWarning,
    NonNumericColumnError,
    RainViewError,
    SmallSampleWarning,
    raincloudplot,
)


def test_reject_non_dataframe():
    with pytest.raises(InvalidDataError, match="DataFrame"):
        raincloudplot([1, 2], "score", "method")


@pytest.mark.parametrize("column", ["score", "method"])
def test_missing_column(frame, column):
    with pytest.raises(ColumnNotFoundError, match=column):
        raincloudplot(frame.drop(columns=column), "score", "method")


@pytest.mark.parametrize("kind", ["string", "bool", "complex"])
def test_reject_non_real_numeric(frame, kind):
    if kind == "string":
        frame["score"] = frame["score"].astype(str)
    elif kind == "bool":
        frame["score"] = True
    else:
        frame["score"] = frame["score"].astype(complex)
    with pytest.raises(NonNumericColumnError, match="числовой столбец"):
        raincloudplot(frame, "score", "method")


def test_duplicate_columns(frame):
    frame.columns = ["score", "score"]
    with pytest.raises(InvalidDataError, match="уникальными"):
        raincloudplot(frame, "score", "method")


def test_multiindex_columns_rejected(frame):
    frame.columns = pd.MultiIndex.from_tuples([("score", "value"), ("method", "group")])
    with pytest.raises(InvalidDataError, match="одноуровневые"):
        raincloudplot(frame, "score", "method")


def test_same_column(frame):
    with pytest.raises(InvalidParameterError, match="разные"):
        raincloudplot(frame, "score", "score")


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinities_not_silently_dropped(frame, bad):
    frame.loc[0, "score"] = bad
    with pytest.raises(InvalidDataError, match="бесконечные"):
        raincloudplot(frame, "score", "method")


def test_empty_group_is_not_silently_lost(frame):
    frame.loc[frame.method == "A", "score"] = np.nan
    with pytest.raises(EmptyGroupError, match='группе "A"'):
        raincloudplot(frame, "score", "method")


def test_minimum_sample_size_after_cleaning(frame):
    frame.loc[0:25, "score"] = np.nan
    with pytest.raises(InsufficientDataError, match="5"):
        raincloudplot(frame, "score", "method")


def test_one_group(frame):
    with pytest.raises(InsufficientGroupsError, match="две группы"):
        raincloudplot(frame.iloc[:30], "score", "method")


def test_no_rows(frame):
    with pytest.raises(InsufficientGroupsError):
        raincloudplot(frame.iloc[:0], "score", "method")


def test_no_group_labels(frame):
    frame["method"] = None
    with pytest.raises(InsufficientGroupsError):
        raincloudplot(frame, "score", "method")


def test_unhashable_groups(frame):
    frame["method"] = [["A"] for _ in range(len(frame))]
    with pytest.raises(InvalidDataError, match="хешируемыми"):
        raincloudplot(frame, "score", "method")


def test_nullable_numeric_dtype(frame):
    frame["score"] = pd.array(np.arange(60), dtype="Int64")
    frame.loc[0, "score"] = pd.NA
    with pytest.warns(MissingDataWarning):
        _, _, summary = raincloudplot(frame, "score", "method")
    assert summary["groups"]["A"]["n"] == 29


def test_unused_categories_ignored_first_observed_order(frame):
    frame["method"] = pd.Categorical(frame.method, categories=["B", "A", "unused"])
    _, ax, summary = raincloudplot(frame, "score", "method")
    assert list(summary["groups"]) == ["A", "B"]
    assert [x.get_text() for x in ax.get_xticklabels()] == ["A", "B"]


@pytest.mark.parametrize("order", [["A"], ["A", "A"], ["A", "C"], "AB", [["A"], ["B"]]])
def test_invalid_order(frame, order):
    with pytest.raises(InvalidParameterError, match="order"):
        raincloudplot(frame, "score", "method", order=order)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"orientation": "diagonal"},
        {"point_alpha": -0.1},
        {"point_alpha": 1.1},
        {"point_alpha": np.nan},
        {"point_size": 0},
        {"point_size": -1},
        {"point_size": "20"},
        {"show_box": "yes"},
        {"show_points": 1},
        {"figsize": (0, 6)},
        {"figsize": (10,)},
        {"figsize": None},
        {"figsize": (10, np.inf)},
        {"random_state": -1},
        {"random_state": True},
        {"random_state": 2.5},
        {"title": 3},
        {"x_label": 3},
        {"y_label": 3},
        {"palette": "nonexistent"},
        {"palette": []},
        {"palette": ["bad", "red"]},
        {"palette": {"A": "red"}},
    ],
)
def test_invalid_display_options(frame, kwargs):
    with pytest.raises(InvalidParameterError):
        raincloudplot(frame, "score", "method", **kwargs)


def test_small_groups_warn_but_render(frame):
    subset = pd.concat([frame.iloc[:5], frame.iloc[30:35]])
    with pytest.warns(SmallSampleWarning, match="5–19"):
        _, _, summary = raincloudplot(subset, "score", "method")
    assert summary["valid_rows"] == 10


def test_many_groups_warn():
    frame = pd.DataFrame({"group": np.repeat(np.arange(11), 20), "x": np.tile(np.arange(20), 11)})
    with pytest.warns(GroupCountWarning, match="10 групп"):
        _, _, summary = raincloudplot(frame, "x", "group")
    assert len(summary["groups"]) == 11


def test_errors_share_base_class():
    assert issubclass(ColumnNotFoundError, RainViewError)
    assert issubclass(InsufficientDataError, RainViewError)
