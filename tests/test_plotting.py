import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection

from rainview import ConstantGroupWarning, raincloudplot


def points(ax):
    return [artist for artist in ax.collections if isinstance(artist, PathCollection)]


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_layers_axes_order_and_render(frame, orientation):
    fig, ax, summary = raincloudplot(
        frame, "score", "method", orientation=orientation, order=["B", "A"], title="Проверка"
    )
    assert fig is ax.figure
    assert len(points(ax)) == 2
    assert sum(len(p.get_offsets()) for p in points(ax)) == 60
    assert len(ax.patches) == 2
    ticks = ax.get_xticklabels() if orientation == "vertical" else ax.get_yticklabels()
    assert [tick.get_text() for tick in ticks] == ["B", "A"]
    expected_labels = ("method", "score") if orientation == "vertical" else ("score", "method")
    assert (ax.get_xlabel(), ax.get_ylabel()) == expected_labels
    assert list(summary["groups"]) == ["B", "A"]
    fig.canvas.draw()


def test_layer_bands_do_not_overlap(frame):
    _, ax, _ = raincloudplot(frame, "score", "method")
    for i, collection in enumerate(points(ax)):
        x = collection.get_offsets()[:, 0]
        assert np.all((x >= i + 0.16) & (x <= i + 0.4))
    density = next(x for x in ax.collections if x.get_gid() == "rainview-density-0")
    vertices = density.get_paths()[0].vertices
    assert vertices[:, 0].max() <= -0.12 + 1e-12
    assert vertices[:, 0].min() >= -0.46 - 1e-12


@pytest.mark.parametrize("box,scatter", [(False, True), (True, False), (False, False)])
def test_optional_layers(frame, box, scatter):
    fig, ax, _ = raincloudplot(frame, "score", "method", show_box=box, show_points=scatter)
    assert len(ax.patches) == (2 if box else 0)
    assert len(points(ax)) == (2 if scatter else 0)
    fig.canvas.draw()


def test_jitter_is_reproducible_and_seed_is_local(frame):
    np.random.seed(101)
    before = np.random.get_state()
    _, ax1, _ = raincloudplot(frame, "score", "method")
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    _, ax2, _ = raincloudplot(frame, "score", "method")
    _, ax3, _ = raincloudplot(frame, "score", "method", random_state=123)
    np.testing.assert_array_equal(points(ax1)[0].get_offsets(), points(ax2)[0].get_offsets())
    assert not np.array_equal(points(ax1)[0].get_offsets(), points(ax3)[0].get_offsets())


def test_constant_groups_do_not_invent_measurements(frame):
    frame["score"] = np.repeat([5.0, 7.0], 30)
    with pytest.warns(ConstantGroupWarning) as record:
        fig, ax, summary = raincloudplot(frame, "score", "method")
    assert len(record) == 2
    assert summary["groups"]["A"]["std"] == 0
    assert np.all(points(ax)[0].get_offsets()[:, 1] == 5)
    fig.canvas.draw()


def test_custom_labels_and_palette(frame):
    fig, ax, _ = raincloudplot(
        frame,
        "score",
        "method",
        palette={"A": "#0099AA", "B": "#EE8855"},
        x_label="Метод",
        y_label="Баллы",
        figsize=(8, 4),
    )
    assert ax.get_xlabel() == "Метод"
    assert ax.get_ylabel() == "Баллы"
    np.testing.assert_allclose(fig.get_size_inches(), [8, 4])
    np.testing.assert_allclose(points(ax)[0].get_facecolors()[0, :3], mpl.colors.to_rgb("#0099AA"))


def test_no_figures_created_for_validation_error(frame):
    start = plt.get_fignums()
    from rainview import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        raincloudplot(frame, "score", "method", palette="invalid")
    assert plt.get_fignums() == start


def test_no_global_style_changes(frame):
    before = mpl.rcParams.copy()
    raincloudplot(frame, "score", "method")
    for key in ("axes.facecolor", "font.size", "axes.grid", "figure.figsize", "axes.prop_cycle"):
        assert mpl.rcParams[key] == before[key]
