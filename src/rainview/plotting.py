"""Публичная функция raincloudplot и независимые визуальные слои."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Hashable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from ._data import prepare_data
from .exceptions import ConstantGroupWarning, InvalidParameterError


def _number(value: Any, name: str, low: float, high: float | None = None) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < low
        or (high is not None and value > high)
    ):
        bounds = f"от {low}" if high is None else f"от {low} до {high}"
        raise InvalidParameterError(f"{name} должен быть конечным числом {bounds}.")
    return float(value)


def _colors(palette: Any, labels: list[Hashable]) -> list[tuple]:
    try:
        if isinstance(palette, str):
            cmap = mpl.colormaps[palette]
            if hasattr(cmap, "colors") and len(cmap.colors) >= len(labels):
                colors = list(cmap.colors[: len(labels)])
            else:
                colors = list(cmap(np.linspace(0.08, 0.92, len(labels))))
        elif isinstance(palette, Mapping):
            colors = [palette[label] for label in labels]
        else:
            colors = list(palette)
            if len(colors) < len(labels):
                raise ValueError("short palette")
            colors = colors[: len(labels)]
        return [to_rgba(color) for color in colors]
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidParameterError(
            "palette: укажите имя палитры Matplotlib (например Set2), "
            "список цветов для всех групп или словарь {группа: цвет}."
        ) from exc


def _density(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    # Для постоянной группы не добавляем искусственный шум к измерениям.
    if np.all(values == values[0]):
        return None
    scale = max(float(np.max(np.abs(values))), 1.0)
    scaled = values / scale
    center, spread = float(scaled.mean()), float(scaled.std())
    if spread == 0:
        return None
    normalized = (scaled - center) / spread
    grid = np.linspace(normalized.min(), normalized.max(), 192)
    try:
        density = gaussian_kde(normalized)(grid)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(density).all() or density.max() <= 0:
        return None
    support = (grid * spread + center) * scale
    return support, density / density.max() * 0.34


def _box_stats(values: np.ndarray, stats: dict) -> dict:
    # Усы Тьюки: крайние реальные точки внутри 1.5 IQR.
    q1, q3 = stats["q1"], stats["q3"]
    with np.errstate(over="ignore"):
        iqr = np.float64(q3) - np.float64(q1)
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    low = max(float(values[values >= lower].min()), stats["min"])
    high = min(float(values[values <= upper].max()), stats["max"])
    return {
        "med": stats["median"],
        "q1": q1,
        "q3": q3,
        "whislo": min(low, q1),
        "whishi": max(high, q3),
        "fliers": values[(values < low) | (values > high)],
    }


def raincloudplot(
    data: pd.DataFrame,
    value: str,
    group: str,
    *,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    palette: str | Sequence | Mapping = "Set2",
    figsize: tuple[float, float] = (10, 6),
    orientation: str = "vertical",
    point_alpha: float = 0.6,
    point_size: float = 20,
    show_box: bool = True,
    show_points: bool = True,
    order: Sequence[Hashable] | None = None,
    random_state: int | None = 42,
) -> tuple[Figure, Axes, dict[str, Any]]:
    """Построить half-violin, boxplot и jitter-точки для двух и более групп.

    Пропуски в value или group исключаются с предупреждением и учётом в summary.
    Требуется минимум пять наблюдений в каждой группе. Исходная таблица не меняется.
    В horizontal числовая ось — X; x_label/y_label всегда обозначают оси рисунка.
    random_state фиксирует jitter и не меняет глобальный генератор NumPy.
    Возвращает (fig, ax, summary); не вызывает show() и не сохраняет файлы.
    """
    if orientation not in ("vertical", "horizontal"):
        raise InvalidParameterError('orientation должен быть "vertical" или "horizontal".')
    for name, flag in (("show_box", show_box), ("show_points", show_points)):
        if not isinstance(flag, (bool, np.bool_)):
            raise InvalidParameterError(f"{name} должен быть True или False.")
    for name, text in (("title", title), ("x_label", x_label), ("y_label", y_label)):
        if text is not None and not isinstance(text, str):
            raise InvalidParameterError(f"{name} должен быть строкой или None.")
    point_alpha = _number(point_alpha, "point_alpha", 0, 1)
    point_size = _number(point_size, "point_size", 0)
    if point_size == 0:
        raise InvalidParameterError("point_size должен быть больше нуля.")
    try:
        if len(figsize) != 2:
            raise TypeError
        size = tuple(_number(v, "figsize", 0) for v in figsize)
    except TypeError as exc:
        raise InvalidParameterError("figsize должен содержать ширину и высоту в дюймах.") from exc
    if min(size) == 0:
        raise InvalidParameterError("Размеры figsize должны быть больше нуля.")
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, Integral)
        or random_state < 0
    ):
        raise InvalidParameterError(
            "random_state должен быть целым неотрицательным числом или None."
        )
    prepared = prepare_data(data, value, group, order)
    colors = _colors(palette, prepared.labels)
    densities = [_density(values) for values in prepared.values]
    for label, density in zip(prepared.labels, densities, strict=True):
        if density is None:
            warnings.warn(
                f'В группе "{label}" плотность не определена из-за отсутствия разброса '
                "или численной вырожденности. Вместо half-violin показана линия.",
                ConstantGroupWarning,
                stacklevel=2,
            )
    rng = np.random.default_rng(random_state)
    vertical = orientation == "vertical"
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    try:
        for index, (label, values, color, density) in enumerate(
            zip(prepared.labels, prepared.values, colors, densities, strict=True)
        ):
            # Три непересекающиеся полосы: плотность слева, ящик в центре, точки справа.
            baseline = index - 0.12
            if density is None:
                location = float(values[0])
                if vertical:
                    (artist,) = ax.plot(
                        [baseline - 0.34, baseline], [location, location], color=color, linewidth=3
                    )
                else:
                    (artist,) = ax.plot(
                        [location, location], [baseline - 0.34, baseline], color=color, linewidth=3
                    )
            else:
                support, width = density
                if vertical:
                    artist = ax.fill_betweenx(
                        support, baseline - width, baseline, color=color, alpha=0.65, linewidth=1
                    )
                else:
                    artist = ax.fill_between(
                        support, baseline - width, baseline, color=color, alpha=0.65, linewidth=1
                    )
            artist.set_gid(f"rainview-density-{index}")
            if show_box:
                direction = (
                    {"orientation": orientation}
                    if "orientation" in inspect.signature(ax.bxp).parameters
                    else {"vert": vertical}
                )
                box = ax.bxp(
                    [_box_stats(values, prepared.summary["groups"][label])],
                    positions=[index],
                    widths=0.14,
                    patch_artist=True,
                    manage_ticks=False,
                    showfliers=not show_points,
                    boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.5},
                    medianprops={"color": "#253447", "linewidth": 1.8},
                    whiskerprops={"color": "#536071", "linewidth": 1.1},
                    capprops={"color": "#536071", "linewidth": 1.1},
                    flierprops={"marker": "o", "markersize": 3, "markeredgecolor": color},
                    zorder=3,
                    **direction,
                )
                box["boxes"][0].set_gid(f"rainview-box-{index}")
            if show_points:
                jitter = index + rng.uniform(0.16, 0.40, size=len(values))
                x, y = (jitter, values) if vertical else (values, jitter)
                points = ax.scatter(
                    x, y, s=point_size, alpha=point_alpha, color=color, linewidths=0, zorder=2
                )
                points.set_gid(f"rainview-points-{index}")
        ticks = np.arange(len(prepared.labels))
        labels = [str(label) for label in prepared.labels]
        if vertical:
            ax.set_xticks(ticks, labels)
            ax.set_xlim(-0.62, len(labels) - 0.42)
            ax.set_xlabel(group if x_label is None else x_label)
            ax.set_ylabel(value if y_label is None else y_label)
            ax.grid(axis="y", color="#E6E9ED", linewidth=0.8)
        else:
            ax.set_yticks(ticks, labels)
            ax.set_ylim(len(labels) - 0.42, -0.62)
            ax.set_xlabel(value if x_label is None else x_label)
            ax.set_ylabel(group if y_label is None else y_label)
            ax.grid(axis="x", color="#E6E9ED", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#B7C0CA")
        ax.tick_params(length=0, pad=8)
        if title is not None:
            ax.set_title(title, loc="left", pad=18, fontsize=14, fontweight="bold")
    except Exception:
        plt.close(fig)
        raise
    return fig, ax, prepared.summary
