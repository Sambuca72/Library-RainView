"""Явный экспорт рисунка и читаемый отчёт о включённых наблюдениях."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

from matplotlib.figure import Figure

from .exceptions import InvalidParameterError


def format_summary(summary: dict) -> str:
    """Вернуть русскую текстовую сводку результата raincloudplot()."""
    lines = [
        f"Всего строк: {summary['total_rows']}; использовано: {summary['valid_rows']}; "
        f"исключено: {summary['dropped_rows']}.",
        f"Из них без метки группы: {summary['missing_group_rows']}.",
    ]
    for label, stats in summary["groups"].items():
        lines.append(
            f"{label}: n={stats['n']}, среднее={stats['mean']:.4g}, "
            f"медиана={stats['median']:.4g}, std={stats['std']:.4g}, "
            f"min={stats['min']:.4g}, max={stats['max']:.4g}, "
            f"исключено в группе={stats['dropped_rows']}."
        )
    return "\n".join(lines)


def save_plot(fig: Figure, path: str | Path, *, dpi: int = 180) -> Path:
    """Сохранить Figure в PNG/SVG по явному запросу пользователя.

    Родительская папка должна существовать. Существующий файл перезаписывается,
    как при fig.savefig(). Figure остаётся открытой для дальнейшей работы.
    """
    if not isinstance(fig, Figure):
        raise InvalidParameterError("fig должен быть объектом matplotlib.figure.Figure.")
    try:
        target = Path(path)
    except TypeError as exc:
        raise InvalidParameterError("path должен быть строкой или pathlib.Path.") from exc
    if target.suffix.lower() not in (".png", ".svg"):
        raise InvalidParameterError("Для save_plot поддерживаются только расширения PNG и SVG.")
    if isinstance(dpi, bool) or not isinstance(dpi, Integral) or dpi <= 0:
        raise InvalidParameterError("dpi должен быть целым положительным числом.")
    fig.savefig(target, format=target.suffix[1:].lower(), dpi=dpi, bbox_inches="tight")
    return target
