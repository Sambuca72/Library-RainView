"""Проверка и подготовка данных без изменения исходного DataFrame."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from .exceptions import (
    ColumnNotFoundError,
    EmptyGroupError,
    GroupCountWarning,
    InsufficientDataError,
    InsufficientGroupsError,
    InvalidDataError,
    InvalidParameterError,
    LargeDatasetWarning,
    MissingDataWarning,
    NonNumericColumnError,
    SmallSampleWarning,
)


@dataclass(frozen=True)
class PreparedData:
    labels: list[Hashable]
    values: list[np.ndarray]
    summary: dict[str, Any]


def prepare_data(
    data: pd.DataFrame,
    value: str,
    group: str,
    order: Sequence[Hashable] | None,
) -> PreparedData:
    if not isinstance(data, pd.DataFrame):
        raise InvalidDataError("Параметр data должен быть pandas.DataFrame.")
    if isinstance(data.columns, pd.MultiIndex):
        raise InvalidDataError("Нужны одноуровневые имена столбцов DataFrame.")
    if not data.columns.is_unique:
        raise InvalidDataError("Имена столбцов DataFrame должны быть уникальными.")
    for column in (value, group):
        if not isinstance(column, str):
            raise InvalidParameterError("Параметры value и group должны быть именами столбцов.")
        if column not in data.columns:
            raise ColumnNotFoundError(f'Столбец "{column}" не найден в DataFrame.')
    if value == group:
        raise InvalidParameterError("value и group должны указывать разные столбцы.")
    dtype = data[value].dtype
    if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
        raise NonNumericColumnError(
            f'Для построения графика требуется числовой столбец: "{value}". '
            "Строки, логические и комплексные значения не поддерживаются."
        )
    try:
        values = data[value].to_numpy(dtype=np.float64, na_value=np.nan)
    except (ValueError, TypeError, OverflowError) as exc:
        raise InvalidDataError("Не удалось представить измерения вещественными числами.") from exc
    if np.isinf(values).any():
        raise InvalidDataError(
            f'Столбец "{value}" содержит бесконечные значения. Исправьте их явно перед построением.'
        )
    try:
        codes, unique = pd.factorize(data[group], sort=False)
    except (TypeError, ValueError) as exc:
        raise InvalidDataError(
            "Метки групп должны быть хешируемыми значениями, например строками или числами."
        ) from exc
    labels = list(unique)
    arrays: dict[Hashable, np.ndarray] = {}
    stats: dict[Hashable, dict[str, int | float]] = {}
    for code, label in enumerate(labels):
        raw = values[codes == code]
        clean = raw[~np.isnan(raw)]
        if clean.size == 0:
            raise EmptyGroupError(
                f'После удаления пропусков в группе "{label}" не осталось наблюдений.'
            )
        if clean.size < 5:
            raise InsufficientDataError(
                f'Минимальное количество наблюдений в группе: 5. В группе "{label}": {clean.size}.'
            )
        arrays[label] = clean
        # Масштабирование защищает среднее и std от промежуточного переполнения.
        scale = max(float(np.max(np.abs(clean))), 1.0)
        scaled = clean / scale
        q1, median, q3 = np.quantile(scaled, [0.25, 0.5, 0.75]) * scale
        stats[label] = {
            "n": int(clean.size),
            "mean": float(scaled.mean() * scale),
            "median": float(median),
            "min": float(clean.min()),
            "max": float(clean.max()),
            "std": float(scaled.std(ddof=1) * scale),
            "q1": float(q1),
            "q3": float(q3),
            "dropped_rows": int(raw.size - clean.size),
        }
    if len(labels) < 2:
        raise InsufficientGroupsError("После очистки данных должно остаться минимум две группы.")
    if order is not None:
        if isinstance(order, (str, bytes)):
            raise InvalidParameterError("order должен содержать список всех групп без повторений.")
        try:
            requested = list(order)
            correct = len(requested) == len(labels) and set(requested) == set(labels)
        except (TypeError, ValueError) as exc:
            raise InvalidParameterError("order должен содержать хешируемые метки групп.") from exc
        if not correct:
            raise InvalidParameterError(
                "order должен содержать все наблюдаемые группы ровно по одному разу."
            )
        labels = requested

    valid_rows = sum(len(array) for array in arrays.values())
    dropped_rows = len(data) - valid_rows
    if dropped_rows:
        warnings.warn(
            f"Исключено строк с пропусками: {dropped_rows} из {len(data)}. "
            "Подробности находятся в summary.",
            MissingDataWarning,
            stacklevel=3,
        )
    small = [str(label) for label in labels if len(arrays[label]) < 20]
    if small:
        warnings.warn(
            "В группах " + ", ".join(small) + ": 5–19 наблюдений. "
            "Оценка плотности на малой выборке может быть неустойчивой.",
            SmallSampleWarning,
            stacklevel=3,
        )
    if len(labels) > 10:
        warnings.warn(
            "Больше 10 групп: подписи и распределения могут быть трудны для сравнения.",
            GroupCountWarning,
            stacklevel=3,
        )
    if len(data) > 100_000:
        warnings.warn(
            "Больше 100 000 строк: время построения выходит за целевой диапазон MVP.",
            LargeDatasetWarning,
            stacklevel=3,
        )
    return PreparedData(
        labels=labels,
        values=[arrays[label] for label in labels],
        summary={
            "total_rows": int(len(data)),
            "valid_rows": int(valid_rows),
            "dropped_rows": int(dropped_rows),
            "missing_group_rows": int(np.count_nonzero(codes == -1)),
            "groups": {label: stats[label] for label in labels},
        },
    )
