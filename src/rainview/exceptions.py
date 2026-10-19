"""Понятные ошибки входных данных и предупреждения RainView."""


class RainViewError(ValueError):
    """Базовый класс ошибок RainView."""


class InvalidDataError(RainViewError):
    """Неверный формат таблицы или недопустимые значения."""


class ColumnNotFoundError(RainViewError):
    """В таблице нет указанного столбца."""


class NonNumericColumnError(RainViewError):
    """Столбец измерений не содержит вещественные числа."""


class EmptyGroupError(RainViewError):
    """Все измерения в наблюдаемой группе пропущены."""


class InsufficientDataError(RainViewError):
    """В группе меньше пяти измерений после очистки."""


class InsufficientGroupsError(RainViewError):
    """Осталось меньше двух наблюдаемых групп."""


class InvalidParameterError(RainViewError):
    """Неверный параметр отображения или экспорта."""


class RainViewWarning(UserWarning):
    """Базовый класс предупреждений RainView."""


class SmallSampleWarning(RainViewWarning):
    """При 5–19 наблюдениях оценку плотности нужно читать осторожно."""


class MissingDataWarning(RainViewWarning):
    """Строки с пропусками исключены из графика и статистики."""


class ConstantGroupWarning(RainViewWarning):
    """Оценка плотности для постоянной группы заменена линией."""


class GroupCountWarning(RainViewWarning):
    """Больше десяти групп может быть трудно сравнивать."""


class LargeDatasetWarning(RainViewWarning):
    """Размер таблицы выходит за целевой диапазон MVP."""
