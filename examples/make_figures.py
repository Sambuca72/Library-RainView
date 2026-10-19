"""Запуск из корня: python examples/make_figures.py --output outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rainview import format_summary, raincloudplot, save_plot

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Демонстрация RainView на учебных данных")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "data" / "education.csv")
    fig, _, summary = raincloudplot(
        data,
        "score",
        "method",
        title="Распределение оценок по формату обучения",
        x_label="Формат обучения",
        y_label="Оценка, баллы",
        palette=["#3D9C9C", "#D88B51", "#7D81B8"],
        figsize=(9, 5.8),
        point_size=19,
        point_alpha=0.65,
    )
    for extension in ("png", "svg"):
        save_plot(fig, args.output / f"education.{extension}")
    plt.close(fig)

    fig, _, _ = raincloudplot(
        data,
        "score",
        "method",
        orientation="horizontal",
        order=["Смешанный", "Интерактивный", "Традиционный"],
        title="Те же данные в горизонтальной ориентации",
        x_label="Оценка, баллы",
        y_label="Формат обучения",
        palette={"Традиционный": "#3D9C9C", "Интерактивный": "#D88B51", "Смешанный": "#7D81B8"},
        figsize=(9, 5.8),
        point_size=19,
    )
    for extension in ("png", "svg"):
        save_plot(fig, args.output / f"education_horizontal.{extension}")
    plt.close(fig)

    text = format_summary(summary)
    (args.output / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nГрафики и сводка сохранены: {args.output.resolve()}")


if __name__ == "__main__":
    main()
