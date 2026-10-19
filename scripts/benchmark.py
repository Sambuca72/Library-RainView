"""Измерение полного построения, включая отрисовку canvas, без порога в unit-тестах."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

from rainview import raincloudplot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("verification/benchmark.json"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats должен быть положительным")
    rng = np.random.default_rng(42)
    results = []
    for rows, groups in ((10_000, 3), (10_000, 10), (100_000, 3)):
        frame = pd.DataFrame(
            {
                "value": rng.normal(size=rows),
                "group": np.arange(rows) % groups,
            }
        )
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            fig, _, summary = raincloudplot(frame, "value", "group")
            fig.canvas.draw()
            durations.append(round(time.perf_counter() - start, 6))
            assert summary["valid_rows"] == rows
            plt.close(fig)
        results.append(
            {
                "rows": rows,
                "groups": groups,
                "seconds": durations,
                "median_seconds": statistics.median(durations),
                "max_seconds": max(durations),
                "target_seconds": 5 if rows == 10_000 else None,
                "target_met": max(durations) <= 5 if rows == 10_000 else None,
            }
        )
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "scipy": scipy.__version__,
        },
        "measurement": "raincloudplot + fig.canvas.draw; imports excluded; all points retained",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
