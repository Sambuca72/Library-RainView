import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def frame():
    rng = np.random.default_rng(17)
    return pd.DataFrame(
        {
            "score": np.r_[rng.normal(60, 9, 30), rng.normal(72, 7, 30)],
            "method": ["A"] * 30 + ["B"] * 30,
        }
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
