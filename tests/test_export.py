from xml.etree import ElementTree

import pytest

from rainview import InvalidParameterError, raincloudplot, save_plot


@pytest.mark.parametrize("extension", ["png", "svg", "PNG"])
def test_actual_export(frame, tmp_path, extension):
    fig, _, _ = raincloudplot(frame, "score", "method")
    target = tmp_path / f"plot.{extension}"
    assert save_plot(fig, target) == target
    assert target.stat().st_size > 1000
    if extension.lower() == "png":
        assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    else:
        assert ElementTree.parse(target).getroot().tag.endswith("svg")


@pytest.mark.parametrize("suffix", [".pdf", ".txt", ""])
def test_unsupported_export_format(frame, tmp_path, suffix):
    fig, _, _ = raincloudplot(frame, "score", "method")
    with pytest.raises(InvalidParameterError, match="PNG и SVG"):
        save_plot(fig, tmp_path / f"plot{suffix}")


@pytest.mark.parametrize("dpi", [0, -10, 1.5, True])
def test_invalid_dpi(frame, tmp_path, dpi):
    fig, _, _ = raincloudplot(frame, "score", "method")
    with pytest.raises(InvalidParameterError, match="dpi"):
        save_plot(fig, tmp_path / "plot.png", dpi=dpi)


def test_plotting_does_not_implicitly_save(frame, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raincloudplot(frame, "score", "method")
    assert list(tmp_path.iterdir()) == []
