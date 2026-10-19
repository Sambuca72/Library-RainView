"""Повторное выполнение всех трёх примеров в текущем Python-окружении."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # KernelSpec только во временной папке: пользовательское Jupyter-окружение не меняется.
    with tempfile.TemporaryDirectory(prefix="rainview-kernel-") as temporary:
        kernel_dir = Path(temporary) / "rainview"
        kernel_dir.mkdir()
        spec = {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "RainView verification",
            "language": "python",
            "env": {
                "PYTHONPATH": str(ROOT / "src"),
                "MPLCONFIGDIR": temporary,
                "MPLBACKEND": "module://matplotlib_inline.backend_inline",
                "IPYTHONDIR": str(Path(temporary) / "ipython"),
            },
        }
        import json

        (kernel_dir / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")
        manager = KernelSpecManager(kernel_dirs=[temporary], ensure_native_kernel=False)
        for path in sorted((ROOT / "examples").glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            client = NotebookClient(
                notebook,
                timeout=120,
                kernel_name="rainview",
                resources={"metadata": {"path": str(ROOT)}},
            )
            client.create_kernel_manager()
            client.km.kernel_spec_manager = manager
            client.execute(env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
            nbformat.write(notebook, path)
            print(f"Выполнен {path.name}", flush=True)


if __name__ == "__main__":
    main()
