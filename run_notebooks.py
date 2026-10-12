"""Execute the analysis notebooks from their jupytext sources.

The source of truth for each notebook is ``notebooks/<name>.py`` (jupytext percent format: plain
Python with ``# %%`` cell markers). This script executes it from the repository root (so
``./results`` and the project modules resolve) and writes the executed notebook to
``<name>.ipynb``, paired with the source. With the jupytext Jupyter extension, or
``jupytext --sync <name>.ipynb``, edits made in either file are copied to the other.

Usage:
    python run_notebooks.py                      # all notebooks
    python run_notebooks.py analysis_FDMC ...    # selected notebooks
"""

import sys
import time
from pathlib import Path

import jupytext
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "notebooks"
PAIRING = "ipynb,notebooks//py:percent"


def run_notebook(name: str, timeout: int = 1800) -> None:
    nb = jupytext.read(SOURCES / f"{name}.py")
    nb.metadata.setdefault("jupytext", {})["formats"] = PAIRING
    t0 = time.perf_counter()
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    jupytext.write(nb, ROOT / f"{name}.ipynb")
    print(f"{name}: executed in {time.perf_counter() - t0:.0f} s")


def main(names: list[str]) -> None:
    names = names or sorted(p.stem for p in SOURCES.glob("*.py"))
    for name in names:
        run_notebook(name)


if __name__ == "__main__":
    main(sys.argv[1:])
