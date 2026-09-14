"""Script to generate and execute analysis_malliavin_vs_FDMC.ipynb using nbclient."""

import sys
import nbformat as nbf
from nbclient import NotebookClient
from generate_notebook import build_notebook

def main():
    print("Building notebook structure...")
    nb = build_notebook()

    print("Executing notebook cells with NotebookClient...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    out_path = "analysis_malliavin_vs_FDMC.ipynb"
    print(f"Saving executed notebook to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("Execution complete! Notebook saved successfully.")

if __name__ == "__main__":
    main()
