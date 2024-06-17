"""A tool to help clean up notebooks for production.

Run with !python3 ../testing/file_formatter.py
"""

import glob
import json

DEFAULT_PATH = "**/*.ipynb"


def clean_notebooks(hide_solutions=True, clear_outputs=True, glob_path=DEFAULT_PATH):
    """Test that code in notebooks run."""
    notebook_files = glob.glob(glob_path, recursive=True)
    for file in notebook_files:
        altered_notebook = False
        print("Cleaning " + file)
        # Get code cells from notebooks
        with open(file, "r") as in_f:
            notebook_json = json.load(in_f)
            test_cells = []
            for cell_idx, cell in enumerate(notebook_json["cells"]):

                def print_update(updated_cell, change, sample_line_idx=0):
                    sample_line = updated_cell["source"][sample_line_idx]
                    print(
                        change
                        + str(cell_idx)
                        + " starting with:    "
                        + sample_line
                    )
                    notebook_json["cells"][cell_idx] = updated_cell

                if hide_solutions:
                    hid_cell = hide_cell(cell)
                    if hid_cell:
                        print_update(hid_cell, "Hid Cell ", sample_line_idx=1)
                if clear_outputs:
                    cleared_output_cell = clear_cell_output(cell)
                    if cleared_output_cell:
                        print_update(cleared_output_cell, "Cleared Outputs ")
                altered_notebook = hid_cell or cleared_output_cell or altered_notebook
        if altered_notebook:
            with open(file, "w") as out_f:
                print("Saving altered notebook " + file)
                json.dump(notebook_json, out_f, indent=1)


def hide_cell(cell):
    first_line = cell["source"][0]
    altered_cell = None
    if "# SOLUTION" in first_line:
        if "jupyter" not in cell["metadata"]:
            cell["metadata"]["jupyter"] = {}
        if not cell["metadata"]["jupyter"].get("source_hidden"):
            cell["metadata"]["jupyter"]["source_hidden"] = True
            altered_cell = cell
    return altered_cell


def clear_cell_output(cell):
    altered_cell = None
    if cell.get("outputs") or cell.get("execution_count"):
        cell["outputs"] = []
        cell["execution_count"] = None
        altered_cell = cell
    return altered_cell


if __name__ == "__main__":
    clean_notebooks()

