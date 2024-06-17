"""A tool to help unit test notebooks.

Runs through all cells in a notebook.
These tests require an active Jupyter environment.
Add "# TODO" or "# EXAMPLE" at the start of a code cell to skip running it.
Add "# SOLUTION" to a cell to automatically hide the cell from students.

Run with !python3 ../testing/test_nbs.py
"""

import glob
import json
from testbook import testbook

DEFAULT_PATH = "**/*.ipynb"

def test_basic_execution(glob_path=DEFAULT_PATH):
    """Test that code in notebooks run."""
    print("Testing code for the notebook files:")
    notebook_files = sorted(glob.glob(glob_path, recursive=True))
    for file in notebook_files:
        altered_notebook=False
        print("Testing notebook code for " + file)
        # Get code cells from notebooks
        with open(file, 'r') as in_f:
            notebook_json = json.load(in_f)
            test_cells = []
            for cell_idx, cell in enumerate(notebook_json["cells"]):
                if cell["cell_type"] == "code" and not skip_execution(cell):
                    test_cells.append(cell_idx)
            # Skip kernel shutdown
            test_cells.pop()
        print("Testing cells:" + str(test_cells))

        @testbook(file, execute=test_cells, timeout=300)
        def test_single_notebook(tb):
            # Just verify
            assert True

        test_single_notebook()
        print(file + ": passed")


def skip_execution(contents):
    """If first line has # TODO, do not run."""
    first_line = contents["source"][0]
    has_todo = first_line.startswith("# TODO")
    has_example = first_line.startswith("# EXAMPLE")
    return has_todo or has_example


def run_test(glob_path=DEFAULT_PATH):
    test_basic_execution(glob_path=glob_path)
    print("tests pass!")


if __name__ == "__main__":
    run_test()

