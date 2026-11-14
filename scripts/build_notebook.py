"""
Join notebook parts in notebooks/parts/*.ipynb into a single master capstone notebook.
Executes all cells and saves the output to notebooks/capstone.ipynb.

Fails loudly on any execution error.
Uses in-process execution to avoid ZeroMQ socket blocking in sandbox environments.
"""

import os
import glob
import sys
import io
import base64
import argparse
import traceback
import nbformat
from IPython.core.interactiveshell import InteractiveShell


class OutputCapture:
    """Helper to capture stdout, stderr, and rich display outputs for nbformat cells."""
    def __init__(self):
        self.stdout_io = io.StringIO()
        self.stderr_io = io.StringIO()
        self.display_outputs = []

    def display_hook(self, obj):
        data = {}
        if hasattr(obj, "_repr_png_"):
            png_bytes = obj._repr_png_()
            if png_bytes:
                data["image/png"] = base64.b64encode(png_bytes).decode("ascii") if isinstance(png_bytes, bytes) else png_bytes
        if hasattr(obj, "_repr_html_"):
            html = obj._repr_html_()
            if html:
                data["text/html"] = html
        if hasattr(obj, "_repr_markdown_"):
            md = obj._repr_markdown_()
            if md:
                data["text/markdown"] = md
        if not data:
            data["text/plain"] = repr(obj)

        self.display_outputs.append(
            nbformat.v4.new_output("display_data", data=data, metadata={})
        )


def execute_notebook_inprocess(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Execute cells in-process using IPython InteractiveShell, capturing all outputs."""
    shell = InteractiveShell.instance()
    proj_root = os.path.abspath(".")
    src_path = os.path.abspath("src")
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    shell.user_ns["__file__"] = os.path.abspath("notebooks/capstone.ipynb")

    exec_count = 1
    total_code = sum(1 for c in nb.cells if c.cell_type == "code")
    curr_code = 0

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        curr_code += 1
        source = cell.source.strip()
        if not source:
            continue

        print(f"Executing cell {curr_code}/{total_code}...", flush=True)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        capture = OutputCapture()
        sys.stdout = capture.stdout_io
        sys.stderr = capture.stderr_io

        try:
            from IPython.core import display_functions
            orig_display = display_functions.display

            def custom_display(*args, **kwargs):
                for arg in args:
                    capture.display_hook(arg)

            display_functions.display = custom_display

            res = shell.run_cell(source, store_history=False)

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            from IPython.core import display_functions
            display_functions.display = orig_display

        cell.execution_count = exec_count
        exec_count += 1
        outputs = []

        stdout_text = capture.stdout_io.getvalue()
        if stdout_text:
            outputs.append(
                nbformat.v4.new_output("stream", name="stdout", text=stdout_text)
            )

        stderr_text = capture.stderr_io.getvalue()
        if stderr_text:
            outputs.append(
                nbformat.v4.new_output("stream", name="stderr", text=stderr_text)
            )

        outputs.extend(capture.display_outputs)

        if res.result is not None:
            res_data = {}
            if hasattr(res.result, "_repr_html_"):
                res_data["text/html"] = res.result._repr_html_()
            if hasattr(res.result, "_repr_png_"):
                png = res.result._repr_png_()
                res_data["image/png"] = base64.b64encode(png).decode("ascii") if isinstance(png, bytes) else png
            res_data["text/plain"] = repr(res.result)

            outputs.append(
                nbformat.v4.new_output(
                    "execute_result",
                    execution_count=cell.execution_count,
                    data=res_data,
                    metadata={}
                )
            )

        if res.error_in_exec is not None:
            err = res.error_in_exec
            tb_list = traceback.format_exception(type(err), err, err.__traceback__)
            outputs.append(
                nbformat.v4.new_output(
                    "error",
                    ename=type(err).__name__,
                    evalue=str(err),
                    traceback=tb_list
                )
            )
            cell.outputs = outputs
            print(f"\nERROR in cell {curr_code}:\n{source}\n", file=sys.stderr)
            print("".join(tb_list), file=sys.stderr)
            raise err

        cell.outputs = outputs

    return nb


def build_and_run_notebook(
    parts_dir: str = "notebooks/parts",
    output_notebook: str = "notebooks/capstone.ipynb",
    execute: bool = True
) -> None:
    """Concatenate ordered notebook parts, execute them, and save to output_notebook."""
    part_files = sorted(glob.glob(os.path.join(parts_dir, "*.ipynb")))
    if not part_files:
        print(f"Warning: No notebook parts found in {parts_dir}.")
        return

    print(f"Found {len(part_files)} notebook parts in {parts_dir}:")
    for p in part_files:
        print(f"  - {os.path.basename(p)}")

    merged_nb = nbformat.v4.new_notebook()
    merged_nb.metadata = {
        "language_info": {"name": "python"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    }

    # Merge cells in alphabetical/numerical order
    for part_path in part_files:
        with open(part_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
            part_name = os.path.basename(part_path)
            merged_nb.cells.append(
                nbformat.v4.new_markdown_cell(f"<!-- PART START: {part_name} -->")
            )
            merged_nb.cells.extend(nb.cells)

    print(f"Merged total of {len(merged_nb.cells)} cells.")

    if execute:
        print("Executing merged notebook in-process...")
        execute_notebook_inprocess(merged_nb)
        print("Execution completed successfully without errors.")

    os.makedirs(os.path.dirname(output_notebook), exist_ok=True)
    with open(output_notebook, "w", encoding="utf-8") as f:
        nbformat.write(merged_nb, f)

    print(f"Successfully saved capstone notebook to {output_notebook}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble and execute capstone notebook.")
    parser.add_argument("--parts-dir", type=str, default="notebooks/parts", help="Directory with part notebooks.")
    parser.add_argument("--output", type=str, default="notebooks/capstone.ipynb", help="Output notebook path.")
    parser.add_argument("--no-exec", action="store_true", help="Join notebooks without executing.")
    args = parser.parse_args()

    build_and_run_notebook(
        parts_dir=args.parts_dir,
        output_notebook=args.output,
        execute=not args.no_exec
    )
