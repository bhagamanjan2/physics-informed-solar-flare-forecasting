"""
Execute notebooks/capstone.ipynb in-process using IPython.
Saves matplotlib plots and stream outputs directly back into the notebook file.
"""
import os, sys, json, traceback, io, base64
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output as ipy_capture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(ROOT, 'notebooks', 'capstone.ipynb')

os.chdir(os.path.join(ROOT, 'notebooks'))

with open(NB_PATH, encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

shell = InteractiveShell.instance()
shell.run_cell("import os; os.chdir(r'%s')" % os.path.join(ROOT, 'notebooks'))
shell.run_cell("import warnings; warnings.filterwarnings('ignore')")

errors = []
for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    
    source = cell.source.strip()
    if not source:
        continue

    print(f"  Cell {i:2d}: {source[:60].replace(chr(10), ' ')!r}...")
    
    outputs = []
    stdout_lines = []
    stderr_lines = []
    
    # Ensure no lingering figures from previous cell
    plt.close('all')
    
    try:
        with ipy_capture() as cap:
            result = shell.run_cell(source, store_history=False)
        
        if cap.stdout:
            stdout_lines.append(cap.stdout)
        
        # Collect display objects
        for obj in cap.outputs:
            if hasattr(obj, 'data'):
                data = obj.data
                metadata = getattr(obj, 'metadata', {})
                if 'image/png' in data:
                    outputs.append(nbformat.v4.new_output(
                        output_type='display_data',
                        data={'image/png': data['image/png']},
                        metadata=metadata or {}
                    ))
                elif 'text/html' in data:
                    outputs.append(nbformat.v4.new_output(
                        output_type='display_data',
                        data=data,
                        metadata=metadata or {}
                    ))
        
        # Collect any open matplotlib figures created in this cell
        fignums = plt.get_fignums()
        for fignum in fignums:
            fig = plt.figure(fignum)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('ascii')
            outputs.append(nbformat.v4.new_output(
                output_type='display_data',
                data={
                    'image/png': img_b64,
                    'text/plain': '<Figure size %dx%d with %d Axes>' % (
                        fig.get_size_inches()[0]*150, fig.get_size_inches()[1]*150, len(fig.axes)
                    )
                },
                metadata={}
            ))
        plt.close('all')
        
        if result.error_in_exec:
            err_text = str(result.error_in_exec)
            stderr_lines.append(err_text)
            errors.append(f"Cell {i}: {err_text[:200]}")
            print(f"    ERROR: {err_text[:120]}")
        
    except Exception as e:
        err_text = traceback.format_exc()
        stderr_lines.append(err_text)
        errors.append(f"Cell {i} exception: {str(e)[:200]}")
        print(f"    EXCEPTION: {str(e)[:120]}")
    
    # Combined stdout
    combined_stdout = ''.join(stdout_lines)
    if combined_stdout.strip():
        outputs.insert(0, nbformat.v4.new_output(
            output_type='stream',
            name='stdout',
            text=combined_stdout
        ))
    
    if stderr_lines:
        outputs.append(nbformat.v4.new_output(
            output_type='stream',
            name='stderr',
            text=''.join(stderr_lines)
        ))
    
    cell.outputs = outputs
    cell.execution_count = i + 1

with open(NB_PATH, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"\nNotebook saved: {NB_PATH}")
if errors:
    print(f"\nErrors ({len(errors)}):")
    for e in errors:
        print(f"  {e}")
else:
    print("All cells executed successfully!")
