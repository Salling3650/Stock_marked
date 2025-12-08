#!/usr/bin/env python3
"""Git filter to strip Jupyter notebook outputs"""
import json
import sys

# Read notebook from stdin
notebook = json.load(sys.stdin)

# Clear outputs from all cells
for cell in notebook.get('cells', []):
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# Write to stdout
json.dump(notebook, sys.stdout, indent=1, ensure_ascii=False)
sys.stdout.write('\n')
