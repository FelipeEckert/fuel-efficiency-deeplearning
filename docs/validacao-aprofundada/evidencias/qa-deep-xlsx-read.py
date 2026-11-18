from pathlib import Path
from openpyxl import load_workbook
import os
root=Path(os.environ['TEMP'])/'fuel-ui-20261118'
for f in sorted(root.glob('qa-deep-xlsx-export-*.xlsx')):
    w=load_workbook(f)
    data=[[(c.value,c.data_type) for c in row[:3]] for row in w.active.iter_rows(min_row=2)]
    print(f.stem, data)
