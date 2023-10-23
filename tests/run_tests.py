from nbdev.test import test_nb
from pathlib import Path

test_nb(fn=Path('./sim_gnn_peptide_impute.ipynb'))
test_nb(fn=Path('./data.ipynb'))
test_nb(fn=Path('./top3.ipynb'))
test_nb(fn=Path('./maxlfq.ipynb'))
print("Done! All tests were run!")