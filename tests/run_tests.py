from nbdev.test import test_nb
from pathlib import Path

test_nb(fn=Path('./sim_gnn_peptide_impute.ipynb'))
print("Done! All tests were run!")