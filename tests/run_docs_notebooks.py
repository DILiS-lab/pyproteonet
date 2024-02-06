from nbdev.test import test_nb
from pathlib import Path

test_nb(fn=Path('../docs/source/notebooks/getting_started.ipynb'), do_print=True)
test_nb(fn=Path('../docs/source/notebooks/simulation.ipynb'), do_print=True)
test_nb(fn=Path('../docs/source/notebooks/evaluate_imputation_abundance.ipynb'), do_print=True)
test_nb(fn=Path('../docs/source/notebooks/evaluate_imputation_fold_change.ipynb'), do_print=True)
test_nb(fn=Path('../docs/source/notebooks/imputation_method_development.ipynb'), do_print=True)
print("Done! All documentation notebooks were run!")