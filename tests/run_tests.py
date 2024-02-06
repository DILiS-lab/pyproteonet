from nbdev.test import test_nb
from pathlib import Path

test_nb(fn=Path('./data.ipynb'), do_print=True)
test_nb(fn=Path('./top3.ipynb'), do_print=True)
test_nb(fn=Path('./maxlfq.ipynb'), do_print=True)
test_nb(fn=Path('./ibaq.ipynb'), do_print=True)
test_nb(fn=Path('./reference_imputation_methods.ipynb'), do_print=True)
print("Done! All tests were run!")