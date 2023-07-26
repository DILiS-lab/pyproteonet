import numpy as np

def eq_nan(a,b):
    return (a == b) | (np.isnan(a) & np.isnan(b))