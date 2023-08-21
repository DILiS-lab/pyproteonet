from typing import Optional, Union

import numpy as np

def get_numpy_random_generator(seed: Optional[Union[np.random.Generator, int]]):
    if isinstance(seed, np.random.Generator):
        return seed
    else:
        return np.random.default_rng(seed=seed)