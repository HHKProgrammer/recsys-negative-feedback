# pipeline utility — reproducibility
# sets a fixed random seed so every run produces the exact same results
# without this, svd init and candidate sampling would differ each run
# and i couldnt tell if a metric change came from my method or just random luck

# Global random seed control for reproducibility. so svd can initialize weights randomly  np uses 42

import random
import numpy as np


def set_global_seed(seed: int) -> None:
    # fix python random and numpy random at the same time
    # important — surprise uses numpy internally so both need to be seeded
    # seed=42 is the standard in ml research, used everywhere in this project
    random.seed(seed)
    np.random.seed(seed)
