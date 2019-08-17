import numpy as np


def _read_feature(cell, idx):
    assert isinstance(cell, (list, tuple))
    out = cell[idx]
    return out


read_feature = np.vectorize(_read_feature, excluded=['idx'])

