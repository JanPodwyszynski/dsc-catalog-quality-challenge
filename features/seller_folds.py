import hashlib

import numpy as np


def seller_folds(sellers, n_splits=5, seed=42):
  assignment = np.array([
      int(hashlib.sha256(f'{seed}:{s}'.encode('utf-8')).hexdigest()[:16], 16)
      % n_splits
      for s in sellers.astype(str)
  ])
  for fold in range(n_splits):
    fit = np.flatnonzero(assignment != fold)
    valid = np.flatnonzero(assignment == fold)
    yield fit, valid