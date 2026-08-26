import math

import numpy as np
from sklearn.model_selection import train_test_split


def train_validation_test_masks(
    labels,
    train_fraction,
    validation_fraction,
    test_fraction,
    seed,
):
    """Create disjoint train, validation, and test masks.

    Fractions must sum to one. For these unsupervised
    detectors, train labels select the score threshold while fitting still
    uses the complete graph without labels.
    """
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("train, validation, and test fractions must sum to 1")
    train_size = max(
        len(np.unique(labels)),
        math.ceil(train_fraction * len(labels)),
    )
    train, remaining = train_test_split(
        indices,
        train_size=train_size,
        random_state=seed,
        stratify=labels,
    )
    validation, test = train_test_split(
        remaining,
        train_size=validation_fraction / (1 - train_fraction),
        random_state=seed + 10000,
        stratify=labels[remaining],
    )
    train_mask = np.zeros(len(labels), dtype=bool)
    validation_mask = np.zeros(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)
    train_mask[train] = True
    validation_mask[validation] = True
    test_mask[test] = True
    return train_mask, validation_mask, test_mask
