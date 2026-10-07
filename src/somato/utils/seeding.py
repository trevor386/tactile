from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    """Seed python, numpy and torch. Returns a torch generator seeded with ``seed``."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
