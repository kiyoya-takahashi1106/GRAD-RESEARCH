import torch
import torch.nn.functional as F

import os
import random
import numpy as np


def set_seed(seed :int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def conpute_similarity(a: torch.Tensor, b: torch.Tensor):
    """
    a: (batch_size, dim)
    b: (batch_size, dim)
    """
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    sim = torch.sum(a_norm * b_norm, dim=-1)
    sim = sim.mean()
    return sim