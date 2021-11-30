import numpy as np

import torch
from torch.autograd import Variable

from train import Batch


# Synthetic Data
def data_gen(V, batch, nbatches):
    """
    - Generate random data for a src-tgt copy task.
    - Given a random set of input symbols from a small vocabulary, the goal is to generate back 
    those same symbols.
    """
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

