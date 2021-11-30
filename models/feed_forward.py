
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """
    - This consists of two linear transformations with a ReLU activation in between.
    - Another way of describing this is as two convolutions with kernel size 1.
    - FFN Equation:  FFN(x)=max(0,xW1+b1)W2+b2
    - Default: d_ff=2048 and d_model=512
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))




