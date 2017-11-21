"""
Tagging loss.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable

from utils import constant

def SequenceLoss(vocab_size):
    weight = torch.ones(vocab_size)
    weight[constant.PAD_TYPE_ID] = 0
    crit = nn.CrossEntropyLoss(weight)
    return crit

