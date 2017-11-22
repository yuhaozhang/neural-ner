"""
Neural layers.
"""

import torch
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable

from utils import constant, torch_utils

class CharLayer(nn.Module):
    def __init__(self, opt, type='rnn'):
        super().__init__()
        self.opt = opt
        self.type = type
        self.use_cuda = opt['cuda']

        self.emb = nn.Embedding(opt['char_vocab_size'], opt['char_emb_dim'], padding_idx=0)
        self.drop = nn.Dropout(opt['dropout'])
        if type == 'rnn':
            self.rnn = nn.LSTM(opt['char_emb_dim'], opt['char_hidden_dim'], 1, batch_first=True, dropout=opt['dropout'], bidirectional=True)
        elif type == 'cnn':
            raise Exception("Not implemented.")
        else:
            raise Exception("Char model type {} not supported.".format(type))
        self.init_weights()

    def init_weights(self):
        self.emb.weight.data[1:,:].uniform_(-1.0,1.0)

    def forward(self, chars):
        """
        chars: LongTensor, B x L x Lc
        """
        b, l, lc = chars.size()
        chars_flat = chars.view(-1, lc)
        chars_emb = self.drop(self.emb(chars_flat))
        d = chars_emb.size()[-1]
        h = self.opt['char_hidden_dim']
        inputs = chars_emb
        if self.type == 'rnn':
            b_flat = inputs.size()[0]
            init_state = lstm_zero_state(b_flat, h, 1, use_cuda=self.use_cuda)
            outputs, (ht, ct) = self.rnn(inputs, init_state)
            hidden = ht[-2:].transpose(0,1).contiguous().view(b_flat, -1)
        elif self.type == 'cnn':
            hidden = None
        # unflat the hidden representations
        hidden = hidden.view(b, l, h*2)
        return hidden

def lstm_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

