"""
Neural layers.
"""

import torch
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable

from utils import constant, torch_utils

class CharRNNLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.use_cuda = opt['cuda']
        self.emb = nn.Embedding(opt['char_vocab_size'], opt['char_emb_dim'], padding_idx=0)
        self.drop = nn.Dropout(opt['dropout'])
        self.rnn = nn.LSTM(opt['char_emb_dim'], opt['char_hidden_dim'], 1, batch_first=True, dropout=opt['dropout'], bidirectional=True)
        self.emb.weight.data[1:,:].uniform_(-1.0,1.0)
        self.out_dim = opt['char_hidden_dim']*2

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
        b_flat = inputs.size()[0]
        init_state = lstm_zero_state(b_flat, h, 1, use_cuda=self.use_cuda)
        outputs, (ht, ct) = self.rnn(inputs, init_state)
        hidden = ht[-2:].transpose(0,1).contiguous().view(b_flat, -1)
        # unflat the hidden representations
        hidden = hidden.view(b, l, h*2)
        return hidden

class CharCNNLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.use_cuda = opt['cuda']
        self.emb = nn.Embedding(opt['char_vocab_size'], opt['char_emb_dim'], padding_idx=0)
        self.drop = nn.Dropout(opt['dropout'])
        self.emb.weight.data[1:,:].uniform_(-1.0,1.0)
        self.init_conv_layers()
        if opt['char_cnn_dim'] > 0:
            self.linear = nn.Linear(self.cnn_output_dim, opt['char_cnn_dim'])
            self.out_dim = opt['char_cnn_dim']
        else: # do not apply linear
            self.out_dim = self.cnn_output_dim

    def init_conv_layers(self):
        conv_layers = []
        for k in range(self.opt['char_fmin'], self.opt['char_fmax']+1):
            insize, outsize = self.opt['char_emb_dim'], self.opt['char_fsize']
            layer = nn.Sequential(
                        nn.Conv1d(insize, outsize, k, padding=1),
                        nn.ReLU()
                    )
            conv_layers += [layer]
        self.conv_layers = nn.ModuleList(conv_layers)
        self.fsize = self.opt['char_fsize']
        self.fnum = self.opt['char_fmax']+1 - self.opt['char_fmin']
        self.cnn_output_dim = self.fsize * self.fnum
    
    def forward(self, chars):
        """
        chars: LongTensor, B x L x Lc
        """
        b, l, lc = chars.size()
        chars_flat = chars.view(-1, lc)
        chars_emb = self.drop(self.emb(chars_flat))
        d = chars_emb.size()[-1]
        inputs = chars_emb
        inputs = inputs.transpose(1,2) # b x d x lc
        b_flat = inputs.size()[0] # new batch_size after flat
        # do cnn
        cnn_outputs = []
        for layer in self.conv_layers:
            outputs = layer(inputs)
            outputs, _ = outputs.max(dim=2) # max pooling over time
            cnn_outputs += [outputs]
        cnn_outputs = torch.cat(cnn_outputs, dim=1)
        hidden = self.linear(cnn_outputs) if self.opt['char_cnn_dim'] > 0 else cnn_outputs
        # unflat the hidden representations
        hidden = hidden.view(b, l, -1)
        return hidden

def lstm_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

