"""
Tagging models.
"""
import torch
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable

from model import layers
from utils import constant, torch_utils

class BLSTM_CRF(nn.Module):
    """A sequence tagging model."""

    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        # input dropout layer
        self.drop = nn.Dropout(opt['dropout'])
        # word embedding matrix
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=0)
        self.char_layer = layers.CharLayer(opt, type='rnn')

        input_size = opt['emb_dim'] + opt['char_hidden_dim']*2
        self.lstm = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True, \
                dropout=opt['dropout'], bidirectional=True)
        self.linear = nn.Linear(opt['hidden_dim']*2, opt['num_class'])
        if opt['crf']:
            # consider make a NN layer?
            self.transition_matrix = torch.rand(opt['num_class'], opt['num_class'])
        
        # keep parameters
        self.crf = opt['crf']
        self.use_cuda = opt['cuda']
        self.opt = opt
        self.emb_matrix = emb_matrix

        self.init_weights()

    def init_weights(self):
        if self.emb_matrix is not None:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        else:
            self.emb.weight.data[1:,:].uniform_(-1.0,1.0)
        self.linear.bias.data.fill_(0.0)
        # use xavier to initialize linear layer
        init.xavier_uniform(self.linear.weight, gain=1)

    def forward(self, inputs):
        words, masks, chars = inputs
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        batch_size = words.size()[0]
        rnn_inputs = self.drop(self.emb(words))
        
        # get character hidden
        char_inputs = self.char_layer(chars)
        rnn_inputs = torch.cat([rnn_inputs, char_inputs], dim=2)

        h0, c0 = zero_state(batch_size, self.opt['hidden_dim'], self.opt['num_layers'], True, self.use_cuda)
        # pack sequence
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.lstm(rnn_inputs, (h0, c0))
        rnn_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        #rnn_outputs = self.drop(rnn_outputs)

        # ht: [num_layers * num_dir, B, H]
        # outputs: [B, T, H * num_dir]
        logits = self.linear(rnn_outputs.contiguous().view(-1, self.opt['hidden_dim']*2))
        logits = logits.view(batch_size, -1, self.opt['num_class'])
        return logits

def zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

