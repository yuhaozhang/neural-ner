"""
A trainer class to handle training and testing of models.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable

from model import model, loss
from data import loader
from utils import constant, torch_utils

class Trainer(object):
    """ A trainer for training models. """
    def __init__(self, opt, emb_matrix=None, joint=False):
        self.opt = opt
        self.model = model.BLSTM_CRF(opt, emb_matrix)
        self.crit = loss.SequenceLoss(opt['num_class'])
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.crit.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        fsize = loader.INPUT_SIZE
        if self.opt['cuda']:
            inputs = [Variable(b.cuda()) if b is not None else None for b in batch[:fsize]]
            labels = Variable(batch[fsize].cuda())
        else:
            inputs = [Variable(b) if b is not None else None for b in batch[:fsize]]
            labels = Variable(batch[fsize])

        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        logits_flat = logits.view(-1, logits.size()[-1])
        loss = self.crit(logits_flat, labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data[0]
        return loss_val

    def predict(self, batch, unsort=True):
        fsize = loader.INPUT_SIZE
        if self.opt['cuda']:
            inputs = [Variable(b.cuda()) if b is not None else None for b in batch[:fsize]]
            labels = Variable(batch[fsize].cuda())
        else:
            inputs = [Variable(b) if b is not None else None for b in batch[:fsize]]
            labels = Variable(batch[fsize])

        orig_idx = batch[-1]
        self.model.eval()
        batch_size = inputs[0].size(0)
        logits = self.model(inputs)
        logits_flat = logits.view(-1, logits.size()[-1])
        loss = self.crit(logits_flat, labels.view(-1))
        predictions = np.argmax(logits_flat.data.cpu().numpy(), axis=1)\
                .reshape([batch_size, -1]).tolist()
        lens = list(inputs[1].data.eq(constant.PAD_ID).long().sum(1).squeeze())
        predictions = [p[:l] for l,p in zip(lens, predictions)] # remove paddings
        if unsort:
            _, predictions = [list(t) for t in zip(*sorted(zip(orig_idx, predictions)))]
        return predictions, loss.data[0]

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")
    
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']


