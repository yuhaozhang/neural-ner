"""
A trainer class to handle training and testing of models.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

from model import model, loss, crf
from data import loader
from utils import constant, torch_utils

def unpack_batch(batch, cuda):
    fsize = loader.INPUT_SIZE
    if cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:fsize]]
        labels = batch[fsize].cuda()
    else:
        inputs = [b if b is not None else None for b in batch[:fsize]]
        labels = batch[fsize]
    masks = inputs[1]
    orig_idx = batch[-1]
    return inputs, labels, masks, orig_idx

class Trainer(object):
    """ A trainer for training models. """
    def __init__(self, opt, emb_matrix=None, joint=False):
        self.opt = opt
        self.model = model.BLSTM_CRF(opt, emb_matrix)
        if opt['crf']:
            print("Using CRF loss...")
            self.crit = crf.CRFLoss(opt['num_class'], True)
        else:
            self.crit = loss.SequenceLoss(opt['num_class'])
        self.parameters = [p for m in (self.model, self.crit) for p in m.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.crit.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'], opt.get('momentum', 0))

    def update(self, batch):
        inputs, labels, masks, orig_idx = unpack_batch(batch, self.opt['cuda'])
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        if self.opt['crf']:
            loss, _ = self.crit(logits, masks, labels)
        else:
            logits_flat = logits.view(-1, logits.size(-1))
            loss = self.crit(logits_flat, labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, labels, masks, orig_idx = unpack_batch(batch, self.opt['cuda'])
        self.model.eval()
        batch_size = inputs[0].size(0)
        logits = self.model(inputs)
        lens = list(masks.data.eq(0).long().sum(1).squeeze())
        if self.opt['crf']:
            loss, trans = self.crit(logits, masks, labels)
            predictions = []
            trans = trans.data.cpu().numpy()
            scores = logits.data.cpu().numpy()
            for i in range(batch_size):
                tags, _ = crf.viterbi_decode(scores[i,:lens[i]], trans)
                predictions += [tags]
        else:
            logits_flat = logits.view(-1, logits.size(-1))
            loss = self.crit(logits_flat, labels.view(-1))
            predictions = np.argmax(logits_flat.data.cpu().numpy(), axis=1)\
                .reshape([batch_size, -1]).tolist()
            predictions = [p[:l] for l,p in zip(lens, predictions)] # remove paddings
        if unsort:
            _, predictions = [list(t) for t in zip(*sorted(zip(orig_idx, predictions)))]
        return predictions, loss.data.item()

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'crit': self.crit.state_dict(),
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
        if 'crit' in checkpoint:
            self.crit.load_state_dict(checkpoint['crit'])
        self.opt = checkpoint['config']


