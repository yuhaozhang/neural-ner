"""
Main training code for tagging.
"""
from datetime import datetime
import sys
import os
import shutil
import time
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data.loader import DataLoader
from model.trainer import Trainer
from utils import scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/conll_iob')
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab_iob')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=100, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=False)
parser.add_argument('--crf', action='store_true')
parser.add_argument('--scheme', type=str, default='iob', help='Either iob or iobes.')

parser.add_argument('--char_emb_dim', type=int, default=20, help='Char embedding dimension.')
parser.add_argument('--char_hidden_dim', type=int, default=50, help='Char layer hidden dimension.')

parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=25)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_best_only', dest='save_best_only', action='store_true', help='Save best model only.')
parser.set_defaults(save_best_only=True)
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# make opt
opt = vars(args)

label2id = constant.TYPE_TO_ID_IOB
opt['num_class'] = len(label2id)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']

char_vocab_file = opt['vocab_dir'] + '/vocab_char.pkl'
char_vocab = Vocab(char_vocab_file, load=True)
opt['char_vocab_size'] = char_vocab.size

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, char_vocab, evaluation=False)
dev_batch = DataLoader(opt['data_dir'] + '/testa.json', opt['batch_size'], opt, vocab, char_vocab, evaluation=True)

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_f1\tbest_dev_f1")

# print model info
helper.print_config(opt)

trainer = Trainer(opt, emb_matrix=emb_matrix)

id2label = dict([(v,k) for k,v in label2id.items()])
dev_f1_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = trainer.update(batch)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    dev_loss = 0
    for i, batch in enumerate(dev_batch):
        preds, loss = trainer.predict(batch)
        predictions += preds
        dev_loss += loss
    predictions = [[id2label[p] for p in ps] for ps in predictions]
    dev_p, dev_r, dev_f1 = scorer.score_by_chunk(dev_batch.gold(), predictions)
    
    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
            train_loss, dev_loss, dev_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1, \
            max([dev_f1] + dev_f1_history)))

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file, epoch)
    if epoch == 1 or dev_f1 > max(dev_f1_history):
        shutil.copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
    if opt['save_best_only'] or epoch % opt['save_epoch'] != 0:
        os.remove(model_file)
    
    # lr schedule
    if len(dev_f1_history) > 30 and dev_f1 <= dev_f1_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_f1_history += [dev_f1]
    print("")

print("Training ended with {} epochs.".format(epoch))


