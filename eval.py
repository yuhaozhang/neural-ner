"""
Run evaluation with saved models.
"""

import os
import random
import argparse
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from data.loader import DataLoader
from model.trainer import Trainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/iob')
parser.add_argument('--dataset', type=str, default='testb', help="Evaluate on dev or test.")
parser.add_argument('--out', type=str, default='', help="Save model outputs along with correct tags into a file.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
trainer = Trainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

char_vocab_file = args.model_dir + '/vocab_char.pkl'
char_vocab = Vocab(char_vocab_file, load=True)
assert opt['char_vocab_size'] == char_vocab.size, "Char vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.jsonl'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, char_vocab, evaluation=True)

helper.print_config(opt)
label2id = constant.TYPE_TO_ID_IOB
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
for i, b in enumerate(tqdm(batch)):
    preds, _ = trainer.predict(b)
    predictions += preds
predictions = [[id2label[p] for p in ps] for ps in predictions]
p, r, f1 = scorer.score_by_chunk(batch.gold(), predictions, scheme=opt['scheme'])
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p*100,r*100,f1*100))

if len(args.out) > 0:
    gold = batch.gold()
    words = batch.words()
    assert len(gold) == len(words) == len(predictions), "Dataset size mismatch."
    out = args.model_dir + '/' + args.out
    with open(out, 'w') as outfile:
        for ws, gs, ps in zip(words, gold, predictions):
            assert len(ws) == len(gs) == len(ps), "Example length mismatch."
            for w,g,p in zip(ws,gs,ps):
                outfile.write("{}\t{}\t{}\n".format(w,g,p))
            outfile.write('\n')
    print("All predictions saved to file {}".format(out))

