"""
Prepare vocabulary and initial word vectors.
"""
import pickle
import argparse
import numpy as np
from collections import Counter

from utils import vocab, constant, helper, jsonl

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for NER.')
    parser.add_argument('data_dir', help='Directory of the jsonl data.')
    parser.add_argument('vocab_dir', help='Output vocab directory.')
    parser.add_argument('--random', action='store_true', help='Randomly initialize vectors.')
    parser.add_argument('--glove_dir', default='dataset/glove', help='GloVe directory.')
    parser.add_argument('--wv_file', default='glove.6B.100d.txt', help='GloVe vector file.')
    parser.add_argument('--wv_dim', type=int, default=100, help='GloVe vector dimension.')
    parser.add_argument('--min_freq', type=int, default=0, help='If > 0, use min_freq as the cutoff.')
    parser.add_argument('--lower', action='store_true', help='If specified, lowercase all words.')
    parser.add_argument('--char_lower', action='store_true', help='If specified, lowercase all characters.')
    parser.add_argument('--all', action='store_true', help='If specified, create vector for all words in train and dev.')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # input files
    train_file = args.data_dir + '/train.jsonl'
    dev_file = args.data_dir + '/testa.jsonl'
    test_file = args.data_dir + '/testb.jsonl'
    wv_file = args.glove_dir + '/' + args.wv_file
    wv_dim = args.wv_dim

    # output files
    helper.ensure_dir(args.vocab_dir)
    vocab_file = args.vocab_dir + '/vocab.pkl'
    char_vocab_file = args.vocab_dir + '/vocab_char.pkl'
    emb_file = args.vocab_dir + '/embedding.npy'

    # load files
    print("loading files...")
    train_tokens, train_chars = load_tokens(train_file)
    dev_tokens, dev_chars = load_tokens(dev_file)
    test_tokens, test_chars = load_tokens(test_file)
    if args.lower:
        train_tokens, dev_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in\
                (train_tokens, dev_tokens, test_tokens)]
    if args.char_lower and train_chars:
        train_chars, dev_chars, test_chars = [[c.lower() for c in chars] for chars in\
            (train_chars, dev_chars, test_chars)]

    # load glove
    print("loading glove...")
    glove_vocab = vocab.load_glove_vocab(wv_file, wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))
    
    print("building vocab...")
    if args.all:
        all_tokens = train_tokens + dev_tokens + test_tokens
    else:
        all_tokens = train_tokens
    v = build_vocab(all_tokens, glove_vocab, args.min_freq)
    
    if train_chars:
        print("building vocab for chars...")
        all_chars = train_chars + dev_chars + test_chars
        char_counter = Counter(all_chars)
        #char_vocab = constant.VOCAB_PREFIX + sorted(char_counter.keys(), key=char_counter.get, reverse=True)
        char_vocab = constant.VOCAB_PREFIX + sorted(list(char_counter.keys()))
        print("vocab built with {} chars.".format(len(char_vocab)))
    else:
        char_vocab = None

    print("calculating oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))
    
    print("building embeddings...")
    if args.random:
        print("using random initialization...")
        embedding = random_embedding(v, wv_dim)
    else:
        embedding = vocab.build_embedding(wv_file, v, wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    if char_vocab:
        with open(char_vocab_file, 'wb') as outfile:
            pickle.dump(char_vocab, outfile)
    np.save(emb_file, embedding)
    print("all done.")

def random_embedding(vocab, wv_dim):
    embeddings = 2 * np.random.rand(len(vocab), wv_dim) - 1.0
    return embeddings

def load_tokens(filename):
    with open(filename) as infile:
        data = jsonl.load(infile)
        tokens = []
        chars = []
        for d in data:
            tokens += d['token']
            if 'char' in d:
                chars += sum(d['char'], [])
        tokens = list(map(vocab.normalize_token, tokens))
    print("{} tokens, {} chars from {} examples loaded from {}.".format(len(tokens), len(chars), len(data), filename))
    chars = chars if len(chars)>0 else None
    return tokens, chars

def build_vocab(tokens, glove_vocab, min_freq):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens tokens
    v = constant.VOCAB_PREFIX + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched

if __name__ == '__main__':
    main()


