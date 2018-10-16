"""
A scorer that provides F1 scores given gold and predicted tags.
"""
import sys
import os
from collections import Counter

from utils import constant

EMPTY_TAG = constant.DEFAULT_TYPE

def score_by_token(gold_tags, predicted_tags, verbose=True):
    """
    Score predicted sequences based on gold sequences on the token-level.
    """
    correct_by_tag = Counter()
    guessed_by_tag = Counter()
    gold_by_tag = Counter()
    assert(len(gold_tags) == len(predicted_tags))
    for gold_seq, pred_seq in zip(gold_tags, predicted_tags):
        assert(len(gold_seq) == len(pred_seq))
        for g, p in zip(gold_seq, pred_seq):
            if g == EMPTY_TAG and p == EMPTY_TAG:
                continue
            elif g == EMPTY_TAG and p != EMPTY_TAG:
                guessed_by_tag[p] += 1
            elif g != EMPTY_TAG and p == EMPTY_TAG:
                gold_by_tag[g] += 1
            else:
                guessed_by_tag[p] += 1
                gold_by_tag[p] += 1
                if g == p:
                    correct_by_tag[p] += 1
    prec_micro = 1.0
    if sum(guessed_by_tag.values()) > 0:
        prec_micro = sum(correct_by_tag.values()) * 1.0 / sum(guessed_by_tag.values())
    rec_micro = 0.0
    if sum(gold_by_tag.values()) > 0:
        rec_micro = sum(correct_by_tag.values()) * 1.0 / sum(gold_by_tag.values())
    f_micro = 0.0
    if prec_micro + rec_micro > 0:
        f_micro = 2.0 * prec_micro * rec_micro / (prec_micro + rec_micro)
    if verbose:
        print("P\tR\tF1")
        print("{:.2f}\t{:.2f}\t{:.2f}".format(prec_micro*100, rec_micro*100, f_micro*100))
    return prec_micro, rec_micro, f_micro

def score_by_chunk(gold_tags, predicted_tags, scheme='iob', verbose=True):
    assert(len(gold_tags) == len(predicted_tags))
    if scheme == 'iobes':
        func_get_chunks = get_chunks_from_IOBES
    else:
        func_get_chunks = get_chunks_from_IOB
    gold_chunks = func_get_chunks(gold_tags)
    gold_chunks = set(gold_chunks)
    predicted_chunks = func_get_chunks(predicted_tags)

    # scoring
    correct_by_type = Counter()
    guessed_by_type = Counter()
    gold_by_type = Counter()
    for p in predicted_chunks:
        guessed_by_type[p[-1]] += 1
        if p in gold_chunks:
            correct_by_type[p[-1]] += 1
    for g in gold_chunks:
        gold_by_type[g[-1]] += 1
    prec_micro = 1.0
    if sum(guessed_by_type.values()) > 0:
        prec_micro = sum(correct_by_type.values()) * 1.0 / sum(guessed_by_type.values())
    rec_micro = 0.0
    if sum(gold_by_type.values()) > 0:
        rec_micro = sum(correct_by_type.values()) * 1.0 / sum(gold_by_type.values())
    f_micro = 0.0
    if prec_micro + rec_micro > 0:
        f_micro = 2.0 * prec_micro * rec_micro / (prec_micro + rec_micro)
    if verbose:
        print("P\tR\tF1")
        print("{:.2f}\t{:.2f}\t{:.2f}".format(prec_micro*100, rec_micro*100, f_micro*100))
    return prec_micro, rec_micro, f_micro

def get_chunks_from_IOB(tag_sequences):
    """
    A signature for a chunk is (seq_id, begin_id, length, type).
    """
    pool = []
    for seq_id, seq in enumerate(tag_sequences):
        buffer = []
        buffer_type = ""
        buffer_start = 0
        for idx, t in enumerate(seq):
            if t == EMPTY_TAG and len(buffer) > 0:
                pool.append((seq_id, buffer_start, len(buffer), buffer_type))
                buffer, buffer_type, buffer_start = reset_buffer()
            elif t.startswith('I'):
                if len(buffer) > 0 and t[2:] == buffer_type:
                    buffer += [t]
                elif len(buffer) > 0 and t[2:] != buffer_type:
                    pool.append((seq_id, buffer_start, len(buffer), buffer_type))
                    buffer, buffer_type, buffer_start = create_new_buffer(t, idx)
                else: 
                    buffer, buffer_type, buffer_start = create_new_buffer(t, idx)
            elif t.startswith('B'):
                if len(buffer) > 0:
                    pool.append((seq_id, buffer_start, len(buffer), buffer_type))
                buffer, buffer_type, buffer_start = create_new_buffer(t, idx)
        if len(buffer) > 0:
            pool.append((seq_id, buffer_start, len(buffer), buffer_type))
            buffer, buffer_type, buffer_start = reset_buffer()
    return pool

def get_chunks_from_IOBES2(tag_sequences):
    """
    A more conservative version that will not count wrong tag sequence
    as a chunk.
    """
    pass

def get_chunks_from_IOBES(tag_sequences):
    """
    A signature for a chunk is (seq_id, begin_id, length, type).
    Note that we need to handle all possible cases, as the predicted sequence
    could be messy.
    """
    pool = []
    for seq_id, seq in enumerate(tag_sequences):
        buffer = []
        buffer_type = ""
        buffer_start = 0
        for idx, t in enumerate(seq):
            if t == EMPTY_TAG and len(buffer) > 0:
                pool.append((seq_id, buffer_start, len(buffer), buffer_type))
                buffer, buffer_type, buffer_start = reset_buffer()
            elif t.startswith('S'):
                if len(buffer) > 0:
                    pool.append((seq_id, buffer_start, len(buffer), buffer_type))
                buffer, buffer_type, buffer_start = reset_buffer()
                pool.append((seq_id, idx, 1, t[2:])) # skip the buffer and directly include this
            elif t.startswith('B'):
                if len(buffer) > 0:
                    pool.append((seq_id, buffer_start, len(buffer), buffer_type))
                buffer = [t]
                buffer_type = t[2:]
                buffer_start = idx
            # for I and E it is possible to have inconsistent previous tags
            elif t.startswith('I'):
                if t[2:] == buffer_type:
                    buffer.append(t)
                else: # handle inconsistencies
                    if len(buffer) > 0:
                        pool.append((seq_id, buffer_start, len(buffer), buffer_type))
                    buffer = [t]
                    buffer_type = t[2:]
                    buffer_start = idx
            elif t.startswith('E'):
                if t[2:] == buffer_type:
                    buffer.append(t)
                    pool.append((seq_id, buffer_start, len(buffer), buffer_type))
                    buffer, buffer_type, buffer_start = reset_buffer()
                else: # handle inconsistencies
                    if len(buffer) > 0:
                        pool.append((seq_id, buffer_start, len(buffer), buffer_type))
                        buffer, buffer_type, buffer_start = reset_buffer()
                    pool.append((seq_id, idx, 1, t[2:]))
        # clean up at the end of each sequence
        if len(buffer) > 0:
            pool.append((seq_id, buffer_start, len(buffer), buffer_type))
            buffer, buffer_type, buffer_start = reset_buffer()
    return pool

def create_new_buffer(tag, idx):
    return [tag], tag[2:], idx

def reset_buffer():
    return [], "", 0

def test():
    pred_sequences = [['O', 'S-LOC', 'O', 'O', 'B-PER', 'E-PER'],
                    ['O', 'S-MISC', 'O', 'E-ORG', 'O', 'B-PER', 'I-PER', 'E-PER']]
    gold_sequences = [['O', 'B-LOC', 'E-LOC', 'O', 'B-PER', 'E-PER'],
                    ['O', 'S-MISC', 'B-ORG', 'E-ORG', 'O', 'B-PER', 'E-PER', 'S-LOC']]
    #print score_by_token(gold_sequences, pred_sequences)
    print(score_by_chunk(gold_sequences, pred_sequences, scheme='iobes'))

if __name__ == '__main__':
    test()


