"""
Convert the original conll2003 data into IOB or IOBES NER schemes.
"""

import os
import sys
import argparse

DATA_FILES = ['eng.train', 'eng.testa', 'eng.testb']

NUM_FIELD = 4

DOC_START_TOKEN = '-DOCSTART-'

def load_original_data(filename):
    cached_lines = []
    examples = []
    with open(filename) as infile:
        for line in infile:
            line = line.strip()
            if len(line) > 0:
                array = line.split('\t')
                if len(array) != NUM_FIELD or array[0] == DOC_START_TOKEN:
                    continue
                else:
                    cached_lines.append(line)
            elif len(cached_lines) > 0:
                example = process_cache(cached_lines)
                examples.append(example)
                cached_lines = []
    return examples

def process_cache(cached_lines):
    tokens = []
    ner_tags = []
    for line in cached_lines:
        array = line.split('\t')
        assert(len(array) == NUM_FIELD)
        tokens.append(array[0])
        ner_tags.append(array[3])
    return (tokens, ner_tags)

def convert_tags_to_IOBES(tags):
    '''
    Convert the original tag sequence into an IOBES scheme.
    '''
    new_tags = []
    pre, cur, nex = '', '', ''
    new_t = ''
    # handle single token sentence
    if len(tags) == 1:
        if tags[0].startswith('I'):
            new_tags.append(tags[0].replace('I-', 'S-'))
        else:
            new_tags.append(tags[0])
        return new_tags
    # sentences that have >=2 tokens
    for i, t in enumerate(tags):
        pre = cur if i > 0 else ''
        cur = t
        nex = tags[i+1] if i < len(tags)-1 else ''
        if i == 0: # first tag
            if cur.startswith('I'):
                if is_different_chunk(cur, nex):
                    new_t = cur.replace('I-', 'S-')
                else:
                    new_t = cur.replace('I-', 'B-')
            else: # cur == 'O'
                new_t = cur
        elif i == len(tags) - 1: # last tag
            if cur.startswith('I'):
                if is_different_chunk(pre,cur):
                    new_t = cur.replace('I-', 'S-')
                else:
                    new_t = cur.replace('I-', 'E-')
            elif cur.startswith('B'):
                new_t = cur.replace('B-', 'S-')
            else: # cur == 'O'
                new_t = cur
        else:
            if cur.startswith('I'):
                if is_different_chunk(pre, cur):
                    if is_different_chunk(cur, nex):
                        new_t = cur.replace('I-', 'S-')
                    else:
                        new_t = cur.replace('I-', 'B-')
                else:
                    if is_different_chunk(cur, nex):
                        new_t = cur.replace('I-', 'E-')
                    else:
                        new_t = cur
            elif cur.startswith('B'):
                if is_different_chunk(cur, nex):
                    new_t = cur.replace('B-', 'S-')
                else:
                    new_t = cur
            else: # cur == 'O'
                new_t = cur
        new_tags.append(new_t)
    return new_tags

def is_different_chunk(tag1, tag2):
    '''tag1 must come before tag2 in sequence'''
    if tag1 == 'O' and tag2 == 'O':
        return False
    if tag1 == 'O' and tag2 != 'O':
        return True
    if tag2 == 'O' and tag1 != 'O':
        return True
    if tag1.startswith('I') and tag2.startswith('B'):
        return True
    if tag1.startswith('I') and tag2.startswith('I'):
        return (tag1[2:] != tag2[2:])
    return False

def main():
    parser = argparse.ArgumentParser(description="Convert the orginal conll data into simple form, either in IOBES or IOB schemes.")
    parser.add_argument('data_dir', help='Original data directory')
    parser.add_argument('target_dir', help='Target directory to write the converted data.')
    parser.add_argument('--scheme', type=str, dest='scheme', default='iob', help='Tagging scheme to use. The original data is in IOB scheme.')

    args = parser.parse_args()

    if args.scheme not in ('iobes', 'iob'):
        raise Exception("Unsupported scheme type: " + args.scheme)

    print("Converting using scheme " + args.scheme.upper())
    
    for f in DATA_FILES:
        fname = os.path.join(args.data_dir, f) 
        examples = load_original_data(fname)
        fname_out = os.path.join(args.target_dir, f + '.' + args.scheme)
        print("Writing to file " + fname_out)
        with open(fname_out, 'w') as outfile:
            for sent, tags in examples:
                if args.scheme == 'iobes':
                    tags = convert_tags_to_IOBES(tags)
                for w, t in zip(sent, tags):
                    print(w + '\t' + t, file=outfile)
                outfile.write('\n')
            print("{} examples written.".format(len(examples)))

def test():
    test_cases = [['I-ORG', 'O', 'O', 'O', 'I-ORG'],
            ['I-ORG', 'B-ORG', 'O', 'I-ORG', 'B-ORG'],
            ['I-ORG', 'I-ORG', 'I-ORG', 'O', 'I-PER', 'I-PER'],
            ['I-ORG', 'I-ORG', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O'],
            ['I-ORG', 'I-ORG', 'I-PER', 'I-PER', 'I-LOC', 'O', 'I-PER', 'I-LOC']]
    for tags in test_cases:
        print(tags)
        print(convert_tags_to_IOBES(tags))
        print("================================")

if __name__ == '__main__':
    main()
    #test()
        
