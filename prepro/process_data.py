"""
Load the column NER data, and process into jsonl format.
"""

import os
import argparse

from utils import jsonl

NUM_FIELD = 2

def parse_args():
    parser = argparse.ArgumentParser(description="Convert column data into json-line format.")
    parser.add_argument('data_dir', help='Original data directory')
    parser.add_argument('target_dir', help='Target directory to write the converted data.')
    parser.add_argument('--scheme', type=str, dest='scheme', default='iob', help='Tagging scheme to use.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    files = ['train', 'testa', 'testb']
    pattern = 'eng.{}.' + args.scheme
    target_pattern = '{}.jsonl'

    if not os.path.exists(args.target_dir):
        print("Creating directory {} ...".format(args.target_dir))
        os.makedirs(args.target_dir)

    for f in files:
        filename = args.data_dir + '/' + pattern.format(f)
        data = load_column_data(filename)
        json_data = [{'token': tk, 'tag': tg} for tk, tg in data]

        # featurize
        char_vocab = set()
        for d in json_data:
            chars = get_chars(d['token'])
            d['char'] = chars
            char_vocab.update(sum(chars, []))

        # save to file
        out = args.target_dir + '/' + target_pattern.format(f)
        with open(out, 'w') as outfile:
            jsonl.dump(json_data, outfile)
        print("Write to jsonl file {}".format(out))
        print("{} unique chars found.".format(len(char_vocab)))

def get_chars(words):
    chars = [list(w) for w in words]
    return chars

def load_column_data(filename):
    """
    Load the converted column NER data.
    """
    cached_lines = []
    examples = []
    with open(filename) as infile:
        for line in infile:
            line = line.strip()
            if len(line) > 0:
                array = line.split('\t')
                if len(array) != NUM_FIELD:
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
        ner_tags.append(array[1])
    return (tokens, ner_tags)

if __name__ == '__main__':
    main()

