"""
Load the column NER data, and process into json.
"""

import os
import json
import argparse

NUM_FIELD = 2

def parse_args():
    parser = argparse.ArgumentParser(description="Convert column data into json.")
    parser.add_argument('data_dir', help='Original data directory')
    parser.add_argument('target_dir', help='Target directory to write the converted data.')
    parser.add_argument('--scheme', type=str, dest='scheme', default='iob', help='Tagging scheme to use.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    files = ['train', 'testa', 'testb']
    pattern = 'eng.{}.' + args.scheme
    target_pattern = '{}.json'

    for f in files:
        filename = args.data_dir + '/' + pattern.format(f)
        data = load_column_data(filename)
        json_data = [{'token': tk, 'tag': tg} for tk, tg in data]
        out = args.target_dir + '/' + target_pattern.format(f)
        with open(out, 'w') as outfile:
            json.dump(json_data, outfile)
        print("Write to json file {}".format(out))

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

