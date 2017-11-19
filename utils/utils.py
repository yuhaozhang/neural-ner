"""
Utility functions.
"""

import data_utils

def convert_ids_to_tags(seqs, id2tag):
    tag_seqs = []
    for seq in seqs:
        tag_seq = []
        for s in seq:
            tag_seq.append(id2tag[s])
        tag_seqs.append(tag_seq)
    return tag_seqs

def test():
    id2tag = dict([(v,k) for k,v in data_utils.LABEL_TO_ID.items()])
    seqs = [[1,2,3], [0,0,1,0,3,2,0]]
    print convert_ids_to_tags(seqs, id2tag)

if __name__ == '__main__':
    test()
