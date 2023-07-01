#!/usr/bin/python
# encoding: utf-8

# usage: python pku_tok.py detok_file pkutok_file
from __future__ import unicode_literals, print_function

from pkunlp import Segmentor, NERTagger, POSTagger
import sys

segmentor = Segmentor("/home/derekfw/taofang/GEC_zh/pkunlp/feature/segment.feat", "/home/derekfw/taofang/GEC_zh/pkunlp/feature/segment.dic")

def tok(detok_file, tok_file):

    pku_tok = open(tok_file, "w")

    for line in open(detok_file):
        pku_segments = segmentor.seg_string(line)
        # print(pku_segments)
        
        pku_tok.write(' '.join(pku_segments))

    pku_tok.close()


if __name__ == "__main__":
    tok(sys.argv[1], sys.argv[2])
    
