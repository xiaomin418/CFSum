#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 02:01:42 2021

@author: zhao
"""
from typing import List, Union, Iterable
from itertools import zip_longest
import sacrebleu
from moverscore_v2 import word_mover_score, get_idf_dict
from bert_score import score as BertScore
from collections import defaultdict
import numpy as np
def sentence_score(hypothesis: str, references: List[str], trace=0):
    
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    
    hypothesis = [hypothesis] * len(references)
    
    sentence_score = 0 

    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    
    sentence_score = np.mean(scores)
    
    if trace > 0:
        print(hypothesis, references, sentence_score)
            
    return sentence_score

def corpus_score(sys_stream: List[str],
                     ref_streams:Union[str, List[Iterable[str]]], trace=0):

    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]

    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]

    fhs = [sys_stream] + ref_streams

    corpus_score = 0
    for lines in zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")
            
        hypo, *refs = lines
        corpus_score += sentence_score(hypo, refs, trace=0)
        
    corpus_score /= len(sys_stream)

    return corpus_score

def test_corpus_score():
    
    refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
            ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']]
    sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
    
    bleu = sacrebleu.corpus_bleu(sys, refs)
    mover = corpus_score(sys, refs)
    print(bleu.score)
    print(mover)
    
def test_sentence_score():
    
    refs = ['The dog bit the man.', 'The dog had bit the man.']
    sys = 'The dog bit the man.'
    
    bleu = sacrebleu.sentence_bleu(sys, refs)
    mover = sentence_score(sys, refs)
    
    print(bleu.score)
    print(mover)

def test_hyp_ref():
    import sys
    import os
    ref_path = sys.argv[-2]
    hyp_path = sys.argv[-1]
    if not (os.path.exists(ref_path) and os.path.exists(hyp_path)):
        print("The ref_path or hyp_path doesn't exists.")
    with open(ref_path, 'r') as fr:
        ref_lines = fr.readlines()
        fr.close()
    with open(hyp_path, 'r') as fh:
        hyp_lines = fh.readlines()
        fh.close()
    ref_lines = [[line.strip() for line in ref_lines]]
    hyp_lines = [line.strip() for line in hyp_lines]
    bleu = sacrebleu.corpus_bleu(hyp_lines, ref_lines)

    idf_dict_hyp = get_idf_dict(hyp_lines)
    idf_dict_ref = get_idf_dict(ref_lines[0])
    mover = word_mover_score(ref_lines[0], hyp_lines, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=2, batch_size=16)
    mover = sum(mover)/len(mover)
    # mover = corpus_score(hyp_lines, ref_lines[0])

    P, R, F1 = BertScore(hyp_lines, ref_lines[0],lang="en")
    print(f"BertScore F1 score: {F1.mean(): .3f}")
    print("Bleu: {}  MoverScore: {}".format(bleu.score, mover))


if __name__ == '__main__':   

    # test_sentence_score()
    
    # test_corpus_score()

    test_hyp_ref()
        
