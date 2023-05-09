from moverscore_v2 import word_mover_score
import sys
from collections import defaultdict
from typing import List, Union, Iterable
import numpy as np
import sacrebleu
from itertools import zip_longest
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
def ms(hyp_file, ref_file):
    refs = open(ref_file,'r').readlines()
    refs = [l.strip() for l in refs]
    refs = [refs]
    hyps = open(hyp_file, 'r').readlines()
    hyps = [l.strip() for l in hyps]
    # refs = ['The dog bit the man.', 'The dog had bit the man.']
    # sys = 'The dog bit the man.'

    # bleu = sacrebleu.sentence_bleu(sys, refs)
    bl = sacrebleu.corpus_bleu(hyps, refs)
    mover = corpus_score(hyps, refs)
    return mover, bl

def sentence_score(hypothesis: str, references: List[str], trace=0):
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)

    hypothesis = [hypothesis] * len(references)

    sentence_score = 0

    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1,
                              remove_subwords=False)

    sentence_score = np.mean(scores)

    if trace > 0:
        print(hypothesis, references, sentence_score)

    return sentence_score


def corpus_score(sys_stream: List[str],
                 ref_streams: Union[str, List[Iterable[str]]], trace=0):
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


if __name__ == '__main__':
    hyp_file = sys.argv[1]
    ref_file = sys.argv[2]
    mo, bl = ms(hyp_file, ref_file)
    print("ms: ",mo)
    print("bleu: ",bl)
