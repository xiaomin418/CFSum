import pdb
import re
import os
import json
import files2rouge
import bert_score
import numpy as np
import math
from moverscore_v2 import get_idf_dict, word_mover_score
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

model_name = ['PGN', 'PGN_multi', 'PGN_enc', 'PGN_dec', 'PGN_both', 'bert_abs', 'bert_abs_multi', 'bert_abs_enc', 'bert_abs_dec', 'bert_abs_both']
modes = ['user', 'agent']
auto_metrics = ['rouge_1', 'rouge_2', 'rouge_l', 'bleu', 'bertscore', 'moverscore']
from get_simple_score import generate_sent_file, get_sents_scores
# tmp_ref_dir = '../result/tmp-ref/'
# tmp_hyp_dir = '../result/tmp-hyp/'
def get_sents_str(file_path):
    sents = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = re.sub(' ', '', line)
            line = re.sub('<q>', '', line)
            sents.append(line)
    return sents

def change_word2id_split(ref, pred):
    ref_id, pred_id = [], []
    tmp_dict = {'%': 0}
    new_index = 1
    words = list(ref)
    for w in words:
        if w not in tmp_dict.keys():
            tmp_dict[w] = new_index
            ref_id.append(str(new_index))
            new_index += 1
        else:
            ref_id.append(str(tmp_dict[w]))
        if w == '。':
            ref_id.append(str(0))
    words = list(pred)
    for w in words:
        if w not in tmp_dict.keys():
            tmp_dict[w] = new_index
            pred_id.append(str(new_index))
            new_index += 1
        else:
            pred_id.append(str(tmp_dict[w]))
        if w == '。':
            pred_id.append(str(0))
    return ' '.join(ref_id), ' '.join(pred_id)

def read_rouge_score(name):
    with open(name, 'r') as f:
        lines = f.readlines()
    r1 = lines[3][21:28]
    r2 = lines[7][21:28]
    rl = lines[11][21:28]
    rl_p = lines[10][21:28]
    rl_r = lines[9][21:28]
    return [float(r1), float(r2), float(rl), float(rl_p), float(rl_r)]

def calculate(pred_file, ref_file, mode, model):
    generate_sent_file(pred_file, ref_file)
    rouge_details = get_sents_scores('')
    rouge_details = [rouge_details[str(i+1)+'.jpg'] for i in range(2000)]
    refs = get_sents_str(ref_file)
    preds = get_sents_str(pred_file)

    scores = []
    #get rouge scores
    # print('Running ROUGE for ' + mode + ' ' + model + '-----------------------------')
    # pred_ids, ref_ids = [], []
    # for ref, pred in zip(refs, preds):
    #     ref_id, pred_id = change_word2id_split(ref, pred)
    #     pred_ids.append(pred_id)
    #     ref_ids.append(ref_id)
    # with open('ref_ids.txt', 'w') as f:
    #     for ref_id in ref_ids:
    #         f.write(ref_id + '\n')
    # with open('pred_ids.txt', 'w') as f:
    #     for pred_id in pred_ids:
    #         f.write(pred_id + '\n')
    os.system('files2rouge {} {} -s rouge.txt'.format(ref_file, pred_file))
    rouge_scores = read_rouge_score('rouge.txt')
    scores.append(rouge_scores[0])
    scores.append(rouge_scores[1])
    scores.append(rouge_scores[2])

    # get bleu scores
    #print('Running BLEU for ' + mode + ' ' + model + '-----------------------------')
    bleu_preds, bleu_refs = [], []
    bleu_scores = []
    for ref, pred in zip(refs, preds):
        bleu_preds.append(list(pred))
        bleu_refs.append([list(ref)])
        bleu_score = sentence_bleu([list(ref)], list(pred))
        bleu_scores.append(bleu_score)
    bleu_score = corpus_bleu(bleu_refs, bleu_preds)
    #bleu_score = np.mean(np.array(bleu_scores))
    scores.append(bleu_score)

    # run bertscore
    print('Running BERTScore for ' + mode + ' ' + model + '-----------------------------')
    prec, rec, f1 = bert_score.score(preds, refs, lang='en')
    scores.append(f1.numpy().mean().item())
    #
    # run moverscore
    #print('Running MoverScore for ' + mode + ' ' + model + '-----------------------------')
    idf_dict_hyp = get_idf_dict(preds)
    idf_dict_ref = get_idf_dict(refs)
    mover_scores = word_mover_score(refs, preds, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=2, batch_size=16)
    scores.append(np.array(mover_scores).mean().item())

    # print('Model: %s, Mode: %s' % (model, mode))
    # for i in range(len(auto_metrics)):
    #     print('%s: %.4f' % (auto_metrics[i], scores[i]))

    score_dict = {}
    for i, metric in enumerate(auto_metrics):
        score_dict[metric] = scores[i]
    return score_dict, scores,(rouge_details,bleu_scores, f1, mover_scores)


if __name__ == '__main__':

    import sys
    import os

    # ref_path = sys.argv[-2]
    # hyp_path = sys.argv[-1]
    # if not (os.path.exists(ref_path) and os.path.exists(hyp_path)):
    #     print("The ref_path or hyp_path doesn't exists.")
    hyp_paths = {
    #               "UniG(T)":"/data/meihuan2/ReAttnMMS_checkpoints/1115-textonly/hyps/hyp_model_38_62000_1668474056.txt",
    #              "UniG":"/data/meihuan2/ReAttnMMS_checkpoints/1114-en-train/hyps/hyp_model_32.txt",
    #              "F3": "/data/meihuan2/ReAttnMMS_checkpoints/0103-prefilter/hyps/hyp15.txt",
    #              "W6":"/data/meihuan2/ReAttnMMS_checkpoints/1109-kl/hyps/hyp_model_13_62000_1667985619.txt",
    #              "S9": "/data/meihuan2/ReAttnMMS_checkpoints/0103-semantic/hyps/hyp_model_6_62000_1672677590.txt",
    #              "W6S9":"/data/meihuan2/ReAttnMMS_checkpoints/1203-pw-nodetach/hyps/hyp16.txt",
    #              "F3W6S9":"/data/meihuan2/ReAttnMMS_checkpoints/1206-c2-ot-l6l9l3/hyps/hyp_model_12_62000_1670233244.txt",
    #              "F9W3S6":"/data/meihuan2/ReAttnMMS_checkpoints/1205-c2-ot/hyps/hyp13.txt",
                 "F3W6":"/data/meihuan2/ReAttnMMS_checkpoints/0118-f3w6/hyps/hyp_model_23_62000_1673979111.txt",
                 "f3S9":"/data/meihuan2/ReAttnMMS_checkpoints/0118-f3p9/hyps/hyp_model_21_62000_1673978527.txt"
                 }
    import csv
    def convert_detail_to_csv(detail_scores, file_name):
        rouge_details = detail_scores[0]
        bleu_details = detail_scores[1]
        bert_details = detail_scores[2]
        movers_details = detail_scores[3]
        f = open(file_name, 'w')
        writer = csv.writer(f, dialect='excel')
        writer.writerow(['r1','r2','rl', 'bleu', 'bert', 'movers'])
        for rg, bl, be, mo in zip(rouge_details, bleu_details, bert_details.tolist(), movers_details):
            if bl<0.00001:
                writer.writerow([rg[0],rg[1],rg[2], 0, be, mo])
            else:
                writer.writerow([rg[0],rg[1],rg[2], bl, be, mo])
        f.close()


    # hyp_paths = {"UniG":"/data/meihuan2/ReAttnMMS_checkpoints/1114-en-train/hyps/hyp_model_32.txt"
    #              }
    ref_path = "/home/meihuan2/document/MMSS4.0/corpus/test_title.txt"
    all_score = []
    for k,hp in hyp_paths.items():
        model = hp.split('/')[-3]
        scores_dict, scores, detail_scores = calculate(hp, ref_path, k, model)
        convert_detail_to_csv(detail_scores, "/".join(hp.split('/')[:-2])+'/{}.csv'.format(k))

        all_score.append((k, scores))
    print("--------------------------------------")
    for k,scores in all_score:
        print('Model: %s' % (k))
        for i in range(len(auto_metrics)):
            print('%s: %.4f' % (auto_metrics[i], scores[i]))
        print('\n')




