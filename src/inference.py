import argparse
import json
from torch.utils.data import DataLoader
import torch
from utils.misc import Struct

import sys
from time import time
import os
import logging
import glob
import pickle
from pytorch_pretrained_bert import BertTokenizer
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm
sys.path.append('../')
from src.datasets.dataset_momu import get_data_loader
from src.datasets.loader import PrefetchLoader
from src.model.mms import MultiModal
import torch.nn.functional as F
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
# torch.backends.cudnn.enabled = True
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(123)
# def convert_tokens_to_string(current_sub_text):
#     return " ".join(self.convert_ids_to_tokens(tokens))
#
# def ids_decode(filtered_tokens):
#     sub_texts = []
#     current_sub_text = []
#     for token in filtered_tokens:
#         if token in self.added_tokens_encoder:
#             if current_sub_text:
#                 sub_texts.append(self.convert_tokens_to_string(current_sub_text))
#                 current_sub_text = []
#             sub_texts.append(token)
#         else:
#             current_sub_text.append(token)
#     if current_sub_text:
#         sub_texts.append(self.convert_tokens_to_string(current_sub_text))
#     text = " ".join(sub_texts)
#
#     if clean_up_tokenization_spaces:
#         clean_text = self.clean_up_tokenization(text)
#         return clean_text
#     else:
#         return text

def merge_sub_word(tokenizer, sentence):
    sub_words = tokenizer.convert_ids_to_tokens(sentence)
    words = []
    i= 0
    len_sub = len(sub_words)
    cur_word = ''
    while i <len_sub:
        if sub_words[i].startswith('##'):
            cur_word = cur_word+sub_words[i][2:]
        else:
            if len(cur_word)!=0:
                words.append(cur_word)
            cur_word = sub_words[i]
        i = i+1
    if len(cur_word)!=0:
        words.append(cur_word)
    return words

def merge_sub_word2score(tokenizer, sentence, scores):
    sub_words = tokenizer.convert_ids_to_tokens(sentence)
    words = {}
    _scores = []
    i= 0
    len_sub = len(sub_words)
    cur_word = ''
    cur_score = []
    while i <len_sub:
        if sub_words[i].startswith('##'):
            cur_word = cur_word+sub_words[i][2:]
            cur_score.append(scores[i])
        else:
            if len(cur_word)!=0:
                if cur_word in words:
                    words[cur_word].append(sum(cur_score)/len(cur_score))
                else:
                    words[cur_word] = [sum(cur_score)/len(cur_score)]
            cur_word = sub_words[i]
            cur_score = [scores[i]]
        i = i+1
    if len(cur_word)!=0:
        if cur_word in words:
            words[cur_word].append(sum(cur_score) / len(cur_score))
        else:
            words[cur_word] = [sum(cur_score) / len(cur_score)]

    words = {k:sum(v)/len(v) for k,v in words.items()}

    return words

def vocab_scores_update(vocabdict, wordict, sid):
    for k,v in wordict.items():
        if k=='[CLS]' or k=='[SEP]':
            continue
        if k in vocabdict:
            vocabdict[k].append((v,sid))
        else:
            vocabdict[k] = [(v,sid)]

def infer(config_name = './configs/config.json', model_path=None):
    BUCKET_SIZE = 8192
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_workers', type=int, default=1,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    args = parser.parse_args()
    with open(config_name, 'r') as f:
        data = json.load(f)
        f.close()
    args.__dict__.update(data)

    # args.guiding = base_config.guiding_in_test
    # args.shuffle = base_config.shuffle_in_test

    device = args.device_id
    print("device id: ",device)
    tokenizer = BertTokenizer.from_pretrained(args.toker, do_lower_case='uncased' in args.toker)
    test_dataloder = get_data_loader(args, args.test_image_path, args.test_text_path, args.test_summri_path, tokenizer,
                                     device, args.test_useful_pic_path, args.test_useless_pic_path,
                                     is_test=True)
    test_dataloder = PrefetchLoader(test_dataloder, device_id=device)

    beam_Search_model = MultiModal(args, tokenizer)
    beam_Search_model.to(device)
    if model_path!=None:
        beam_Search_model.setup_train(model_path)
    beam_Search_model.eval()
    iter_bar = tqdm(total=len(test_dataloder.loader.dataset), desc='Iter (loss=X.XXX)')

    hyps_dir = "/".join(model_path.split('/')[:-2])
    hyps_dir = os.path.join(hyps_dir, 'hyps')
    model_name = model_path.split('/')[-1]
    if not os.path.exists(hyps_dir):
        os.mkdir(hyps_dir)

    hyp_file_path = hyps_dir+'/hyp_{}.txt'.format(model_name)
    hyp_file = open(hyp_file_path, 'w')
    datas = []
    loss_weights = []
    attns = []
    ot_dists = []
    tokens = []
    less_pool = 0
    vocabdict = {}
    for iter, batch in enumerate(test_dataloder):
        # Run beam search to get best Hypothesis
        best_summary, all_attention_scores, loss_weight, pool_diff, mutual_info, phrase_info = beam_Search_model.beam_search(batch)
        ids_len = batch['input_ids'][0].shape[0]
        # wordict = merge_sub_word2score(tokenizer, batch['input_ids'][0].tolist(), mutual_info[0,:ids_len].tolist())
        # vocab_scores_update(vocabdict, wordict, 'd{}'.format(iter))
        ot_dists.append(pool_diff.item())
        # if pool_diff.item()<args.ot_trunc:
        #     less_pool = less_pool + 1
        # if less_pool %10==0:
        #     print("iter: {} less_pool:{}".format(iter, less_pool))
        txt_lens = batch['txt_lens'][0]
        # data = [(all_attention_scores[i][0][0][:txt_lens,txt_lens:].max().item(),all_attention_scores[i][0][0][:42,42:].mean().item()) for i in range(12)]
        # Extract the output ids from the hypothesis and convert back to words
        # datas.append(data)
        output_ids = [[int(t) for t in best_summary.tokens[1:]]]
        hyp_tokens = [merge_sub_word(tokenizer, sentence) for sentence in output_ids]

        # Remove the [STOP] token from decoded_words, if necessary
        _hyp_tokens = []
        for sentence in hyp_tokens:
            if '[SEP]' in sentence:
                _hyp_tokens.append(sentence[:sentence.index('[SEP]')])
            elif '<PAD>' in sentence:
                _hyp_tokens.append(sentence[:sentence.index('<PAD>')])
            else:
                _hyp_tokens.append(sentence)
        hyp_tokens = [" ".join(sentence) for sentence in _hyp_tokens]
        for hyp in hyp_tokens:
            if len(hyp.replace('\n', '')) == 0:
                continue
            hyp_file.write(hyp.replace('\n', '').strip() + '\n')
        if loss_weight!=None:
            loss_weights.append(loss_weight[0].data.tolist())
        attns.append(F.softmax(all_attention_scores[6].mean(dim=1)[0,:,:],dim=-1).data.cpu())
        tokens.append(tokenizer.convert_ids_to_tokens(batch['input_ids'][0].tolist()))
        iter_bar.update(1)
    print("avg ot dist: ",sum(ot_dists)/len(ot_dists))
    hyp_file.close()

    res_file_path = hyps_dir + '/rouge_{}.txt'.format(model_name)
    df = os.popen('files2rouge /home/meihuan2/document/MMSS4.0/corpus/test_title.txt {}'.format(hyp_file_path)).read()
    with open(res_file_path,'w') as fres:
        fres.write(df)
        fres.close()
    # ot_file_path = hyps_dir + '/otdist_{}.pkl'.format(model_name)
    # with open(ot_file_path,'wb') as fres:
    #     pickle.dump(ot_dists, fres)
    #     fres.close() #vocabdict
    # ot_file_path = hyps_dir + '/vocabdict_{}.pkl'.format(model_name)
    # with open(ot_file_path, 'wb') as fres:
    #     pickle.dump(vocabdict, fres)
    #     fres.close()  # vocabdict
    # weight_file_path = hyps_dir + '/weight_{}.pickle'.format(model_name)
    # with open(weight_file_path,'wb') as fw:
    #     pickle.dump(loss_weights,fw)
    #     fw.close()
    attn_file = hyps_dir + '/attn_{}.pickle'.format(model_name)
    with open(attn_file,'wb') as fw:
        pickle.dump((attns,tokens),fw)
        fw.close()

def sort_rouge(model_dir):
    rouge_paths = glob.glob(model_dir + '/hyps/rouge*')
    all_result_dict = {}
    for rouge_p in rouge_paths:
        with open(rouge_p, 'r') as frouge:
            df = frouge.read()
            frouge.close()
        rouge1, rouge2, rougeL = df.split('\n')[5].split(' ')[3], df.split('\n')[9].split(' ')[3], \
                                 df.split('\n')[13].split(' ')[3]
        rouge1, rouge2, rougeL = float(rouge1), float(rouge2), float(rougeL)
        all_result_dict[rouge_p] = [rouge1, rouge2, rougeL]
    sorted_result = sorted(all_result_dict.items(), key=lambda x: x[1][2])

    fresult = open(model_dir + '/rouge.txt', 'w')
    for model_pth ,rouges in sorted_result:
        fresult.write(model_pth+' '+str(rouges[0])+' '+str(rouges[1])+' '+str(rouges[2])+'\n')
        print(model_pth+' '+str(rouges[0])+' '+str(rouges[1])+' '+str(rouges[2])+'\n')
    fresult.close()

def infer_many(config_name, model_dir, tgt_epoch_ids = []):
    # model_dir = '/data/meihuan2/PreEMMS_checkpoints/0324-base'
    model_paths = glob.glob(model_dir+'/model/*')
    if len(tgt_epoch_ids)==0:
        print("No test model.")
        return
    for model_id, model_pth in enumerate(model_paths):
        model_epoch = model_pth.split('/')[-1].split('_')[1]
        model_iter = model_pth.split('/')[-1].split('_')[2]
        model_epoch = int(model_epoch)
        if model_epoch in tgt_epoch_ids:
            print("id: {} valid: {}".format(model_id, model_pth))
            infer(config_name, model_pth)
    sort_rouge(model_dir)



if __name__ == '__main__':
    import sys
    config_name = sys.argv[1]
    model_filename = sys.argv[2]
    sys.argv = sys.argv[:-2]
    # infer(config_name, model_filename)
    infer_many(config_name, model_filename,[10,11,12,13,14,15,18,20,21,22,23]) # 18,19,21,22,23,24 #10,11,12,13,14,15
