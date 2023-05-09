import glob

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import pickle
import spacy
eng_model = spacy.load('en_core_web_sm')
import numpy as np
import math
from numpy import random
import json
import argparse
import sys
import csv
from pytorch_pretrained_bert import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from stanfordcorenlp import StanfordCoreNLP

sys.path.append('../')
from src.datasets.convert_imgdir import load_npz
from src.datasets.data import pad_tensors, get_gather_index
from src.datasets.sampler import TokenBucketSampler
from src.datasets.loader import PrefetchLoader
from src.configs import base_config
from src.utils.utils import merge_sub_word
from src.utils import const
from src.utils import structure_parse
csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

# device_id = config.device_id

# torch.manual_seed(123)

def shuffle_image(img_lens, image_useful, random_seed):
    random.seed(random_seed)
    shuffle_image_proj = {}
    random_index = [str(i+1) for i in range(img_lens) if str(i+1)+'.jpg' not in image_useful]
    random.shuffle(random_index)
    for index in range(img_lens):
        iname = str((index + 1)) + '.jpg'
        if iname in image_useful:
            shuffle_image_proj[str((index + 1))] = str((index + 1))
        else:
            shuffle_image_proj[str((index + 1))] = random_index.pop()
    return shuffle_image_proj

def sample_balance_useless(img_lens, image_useful, random_seed):
    # import pdb
    # pdb.set_trace()
    random.seed(random_seed)
    useless_index = [i for i in range(img_lens) if str(i + 1) + '.jpg' not in image_useful]
    useful_index = [int(img_str.split('.')[0])-1 for img_str in image_useful]
    random.shuffle(useless_index)
    useless_index = useless_index[:len(useful_index)]

    balance_index = useful_index + useless_index
    random.shuffle(balance_index)
    sample_image_proj = {ind:origin_ind for ind, origin_ind in enumerate(balance_index)}
    return sample_image_proj

def _compute_ot_scatter(txt_lens, max_txt_len, joint_len):
    ot_scatter = torch.arange(0, joint_len, dtype=torch.long
                              ).unsqueeze(0).repeat(len(txt_lens), 1)
    for i, tl in enumerate(txt_lens):
        max_ind = max_txt_len + (joint_len-tl)
        ot_scatter.data[i, tl:] = torch.arange(max_txt_len, max_ind,
                                               dtype=torch.long).data
    return ot_scatter


def _compute_pad(lens, max_len):
    pad = torch.zeros(len(lens), max_len, dtype=torch.uint8)
    for i, l in enumerate(lens):
        pad.data[i, l:].fill_(1)
    return pad

class MultiDataset(Dataset):
    def __init__(self, config, images_path, titles_file,sent_summaris_file, tokenizer, image_useful_file,
                 image_useless_file, random_seed, is_test=False, balance_useful= False):
        self.config = config
        self.images_dir = images_path
        self.titles_file = titles_file
        data_mode = self.images_dir.split('_')[-1]
        self.phrase_dir = "/".join(self.titles_file.split('/')[:-1]) + '/{}_phrase'.format(data_mode)
        self.txt_feat_dir = "/".join(self.titles_file.split('/')[:-1]) + '/{}_txt_feat'.format(data_mode)
        self.titles=open(titles_file,'r').readlines()
        self.tokenizer =  tokenizer
        self.sent_summarizations=open(sent_summaris_file,'r').readlines()

        if config.comple_of_high_freq:
            image_useful_file = image_useless_file[:-7]+'_comple.pickle'
        self.image_useful =  pickle.load(open(image_useful_file, 'rb'))
        self.image_useless = pickle.load(open(image_useless_file, 'rb'))
        if self.config.remove_high_freq:
            self.image_useful = set(self.image_useful)-self.image_useless
        # Init and Build vocab
        self.random_seed = random_seed
        self.balance_useful = balance_useful

        self.start_num = 0

        meta = json.load(open(config.meta_file, 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

        # mode_str = titles_file.split('/')[-1][:-9]
        # load_path = self.config.vg_words_dir + '/txt_{}.pickle'.format(mode_str)
        self.input_ids, self.input_poses, self.txtlens, self.dec_ids, self.dec_poses = \
            self._get_ids_and_lens(self.titles, self.sent_summarizations)
        name2nbb_name = "/".join(images_path.split('/')[:-1])+'/'+images_path.split('/')[-1].split('_')[1]+"_name2nbb.pkl"
        if not os.path.exists(name2nbb_name):
            self.name2nbb = self._get_name_to_nbb(self.images_dir)
            with open(name2nbb_name, 'wb') as fnb:
                pickle.dump(self.name2nbb, fnb)
                fnb.close()
        else:
            with open(name2nbb_name, 'rb') as fnb:
                self.name2nbb = pickle.load(fnb)
                fnb.close()

        assert len(self.txtlens) == len(self.name2nbb)
        self.sample_image_proj = None

        self.lens = [tl + self.name2nbb[str(id + 1) + '.npz'] for id, tl in enumerate(self.txtlens)]


    def __len__(self):
        return len(self.lens)

    def __getitem__(self, index: int):
        # print("dataset.py 160 index: ",index)
        input_str, tgt_str = self.titles[index], self.sent_summarizations[index]
        if self.config.textonly:
            input_lens = torch.tensor(self.lens[index])
            img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(self.config.avg_img_npz)
            img_useful = 1
        else:
            input_lens = torch.tensor(self.lens[index])
            img_index = str(index + 1)
            img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(
                self.images_dir + '/{}.npz'.format(img_index))
            img_useful = 1
        # input_ids, input_poses, dec_batch, target_batch, dec_padding_mask, dec_len, copy_position = self._get_txt_feat(
        #     index)
        with open(self.txt_feat_dir+'/{}.pkl'.format(index),'rb') as pf:
            input_ids, dec_batch, target_batch, dec_padding_mask, dec_len, \
            copy_position,phrase_copy_score = pickle.load(pf)
            pf.close()
        phrase_padding_mask = torch.ones_like(phrase_copy_score)


        if self.config.key_w_loss:
            dec_pos = [self.config.key_w_pos[0]] + self.dec_poses[index]
            dec_pos_f = [
                self.config.key_loss_weight[0] if p in self.config.key_w_pos else self.config.key_loss_weight[1]
                for p in dec_pos]
            dec_pos_f = torch.tensor(dec_pos_f)
            if len(dec_pos_f) > dec_len:
                dec_pos_f = dec_pos_f[:dec_len]
            assert len(dec_pos_f) == dec_len
        else:
            dec_pos_f = torch.tensor([0.0])
        len_txt = len(input_ids)
        attn_masks = torch.ones(len_txt + num_bb, dtype=torch.long)
        soft_labels = torch.tensor(soft_labels)
        with open(self.phrase_dir+'/{}.pkl'.format(index),'rb') as pf:
            phrase_tensor = pickle.load(pf)
            pf.close()
            if phrase_tensor.shape[0]==1 and phrase_tensor.sum()==0:
                phrase_tensor[0][0]=1


        return index, input_lens, input_ids, copy_position, phrase_tensor,phrase_copy_score, phrase_padding_mask, \
               None, img_feat, soft_labels, img_useful, img_pos_feat, attn_masks, \
               dec_batch, target_batch, dec_padding_mask, dec_len, dec_pos_f,input_str, tgt_str

    def _get_img_feat(self, filename):
        name, dump, nbb = load_npz(self.config.conf_th,
                                   self.config.max_bb,
                                   self.config.min_bb,
                                   self.config.num_bb,
                                   filename)
        img_feat = dump['features']
        img_bb = dump['norm_bb']
        soft_labels = dump['soft_labels']

        img_feat = torch.tensor(img_feat[:nbb, :]).float()
        img_bb = torch.tensor(img_bb[:nbb, :]).float()

        img_bb = torch.cat([img_bb, img_bb[:, 4:5] * img_bb[:, 5:]], dim=-1)

        return img_feat, img_bb, nbb, soft_labels

    def _get_name_to_nbb(self, image_dir):
        name2nbb = {}
        pts = glob.glob(image_dir+'/*.npz')
        for pt in pts:
            name, dump, nbb = load_npz(self.config.conf_th,
                                       self.config.max_bb,
                                       self.config.min_bb,
                                       self.config.num_bb,
                                       pt)
            name = pt.split('/')[-1]
            name2nbb[name] = nbb

        return name2nbb

    def merge_sub_word(self, sentence):
        sub_words = self.tokenizer.convert_ids_to_tokens(sentence)
        words = []
        i = 0
        len_sub = len(sub_words)
        cur_word = ''
        while i < len_sub:
            if sub_words[i].startswith('##'):
                cur_word = cur_word + sub_words[i][2:]
            else:
                if len(cur_word) != 0:
                    words.append(cur_word)
                cur_word = sub_words[i]
            i = i + 1
        if len(cur_word) != 0:
            words.append(cur_word)
        return words, sub_words

    def _get_ids_and_lens(self, titles, summaris):
        assert len(titles) == len(summaris)
        lens = []
        input_ids = []
        dec_poses = []
        dec_ids = []
        # import pdb
        # pdb.set_trace()
        # 1. judge generate or not
        if not os.path.exists(self.phrase_dir):
            os.mkdir(self.phrase_dir)
        phrase_pts = glob.glob(self.phrase_dir+'/*.pkl')
        if len(phrase_pts)<len(titles):
            phrase_gen = True
            self.nlp = StanfordCoreNLP('/home/meihuan2/download/stanford-corenlp-full-2018-02-27')
        else:
            phrase_gen = False
            self.nlp = None

        if not os.path.exists(self.txt_feat_dir):
            os.mkdir(self.txt_feat_dir)
        txtfeat_pts = glob.glob(self.txt_feat_dir+'/*.pkl')
        if len(txtfeat_pts)<len(titles):
            txtfeat_gen = True
        else:
            txtfeat_gen = False

        for ind in range(len(titles)):
            inp = self.bert_tokenize(titles[ind].strip())
            if len(inp) > self.config.max_txt_len:
                inp = [self.cls_] + inp[:self.config.max_txt_len] + [self.sep]
            else:
                inp = [self.cls_] + inp + [self.sep]

            input_ids.append(inp)
            lens.append(len(inp))

            dec = self.bert_tokenize(summaris[ind].strip())
            if self.config.key_w_loss:
                dec_pos, org_pos, dec_to_tokens = self.get_inp_pos(dec)
            else:
                dec_pos = [0]
            dec_ids.append(dec)
            dec_poses.append(dec_pos)

            #Phrase Generate
            if phrase_gen:
                if os.path.exists(self.phrase_dir+'/{}.pkl'.format(ind)):
                    pass
                # else:
                #     import pdb
                #     pdb.set_trace()
                else:
                    if ind % 500 == 0:
                        print("Phrase Generator: {}/{}".format(ind, len(titles)))
                    inp_to_tokens, sub_words = self.merge_sub_word(inp)
                    phrase_tensor = structure_parse.to_phrase(self.nlp, inp_to_tokens, sub_words)
                    with open(self.phrase_dir + '/{}.pkl'.format(ind), 'wb') as pf:
                        pickle.dump(phrase_tensor, pf)
                        pf.close()

            # Txt Feat Generate
            if txtfeat_gen:
                if os.path.exists(self.txt_feat_dir+'/{}.pkl'.format(ind)):
                    pass
                else:
                    with open(self.phrase_dir + '/{}.pkl'.format(ind), 'rb') as pf:
                        phrase_tensor = pickle.load(pf)
                        pf.close()
                    if ind % 500 == 0:
                        print("TxtFeat Generator: {}/{}".format(ind, len(titles)))
                    inp_ids, dec_inp, dec_tgt, dec_padding_mask, dec_len, copy_position = \
                        self.F_get_txt_feat(inp, dec)
                    phrase_copy = torch.zeros_like(phrase_tensor)
                    phrase_len = phrase_tensor.shape[0]
                    words_len = inp_ids.shape[0]
                    phrase2vocab = phrase_tensor * inp_ids.view(1,-1).expand(phrase_len,words_len)
                    for dt in dec_tgt:
                        if dt==0:
                            break
                        phrase_copy = phrase_copy+ (phrase2vocab==dt).long()
                    phrase_copy = phrase_copy.ge(1).long()
                    phrase_copy_score = phrase_copy.sum(dim=1)/phrase_tensor.sum(dim=1)
                    with open(self.txt_feat_dir + '/{}.pkl'.format(ind), 'wb') as pf:
                        pickle.dump((inp_ids,
                                     dec_inp,
                                     dec_tgt,
                                     dec_padding_mask,
                                     dec_len,
                                     copy_position,
                                     phrase_copy_score), pf)
                        pf.close()

        if self.nlp!=None:
            self.nlp.close()
        return input_ids, None, lens, dec_ids, dec_poses

    def get_inp_pos(self, inp):
        inp_to_tokens, sub_words = merge_sub_word(self.tokenizer, inp)
        org_pos = eng_model(" ".join(inp_to_tokens))
        org_pos = [w.pos_ for w in org_pos]
        cur_index = 0
        inp_pos = []
        for subw in sub_words:
            if subw.startswith('##'):
                inp_pos.append(org_pos[cur_index-1])
            else:
                inp_pos.append(org_pos[cur_index])
                cur_index = cur_index+1
        return inp_pos, org_pos, inp_to_tokens

    def get_inp_ef(self, ind, inp):
        inp_to_tokens, sub_words = merge_sub_word(self.tokenizer, inp)
        if len(self.mg_ef[ind])!=0:
            org_str, org_pos = self.mg_ef[ind]
        else:
            org_str, org_pos = [],[]
        cur_index = 0
        inp_pos = []
        for iter, subw in enumerate(sub_words):
            if cur_index>=len(org_pos):
                inp_pos.append(0)
                continue
            if subw.startswith('##') and subw.replace('##','') in org_str[cur_index - 1]:
                inp_pos.append(org_pos[cur_index - 1])
            elif subw in org_str[cur_index]:
                inp_pos.append(org_pos[cur_index])
                cur_index = cur_index + 1
            else:
                cur_index = cur_index + 1
        return inp_pos

    def _get_txt_feat(self,index):
        input_ids = torch.tensor(self.input_ids[index])
        # input_poses = self.input_poses[index]
        _dec_id = self.dec_ids[index]

        dec_inp, dec_tgt = self.get_dec_inp_targ_seqs(_dec_id,
                                                      self.config.max_dec_steps,
                                                      self.cls_,
                                                      self.sep)
        dec_len = len(dec_inp)
        dec_inp, dec_tgt = self.pad_decoder_inp_targ(self.config.max_dec_steps,
                                                     0,
                                                     dec_inp,
                                                     dec_tgt)
        dec_inp = torch.tensor(dec_inp)
        dec_tgt = torch.tensor(dec_tgt)
        dec_padding_mask = torch.ones((dec_len))
        dec_len = torch.tensor(dec_len)
        copy_position = torch.zeros(input_ids.shape[0])
        for i, x in enumerate(input_ids):
            if x in dec_tgt and x!=self.sep:
                copy_position[i] = 1
        return input_ids ,None, dec_inp, dec_tgt, dec_padding_mask, dec_len, copy_position

    def F_get_txt_feat(self,input_ids, _dec_id):
        # input_ids = torch.tensor(self.input_ids[index])
        # # input_poses = self.input_poses[index]
        # _dec_id = self.dec_ids[index]
        input_ids = torch.tensor(input_ids)
        dec_inp, dec_tgt = self.get_dec_inp_targ_seqs(_dec_id,
                                                      self.config.max_dec_steps,
                                                      self.cls_,
                                                      self.sep)
        dec_len = len(dec_inp)
        dec_inp, dec_tgt = self.pad_decoder_inp_targ(self.config.max_dec_steps,
                                                     0,
                                                     dec_inp,
                                                     dec_tgt)
        dec_inp = torch.tensor(dec_inp)
        dec_tgt = torch.tensor(dec_tgt)
        dec_padding_mask = torch.ones((dec_len))
        dec_len = torch.tensor(dec_len)
        copy_position = torch.zeros(input_ids.shape[0])
        for i, x in enumerate(input_ids):
            if x in dec_tgt and x!=self.sep:
                copy_position[i] = 1
        return input_ids ,dec_inp, dec_tgt, dec_padding_mask, dec_len, copy_position

    def bert_tokenize(self, text):
        ids = []
        for word in text.strip().split():
            ws = self.tokenizer.tokenize(word)
            if not ws:
                # some special char
                continue
            ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
        return ids

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id,numericalized_inp,numericalized_tgt):
        while len(numericalized_inp) < max_len:
            numericalized_inp.append(pad_id)
        while len(numericalized_tgt) < max_len:
            numericalized_tgt.append(pad_id)
        return numericalized_inp,numericalized_tgt

def vqa_eval_collate(inputs):
    def sorted_batch(inputs):
        (qids, input_lens, input_ids, copy_position,
         phrase_tensor,phrase_copy_score, phrase_padding_mask,
         input_poses, img_feats, soft_labels, img_useful, img_pos_feats, attn_masks,
         dec_batch, target_batch, dec_padding_mask,
         dec_len, dec_pos_f
         ) = map(list, unzip(inputs))
        input_lens = torch.stack(input_lens, dim=0)
        sorted_input_lens = torch.argsort(input_lens)
        qids = [qids[i] for i in sorted_input_lens]
        input_ids = [input_ids[i] for i in sorted_input_lens]
        img_feats = [img_feats[i] for i in sorted_input_lens]
        img_pos_feats = [img_pos_feats[i] for i in sorted_input_lens]
        attn_masks = [attn_masks[i] for i in sorted_input_lens]
        dec_batch = [dec_batch[i] for i in sorted_input_lens]
        target_batch = [target_batch[i] for i in sorted_input_lens]
        dec_padding_mask = [dec_padding_mask[i] for i in sorted_input_lens]
        return qids, input_lens, input_ids, img_feats, img_pos_feats, attn_masks, dec_batch, target_batch, dec_padding_mask, dec_len

    # (qids, input_lens, input_ids, img_feats, img_pos_feats, attn_masks, dec_batch, target_batch, dec_padding_mask, dec_len
    #  ) = sorted_batch(inputs)

    (qids, input_lens, input_ids, copy_position, phrase_tensor,phrase_copy_score, phrase_padding_mask,
     input_poses, img_feats, soft_labels, img_useful, img_pos_feats, attn_masks,
     dec_batch, target_batch, dec_padding_mask,
     dec_len, dec_pos_f,input_str, tgt_str
     ) = map(list, unzip(inputs))
    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    copy_position = pad_sequence(copy_position, batch_first=True, padding_value=0)
    copy_position = copy_position.long()
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    # phrase_tensor = pad_sequence(phrase_tensor, batch_first=True, padding_value=0)
    _phrase_tensor = []
    _phrase_copy_score = []
    p_lens = [pte.shape[1] for pte in phrase_tensor]
    phrase_nums = [pte.shape[0] for pte in phrase_tensor]
    max_p_len = max(p_lens)
    for pi, p_len in enumerate(p_lens):
        # p_len = p_lens[pi]
        pte = phrase_tensor[pi]
        if p_len < max_p_len:
            # print(max_p_len - p_len, p_len)
            cur_pad = nn.ZeroPad2d(padding=(0, max_p_len - p_len, 0, 0))
            pte = cur_pad(pte)
        _phrase_tensor.append(pte)
    _phrase_tensor = pad_sequence(_phrase_tensor, batch_first=True, padding_value=0)
    phrase_copy_score = pad_sequence(phrase_copy_score, batch_first=True, padding_value=0)
    phrase_padding_mask = pad_sequence(phrase_padding_mask, batch_first=True, padding_value=0)
    phrase_padding_mask = phrase_padding_mask.unsqueeze(2)

    # attn_masks_s0, attn_masks_s1 = attn_masks.shape
    # attn_masks = attn_masks.unsqueeze(1)
    # attn_masks = attn_masks.expand(attn_masks_s0, attn_masks_s1, attn_masks_s1)

    # if targets[0] is None:
    #     targets = None
    # else:
    #     targets = torch.stack(targets, dim=0)
    dec_batch = pad_sequence(dec_batch, batch_first=True, padding_value=0)
    targets = pad_sequence(target_batch, batch_first=True, padding_value=0)
    dec_padding_mask = pad_sequence(dec_padding_mask, batch_first=True, padding_value=0)
    dec_pos_f = pad_sequence(dec_pos_f, batch_first=True, padding_value=0)
    soft_labels = pad_sequence(soft_labels, batch_first=True, padding_value=0)
    soft_labels = soft_labels.float()

    num_bbs = [f.size(0) for f in img_feats]
    num_bbs = torch.tensor(num_bbs)
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
    dec_len = torch.stack(dec_len, dim=0)
    txt_lens = torch.tensor(txt_lens)
    img_useful = [base_config.useful_weight if im_u ==1  else base_config.useless_weight for im_u in img_useful]
    img_useful = torch.tensor(img_useful).unsqueeze(1).unsqueeze(1)

    max_nbb = max(num_bbs)
    ot_scatter = _compute_ot_scatter(txt_lens, max_tl, attn_masks.size(1))
    txt_pad = _compute_pad(txt_lens, max_tl)
    img_pad = _compute_pad(num_bbs, max_nbb)
    ot_inputs = {'ot_scatter': ot_scatter,
                 'scatter_max': ot_scatter.max().item(),
                 'txt_pad': txt_pad,
                 'img_pad': img_pad,
                 'input_str':input_str,
                 'tgt_str':tgt_str}

    batch = {'qids': qids,
             'input_ids': input_ids,
             'copy_position': copy_position,
             'phrase_tensor': _phrase_tensor,
             'max_phrase_len': max_p_len,
             'phrase_nums': phrase_nums,
             'phrase_copy_score': phrase_copy_score,
             'phrase_padding_mask': phrase_padding_mask,
             'input_poses': input_poses,
             'txt_lens':txt_lens,
             'num_bbs':num_bbs,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'soft_labels': soft_labels,
             'img_useful': img_useful,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             "dec_batch":dec_batch,
             "dec_len":dec_len,
             'targets': targets,
             'dec_mask':dec_padding_mask,
             'dec_pos_f':dec_pos_f,
             'ot_inputs':ot_inputs}
    # print("dataset.py 392:", batch['input_ids'][0])
    return batch

def get_data_loader(args, dev_image_path, dev_text_path, dev_summri_path, tokenizer,device, image_useful_file='', image_useless_file='', random_seed=1, is_test=False, balance_useful = False):
    BUCKET_SIZE = 8
    if is_test:
        train_dataset = MultiDataset(args,
                                     dev_image_path,
                                     dev_text_path,
                                     dev_summri_path,
                                     tokenizer,
                                     image_useful_file,
                                     image_useless_file,
                                     random_seed,
                                     is_test,
                                     balance_useful)
        # sampler = TokenBucketSampler(train_dataset.lens, bucket_size=BUCKET_SIZE,
        #                              batch_size=args.batch_size, droplast=False)
        eval_dataloader = DataLoader(train_dataset,
                                     # batch_sampler=sampler,
                                     # batch_size=4,
                                     shuffle=False,
                                     num_workers=args.n_workers,
                                     pin_memory=args.pin_mem,
                                     collate_fn=vqa_eval_collate)
        eval_dataloader = PrefetchLoader(eval_dataloader, device_id=device)
    else:
        train_dataset = MultiDataset(args,
                                     dev_image_path,
                                     dev_text_path,
                                     dev_summri_path,
                                     tokenizer,
                                     image_useful_file,
                                     image_useless_file,
                                     random_seed,
                                     is_test,
                                     balance_useful)
        sampler = TokenBucketSampler(train_dataset.lens, bucket_size=BUCKET_SIZE,
                                     batch_size=args.batch_size, droplast=False)
        eval_dataloader = DataLoader(train_dataset,
                                     batch_sampler=sampler,
                                     # batch_size=4,
                                     num_workers=args.n_workers,
                                     pin_memory=args.pin_mem,
                                     collate_fn=vqa_eval_collate)
        eval_dataloader = PrefetchLoader(eval_dataloader, device_id=device)
    return eval_dataloader



def main_tmp():
    from sampler import TokenBucketSampler
    BUCKET_SIZE = 8192

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    args = parser.parse_args()
    with open('./configs/config.json', 'r') as f:
        data = json.load(f)
        f.close()
    args.__dict__.update(data)
    train_dataset = MultiDataset(args, args.dev_image_path,
                                 args.dev_text_path,
                                 args.dev_summri_path,
                                 'image_useful_file',
                                 'image_useless_file',
                                 1,
                                 is_test=False)
    sampler = TokenBucketSampler(train_dataset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=args.batch_size, droplast=False)
    eval_dataloader = DataLoader(train_dataset,
                                 batch_sampler=sampler,
                                 # batch_size=args.batch_size,
                                 num_workers=args.n_workers,
                                 pin_memory=args.pin_mem,
                                 collate_fn=vqa_eval_collate)
    for idb, batch in enumerate(eval_dataloader):
        print("*******{}*******".format(idb))
        import pdb
        pdb.set_trace()
        print(batch['input_ids'])


if __name__ == '__main__':
    main_tmp()
