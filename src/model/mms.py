from __future__ import unicode_literals, print_function, division

import sys
import os
import json
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F
sys.path.append('../')
from src.model.TextEncoder import UniterForVisualSummEnc, UniterForText
from src.utils.const import BUCKET_SIZE, IMG_DIM
from src.utils.beam import Beam
from src.model.ot import optimal_transport_dist
from torch.nn.utils.rnn import pad_sequence

def init_lstm_wt(config, lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_wt_normal(config, wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_linear_wt(config, linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ImgPooler(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ImgPooler, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Attention(nn.Module):
    def __init__(self, config, input_dim, hidden_dim):
        super(Attention, self).__init__()
        # attention
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.decode_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1, bias=False)
        init_linear_wt(config, self.decode_proj)
        init_linear_wt(config, self.v)



    def forward(self, s_t_hat, encoder_outputs, encoder_feature,coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim

        e = F.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k
        # print("Internal-Shape-enc pad mask:{}  enc_out:{}".format(enc_padding_mask.shape,
        #                                                  encoder_outputs.shape))
        attn_dist_ = F.softmax(scores, dim=1)
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(b, -1)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k


        return c_t, attn_dist,coverage

class GRUDecoder(nn.Module):
    def __init__(self, config, vocab_size,embedding_dim,hidden_dim):
        super(GRUDecoder, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.attention_network = Attention(config, config.hidden_dim, config.hidden_dim)
        # decoder
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        init_wt_normal(config, self.embedding.weight)

        self.x_context = nn.Linear(self.hidden_dim + self.embedding_dim, self.embedding_dim)
        self.W_h = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.GRU_layer = nn.GRU(self.embedding_dim,self.hidden_dim,batch_first=True)

        init_lstm_wt(config, self.GRU_layer)


        # p_vocab
        self.out1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.out2 = nn.Linear(self.hidden_dim, self.vocab_size)
        init_linear_wt(config, self.out2)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, y_t_1,
                s_t_0,
                s_t_1,
                encoder_outputs,
                c_t_1,
                coverage,
                step):
        y_t_1_embd = self.embedding(y_t_1)
        y_t_1_embd = y_t_1_embd.detach()

        y_t_1_embd = self.dropout(y_t_1_embd)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        self.GRU_layer.flatten_parameters()
        lstm_out, s_t = self.GRU_layer(x.unsqueeze(1), s_t_1)

        # h_decoder, c_decoder = s_t
        # s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_dim),
        #                      c_decoder.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim
        encoder_feature = encoder_outputs.view(-1, self.hidden_dim)
        encoder_feature = self.W_h(encoder_feature)
        s_t_hat=s_t.squeeze(0)
        c_t, attn_dist ,coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               coverage)


        if step > 0:
            coverage = coverage_next

        output = torch.cat((lstm_out.view(-1, self.hidden_dim), c_t), 1)  # B x hidden_dim * 3
        output = self.out1(output)  # B x hidden_dim

        # output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist,coverage

class LossGlobalPart(nn.Module):
    def __init__(self, config, hidden_dim):
        super(LossGlobalPart, self).__init__()
        self.config = config
        self.fc_vg = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mg = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.out =  nn.Linear(hidden_dim,1)
        self.pad = False

    def forward(self, hidden_state, txt_enc, img_enc):
        vg_score = self.fc_vg(hidden_state)
        vg_score = self.activation(vg_score)
        vg_score = self.dropout(vg_score)
        vg_score = self.out(vg_score)

        mg_score = self.fc_mg(hidden_state)
        mg_score = self.activation(mg_score)
        mg_score = self.dropout(mg_score)
        mg_score = self.out(mg_score)

        scores = torch.cat((vg_score, mg_score), dim=1)
        scores = F.softmax(scores, dim=-1)
        # scores = F.sigmoid(scores)
        return scores

class MultiModal(nn.Module):
    def __init__(self, config, tokenizer, model_file_path=None, is_eval=False):
        super(MultiModal, self).__init__()
        self.config = config
        vocab_size=len(tokenizer.vocab)
        meta = json.load(open(config.meta_file, 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.tokenizer = tokenizer
        embedding_dim= config.hidden_dim
        hidden_dim= config.hidden_dim

        ckpt_file = config.checkpoint
        checkpoint = torch.load(ckpt_file)
        mutual_score = config.mutual_score
        mutual_detach = config.mutual_layer_detach
        self.cross_encoder = UniterForVisualSummEnc.from_pretrained(
            config.model_config, checkpoint, img_dim=IMG_DIM, mutual_layer = config.mutual_layer,
            mutual_score = mutual_score, mutual_detach = mutual_detach,
            ot_layer = config.ot_layer, ot_detach = config.ot_layer_detach,
            phrase_layer=config.phrase_layer, phrase_score=config.phrase_score, phrase_detach=config.phrase_detach,
            output_all_encoded_layers = config.cross_output_all_encoded_layers)
        if config.mutual_kl_loss or config.mutual_phrase_loss or config.pre_filter:
            self.text_encoder = UniterForText.from_pretrained(
                config.model_config, checkpoint, img_dim=IMG_DIM,
                mutual_layer=config.mutual_layer,mutual_score=mutual_score, mutual_detach=mutual_detach,
                ot_layer = config.ot_layer, ot_detach = config.ot_layer_detach,
                phrase_layer=config.phrase_layer, phrase_score=config.phrase_score, phrase_detach=config.phrase_detach,
                output_all_encoded_layers=config.cross_output_all_encoded_layers
            )
            if config.params_sharing:
                txt_encoder_layer = max(config.mutual_layer, config.phrase_layer, config.ot_layer)
                self.text_encoder.uniter.encoder.layer = self.cross_encoder.uniter.encoder.layer[:txt_encoder_layer]
            self.text_encoder.uniter.encoder.layer.eval()

        self.cross_encoder.uniter.encoder.layer[:config.mutual_layer].eval()
        self.decoder=GRUDecoder(config, vocab_size,embedding_dim,hidden_dim)

        # shared the embedding between encoder and decoder
        self.decoder.embedding.weight = self.cross_encoder.uniter.embeddings.word_embeddings.weight
        if config.loss_modulate:
            self.loss_scorer = LossGlobalPart(config, hidden_dim)
        else:
            self.loss_scorer = None
        if is_eval:
            self.cross_encoder = self.cross_encoder.eval()
            self.decoder = self.decoder.eval()
            if config.loss_modulate:
                self.loss_scorer = self.loss_scorer.eval()
            if config.mutual_kl_loss:
                self.text_encoder = self.text_encoder.eval()


        if model_file_path is not None:
            self.setup_train(model_file_path)

        if config.score_ref:
            self.txt_pooler = Pooler(config.hidden_dim)
            self.txt_img_pooler = Pooler(config.hidden_dim)
            self.out_pooler = Pooler(config.hidden_dim)
            init_linear_wt(config, self.txt_pooler.dense)
            init_linear_wt(config, self.txt_img_pooler.dense)
            init_linear_wt(config, self.out_pooler.dense)
        if config.decoder_init and config.decoder_init_methods=='global':
            self.img_pooler = ImgPooler(2048, config.hidden_dim)
        elif config.decoder_init and config.decoder_init_methods=='softlb':
            self.img_pooler = ImgPooler(1601, config.hidden_dim)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        if config.mutual_cosine:
            self.cosine_2 = nn.CosineSimilarity(dim=2, eps=1e-6)
        if config.consistency_ways=='cosine':
            self.cosine_1 = nn.CosineSimilarity(dim=1, eps=1e-6)

    def setup_train(self, model_file_path=None):
        start_iter, start_loss = 0,0
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            # start_iter = state['iter']
            # start_loss = state['current_loss']
            if "encoder_state_dict" in state:
                self.text_encoder.load_state_dict(state['encoder_state_dict'], strict=False)
                self.cross_encoder.load_state_dict(state['encoder_state_dict'], strict=False)
            if "text_encoder_state_dict" in state and self.config.mutual_kl_loss:
                self.text_encoder.load_state_dict(state['text_encoder_state_dict'], strict=False)
            if "cross_encoder_state_dict" in state:
                self.cross_encoder.load_state_dict(state['cross_encoder_state_dict'], strict=False)
            if "loss_scorer_state_dict" in state:
                self.loss_scorer.load_state_dict(state['loss_scorer_state_dict'], strict=False)
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
        return start_iter, start_loss

    def get_ot_dist(self, batch, encoder_output):
        ot_inputs = batch['ot_inputs']
        if ot_inputs is not None:
            ot_scatter = ot_inputs['ot_scatter']

            b = encoder_output.size(0)
            tl = batch['input_ids'].size(1)
            il = batch['img_feat'].size(1)
            max_l = max(ot_inputs['scatter_max'] + 1, tl+il)

            ot_scatter = ot_scatter.unsqueeze(-1).expand_as(encoder_output)
            ctx_emb = torch.zeros(b, max_l, self.config.hidden_dim,
                                  dtype=encoder_output.dtype,
                                  device=encoder_output.device
                                  ).scatter_(dim=1, index=ot_scatter,
                                             src=encoder_output)
            txt_emb = ctx_emb[:, :tl, :]
            img_emb = ctx_emb[:, tl:tl+il, :]

            txt_pad = ot_inputs['txt_pad']
            img_pad = ot_inputs['img_pad']
            txt_pad = txt_pad.bool()
            img_pad = img_pad.bool()
            # NOTE: run in fp32 for stability
            ot_dist, cost = optimal_transport_dist(txt_emb.float(), img_emb.float(),
                                             txt_pad, img_pad)
            ot_dist = ot_dist.to(txt_emb)
            cost = cost.to(txt_emb)
        else:
            ot_dist = None
            cost = None
        return ot_dist, cost

    def get_attn_ctx(self, batch, attn):
        ot_inputs = batch['ot_inputs']
        ot_scatter = ot_inputs['ot_scatter']

        b = attn.size(0)
        tl = batch['input_ids'].size(1)
        il = batch['img_feat'].size(1)
        max_l = max(ot_inputs['scatter_max'] + 1, tl + il)

        ot_scatter = ot_scatter.unsqueeze(-1).expand_as(attn)
        attn_emb1 = torch.zeros(b, max_l, max_l,
                              dtype=attn.dtype,
                              device=attn.device
                              ).scatter_(dim=1, index=ot_scatter,
                                         src=attn)
        attn_emb = torch.zeros(b, max_l, max_l,
                              dtype=attn.dtype,
                              device=attn.device
                              ).scatter_(dim=2, index=ot_scatter.transpose(1,2),
                                         src=attn_emb1)
        attn_t2i_RU = attn_emb[:, :tl, tl:tl+il]
        attn_t2i_LD = attn_emb[:, tl:tl+il, :tl]
        attn_t2i = (attn_t2i_RU + attn_t2i_LD.transpose(1,2))/2
        return attn_t2i

    def gen_attn(self, batch):
        attn_masks = batch['attn_masks']
        # input_poses = batch['input_poses']
        batch_size = len(batch['qids'])
        txt_lens = batch['txt_lens']
        img_lens = batch['num_bbs']
        attn_masks_s0, attn_masks_s1 = attn_masks.shape

        # attn_masks_ones = torch.ones((attn_masks_s0, attn_masks_s1, attn_masks_s1), dtype=torch.long)
        # attn_masks_ones = attn_masks_ones.to(device=attn_masks.device)
        # attn_masks = attn_masks.unsqueeze(1)
        # attn_masks = attn_masks_ones * attn_masks

        attn_vis_RU = torch.zeros((attn_masks_s0, attn_masks_s1, attn_masks_s1), dtype=torch.long)
        attn_vis_RU = attn_vis_RU.to(device=attn_masks.device)
        # attn_vis_LD = torch.zeros((attn_masks_s0, attn_masks_s1, attn_masks_s1), dtype=torch.long)
        # attn_vis_LD = attn_vis_LD.to(device=attn_masks.device)

        # attn_vis_self = torch.ones((attn_masks_s0, attn_masks_s1, attn_masks_s1), dtype=torch.long)
        # attn_vis_self = attn_vis_self.to(device=attn_masks.device)

        # attn_vis_all = torch.zeros((attn_masks_s0, attn_masks_s1), dtype=torch.long)
        # attn_vis_all = attn_vis_all.to(device=attn_masks.device)
        attn_txt_all = torch.zeros((attn_masks_s0, attn_masks_s1), dtype=torch.long)
        attn_txt_all = attn_txt_all.to(device=attn_masks.device)


        attn_txt_LU= torch.zeros((attn_masks_s0, attn_masks_s1, attn_masks_s1), dtype=torch.long)
        attn_txt_LU = attn_txt_LU.to(device=attn_masks.device)

        # attn_eys = torch.eye((attn_masks_s1), dtype=torch.long)
        # attn_eys = attn_eys.to(device=attn_masks.device)
        # attn_eys = attn_eys.unsqueeze(0)
        # attn_eys = attn_eys.expand(attn_masks_s0, attn_masks_s1, attn_masks_s1)
        # _attn_eys = 1- attn_eys
        for b in range(batch_size):
            # attn_vis_self[b,:txt_lens[b],:] = 0
            # attn_vis_self[b,:,:txt_lens[b]] = 0

            # attn_vis_all[b,txt_lens[b]:txt_lens[b]+img_lens[b]] = 1
            attn_txt_all[b, :txt_lens[b]] = 1
            attn_txt_LU[b, :txt_lens[b], :txt_lens[b]] = 1

            attn_vis_RU[b, :txt_lens[b], txt_lens[b]:txt_lens[b]+img_lens[b]] = 1
            # attn_vis_LD[b, txt_lens[b]:, :txt_lens[b]] = 1

        # attn_vis_self = attn_vis_self * _attn_eys
        return attn_vis_RU, attn_txt_all, attn_txt_LU

    def get_mutual_info(self, text_output_distri, mutual_encoder_distri, copy_position):
        seq_len = text_output_distri.shape[1]
        copy_len = copy_position.shape[1]
        pad = nn.ZeroPad2d(padding=(0, seq_len-copy_len, 0, 0))
        text_output_distri = torch.gather(text_output_distri, 2, copy_position.unsqueeze(2)).squeeze()
        mutual_encoder_distri = torch.gather(mutual_encoder_distri, 2, copy_position.unsqueeze(2)).squeeze()
        if len(text_output_distri.shape)==1:
            text_output_distri = text_output_distri.unsqueeze(0)
        if len(mutual_encoder_distri.shape)==1:
            mutual_encoder_distri = mutual_encoder_distri.unsqueeze(0)
        text_output_distri = pad(text_output_distri.log())
        mutual_encoder_distri = pad(mutual_encoder_distri.log())
        mutual_info = text_output_distri - mutual_encoder_distri
        """
        text_output_distri = F.softmax(text_output, dim=-1)
        text_output_entr = -(text_output_distri * text_output_distri.log()).sum(dim=-1)
        # mutual_encoder_distri = F.softmax(mutual_encoder_layer, dim=-1)
        mutual_encoder_entr = -(mutual_encoder_distri * mutual_encoder_distri.log()).sum(dim=-1)
        if self.config.mutual_positive:
            mutual_info = mutual_encoder_entr - text_output_entr
        else:
            mutual_info = -mutual_encoder_entr + text_output_entr
        """
        return mutual_info

    def get_phrase_info(self, batch, txt_phrase_score, cross_phrase_score):
        phrase_target_score = batch['phrase_copy_score'].unsqueeze(2)
        txt_phrase_dist = (txt_phrase_score - phrase_target_score).abs()
        cross_phrase_dist = (cross_phrase_score - phrase_target_score).abs()
        phrase_attn = txt_phrase_dist - cross_phrase_dist
        phrase_attn2w = phrase_attn * batch['phrase_tensor']
        phrase_padding_mask = batch['phrase_padding_mask']
        b, pl, _ = phrase_padding_mask.shape
        # phrase_padding_mask = phrase_padding_mask.expand(b,pl,batch['max_phrase_len'])
        phrase_attn2w = phrase_attn2w + (1 - batch['phrase_tensor']) * -10000
        phrase_attn2w = phrase_attn2w.max(dim=1).values
        return phrase_attn2w, txt_phrase_score, cross_phrase_score

    def get_mutual_info_cos(self, text_output, mutual_encoder_layer, copy_position):
        sim = self.cosine_2(text_output, mutual_encoder_layer)
        sim = 1- sim
        return sim

    def get_slice_modals(self, batch, encoder_output):
        batch_size = len(batch['qids'])
        txt_lens = batch['txt_lens']
        img_lens = batch['num_bbs']
        img_dim = batch['img_feat'].shape[1]
        txt_enc = []
        img_enc = []
        for b in range(batch_size):
            txt_pos = txt_lens[b]
            img_pos = txt_pos + img_lens[b]
            txt_enc.append(encoder_output[b,txt_pos-1,:])
            img_enc.append(encoder_output[b,img_pos-1,:])
        txt_enc = torch.stack(txt_enc)
        img_enc = torch.stack(img_enc)
        return txt_enc, img_enc

    def get_vis_attn(self, config, all_attention_scores,mutual_info, attn_vis_RU, attn_txt_all, img_lens):
        numls = config.mutual_layer
        mutual_guiding_nums = config.mutual_guiding_nums
        all_attention_scores = all_attention_scores[numls:numls+mutual_guiding_nums]
        all_attention_scores = [attn.mean(dim=1) for attn in all_attention_scores]
        avg_attention_scores = sum(all_attention_scores)/len(all_attention_scores)

        visual_attn_scores = avg_attention_scores*attn_vis_RU+(avg_attention_scores*attn_vis_RU.transpose(-1,-2)).transpose(-1,-2)
        visual_attn_scores = visual_attn_scores.sum(dim=-1)/(img_lens.unsqueeze(1))

        visual_attn_scores = visual_attn_scores + attn_vis_RU[:, 0] * -10000
        visual_attn_scores = F.softmax(visual_attn_scores, dim=-1)
        visual_attn_scores = visual_attn_scores + 1e-6

        # mutual_info = mutual_info
        mask_mutual = (1 - attn_txt_all) * -10000
        mutual_info = mutual_info + mask_mutual
        mutual_info = F.softmax(mutual_info, dim=-1)
        kl_loss = F.kl_div(visual_attn_scores.log(), mutual_info, reduction='none').sum(dim=-1)
        # kl_loss = kl_loss.mean()
        return kl_loss

    def get_phrase_attn(self, config, all_attention_scores,phrase_info, phrase_tensor, attn_vis_RU, attn_txt_all, img_lens):
        numls = config.phrase_layer
        mutual_guiding_nums = config.mutual_guiding_nums
        all_attention_scores = all_attention_scores[numls:numls+mutual_guiding_nums]
        all_attention_scores = [attn.mean(dim=1) for attn in all_attention_scores]
        avg_attention_scores = sum(all_attention_scores)/len(all_attention_scores)

        visual_attn_scores = avg_attention_scores*attn_vis_RU+(avg_attention_scores*attn_vis_RU.transpose(-1,-2)).transpose(-1,-2)
        visual_attn_scores = visual_attn_scores.sum(dim=-1)/(img_lens.unsqueeze(1))

        visual_attn_scores = visual_attn_scores + attn_vis_RU[:, 0] * -10000
        txt_len = phrase_tensor.shape[2]
        phrase_tensor_mask = phrase_tensor.max(dim=1).values
        visual_attn_scores = visual_attn_scores[:,:txt_len] + (1-phrase_tensor_mask) * -10000
        visual_attn_scores = F.softmax(visual_attn_scores, dim=-1)
        visual_attn_scores = visual_attn_scores + 1e-6

        # phrase_info
        phrase_info = F.softmax(phrase_info, dim=-1)
        phrase_loss = F.kl_div(visual_attn_scores.log(), phrase_info, reduction='none').sum(dim=-1)
        # kl_loss = kl_loss.mean()
        return phrase_loss

    def get_supplement_attn(self, config, batch, all_attention_scores,cost, attn_vis_RU, attn_txt_all, img_lens):
        numls = config.ot_layer
        tl = batch['input_ids'].size(1)
        il = batch['img_feat'].size(1)
        mutual_guiding_nums = config.mutual_guiding_nums
        all_attention_scores = all_attention_scores[numls:numls + mutual_guiding_nums]
        all_attention_scores = [attn.mean(dim=1) for attn in all_attention_scores]
        avg_attention_scores = sum(all_attention_scores)/len(all_attention_scores)
        # attn_t2i = self.get_attn_ctx(batch, avg_attention_scores)
        attn_t2i = avg_attention_scores * attn_vis_RU + (
                    avg_attention_scores * attn_vis_RU.transpose(-1, -2)).transpose(-1, -2)
        attn_t2i = attn_t2i.sum(dim=-1) / (img_lens.unsqueeze(1))

        attn_t2i = attn_t2i + (1-attn_txt_all) * -10000
        attn_t2i = attn_t2i[:,:tl]
        attn_t2i = F.softmax(attn_t2i, dim=-1)
        attn_t2i = attn_t2i + 1e-6

        # mutual_info = mutual_info
        mask_mutual = (1 - attn_txt_all[:,:tl]) * -10000
        cost = cost.sum(dim=-1) / (img_lens.unsqueeze(1))
        cost_info = cost + mask_mutual
        cost_info = F.softmax(cost_info, dim=-1)
        cos_loss = F.kl_div(attn_t2i.log(), cost_info, reduction='none').sum(dim=-1)
        # kl_loss = kl_loss.mean()
        return cos_loss

    def get_abandon_attn(self, all_attention_scores,attn_vis_RU, attn_txt_LU, img_lens, txt_lens):
        # numls = config.ot_layer
        # tl = batch['input_ids'].size(1)
        # il = batch['img_feat'].size(1)
        # mutual_guiding_nums = config.mutual_guiding_nums
        all_attention_scores = all_attention_scores[9:10]
        all_attention_scores = [attn.mean(dim=1) for attn in all_attention_scores]
        avg_attention_scores = sum(all_attention_scores) / len(all_attention_scores)
        # attn_t2i = self.get_attn_ctx(batch, avg_attention_scores)
        avg_txt = avg_attention_scores*attn_txt_LU
        avg_txt = avg_txt.sum(dim=-1).sum(dim=-1)/(txt_lens*txt_lens)

        avg_img = avg_attention_scores+(1-attn_vis_RU)*-10000
        avg_img = avg_img.max(dim=-1).values.max(dim=-1).values

        filter = avg_txt>avg_img
        return filter.long()

    def forward(self, batch, encoder_training=False):
        if self.config.guiding:
            batch['img_feat'] = batch['img_feat'] * batch['img_useful']
        attn_vis_RU, attn_txt_all, attn_txt_LU = self.gen_attn(batch)
        batch['attn_vis_RU'] = attn_vis_RU
        batch['attn_txt_all'] = attn_txt_all
        copy_position = batch['copy_position']
        if self.config.mutual_kl_loss or self.config.mutual_phrase_loss or self.config.pre_filter:
            txt_encoder_output,txt_phrase_score, txt_ot_layer = \
                self.text_encoder(batch, training=encoder_training)
            # txt_embedding_output, txt_attn_masks, txt_all_attention_scores = \
        else:
            txt_encoder_output,txt_phrase_score, txt_ot_layer = None, None, None
        encoder_output, pooled_output, embedding_output, attn_masks, all_attention_scores, \
        mutual_encoder_layer, ot_encoder_layer, cross_phrase_score = \
            self.cross_encoder(batch,
                               present_params = (self.config.ot_layer, False, txt_ot_layer, attn_vis_RU,
                                                 batch['attn_masks'], batch['attn_txt_all'],self.config.ot_trunc),
                               training=encoder_training)
        # cross_pool = (ot_encoder_layer*batch['attn_masks'].unsqueeze(2)).sum(dim=1)
        # cross_pool = cross_pool/(batch['attn_masks'].sum(dim=1).unsqueeze(1))
        # txt_pool = (txt_ot_layer * batch['attn_txt_all'].unsqueeze(2)).sum(dim=1)
        # txt_pool = txt_pool / (batch['attn_txt_all'].sum(dim=1).unsqueeze(1))
        # pool_diff = self.cos(cross_pool, txt_pool)
        # pool_diff_trunc = (pool_diff.topk(2, largest=False).values)[1]
        # pool_diff_bool = (pool_diff>pool_diff_trunc).long()
        if self.config.mutual_kl_loss:
            if self.config.mutual_cosine:
                mutual_info = self.get_mutual_info_cos(txt_encoder_output, mutual_encoder_layer, copy_position)
            else:
                mutual_info = self.get_mutual_info(txt_encoder_output, mutual_encoder_layer, copy_position)
        else:
            mutual_info = None

        if self.config.mutual_phrase_loss:
            phrase_info, txt_phrase_score, cross_phrase_score = self.get_phrase_info(batch, txt_phrase_score, cross_phrase_score)
        else:
            phrase_info, txt_phrase_score, cross_phrase_score = None, None, None

        dec_batch = batch['dec_batch']
        max_dec_len = max(batch['dec_len'])
        b, t_k, n = list(encoder_output.size())
        s_t_1 = pooled_output.unsqueeze(0)
        if self.config.decoder_init and self.config.decoder_init_methods=='global':
            s_t_1 = self.get_decoder_init(s_t_1, batch['img_feat'], batch['img_useful'])
        elif self.config.decoder_init and self.config.decoder_init_methods=='softlb':
            s_t_1 = self.get_decoder_init_softlabels(s_t_1, batch['soft_labels'], batch['img_useful'])
        s_t_0 = s_t_1.unsqueeze(0)
        c_t = Variable(torch.zeros((b, self.config.hidden_dim)))
        coverage = Variable(torch.zeros((b, t_k)))
        c_t = c_t.to(device = encoder_output.device)
        coverage = coverage.to(device = encoder_output.device)
        final_dists = []
        attn_dists = []
        Ias = []
        for di in range(min(max_dec_len, self.config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t, attn_dist , next_coverage = self.decoder(y_t_1,
                                                          s_t_0,
                                                          s_t_1,
                                                          encoder_output,
                                                          c_t,
                                                          coverage,
                                                          di)
            final_dists.append(final_dist)
            attn_dists.append(attn_dist)
        sim_loss = None
        ot_dist, cost = self.get_ot_dist(batch, ot_encoder_layer)
        if self.config.loss_modulate:
            txt_enc, img_enc = self.get_slice_modals(batch, encoder_output)
            loss_weight = self.loss_scorer(pooled_output, txt_enc, img_enc)
        else:
            loss_weight = torch.ones(b,2).to(device = encoder_output.device)

        return torch.stack(final_dists, dim=1), attn_dists, coverage, Ias,sim_loss, all_attention_scores, \
               mutual_info, txt_encoder_output, mutual_encoder_layer, \
               phrase_info, txt_phrase_score, cross_phrase_score, \
               ot_dist, cost, loss_weight

    def score_ref_loss(self, encoder_output,pooled_output,s_t_end, txt_lens, img_useful):
        bs =  encoder_output.shape[0]
        text_pooled = []
        for i in range(bs):
            txt_len = txt_lens[i]
            cur_pool = self.text_encoder.uniter.pooler(encoder_output[i:i+1,:txt_len,:])
            text_pooled.append(cur_pool)
        text_pooled = torch.stack(text_pooled,dim=0)

        emb_text = self.txt_pooler(text_pooled)
        emb_img_txt = self.txt_img_pooler(pooled_output)
        emb_out = self.out_pooler(s_t_end)
        sc_txt = self.cos(emb_text.squeeze(1), emb_out)
        sc_img_txt =self.cos(emb_img_txt, emb_out)

        img_use = img_useful[:,0,0]
        ones=torch.ones_like(img_use).float()
        neg_ones = -ones
        zeros = torch.zeros_like(ones).float()
        sim_loss_w = torch.where(img_use>=1, ones, neg_ones)
        sim_loss = -sim_loss_w*sc_img_txt
        sim_loss = torch.where(sim_loss>=0, sim_loss,zeros)

        return sim_loss

    def get_decoder_init(self, s_t_1, img_feat, img_useful):
        # 'global method init the s_t_0'
        img_pooled = self.img_pooler(img_feat)
        img_pooled = torch.mean(img_pooled, dim=1)
        img_pooled_use = (img_pooled.unsqueeze(1)*img_useful).transpose(0,1)
        _s_t_1 = img_pooled_use + s_t_1
        return _s_t_1

    def get_decoder_init_softlabels(self, s_t_1, soft_labels, img_useful):
        img_pooled = self.img_pooler(soft_labels)
        img_pooled = torch.mean(img_pooled, dim=1)
        img_pooled_use = (img_pooled.unsqueeze(1) * img_useful).transpose(0, 1)
        _s_t_1 = img_pooled_use + s_t_1
        return _s_t_1

    def decode(self, batch):
        if self.config.guiding:
            batch['img_feat'] = batch['img_feat'] * batch['img_useful']
        attn_vis_RU, attn_txt_all, attn_txt_LU = self.gen_attn(batch)
        batch['attn_vis_RU'] = attn_vis_RU
        batch['attn_txt_all'] = attn_txt_all
        copy_position = batch['copy_position']

        if self.config.mutual_kl_loss or self.config.mutual_phrase_loss or self.config.pre_filter:
            txt_encoder_output,txt_phrase_score, txt_ot_layer = \
                self.text_encoder(batch)
            # txt_embedding_output, txt_attn_masks, txt_all_attention_scores = \

        else:
            txt_encoder_output,txt_phrase_score, txt_ot_layer = None, None, None
        encoder_output, pooled_output, embedding_output, attn_masks, all_attention_scores, \
        mutual_encoder_layer, ot_encoder_layer, cross_phrase_score = \
            self.cross_encoder(batch,
                               present_params=(self.config.ot_layer, False, txt_ot_layer, attn_vis_RU, batch['attn_masks'],
                                               batch['attn_txt_all'],self.config.ot_trunc))
        if self.config.mutual_kl_loss:
            if self.config.mutual_cosine:
                mutual_info = self.get_mutual_info_cos(txt_encoder_output, mutual_encoder_layer, copy_position)
            else:
                mutual_info = self.get_mutual_info(txt_encoder_output, mutual_encoder_layer, copy_position)
        else:
            mutual_info = None

        if self.config.mutual_phrase_loss:
            phrase_info, txt_phrase_score, cross_phrase_score = self.get_phrase_info(batch, txt_phrase_score, cross_phrase_score)
        else:
            phrase_info, txt_phrase_score, cross_phrase_score = None, None, None

        max_dec_len = max(batch['dec_len'])
        b, t_k, n = list(encoder_output.size())
        s_t_1 = pooled_output.unsqueeze(0)
        if self.config.decoder_init and self.config.decoder_init_methods == 'global':
            s_t_1 = self.get_decoder_init(s_t_1, batch['img_feat'], batch['img_useful'])
        elif self.config.decoder_init and self.config.decoder_init_methods == 'softlb':
            s_t_1 = self.get_decoder_init_softlabels(s_t_1, batch['soft_labels'], batch['img_useful'])
        s_t_0 = s_t_1.unsqueeze(0)
        c_t = Variable(torch.zeros((b, self.config.hidden_dim)))
        coverage = Variable(torch.zeros((b, t_k)))
        c_t = c_t.to(device = encoder_output.device)
        coverage = coverage.to(device = encoder_output.device)
        final_dists = []
        attn_dists = []
        Ias = []
        latest_tokens = [self.cls_ for _ in range(b)]
        for di in range(min(max_dec_len, self.config.max_dec_steps)):
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            y_t_1 = y_t_1.to(encoder_output.device)
            final_dist, s_t_1, c_t, attn_dist , next_coverage = self.decoder(y_t_1,
                                                          s_t_0,
                                                          s_t_1,
                                                          encoder_output,
                                                          c_t,
                                                          coverage,
                                                          di)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, 1)
            latest_tokens = [topk_ids[i][0] for i in range(b)]

            final_dists.append(final_dist)
            attn_dists.append(attn_dist)
        sim_loss = None
        ot_dist, cost = self.get_ot_dist(batch, ot_encoder_layer)

        if self.config.loss_modulate:
            txt_enc, img_enc = self.get_slice_modals(batch, encoder_output)
            loss_weight = self.loss_scorer(pooled_output, txt_enc, img_enc)
        else:
            loss_weight = torch.ones(b,2).to(device = encoder_output.device)
        return torch.stack(final_dists, dim=1), attn_dists, coverage, Ias,sim_loss, all_attention_scores, \
               mutual_info, txt_encoder_output, mutual_encoder_layer, \
               phrase_info, txt_phrase_score, cross_phrase_score, \
               ot_dist, loss_weight

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def filter_pos_words(self,batch,mutual_info):
        win = 0
        exchange = 0
        earthquake = 0
        held = 0
        subwords = self.tokenizer.convert_ids_to_tokens(batch['input_ids'][0].tolist())
        if 'win' in subwords:
            index = subwords.index('win')
            wg = mutual_info[0,index]
            if wg>0:
                win = 1
        if 'exchange' in subwords:
            index = subwords.index('exchange')
            wg = mutual_info[0,index]
            if wg>0:
                exchange = 1
        if 'earthquake' in subwords:
            index = subwords.index('earthquake')
            wg = mutual_info[0,index]
            if wg>0:
                earthquake = 1
        if 'held' in subwords:
            index = subwords.index('held')
            wg = mutual_info[0,index]
            if wg>0:
                held = 1
        if 'hold' in subwords:
            index = subwords.index('hold')
            wg = mutual_info[0,index]
            if wg>0:
                held = 1
        return win, exchange, earthquake,held

    def beam_search(self, batch):
        if self.config.guiding:
            batch['img_feat'] = batch['img_feat'] * batch['img_useful']
        attn_vis_RU, attn_txt_all, attn_txt_LU = self.gen_attn(batch)
        batch['attn_vis_RU'] = attn_vis_RU
        batch['attn_txt_all'] = attn_txt_all
        copy_position = batch['copy_position']
        if self.config.mutual_kl_loss or self.config.mutual_phrase_loss or self.config.pre_filter:
            txt_encoder_output, txt_phrase_score, txt_ot_layer = \
                self.text_encoder(batch)
            # txt_embedding_output, txt_attn_masks, txt_all_attention_scores = \
        else:
            txt_encoder_output, txt_phrase_score, txt_ot_layer = None, None, None

        encoder_output, pooled_output, embedding_output, attn_masks, all_attention_scores, \
        mutual_encoder_layer, ot_encoder_layer, cross_phrase_score = \
            self.cross_encoder(batch,
                               present_params=(
                               self.config.ot_layer, True, txt_ot_layer, attn_vis_RU, batch['attn_masks'],
                               batch['attn_txt_all'],self.config.ot_trunc))
        if self.config.mutual_kl_loss:
            mutual_info = self.get_mutual_info(txt_encoder_output, mutual_encoder_layer, copy_position)
        else:
            mutual_info = None
        if self.config.mutual_phrase_loss:
            phrase_info, txt_phrase_score, cross_phrase_score = self.get_phrase_info(batch, txt_phrase_score,
                                                                                 cross_phrase_score)
        else:
            phrase_info = None
        # encoder_output, pooled_output, embedding_output, attn_masks, all_attention_scores, \
        # mutual_encoder_layer, ot_encoder_layer, phrase_encoder_layer = \
        #     self.cross_encoder(batch)
        if self.config.pre_filter:
            cross_pool = (ot_encoder_layer * batch['attn_masks'].unsqueeze(2)).sum(dim=1)
            cross_pool = cross_pool / (batch['attn_masks'].sum(dim=1).unsqueeze(1))
            txt_pool = (txt_ot_layer * batch['attn_txt_all'].unsqueeze(2)).sum(dim=1)
            txt_pool = txt_pool / (batch['attn_txt_all'].sum(dim=1).unsqueeze(1))
            pool_diff = self.cos(cross_pool, txt_pool)
        else:
            pool_diff = torch.tensor(0)
        # pool_diff, cost = self.get_ot_dist(batch, ot_encoder_layer)
        # if self.config.mutual_kl_loss:
        #     if self.config.mutual_cosine:
        #         mutual_info = self.get_mutual_info_cos(txt_encoder_output, mutual_encoder_layer, copy_position)
        #     else:
        #         mutual_info = self.get_mutual_info(txt_encoder_output, mutual_encoder_layer, copy_position)
        # else:
        #     mutual_info = None
        # win, exchange, earthquake,held = self.filter_pos_words(batch, mutual_info)
        # if win+exchange+earthquake+held>=1:
        #     print(batch['qids'], win, exchange, earthquake,held)
        if self.config.loss_modulate:
            txt_enc, img_enc = self.get_slice_modals(batch, encoder_output)
            loss_weight = self.loss_scorer(pooled_output, txt_enc, img_enc)
        else:
            loss_weight = torch.ones(1,2).to(device = encoder_output.device)
        b, t_k, n = list(encoder_output.size())
        encoder_output = encoder_output.expand(self.config.beam_size, t_k, n).contiguous()
        s_t_1 = pooled_output.unsqueeze(0)
        if self.config.decoder_init and self.config.decoder_init_methods == 'global':
            s_t_1 = self.get_decoder_init(s_t_1, batch['img_feat'], batch['img_useful'])
        elif self.config.decoder_init and self.config.decoder_init_methods == 'softlb':
            s_t_1 = self.get_decoder_init_softlabels(s_t_1, batch['soft_labels'], batch['img_useful'])
        s_t_0 = pooled_output
        c_t = Variable(torch.zeros((self.config.beam_size, self.config.hidden_dim)))
        coverage = Variable(torch.zeros((self.config.beam_size, t_k)))
        c_t = c_t.to(device=encoder_output.device)
        coverage = coverage.to(device=encoder_output.device)

        beams = [Beam(tokens=[self.cls_],
                      log_probs=[0.0],
                      state=s_t_1[:, 0],
                      context=c_t[0],
                      coverage=coverage[0])
                 for _ in range(self.config.beam_size)]
        results = []
        steps = 0
        while steps < self.config.max_dec_steps and len(results) < self.config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            # latest_tokens = [t if t < len(self.vocab) else self.vocab.stoi[self.config.UNKNOWN_TOKEN] \
            #                  for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            y_t_1 = y_t_1.to(encoder_output.device)
            all_state_h = []

            all_context = []

            for h in beams:
                state_h = h.state
                all_state_h.append(state_h)

                all_context.append(h.context)
            s_t_1 = torch.stack(all_state_h, 0).transpose(1, 0)
            c_t = torch.stack(all_context, 0)

            coverage_t_1 = None
            if self.config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t_1, c_t, attn_dist , next_coverage = self.decoder(y_t_1,
                                                          s_t_0,
                                                          s_t_1,
                                                          encoder_output,
                                                          c_t,
                                                          coverage,
                                                          steps)

            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, self.config.beam_size * 2)

            dec_h = s_t_1
            dec_h = dec_h.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i].unsqueeze(0))
                context_i = c_t[i]
                # coverage_i = next_coverage[i] if next_coverage else None
                coverage_i = next_coverage[i]

                for j in range(self.config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.sep:
                    if steps >= self.config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == self.config.beam_size or len(results) == self.config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0], all_attention_scores, loss_weight, pool_diff, mutual_info, phrase_info

