"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""
from collections import defaultdict

from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU
from .model import UniterPreTrainedModel, UniterModel, UniterModelForText
import torch
from torch.nn.utils.rnn import pad_sequence

class GroudClassifier(nn.Module):
    def __init__(self, hidden_dim, output_class):
        super(GroudClassifier, self).__init__()
        self.fc =  nn.Linear(hidden_dim,hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.out =  nn.Linear(hidden_dim,output_class)

    def forward(self, hidden_state):
        output = self.fc(hidden_state)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.out(output)
        output = F.softmax(output, dim=-1)
        return output

class PhraseCopyScorer(nn.Module):
    def __init__(self, hidden_dim, output_class):
        super(PhraseCopyScorer, self).__init__()
        self.fc =  nn.Linear(hidden_dim,hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.out =  nn.Linear(hidden_dim,output_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_state):
        output = self.fc(hidden_state)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.out(output)
        output = self.sigmoid(output)
        return output

class OTMapping(nn.Module):
    def __init__(self, hidden_dim):
        super(OTMapping, self).__init__()
        self.fc =  nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.out =  nn.Linear(hidden_dim,hidden_dim)

    def forward(self, hidden_state):
        output = self.fc(hidden_state)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.out(output)
        return output

class UniterForVisualSummEnc(UniterPreTrainedModel):
    """ Finetune UNITER for Multi-modal Summarizaiton Encoder
    """
    def __init__(self, config, img_dim,
                 mutual_layer, mutual_score = False, mutual_detach = False,
                 ot_layer = 12, ot_detach = False,
                 phrase_layer=6, phrase_score=False, phrase_detach=False,
                 output_all_encoded_layers = False):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim, mutual_layer, mutual_detach)
        self.apply(self.init_weights)
        self.mutual_score = mutual_score
        self.mutual_detach = mutual_detach
        self.mutual_layer = mutual_layer

        self.phrase_layer = phrase_layer
        self.phrase_score = phrase_score
        self.phrase_detach = phrase_detach

        self.ot_detach = ot_detach
        self.ot_layer = ot_layer
        self.output_all_encoded_layers = output_all_encoded_layers
        if mutual_score:
            self.mutual_scorer = GroudClassifier(config.hidden_size, 2)
        if phrase_score:
            self.phrase_scorer = PhraseCopyScorer(config.hidden_size, 1)
        # self.ot_mapping = OTMapping(config.hidden_size)

    def forward(self, batch, present_params=(),training=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        phrase_tensor = batch['phrase_tensor']
        phrase_padding_mask = batch['phrase_padding_mask']
        sequence_output, embedding_output, all_attention_scores = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      present_params,
                                      output_all_encoded_layers=self.output_all_encoded_layers)
        if self.output_all_encoded_layers:
            pooled_output = self.uniter.pooler(sequence_output[-1])
        else:
            pooled_output = self.uniter.pooler(sequence_output)

        mutual_encoder_layer = sequence_output[self.mutual_layer-1]
        ot_encoder_layer = sequence_output[self.ot_layer-1]
        phrase_encoder_layer = sequence_output[self.phrase_layer-1]
        if self.mutual_detach:
            mutual_encoder_layer = mutual_encoder_layer.detach()
        if self.ot_detach:
            ot_encoder_layer = ot_encoder_layer.detach()
        if self.phrase_detach:
            phrase_encoder_layer = phrase_encoder_layer.detach()
        # if training==False:
        #     sequence_output = sequence_output.detach()
        #     pooled_output = pooled_output.detach()
        if self.mutual_score:
            mutual_encoder_layer = self.mutual_scorer(mutual_encoder_layer)
        if self.phrase_score:
            phrase_feat = []
            for pi, pte in enumerate(phrase_tensor):
                p_len = pte.shape[1]
                cur_feat = torch.matmul(pte, phrase_encoder_layer[pi][:p_len, :])
                pte_mask = phrase_padding_mask[pi]
                pte_sum = pte.sum(dim=1).view(-1,1) + (1-pte_mask)*10000
                cur_feat = cur_feat/pte_sum
                phrase_feat.append(cur_feat)
            phrase_feat = pad_sequence(phrase_feat, batch_first=True, padding_value=0)
            phrase_encoder_layer = self.phrase_scorer(phrase_feat)
        # ot_encoder_layer = self.ot_mapping(ot_encoder_layer)
        return sequence_output[-1], pooled_output, embedding_output, attn_masks, all_attention_scores, \
               mutual_encoder_layer, ot_encoder_layer, phrase_encoder_layer

class UniterForText(UniterPreTrainedModel):
    """ Finetune UNITER for Multi-modal Summarizaiton Encoder
    """
    def __init__(self, config, img_dim,
                 mutual_layer =3, mutual_score = False, mutual_detach = False,
                 ot_layer = 6, ot_detach = False,
                 phrase_layer=6, phrase_score=False, phrase_detach=False,
                 output_all_encoded_layers = False
                 ):
        super().__init__(config)
        encoder_layer = max(mutual_layer, phrase_layer, ot_layer)
        self.uniter = UniterModelForText(config, img_dim, encoder_layer)
        self.apply(self.init_weights)
        self.mutual_layer = mutual_layer
        self.mutual_score = mutual_score
        self.mutual_detach = mutual_detach

        self.phrase_layer = phrase_layer
        self.phrase_score = phrase_score
        self.phrase_detach = phrase_detach

        self.ot_detach = ot_detach
        self.ot_layer = ot_layer
        self.output_all_encoded_layers = output_all_encoded_layers
        if mutual_score:
            self.mutual_scorer = GroudClassifier(config.hidden_size, 2)
        if phrase_score:
            self.phrase_scorer = PhraseCopyScorer(config.hidden_size, 1)

    def forward(self, batch, training=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_txt_all']
        gather_index = batch['gather_index']
        phrase_tensor = batch['phrase_tensor']
        phrase_padding_mask = batch['phrase_padding_mask']
        sequence_output, embedding_output, all_attention_scores = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=self.output_all_encoded_layers)
        # pooled_output = self.uniter.pooler(sequence_output)
        mutual_encoder_layer = sequence_output[self.mutual_layer - 1]
        ot_encoder_layer = sequence_output[self.ot_layer - 1]
        phrase_encoder_layer = sequence_output[self.phrase_layer - 1]
        if self.mutual_detach:
            mutual_encoder_layer = mutual_encoder_layer.detach()
        if self.ot_detach:
            ot_encoder_layer = ot_encoder_layer.detach()
        if self.phrase_detach:
            phrase_encoder_layer = phrase_encoder_layer.detach()
        # import pdb
        # pdb.set_trace()
        if self.mutual_score:
            mutual_encoder_layer = self.mutual_scorer(mutual_encoder_layer)
        if self.phrase_score:
            phrase_feat = []
            for pi, pte in enumerate(phrase_tensor):
                p_len = pte.shape[1]
                cur_feat = torch.matmul(pte, phrase_encoder_layer[pi][:p_len, :])
                pte_mask = phrase_padding_mask[pi]
                pte_sum = pte.sum(dim=1).view(-1,1) + (1-pte_mask)*10000
                cur_feat = cur_feat/pte_sum
                phrase_feat.append(cur_feat)
            phrase_feat = pad_sequence(phrase_feat, batch_first=True, padding_value=0)
            phrase_encoder_layer = self.phrase_scorer(phrase_feat)

        return mutual_encoder_layer, phrase_encoder_layer, ot_encoder_layer
               #\ embedding_output, attn_masks, all_attention_scores
