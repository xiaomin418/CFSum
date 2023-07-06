import argparse
import json
from torch.utils.data import DataLoader
import torch
from utils.misc import Struct
#
import sys
from time import time
import os
import logging
from pytorch_pretrained_bert import BertTokenizer
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm
sys.path.append('../')
from src.datasets.dataset_momu import get_data_loader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.model.mms import MultiModal
from utils.logger import LOGGER
from utils.logger import print_config
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
                    datefmt='%Y-%m  -%d %H:%M:%S',
                    level=logging.INFO)
# torch.backends.cudnn.enabled = True
torch.manual_seed(123)
# import pdb
# pdb.set_trace()
def validate_batches(config, epoch, dev_loader,model):
        pure_mean_loss = []
        mean_kl_loss = []
        mean_copy_loss = []
        mean_ot_loss = []
        mean_norm_loss = []
        mean_phrase_attn_loss = []
        mean_phrase_copy_loss = []
        model.module.eval() if hasattr(
                model, 'module') else model.eval()
        mse_fn = torch.nn.MSELoss(reduce=False, size_average=False)
        for iter, batch in enumerate(dev_loader):
            final_dists, attn_dist, coverage, Ias, sim_loss, all_attention_scores, \
            mutual_info, txt_encoder_output, cross_encoder_output, \
            phrase_attn2w, txt_phrase_score, cross_phrase_score, \
            ot_dist, loss_weight = model.decode(batch)
            target_batch = batch['targets']
            dec_padding_mask = batch['dec_mask']
            dec_lens = batch['dec_len']
            gen_len = final_dists.shape[1]
            gold_probs = torch.gather(final_dists, 2, target_batch[:, :gen_len].unsqueeze(2)).squeeze()
            pure_step_losses = -torch.log(gold_probs + config.eps)

            if config.key_w_loss:
                step_loss = pure_step_losses * dec_padding_mask[:, :gen_len] * batch['dec_pos_f']
            else:
                step_loss = pure_step_losses * dec_padding_mask[:, :gen_len]

            sum_losses = torch.sum(step_loss, 1)
            batch_avg_loss = sum_losses / dec_lens
            if config.score_ref:
                batch_avg_loss = batch_avg_loss + sim_loss * config.sim_loss_weight

            loss = torch.mean(batch_avg_loss)
            pure_loss = loss
            if config.copy_position_loss and epoch >= config.other_loss_train_epoch:
                copy_position = batch['copy_position']
                if config.mutual_kl_loss:
                    copy_loss_txt = torch.gather(txt_encoder_output, 2, copy_position.unsqueeze(2)).squeeze()
                    copy_loss_txt = -torch.log(copy_loss_txt + config.eps)
                else:
                    copy_loss_txt = 0
                copy_loss_cross = torch.gather(cross_encoder_output, 2, copy_position.unsqueeze(2)).squeeze()
                copy_loss_cross = -torch.log(copy_loss_cross + config.eps)
                copy_loss_all = copy_loss_txt + copy_loss_cross
                src_len = copy_loss_all.shape[1]
                copy_loss_all = copy_loss_all * batch['attn_txt_all'][:, :src_len]
                copy_loss_all = torch.sum(copy_loss_all, 1)
                copy_loss_all = copy_loss_all / batch['txt_lens']
                copy_loss_all = torch.mean(copy_loss_all)
                copy_loss_all = copy_loss_all * config.copy_position_weight
                loss = loss + copy_loss_all
                # loss = copy_loss_all
            else:
                copy_loss_all = torch.tensor(0)

            if config.mutual_kl_loss and epoch >= config.other_loss_train_epoch:
                kl_loss = model.get_vis_attn(config, all_attention_scores, mutual_info, batch['attn_vis_RU'],
                                             batch['attn_txt_all'], batch['num_bbs'])
                kl_loss = kl_loss * config.mutual_loss_weight
                kl_loss = kl_loss * loss_weight[:, 0]
                kl_loss = kl_loss.mean()
                loss = loss + kl_loss
            else:
                kl_loss = torch.tensor(0)

            if config.mutual_phrase_loss and epoch >= config.other_loss_train_epoch:
                phrase_attn_loss = model.get_phrase_attn(config, all_attention_scores, phrase_attn2w,
                                                         batch['phrase_tensor'],
                                                         batch['attn_vis_RU'], batch['attn_txt_all'], batch['num_bbs'])
                phrase_attn_loss = phrase_attn_loss * config.phrase_loss_weight
                phrase_attn_loss = phrase_attn_loss * loss_weight[:, 1]
                phrase_attn_loss = phrase_attn_loss.mean()
                loss = loss + phrase_attn_loss
            else:
                phrase_attn_loss = torch.tensor(0)

            if config.phrase_copy_loss and epoch >= config.other_loss_train_epoch:
                phrase_padding_mask = batch['phrase_padding_mask']
                phrase_copy_score = batch['phrase_copy_score']
                phrase_copy_loss = mse_fn(txt_phrase_score, phrase_copy_score.unsqueeze(2)) + \
                                   mse_fn(cross_phrase_score, phrase_copy_score.unsqueeze(2))
                phrase_copy_loss = phrase_copy_loss * phrase_padding_mask
                phrase_copy_loss = phrase_copy_loss.sum(dim=1) / phrase_padding_mask.sum(dim=1)
                phrase_copy_loss = phrase_copy_loss.mean()
                loss = loss + phrase_copy_loss
            else:
                phrase_copy_loss = torch.tensor(0)

            if config.ot_loss:
                ot_loss = ot_dist * loss_weight[:, 1]
                ot_loss = ot_loss.mean()
                loss = loss + ot_loss
            else:
                ot_loss = torch.tensor(0)

            if config.norm_loss:
                norm_loss = loss_weight.sigmoid()
                norm_loss = config.norm_weight * (1 / norm_loss)
                norm_loss = norm_loss.sum(dim=-1)
                norm_loss = norm_loss.mean()
                loss = loss + norm_loss
            else:
                norm_loss = torch.tensor(0)

            pure_mean_loss.append(pure_loss.item())
            mean_kl_loss.append(kl_loss.item())
            mean_copy_loss.append(copy_loss_all.item())
            mean_ot_loss.append(ot_loss.item())
            mean_norm_loss.append(norm_loss.item())
            mean_phrase_attn_loss.append(phrase_attn_loss.item())
            mean_phrase_copy_loss.append(phrase_copy_loss.item())
            if iter>500:
                break
        avg_loss = round(sum(pure_mean_loss) / len(pure_mean_loss), 3)
        avg_kl_loss = round(sum(mean_kl_loss) / len(mean_kl_loss), 3)
        avg_copy_loss = round(sum(mean_copy_loss) / len(mean_copy_loss), 3)
        avg_ot_loss = round(sum(mean_ot_loss) / len(mean_ot_loss), 3)
        avg_norm_loss = round(sum(mean_norm_loss) / len(mean_norm_loss), 3)
        avg_phrase_attn_loss = round(sum(mean_phrase_attn_loss) / len(mean_phrase_attn_loss), 3)
        avg_phrase_copy_loss = round(sum(mean_phrase_copy_loss) / len(mean_phrase_copy_loss), 3)
        return avg_loss, avg_kl_loss, avg_copy_loss, avg_ot_loss, avg_norm_loss,\
               avg_phrase_attn_loss, avg_phrase_copy_loss

def save_model(model, model_dir, running_avg_loss, epoch=0, iter=0, encoder_training=False):

    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self
    if encoder_training:
        state = {
            'cross_encoder_state_dict': model_to_save.cross_encoder.state_dict(),
            'decoder_state_dict': model_to_save.decoder.state_dict(),
            'current_loss': running_avg_loss
        }
        if model.config.loss_modulate:
            state['loss_scorer_state_dict'] = model_to_save.loss_scorer.state_dict()
        if model.config.mutual_kl_loss:
            state['text_encoder_state_dict'] = model_to_save.text_encoder.state_dict()
    else:
        state = {
            'decoder_state_dict': model_to_save.decoder.state_dict(),
            'current_loss': running_avg_loss
        }
    model_save_path = os.path.join(model_dir, 'model_%d_%d_%d' % (epoch, iter, int(time())))
    torch.save(state, model_save_path)

def trainIters(config, epoch,train_loader, iter_bar,model, optimizer, encoder_training):
    print("Begin training epoch={}!".format(epoch))
    pure_mean_loss = []
    mean_kl_loss = []
    mean_copy_loss = []
    mean_ot_loss = []
    mean_norm_loss = []
    mean_phrase_attn_loss = []
    mean_phrase_copy_loss = []

    for iter, batch in enumerate(train_loader):
        # import pdb
        # pdb.set_trace()
        batch_size = len(batch['qids'])
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        mse_fn = torch.nn.MSELoss(reduce=False, size_average=False)
        optimizer.zero_grad()
        final_dists, attn_dists, coverage, Ias, sim_loss, all_attention_scores, \
        mutual_info, txt_encoder_output, cross_encoder_output, \
        phrase_attn2w, txt_phrase_score, cross_phrase_socre, \
        ot_dist, cost, loss_weight = model(batch,encoder_training)
        target_batch = batch['targets']
        dec_padding_mask = batch['dec_mask']
        dec_lens = batch['dec_len']
        gen_len = final_dists.shape[1]
        gold_probs = torch.gather(final_dists, 2, target_batch[:, :gen_len].unsqueeze(2)).squeeze()
        pure_step_losses = -torch.log(gold_probs + config.eps)
        if config.key_w_loss:
            step_loss = pure_step_losses * dec_padding_mask[:, :gen_len]*batch['dec_pos_f']
        else:
            step_loss = pure_step_losses * dec_padding_mask[:, :gen_len]

        sum_losses = torch.sum(step_loss, 1)
        batch_avg_loss = sum_losses / dec_lens
        if config.score_ref:
            batch_avg_loss=batch_avg_loss+sim_loss*config.sim_loss_weight

        loss = torch.mean(batch_avg_loss)

        pure_loss = loss
        if epoch>=config.other_loss_train_epoch:
            model.cross_encoder.mutual_detach = config.mutual_layer_detach
        else:
            model.cross_encoder.mutual_detach = False

        if config.copy_position_loss and epoch>=config.other_loss_train_epoch:
            copy_position = batch['copy_position']
            if config.mutual_kl_loss:
                copy_loss_txt = torch.gather(txt_encoder_output, 2, copy_position.unsqueeze(2)).squeeze()
                copy_loss_txt = -torch.log(copy_loss_txt + config.eps)
            else:
                copy_loss_txt = 0
            copy_loss_cross = torch.gather(cross_encoder_output, 2, copy_position.unsqueeze(2)).squeeze()
            copy_loss_cross = -torch.log(copy_loss_cross + config.eps)
            copy_loss_all = copy_loss_txt + copy_loss_cross
            src_len = copy_loss_all.shape[1]
            copy_loss_all = copy_loss_all*batch['attn_txt_all'][:,:src_len]
            copy_loss_all = torch.sum(copy_loss_all, 1)
            copy_loss_all = copy_loss_all/batch['txt_lens']
            copy_loss_all = torch.mean(copy_loss_all)
            copy_loss_all = copy_loss_all * config.copy_position_weight
            loss = loss + copy_loss_all
            # loss = copy_loss_all
        else:
            copy_loss_all = torch.tensor(0)
        if config.mutual_kl_loss and epoch>=config.other_loss_train_epoch:
            kl_loss = model.get_vis_attn(config, all_attention_scores, mutual_info, batch['attn_vis_RU'],
                                         batch['attn_txt_all'], batch['num_bbs'])
            kl_loss = kl_loss * config.mutual_loss_weight
            if config.hieracrchical:
                kl_loss = kl_loss.detach()
            kl_loss = kl_loss * loss_weight[:,0]
            kl_loss = kl_loss.mean()
            loss = loss + kl_loss
        else:
            kl_loss = torch.tensor(0)

        if config.mutual_phrase_loss and epoch>=config.other_loss_train_epoch:
            phrase_attn_loss = model.get_phrase_attn(config, all_attention_scores, phrase_attn2w,batch['phrase_tensor'],
                                                batch['attn_vis_RU'],batch['attn_txt_all'], batch['num_bbs'])
            phrase_attn_loss = phrase_attn_loss * config.phrase_loss_weight
            phrase_attn_loss = phrase_attn_loss * loss_weight[:,1]
            phrase_attn_loss = phrase_attn_loss.mean()
            loss = loss + phrase_attn_loss
        else:
            phrase_attn_loss = torch.tensor(0)

        if config.phrase_copy_loss and epoch>=config.other_loss_train_epoch:
            phrase_padding_mask = batch['phrase_padding_mask']
            phrase_copy_score = batch['phrase_copy_score']
            phrase_copy_loss = mse_fn(txt_phrase_score, phrase_copy_score.unsqueeze(2))+\
                               mse_fn(cross_phrase_socre, phrase_copy_score.unsqueeze(2))
            phrase_copy_loss = phrase_copy_loss*phrase_padding_mask
            phrase_copy_loss = phrase_copy_loss.sum(dim=1)/phrase_padding_mask.sum(dim=1)
            phrase_copy_loss = phrase_copy_loss.mean()
            loss = loss + phrase_copy_loss
        else:
            phrase_copy_loss = torch.tensor(0)
        if phrase_copy_loss!=phrase_copy_loss:
            import pdb
            pdb.set_trace()
        if config.ot_loss:
            ot_loss = model.get_supplement_attn(config, batch, all_attention_scores, cost, batch['attn_vis_RU'],
                                         batch['attn_txt_all'], batch['num_bbs'])
            # ot_loss = ot_dist * config.ot_weight
            ot_loss = ot_loss * loss_weight[:,1]
            ot_loss = ot_loss.mean()
            loss = loss + ot_loss
        else:
            ot_loss = torch.tensor(0)

        if config.norm_loss:
            # norm_loss = loss_weight.sigmoid()
            # import pdb
            # pdb.set_trace()
            norm_loss = config.norm_weight* (1/loss_weight)
            norm_loss = norm_loss.sum(dim=-1)
            norm_loss = norm_loss.mean()
            loss = loss + norm_loss
        else:
            norm_loss = torch.tensor(0)

        loss.backward()

        del batch
        model_to_clip = model.module if hasattr(
            model, 'module') else model
        if config.optimize_encoder:
            clip_grad_norm_(model_to_clip.parameters(), config.max_grad_norm)
        else:
            clip_grad_norm_(model_to_clip.decoder.parameters(), config.max_grad_norm)

        optimizer.step()
        iter_bar.update(batch_size)
        iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())
        # if iter!=0 and iter%100==0:
        #     print("supervised loss: ",round(sum(mean_kl_loss[-100:]) / len(mean_kl_loss[-100:]), 3))
        if config.mutual_kl_loss:
            mean_kl_loss.append(kl_loss.item())
        else:
            mean_kl_loss.append(0)
        pure_mean_loss.append(pure_loss.item())
        mean_copy_loss.append(copy_loss_all.item())
        mean_phrase_attn_loss.append(phrase_attn_loss.item())
        mean_phrase_copy_loss.append(phrase_copy_loss.item())
        mean_ot_loss.append(ot_loss.item())
        mean_norm_loss.append(norm_loss.item())

    if len(pure_mean_loss) == 0:
        return -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01
    else:
        return round(sum(pure_mean_loss) / len(pure_mean_loss), 3), \
               round(sum(mean_kl_loss) / len(mean_kl_loss), 3), \
               round(sum(mean_copy_loss) / len(mean_copy_loss), 3), \
               round(sum(mean_ot_loss) / len(mean_ot_loss), 3), \
               round(sum(mean_norm_loss) / len(mean_norm_loss), 3), \
               round(sum(mean_phrase_attn_loss) / len(mean_phrase_attn_loss), 3), \
               round(sum(mean_phrase_copy_loss) / len(mean_phrase_copy_loss), 3)

def build_optimizer(model, opts):
    if opts.optimize_encoder:
        if opts.mutual_kl_loss:
            optimizer_grouped_parameters = list(model.decoder.parameters()) + list(
                model.text_encoder.parameters()) + list(
                model.cross_encoder.parameters())
        else:
            optimizer_grouped_parameters = list(model.decoder.parameters()) + list(
                model.cross_encoder.parameters())
    else:
        if opts.mutual_kl_loss:
            optimizer_grouped_parameters = list(model.decoder.parameters()) + list(
                model.text_encoder.mutual_scorer.parameters()) + list(
                model.cross_encoder.mutual_scorer.parameters())
        else:
            optimizer_grouped_parameters = list(model.decoder.parameters()) + list(
                model.cross_encoder.mutual_scorer.parameters())
    if opts.loss_modulate:
        optimizer_grouped_parameters = optimizer_grouped_parameters + list(model.loss_scorer.parameters())
    optimizer = Adam(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    if opts.learning_adpative:
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True, min_lr=0,
                                      eps=1e-08)
    else:
        scheduler = None
    return optimizer, scheduler

def train(config_name = './configs/config.json'):
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
    print_config(config_name)

    device = args.device_id
    tokenizer = BertTokenizer.from_pretrained(args.toker, do_lower_case='uncased' in args.toker)

    train_dir = os.path.join(args.log_root, 'train_%d' % (int(time())))
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    model_dir = os.path.join(train_dir, 'model')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # model_dir = "/data/meihuan2/PreEMMS_checkpoints/0324-en-train/model"

    model = MultiModal(args, tokenizer)
    model_init_path = args.model_init_path
    if len(model_init_path)!=0 and os.path.exists(model_init_path):
        model.setup_train(model_init_path)
    model.to(device)


    # evaluate(model,eval_dataloder, ans2label)
    optimizer, scheduler = build_optimizer(model, args)

    random_seeds = [i + 1 for i in range(args.epoches)]
    encoder_training = model.config.optimize_encoder
    model.module.train() if hasattr(
        model, 'module') else model.train()
    for epoch in range(args.epoches):
        dev_loader = get_data_loader(args, args.dev_image_path, args.dev_text_path, args.dev_summri_path, tokenizer,
                                     device, None, None,
                                     random_seeds[epoch])
        train_loader = get_data_loader(args, args.train_image_path, args.train_text_path, args.train_summri_path,
                                       tokenizer, device, None, None,
                                       random_seeds[epoch])
        model.module.train() if hasattr(
            model, 'module') else model.train()

        iter_bar = tqdm(total = len(train_loader.loader.dataset), desc='Iter (loss=X.XXX)')
        # model.cross_encoder.train()
        # if model.config.mutual_kl_loss:
        #     model.text_encoder.train()

        avg_loss, avg_kl_loss, avg_copy_loss, avg_ot_loss, avg_norm_loss,\
            avg_phrase_attn_loss, avg_phrase_copy_loss= trainIters(args, epoch, train_loader,iter_bar, model, optimizer, encoder_training)
        save_model(model, model_dir, avg_loss, epoch, len(train_loader.dataset), encoder_training)

        logging.info("\nEpoch:{}-Training\nsummary loss is :{}\nkl_loss is: {}\ncopy_loss is: {}"
                     "\not_loss is: {}\nnorm_loss is: {}\nphrase_attn_loss is: {}\nphrase_copy_loss is: {}\n".
                     format(epoch, avg_loss, avg_kl_loss, avg_copy_loss,
                            avg_ot_loss,avg_norm_loss, avg_phrase_attn_loss, avg_phrase_copy_loss))
        logging.info("Enter dev")
        dev_loss, avg_kl_loss, avg_copy_loss, avg_ot_loss, avg_norm_loss,\
            avg_phrase_attn_loss, avg_phrase_copy_loss= validate_batches(args, epoch, dev_loader, model)
        if args.learning_adpative:
            scheduler.step(dev_loss)
        logging.info("\nEpoch:{}-Dev\nsummary loss is :{}\nkl_loss is: {}\ncopy_loss is: {}"
                     "\not_loss is: {}\nnorm_loss is: {}\nphrase_attn_loss is: {}\nphrase_copy_loss is: {}\n".
                     format(epoch, avg_loss, avg_kl_loss, avg_copy_loss,
                            avg_ot_loss, avg_norm_loss, avg_phrase_attn_loss, avg_phrase_copy_loss))
        # logging.info("\nEpoch:{}-Dev\nsummary loss is:{}\nkl_loss is: {}\ncopy_loss is: {}\not_loss is: {}\nnorm_loss is: {}\n".
        #              format(epoch, dev_loss, avg_kl_loss, avg_copy_loss, avg_ot_loss,avg_norm_loss))


if __name__ == "__main__":
    # config_name = './configs/base.json'
    import sys
    config_name = sys.argv[1]
    sys.argv = sys.argv[:-1]
    train(config_name)


