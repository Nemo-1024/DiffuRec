import torch.nn as nn
import torch
import math
from diffurec import DiffuRec,Mlp_dif
import torch.nn.functional as F
import copy
import numpy as np
from step_sample import LossAwareSampler
import torch as th
import einops
from common import *

class Att_Diffuse_model(nn.Module):
    def __init__(self, diffu, args):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num+1
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim,padding_idx=0)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = diffu
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=0)
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
        self.per_token_ag = args.per_token_ag
        self.ag_encoder = Transformer_rep(args)
    def diffu_pre(self, item_rep, tag_emb, mask_seq,mask_tag):
        tgt_rep_diffu, seq_rep_out, weights, t  = self.diffu(item_rep, tag_emb, mask_seq,mask_tag)
        return tgt_rep_diffu, seq_rep_out, weights, t

    def reverse(self, item_rep, noise_x_t, mask_seq):
        reverse_pre = self.diffu.reverse_p_sample(item_rep, noise_x_t, mask_seq)
        return reverse_pre

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_diffu(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        scores_pos = scores.gather(1 , labels)  ## labels: b x 1
        scores_neg_mean = (torch.sum(scores, dim=-1).unsqueeze(-1)-scores_pos)/(scores.shape[1]-1)

        loss = torch.min(-torch.log(torch.mean(torch.sigmoid((scores_pos - scores_neg_mean).squeeze(-1)))), torch.tensor(1e8))
       
        # if isinstance(self.diffu.schedule_sampler, LossAwareSampler):
        #     self.diffu.schedule_sampler.update_with_all_losses(t, loss.detach())
        # loss = (loss * weights).mean()
        return loss   

    def loss_diffu_ce(self, out_seq, last_item, labels):
        # if self.per_token_ag:
        scores = torch.matmul(out_seq, self.item_embeddings.weight.t()) #B,L,K
        #
        # else:
        #     scores = torch.matmul(last_item, self.item_embeddings.weight.t()) #B,K labels: B
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        """
        return self.loss_ce(scores.reshape(-1,scores.shape[-1]), labels.reshape(-1))
        # return self.loss_ce(scores, labels.squeeze(-1))

    def diffu_rep_pre(self, rep_diffu):
        scores = torch.matmul(rep_diffu.reshape(-1,rep_diffu.shape[-1]), self.item_embeddings.weight.t())
        return scores
    
    def loss_rmse(self, rep_diffu, labels):
        rep_gt = self.item_embeddings(labels).squeeze(1)
        return torch.sqrt(self.loss_mse(rep_gt, rep_diffu))
    
    def routing_rep_pre(self, rep_diffu):
        item_norm = (self.item_embeddings.weight**2).sum(-1).view(-1, 1)  ## N x 1
        rep_norm = (rep_diffu**2).sum(-1).view(-1, 1)  ## B x 1
        sim = torch.matmul(rep_diffu, self.item_embeddings.weight.t())  ## B x N
        dist = rep_norm + item_norm.transpose(0, 1) - 2.0 * sim
        dist = torch.clamp(dist, 0.0, np.inf)
        
        return -dist

    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep/seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep/item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep/seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1)/torch.sum(mask_seq, dim=-1))

    def forward(self, sequence, tag, train_flag=True):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)

        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)  ## dropout first than layernorm

        item_embeddings = item_embeddings + position_embeddings

        item_embeddings = self.LayerNorm(item_embeddings)

        mask_seq = (sequence>0).float()
        mask_tag = (tag>0).float()

        # item_embeddings = self.ag_encoder(item_embeddings, mask_seq)

        if train_flag:
            tag_emb = self.item_embeddings(tag)  ## B x H
            # print("tag_emb",tag_emb.shape)
            last_item, out_seq, weights, t = self.diffu_pre(item_embeddings, tag_emb, mask_seq, mask_tag)
            # tgt, seq, weights, t

            # item_rep_dis = self.regularization_rep(rep_item, mask_seq)
            # seq_rep_dis = self.regularization_seq_item_rep(last_item, rep_item, mask_seq)

            item_rep_dis = None
            seq_rep_dis = None
        else:
            # noise_x_t = th.randn_like(tag_emb)
            noise_x_t = th.randn_like(item_embeddings[:,-1,:]).unsqueeze(1)
            # print("noise_x_t",noise_x_t.shape)
            last_item = self.reverse(item_embeddings, noise_x_t, mask_seq)
            weights, t, item_rep_dis, seq_rep_dis = None, None, None, None
            out_seq = None
        # item_rep = self.model_main(item_embeddings, last_item, mask_seq)
        # seq_rep = item_rep[:, -1, :]
        # scores = torch.matmul(seq_rep, self.item_embeddings.weight.t())
        # scores = None
        return out_seq, last_item, weights, t, item_rep_dis, seq_rep_dis

def create_model_diffu(args):
    # if args.model =='diffurec':
    #     diffu_pre = DiffuRec(args)
    # elif args.model == 'mlpdif':
    #     diffu_pre = Mlp_dif(args)
    # else:
    #     diffu_pre=None
    #     print('args.model is wrong')
    # return diffu_pre
    return DiffuRec(args)
