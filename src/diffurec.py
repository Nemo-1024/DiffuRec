import torch.nn as nn
import torch as th
from step_sample import create_named_schedule_sampler
import numpy as np
import math
import torch
import torch.nn.functional as F
from common import *
from utils import _extract_into_tensor
from step_sample import *

class Diffu_xstart(nn.Module):
    def __init__(self, hidden_size, args):
        super(Diffu_xstart, self).__init__()
        self.hidden_size = hidden_size
        # self.linear_item = nn.Linear(self.hidden_size, self.hidden_size)
        # self.linear_xt = nn.Linear(self.hidden_size, self.hidden_size)
        # self.linear_t = nn.Linear(self.hidden_size, self.hidden_size)
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim),
                                        SiLU(),
                                        nn.Linear(time_embed_dim, self.hidden_size)
                                        )
        # self.fuse_linear = nn.Linear(self.hidden_size*3, self.hidden_size)
        self.att = Transformer_rep(args)
        # self.mlp_model = nn.Linear(self.hidden_size, self.hidden_size)
        # self.gru_model = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # self.gru_model = nn.GRU(self.hidden_size, self.hidden_size, num_layers=args.num_blocks, batch_first=True)
        self.lambda_uncertainty = args.lambda_uncertainty
        self.dropout = nn.Dropout(args.dropout)
        self.norm_diffu_rep = LayerNorm(self.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 4 * self.hidden_size),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(4 * self.hidden_size, self.hidden_size)
        )
        self.model = args.dif_decoder

    def dif_decoder(self, seq, mask):
        if self.model == 'mlp':
            return self.mlp(seq)
        elif self.model == 'att':
            return self.att(seq,mask)
        else:
            print('wrong dif_decoder')
            return None

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        assert dim % 2 == 0
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, x_t, t, mask_seq):
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        # print(x_t.shape,emb_t.shape) #[512, 50, 128] or [512, 1, 128],   [512,128]
        x_t = x_t + emb_t.unsqueeze(1)

        # lambda_uncertainty = th.normal(mean=th.full(rep_item.shape, 1.0), std=th.full(rep_item.shape, 1.0)).to(x_t.device)

        lambda_uncertainty = th.normal(mean=th.full(rep_item.shape, self.lambda_uncertainty), std=th.full(rep_item.shape, self.lambda_uncertainty)).to(x_t.device)  ## distribution
        # lambda_uncertainty = self.lambda_uncertainty  ### fixed

        # print(lambda_uncertainty.shape, x_t.shape, rep_item.shape)
        rep_diffu = self.dif_decoder(rep_item + lambda_uncertainty * x_t, mask_seq)
        out = rep_diffu[:, -1, :]

        ####  Attention
        # rep_diffu = self.att(rep_item + lambda_uncertainty * x_t, mask_seq)
        # # rep_diffu = self.att(rep_item,mask_seq)
        # rep_diffu = self.norm_diffu_rep(self.dropout(rep_diffu))
        # out = rep_diffu[:, -1, :]

        ### MLP
        # rep_diffu = self.mlp(rep_item + lambda_uncertainty * x_t)
        # # rep_diffu = self.norm_diffu_rep(self.dropout(rep_diffu))
        # out = rep_diffu[:, -1, :]


        ## rep_diffu = self.att(rep_item, mask_seq)  ## do not use
        ## rep_diffu = self.dropout(self.norm_diffu_rep(rep_diffu))  ## do not use

        ####

        #### GRU
        # output, hn = self.gru_model(rep_item + lambda_uncertainty * x_t.unsqueeze(1))
        # output = self.norm_diffu_rep(self.dropout(output))
        # out = output[:,-1,:]
        ## # out = hn.squeeze(0)
        # rep_diffu = None
        ####

        ### MLP
        # output = self.mlp_model(rep_item + lambda_uncertainty * x_t.unsqueeze(1))
        # output = self.norm_diffu_rep(self.dropout(output))
        # out = output[:,-1,:]
        # rep_diffu = None
        ###

        # out = out + self.lambda_uncertainty * x_t

        return out, rep_diffu

class DiffuRec(nn.Module):
    def __init__(self, args,):
        super(DiffuRec, self).__init__()
        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])

        self.noise_schedule = args.noise_schedule
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
         # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])
       
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)  ## lossaware (schedule_sample)
        self.timestep_map = self.time_map()
        self.rescale_timesteps = args.rescale_timesteps
        self.original_num_steps = len(betas)

        # self.xstart_model = self.dif_model(args)
        self.net = Diffu_xstart(self.hidden_size, args)

    def get_betas(self, noise_schedule, diffusion_steps):
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)  ## array, generate beta
        return betas

    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise  ## reparameter trick
        )  ## genetrate x_t based on x_0 (x_start) with reparameter trick

        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return th.where(mask==0, x_start, x_t)  ## replace the output_target_seq embedding (x_0) as x_t

    def time_map(self):
        timestep_map = []
        for i in range(len(self.alphas_cumprod)):
            if i in self.use_timesteps:
                timestep_map.append(i)
        return timestep_map

    # def scale_t(self, ts):
    #     map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
    #     new_ts = map_tensor[ts]
    #     # print(new_ts)
    #     if self.rescale_timesteps:
    #         new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
    #     return new_ts

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  ## \mu_t
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq):
        # print("func p_mean_variance", rep_item.shape,x_t.shape)
        model_output, _ = self.net(rep_item, x_t, self._scale_timesteps(t), mask_seq)

        x_0 = model_output.unsqueeze(1)  ##output predict
        # x_0 = self._predict_xstart_from_eps(x_t, t, model_output)  ## eps predict

        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)  ## x_start: candidante item embedding, x_t: inputseq_embedding + outseq_noise, output x_(t-1) distribution
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq)
        noise = th.randn_like(noise_x_t)
        # print("noise shape in func p_sample",noise.shape)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))  # no noise when t == 0
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick
        return sample_xt

    def reverse_p_sample(self, item_rep, noise_x_t, mask_seq):
        # return self.xstart_model(item_rep, noise_x_t, th.tensor([1] * item_rep.shape[0], device=item_rep.device), mask_seq)[0]
        device = item_rep.device
        indices = list(range(self.num_timesteps))[::-1]
        
        for i in indices: # from T to 0, reversion iteration  
            t = th.tensor([i] * item_rep.shape[0], device=device)
            with th.no_grad():
                noise_x_t = self.p_sample(item_rep, noise_x_t, t, mask_seq)
        return noise_x_t

    def forward(self, item_rep, item_tag, mask_seq,mask_tag):
        noise = th.randn_like(item_tag)
        t, weights = self.schedule_sampler.sample(item_rep.shape[0], item_tag.device) ## t is sampled from schedule_sampler
        x_t = self.q_sample(item_tag, t, noise=noise,mask=mask_tag)

        # eps, item_rep_out = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)  ## eps predict
        # x_0 = self._predict_xstart_from_eps(x_t, t, eps)
        # print("func forward", item_rep.shape, x_t.shape)
        x_0, item_rep_out = self.net(item_rep, x_t, self._scale_timesteps(t), mask_seq)  ##output predict
        return x_0, item_rep_out, weights, t
        # tgt, seq, weights, t

class Mlp_dif(DiffuRec):
    def __init__(self, args):
        super().__init__(args)
    def xstart_model(self,item_rep, x_t, t, mask_seq):
        return self.mlp(item_rep)