import torch.nn as nn
import torch as th
import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
def exists(v):
    return v is not None
def default(v, d):
    return v if exists(v) else d
def divisible_by(num, den):
    return (num % den) == 0
def generate_square_subsequent_mask(sz: int, device):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=torch.float32, device=device),
        diagonal=1,
    )

class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (
                    1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, padding_mask=None,is_causal=False):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in
                   zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if padding_mask is not None:
            # assert padding_mask.shape == [corr.shape[0],corr.shape[-1]]
            padding_mask = padding_mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1, 1, 1, corr.shape[-1]])
            corr = corr.masked_fill(padding_mask == 0, -1e9)
        if is_causal:
            causal_mask = generate_square_subsequent_mask(corr.shape[-1],device=corr.device)
            corr += causal_mask.unsqueeze(0).unsqueeze(0).repeat([corr.shape[0],corr.shape[1],1,1])
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout,is_causal):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.is_causal = is_causal
    def forward(self, hidden, padding_mask):
        hidden = self.input_sublayer(hidden,
                                     lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, padding_mask=padding_mask, is_causal=self.is_causal))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Transformer_rep(nn.Module):
    def __init__(self, args):
        super(Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.is_causal = args.is_causal
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout,self.is_causal) for _ in range(self.n_blocks)])

    def forward(self, hidden, padding_mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, padding_mask)
        return hidden
    def make_causal(self,is_causal):
        self.is_causal = is_causal

class AdaptiveLayerNorm(nn.Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        dim_condition = default(dim_condition, dim)

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.to_gamma = nn.Linear(dim_condition, dim, bias = False)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        normed = self.ln(x)
        gamma = self.to_gamma(condition)
        return normed * (gamma + 1.)


class MLP(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_cond=None,
        depth = 3,
        width = 1024,
        dropout = 0.
    ):
        super().__init__()
        layers = nn.ModuleList([])
        dim_cond = default(dim_cond, dim_input)
        self.to_time_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim_cond), # B,D+1
            nn.Linear(dim_cond + 1, dim_cond),
        )

        for _ in range(depth):

            adaptive_layernorm = AdaptiveLayerNorm(
                dim_input,
                dim_condition = dim_cond
            )

            block = nn.Sequential(
                nn.Linear(dim_input, width),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width, dim_input)
            )

            block_out_gamma = nn.Linear(dim_cond, dim_input, bias = False)
            nn.init.zeros_(block_out_gamma.weight)

            layers.append(nn.ModuleList([
                adaptive_layernorm,
                block,
                block_out_gamma
            ]))

        self.layers = layers

    def forward(
        self,
        noised,
        *,
        times,
        cond
    ):
        assert noised.ndim == 2

        time_emb = self.to_time_emb(times)
        cond = F.silu(time_emb + cond)

        denoised = noised

        for adaln, block, block_out_gamma in self.layers:
            residual = denoised
            denoised = adaln(denoised, condition = cond)

            block_out = block(denoised) * (block_out_gamma(cond) + 1.)
            denoised = block_out + residual

        return denoised
class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * torch.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered    # B,D+1