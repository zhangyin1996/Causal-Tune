from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor

import cv2
import numpy as np
from scipy.fftpack import dct

@MODELS.register_module()
class Reins(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        ### oursv2-3 ###
        self.learnable_tokens2 = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.mlp_token2feat2 = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f2 = nn.Linear(self.embed_dims, self.embed_dims)
        ### oursv2-3 ###
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        ### oursv2-3 ###
        nn.init.uniform_(self.learnable_tokens2.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f2.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat2.weight, a=math.sqrt(5))
        ### oursv2-3 ###
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)
            ### oursv2-3 ###
            nn.init.zeros_(self.mlp_delta_f2.weight)
            nn.init.zeros_(self.mlp_delta_f2.bias)
            ### oursv2-3 ###

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]
    ### oursv2-3 ###
    def get_tokens2(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens2
        else:
            return self.learnable_tokens2[layer]
    ### oursv2-3 ###

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)

        N, B, C = feats.shape
        h = int(np.sqrt(N))
        w = h
        feats = feats.reshape(B, C, h, w)
        _, cal_feat_dct = MAD_decompose(feats)

        ##non_feat_dct = non_feat_dct.reshape(N, B, C)
        cal_feat_dct = cal_feat_dct.reshape(N, B, C)
        feats = feats.reshape(N, B, C)

        tokens1 = self.get_tokens(layer)
        ##tokens2 = self.get_tokens2(layer)
        delta_feat1 = self.forward_delta_feat(
            cal_feat_dct,
            tokens1,
            layer,
        )
        # delta_feat2 = self.forward_delta_feat2(
        #     non_feat_dct,
        #     tokens2,
        #     layer,
        # )
        delta_feat = MAD_compose_fast(
            delta_feat1,
            ##delta_feat2
        )
        
        
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )
        
        
        delta_f = self.mlp_delta_f(delta_f + feats)
        return delta_f

    def forward_delta_feat2(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        attn = F.instance_norm(attn)
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat2(tokens[1:, :]),
        )
        
        delta_f = self.mlp_delta_f2(delta_f + feats)
        return delta_f
    ## oursv3 ###

def pcnorm(residual):
    # norm = F.batch_norm(
    #     residual, bn.running_mean, bn.running_var,
    #     None, None, bn.training, bn.momentum, bn.eps)
    norm = F.instance_norm(residual)

    phase, _ = decompose(residual, 'phase')
    _, norm_amp = decompose(norm, 'amp')

    residual = compose(
        phase,
        norm_amp
    )
    
    # residual = residual * bn.weight.view(1, -1, 1, 1) + bn.bias.view(1, -1, 1, 1)
    return residual

def replace_denormals(x, threshold=1e-5):
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y

def decompose(x, mode='all'):
    fft_im = torch.view_as_real(torch.fft.fft2(x, norm='backward'))
    if mode == 'all' or mode == 'amp':
        fft_amp = fft_im.pow(2).sum(dim=-1, keepdim=False)
        fft_amp = torch.sqrt(replace_denormals(fft_amp))
    else:
        fft_amp = None

    if mode == 'all' or mode == 'phase':
        fft_pha = torch.atan2(fft_im[..., 1], replace_denormals(fft_im[..., 0])) 
    else:
        fft_pha = None
    return fft_pha, fft_amp

def compose(phase, amp):
    x = torch.stack([torch.cos(phase) * amp, torch.sin(phase) * amp], dim=-1) 
    x = x / math.sqrt(x.shape[2] * x.shape[3])
    x = torch.view_as_complex(x)
    # return torch.fft.irfft2(x, s=x.shape[2:], norm='ortho')
    return torch.fft.irfft2(x, s=x.shape[1:], norm='ortho')


def dct_2d_batch(x_np):
    return np.array([dct(dct(img.T, norm='ortho').T, norm='ortho') for img in x_np])

def FreCom(x):

    x = x.detach().cpu().numpy().astype(np.float32)
    b, c, h, w = x.shape
    x_dct = np.zeros_like(x)
    
    for i in range(c):
        x_dct[:, i] = dct_2d_batch(x[:, i])

    return x_dct


def MAD_decompose(feat):

    ### V2 
    epsilon = 1e-10
    b, c, h, w = feat.shape
    feat_dct = FreCom(feat)

    x_coords = np.arange(h).reshape(h, 1)
    y_coords = np.arange(w).reshape(1, w)
    max_xy = np.maximum(x_coords, y_coords)  # shape: (h, w)

    v1 = int(min(h, w) * 0.1)
    v2 = int(min(h, w) * 0.7)
    v3 = min(h, w)

    mask_2d = np.zeros((h, w))

    region1 = max_xy <= v1
    mask_2d[region1] = 1 - max_xy[region1] / (v1 + epsilon) * 0.95

    region2 = (max_xy > v1) & (max_xy <= v2)
    mask_2d[region2] = 0.01

    region3 = (max_xy > v2) & (max_xy <= v3)
    mask_2d[region3] = (max_xy[region3] - v2) / (v3 - v2) * 0.3

    region4 = max_xy > v3
    mask_2d[region4] = 0.5

    mask = np.broadcast_to(mask_2d, (b, c, h, w))

    n_mask = 1 - mask

    non_feat_dct = torch.from_numpy(feat_dct * mask).float().to(feat.device)
    cal_feat_dct = torch.from_numpy(feat_dct * n_mask).float().to(feat.device)

    return non_feat_dct, cal_feat_dct


def MAD_compose(non_feat_dct, cal_feat_dct):

    feat_fc = non_feat_dct + cal_feat_dct
    N, b, c = feat_fc.shape
    h = int(np.sqrt(N))
    w = h
    feat_fc = feat_fc.reshape(b, c, h, w)

    feat_out = np.zeros((b, c, h, w))

    for i in range(c):
        feat_fc_ = feat_fc[:, i, :, :]
        feat_fc_ = feat_fc_.detach().cpu().numpy().astype(np.float32)
        for j in range(b): 
            feat_out[j, i, :, :] = cv2.idct(feat_fc_[j]).clip(0,255)
    
    feat_out = feat_out.reshape(N, b, c)
    feat_out = torch.from_numpy(feat_out).float().to(feat_fc.device)
    return feat_out


def idct_2d_approx(x):
    # x: (B, C, H, W)
    return torch.real(torch.fft.ifft2(x, norm='ortho')).clamp(0, 255)

# Fast version
def MAD_compose_fast(cal_feat_dct, non_feat_dct=None):
    if non_feat_dct is None:
        feat_fc = cal_feat_dct
    else:
        feat_fc = non_feat_dct + cal_feat_dct
    N, b, c = feat_fc.shape
    h = int(np.sqrt(N))
    w = h
    feat_fc = feat_fc.reshape(b, c, h, w).to(torch.complex64)

    feat_out = idct_2d_approx(feat_fc)
    feat_out = feat_out.reshape(N, b, c).float()
    return feat_out




@MODELS.register_module()
class LoRAReins(Reins):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.learnable_tokens
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)

    def get_tokens(self, layer):
        if layer == -1:
            return self.learnable_tokens_a @ self.learnable_tokens_b
        else:
            return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]
