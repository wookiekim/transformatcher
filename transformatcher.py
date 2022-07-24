import functools
from operator import add

import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

from geometry import Geometry
import backbone as resnet

from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        nn.Linear(dim, int(dim * mult)),
        nn.GELU(),
        nn.Linear(int(dim * mult), dim)
    )

class FastAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 64,
        max_seq_len = 15**4,
        pos_emb = None
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.pos_emb = pos_emb
        self.max_seq_len = max_seq_len

        self.to_q_attn_logits = nn.Linear(dim_head, 1, bias = False) 
        self.to_k_attn_logits = nn.Linear(dim_head // 2, 1, bias = False) 
        self.to_r = nn.Linear(dim_head // 2, dim_head)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        n, device, h = x.shape[1], x.device, self.heads
        use_rotary_emb = True

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        freqs = self.pos_emb(torch.arange(self.max_seq_len, device = device), cache_key = self.max_seq_len)
        freqs = rearrange(freqs[:n], 'n d -> () () n d')
        q_aggr, k_aggr = map(lambda t: apply_rotary_emb(freqs, t), (q, k))
        
        v_aggr = v

        q_attn_logits = rearrange(self.to_q_attn_logits(q), 'b h n () -> b h n') * self.scale
        q_attn = q_attn_logits.softmax(dim = -1)

        global_q = torch.einsum('b h n, b h n d -> b h d', q_attn, q_aggr)
        global_q = rearrange(global_q, 'b h d -> b h () d')

        k = k * global_q
        k = reduce(k, 'b h n (d r) -> b h n d', 'sum', r = 2)

        k_attn_logits = rearrange(self.to_k_attn_logits(k), 'b h n () -> b h n') * self.scale
        k_attn = k_attn_logits.softmax(dim = -1)

        global_k = torch.einsum('b h n, b h n d -> b h d', k_attn, k_aggr)
        global_k = rearrange(global_k, 'b h d -> b h () d')

        u = v_aggr * global_k
        u = reduce(u, 'b h n (d r) -> b h n d', 'sum', r = 2)

        r = self.to_r(u)

        r = r + q

        r = rearrange(r, 'b h n d -> b n (h d)')
        return self.to_out(r)


class Match2Match(nn.Module):

    def __init__(self, feat_dims,luse):
        super(Match2Match, self).__init__()

        input_dim = 16
        layer_num = 6 
        expand_ratio = 4
        bottlen = 26

        self.to_embedding = nn.Sequential(
            Rearrange('b c h1 w1 h2 w2 -> b (h1 w1 h2 w2) c'),
            nn.Linear(bottlen, input_dim)
        )

        self.posenc = nn.Parameter(torch.randn(15,15,15,15,input_dim), requires_grad=True)

        layer_pos_emb = RotaryEmbedding(dim=4, freqs_for = 'pixel')

        self.to_original = nn.Sequential(
            nn.Linear(input_dim, 1),
            Rearrange('b (h1 w1 h2 w2) c -> b c h1 w1 h2 w2', h1=15, w1=15, h2=15, w2=15),
        )
        
        self.trans_nc = nn.ModuleList([])
        for _ in range(layer_num):
            self.trans_nc.append(nn.ModuleList([
                PreNorm(input_dim, FastAttention(input_dim, heads = 8, dim_head = 4, pos_emb=layer_pos_emb)),
                PreNorm(input_dim, FeedForward(input_dim)),
            ]))
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, src_feats, trg_feats):
        
        correlations = Geometry.cosine_similarity(src_feats, trg_feats)
        correlations = torch.stack(correlations, dim=1) 
        correlations = correlations.squeeze(2)

        correlations = self.relu(correlations)

        bsz, ch, side, _, _, _ = correlations.size()
        
        embedded_features = self.to_embedding(correlations)
        
        # Match-to-match attention blocks
        for attn, ff in self.trans_nc:
            embedded_features = attn(embedded_features) + embedded_features
            embedded_features = ff(embedded_features) + embedded_features
        
        refined_corr = self.to_original(embedded_features)

        # Geometry upsample
        correlations = Geometry.interpolate4d(refined_corr.squeeze(1), Geometry.upsample_size).unsqueeze(1)

        side = correlations.size(-1) ** 2
        correlations = correlations.view(bsz, side, side).contiguous()

        return correlations



class TransforMatcher(nn.Module):
    def __init__(self, backbone, luse, device, imside=240):
        super(TransforMatcher, self).__init__()

        # 1. Backbone network initialization
        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True).to(device)
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True).to(device)
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.luse = luse
        self.bottleneck_ids = functools.reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = functools.reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.nbottlenecks = nbottlenecks

        if 'resnet' in backbone:
            self.feature_extractor = self.extract_bottleneck_features
        else:
            self.feature_extractor = self.extract_features_pvt

        # 2. Pass dummy input to get channel & correlation tensor dimensions
        with torch.no_grad():
            dummy = torch.randn((2, 3, imside, imside)).to(device)
            dummy_feat = self.feature_extractor(dummy, dummy)[0]
            feat_dim = [f.size(1) for f in dummy_feat]
        Geometry.initialize(imside, device)

        # 3. Learnable match-to-match attention module
        self.match2match = Match2Match(feat_dim, luse).to(device)

    def forward(self, src_img, trg_img):
        src_feats, trg_feats = self.feature_extractor(src_img, trg_img)

        correlation_ts = self.match2match(src_feats, trg_feats)
        return correlation_ts, None

    def extract_bottleneck_features(self, src_img, trg_img):
        src_feats, trg_feats = [], []

        feat = self.backbone.conv1.forward(torch.cat([src_img, trg_img], dim=1))
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)

        for idx in range(1, 5):
            if idx not in self.luse:
                feat = self.backbone.__getattr__('layer%d' % idx)(feat)
            else:
                layer = self.backbone.__getattr__('layer%d' % idx)
                for bid in range(len(layer)):
                    feat = layer[bid](feat)
                    src_feats.append(feat.narrow(1, 0, feat.size(1) // 2).clone())
                    trg_feats.append(feat.narrow(1, feat.size(1) // 2, feat.size(1) // 2).clone())

                if idx == max(self.luse): break
        return src_feats, trg_feats