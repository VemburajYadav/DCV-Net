import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .linear_attention import LinearAttention, FullAttention
from .swin_attention import ShiftedWindowAttention
# from einops.einops import rearrange


class FeatureAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 share_qk_proj=False,
                 mlp_layer=True,
                 gelu_activation=False,
                 num_splits=1):
        super(FeatureAttentionLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.share_qk_proj = share_qk_proj
        self.use_mlp = mlp_layer

        # multi-head attention
        if share_qk_proj:
            self.qk_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)

        if attention == 'linear':
            self.attention = LinearAttention()
        elif attention == 'swin':
            self.attention = ShiftedWindowAttention(num_splits=num_splits)
        else:
            self.attention = FullAttention()

        self.attention_type = attention

        self.merge = nn.Linear(d_model, d_model, bias=False)

        if self.use_mlp:
            # feed-forward network
            self.mlp = nn.Sequential(
                nn.Linear(d_model*2, d_model*2, bias=False),
                nn.GELU() if gelu_activation else nn.ReLU(True),
                nn.Linear(d_model*2, d_model, bias=False),
            )
            self.norm2 = nn.LayerNorm(d_model)

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None, attn_mask=None, with_shift=False, h=None, w=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)

        query, key, value = x, source, source

        # multi-head attention
        if self.share_qk_proj:
            query = self.qk_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            key = self.qk_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        else:
            query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)

        if self.attention_type == 'swin':
            query = query.view(bs, -1, self.dim)
            key = key.view(bs, -1, self.dim)
            value = value.view(bs, -1, self.dim)
            message = self.attention(query, key, value, h=h, w=w, with_shift=with_shift, attn_mask=attn_mask)
        else:
            message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]

        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        if self.use_mlp:
            # feed-forward network
            message = self.mlp(torch.cat([x, message], dim=2))
            message = self.norm2(message)

        return x + message


class FeatureAttentionLayerPreLN(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 share_qk_proj=False):
        super(FeatureAttentionLayerPreLN, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.share_qk_proj = share_qk_proj

        # multi-head attention
        if share_qk_proj:
            self.qk_proj = nn.Linear(d_model, d_model, bias=False)
            # norm and dropout
            self.norm1_qk = nn.LayerNorm(d_model)
            self.norm1_v = nn.LayerNorm(d_model)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            # norm and dropout
            self.norm1_q = nn.LayerNorm(d_model)
            self.norm1_kv = nn.LayerNorm(d_model)

        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        q, kv = x, source

        # multi-head attention
        if self.share_qk_proj:
            query = self.norm1_qk(q)
            key = self.norm1_qk(kv)
            value = self.norm1_v(kv)

            query = self.qk_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            key = self.qk_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
            value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        else:
            query = self.norm1_q(q)
            key_value = self.norm1_kv(kv)

            query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            key = self.k_proj(key_value).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
            value = self.v_proj(key_value).view(bs, -1, self.nhead, self.dim)

        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        out1 = q + message

        out2 = out1 + self.mlp(self.norm2(out1))
        return out2


class FeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(FeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.is_pre_ln = config["layer_norm"] == "pre"
        self.use_mlp = config["use_mlp"]
        self.is_gelu = config["activation"] == "GELU"
        self.attention_type = config['attention']

        if self.is_pre_ln:
            encoder_layer = FeatureAttentionLayerPreLN(config['d_model'],
                                                       config['nhead'], config['attention'],
                                                       share_qk_proj=config['share_qk_proj'])
        else:
            encoder_layer = FeatureAttentionLayer(config['d_model'], config['nhead'], config['attention'],
                                                  share_qk_proj=config['share_qk_proj'],
                                                  mlp_layer=self.use_mlp,
                                                  gelu_activation=self.is_gelu,
                                                  num_splits=config['swin_att_num_splits'])

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, attn_mask=None, with_shift=False, h=None, w=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        b = feat0.shape[0]

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat_cat = torch.cat([feat0, feat1], dim=0)
                if self.attention_type == 'swin':
                    feat_cat = layer(feat_cat, feat_cat, attn_mask=attn_mask, with_shift=with_shift, h=h, w=w)
                else:
                    feat_cat = layer(feat_cat, feat_cat, mask0, mask0)
                feat0, feat1 = torch.split(feat_cat, [b, b], dim=0)
            elif name == 'cross':
                feat_cat = torch.cat([feat0, feat1], dim=0)
                feat_rev = torch.cat([feat1, feat0], dim=0)
                if self.attention_type == 'swin':
                    feat_cat = layer(feat_cat, feat_rev, attn_mask=attn_mask, with_shift=with_shift, h=h, w=w)
                else:
                    feat_cat = layer(feat_cat, feat_rev, mask0, mask1)
                feat0, feat1 = torch.split(feat_cat, [b, b], dim=0)
            else:
                raise KeyError

        return feat0, feat1


class FeatureFlowTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(FeatureFlowTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.is_pre_ln = config["layer_norm"] == "pre"

        if self.is_pre_ln:
            print("Pre LN")
            encoder_layer = FeatureAttentionLayerPreLN(config['d_model'], config['nhead'], config['attention'])
        else:
            encoder_layer = FeatureAttentionLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, sparse_flow0=None, sparse_flow1=None, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            sparse_flow0 (torch.Tensor): [N, L, C] (optional)
            sparse_flow1 (torch.Tensor): [N, L, C] (optional)
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            elif name == 'flow-feature':
                feat0 = layer(feat0, sparse_flow0, mask0, mask0)
                feat1 = layer(feat1, sparse_flow1, mask1, mask1)
            elif name == 'flow-self':
                sparse_flow0 = layer(sparse_flow0, sparse_flow0, mask0, mask0)
                sparse_flow1 = layer(sparse_flow1, sparse_flow1, mask1, mask1)
            elif name == 'flow-cross':
                sparse_flow0 = layer(sparse_flow0, sparse_flow1, mask0, mask1)
                sparse_flow1 = layer(sparse_flow1, sparse_flow0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1
