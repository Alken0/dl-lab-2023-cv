import torch
import torch.nn as nn
from ..vit.vit import Mlp, Attention


# Attention mechanism that uses the same projection for queries and keys.

class AttentionSameQK(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = args[0]
        qkv_bias = False
        self.qkv = None
        # START TODO #################
        # re-build the self.qkv layer for shared Q and K projections
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # END TODO ###################

    def forward(self, x):
        # START TODO #################
        # re-implement the forward pass with shared Q and K. See models/vit/vit.py
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[0],
            qkv[1],
        )  # make torchscript happy (cannot use tensor as tuple)
        # END TODO ###################

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Implementation of a transformer head for segmentation, using extra learnable class embeddings.

class TransformerSegmentationHead(nn.Module):
    def __init__(self, num_classes, dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, shared_qk=False):
        super().__init__()
        self.num_classes = num_classes

        # START TODO #################
        # add learnable class parameters/embeddings
        self.embed = nn.Parameter(torch.randn(num_classes, dim))
        # END TODO ###################

        # normalization layer before attention
        self.norm1 = norm_layer(dim)

        # build attention module
        attention_mechanism = Attention if not shared_qk else AttentionSameQK
        self.attn = attention_mechanism(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

        # normalization layer after attention
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        # build TWO MLPs: one for the patch tokens, one for the class tokens
        self.patches_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.classes_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        # final normalization layer
        self.mask_norm = norm_layer(num_classes)

    def forward(self, x):
        # reshape (patch) embeddings from encoder
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # START TODO #################
        # concatenate patch embeddings with class embeddings
        embed = self.embed.expand(b, -1, -1)
        x = torch.cat([embed, x], dim=1)
        residual = x
        # END TODO ###################

        # START TODO #################
        # normalize and compute self attention
        x = self.norm1(x)
        x = self.attn(x)
        # END TODO ###################

        # START TODO #################
        # residual connection and normalization
        x = residual + x
        x = self.norm2(x)
        # END TODO ###################

        # START TODO #################
        # split patch and class embeddings, apply mlps
        classes, patches = x[:, 0:self.num_classes], x[:, self.num_classes:]
        classes = self.classes_mlp(classes)
        patches = self.patches_mlp(patches)
        # END TODO ###################

        # START TODO #################
        # compute segmentation masks via patch-class similarity (you can use matmul or the @ operator) and normalize.
        masks = patches @ self.embed.transpose(-1, -2)
        masks = self.mask_norm(masks)
        # END TODO ###################

        # reshape masks from (B, H*W, C) to (B, C, H, W)
        masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)

        return masks
