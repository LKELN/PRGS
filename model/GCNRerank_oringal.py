import logging
import os
import re

import torch
import torch.nn as nn
from functools import partial

from einops import rearrange
from ptflops import get_model_complexity_info
from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed, Block
import argparse
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from itertools import repeat
import collections.abc
import time
import math
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from parser_1 import parse_arguments
from copy import deepcopy

save_number = 0


def max_min_norm_tensor(mat):
    v_min, _ = torch.min(mat, -1, keepdims=True)
    v_max, _ = torch.max(mat, -1, keepdims=True)
    mat = (mat - v_min) / (v_max - v_min)
    return mat


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNorm2d, self).__init__()

        self.channels = channels
        self.eps = torch.tensor(eps)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = input.mean(1, keepdim=True)
        std = torch.sqrt(input.var(1, unbiased=False, keepdim=True) + self.eps)
        out = (input - mean) / std
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out

    def extra_repr(self):
        return '{channels}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class FixedAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()

        # self.position_embedding_type = position_embedding_type
        self.is_absolute = True

        inv_freq = 1. / (10000 ** (torch.arange(0, hidden_size, 2, dtype=torch.float) / hidden_size))
        position = torch.arange(max_position_embeddings, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j -> ij', position, inv_freq)
        embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('embeddings', embeddings)

    def forward_fixed(self, x):
        """
        return (b l d)
        """
        return x + self.embeddings[None, :x.size(1), :]

    def _forward(self, x):
        return self.forward_fixed(x)

    def forward(self, x):
        if x.dim() == 3:
            return self._forward(x)
        elif x.dim() == 4:
            h = x.size(1)
            x = rearrange(x, 'b h l d -> (b h) l d')
            x = self._forward(x)
            x = rearrange(x, '(b h) l d -> b h l d', h=h)
            return x


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Projection(nn.Module):
    def __init__(self, input_channels, output_channels, act_layer=nn.GELU, mode='single'):
        super().__init__()
        tmp = []
        self.mode = mode
        if mode == "single":
            # stride = int(re.findall(r"\d+",mode)[0])
            ks = 3
            padding = 1
            stride = 2
            tmp.append(
                nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=ks, stride=stride,
                          padding=padding, groups=input_channels))
            tmp.extend([
                LayerNorm2d(input_channels),
                act_layer(),
            ])

            self.proj = nn.Sequential(*tmp)
        elif mode == "multi":
            ks = 2
            padding = 0
            stride = 2
            tmp = []
            tmp.append(
                nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=ks, stride=stride,
                          padding=padding, groups=input_channels))
            tmp.extend([
                LayerNorm2d(input_channels),
                act_layer(),
            ])
            self.proj1 = nn.Sequential(*tmp)
            ks = 4
            padding = 0
            stride = 4
            tmp = []
            tmp.append(
                nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=ks, stride=stride,
                          padding=padding, groups=input_channels))
            tmp.extend([
                LayerNorm2d(input_channels),
                act_layer(),
            ])
            self.proj2 = nn.Sequential(*tmp)

        else:

            tmp.append(nn.Identity())
            self.proj = nn.Sequential(*tmp)
        self.peg = nn.Conv2d(output_channels, output_channels, kernel_size=1, padding=0, groups=output_channels,
                             bias=False)

    def forward(self, x, H, W):
        if self.mode == "multi":
            B, N, HW, C = x.shape
            x = x.transpose(-1, -2)
            x = x.reshape(B * N, C, H, W)
            input = x.clone()
            pos = self.peg(input)
            input = (input + pos).flatten(2)
            x_1 = self.proj1(x)
            pos = self.peg(x_1)
            x_1 = (x_1 + pos).flatten(2)
            x_2 = self.proj2(x)
            pos = self.peg(x_2)
            x_2 = (x_2 + pos).flatten(2)
            x = torch.cat((input,x_1, x_2), dim=-1)
            x = x.reshape(B, N, C, -1).transpose(-1, -2)
            Hout, Wout = H, W
        else :
            B, N, HW, C = x.shape  # 30,40
            x = x.transpose(-1, -2)
            x = x.reshape(B * N, C, H, W)
            x = self.proj(x)
            pos = self.peg(x)
            x = x + pos
            _, D, Hout, Wout = x.shape
            # torch.save(x, "single.pt")
            x = x.reshape(B, N, D, Hout, Wout).flatten(3).transpose(-1, -2)



        return x, Hout, Wout


class AttentionGraphLayer(nn.Module):
    """
    Attention graphlayer.
    """

    def __init__(self, in_dim, out_dim, learn_adj=True, dist_method='l2', norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 num_heads=4, temperature=0.5, mode="cross",
                 ):
        """
        :param in_dim: input feature size.
        :param out_dim: output feature size.
        :param learn_adj: learn a adj mat  or not.
        :param alpha:Input-Output Weight Ratio.
        :param dist_method: calculate the similarity between the vertex.
        :param k: nearest neighbor size.
        :param
        """
        super(AttentionGraphLayer, self).__init__()
        self.temperature = temperature
        self.learn_adj = learn_adj
        self.alpha = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        self.lamda = nn.Parameter(torch.ones(1) * 0.1, requires_grad=True)
        self.dist_method = dist_method
        self.mode = mode
        self.norm_layer = norm_layer(out_dim)
        self.relu = nn.GELU()
        self.num_heads = num_heads
        self.sm = nn.Softmax(dim=-1)
        if self.dist_method == "dot":
            hid_dim = self.in_dim
            self.emb_q = nn.Linear(in_dim, hid_dim)
            self.emb_k = nn.Linear(in_dim, hid_dim)
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_adj_matrix(self, all_features, query_feature=None):
        """
        generate similarity matrix
        :param query_features: (batch, num_perbatch, num_local)
        :param candidate_feature:(batch,num_perbatch,num_local)
        :return: adj_matrix: (batch,num_perbatch,num_perbatch)
        """
        global save_number
        if query_feature == None:
            query_feature = all_features.clone()
        if self.dist_method == 'dot':
            query_feature = self.emb_q(query_feature)
            all_features = self.emb_k(all_features)
            adj_matrix = torch.matmul(query_feature, all_features.transpose(-1, -2))  # 有可能会出现较大的负数
        elif self.dist_method == 'l2':
            distmat = torch.cdist(query_feature, all_features, p=2)
            adj_matrix = -distmat / self.temperature
        else:
            raise NotImplementedError

        if len(all_features.shape) == 4:
            adj_matrix = adj_matrix + self.lamda * torch.eye(adj_matrix.shape[-1], device=adj_matrix.device).unsqueeze(
                0).unsqueeze(0).expand(adj_matrix.shape[0], adj_matrix.shape[1], -1, -1)
        else:
            adj_matrix = adj_matrix + self.lamda * torch.eye(adj_matrix.shape[-1], device=adj_matrix.device).unsqueeze(
                0).expand(adj_matrix.shape[0], -1, -1)
        adj_matrix = self.sm(adj_matrix)

        return adj_matrix

    def forward(self, x, adj=None):
        """
        :param input: (batch, num_perbatch, num_local)
        :param adj: (batch, num_perbatch, num_local)
        :return:(batch,num_perbatch,num_local)
        """
        input = x.clone()
        x = self.linear(x)
        if self.mode == "cross":
            y = x[:, :1, :, :].clone()
        else:
            y = None
        if self.learn_adj or adj == None:
            graph = self.get_adj_matrix(query_feature=y, all_features=x)
        else:
            graph = adj
        x = torch.matmul(graph, x)
        x = self.norm_layer(x)
        x = self.relu(x)
        return (1 - self.alpha.clamp(min=0.1, max=0.9)) * input + self.alpha.clamp(min=0.1, max=0.9) * x


class GrapgRerank(nn.Module):
    def __init__(self, infeature_dim, outfeature_dim, train_batch_size, rerank_batch_size,
                 infer_batch_size, temperature,
                 dist_method='l2', num_classes=2,
                 num_graph_layers=2, pool=[None, "single", "multi"],
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 graph_num_heads=4, learn_adj=True,
                 ):
        super(GrapgRerank, self).__init__()
        # self.patch_size = patch_size
        self.infer_batch_size = infer_batch_size
        self.train_batch_size = train_batch_size
        self.rerank_batch_size = rerank_batch_size
        self.num_graph_layers = num_graph_layers
        self.num_classes = num_classes
        self.num_local_dyn =385
        self.fine2cor = nn.ModuleList()
        for i in range(num_graph_layers):
            self.fine2cor.append(Projection(input_channels=infeature_dim,
                                            output_channels=outfeature_dim,
                                            act_layer=nn.GELU,
                                            mode=pool[i]))

        self.selfnode_graph_layers = nn.ModuleList(
            [AttentionGraphLayer(in_dim=infeature_dim, out_dim=outfeature_dim, learn_adj=learn_adj,
                                 dist_method=dist_method,
                                 norm_layer=norm_layer,
                                 num_heads=graph_num_heads,
                                 temperature=temperature,
                                 mode="self") for _ in
             range(num_graph_layers)])
        self.crossnode_graph_layers = nn.ModuleList(
            [AttentionGraphLayer(in_dim=infeature_dim, out_dim=outfeature_dim, learn_adj=learn_adj,
                                 dist_method=dist_method,
                                 norm_layer=norm_layer,
                                 num_heads=graph_num_heads,

                                 temperature=temperature,
                                 mode="cross") for _ in
             range(num_graph_layers)])
        self.image_graph_layers = nn.ModuleList(
            [AttentionGraphLayer(in_dim=self.num_local_dyn, out_dim=self.num_local_dyn, learn_adj=learn_adj,
                                 dist_method=dist_method,
                                 norm_layer=norm_layer,
                                 num_heads=graph_num_heads, temperature=temperature,
                                 mode="image") for _ in
             range(num_graph_layers)])

        self.dim_reduce = nn.Sequential(
            *[nn.Linear(infeature_dim, 64), nn.LayerNorm(64), nn.GELU(),
              nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU(),
              nn.Linear(32, 16), nn.LayerNorm(16), nn.GELU(),
              nn.Linear(16, 1), nn.GELU()])

        self.classifier = nn.Linear(self.num_local_dyn, num_classes, bias=True)
        self.sm = torch.nn.Softmax(dim=-1)
        self._init_params()

    def _init_params(self):
        # trunc_normal_(self.pos_embed, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, query_global, candidate_global=None, x_rerank=None, adj=None):
        """
        :param query_global: (batchsize,feature-dim)
        :param candidate_global: (batchsize*11,feature-dim)
        :param x_rerank:(batchsize,12,num_local,feature_dim)
        :param adj:
        :return:
        """
        Hout, Wout = 30, 40
        for i in range(self.num_graph_layers):
#             if i<3:
            x_rerank, Hout, Wout = self.fine2cor[i](x_rerank, Hout, Wout)
            x_rerank = self.selfnode_graph_layers[i](x_rerank)
            x_rerank = self.crossnode_graph_layers[i](x_rerank)

        x_rerank = self.dim_reduce(x_rerank).flatten(2)

        for graph_layer in self.image_graph_layers:
            x_rerank = graph_layer(x_rerank, adj=adj)

        x_rerank = self.classifier(x_rerank)  # (2,12,2)
        if self.num_classes == 1:
            local_score = torch.sigmoid(x_rerank)
            # local_score = local_score[:, 1:, :].reshape(query_batch_size, -1)
        elif self.num_classes == 2:
            local_score = x_rerank[:, 1:, :]  # (2,11,2)
        return local_score, None  # (2,11,2)  ,(2,11)


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def getGcnRerank(args):
    model = GrapgRerank(infeature_dim=args.local_dim+3 , outfeature_dim=args.local_dim+3, learn_adj=args.learn_adj,
                        dist_method=args.dist_method,
                        num_classes=args.num_classes, infer_batch_size=args.infer_batch_size,
                        train_batch_size=args.train_batch_size, rerank_batch_size=args.rerank_batch_size,
                        num_graph_layers=args.num_graph_layers,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),

                        temperature=args.temperature
                        )

    return model


if __name__ == "__main__":
    args = parse_arguments()
    device = "cuda"
    model = getGcnRerank(args)
    model.to(device)
    query_global, candidate_global, x_rerank = torch.rand(1, 256), torch.rand(100, 256), torch.rand(1, 101, 1200, 131)
    from thop import profile
    # model_1=torch.load("/home/think/PycharmProjects/GCNcode/resume/CVPR23_DeitS_Rerank.pth")
    query_global = query_global.to(device)
    candidate_global = candidate_global.to(device)
    x_rerank = x_rerank.to(device)
    torch.cuda.synchronize()
    time_start = time.time()

    flops, params = profile(model, inputs=(query_global, candidate_global, x_rerank))
    torch.cuda.synchronize()
    time_end = time.time()
    time_sum = time_end - time_start
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    print("run tims is {}".format(time_sum))
