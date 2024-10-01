import logging
import os
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


class MLPLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        self.linear = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            # nn.Linear(in_dim, out_dim),
            # nn.GELU()
        )

    def forward(self, x):
        return x + self.linear(x)


class graph_MLPLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        self.linear = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.linear(x)


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


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, :, 0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, :, 1])  # (H*W, D/2)
    emb = torch.cat([emb_h, emb_w], dim=2)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32).cuda()
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    # pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('bm,d->bmd', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=2)  # (M, D)
    return emb


class Mlp_self(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, p=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout1 = nn.Dropout(p=p)
        # self.dropout2 = nn.Dropout(p=p)
        self.act = act_layer()
        self.ln_1 = nn.LayerNorm(in_features)

    def forward(self, x):
        input = x.clone()
        x = self.ln_1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        # x = self.dropout1(x)
        x = input + x
        return x


class Mlp_cross(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, p=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.fc4 = nn.Linear(hidden_features, out_features)
        self.dropout1 = nn.Dropout(p=p)
        self.dropout2 = nn.Dropout(p=p)
        # self.dropout3 = nn.Dropout(p=p)
        # self.dropout4 = nn.Dropout(p=p)
        self.act = act_layer()
        self.ln_1 = nn.LayerNorm(in_features)
        self.ln_2 = nn.LayerNorm(in_features)

    def forward(self, x):
        input = x.clone()
        x = self.ln_1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        # x = self.dropout1(x)
        x = input + x
        input = x.clone()
        x = self.ln_2(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.fc4(x)
        x = self.act(x)
        # x = self.dropout2(x)
        x = input + x
        return x


class AttentionGraphLayer(nn.Module):
    """
    Attention graphlayer.
    """

    def __init__(self, in_dim, out_dim, learn_adj=True, dist_method='l2', norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 num_heads=4, decoder_mlp_ratio=4, temperature=0.5, attention_depth=1, mode="cross", cross_depth=1,
                 self_depth=1,
                 num_local=500, cross_limit=30,
                 graph_depth=1):
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
        self.cross_limit = cross_limit
        self.in_dim = in_dim
        self.out_dim = out_dim
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
        if mode == "cross":
            self.linear = nn.Linear(in_dim, out_dim, bias=True)
        elif mode == "self":
            self.linear = nn.Linear(in_dim, out_dim, bias=True)
            # nn.Sequential(*[MLPLayer(in_dim=in_dim,hidden_dim=in_dim,out_dim=in_dim) for _ in range(self_depth)])
        elif mode == "image":
            self.linear = nn.Linear(in_dim, out_dim, bias=True)
        else:
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

        # if self.mode == "cross":
        #     sort_each = torch.argsort(adj_matrix, descending=True, dim=-1)
        #     new_adj_matrix = torch.zeros_like(adj_matrix).scatter_(-1, sort_each[:, :, :, :self.cross_limit], 1)
        #     adj_matrix = torch.mul(new_adj_matrix, adj_matrix)
        #     adj_matrix[adj_matrix == 0] = -1e8
        # if self.mode == "self":
        #     sort_each = torch.argsort(adj_matrix, descending=True, dim=-1)
        #     new_adj_matrix = torch.zeros_like(adj_matrix).scatter_(-1, sort_each[:, :, :, :30], 1)
        #     adj_matrix = torch.mul(new_adj_matrix, adj_matrix)
        #     adj_matrix[adj_matrix == 0] = -1e8

        adj_matrix = self.sm(adj_matrix)
        # path='../visible'
        # if self.training == False and save_number < 9:
        #     torch.save(adj_matrix,os.path.join(path,"adj_matrix_" + str(save_number) +self.mode+ ".pt"))

        return adj_matrix

    def forward(self, x, adj=None):
        """
        :param input: (batch, num_perbatch, num_local)
        :param adj: (batch, num_perbatch, num_local)
        :return:(batch,num_perbatch,num_local)
        """
        input = x.clone()
        # print(input.shape)
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
                 aggregation_select, infer_batch_size, local_dim, temperature, cross_limit=30,
                 dist_method='l2', num_classes=2,
                 num_graph_layers=2,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 graph_num_heads=4, decoder_mlp_ratio=4, attention_depth=1, num_local=500, learn_adj=True):
        super(GrapgRerank, self).__init__()

        self.in_dim = infeature_dim
        self.out_dim = outfeature_dim
        self.num_local = num_local
        self.local_dim = local_dim
        self.infer_batch_size = infer_batch_size
        self.train_batch_size = train_batch_size
        self.rerank_batch_size = rerank_batch_size
        self.num_graph_layers = num_graph_layers
        self.num_classes = num_classes
        self.selfnode_graph_layers = nn.ModuleList(
            [AttentionGraphLayer(in_dim=infeature_dim, out_dim=outfeature_dim, learn_adj=learn_adj,
                                 dist_method=dist_method,
                                 norm_layer=norm_layer,
                                 num_heads=graph_num_heads,
                                 decoder_mlp_ratio=decoder_mlp_ratio,
                                 attention_depth=attention_depth, cross_limit=cross_limit, temperature=temperature,
                                 mode="self") for _ in
             range(num_graph_layers)])
        self.crossnode_graph_layers = nn.ModuleList(
            [AttentionGraphLayer(in_dim=infeature_dim, out_dim=outfeature_dim, learn_adj=learn_adj,
                                 dist_method=dist_method,
                                 norm_layer=norm_layer,
                                 num_heads=graph_num_heads,
                                 decoder_mlp_ratio=decoder_mlp_ratio,
                                 attention_depth=attention_depth, cross_limit=cross_limit, temperature=temperature,
                                 mode="cross") for _ in
             range(num_graph_layers)])
        self.image_graph_layers = nn.ModuleList(
            [AttentionGraphLayer(in_dim=num_local, out_dim=num_local, learn_adj=learn_adj,
                                 dist_method=dist_method,
                                 norm_layer=norm_layer,
                                 num_heads=graph_num_heads,
                                 decoder_mlp_ratio=decoder_mlp_ratio,
                                 attention_depth=attention_depth, cross_limit=cross_limit, temperature=temperature,
                                 mode="image") for _ in
             range(num_graph_layers)])
        # if self.in_dim == 131:
        self.dim_reduce = nn.Sequential(
            *[nn.Linear(infeature_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32), nn.LayerNorm(32),
              nn.GELU(),
              nn.Linear(32, 16), nn.LayerNorm(16), nn.GELU(),
              nn.Linear(16, 1), nn.GELU()])
        # elif self.in_dim == 259:
        #     self.dim_reduce = nn.Sequential(
        #         *[nn.Linear(infeature_dim, 128), nn.LayerNorm(128), nn.GELU(),
        #           nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
        #           nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU(),
        #           nn.Linear(32, 16), nn.LayerNorm(16), nn.GELU(),
        #           nn.Linear(16, 1), nn.GELU()])
        # self.dim_linear = nn.ModuleList(
        #     [MLPLayer(in_dim=infeature_dim, hidden_dim=infeature_dim, out_dim=infeature_dim) for _ in range(3)])
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_local, self.in_dim),requires_grad=True)
        # self.dim_reduce = Mlp(in_features=infeature_dim, hidden_features=64, out_features=1)
        # self.emndding_attention = nn.Embedding(self.num_local + 1, self.in_dim)
        self.classifier = nn.Linear(self.num_local, num_classes, bias=True)

        self.cos = nn.CosineSimilarity(dim=-1)
        self.ratio = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
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
        query_batch_size, global_feature_num = query_global.size()
        query_global = query_global.unsqueeze(1)
        candidate_global = candidate_global.view(query_batch_size, -1, global_feature_num)  # (2,11,256)
        global_score = self.cos(query_global.detach(), candidate_global.detach())  # (2,11)

        for i in range(self.num_graph_layers):
            x_rerank = self.selfnode_graph_layers[i](x_rerank)
            x_rerank = self.crossnode_graph_layers[i](x_rerank)
        x_rerank = self.dim_reduce(x_rerank).squeeze(-1)
        for graph_layer in self.image_graph_layers:
            x_rerank = graph_layer(x_rerank, adj=adj)

        x_rerank = self.classifier(x_rerank)  # (2,12,2)
        if self.num_classes == 1:
            local_score = torch.sigmoid(x_rerank)
            local_score = local_score[:, 1:, :].reshape(query_batch_size, -1)
            final_score = global_score.detach() * self.ratio.clamp(min=0.1, max=0.9) + local_score.detach() * (
                    1 - self.ratio.clamp(min=0.1, max=0.9))
        elif self.num_classes == 2:
            local_score = x_rerank[:, 1:, :]  # (2,11,2)
            final_score = global_score.detach() * self.ratio.clamp(min=0.1, max=0.9) + self.sm(local_score).detach()[:,
                                                                                       :, 1] * (
                                  1 - self.ratio.clamp(min=0.1, max=0.9))  # (2,11)+(2,11)
        return local_score, final_score  # (2,11,2)  ,(2,11)


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
    model = GrapgRerank(infeature_dim=args.local_dim + 3, outfeature_dim=args.local_dim + 3, learn_adj=args.learn_adj,
                        dist_method=args.dist_method,
                        num_classes=args.num_classes, infer_batch_size=args.infer_batch_size,
                        train_batch_size=args.train_batch_size, rerank_batch_size=args.rerank_batch_size,
                        num_graph_layers=args.num_graph_layers,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), local_dim=args.local_dim,
                        graph_num_heads=4, decoder_mlp_ratio=4, attention_depth=args.attention_depth,
                        aggregation_select=args.aggregation_select, num_local=args.num_local,
                        temperature=args.temperature
                        )

    return model


if __name__ == "__main__":
    args = parse_arguments()
    device = "cuda"
    model = getGcnRerank(args)
    model.to(device)
    query_global, candidate_global, x_rerank = torch.rand(1, 256), torch.rand(100, 256), torch.rand(1, 101, 1000, 131)
    from thop import profile
    # a=np.load("/home/flztiii/llg/R2Former/result/Pittsburgh30k_v2_deit_hard_final_distance.npy")
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
