import logging
import os
import torch
import torch.nn as nn
from functools import partial

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


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AttentionGraphLayer(nn.Module):
    """
    Attention graphlayer.
    """

    def __init__(self, in_dim, out_dim, learn_adj=True, dist_method='l2', norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 num_heads=4, decoder_mlp_ratio=4, attention_depth=1, mode="cross"):
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
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.learn_adj = learn_adj
        self.alpha = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        self.lamda = nn.Parameter(torch.ones(1) * 0.1, requires_grad=True)
        self.dist_method = dist_method
        self.mode=mode
        self.norm_layer = norm_layer(out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.LeakyReLU(0.1)
        self.num_heads = num_heads
        # assert self.in_dim % self.num_heads == 0, "make sure in_dim % num_heads=0"
        if self.dist_method == "dot":
            hid_dim = self.in_dim // 4
            self.emb_q = nn.Linear(in_dim, hid_dim)
            self.emb_k = nn.Linear(in_dim, hid_dim)
        if mode == "cross":
            self.linear = Mlp(in_features=in_dim)
        else :
            self.linear = nn.Linear(in_dim, out_dim, bias=False)

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
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def get_adj_matrix(self, all_features, query_feature=None):
        """
        generate similarity matrix
        :param query_features: (batch, num_perbatch, num_local)
        :param candidate_feature:(batch,num_perbatch,num_local)
        :return: adj_matrix: (batch,num_perbatch,num_perbatch)
        """
        if query_feature == None:
            query_feature = all_features.clone()
        if self.dist_method == 'dot':
            emb_q = self.emb_q(query_feature)
            emb_k = self.emb_k(all_features)
            adj_matrix = torch.matmul(emb_q, emb_k.transpose(-1, -2))  # 有可能会出现较大的负数
            adj_matrix = torch.softmax(adj_matrix, dim=-1)
            # adj_matrix = F.normalize(adj_matrix, p=1, dim=2)
            adj_matrix = (adj_matrix + adj_matrix.transpose(-1, -2)) * 0.5
        elif self.dist_method == 'l2':
            distmat = torch.cdist(query_feature, all_features, p=2)
            adj_matrix = 1 / (torch.exp(distmat))
        else:
            raise NotImplementedError

        if len(all_features.shape) == 4:
            adj_matrix = adj_matrix + self.lamda * torch.eye(adj_matrix.shape[-1], device=adj_matrix.device).unsqueeze(
                0).unsqueeze(0).expand(adj_matrix.shape[0], adj_matrix.shape[1], -1, -1)
        else:
            adj_matrix = adj_matrix + self.lamda * torch.eye(adj_matrix.shape[-1], device=adj_matrix.device).unsqueeze(
                0).expand(adj_matrix.shape[0], -1, -1)
        # d_row = torch.sqrt(1 / torch.sum(adj_matrix, dim=-1))
        # diag_matrix = torch.diag_embed(d_row)
        # adj_matrix = torch.matmul(diag_matrix, adj_matrix)
        # adj_matrix = torch.matmul(adj_matrix, diag_matrix)
        return adj_matrix

    def forward(self, x, cross=False, adj=None):
        """
        :param input: (batch, num_perbatch, num_local)
        :param adj: (batch, num_perbatch, num_local)
        :return:(batch,num_perbatch,num_local)
        """
        input = x.clone()
        x = self.linear(x)
        if cross:
            y = x[:, :1, :, :].clone()
        else:
            y = None
        if self.learn_adj or adj == None:
            graph = self.get_adj_matrix(query_feature=y, all_features=x)
        else:
            graph = adj
        x = torch.matmul(graph, x)
        if type(self.norm_layer) == nn.LayerNorm:
            x = self.norm_layer(x)
        x = self.relu(x)
        return (1 - self.alpha.clamp(min=0.1, max=0.9)) * input + self.alpha.clamp(min=0.1, max=0.9) * x


class GrapgRerank(nn.Module):
    def __init__(self, infeature_dim, outfeature_dim, train_batch_size, rerank_batch_size,
                 aggregation_select, infer_batch_size, local_dim,
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
                                 attention_depth=attention_depth, mode="self") for _ in range(num_graph_layers)])
        self.crossnode_graph_layers = nn.ModuleList(
            [AttentionGraphLayer(in_dim=infeature_dim, out_dim=outfeature_dim, learn_adj=learn_adj,
                                 dist_method=dist_method,
                                 norm_layer=norm_layer,
                                 num_heads=graph_num_heads,
                                 decoder_mlp_ratio=decoder_mlp_ratio,
                                 attention_depth=attention_depth, mode="cross") for _ in range(num_graph_layers)])
        self.image_graph_layers = nn.ModuleList(
            [AttentionGraphLayer(in_dim=num_local, out_dim=num_local, learn_adj=learn_adj,
                                 dist_method=dist_method,
                                 norm_layer=norm_layer,
                                 num_heads=graph_num_heads,
                                 decoder_mlp_ratio=decoder_mlp_ratio,
                                 attention_depth=attention_depth, mode="image") for _ in range(num_graph_layers)])
        self.fc1 = nn.Linear(local_dim + 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.active = nn.GELU()
        self.dim_reduce = nn.Sequential(
            *[self.fc1, nn.LayerNorm(64), self.active, self.fc2, nn.LayerNorm(32), self.active,
              self.fc3, nn.LayerNorm(16), self.active,
              self.fc4, self.active])
        # self.dim_reduce = Mlp(in_features=infeature_dim, hidden_features=64, out_features=1)
        self.classifier = nn.Linear(self.num_local, num_classes, bias=True)
        self.cos = nn.CosineSimilarity(dim=-1)
        self.ratio = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        self.sm = torch.nn.Softmax(dim=2)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, query_global, candidate_global=None, x_rerank=None, adj=None):
        """
        :param query_global: (batchsize,feature-dim)
        :param candidate_global: (batchsize*11,feature-dim)
        :param x_rerank:(batchsize,12,num_local,feature_dim)
        :param adj:
        :return: (batchsize
        """
        query_batch_size, global_feature_num = query_global.size()
        query_global = query_global.view(query_batch_size, -1, global_feature_num)  # (2,1,256)
        candidate_global = candidate_global.view(query_batch_size, -1, global_feature_num)  # (2,11,256)
        global_score = self.cos(query_global.detach(), candidate_global.detach())  # (2,11)
        for i in range(self.num_graph_layers):
            x_rerank = self.selfnode_graph_layers[i](x_rerank)
            x_rerank = self.crossnode_graph_layers[i](x_rerank, cross=True)

        x_rerank = self.dim_reduce(x_rerank).squeeze(-1)
        x_rerank = x_rerank.view(query_batch_size, -1, self.num_local)  # (2,12,500)

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
        # logging.info(f"self.ration{self.ratio}")
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
                        aggregation_select=args.aggregation_select, num_local=args.num_local
                        )

    return model


if __name__ == "__main__":
    args = parse_arguments()

    model = getGcnRerank(args)
    model.to(device="cuda")
    query_global, candidate_global, x_rerank = torch.rand(1, 256), torch.rand(100, 256), torch.rand(1, 101, 500, 131)
    from thop import profile

    query_global = query_global.to(device="cuda")
    candidate_global = candidate_global.to(device="cuda")
    x_rerank = x_rerank.to(device="cuda")
    torch.cuda.synchronize()
    time_start = time.time()

    flops, params = profile(model, inputs=(query_global, candidate_global, x_rerank))
    torch.cuda.synchronize()
    time_end = time.time()
    time_sum = time_end - time_start
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    print("run tims is {}".format(time_sum))
