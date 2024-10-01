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


#
#
# def _ntuple(n):
#     def parse(x):
#         if isinstance(x, collections.abc.Iterable):
#             return x
#         return tuple(repeat(x, n))
#
#     return parse
#
#
# to_2tuple = _ntuple(2)
#
#
# class Mlp(nn.Module):
#     """ MLP as used in Vision Transformer, MLP-Mixer and related networks
#     """
#
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         drop_probs = to_2tuple(drop)
#
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.drop1 = nn.Dropout(drop_probs[0])
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop2 = nn.Dropout(drop_probs[1])
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x
#

def MLP(channels) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=False))
        layers.append(nn.BatchNorm1d(channels[i]))
        layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)


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
        self.alpha = nn.Parameter(torch.ones(1) * 0.8, requires_grad=True)
        self.dist_method = dist_method
        self.lamda=nn.Parameter(torch.ones(1)*0.1,requires_grad=True)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.LeakyReLU(0.1)
        self.num_heads = num_heads
        # assert self.in_dim % self.num_heads == 0, "make sure in_dim % num_heads=0"
        if self.dist_method == "dot":
            hid_dim = self.in_dim // 2
            self.emb_q = nn.Linear(in_dim, hid_dim)
            self.emb_k = nn.Linear(in_dim, hid_dim)
        self.sm = nn.Softmax(dim=-1)
        # self.linear = nn.Linear(in_dim, out_dim, bias=False)
        # self.mode = mode
        self.message_pass = MLP([256, 256, 256])

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

    def get_adj_matrix(self, all_features):
        """
        generate similarity matrix
        :param query_features: (batch, num_perbatch, num_local)
        :param candidate_feature:(batch,num_perbatch,num_local)
        :return: adj_matrix: (batch,num_perbatch,num_perbatch)
        """

        if self.dist_method == 'dot':
            emb_q = self.emb_q(all_features)
            emb_k = self.emb_k(all_features)
            adj_matrix = torch.matmul(emb_q, emb_k.transpose(-1, -2))  # 有可能会出现较大的负数
        elif self.dist_method == 'l2':
            distmat = torch.cdist(all_features.contiguous(), all_features.contiguous(), p=2)
            adj_matrix =torch.exp(-distmat/0.2)
        else:
            raise NotImplementedError


        return adj_matrix

    def forward(self, x, adj=None):
        """
        :param input: (batch, num_perbatch, num_local)
        :param adj: (batch, num_perbatch, num_local)
        :return:(batch,num_perbatch,num_local)
        """
        x = self.message_pass(x)
        input = x.clone()
        x = x.transpose(-1, -2)
        if self.learn_adj or adj == None:
            graph = self.get_adj_matrix(x)
        else:
            graph = adj
        if len(graph.shape) == 4:
            graph = graph +torch.eye(graph.shape[-1], device=graph.device).unsqueeze(0).unsqueeze(0).expand(
                graph.shape[0], graph.shape[1], -1, -1) *self.lamda
        else:
            graph = graph + torch.eye(graph.shape[-1], device=graph.device).unsqueeze(0).expand(graph.shape[0], -1,
                                                                                  -1)*self.lamda
        sort_each = torch.argsort(graph ,dim=-1, descending=True)
        graph[sort_each > 15] = 0
        d_row = torch.sqrt(1 / torch.sum(graph, dim=-1))
        diag_matrix = torch.diag_embed(d_row)
        graph = torch.matmul(diag_matrix,graph)
        graph = torch.matmul(graph, diag_matrix)
        x = torch.matmul(graph, x)
        x = self.bn(x.transpose(-1, -2))
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
        self.bn = nn.BatchNorm1d(256)

        self.selfnode_graph_layers = nn.ModuleList(
            [AttentionGraphLayer(in_dim=256, out_dim=256, learn_adj=learn_adj,
                                 dist_method=dist_method,
                                 norm_layer=norm_layer,
                                 num_heads=graph_num_heads,
                                 decoder_mlp_ratio=decoder_mlp_ratio,
                                 attention_depth=attention_depth, mode="self") for _ in range(num_graph_layers)])
        self.classifier = nn.Sequential(nn.Linear(256, self.num_classes))
        self.cos = nn.CosineSimilarity(dim=-1)
        self.ratio = nn.Parameter(torch.ones(1) * 0.1, requires_grad=True)
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

    def forward(self, query_global, candidate_global=None):
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
        # differ_input = torch.pow(query_global - candidate_global, 2)
        # differ_input = self.bn(differ_input.transpose(-1, -2))
        all_feature=torch.cat((query_global, candidate_global), dim=1).transpose(-1,-2)
        for i in range(self.num_graph_layers):
            all_feature= self.selfnode_graph_layers[i](all_feature)
        if self.num_classes == 1:
            local_score = self.classifier(all_feature.transpose(-1,-2))[:,1:,:]
            local_score = local_score.reshape(query_batch_size, -1)
            final_score = global_score.detach() * self.ratio.clamp(min=0.1, max=0.9) + torch.sigmoid(local_score).detach() * (
                    1 - self.ratio.clamp(min=0.1, max=0.9))
        elif self.num_classes == 2:
            local_score = self.classifier(all_feature.transpose(-1, -2)[:,1:,:])  # (2,11,2)
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
    model = GrapgRerank(infeature_dim=args.local_dim, outfeature_dim=args.local_dim, learn_adj=args.learn_adj,
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
    query_global, candidate_global, x_rerank = torch.rand(1, 256), torch.rand(10, 256), torch.rand(1, 11, 500, 131)
    from thop import profile

    query_global = query_global.to(device="cuda")
    candidate_global = candidate_global.to(device="cuda")
    x_rerank = x_rerank.to(device="cuda")
    torch.cuda.synchronize()
    time_start = time.time()

    flops, params = profile(model, inputs=(query_global, candidate_global))
    torch.cuda.synchronize()
    time_end = time.time()
    time_sum = time_end - time_start
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    print("run tims is {}".format(time_sum))
