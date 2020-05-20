import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import random

from torch.nn import init

"""
Set of modules for aggregating embeddings of neighbors.
"""


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, features_dim=4096, cuda=False, gcn=False, kernel="GCN"):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        self.softmax = nn.Softmax(dim=1)

        self.kernel = kernel
        self.attention = True if kernel == "GAT" else "False"

        self.in_features = features_dim
        self.out_features = features_dim

        self.weight = nn.Parameter(
            torch.FloatTensor(self.in_features, self.out_features, ))
        self.a = nn.Parameter(
            torch.FloatTensor(2 * self.out_features, 1))
        init.xavier_uniform(self.weight)
        init.xavier_uniform(self.a)

        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, nodes, to_neighs, num_sample=10, average="mean"):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                                        num_sample,
                                        )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        if self.gcn:
            samp_neighs = [samp_neigh | set([nodes[i]])
                           for i, samp_neigh in enumerate(samp_neighs)]

        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # print('agg:', nodes)
        # print('agg unique:', unique_nodes_list)

        column_indices = [unique_nodes[n]
                          for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs))
                       for j in range(len(samp_neighs[i]))]

        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        mask[row_indices, column_indices] = 1

        attention_mask = Variable(torch.full(
            (len(samp_neighs), len(unique_nodes)), np.inf))
        attention_mask[row_indices, column_indices] = 0

        zero_vec = -9e15 * torch.ones_like(mask)

        if self.cuda:
            mask = mask.cuda()
            attention_mask.cuda()

        num_neigh = mask.sum(1, keepdim=True)
        for ni, num in enumerate(num_neigh):
            if num == 0:
                num_neigh[ni] = 1

        if self.cuda:
            embed_matrix = self.features(
                torch.LongTensor(unique_nodes_list).cuda())
            feature_matrix = self.features(torch.LongTensor(nodes).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            feature_matrix = self.features(torch.LongTensor(nodes))

        if self.kernel == "GAT":
            # attention_matrix = feature_matrix.mm(embed_matrix.t())
            #             # attention_matrix = attention_matrix.mul(mask)
            #             # attention_matrix = attention_matrix - attention_mask
            #             # attention = self.softmax(attention_matrix)
            # mask = mask.mul(attention)
            feature_matrix = torch.mm(feature_matrix, self.weight)
            embed_matrix = torch.mm(embed_matrix, self.weight)
            N = feature_matrix.size()[0]
            M = embed_matrix.size()[0]

            a_input = torch.cat([feature_matrix.repeat(1, M).view(N * M, -1), embed_matrix.repeat(N, 1)],
                                dim=1).view(N, -1, 2 * self.out_features)
            attention_matrix = self.leakyrelu(
                torch.matmul(a_input, self.a).squeeze(2))
            # print(attention_matrix.size())
            # print(mask.size())

            # attention_matrix = feature_matrix.mm(embed_matrix.t())

            attention = torch.where(mask > 0, attention_matrix, zero_vec)
            attention = self.softmax(attention)
            # mask = mask.mul(attention)
            to_feats = attention.mm(embed_matrix)
        elif self.kernel == "GCN":
            if average == "mean":
                mask = mask.div(num_neigh)
            to_feats = mask.mm(embed_matrix)
        elif self.kernel == "GIN":
            to_feats = mask.mm(embed_matrix)

        return to_feats


class AttentionAggregator(nn.Module):
    """
    Aggregates a node's embeddings with attention
    """

    def __init__(self, features, in_features=4096, out_features=1024, cuda=False, gcn=False, attention_dim=512):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(AttentionAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        self.softmax = nn.Softmax(dim=1)
        self.attention = True

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))

        self.a = nn.Parameter(
            torch.FloatTensor(2 * out_features, 1))

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                                        num_sample,
                                        )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        if self.gcn:
            samp_neighs = [samp_neigh | set([nodes[i]])
                           for i, samp_neigh in enumerate(samp_neighs)]

        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        column_indices = [unique_nodes[n]
                          for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs))
                       for j in range(len(samp_neighs[i]))]

        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        mask[row_indices, column_indices] = 1

        zero_vec = -9e15 * torch.ones_like(mask)

        if self.cuda:
            mask = mask.cuda()

        num_neigh = mask.sum(1, keepdim=True)
        for ni, num in enumerate(num_neigh):
            if num == 0:
                num_neigh[ni] = 1

        if self.cuda:
            embed_matrix = self.features(
                torch.LongTensor(unique_nodes_list).cuda())
            feature_matrix = self.features(torch.LongTensor(nodes).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            feature_matrix = self.features(torch.LongTensor(nodes))

        attention_matrix = feature_matrix.mm(embed_matrix.t())
        attention = torch.where(mask > 0, attention_matrix, zero_vec)
        attention = self.softmax(attention)

        to_feats = attention.mm(embed_matrix)
        return to_feats
