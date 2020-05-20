import functools

import sklearn
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc

from collections import defaultdict

from Model.encoders import Encoder
from Model.aggregators import MeanAggregator

import warnings
import sklearn.exceptions

from Utils.RARE_INFO import RareInfo

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start_time = time.time()
        res = func(*args, **kw)
        end_time = time.time()
        print('%s executed in %ss' % (func.__name__, end_time - start_time))
        return res

    return wrapper


class UnsupervisedGraphSage(nn.Module):
    def __init__(self, enc):
        super(UnsupervisedGraphSage, self).__init__()
        self.enc = enc
        self.weight = nn.Parameter(torch.FloatTensor(1, enc.embed_dim))
        self.logSig = nn.LogSigmoid()
        self.MSE = nn.MSELoss()
        self.sig = nn.Sigmoid()
        nn.init.xavier_uniform_(self.weight)

    def forward(self, u, v):
        embed_u = self.enc(u)
        embed_v = self.enc(v)
        scores = nn.functional.cosine_similarity(embed_u.t(), embed_v.t())
        return scores

    def sigmoid(self, u, v):
        scores = self.forward(u, v)
        return scores

    def mse_loss(self, nodes, labels):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return self.MSE(scores.squeeze(), labels.squeeze())

    def pos_loss(self, u, v):
        scores = self.forward(u, v)
        return self.logSig(scores)

    def neg_loss(self, nodes_u, neg_samples):
        neg_loss = 0
        for u in nodes_u:
            embed_u = self.enc([u])
            embed_negs = self.enc(list(neg_samples[u]))
            scores = nn.functional.cosine_similarity(embed_u.t(), embed_negs.t())
            neg_loss += self.logSig(torch.mean(-scores))
        return neg_loss

    def loss(self, nodes_u, nodes_v, neg_samples):
        return -sum(self.pos_loss(nodes_u, nodes_v)) - self.neg_loss(nodes_u, neg_samples)


def evaluate(data_name, val_output, test_labels, val, roc=True):
    print("----" * 25)
    print()
    print("%s: " % data_name)

    if roc:
        fpr, tpr, threshold = roc_curve(test_labels[val], val_output.data.numpy().argmax(axis=1))  # 计算真正率和假正率
        roc_auc = auc(fpr, tpr)  # 计算auc的值
        # print('roc_auc:', roc_auc)
        print("roc_auc_score:", roc_auc_score(test_labels[val], val_output.data.numpy().max(axis=1)))

    print("Precision - Recall - F1 score:")
    print(sklearn.metrics.precision_recall_fscore_support(test_labels[val],
                                                          val_output.data.numpy().argmax(axis=1)))
    print()
    print("Macro F1:", f1_score(test_labels[val],
                                val_output.data.numpy().argmax(axis=1), average="macro"))
    print("Macro Recall:", recall_score(test_labels[val],
                                        val_output.data.numpy().argmax(axis=1), average="macro"))
    print("Macro Precision:", precision_score(test_labels[val],
                                              val_output.data.numpy().argmax(axis=1), average="macro"))
    print()
    print("weighted F1:", f1_score(test_labels[val],
                                   val_output.data.numpy().argmax(axis=1), average="weighted"))
    print("weighted Recall:", recall_score(test_labels[val],
                                           val_output.data.numpy().argmax(axis=1), average="weighted"))
    print("weighted Precision:", precision_score(test_labels[val],
                                                 val_output.data.numpy().argmax(axis=1), average="weighted"))
    print()


class RarePredictor:
    def __init__(self, feat_data, b_labels, m_labels, adj_lists, feature_dim,
                 train_enc_num, train_enc_dim, train_sample_num,
                 train, test,
                 pos_samp_num=1, neg_samp_num=1,
                 attention=False, weights_flag=False, weights=[0.5, 0.5],
                 cuda=False,
                 N_WALK_LEN=5, N_WALKS=6):

        self.cuda = cuda

        self.train = train
        self.test = test

        self.pos_samp_num = pos_samp_num
        self.neg_samp_num = neg_samp_num

        self.N_WALK_LEN = N_WALK_LEN
        self.N_WALKS = N_WALKS

        # self.train = [i for i in np.where((m_labels < RareInfo().OTHERS))[0].squeeze() if i in self.train]
        self.test = [i for i in np.where((m_labels < RareInfo().OTHERS))[0].squeeze() if i in self.test]

        self.multi_train = [i for i in np.where(
            (m_labels > RareInfo().NON_RARE) & (m_labels < RareInfo().OTHERS))[0].squeeze() if i in self.train]
        self.multi_test = [i for i in np.where(
            (m_labels > RareInfo().NON_RARE) & (m_labels < RareInfo().OTHERS))[0].squeeze() if i in self.test]
        # print(np.where((m_labels > 0) & (m_labels < 4))[0].squeeze())

        self.b_labels = b_labels  # labels for the first step prediction
        self.m_labels = m_labels  # labels for the multi-classification (the 2nd step)

        self.bi_class_num = 2
        self.multi_class_num = RareInfo().RARE

        # self.multi_class_num = len(set(list(m_labels.squeeze())))
        # print(set(list(m_labels.squeeze())))

        self.features = nn.Embedding(len(feat_data), feature_dim)  # nodes' features (random setting)
        self.features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
        self.adj_lists = adj_lists  # edges' information

        # model parameters
        self.train_enc_dim = train_enc_dim
        self.train_enc_num = train_enc_num
        self.attention = attention
        self.feature_dim = feature_dim
        self.train_sample_num = train_sample_num

        # weighted cross-entropy
        self.weights_flag = weights_flag
        self.class_weights = torch.FloatTensor(weights)

        # labels for test
        self.test_b_labels = b_labels
        self.test_m_labels = m_labels
        self.test_adj = adj_lists

        # inductive settings
        self.is_inductive = False
        self.test_sample_num = train_sample_num
        self.test_features = self.features

        # build aggregator and encoders
        # default: transductive setting
        self.agg1 = MeanAggregator(self.features, cuda=self.cuda, attention=attention)
        self.enc1 = Encoder(self.features, feature_dim, train_enc_dim[0], adj_lists, self.agg1, gcn=True,
                            cuda=self.cuda)

        self.agg2 = MeanAggregator(lambda nodes: self.enc1(nodes).t(), cuda=self.cuda, attention=attention)
        self.enc2 = Encoder(lambda nodes: self.enc1(nodes).t(), self.enc1.embed_dim, train_enc_dim[1], adj_lists,
                            self.agg2, base_model=self.enc1, gcn=True, cuda=self.cuda)

        self.enc1.num_samples = self.train_sample_num[0]
        self.enc2.num_samples = self.train_sample_num[1]

        self.neg_samples = self.get_neg_samples()

    def get_neg_samples(self):
        neg_samples = {}

        negative_pairs = []
        node_negative_pairs = {}

        for node in range(0, len(self.adj_lists)):
            neighbors = {node}
            frontier = {node}
            for i in range(self.N_WALK_LEN):
                current = set()
                for outer in frontier:
                    current |= self.adj_lists[int(outer)]
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(self.train) - neighbors
            node_neg_samples = random.sample(far_nodes, self.neg_samp_num) if self.neg_samp_num < len(
                far_nodes) else far_nodes
            negative_pairs.extend([(node, neg_node) for neg_node in node_neg_samples])
            node_negative_pairs[node] = [(node, neg_node) for neg_node in node_neg_samples]
            neg_samples[node] = node_neg_samples
        return neg_samples

    def run_unsupervised(self, loop_num=100, batch_num=512, lr=0.01):
        np.random.seed(123)
        random.seed(123)
        _set = set
        _sample = random.sample

        train = self.train
        train_enc_num = self.train_enc_num

        if train_enc_num == 1:
            model = UnsupervisedGraphSage(self.enc1)
        else:  # elif self.train_enc_num == 2:
            model = UnsupervisedGraphSage(self.enc2)

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.99))
        times = []

        for batch in range(loop_num):
            batch_nodes = train[:batch_num]
            random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()

            # positive samples
            to_neighs = [self.adj_lists[int(node)] for node in batch_nodes]
            if not self.pos_samp_num is None:
                samp_neighs = [_set(_sample(to_neigh, self.pos_samp_num, ))
                               if len(to_neigh) >= self.pos_samp_num else to_neigh
                               for to_neigh in to_neighs]
            else:
                samp_neighs = to_neighs

            nodes_u = []
            nodes_v = []
            for i, node in enumerate(batch_nodes):
                for v in samp_neighs[i]:
                    nodes_u.append(node)
                    nodes_v.append(v)

            loss = model.loss(nodes_u, nodes_v, self.neg_samples)
            loss.backward()
            optimizer.step()

            end_time = time.time()
            times.append(end_time - start_time)

            print(batch, loss.data)

        print()
        print("Average batch time:", np.mean(times))
        print()
