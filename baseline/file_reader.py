from collections import defaultdict

import numpy as np


def load_medicine(num_malacards, num_testset):
    num_nodes = num_malacards + num_testset
    num_feats = 8090
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("../graphsage_simple/my/feat_190203_2.txt") as fp:
        for i, line in enumerate(fp):
            info = line.strip('\n').split(' ')
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("../graphsage_simple/my/adj_190203_2.txt") as fp:
        for i, line in enumerate(fp):
            info = line.strip('\n').split(' ')
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists
