import random
import time
from collections import defaultdict

import numpy as np
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.neighbors import NearestCentroid
from sklearn.multiclass import OneVsRestClassifier

from Utils.RARE_INFO import RareInfo
from Model.load_dataset import load_diseases_map


def evaluation(name, labels, output, topk=(1, 2, 3, 4, 5)):
    print('baseline:', name)
    # print("weighted f1, recall, precision:", )
    # w_f1 = f1_score(labels, output, average="weighted")
    # w_recall = recall_score(labels, output, average="weighted")
    # w_p = precision_score(labels, output, average="weighted")
    # print("%f\t%f\t%f" % (w_f1, w_recall, w_p))

    # if " Binary " in name:
    #     print("roc_auc_score:", roc_auc_score(labels, output))

    # print()
    # print("macro f1, recall, precision:", )
    # m_f1 = f1_score(labels, output, average="macro")
    # m_recall = recall_score(labels, output, average="macro")
    # m_p = precision_score(labels, output, average="macro")
    # print("%f\t%f\t%f" % (m_f1, m_recall, m_p))
    # print()

    # shape: batchnum * classnum
    target = labels
    output = output  # shape: batchnum * classnum

    # for line in output:
    #     print(line)
    # print(target.shape, output.shape)
    maxk = max(topk)
    batch_size = target.shape[0]

    def partition_arg_topK(matrix, K, axis=-1):
        """
        perform topK based on np.argpartition
        :param matrix: to be sorted
        :param K: select and sort the top K items
        :param axis: 0 or 1. dimension to be sorted.
        :return:
        """
        a_part = np.argpartition(matrix, K, axis=axis)
        if axis == 0:
            row_index = np.arange(matrix.shape[1 - axis])
            a_sec_argsort_K = np.argsort(
                matrix[a_part[0:K, :], row_index], axis=axis)
            return a_part[0:K, :][a_sec_argsort_K, row_index]
        else:
            column_index = np.arange(matrix.shape[1 - axis])[:, None]
            a_sec_argsort_K = np.argsort(
                matrix[column_index, a_part[:, 0:K]], axis=axis)
            return a_part[:, 0:K][column_index, a_sec_argsort_K]

    pred = partition_arg_topK(-output, maxk, axis=1)

    correct = np.zeros((batch_size, maxk))
    for i in range(batch_size):
        for k in range(maxk):
            correct[i, k] = 1 if target[i][pred[i, k]] == 1 else 0

    correct = correct.T
    # print(correct)

    correct_target = target.sum(axis=1)
    # print(correct_target)

    for k in topk:
        correct_k = correct[:k].sum(axis=0)
        # print("correct k:", correct_k)

        precision_k = 0.0
        recall_k = 0.0
        for i in range(0, batch_size):
            # _k = k if k < int(correct_target[i]) else int(correct_target[i])
            _k = k
            precision_k += float(correct_k[i]) / _k
            recall_k += float(correct_k[i]) / float(correct_target[i])
        # print("sum precision:", precision_k, "sum recall:", recall_k)
        precision_k = precision_k / batch_size
        recall_k = recall_k / batch_size

        f1_k = 2 * precision_k * recall_k / (precision_k + recall_k)

        print("precision @ %d : %.5f, recall @ %d : %.5f, f1 @ %d : %.5f" % (
            k, precision_k, k, recall_k, k, f1_k))


def load_multi_graph(graph_path):
    pos = graph_path.find("/Datasets/disease_prediction")
    data_path = graph_path[:pos + 28]
    graph_date = graph_path[pos + 40: pos + 46]
    graph_type = graph_path[pos + 53]
    map_path = "{}/graph_data/{}/diseases-map-{}.txt".format(
        data_path, graph_date, graph_date)
    diseases_map = load_diseases_map(map_path)

    node_list = []
    node_label = {}
    node_attr = {}
    adj_lists = defaultdict(set)
    with open(graph_path + ".node", "r", encoding="utf8") as f:
        for line in f:
            line = line.strip("\n").split("\t")
            if line[0] not in node_list:
                node_list.append(line[0])
            n_label = np.zeros(len(diseases_map))
            if graph_type == 'M':
                n_label[diseases_map[line[0]] - 1] = 1
            if graph_type == 'P':
                n_label[list(
                    map(lambda x: diseases_map[x] - 1, line[1:-2]))] = 1
            main_disease = int(line[-2]) - 1
            rare_flag = int(line[-1])
            node_label[line[0]] = n_label
            node_attr[line[0]] = (main_disease, rare_flag)

    with open(graph_path + ".edge", "r", encoding="utf8") as f:
        for line in f:
            line = line.strip("\n").split("\t")
            adj_lists[line[0]].add(line[1])

    node_map = {}
    feature_map = {}
    for node in node_list:
        node_map[node] = len(node_map)
        for adj in adj_lists[node]:
            if adj not in feature_map:
                feature_map[adj] = len(feature_map)

    feat_data = np.zeros((len(node_map), len(feature_map)), dtype=np.float64)
    labels = np.zeros((len(node_list), len(diseases_map)), dtype=np.int64)
    main_disease = np.zeros((len(node_list), 1), dtype=np.int64)
    rare_type = np.zeros((len(node_list), 1), dtype=np.int64)

    for node in node_list:
        node_id = node_map[node]
        labels[node_id] = node_label[node]
        rare_type[node_id] = node_attr[node][1]
        main_disease[node_id] = node_attr[node][0]
        for neighbor in adj_lists[node]:
            feat_data[node_id, feature_map[neighbor]] = 1

    file_name_train = graph_path + "-transductive-train.index"
    file_name_test = graph_path + "-transductive-test.index"

    with open(file_name_train, "r", encoding="utf8") as f:
        train = [node_map[line.strip("\n")] for line in f]
    with open(file_name_test, "r", encoding="utf8") as f:
        test = [node_map[line.strip("\n")] for line in f]

    multi_test = [i for i in np.where(rare_type > 0)[0].squeeze() if i in test]

    return feature_map, train, test, multi_test, feat_data, labels, main_disease


def run(data_path, file_date, file_suffix, MODEL="Random Forest",):
    data_path = data_path + "/graph_data/" + file_date
    file_patient_graph = data_path + "/graph-P-" + file_date + "-" + file_suffix

    feature_map, train, test, multi_test, feat_data, labels, main_disease = load_multi_graph(
        file_patient_graph)

    # print(feature_map_medical.keys() & feature_map_patient.keys())
    # print(len(feature_map_medical.keys() & feature_map_patient.keys()))

    feat_train = feat_data[train]
    label_train = labels[train]
    feat_test = feat_data[test]
    label_test = labels[test]
    main_disease_train = main_disease[train]
    main_disease_test = main_disease[test]

    feat_test_multi = feat_data[multi_test]
    label_test_multi = labels[multi_test]
    main_disease_test_multi = main_disease[multi_test]

    test_rare_index = [test.index(i) for i in multi_test]

    # MODEL = "Nearest Centroid"
    # MODEL = "SVM"
    # MODEL = "Random Forest"
    # MODEL = "Random Forest"

    if MODEL == "SVM":
        # SVM
        clf = svm.SVC(kernel='linear', probability=True)
        clf_multi = svm.SVC(kernel='linear', probability=True)
    elif MODEL == "Random Forest":
        # Random Forest
        clf = RandomForestClassifier(n_estimators=10)
        clf_multi = RandomForestClassifier(n_estimators=10)
    elif MODEL == "Nearest Centroid":
        # Nearest Centroid
        clf = NearestCentroid()
        clf_multi = NearestCentroid()
    elif MODEL == "Decision Tree":
        # Decision Tree
        clf = tree.DecisionTreeClassifier()
        clf_multi = tree.DecisionTreeClassifier()

    t_time = time.time()
    clf = clf = OneVsRestClassifier(clf, n_jobs=-1)
    clf.fit(feat_train, label_train)
    print('train.', time.time() - t_time)
    t_time = time.time()
    # output_test = clf.predict(feat_test)
    pred = clf.predict_proba(feat_test)
    print('predict.', time.time() - t_time)
    print()
    print(pred.shape)

    # overall:
    evaluation(MODEL + ', overall:', label_test, pred)

    # rare:
    print(len(test_rare_index))
    evaluation(MODEL + ', rare:', label_test[test_rare_index],
               pred[test_rare_index])

    # t_time = time.time()
    # clf = RandomForestClassifier(n_estimators=10)
    # clf.fit(feat_train, label_train)
    # print('train.', time.time() - t_time)
    # t_time = time.time()
    # output_test = clf.predict(feat_test)
    # print('predict.', time.time() - t_time)
    # print()
    # evaluation('Random Forest', label_test, output_test)
    #
    # t_time = time.time()
    # clf = NearestCentroid()
    # clf.fit(feat_train, label_train)
    # print('train.', time.time() - t_time)
    # t_time = time.time()
    # output_test = clf.predict(feat_test)
    # print('predict.', time.time() - t_time)
    # print()
    # print('baseline: Nearest Centroid')
    # evaluation('Nearest Centroid', label_test, output_test)
    #
    # t_time = time.time()
    # clf = tree.DecisionTreeClassifier()
    # clf.fit(feat_train, label_train)
    # print('train.', time.time() - t_time)
    # t_time = time.time()
    # output_test = clf.predict(feat_test)
    # print('predict.', time.time() - t_time)
    # print()
    # evaluation('Decision Tree', label_test, output_test)


if __name__ == "__main__":
    run("../data", file_date="191210", file_suffix="00")
