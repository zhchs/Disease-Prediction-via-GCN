from collections import defaultdict
from queue import Queue

import numpy as np

from Extraction.HPO import HPO


def load_diseases_map(file_path):
    diseases_map = {}
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip("\n").split("\t")
            diseases_map[line[1]] = int(line[0])
    return diseases_map


def load_graph(graph_path, hpo_data=False, add_parent=False):
    pos = graph_path.find("/data")
    data_path = graph_path[:pos +5]
    graph_date = graph_path[pos +17: pos + 23]
    graph_type = graph_path[pos + 30]
    map_path = "{}/graph_data/{}/diseases-map-{}.txt".format(
        data_path, graph_date, graph_date)
    diseases_map = load_diseases_map(map_path)

    hpo_handler = HPO(data_path)
    hpo_id, hpo_parents, hpo_alt_id = hpo_handler.extract_hpo()

    node_list = []
    node_label = {}
    node_attr = {}
    adj_lists = defaultdict(set)
    feature_map = {}
    with open(graph_path + ".node", "r", encoding="utf8") as f:
        for line in f:
            line = line.strip("\n").split("\t")
            if line[0] not in node_list:
                node_list.append(line[0])
            n_label = np.zeros(len(diseases_map))
            if graph_type == 'M':
                n_label[int(line[-1]) - 1] = 1
                main_disease = int(line[-1])
                rare_flag = 1
            if graph_type == 'P':
                n_label[list(
                    map(lambda x: diseases_map[x] - 1, line[1:-2]))] = 1
                main_disease = int(line[-2])
                rare_flag = int(line[-1])
            node_label[line[0]] = n_label
            node_attr[line[0]] = (main_disease, rare_flag)

    with open(graph_path + ".edge", "r", encoding="utf8") as f:
        for line in f:
            line = line.strip("\n").split("\t")
            hpo_node = line[1]
            if hpo_data and hpo_node in hpo_alt_id:
                hpo_node = hpo_alt_id[hpo_node]

            if not add_parent:
                adj_lists[line[0]].add(hpo_node)
                if hpo_node not in feature_map:
                    feature_map[hpo_node] = len(feature_map)
            else:
                q = Queue()
                q.put(hpo_node)
                while not q.empty():
                    q_term = q.get()
                    if q_term != "HP:0000001":
                        adj_lists[line[0]].add(q_term)
                        if q_term not in feature_map:
                            feature_map[q_term] = len(q_term)

                    if q_term in hpo_parents:
                        for t in hpo_parents[q_term]:
                            q.put(t)

    if hpo_data:
        for term in hpo_id:
            for p_id in hpo_parents["term"]:
                adj_lists[term].add(p_id)

    print("graph_path:", graph_path)
    print("graph type:", graph_type)
    print("add hpo term ?", hpo_data)
    print("add hpo parents ?", add_parent)
    print("node list length: %d, feature(adj) number: %d" %
          (len(node_list), len(feature_map)))

    return node_list, node_label, adj_lists, feature_map, diseases_map, graph_type, node_attr


def load_dataset(graph_name, hpo_data=False, add_parent=False, mix=True):
    node_list, node_label, node_adj_lists, _, diseases_map, graph_type, node_attr = load_graph(
        graph_name, add_parent=add_parent, hpo_data=hpo_data)

    medical_graph_path = graph_name.replace("-P-", "-M-")

    node_list_M, node_label_M, node_adj_lists_M, _, __, graph_type_M, node_attr_M = load_graph(
        medical_graph_path, add_parent=add_parent, hpo_data=hpo_data)

    node_map = {}

    node_num = len(node_list) + len(node_list_M) if mix else len(node_list)

    labels = np.zeros((node_num, len(diseases_map)), dtype=np.int64)
    main_disease = np.zeros((node_num, 1), dtype=np.int64)
    rare_patient = np.zeros((node_num, 1), dtype=np.int64)

    adj_lists = defaultdict(set)
    for node in node_list:
        node_map[node] = len(node_map)
        labels[node_map[node]] = node_label[node]
        main_disease[node_map[node]] = node_attr[node][0] - 1
        rare_patient[node_map[node]] = node_attr[node][1]
    if mix:
        for node in node_list_M:
            node_map[node] = len(node_map)
            labels[node_map[node]] = node_label_M[node]
            main_disease[node_map[node]] = node_attr_M[node][0] - 1
            rare_patient[node_map[node]] = node_attr_M[node][1]

    node_num = len(node_list)
    node_map_input = node_map.copy()

    for node in node_adj_lists:
        if node not in node_map:
            node_map[node] = len(node_map)
        for neighbor in node_adj_lists[node]:
            if neighbor not in node_map:
                node_map[neighbor] = len(node_map)
            n1 = node_map[node]
            n2 = node_map[neighbor]
            adj_lists[n1].add(n2)
            adj_lists[n2].add(n1)
    if mix:
        for node in node_adj_lists_M:
            if node not in node_map:
                node_map[node] = len(node_map)
            for neighbor in node_adj_lists[node]:
                if neighbor not in node_map:
                    node_map[neighbor] = len(node_map)
                n1 = node_map[node]
                n2 = node_map[neighbor]
                adj_lists[n1].add(n2)
                adj_lists[n2].add(n1)

    file_name_train = graph_name + "-transductive-train.index"
    file_name_test = graph_name + "-transductive-test.index"

    with open(file_name_train, "r", encoding="utf8") as f:
        train = [node_map[line.strip("\n")] for line in f]
    with open(file_name_test, "r", encoding="utf8") as f:
        test = [node_map[line.strip("\n")] for line in f]

    if mix:
        for node in node_list_M:
            train.append(node_map[node])

    node_attr = (main_disease, rare_patient)

    # print(train)
    # print(test)

    return node_list, node_attr, labels, adj_lists, train, test
