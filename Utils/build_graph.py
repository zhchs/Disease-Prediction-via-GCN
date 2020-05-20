import csv
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import random

from Extraction.MedDatabases import MedDatabases
from Extraction.orphanet import extract_orphanet
from Utils.patient_record_handler import Patient_Record_Handler


class MedGraph:
    def __init__(self, data_dir="../data", debug=False):
        self.med_db = MedDatabases(path=data_dir)
        self.OMIM, self.ORPHA, self.DECIPHER = self.med_db.extract_from_hpo_annotation()
        self.MAP_ORPHA_TO_OMIM, self.MAP_OMIM_TO_ORPHA = extract_orphanet(
            filepath=data_dir)
        self.__data_dir__ = data_dir
        self.__debug = debug

        self.patient_record_handler = Patient_Record_Handler(
            data_path=self.__data_dir__)

        self.RARE = self.patient_record_handler.RARE
        self.NON_RARE = self.patient_record_handler.NON_RARE
        self.diseases_map = self.patient_record_handler.diseases_map

    def __merge_diseases__(self):
        # count object number
        count = 0
        count_E = 0
        count_BTNT = 0
        count_E_match = 0
        count_BTNT_match = 0

        # init map relations
        map_to_orpha = {}
        # objects' labels
        object_label = {}

        # objects in ORPHA are all RARE DISEASES
        for key in self.ORPHA:
            object_label[key] = self.RARE
            if key in self.patient_record_handler.diseases_map:
                object_label[key] = self.patient_record_handler.diseases_map[key]

        for key in self.OMIM:
            # if the object(item) could map to ORPHA
            if key in self.MAP_OMIM_TO_ORPHA:
                # init
                item_count = 0
                max_map = "BTNT"
                object_label[key] = self.NON_RARE  # default take as NON-RARE

                if self.__debug:
                    print("===" * 25)
                    print(key)

                # check MAP RELATIONS
                for item in self.MAP_OMIM_TO_ORPHA[key]:
                    item_count += 1

                    if self.__debug:
                        print("---" * 25)
                        print(item_count, "<-", item)
                        print(item_count, "  ",
                              self.MAP_OMIM_TO_ORPHA[key][item]["NAME"])
                        print(item_count, "  ",
                              self.MAP_OMIM_TO_ORPHA[key][item]["TYPE"])
                        print(item_count, "  ",
                              self.MAP_OMIM_TO_ORPHA[key][item]["RELATION_TYPE"])
                        print(
                            item_count, "  ", self.MAP_OMIM_TO_ORPHA[key][item]["RELATION_NUMBER"])

                    """
                    * The relation is from #ORPHA# to #OMIM#
                    * TYPE:     ORPHA -> OMIM
                    * E:        ORPHA:377807
                    * NTBT:     ORPHA:377808
                    * BTNT:     ORPHA:377809
                    """
                    if self.MAP_OMIM_TO_ORPHA[key][item]["RELATION_NUMBER"] == "ORPHA:377807":  # E
                        # the object can map to ORPHA Exactly
                        object_label[key] = self.RARE
                        if item in self.patient_record_handler.diseases_map:
                            object_label[key] = self.patient_record_handler.diseases_map[item]
                        # if mapped object also in the annotation
                        if item in self.ORPHA:
                            if key not in map_to_orpha:
                                map_to_orpha[key] = []
                            map_to_orpha[key].append(item)  # add map relation
                            count_E_match += 1

                        # for debug, and count number
                        max_map = "E"
                        count_E += 1

                    # BTNT
                    elif self.MAP_OMIM_TO_ORPHA[key][item]["RELATION_NUMBER"] == "ORPHA:377809":
                        # broader #ORPHA# term maps to a narrower #OMIM# term.
                        # we take this #OMIM# term as RARE
                        object_label[key] = self.RARE
                        if item in self.patient_record_handler.diseases_map:
                            object_label[key] = self.patient_record_handler.diseases_map[item]
                        # for debug, and count number
                        if max_map == "NTBT":
                            max_map = "BTNT"
                        count_BTNT += 1
                        if item in self.ORPHA:
                            count_BTNT_match += 1
                count += 1  # count number
            else:
                object_label[key] = self.NON_RARE

        if self.__debug:
            print("===" * 25)
            print("COUNT:", count)
            print("E:", count_E)
            print("BTNT (Orpha -> OMIM):", count_BTNT)
            print("Match E:", count_E_match)
            print("Match BTNT (Orpha -> OMIM):", count_BTNT_match)
            print()
            print("OMIM COUNT:", len(self.OMIM))
            print("ORPHA COUNT:", len(self.ORPHA))
            print("LABELS COUNT:", len(object_label))
            print("MAPPED COUNT:", len(map_to_orpha))
            print()
            for key in map_to_orpha:
                print(key, map_to_orpha[key])
            print()
        return object_label, map_to_orpha

    def build_graph_hpo(self, file_date="190726", file_suffix="00"):
        file_name = self.__data_dir__ + "/graph_data/" + \
            file_date + "/graph-" + file_date + "-" + file_suffix
        object_label, map_to_orpha = self.__merge_diseases__()
        list_object_label = []  # supposed to be 9150 objects, 7201 rare
        node_map = {}
        list_hpo = []

        G = nx.Graph()

        file_path = self.__data_dir__ + "/HPO/phenotype_annotation.tab"
        with open(file_path) as tsv:
            """
                *       Column	Content	Required	Example
                *   0	DB	required	OMIM, ORPHA, DECIPHER, MONDO
                *   1	DB_Object_ID	required	154700
                *   2	DB_Name	required	Achondrogenesis, type IB
                *   3	Qualifier	optional	NOT
                *   4	HPO_ID	required	HP:0002487
                *   5	DB_Reference	required	OMIM:154700 or PMID:15517394
                *   6	Evidence_Code	required	IEA
                *   7	Onset modifier	optional	HP:0003577
                *   8	Frequency	optional	HP:0003577 or 12/45 or 22%
                *   9	Sex	optional	MALE or FEMALE
                *   10	Modifier	optional	HP:0025257(“;”-separated list)
                *   11	Aspect	required	P, C, or I
                *   12	Date_Created	required	YYYY-MM-DD
                *   13	Assigned_By	required	HPO
            """
            for line in csv.reader(tsv, dialect="excel-tab"):
                db_source = line[0]
                db_object_id = line[0] + ":" + line[1]
                db_object_name = line[2]
                hpo_id = line[4]
                map_object_id = db_object_id
                if hpo_id not in list_hpo:
                    list_hpo.append(hpo_id)

                if db_source == "OMIM" or db_source == "ORPHA":
                    if db_object_id in map_to_orpha:
                        map_object_id = map_to_orpha[db_object_id][0]
                    if map_object_id not in node_map:
                        list_object_label.append(
                            (map_object_id, object_label[db_object_id]))
                        node_map[map_object_id] = len(node_map)
                    G.add_edge(map_object_id, hpo_id)

                if self.__debug:
                    pass
                    # print(db_object_id, hpo_id, db_object_name, map_object_id)
                    # if db_source == "OMIM":
                    #     print(db_object_id, hpo_id, map_object_id, "R:" + str(object_label[db_object_id]))
                    # elif db_source == "ORPHA":
                    #     print(db_object_id, hpo_id, map_object_id, "R:" + str(object_label[db_object_id]))
                    # else:
                    #     print(db_object_id, hpo_id, map_object_id)
        if self.__debug:
            print("object labels:", len(list_object_label))
            print("object map:", len(node_map))

        for hpo_id in list_hpo:
            node_map[hpo_id] = len(node_map)

        if self.__debug:
            print("node map:", len(node_map))
            print("nodes in G:", G.number_of_nodes())

        # write to files
        node_file = open(file_name + ".node", "w", encoding="utf8")
        edge_file = open(file_name + ".edge", "w", encoding="utf8")
        for object_id, label in list_object_label:
            node_file.write(
                str(node_map[object_id]) + "\t" + str(label) + "\n")
            for neighbor in G.neighbors(object_id):
                edge_file.write(
                    str(node_map[object_id]) + "\t" + str(node_map[neighbor]) + "\n")
        node_file.close()
        edge_file.close()

    def build_graph(self, file_date="190830", file_suffix="00", write_file=True):
        object_label, map_to_orpha = self.__merge_diseases__()

        patient_records = self.patient_record_handler.cn_record_handler()

        list_object_label = []  # supposed to be 9150 objects, 7201 rare
        # list of Medical Concepts (diseases in HPO or Other Sources)
        list_concept = []

        list_patient_label = []  # does a patient have any rare diseases?
        list_patient = []  # patient list in EHR

        list_hpo = []  # list of HPO Terms

        node_map = {}
        medical_graph = nx.Graph()  # medical concepts graph
        patient_graph = nx.Graph()  # patient records graph

        file_path = self.__data_dir__ + "/HPO/phenotype_annotation.tab"
        rare_count = 0
        with open(file_path) as tsv:
            for line in csv.reader(tsv, dialect="excel-tab"):
                db_source = line[0]
                db_object_id = line[0] + ":" + line[1]
                db_object_name = line[2]
                hpo_id = line[4]
                map_object_id = db_object_id
                if hpo_id not in list_hpo:
                    list_hpo.append(hpo_id)

                if db_source == "OMIM" or db_source == "ORPHA":
                    if db_object_id in map_to_orpha:
                        map_object_id = map_to_orpha[db_object_id][0]

                    # 191210: all the diseases in medical graph are RARE DISEASES !!!
                    if object_label[db_object_id] > 0:
                        if map_object_id not in node_map:
                            node_map[map_object_id] = len(node_map)
                            list_object_label.append(
                                [map_object_id, str(object_label[db_object_id])])
                            rare_count += 1 if object_label[db_object_id] >= self.RARE else 0
                        medical_graph.add_edge(map_object_id, hpo_id)

                    # if map_object_id not in node_map:
                    #     node_map[map_object_id] = len(node_map)
                    #     list_object_label.append(
                    #         (map_object_id, object_label[db_object_id]))
                    #     rare_count += 1 if object_label[db_object_id] > 0 else 0
                    # medical_graph.add_edge(map_object_id, hpo_id)

        print("medical concepts number: %d, rare: %d " %
              (len(list_object_label), rare_count))

        db_node_map = node_map.copy()

        rare_count = 0
        for patient in patient_records:
            patient_id = patient["id"]
            patient_phenotypes = patient["Symptom"]
            patient_type = patient["Rare_Type"]
            diseases_list = list(patient["Diseases"])
            main_type = patient["Main_Type"]
            rare_count += 1 if patient_type > 0 else 0

            # patients who not have rare diseases and the number of symptoms less than 3
            if patient_type == 0 and len(patient_phenotypes) < 3:
                print("Skip", patient_id, "non-rare, less than 3 symptoms")
                continue

            if main_type == 0:
                print("Skip", patient_id, "no disease id")
                continue

            if patient_id not in node_map:
                node_map[patient_id] = len(node_map)
                list_patient_label.append(
                    [patient_id] + diseases_list + [str(main_type), str(patient_type)])
            for phenotype in patient_phenotypes:
                if phenotype not in list_hpo:
                    list_hpo.append(phenotype)
                    # print(phenotype)
                patient_graph.add_edge(patient_id, phenotype)
        print("patient number: %d, rare: %d " %
              (len(list_patient_label), rare_count))

        def write_graph(G, label_list, graph_name):
            node_file = open(graph_name + ".node", "w", encoding="utf8")
            edge_file = open(graph_name + ".edge", "w", encoding="utf8")
            for line in label_list:
                object_id = line[0]
                print(object_id, line[1:])
                node_file.write("\t".join(line) + "\n")
                for neighbor in G.neighbors(object_id):
                    edge_file.write(object_id + "\t" + neighbor + "\n")
            node_file.close()
            edge_file.close()

        file_path = self.__data_dir__ + "/graph_data/" + file_date
        file_name_medical_graph = file_path + "/graph-M-" + file_date + "-" + file_suffix
        file_name_patient_graph = file_path + "/graph-P-" + file_date + "-" + file_suffix

        file_diseases_code = file_path + "/diseases-map-" + file_date + ".txt"
        with open(file_diseases_code, "w", encoding="utf8") as f:
            for key in self.diseases_map:
                f.write(str(self.diseases_map[key])+"\t"+key+"\n")

        if write_file:
            write_graph(medical_graph, list_object_label,
                        file_name_medical_graph)
            write_graph(patient_graph, list_patient_label,
                        file_name_patient_graph)

        return node_map

    def generate_dataset(self, file_date, file_suffix, out_date, out_suffix, match_hpo=False):
        np.random.seed(1)
        random.seed(1)
        file_path = self.__data_dir__ + "/graph_data/" + file_date
        file_name_medical_graph = file_path + "/graph-M-" + file_date + "-" + file_suffix
        file_name_patient_graph = file_path + "/graph-P-" + file_date + "-" + file_suffix

        def load_graph(file_name):
            nodes_list = []
            nodes_label = {}
            adj_lists = defaultdict(set)
            hpo_nodes = set()
            with open(file_name + ".node", "r", encoding="utf8") as node_file:
                for line in node_file:
                    line = line.strip("\n").split("\t")
                    if line[0] not in nodes_list:
                        nodes_list.append(line[0])
                    nodes_label[line[0]] = line[1]
            with open(file_name + ".edge", "r", encoding="utf8") as edge_file:
                for line in edge_file:
                    line = line.strip("\n").split("\t")
                    adj_lists[line[0]].add(line[1])
                    hpo_nodes.add(line[1])

            return nodes_list, nodes_label, adj_lists, hpo_nodes

        def nodes_index(length, r=0.7):
            index = [i for i in range(length)]
            random.shuffle(index)
            pos = int(length * r)
            train = index[:pos]
            test = index[pos:]
            return train, test

        def write_index(file_name, node_list, node_index):
            with open(file_name + ".index", "w", encoding="utf8") as f:
                for i in node_index:
                    f.write(node_list[i] + "\n")

        output_name_medical = file_path + "/graph-M-" + out_date + "-" + out_suffix + "-"
        output_name_patient = file_path + "/graph-P-" + out_date + "-" + out_suffix + "-"

        nodes_list_patient, nodes_label_patient, adj_lists_patient, hpo_nodes_patient = load_graph(
            file_name_patient_graph)
        train_patient, test_patient = nodes_index(len(nodes_list_patient))
        write_index(output_name_patient + "transductive-train",
                    nodes_list_patient, train_patient)
        write_index(output_name_patient + "transductive-test",
                    nodes_list_patient, test_patient)
        # write_file(output_name_patient + "transductive-train",
        #            nodes_list_patient, nodes_label_patient, adj_lists_patient, train_patient)
        # write_file(output_name_patient + "transductive-test",
        #            nodes_list_patient, nodes_label_patient, adj_lists_patient, test_patient)

        nodes_list_medical, nodes_label_medical, adj_lists_medical, _ = load_graph(
            file_name_medical_graph)

        if match_hpo:
            nodes_list_medical_match = [node for node in nodes_list_medical if
                                        len(adj_lists_medical[node] & hpo_nodes_patient) > 0]
            train_medical, test_medical = nodes_index(
                len(nodes_list_medical_match))
        else:
            train_medical, test_medical = nodes_index(len(nodes_list_medical))

        write_index(output_name_medical + "transductive-train",
                    nodes_list_medical, train_medical)
        write_index(output_name_medical + "transductive-test",
                    nodes_list_medical, test_medical)
        # write_file(output_name_medical + "transductive-train",
        #            nodes_list_medical, nodes_label_medical, adj_lists_medical, train_medical)
        # write_file(output_name_medical + "transductive-test",
        #            nodes_list_medical, nodes_label_medical, adj_lists_medical, test_medical)
