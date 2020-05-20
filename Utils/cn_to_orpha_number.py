import csv

from Utils.build_graph import MedGraph
from Utils.patient_record_handler import Patient_Record_Handler


def load_clean_orphanet(data_path="../data"):
    file_path = data_path + "/orphanet-cn2en-clean.tab"
    _ = []
    o_cn2en = {}
    o_en2cn = {}
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            cn_name, en_name = line.strip("\n").split("\t")
            o_cn2en[cn_name] = en_name
            o_en2cn[en_name] = cn_name
    return _, o_cn2en, o_en2cn


def print_cn_to_en():
    data_path = "../data"
    f_freq_cn_path = data_path + "/rare diseases table.csv"

    _, o_cn2en, o_en2cn = load_clean_orphanet()

    with open(f_freq_cn_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip("\n").split(",")
            if line[0] in o_cn2en:
                print("%s\t%s\t%s\t" % (line[0], o_cn2en[line[0]], line[1]))
            else:
                print("%s\t%s\t%s\t" % (line[0], "!!!", line[1]))


def print_cn_to_id():
    data_path = "../data"
    f_cn_to_id_path = data_path + "/patient-records/cn_to_orpha_id.csv"
    with open(f_cn_to_id_path, "r", newline="") as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[3] == "K" or len(row[2]) < 1:
                continue
            print(row[:3])


def find_rare_in_MedDataBase():
    target_map = {"ORPHA:3389": 2,
                  "ORPHA:761": 3,
                  "ORPHA:94058": 4}

    patient_handler = Patient_Record_Handler()
    graph_builder = MedGraph(data_dir="../data")
    _, map_to_orpha = graph_builder.__merge_diseases__()
    # db_node_map = graph_builder.build_graph(write_file=False)

    data_path = "../data"
    f_cn_to_id_path = data_path + "/patient-records/cn_to_orpha_id.csv"

    cn_rare_disease_to_orpha_id = {}
    cn_rare_disease_to_orpha_id_list = []
    orpha_id_dict = {}

    with open(f_cn_to_id_path, "r", newline="") as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[3] == "K" or len(row[2]) < 1:
                continue

            if row[2] in target_map:
                print("+ %s %s" % (row[0], row[2]))
            cn_rare_disease_to_orpha_id[row[0]] = row[2]
            cn_rare_disease_to_orpha_id_list.append(row[2])
            if row[2] not in orpha_id_dict:
                orpha_id_dict[row[2]] = len(orpha_id_dict) + 1

    for key in patient_handler.cn_rare_diseases:
        cn_disease = patient_handler.cn_rare_diseases[key]
        if cn_disease in cn_rare_disease_to_orpha_id:
            orpha_id = cn_rare_disease_to_orpha_id[cn_disease]
            if orpha_id in target_map:
                print("+ %s %s %s %d" % (key, cn_disease, orpha_id, target_map[orpha_id]))
            else:
                print("- %s %s %s" % (key, cn_disease, orpha_id,))
    print("---" * 25)
    for key in cn_rare_disease_to_orpha_id_list:
        print("\"%s\": %d, " % (key, orpha_id_dict[key]))


find_rare_in_MedDataBase()
