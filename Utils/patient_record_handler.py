import csv
from difflib import SequenceMatcher

from Utils.RARE_INFO import RareInfo


class Patient_Record_Handler:
    def __init__(self, data_path="../data"):
        self.__data_path__ = data_path
        self.cn_symptom_map = {}
        self.raw_patient_record = []
        self.chpo_map = {}
        self.cn_rare_diseases = {}
        self.cn_rare_to_orpha_id = {}
        self.target_map = RareInfo().TARGET_MAP
        self.rare_map = {}
        self.cn_rare_map = {}
        self.NON_RARE = RareInfo().NON_RARE
        self.RARE = RareInfo().RARE
        self.diseases_map = {}
        self.common_diseases = {}

        file_path_cn_records = self.__data_path__ + \
            "/patient-records/patient-records-190715.csv"
        with open(file_path_cn_records, "r", encoding="utf8") as f:
            spamreader = csv.reader(f)
            for row in spamreader:
                patient_id = [item for item in row if item[:3] == "id_"]
                patient_info = [item for item in row if item[:2] == "p_"]
                patient_symptoms = [item for item in row if item[:2] == "s_"]
                patient_diseases = [item for item in row if item[:2] == "d_"]
                self.raw_patient_record.append(
                    patient_id + patient_info + patient_symptoms + patient_diseases)

        file_path_cn_symptom_map = self.__data_path__ + \
            "/patient-records/patient-symptom-190723-UTF8.tab"
        with open(file_path_cn_symptom_map, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip("\n").split("\t")
                if len(line) > 1 and line[0] not in self.cn_symptom_map:
                    self.cn_symptom_map[line[0]] = []
                    for item in line[1:]:
                        self.cn_symptom_map[line[0]].append(item)

        file_path_chpo = self.__data_path__ + "/HPO/chpo-name.csv"
        with open(file_path_chpo, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip("\n").split("\t")
                if line[2] not in self.chpo_map:
                    self.chpo_map[line[2]] = line[0]

        # load rare diseases match table (Raw CN -> CN)
        file_path_cn_rare_diseases = self.__data_path__ + \
            "/patient-records/patient-diseases-stupid-match-190830.tab"
        with open(file_path_cn_rare_diseases, "r", encoding="utf8", newline="") as f:
            for line in csv.reader(f, dialect="excel-tab"):
                if line[0] not in self.cn_rare_diseases:
                    self.cn_rare_diseases[line[0]] = line[1]
                # if line[1] not in self.cn_rare_map:
                    # self.cn_rare_map[line[1]] = len(self.rare_map) + 1
                    # self.cn_rare_map[line[1]] = self.RARE

        # load rare diseases match table (CN -> Orpha ID)
        file_path_cn_to_orpha_id = self.__data_path__ + \
            "/patient-records/cn_to_orpha_id_191210.csv"
        with open(file_path_cn_to_orpha_id, "r", newline="") as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                if row[3] == "K" or len(row[2]) < 1:
                    continue
                self.cn_rare_to_orpha_id[row[0]] = row[2]
                # 1st step of encoder: add cn rare diseases with orpha id
                if row[2] not in self.diseases_map:
                    self.diseases_map[row[2]] = len(self.diseases_map) + 1
                self.rare_map[row[2]] = self.diseases_map[row[2]]

        # 2nd step of encoder: add cn common diseases (sorted by frequency)
        file_path_cn_common_diseases = self.__data_path__ + \
            "/patient-records/patient-diseases-191210-clean.csv"
        with open(file_path_cn_common_diseases, "r", newline="") as f:
            d_flag = ""
            for line in f:
                line = line.strip("\n")
                if len(line) == 0:
                    d_flag = ""
                else:
                    line = line.split(",")
                    for item in line[1:]:
                        if item not in self.cn_rare_to_orpha_id and line[0] not in self.cn_rare_to_orpha_id:
                            if line[0] not in self.diseases_map:
                                self.diseases_map[line[0]] = len(
                                    self.diseases_map) + 1
                            if line[0] not in self.common_diseases:
                                self.common_diseases[line[0]] = [item]
                            else:
                                self.common_diseases[line[0]].append(item)
                            if len(d_flag) == 0:
                                d_flag = line[0]
                            else:
                                self.common_diseases[d_flag].append(item)

        # map chinese rare diseases in record to ORPHA number

    def cn_symptom_handler(self, symptoms):
        cn_name = []
        en_name = []
        for item in symptoms:
            if item in self.cn_symptom_map:
                cn_name += self.cn_symptom_map[item]
        for item in cn_name:
            if item[:3] == "HP:":
                en_name.append(item)
            elif item in self.chpo_map:
                en_name.append(self.chpo_map[item])
        # print(symptoms, cn_name, en_name)
        return en_name, cn_name

    def cn_disease_handler(self, diseases):
        rare_flag = 0
        output_diseases = set()
        main_type = 0
        for raw_disease in diseases:
            if raw_disease in self.cn_rare_diseases:
                cn_disease = self.cn_rare_diseases[raw_disease]
                if cn_disease in self.cn_rare_to_orpha_id:
                    orpha_id = self.cn_rare_to_orpha_id[cn_disease]
                    if orpha_id in self.rare_map:
                        output_diseases.add(orpha_id)
                        rare_flag = 1
                        if main_type == 0:
                            main_type = self.rare_map[orpha_id]

            for key in self.common_diseases:
                if raw_disease in self.common_diseases[key]:
                    output_diseases.add(key)
                    if main_type == 0:
                        main_type = self.diseases_map[key]

        return output_diseases, rare_flag, main_type

    def cn_record_handler(self):
        patient_records = []
        for record in self.raw_patient_record:
            raw_infos = [item[2:] for item in record if item[:2] == "p_"]
            raw_symptoms = [item[2:] for item in record if item[:2] == "s_"]
            raw_diseases = [item[2:] for item in record if item[:2] == "d_"]

            hpo_symptoms, chpo_symptoms = self.cn_symptom_handler(raw_symptoms)
            # rare_d = self.cn_disease_handler(raw_diseases)
            diseases_set, is_rare, main_type = self.cn_disease_handler(
                raw_diseases)

            if len(raw_infos) > 0 and len(hpo_symptoms) > 0 and len(diseases_set) > 0:
                patient_records.append({"id": record[0],
                                        "Profile": raw_infos,
                                        "Symptom": hpo_symptoms,
                                        "Symptom_CN": raw_symptoms,
                                        "Disease_CN": raw_diseases,
                                        "Rare_Type": is_rare,
                                        "Diseases": diseases_set,
                                        "Main_Type": main_type})
        return patient_records

    # ------------------------------------------
    #       codes below for data cleaning
    # ------------------------------------------

    def __test_01__(self):
        for key in self.cn_rare_diseases:
            print(key, self.cn_rare_diseases[key])

    def __extract_diseases__(self, file_date="191210"):
        raw_diseases = {}
        for record in self.raw_patient_record:
            for item in record:
                if item[:2] == "d_":
                    if item[2:] not in raw_diseases:
                        raw_diseases[item[2:]] = 1
                    else:
                        raw_diseases[item[2:]] += 1
        freq_diseases = [(key, raw_diseases[key]) for key in raw_diseases]
        freq_diseases.sort(key=lambda x: x[1], reverse=True)
        with open(self.__data_path__ + "/temp/patient-diseases-" + file_date + ".csv", "w", encoding="utf8") as f:
            for key, value in freq_diseases:
                f.write(key + "\n")
                print(key, value)
        with open(self.__data_path__ + "/temp/patient-diseases-with-f-" + file_date + ".csv", "w", encoding="utf8") as f:
            for key, value in freq_diseases:
                f.write(key + " " + str(value) + "\n")
                # print(key, value)
        print(len(freq_diseases))

    def __stupid_diseases_mathcer__(self):
        patient_diseases = []
        with open(self.__data_path__ + "/patient-records/patient-diseases-190724-UTF8.csv", "r", encoding="utf8") as f:
            for line in f:
                patient_diseases.append(line.strip("\n"))

        orpha_cn = []
        with open(self.__data_path__ + "/orphanet/orphanet-cn2en-190517.tab", "r", encoding="utf8") as f:
            for line in f:
                line = line.strip("\n").split("\t")
                orpha_cn.append(line[0])

        print("diseases in EHR:", len(patient_diseases))
        print("diseases in CN Orpah:", len(orpha_cn))

        def similarity(a, b):
            return SequenceMatcher(None, a, b).ratio()

        match_result = []
        for index, p_disease in enumerate(patient_diseases):
            result = [p_disease]
            s_result = []
            for item in orpha_cn:
                s = similarity(p_disease, item)
                s_result.append((item, s))
            s_result.sort(key=lambda x: x[1], reverse=True)
            result += [item[0] for item in s_result[:10]]
            match_result.append(result)
            if index % 100 == 0 and index > 0:
                print("index %d done." % index)

        with open(self.__data_path__ + "/temp/patient-diseases-stupid-match-190724.csv", "w", encoding="utf8") as f:
            for line in match_result:
                f.write("\t".join(line) + "\n")
        print("result:", len(match_result))

    def __list_rare__(self):
        patient_records = []
        rare_diseases = {}
        for record in self.raw_patient_record:
            raw_infos = [item[2:] for item in record if item[:2] == "p_"]
            raw_symptoms = [item[2:] for item in record if item[:2] == "s_"]
            raw_diseases = [item[2:] for item in record if item[:2] == "d_"]

            hpo_symptoms, chpo_symptoms = self.cn_symptom_handler(raw_symptoms)

            rare_d = self.cn_disease_handler(raw_diseases)

            if len(rare_d) > 1:
                print("patient:%s," % (record[0]), rare_d)

            if len(raw_infos) > 0 and len(hpo_symptoms) > 0:
                patient_records.append({"id": record[0],
                                        "Profile": raw_infos,
                                        "Symptom": hpo_symptoms,
                                        "Symptom_CN": raw_symptoms,
                                        "Disease": [],
                                        "Disease_CN": raw_diseases,
                                        "Rare": 0})
        return patient_records, rare_diseases

    def diseases_encoder(self):
        print(len(self.cn_rare_to_orpha_id))

    def __list_common__(self):
        for key in self.common_diseases:
            print(self.diseases_map[key], key, self.common_diseases[key])
