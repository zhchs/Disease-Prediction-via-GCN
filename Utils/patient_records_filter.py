import csv


def patient_records_filter(data_path="../data", input_date_suffix="-190526.csv", output_date_suffix=""):
    file_path = data_path + "/patient-records/patient-records" + input_date_suffix
    new_file_path = data_path + "/patient-records/patient-records" + output_date_suffix
    new_csvfile = open(new_file_path, "w", encoding="utf8")
    with open(file_path, "r", encoding="utf8") as csvfile:
        spamreader = csv.reader(csvfile)
        patient_count = 0
        for row in spamreader:
            patient_symptoms = [item[2:] for item in row if item[:2] == "s_"]
            patient_diseases = [item[2:] for item in row if item[:2] == "d_"]
            if len(patient_symptoms) < 3:
                continue
            plain_text = "".join(row)
            if "孕" in plain_text or "待产" in plain_text or "胎方位" in plain_text or "妊娠" in plain_text:
                continue
            if len(patient_diseases) < 1 or len(patient_diseases) < 1:
                continue
            new_csvfile.write(",".join(row) + "\n")
            print(", ".join(row))
            patient_count += 1
    print(patient_count)
    new_csvfile.close()


def patient_records_filter_slim(data_path="../data", input_date_suffix="-full.tab",
                                output_date_suffix="-full-F-190703.tab"):
    file_path = data_path + "/patient-records/patient-records" + input_date_suffix
    new_file_path = data_path + "/patient-records/patient-records" + output_date_suffix
    new_file = open(new_file_path, "w", encoding="utf8")
    with open(file_path, "r", encoding="utf8") as f:
        patient_count = 0
        for line in f:
            plain_text = line
            if "孕" in plain_text or "待产" in plain_text or "胎方位" in plain_text or "妊娠" in plain_text:
                continue
            new_file.write(line)
            patient_count += 1
    print(patient_count)
    new_file.close()


def patient_diseases_symptoms(data_path="../data"):
    # file_path = data_path + "/patient-records/patient-records-k3-190518-clean.csv"
    file_path = data_path + "/patient-records/filtered-patient-records-190521.csv"
    diseases_set = {}
    symptoms_set = {}
    with open(file_path, "r", encoding="utf8") as csvfile:
        spamreader = csv.reader(csvfile)
        patient_count = 0
        for row in spamreader:
            patient_symptoms = [item[2:] for item in row if item[:2] == "s_"]
            patient_diseases = [item[2:] for item in row if item[:2] == "d_"]
            for item in patient_symptoms:
                if item not in symptoms_set:
                    symptoms_set[item] = 1
                else:
                    symptoms_set[item] += 1
            for item in patient_diseases:
                if item not in diseases_set:
                    diseases_set[item] = 1
                else:
                    diseases_set[item] += 1
            print(", ".join(row))
            patient_count += 1
    print(patient_count)

    diseases_freq = [(disease, diseases_set[disease]) for disease in diseases_set]
    symptoms_freq = [(symptom, symptoms_set[symptom]) for symptom in symptoms_set]

    diseases_freq.sort(key=lambda x: x[1], reverse=True)
    symptoms_freq.sort(key=lambda x: x[1], reverse=True)

    with open(data_path + "/patient-records/diseases-list-190521.txt", "w", encoding="utf-8") as f:
        for disease, freq in diseases_freq:
            f.write(disease + "\n")

    with open(data_path + "/patient-records/symptoms-list-190521.txt", "w", encoding="utf-8") as f:
        for symptom, freq in symptoms_freq:
            f.write(symptom + "\n")


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


if __name__ == "__main__":
    # patient_records_filter(data_path="../data", output_date_suffix="-k3F-190702.csv")
    patient_records_filter_slim(input_date_suffix="-new-method-190704.csv",
                                output_date_suffix="-new-method-F-190704.csv")
    # patient_diseases_symptoms("../data")
    # load_clean_orphanet()
