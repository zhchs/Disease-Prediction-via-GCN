import numpy as np
import time
import random
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from baseline.file_reader import load_medicine

num_malacards = 8443
num_testset = 10143
num_nodes = num_malacards + num_testset

random.seed(1)

feat_data, labels, adj_lists = load_medicine(num_malacards=num_malacards, num_testset=num_testset)

val = [i for i in range(num_malacards, num_nodes)]
random.shuffle(val)
val_output = [random.randint(0, 1) for i in range(num_malacards, num_nodes)]

print('baseline: random')
print("F1:", f1_score(labels[val], val_output, average="micro"))
print("recall_score:", recall_score(labels[val], val_output))
print("precision_score:", precision_score(labels[val], val_output))
print("roc_auc_score:", roc_auc_score(labels[val], val_output))
