import numpy as np
from Model.load_dataset import load_dataset
from Model.model_multi import DiseasesPredictor
import time

node_list, node_attr, labels, adj_lists, train, test = load_dataset(
    "./data/graph_data/191210/graph-P-191210-00"
)

main_disease = node_attr[0]
rare_patient = node_attr[1]

feature_dim = 10000
feat_data = np.random.random((50000, feature_dim))
train_enc_dim = [1000, 1000, 1000, 1000]
t1= time.time()
model = DiseasesPredictor(feat_data=feat_data,
                          b_labels=rare_patient,
                          m_labels=main_disease,
                          labels=labels,
                          adj_lists=adj_lists,
                          feature_dim=feature_dim,
                          train_enc_num=1,
                          train_enc_dim=train_enc_dim,
                          train_sample_num=[5, 5, 5, 5],
                          train=train, test=test,
                          kernel='GCN',
                          topk=(1, 2, 3, 4,5,))

model.run(8000, 200, 0.3)
print(feature_dim, train_enc_dim)
print("running time:", time.time()-t1, "s")
