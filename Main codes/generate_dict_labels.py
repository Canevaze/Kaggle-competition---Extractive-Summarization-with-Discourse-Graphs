# ------------ Script to generate dictionary of labels for each egde in the dataset ------------ #

import os
import sys
import json
import numpy as np

# ----------------- Load Training Labels -----------------#

training_label_path = 'training_labels.json'
print(os.path.dirname(os.path.abspath(training_label_path)))
try :
    labels_data = json.load(open(training_label_path))
    print(training_label_path + ' loaded')

except FileNotFoundError:
    print('File not found: ', training_label_path)

# ----------------- Load Training Data -----------------#

training_data_path = 'training/'
discussion_ids = []
for key in labels_data:
    discussion_ids.append(key)

# --------------- Concat all edge labels ---------------#
edge_labels = np.array([])

for discussion_id in discussion_ids:
    discussion_path = training_data_path + discussion_id + '.json'
    try :
        edges_path = training_data_path + discussion_id + '.txt'
        edges = np.array(np.loadtxt(edges_path, dtype=str))
        edge_labels = np.concatenate((edge_labels, edges[:, 1]))
        print('Edges loaded from ', edges_path)
    except FileNotFoundError:
        print('File not found: ', edges_path)
        continue
edge_labels = np.unique(edge_labels)
print('Edge labels: ', edge_labels)
print('Number of edge labels: ', len(edge_labels))

# --------------- Generate dictionary of labels ---------------#

label_dict = {}
for i in range(len(edge_labels)):
    label_dict[edge_labels[i]] = i
print(label_dict)

# --------------- Save dictionary of labels ---------------#

with open('label_dict.json', 'w') as fp:
    json.dump(label_dict, fp)
print('Saved label dictionary to label_dict.json')