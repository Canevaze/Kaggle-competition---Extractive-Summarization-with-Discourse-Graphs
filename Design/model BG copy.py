# -------------------- Imports -------------------- #

import json
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import tqdm



# --------------------- Paths --------------------- #

path_to_training = Path("training")
path_to_test = Path("test")

# ------------------- Embedding ------------------- #

model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
model_embedding.encode("Test sentence")

# --------------------- Data ---------------------- #

print("Loading data...")
        # --- Loading Training Labels --- #

try :
    training_labels = json.load(open("training_labels.json", "r"))
except FileNotFoundError:
    print('File not found: training_labels.json')

        # ---- Loading Training Data ---- #

try :
    training_transcription_ids = json.load(open("training_labels.json", "r")).keys()
    print("Training on ", len(training_transcription_ids), " transcriptions.")
except FileNotFoundError:
    print('File not found: training_labels.json')
    

print("Data loaded.")

        # ---- Loading Label Dict ---- #

label_dict = json.load(open("label_dict.json", "r"))
size_dict = len(label_dict)
print("Using label_dict.json of size", size_dict, "to transform x_edge_values...")

        # ------- Building tensor ------- #


print("Building tensors...")
X_list = []
edge_index_list = []
labels_list = []

for transcription_id in tqdm.tqdm(training_transcription_ids, desc="Processing transcriptions"):
    training_label = training_labels[transcription_id]
    with open(path_to_training / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
        edges = np.array(np.loadtxt(path_to_training / f"{transcription_id}.txt", dtype=str))
        formatted_edges = edges[:, [0, -1]].astype(int)
        X = [transcription[i]["text"] for i in range(len(transcription))]
        edge_index = torch.tensor(formatted_edges, dtype=torch.long)
        embedding_text = model_embedding.encode(X)
        X = embedding_text.tolist()
        X = torch.tensor(X, dtype=torch.float)
        labels = torch.tensor(training_label, dtype=torch.long)
        edge_values = edges[:, [1]]
        individual_edge_values = []
        for x in range(0, len(X)):
            x_neighbors_indexes = [i for i in range(len(formatted_edges)) if formatted_edges[i][0] == x or formatted_edges[i][1] == x]
            x_edge_values = edge_values[x_neighbors_indexes]
            x_edge_values = np.unique(x_edge_values)

            # Transform x_edge_values using label_dict.json
            x_edge_values = [label_dict[str(value)] for value in x_edge_values]
            edge_labels_weights = torch.zeros(size_dict, dtype=torch.float)
            for i in range(size_dict):
                edge_labels_weights[i] = x_edge_values.count(i)
            edge_labels_weights = edge_labels_weights / torch.sum(edge_labels_weights)
            individual_edge_values.append(edge_labels_weights)
        X = torch.cat((X, torch.stack(individual_edge_values)), 1)


        # Create a Data object for each transcription
        data = Data(x=X, edge_index=edge_index.t(), y=labels)
        
        # Append the Data object to the lists
        X_list.append(data.x)
        edge_index_list.append(data.edge_index)
        labels_list.append(data.y)

        # ------- Building dataset ------- #

validation_percentage = 0.25
validation_size = int(len(X_list) * validation_percentage)

print("Splitting dataset...")
X_list_train = X_list[validation_size:]
X_list_val = X_list[:validation_size]
edge_index_list_train = edge_index_list[validation_size:]
edge_index_list_val = edge_index_list[:validation_size]
labels_list_train = labels_list[validation_size:]
labels_list_val = labels_list[:validation_size]
print("Dataset splitted in train and validation sets of size", len(X_list_train), "and", len(X_list_val), "respectively.")

print("Building dataset...")
dataset_train = [Data(x=X, edge_index=edge_index, y=labels) for X, edge_index, labels in zip(X_list_train, edge_index_list_train, labels_list_train)]
dataset_val = [Data(x=X, edge_index=edge_index, y=labels) for X, edge_index, labels in zip(X_list_val, edge_index_list_val, labels_list_val)]
print("Dataset built:", len(dataset_train) + len(dataset_val), "Graphs (", len(dataset_train), "for training and", len(dataset_val), "for validation).")

# # Select nodes with most connections and print the number of maximum connections
# max_connections = 0
# for data in tqdm.tqdm(dataset_train):
#     edges = data.edge_index
#     for i in range (len(data.x)):
#         connections = [1 if x == i else 0 for x in edges[0]].count(1) + [1 if x == i else 0 for x in edges[1]].count(1)
#         if connections > max_connections:
#             max_connections = connections
# print("Maximum number of connections:", max_connections)

# Model definition #
input_size = 384 + size_dict
intermediate_size = 384
dropout = 0.2
final_size = 1
threshold = 0.29
class Model(torch.nn.Module):
    def __init__(self, input_size=input_size, intermediate_size=intermediate_size, dropout=dropout, final_size=final_size):
        super(Model, self).__init__()
        self.fc1_first = torch.nn.Linear(input_size - size_dict, final_size)
        self.fc2_last = torch.nn.Linear(size_dict, final_size)
        self.fc3_combined = torch.nn.Linear(final_size * 2, final_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        # # Initialize weights so that the model predicts 0.5 for all inputs
        # self.fc1_first.weight.data.fill_(1)
        # self.fc1_first.bias.data.fill_(1)
        # self.fc2_last.weight.data.fill_(1)
        # self.fc2_last.bias.data.fill_(1)
        # self.fc3_combined.weight.data.fill_(1)
        # self.fc3_combined.bias.data.fill_(1)


    def forward(self, data):
        x = data.x
        x_first = x[:, :input_size - size_dict]
        x_first = self.fc1_first(x_first)
        x_first = self.dropout(x_first)
        x_first = self.relu(x_first)
        x_last = x[:, -size_dict:]
        x_last = self.fc2_last(x_last)
        x_last = self.dropout(x_last)
        x_last = self.relu(x_last)
        x_combined = torch.cat([x_first, x_last], dim=1)
        x_combined = self.fc3_combined(x_combined)
        x_combined = self.dropout(x_combined)
        x_combined = self.relu(x_combined)
        return x_combined.squeeze(1)
# Model training #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
criterion = torch.nn.MSELoss()
# criterion = torch.nn.BCEWithLogitsLoss()
dataset_train = [data.to(device) for data in dataset_train]

losses_train = []
losses_validation = []
model.train()
for epoch in range(300):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    total_loss = 0
    for data in dataset_train:
        optimizer.zero_grad()
        out = model(data)
        target = data.y.unsqueeze(1).float().view(-1)
        predictions = [1 if x > threshold else 0 for x in out]
        total_1 += (target == 1).sum().item()
        data_total_1 = (target == 1).sum().item()
        data_total_0 = (target == 0).sum().item()
        total_0 += (target == 0).sum().item()
        tp += [predictions[i] == 1 and target[i] == 1 for i in range(len(predictions))].count(True)
        tn += [predictions[i] == 0 and target[i] == 0 for i in range(len(predictions))].count(True)
        fp += [predictions[i] == 1 and target[i] == 0 for i in range(len(predictions))].count(True)
        fn += [predictions[i] == 0 and target[i] == 1 for i in range(len(predictions))].count(True)
        loss = criterion(out, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    losses_train.append(total_loss)
    print("Epoch:", epoch, "Loss:", total_loss, "TP:", tp, "/", total_1, "FP:", fp, "/", total_1, "TN:", tn, "/", total_0, "FN:", fn, "/", total_0) 
    # Calculate f1 score on validation set
    model.eval()
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    total_loss = 0
    for data in dataset_val:
        out = model(data)
        target = data.y.unsqueeze(1).float().view(-1)
        predictions = [1 if x > threshold else 0 for x in out]
        total_1 += (target == 1).sum().item()
        total_0 += (target == 0).sum().item()
        tp += [predictions[i] == 1 and target[i] == 1 for i in range(len(predictions))].count(True)
        tn += [predictions[i] == 0 and target[i] == 0 for i in range(len(predictions))].count(True)
        fp += [predictions[i] == 1 and target[i] == 0 for i in range(len(predictions))].count(True)
        fn += [predictions[i] == 0 and target[i] == 1 for i in range(len(predictions))].count(True)
        loss = criterion(out, target)
        total_loss += loss.item()
    losses_validation.append(total_loss/validation_percentage)
    print("Validation Loss:", total_loss, "TP:", tp, "/", total_1, "FP:", fp, "/", total_1, "TN:", tn, "/", total_0, "FN:", fn, "/", total_0)
    if (tp+0.5*(fp+fn)) == 0:
        print("F1 score:", 0)
    else:
        print("F1 score:", tp / (tp + 0.5 * (fp + fn)))
    print('-------------------')
    model.train()
    scheduler.step(total_loss)

# Plotting the loss function
plt.plot(range(len(losses_validation)), losses_validation)
plt.plot(range(len(losses_train)), losses_train)
plt.xlabel('Epoch')
plt.ylabel('Loss') 
plt.title('Loss Function')
plt.legend(['Validation', 'Train'])
plt.show()



# Model evaluation #
# use threshold betha to determine if the prediction is 1 or 0
B = np.linspace(0,1,100)
F1_score_list = []
best_b = 0
max_F1_score = 0
model.eval()
for b in tqdm.tqdm(B) : 

    number_of_predicted_1 = 0
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    total_loss = 0
    

    for data in dataset_val:
        out = model(data)
        target = data.y.unsqueeze(1).float().view(-1)
        predictions = [1 if x > b else 0 for x in out]
        total_1 += (target == 1).sum().item()
        total_0 += (target == 0).sum().item()
        tp += [predictions[i] == 1 and target[i] == 1 for i in range(len(predictions))].count(True)
        tn += [predictions[i] == 0 and target[i] == 0 for i in range(len(predictions))].count(True)
        fp += [predictions[i] == 1 and target[i] == 0 for i in range(len(predictions))].count(True)
        fn += [predictions[i] == 0 and target[i] == 1 for i in range(len(predictions))].count(True)
        loss = criterion(out, target)
        total_loss += loss.item()
    if (tp+0.5*(fp+fn)) == 0:
        F1_score = 0
    else:
        F1_score = tp / (tp + 0.5 * (fp + fn))
    F1_score_list.append(F1_score)

    if F1_score > max_F1_score :
        best_b = b
        max_F1_score = F1_score


plt.figure()
plt.plot(B,F1_score_list)
plt.legend(['F1 score'])
plt.xlabel('Threshold')
plt.ylabel('F1 score')
plt.title('F1 score vs Threshold')
plt.show()
print("Best threshold :", best_b, "with F1 score :", max_F1_score)


# ------------------- Testing ------------------- #

print("Testing...")
        # --- Loading Test Labels --- #

try :
    test_labels = json.load(open("test_labels.json", "r"))
except FileNotFoundError:
    print('File not found: test_labels.json')
    
        # ---- Loading Test Data ---- #


test_transcription_ids = np.array([])               # Labels
for file in path_to_test.iterdir():
    test_transcription_ids = np.append(test_transcription_ids, file.stem)
test_transcription_ids = np.unique(test_transcription_ids)


# Evaluation

test_labels = {}
model.eval()
for transcription_id in tqdm.tqdm(test_transcription_ids):
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
        edges = np.array(np.loadtxt(path_to_test / f"{transcription_id}.txt", dtype=str))
        formatted_edges = edges[:, [0, -1]].astype(int)
        X = [transcription[i]["text"] for i in range(len(transcription))]
        edge_index = torch.tensor(formatted_edges, dtype=torch.long)
        embedding_text = model_embedding.encode(X)
        X = embedding_text.tolist()
        X = torch.tensor(X, dtype=torch.float)
        edge_values = edges[:, [1]]
        individual_edge_values = []
        for x in range(0, len(X)):
            x_neighbors_indexes = [i for i in range(len(formatted_edges)) if formatted_edges[i][0] == x or formatted_edges[i][1] == x]
            x_edge_values = edge_values[x_neighbors_indexes]
            x_edge_values = np.unique(x_edge_values)

            # Transform x_edge_values using label_dict.json
            x_edge_values = [label_dict[str(value)] for value in x_edge_values]
            edge_labels_weights = torch.zeros(size_dict, dtype=torch.float)
            for i in range(size_dict):
                edge_labels_weights[i] = x_edge_values.count(i)
            edge_labels_weights = edge_labels_weights / torch.sum(edge_labels_weights)
            individual_edge_values.append(edge_labels_weights)
        X = torch.cat((X, torch.stack(individual_edge_values)), 1)
        data = Data(x=X, edge_index=edge_index.t())
        labels = model(data)
        predictions = [1 if x > best_b else 0 for x in labels]
    test_labels[transcription_id] = predictions

with open("test_labels.json", "w") as file:
    json.dump(test_labels, file, indent=4)
    



