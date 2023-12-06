# -------------------- Imports -------------------- #

import json
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
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

        # Create a Data object for each transcription
        data = Data(x=X, edge_index=edge_index.t(), y=labels)
        
        # Append the Data object to the lists
        X_list.append(data.x)
        edge_index_list.append(data.edge_index)
        labels_list.append(data.y)

        # ------- Building dataset ------- #

validation_percentage = 0.2
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


# Model definition (GCN) #
input_size = 384
intermediate_size = 384
dropout = 0.2
final_size = 1
threshold = 0.25
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, final_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x.squeeze(1)
    
    
# Model training #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
dataset_train = [data.to(device) for data in dataset_train]

losses_train = []
losses_validation = []
model.train()
for epoch in range(100):
    number_of_predicted_1 = 0
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

        loss = torch.nn.MSELoss()(out, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    losses_train.append(total_loss)
    print('-------------------')
    print("Epoch:", epoch, "Loss:", total_loss, "TP:", tp, "/", total_1, "FP:", fp, "/", total_1, "TN:", tn, "/", total_0, "FN:", fn, "/", total_0) 
    # Calculate f1 score on validation set
    model.eval()
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
        predictions = [1 if x > threshold else 0 for x in out]
        total_1 += (target == 1).sum().item()
        total_0 += (target == 0).sum().item()
        tp += [predictions[i] == 1 and target[i] == 1 for i in range(len(predictions))].count(True)
        tn += [predictions[i] == 0 and target[i] == 0 for i in range(len(predictions))].count(True)
        fp += [predictions[i] == 1 and target[i] == 0 for i in range(len(predictions))].count(True)
        fn += [predictions[i] == 0 and target[i] == 1 for i in range(len(predictions))].count(True)
        loss = torch.nn.MSELoss()(out, target)
        total_loss += loss.item()
    losses_validation.append(total_loss)
    print("Validation Loss:", total_loss, "TP:", tp, "/", total_1, "FP:", fp, "/", total_1, "TN:", tn, "/", total_0, "FN:", fn, "/", total_0)
    if (tp+0.5*(fp+fn)) == 0:
        print("F1 score:", 0)
    else:
        print("F1 score:", tp / (tp + 0.5 * (fp + fn)))
    print('-------------------')
    model.train()

# Plotting the loss function
plt.plot(range(len(losses_validation)), losses_validation)
plt.plot(range(len(losses_train)), losses_train)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function')
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
        loss = torch.nn.MSELoss()(out, target)
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
        data = Data(x=X, edge_index=edge_index.t())
        labels = model(data)
        predictions = [1 if x > best_b else 0 for x in labels]
    test_labels[transcription_id] = predictions

with open("test_labels.json", "w") as file:
    json.dump(test_labels, file, indent=4)
    



