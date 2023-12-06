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



# --------------------- Paths --------------------- #

path_to_training = Path("training")
path_to_test = Path("test")

# ------------------- Embedding ------------------- #

model = SentenceTransformer('all-MiniLM-L6-v2')
model.encode("Test sentence")

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
        embedding_text = model.encode(X)
        X = embedding_text.tolist()
        X = torch.tensor(X, dtype=torch.float)
        labels = torch.tensor(training_label, dtype=torch.long)

        # Create a Data object for each transcription
        data = Data(x=X, edge_index=edge_index.t(), y=labels)
        
        # Append the Data object to the lists
        X_list.append(data.x)
        edge_index_list.append(data.edge_index)
        labels_list.append(data.y)


# --------------------- Augmenting data ---------------------- #

print("Augmenting data...")
X_list_augmented = []
edge_index_list_augmented = []
labels_list_augmented = []



        # ------- Building dataset ------- #

print("Building dataset...")
dataset = [Data(x=X, edge_index=edge_index, y=labels) for X, edge_index, labels in zip(X_list, edge_index_list, labels_list)]
print("Dataset built:", len(dataset), "Graphs.")


# Model definition (GCN) #

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(384, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
# Model training #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
dataset_train = [data.to(device) for data in dataset]


losses = []
model.train()
for epoch in range(200):
    number_of_predicted_1 = 0
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    for data in dataset_train:
        optimizer.zero_grad()
        out = model(data)
        predicted = torch.argmax(out, dim=1)
        correspondancy_1 = ((predicted == data.y) & (data.y == 1))
        correspondancy_0 = ((predicted == data.y) & (data.y == 0))
        total_0 += torch.sum(data.y == 0).item()
        total_1 += torch.sum(data.y == 1).item()
        tp += torch.sum(correspondancy_1).item()
        tn += torch.sum(correspondancy_0).item()    
        fn += torch.sum((predicted != data.y) & (data.y == 1)).item()
        fp += torch.sum((predicted != data.y) & (data.y == 0)).item()
        loss = F.cross_entropy(out, data.y, weight=torch.tensor([1, 10], dtype=torch.float).to(device))
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    print("Epoch:", epoch, "Loss:", loss.item(), "TP:", tp, "/", total_1, "FP:", fp, "/", total_1, "TN:", tn, "/", total_0, "FN:", fn, "/", total_0) 

# Plotting the loss function
plt.plot(range(len(losses)), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function')
plt.show()



# Model evaluation #