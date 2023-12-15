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
        edge_index = torch.tensor(formatted_edges, dtype=torch.long)
        X = []

        for i in range(len(transcription)):
            embedding_text = model.encode(transcription[i]["text"])
            X.append(embedding_text)

        X = torch.tensor(X, dtype=torch.float)
        labels = torch.tensor(training_label, dtype=torch.long)

        # Create a Data object for each transcription
        data = Data(x=X, edge_index=edge_index.t(), y=labels)
        
        # Append the Data object to the lists
        X_list.append(data.x)
        edge_index_list.append(data.edge_index)
        labels_list.append(data.y)

        # ------- Building dataset ------- #

print("Building dataset...")
dataset = [Data(x=X, edge_index=edge_index, y=labels) for X, edge_index, labels in zip(X_list, edge_index_list, labels_list)]
print("Dataset built:", len(dataset), "Graphs.")

# ------------------- GCN Model ------------------- #
num_node_features = 384
num_classes = 2
relative_weight = 3.5
dim_intermediate = 112

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, dim_intermediate)
        self.conv2 = GCNConv(dim_intermediate, dim_intermediate)
        self.conv3 = GCNConv(dim_intermediate, dim_intermediate)
        self.conv4 = GCNConv(dim_intermediate, dim_intermediate)
        self.conv5 = GCNConv(dim_intermediate, dim_intermediate)
        self.conv6 = GCNConv(dim_intermediate, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.conv6(x, edge_index)

        return x  # Remove the log_softmax layer

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.0001)

# Convert the entire dataset to device
dataset = [data.to(device) for data in dataset]

# Training loop
model.train()
accuracies = []

for epoch in range(200):
    predicted_1 = 0
    real_1 = 0
    optimizer.zero_grad()
    total_loss = 0
    for data in dataset:
        out = model(data)
        predicted_1 += out.max(dim=1)[1].sum().item()
        real_1 += data.y.sum().item()
        loss = F.cross_entropy(out, data.y, weight=torch.tensor([1, relative_weight], dtype=torch.float).to(device))
        total_loss += loss.item()
        loss.backward()
    optimizer.step()
    accuracy = predicted_1 / real_1
    accuracies.append(accuracy)
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataset):.4f}, Predicted 1: {predicted_1}, Real 1: {real_1}')

# Plotting the accuracy graph
epochs = range(1, len(accuracies) + 1)
plt.plot(epochs, accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy (Predicted 1) vs Epoch')
plt.show()
print("Training done.")

# Evaluation
model.eval()
total_correct = 0
number_of_1 = 0
number_of_true_1 = 0
real_number_of_1 = 0
number_of_0 = 0
number_of_true_0 = 0
real_number_of_0 = 0
total_samples = 0

for data in dataset:
    _, pred = model(data).max(dim=1)
    total_correct += pred.eq(data.y).sum().item()
    total_samples += len(data.y)
    number_of_1 += pred.sum().item()
    number_of_true_1 += (pred * data.y).sum().item()
    real_number_of_1 += data.y.sum().item()
    number_of_0 += len(data.y) - pred.sum().item()
    number_of_true_0 += ((1 - pred) * (1 - data.y)).sum().item()
    real_number_of_0 += len(data.y) - data.y.sum().item()

accuracy = total_correct / total_samples

print('Accuracy on training set: {:.4f}'.format(accuracy), "(", total_correct, "/", total_samples, ")")
print('Number of true 1:', number_of_true_1,"( on ",number_of_1," predicted) and ", real_number_of_1, " real")
print('Number of true 0:', number_of_true_0,"( on ",number_of_0," predicted) and ", real_number_of_0, " real")

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
for transcription_id in test_transcription_ids:
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
        edges = np.array(np.loadtxt(path_to_test / f"{transcription_id}.txt", dtype=str))
        formatted_edges = edges[:, [0, -1]].astype(int)
        edge_index = torch.tensor(formatted_edges, dtype=torch.long)
        X = []

        for i in range(len(transcription)):
            embedding_text = emb.get_sentence_embedding(transcription[i]["text"], False)
            X.append(embedding_text)

        X = torch.tensor(X, dtype=torch.float)

        # Create a Data object for each transcription
        data = Data(x=X, edge_index=edge_index.t())
        
        Y_test = model(data)
        Y_test = Y_test.max(dim=1)[1]
        test_labels[transcription_id] = Y_test.tolist()

with open("test_labels.json", "w") as file:
    json.dump(test_labels, file, indent=4)
        








