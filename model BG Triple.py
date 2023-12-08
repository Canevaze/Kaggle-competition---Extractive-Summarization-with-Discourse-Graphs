### _____________________________ Introduction _____________________________ ###

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

### _________________________________________________________________________ ###








### ________________________ Building the dataset ___________________________ ###

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
        embedding_text = model_embedding.encode(X)
        X = embedding_text.tolist()
        X = torch.tensor(X, dtype=torch.float)
        labels = torch.tensor(training_label, dtype=torch.long)
        edge_values = edges[:, [1]]
        individual_edge_values = []
        for x in range(0, len(X)):
            x_neighbors_indexes = [i for i in range(len(formatted_edges)) if formatted_edges[i][1] == x]
            if len(x_neighbors_indexes) == 0:
                edge_labels_weights = torch.zeros(size_dict, dtype=torch.float)
                edge_labels_weights[labels[x]] = 0
                individual_edge_values.append(edge_labels_weights)
                continue 
            x_edge_values = edge_values[x_neighbors_indexes]
            x_edge_values = np.unique(x_edge_values)

            # Transform x_edge_values using label_dict.json
            x_edge_values = [label_dict[str(value)] for value in x_edge_values]
            edge_labels_weights = torch.zeros(size_dict, dtype=torch.float)
            for i in range(size_dict):
                if i in x_edge_values:
                    edge_labels_weights[i] = x_edge_values.count(i)
                else:
                    edge_labels_weights[i] = 0
            edge_labels_weights = edge_labels_weights / torch.sum(edge_labels_weights)
            individual_edge_values.append(edge_labels_weights)
        edge_index_np = np.array([tensor.numpy() for tensor in individual_edge_values])
        edge_index = torch.tensor(edge_index_np, dtype=torch.float)


        # Create a Data object for each transcription
        data = Data(x=X, edge_index=edge_index.t(), y=labels)
        
        # Append the Data object to the lists
        X_list.append(data.x)
        edge_index_list.append(data.edge_index)
        labels_list.append(data.y)

# ---------------- Splitting the dataset ---------------- #

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

### _____________________________________________________________________________ ###






### _________________________ Building the model(s) _____________________________ ###

# --------------- Model 1 definition ------------- #

input_size = 384
intermediate_size = 384
dropout = 0.2
final_size = 1
threshold = 0.25
class Model_1(torch.nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, final_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, data):
        x = data.x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x.squeeze(1)
    
# --------------- Model 1  training ------------- #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model_1().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
criterion = torch.nn.MSELoss()
# criterion = torch.nn.BCEWithLogitsLoss()
dataset_train = [data.to(device) for data in dataset_train]

losses_train = []
losses_validation = []
model.train()
for epoch in range(100):
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

# Plotting the loss function for Model 1
plt.plot(range(len(losses_validation)), losses_validation)
plt.plot(range(len(losses_train)), losses_train)
plt.xlabel('Epoch')
plt.ylabel('Loss') 
plt.title('Loss Function for Model 1')
plt.legend(['Validation', 'Train'])
plt.show()

# --------------- Model 1 threshold determination ------------- #

B = np.linspace(0,1,100)
F1_score_1_list = []
best_b_1 = 0
max_F1_score_1 = 0
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
        F1_score_1 = 0
    else:
        F1_score_1 = tp / (tp + 0.5 * (fp + fn))
    F1_score_1_list.append(F1_score_1)

    if F1_score_1 > max_F1_score_1 :
        best_b_1 = b
        max_F1_score_1 = F1_score_1


plt.figure()
plt.plot(B,F1_score_1_list)
plt.legend(['F1 score'])
plt.xlabel('Threshold')
plt.ylabel('F1 score')
plt.title('F1 score vs Threshold for Model 1')
plt.show()
print("Best threshold for Model 1 :", best_b_1, "with F1 score :", max_F1_score_1)











# --------------- Model 2 definition ------------- #

input_size_2 = size_dict
intermediate_size_2 = size_dict
dropout_2 = 0.1
final_size_2 = 1
threshold_2 = 0.16
class Model_2(torch.nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.fc1 = torch.nn.Linear(input_size_2, final_size_2)
        self.dropout = torch.nn.Dropout(dropout_2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        #transpose edge_index
        x = data.edge_index.t()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x.squeeze(1)

# --------------- Model 2  training ------------- #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_2 = Model_2().to(device)
optimizer = torch.optim.Adam(model_2.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
criterion = torch.nn.MSELoss()

dataset_train = [data.to(device) for data in dataset_train]

losses_train = []
losses_validation = []
model_2.train()
for epoch in range(30):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    total_loss = 0
    for data in dataset_train:
        optimizer.zero_grad()
        out = model_2(data)
        target = data.y.unsqueeze(1).float()
        predictions = [1 if x > threshold_2 else 0 for x in out]
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
    model_2.eval()
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    total_loss = 0
    for data in dataset_val:
        out = model_2(data)
        target = data.y.unsqueeze(1).float()
        predictions = [1 if x > threshold_2 else 0 for x in out]
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
    model_2.train()
    scheduler.step(total_loss)

# Plotting the loss function for Model 2
plt.plot(range(len(losses_validation)), losses_validation)
plt.plot(range(len(losses_train)), losses_train)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function for Model 2')
plt.legend(['Validation', 'Train'])
plt.show()

weights = np.abs(model_2.fc1.weight.detach().numpy().flatten())

# Plotting the weights
plt.bar(range(1, size_dict+1), weights)
plt.xlabel('Weight Index')
plt.ylabel('Weight Value')
plt.title('Weights for Model 2')
plt.xticks(range(1, size_dict+1), label_dict.values())
plt.show()
print("Corresponding labels:")
for key, value in label_dict.items():
    print(key, ":", value)

    #total weight value
print("Total weight value:", np.sum(weights))

# --------------- Model 2 threshold determination ------------- #

B_2 = np.linspace(0.0,0.2,500)
F1_score_2_list = []
best_b_2 = 0
max_F1_score_2 = 0
model_2.eval()
for b in tqdm.tqdm(B_2) :
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    total_loss = 0
    for data in dataset_val:
        out = model_2(data)
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
        F1_score_2 = 0
    else:
        F1_score_2 = tp / (tp + 0.5 * (fp + fn))
    F1_score_2_list.append(F1_score_2)

    if F1_score_2 > max_F1_score_2 :
        best_b_2 = b
        max_F1_score_2 = F1_score_2

# Plotting the F1 score for Model 2
plt.figure()
plt.plot(B_2,F1_score_2_list)
plt.legend(['F1 score'])
plt.xlabel('Threshold')
plt.ylabel('F1 score')
plt.title('F1 score vs Threshold for Model 2')
plt.show()
print("Best threshold for Model 2 :", best_b_2, "with F1 score :", max_F1_score_2)











# --------------- Store model 1 and 2 ------------- #

torch.save(model.state_dict(), 'model_1.pt')
torch.save(model_2.state_dict(), 'model_2.pt')














# --------------- Model 3 definition ------------- #

input_size_3 = 2
intermediate_size_3 = 2
dropout_3 = 0.1
final_size_3 = 1
threshold_3 = 0.25

# loading  model 1 and 2
model = Model_1()
model_2 = Model_2()
model.load_state_dict(torch.load('model_1.pt'))
model_2.load_state_dict(torch.load('model_2.pt'))


model.eval()
model_2.eval()

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False
for param in model_2.parameters():
    param.requires_grad = False

class Model_3(torch.nn.Module):
    def __init__(self, model=model, model_2=best_b_1, threshold_1=best_b_2, threshold_2=threshold_2):
        super(Model_3, self).__init__()
        self.model_1 = model
        self.model_2 = model_2
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.alpha = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.layer = torch.nn.Linear(input_size_3, final_size_3)
        self.activation = torch.nn.Sigmoid()

    def forward(self, data):
        predicted_1 = self.model_1(data).view(-1,1)
        predicted_2 = self.model_2(data).view(-1,1)
        prediction_1 = [1 if x > self.threshold_1 else 0 for x in predicted_1]
        prediction_2 = [1 if x > self.threshold_2 else 0 for x in predicted_2]
        prediction_1 = torch.tensor(prediction_1, dtype=torch.float).view(-1,1)* self.alpha
        prediction_2 = torch.tensor(prediction_2, dtype=torch.float).view(-1,1)* (1-self.alpha)
        prediction_concat = torch.cat((prediction_1, prediction_2), 1)
        x = self.layer(prediction_concat)
        x = self.activation(x)
        
        return x
    
# --------------- Model 3  training ------------- #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_3 = Model_3(model=model, model_2=model_2, threshold_1=best_b_1, threshold_2=best_b_2).to(device)
optimizer = torch.optim.Adam(model_3.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
criterion = torch.nn.MSELoss()

dataset_train = [data.to(device) for data in dataset_train]

losses_train = []
losses_validation = []
model_3.train()
for epoch in range(100):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    total_loss = 0
    for data in dataset_train:
        optimizer.zero_grad()
        out = model_3(data)
        target = data.y.unsqueeze(1).float()
        predictions = [1 if x > threshold_3 else 0 for x in out]
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
    print("Epoch:", epoch, "Loss:", total_loss, "Alpha:", model_3.alpha.item(), "TP:", tp, "/", total_1, "FP:", fp, "/", total_1, "TN:", tn, "/", total_0, "FN:", fn, "/", total_0)
    # Calculate f1 score on validation set
    model_3.eval()
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    total_loss = 0
    for data in dataset_val:
        out = model_3(data)
        target = data.y.unsqueeze(1).float()
        predictions = [1 if x > threshold_3 else 0 for x in out]
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
    model_3.train()
    scheduler.step(total_loss)

# Plotting the loss function for Model 3
plt.plot(range(len(losses_validation)), losses_validation)
plt.plot(range(len(losses_train)), losses_train)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function for Model 3')
plt.legend(['Validation', 'Train'])
plt.show()

# --------------- Model 3 threshold determination ------------- #

B_3 = np.linspace(0,1,100)
F1_score_3_list = []
best_b_3 = 0
max_F1_score_3 = 0
model_3.eval()
for b in tqdm.tqdm(B_3) :
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    total_loss = 0
    for data in dataset_val:
        out = model_3(data)
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
        F1_score_3 = 0
    else:
        F1_score_3 = tp / (tp + 0.5 * (fp + fn))
    F1_score_3_list.append(F1_score_3)

    if F1_score_3 > max_F1_score_3 :
        best_b_3 = b
        max_F1_score_3 = F1_score_3

# Plotting the F1 score for Model 3
plt.figure()
plt.plot(B_3,F1_score_3_list)
plt.legend(['F1 score'])
plt.xlabel('Threshold')
plt.ylabel('F1 score')
plt.title('F1 score vs Threshold for Model 3')
plt.show()
print("Best threshold for Model 3 :", best_b_3, "with F1 score :", max_F1_score_3)







# --------------- Model 4 definition ------------- #

input_size_4 = 384 + size_dict
final_size_4 = 1
threshold_4 = 0.25
dropout = 0.2

class Model_4(torch.nn.Module):
    def __init__(self):
        super(Model_4, self).__init__()
        self.fc1 = torch.nn.Linear(input_size_4, final_size_4)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, data):
        x = data.x 
        edge_index = data.edge_index.t()
        concat = torch.cat((x, edge_index), 1)
        x = self.fc1(concat)
        x = self.relu(x)
        x = self.dropout(x)
        return x.squeeze(1)
    
# --------------- Model 4  training ------------- #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_4 = Model_4().to(device)
optimizer = torch.optim.Adam(model_4.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
criterion = torch.nn.MSELoss()

dataset_train = [data.to(device) for data in dataset_train]

losses_train = []
losses_validation = []
model_4.train()
for epoch in range(100):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    total_loss = 0
    for data in dataset_train:
        optimizer.zero_grad()
        out = model_4(data)
        target = data.y.unsqueeze(1).float().view(-1)
        predictions = [1 if x > threshold_4 else 0 for x in out]
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
    model_4.eval()
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    total_loss = 0
    for data in dataset_val:
        out = model_4(data)
        target = data.y.unsqueeze(1).float().view(-1)
        predictions = [1 if x > threshold_4 else 0 for x in out]
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
    model_4.train()
    scheduler.step(total_loss)

# Plotting the loss function for Model 4
plt.plot(range(len(losses_validation)), losses_validation)
plt.plot(range(len(losses_train)), losses_train)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function for Model 4')
plt.legend(['Validation', 'Train'])
plt.show()

# --------------- Model 4 threshold determination ------------- #

B_4 = np.linspace(0,1,100)
F1_score_4_list = []
best_b_4 = 0
max_F1_score_4 = 0
model_4.eval()
for b in tqdm.tqdm(B_4) :
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    total_loss = 0
    for data in dataset_val:
        out = model_4(data)
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
        F1_score_4 = 0
    else:
        F1_score_4 = tp / (tp + 0.5 * (fp + fn))
    F1_score_4_list.append(F1_score_4)

    if F1_score_4 > max_F1_score_4 :
        best_b_4 = b
        max_F1_score_4 = F1_score_4

# Plotting the F1 score for Model 4
plt.figure()
plt.plot(B_4,F1_score_4_list)
plt.legend(['F1 score'])
plt.xlabel('Threshold')
plt.ylabel('F1 score')
plt.title('F1 score vs Threshold for Model 4')
plt.show()

print("Best threshold for Model 4 :", best_b_4, "with F1 score :", max_F1_score_4)

# --------------- Model 4 threshold determination (zoom) ------------- #

B_4_zoom = np.linspace(max(0, best_b_4-0.05), min(1, best_b_4+0.05), 100)
F1_score_4_list_zoom = []
best_b_4_zoom = 0
max_F1_score_4_zoom = 0
model_4.eval()
for b in tqdm.tqdm(B_4_zoom) :
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_1 = 0
    total_0 = 0
    total_loss = 0
    for data in dataset_val:
        out = model_4(data)
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
        F1_score_4_zoom = 0
    else:
        F1_score_4_zoom = tp / (tp + 0.5 * (fp + fn))
    F1_score_4_list_zoom.append(F1_score_4_zoom)

    if F1_score_4_zoom > max_F1_score_4_zoom :
        best_b_4_zoom = b
        max_F1_score_4_zoom = F1_score_4_zoom

#Plotting the F1 score for Model 4
plt.figure()
plt.plot(B_4_zoom,F1_score_4_list_zoom)
plt.legend(['F1 score'])
plt.xlabel('Threshold')
plt.ylabel('F1 score')
plt.title('F1 score vs Threshold for Model 4')
plt.show()
print("Best threshold for Model 4 (After Zoom) :", best_b_4_zoom, "with F1 score :", max_F1_score_4_zoom) 

# --------------- Store model 4 ------------- #

torch.save(model_4.state_dict(), 'model_4.pt')

### _____________________________________________________________________________ ###




        


















### ___________________________ Testing the model _______________________________ ###

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
model_4.eval()
for transcription_id in tqdm.tqdm(test_transcription_ids):
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
        edges = np.array(np.loadtxt(path_to_test / f"{transcription_id}.txt", dtype=str))
        formatted_edges = edges[:, [0, -1]].astype(int)
        X = [transcription[i]["text"] for i in range(len(transcription))]
        edge_index = torch.tensor(formatted_edges, dtype=torch.float)
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
        X = torch.tensor(X, dtype=torch.float)
        edge_index_np = np.array([tensor.numpy() for tensor in individual_edge_values])
        edge_index = torch.tensor(edge_index_np, dtype=torch.float)
        data = Data(x=X, edge_index=edge_index.t())
        labels = model_4(data)
        predictions = [1 if x > best_b_4_zoom else 0 for x in labels]
    test_labels[transcription_id] = predictions

with open("test_labels.json", "w") as file:
    json.dump(test_labels, file, indent=4)
    



