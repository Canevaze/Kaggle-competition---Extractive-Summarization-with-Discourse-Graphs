### _____________________________ Introduction _____________________________ ###

# -------------------- Imports -------------------- #

import json
import os
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] ='0'

# --------------------- Paths --------------------- #

path_to_training = Path("training")
path_to_test = Path("test")

# ------------------- Embedding ------------------- #

model_embedding = SentenceTransformer('all-MiniLM-L6-v2')

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
edge_attr_list = []
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
        edge_index = torch.tensor(formatted_edges, dtype=torch.long)
        list_edge_attr = []
        for i in range(len(edges)):
            edge_attr_part = np.zeros(size_dict)
            edge_attr_part[label_dict[edges[i, 1]]] = 1
            list_edge_attr.append(edge_attr_part)
        edge_attr = torch.tensor(list_edge_attr, dtype=torch.float)
        # Create a Data object for each transcription
        data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=labels)
        
        # Append the Data object to the lists
        X_list.append(data.x)
        edge_index_list.append(data.edge_index)
        labels_list.append(data.y)
        edge_attr_list.append(data.edge_attr)
print("Tensors built.")


# ------------------- percetenage of edge_attr occurence ------------------- #
edge_attr_list_predict_1 = []
edge_attr_list_predict_0 = []
for i in range(len(edge_attr_list)):
    for j in range(len(edge_attr_list[i])):
        if labels_list[i][j] == 1:
            edge_attr_list_predict_1.append(edge_attr_list[i][j].numpy())
        else :
            edge_attr_list_predict_0.append(edge_attr_list[i][j].numpy())
print(len(edge_attr_list_predict_1))
edge_attr_list_predict_1 = np.array(edge_attr_list_predict_1)
edge_attr_list_predict_1 = np.sum(edge_attr_list_predict_1,axis=0)
edge_attr_list_predict_0 = np.array(edge_attr_list_predict_0)
edge_attr_list_predict_0 = np.sum(edge_attr_list_predict_0,axis=0)
edge_attr_list_predict_1 = edge_attr_list_predict_1/np.sum(edge_attr_list_predict_1)
edge_attr_list_predict_0 = edge_attr_list_predict_0/np.sum(edge_attr_list_predict_0)
print("Percentage of edge_attr occurence:", edge_attr_list_predict_1)
plt.bar(range(len(edge_attr_list_predict_1)),edge_attr_list_predict_1)
plt.title("Percentage of edge_attr occurence when label = 1")
plt.show()
plt.bar(range(len(edge_attr_list_predict_0)),edge_attr_list_predict_0)
plt.title("Percentage of edge_attr occurence when label = 0")
plt.show()

plt.bar(range(len(edge_attr_list_predict_1)),edge_attr_list_predict_1-edge_attr_list_predict_0)

select_labels = [3,4,6,9,15]
selected_labels_size = len(select_labels)
    

# ---------------- Splitting the dataset ---------------- #

validation_percentage = 0.15
validation_size = int(len(X_list) * validation_percentage)

print("Splitting dataset...")
X_list_train = X_list[validation_size:]
X_list_val = X_list[:validation_size]
edge_index_list_train = edge_index_list[validation_size:]
edge_index_list_val = edge_index_list[:validation_size]
labels_list_train = labels_list[validation_size:]
labels_list_val = labels_list[:validation_size]
edge_attr_list_train = edge_attr_list[validation_size:]
edge_attr_list_val = edge_attr_list[:validation_size]
print("Dataset splitted in train and validation sets of size", len(X_list_train), "and", len(X_list_val), "respectively.")

print("Building dataset...")
dataset_train = [Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=labels).cuda() for X, edge_index, edge_attr, labels in zip(X_list_train, edge_index_list_train, edge_attr_list_train, labels_list_train)]
dataset_val = [Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=labels).cuda() for X, edge_index, edge_attr, labels in zip(X_list_val, edge_index_list_val, edge_attr_list_val, labels_list_val)]
print("Dataset built:", len(dataset_train) + len(dataset_val), "Graphs (", len(dataset_train), "for training and", len(dataset_val), "for validation).")

### _____________________________________________________________________________ ###









# ___________________________ Building the model ________________________________ #

# --------------------- Model --------------------- #

size_input = 384
size_hidden = 384
size_output = 1
threshold = 0.26
calculate_f1 = False

learning_rate = 0.001
weight_decay = 5e-4
epochs = 10000


class Model(torch.nn.Module):
    def __init__(self, num_features=size_input, hidden_size=size_hidden, num_classes=size_output):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = num_classes
        self.convs = [
            GATConv(self.num_features, self.hidden_size, edge_dim=size_dict).cuda()]           
        self.linear = torch.nn.Linear(self.hidden_size, self.target_size).cuda()
        self.relu = torch.nn.ReLU().to('cuda')
        self.dropout = torch.nn.Dropout(0.15).to('cuda')

    def forward(self, data):
        x, edge_index,edge_attr = data.x, data.edge_index.t(), data.edge_attr.unsqueeze(1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        x = self.linear(x)
        return self.relu(x)
    

# --------------------- Training --------------------- #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_GAT = Model()
model_GAT = model_GAT.cuda()
optimizer = torch.optim.Adam(model_GAT.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=15, min_lr=0.000001, verbose=True)
dataset_train = [data.cuda() for data in dataset_train]
losses_train = []
losses_val = []

for epoch in range(epochs):
    model_GAT.train()
    loss_train = 0
    tp_t = 0
    tn_t = 0
    fn_t = 0
    fp_t = 0
    total_1_t = 0
    total_0_t = 0
    if epoch % 100 == 0 and epoch != 0:
        calculate_f1 = False
    else:
        calculate_f1 = False
    for data in dataset_train :
        optimizer.zero_grad()
        out = model_GAT(data.cuda())
        if calculate_f1:
            target = data.y.unsqueeze(1).float().view(-1)
            predictions = [1 if x > threshold else 0 for x in out]
            total_1_t += (target == 1).sum().item()
            data_total_1_t = (target == 1).sum().item()
            data_total_0_t = (target == 0).sum().item()
            total_0_t += (target == 0).sum().item()
            tp_t += [predictions[i] == 1 and target[i] == 1 for i in range(len(predictions))].count(True)
            tn_t += [predictions[i] == 0 and target[i] == 0 for i in range(len(predictions))].count(True)
            fp_t += [predictions[i] == 1 and target[i] == 0 for i in range(len(predictions))].count(True)
            fn_t += [predictions[i] == 0 and target[i] == 1 for i in range(len(predictions))].count(True)
        loss = F.mse_loss(out, data.y.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    loss_train = loss_train / len(dataset_train)
    losses_train.append(loss_train)
    model_GAT.eval()

    loss_val = 0
    tp_v = 0
    tn_v = 0
    fn_v = 0
    fp_v = 0
    total_1_v = 0
    total_0_v = 0
    for data in dataset_val :
        out = model_GAT(data)
        if calculate_f1:
            target = data.y.unsqueeze(1).float().view(-1)
            predictions = [1 if x > threshold else 0 for x in out]
            total_1_v += (target == 1).sum().item()
            total_0_v += (target == 0).sum().item()
            tp_v += [predictions[i] == 1 and target[i] == 1 for i in range(len(predictions))].count(True)
            tn_v += [predictions[i] == 0 and target[i] == 0 for i in range(len(predictions))].count(True)
            fp_v += [predictions[i] == 1 and target[i] == 0 for i in range(len(predictions))].count(True)
            fn_v += [predictions[i] == 0 and target[i] == 1 for i in range(len(predictions))].count(True)
        loss = F.mse_loss(out, data.y.unsqueeze(1).float())
        loss_val += loss.item()
    loss_val = loss_val / len(dataset_val)
    losses_val.append(loss_val)
    print("---------------------------------")
    print("Epoch", epoch + 1, "/", epochs)
    if calculate_f1:
        if (0.5*(fp_t + fn_t) == 0):
            print("F1 score on training set:", 0)
        else :
            print("F1 score on training set:", tp_t/(tp_t + 0.5*(fp_t + fn_t)))
        if (0.5*(fp_v + fn_v) == 0):
            print("F1 score on validation set:", 0)
        else :
            print("F1 score on validation set:", tp_v/(tp_v + 0.5*(fp_v + fn_v)))
    print("Validation loss:", loss_val)

    scheduler.step(loss_val)

    # Print the losses
plt.plot(losses_train, label="Training loss")
plt.plot(losses_val, label="Validation loss")
plt.legend()
plt.show()

#Store the model_GAT
torch.save(model_GAT.state_dict(), "model_GAT.pt")


### _____________________________________________________________________________ ###

## optimize threshold

B = np.linspace(0,1,100)
F1 = []

for b in tqdm.tqdm( B, desc="Optimizing threshold"):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for data in dataset_val :
        out = model_GAT(data)
        target = data.y.unsqueeze(1).float().view(-1)
        predictions = [1 if x > b else 0 for x in out]
        tp += [predictions[i] == 1 and target[i] == 1 for i in range(len(predictions))].count(True)
        tn += [predictions[i] == 0 and target[i] == 0 for i in range(len(predictions))].count(True)
        fp += [predictions[i] == 1 and target[i] == 0 for i in range(len(predictions))].count(True)
        fn += [predictions[i] == 0 and target[i] == 1 for i in range(len(predictions))].count(True)
    if (0.5*(fp + fn) == 0):
        F1.append(0)
    else :
        F1.append(tp/(tp + 0.5*(fp + fn)))

plt.plot(B,F1)
plt.show()
F1_max_GAT = np.max(F1)
print("Best threshold:", B[np.argmax(F1)], "with F1 score:", np.max(F1))


### _____________________________________________________________________________ ###










### _____________________________ Linear Model ______________________________________ ###

# --------------------- Model --------------------- #


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
model_l = Model_1().to(device)
optimizer = torch.optim.Adam(model_l.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
criterion = torch.nn.MSELoss()
# criterion = torch.nn.BCEWithLogitsLoss()
dataset_train = [data.to(device) for data in dataset_train]

losses_train = []
losses_validation = []
model_l.train()
for epoch in range(500):

    total_loss = 0
    for data in dataset_train:
        optimizer.zero_grad()
        out = model_l(data)
        target = data.y.unsqueeze(1).float().view(-1)
        loss = criterion(out, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    losses_train.append(total_loss)
    print("Epoch:", epoch, "Loss:", total_loss)
    # Calculate f1 score on validation set
    model_l.eval()
    total_loss = 0
    for data in dataset_val:
        out = model_l(data)
        target = data.y.unsqueeze(1).float().view(-1)
        loss = criterion(out, target)
        total_loss += loss.item()
    losses_validation.append(total_loss/validation_percentage)
    print("Validation Loss:", total_loss)
    print('-------------------')
    model_l.train()
    scheduler.step(total_loss)

# Plotting the loss function for Model 1
plt.plot(range(len(losses_validation)), losses_validation)
plt.plot(range(len(losses_train)), losses_train)
plt.xlabel('Epoch')
plt.ylabel('Loss') 
plt.title('Loss Function for Model 1')
plt.legend(['Validation', 'Train'])
plt.show()

# Store the model
torch.save(model_l.state_dict(), "model_linear.pt")
# --------------- Model 1 threshold determination ------------- #

B = np.linspace(0,1,100)
F1_score_1_list = []
best_b_1 = 0
max_F1_score_1 = 0
model_l.eval()
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
        out = model_l(data)
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
plt.title('F1 score vs Threshold for Linear model')
plt.show()
print("Best threshold for Model 1 :", best_b_1, "with F1 score :", max_F1_score_1)


### _____________________________________________________________________________ ###



### _____________________________ Combined Model ______________________________________ ###

# --------------------- Model --------------------- #

# --------------- Combined Model definition ------------- #

input_size = 384
intermediate_size = 384
dropout = 0.2
final_size = 1
threshold = 0.25

#Load the two models
model_lin_2 = Model_1().cuda()
model_lin_2.load_state_dict(torch.load("model_linear.pt"))
model_GAT_2 = Model().cuda()
model_GAT_2.load_state_dict(torch.load("model_GAT.pt"))

# freeze the weights
for param in model_lin_2.parameters():
    param.requires_grad = False
for param in model_GAT_2.parameters():
    param.requires_grad = False


class Model_combined(torch.nn.Module):
    def __init__(self,model_lin=model_lin_2,model_GAT=model_GAT_2):
        super(Model_combined, self).__init__()
        # load the two models
        self.model_lin = model_lin
        self.model_GAT = model_GAT
        # define the classifier
        self.fc1 = torch.nn.Linear(2, final_size).cuda()
        self.dropout = torch.nn.Dropout(dropout).cuda()
        self.relu = torch.nn.ReLU().cuda()
        self.sigmoid = torch.nn.Sigmoid().cuda()


    def forward(self, data):
        x_1 = self.model_lin(data)
        x_2 = self.model_GAT(data)
        x = torch.cat((x_1.unsqueeze(1),x_2),dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x.squeeze(1)
    
    
# --------------- Combined model training ------------- #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_combined = Model_combined().cuda()
optimizer = torch.optim.Adam(model_combined.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
criterion = torch.nn.MSELoss().cuda()
# criterion = torch.nn.BCEWithLogitsLoss()
dataset_train = [data.to(device) for data in dataset_train]

losses_train = []
losses_validation = []
model_combined.train()

for epoch in range(400):
    total_loss = 0
    for data in dataset_train:
        optimizer.zero_grad()
        out = model_combined(data)
        target = data.y.unsqueeze(1).float().view(-1)
        loss = criterion(out, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    losses_train.append(total_loss)
    print("Epoch:", epoch, "Loss:", total_loss)
    # Calculate f1 score on validation set
    model_combined.eval()
    total_loss = 0
    for data in dataset_val:
        out = model_combined(data)
        target = data.y.unsqueeze(1).float().view(-1)
        loss = criterion(out, target)
        total_loss += loss.item()
    losses_validation.append(total_loss/validation_percentage)
    print("Validation Loss:", total_loss)
    print('-------------------')
    model_combined.train()
    scheduler.step(total_loss)

# Plotting the loss function for Combined Model
plt.plot(range(len(losses_validation)), losses_validation)
plt.plot(range(len(losses_train)), losses_train)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function for Combined Model')
plt.legend(['Validation', 'Train'])
plt.show()

# Store the model
torch.save(model_combined.state_dict(), "model_combined.pt")

# --------------- Combined model threshold determination ------------- #

B = np.linspace(0.2,0.3,100)
F1_score_2_list = []
best_b_2 = 0
max_F1_score_2 = 0
model_combined.eval()

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
        out = model_combined(data)
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


plt.figure()
plt.plot(B,F1_score_2_list)
plt.legend(['F1 score'])
plt.xlabel('Threshold')
plt.ylabel('F1 score')
plt.title('F1 score vs Threshold for Combined model')
plt.show()
print("Best threshold for Combined Model :", best_b_2, "with F1 score :", max_F1_score_2)

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
model_combined.eval()
for transcription_id in tqdm.tqdm(test_transcription_ids):
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
        edges = np.array(np.loadtxt(path_to_test/ f"{transcription_id}.txt", dtype=str))
        formatted_edges = edges[:, [0, -1]].astype(int)
        X = [transcription[i]["text"] for i in range(len(transcription))]
        embedding_text = model_embedding.encode(X)
        X = embedding_text.tolist()
        X = torch.tensor(X, dtype=torch.float)
        edge_index = torch.tensor(formatted_edges, dtype=torch.long)
        list_edge_attr = []
        for i in range(len(edges)):
            edge_attr_part = np.zeros(size_dict)
            edge_attr_part[label_dict[edges[i, 1]]] = 1
            list_edge_attr.append(edge_attr_part)
        edge_attr = torch.tensor(list_edge_attr, dtype=torch.float)
        # Create a Data object for each transcription
        data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr)
        data = data.cuda()
        labels = model_combined(data)
        predictions = [1 if x > best_b_2 else 0 for x in labels] 

    test_labels[transcription_id] = predictions

with open("test_labels.json", "w") as file:
    json.dump(test_labels, file, indent=4)
    



