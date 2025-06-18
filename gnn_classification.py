import json
import os
import glob
import networkx as nx
import pandas as pd
from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
import numpy as np
import concurrent.futures
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import ast 


with open('./user_data/users.json', 'r') as file:
    users = json.load(file)

label_dic = {"not related": 0, "suicide ideation":1}


train1_users = []

def load_graph(username):
    G_loaded = nx.read_graphml(f"graphs/{username}.graphml")
    for node in G_loaded.nodes():
        new_feat = np.array(ast.literal_eval(G_loaded.nodes[node]['feature']), dtype=np.float32) 
        G_loaded.nodes[node]['feature'] = new_feat
    return G_loaded


def from_networkx_to_torch_geometric(G, label):
    # Get node features and indices
    node_features = torch.tensor([G.nodes[n]['feature'] for n in G.nodes()], dtype=torch.float)
    edges_int = [(int(src), int(dst)) for src, dst in G.edges()]
    edge_list = torch.tensor(edges_int, dtype=torch.long).t().contiguous()
    
    label = torch.tensor([label], dtype=torch.long) 

    # Create the PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_list, y=label)

    return data

test_graphs = []
test_labels = []
for t in tqdm(test1_users):
    test_graphs.append(load_graph(t))
    test_labels.append(label_dic[users[t]])

test_dataset = []
for i in range(len(test_graphs)):
    try:
        test_dataset.append(from_networkx_to_torch_geometric(test_graphs[i], test_labels[i]))
    except KeyError:
        print(i)


train_graphs = []
train_labels = []
for t in tqdm(train1_users):
    train_graphs.append(load_graph(t))
    train_labels.append(label_dic[users[t]])

train_dataset = []
for i in range(len(train_graphs)):
    try:
        train_dataset.append(from_networkx_to_torch_geometric(train_graphs[i], train_labels[i]))
    except KeyError:
        print(i)


class GraphClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First graph convolution layer
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second graph convolution layer
        x = F.relu(self.conv2(x, edge_index))
        
        # Global pooling (mean pooling)
        x = global_mean_pool(x, batch)  # Aggregate node features to graph features
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


NODE_FEATURE = 768
num_classes = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphClassifier(input_dim=NODE_FEATURE, hidden_dim=64, output_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

# Training loop for each fold
for epoch in range(50):  # Adjust number of epochs according to your needs
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data)  # Model outputs log probabilities
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

# Convert labels and predictions to binary format for multiclass ROC AUC
all_labels_binary = label_binarize(all_labels, classes=[i for i in range(num_classes)])
all_preds_binary = label_binarize(all_preds, classes=[i for i in range(num_classes)])

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

# **Calculate per-class precision & recall**
precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
recall_per_class = recall_score(all_labels, all_preds, average=None)

try:
    roc_auc = roc_auc_score(all_labels_binary, all_preds_binary, average='macro', multi_class='ovr')
except ValueError as e:
    roc_auc = float('nan')  # Handle cases where ROC AUC cannot be computed

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')

# Print per-class metrics
for i, (prec, rec) in enumerate(zip(precision_per_class, recall_per_class)):
    print(f"Class {i}: Precision = {prec:.4f}, Recall = {rec:.4f}")

    