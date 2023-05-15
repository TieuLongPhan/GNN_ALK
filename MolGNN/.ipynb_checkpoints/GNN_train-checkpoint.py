import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from torch.optim.lr_scheduler import ReduceLROnPlateau
import deepchem as dc
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import f1_score, average_precision_score
device = torch.device("cuda")
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
def train(model, criterion, optimizer, train_loader):
    model.train()
    trainning_loss = 0
    truelabels = []
    probas = []
    proba_flat = []
    pred_flat = []
    for _, batch in enumerate(train_loader):
        batch.to(device)
        optimizer.zero_grad()
        output = model(batch.node_features.float(),batch.edge_attr.float(),batch.edge_index,batch.batch).to(device)
        loss = criterion(output[:,0], batch.y.float()) 
        loss.backward()
        optimizer.step()
        trainning_loss += loss.item()
        
        for label in batch.y.cpu():
            truelabels.append(np.asarray(label))

        output_probas = F.sigmoid(output)[:,0]
        probas.append(np.asarray(output_probas.detach().cpu()))
        
    for i in probas:
        for j in i:
            proba_flat.append(j)
    
    pred_flat = proba_flat.copy()
    for key, value in enumerate(proba_flat):
        if value < 0.5:
            pred_flat[key] = 0
        else:
            pred_flat[key] = 1
            
    loss = trainning_loss/len(train_loader)
    f1 = f1_score(truelabels,pred_flat)
    ap = average_precision_score(truelabels,proba_flat)
    return loss, f1, ap
