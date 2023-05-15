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


def external_evaluate(model, criterion, valid_loader):
    model.eval()
    validation_loss = 0
    truelabels_val = []
    probas_val = []
    proba_flat_val = []
    pred_flat_val = []
    with torch.no_grad():
        for _, batch in enumerate(valid_loader):
            batch.to(device)
            output_val = model(batch.node_features.float(),batch.edge_attr.float(),batch.edge_index,batch.batch).to(device)
            loss = criterion(output_val[:,0], batch.y.float())
            validation_loss += loss.item()
      
        
            for label in batch.y.cpu():
                truelabels_val.append(np.asarray(label))

            output_probas_val = F.sigmoid(output_val)[:,0]
            probas_val.append(np.asarray(output_probas_val.detach().cpu()))

        
    for i in probas_val:
        for j in i:
            proba_flat_val.append(j)
    
    pred_flat_val = proba_flat_val.copy()
    for key, value in enumerate(proba_flat_val):
        if value < 0.5:
            pred_flat_val[key] = 0
        else:
            pred_flat_val[key] = 1
            
    loss_val = validation_loss/len(valid_loader)
    f1_val = f1_score(truelabels_val,pred_flat_val)
    ap_val = average_precision_score(truelabels_val,proba_flat_val)
    print("Classification report")
    print(classification_report(truelabels_val, pred_flat_val))
    print("Average precision:",ap_val)
    print("AUC:",roc_auc_score(truelabels_val, proba_flat_val))
        