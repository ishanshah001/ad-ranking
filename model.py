import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

class TwinTowerModel(nn.Module):
    def __init__(self, user_input_dim, ad_vocab_size, embed_dim=32):
        super(TwinTowerModel, self).__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        self.ad_embedding = nn.Embedding(ad_vocab_size, embed_dim)

    def forward(self, user_feats, ad_ids):
        user_emb = self.user_tower(user_feats)              # shape: (batch_size, embed_dim)
        ad_emb = self.ad_embedding(ad_ids)                   # shape: (batch_size, embed_dim)
        dot = (user_emb * ad_emb).sum(dim=1)                 # dot product per example
        prob = torch.sigmoid(dot)
        return prob