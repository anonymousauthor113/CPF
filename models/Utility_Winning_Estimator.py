import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from User_Traffic_Forecaster import MLP

class SharedRepresentation(nn.Module): 
    def __init__(
            self, 
            emb_dim, 
            hidden_dim, 
            num_heads, 
            num_layers
    ):
        super().__init__()
        self.embedding = nn.Linear(emb_dim, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads),
            num_layers
        )

    def forward(self, x): 
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return x
    

class UtilityHead(nn.Module): 
    def __init__(
            self, 
            layer_dimensions, 
            activation = 'relu', 
            dropout = 0
    ): 
        super().__init__()
        self.ctr_model = MLP(layer_dimensions, activation, dropout)
        self.cvr_model = MLP(layer_dimensions, activation, dropout)

    def forward(self, ad_feature): 
        pctr = self.ctr_model(ad_feature)
        pcvr = self.cvr_model(ad_feature)
        pctcvr = pctr * pcvr
        return pctr, pctcvr
    

class WinningProbabilityHead(nn.Module):
    def __init__(
            self,
            layer_dimensions, 
            activation = 'relu', 
            dropout = 0
    ): 
        super().__init__()
        self.win_model = MLP(layer_dimensions, activation, dropout)

    def forward(self, ad_feature): 
        win_logit = self.win_model(ad_feature)
        return win_logit

    
class FutureImpressionEnv(Dataset):
    """
    This dataset relies on the internal API to access and resolve the log server. Each sample has the following form: (variable_lenghth_candidate_ads, ad_list_length, user_feature, contexts, masked_user_historical_behaviors). The ad_list_length is used for slicing the encoded ad embeddings from the shared representation layer. The target contains utility labels for each ad and the final display result.  
    """
    pass

class JointModel(nn.Module):
    def __init__(
            self, 
            shared_representation, 
            utiltiy_head, 
            winning_head,
            ad_list_length,
            billing_type
    ):
        super().__init__()
        self.shared_representation = shared_representation
        self.utiltiy_head = utiltiy_head
        self.winning_head = winning_head
        self.ad_list_length = ad_list_length
        self.billing_type = billing_type

    def forward(self, ad): 
        ad_features = self.shared_representation(ad)[:self.ad_list_length]
        pctr, pctcvr = self.utiltiy_head(ad_features)
        utility = pctr if self.billing_type else pctcvr
        logits = self.winning_head(ad_features)
        return utility, logits


def loss_fn(target, display, utility, logits, alpha): 
    loss1 = F.binary_cross_entropy_with_logits(utility, target)
    loss2 = F.cross_entropy(logits, display)
    return loss1 + alpha * loss2

