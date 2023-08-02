import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class TrafficDataset(Dataset):
    """
    The traffic dataset can be treated as a high-dimension 2D matrix, and we slice this matrix with a sliding window to generate pairs of (look-back, forecasting horizon). 
    """
    def __init__(self, traffic_data, h, tau):
        super().__init__()
        self.data = traffic_data
        self.h = h
        self.tau = tau

    def __len__(self):
        return self.data.shape[-1] - self.h -self.tau + 1
    
    @property
    def dimension_of_attribute_combinations(self):
        return self.data.shape[0]
    
    def __getitem__(self, index): 
        look_back_window = self.data[:, index:index + self.h].float()
        forecasting_horizon = self.data[:, index + self.h:index + self.h + self.tau]
        return look_back_window, forecasting_horizon


class MLP(nn.Module):
    def __init__(
        self,    
        dimensions = [256, 128, 64], 
        activation = "ReLu",
        dropout = 0.2
    ): 
        super().__init__()
        layers = []
        for i in range(len(dimensions) - 1):
            in_feature = dimensions[i] 
            out_feature = dimensions[i + 1]
            layers.append(nn.Linear(in_feature, out_feature))
            if i != len(dimensions) - 1:
                if activation.lower() == "relu":
                    layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, input): 
        return self.mlp(input)


class LatentReularizer(nn.Module): 
    """The authors of Crossformer have released their implementation, so we omit the pytorhc rebuild for clarity. In this code, we user GRU as a demostration. The original implementation of Crossformer is here: https://github.com/Thinklab-SJTU/Crossformer
    """
    pass

class UserTrafficForecaster(nn.Module): 
    
    def __init__(
            self, 
            encoder_dimensions, 
            encoder_activation, 
            encoder_dropout, 
            h, 
            tau
            ):
        super().__init__()
        self.h = h
        self.tau = tau
        self.encoder = MLP(encoder_dimensions, encoder_activation, encoder_dropout)
        self.decoder = MLP(encoder_dimensions[::-1], encoder_activation, encoder_dropout)
        self.latent_regularizer = nn.GRU(input_size = encoder_dimensions[-1], hidden_size = 64, num_layers = 4)

    def forward(self, Y): 
        X = self.encoder(Y.permute([0, 2, 1]))
        Y_rec = self.decoder(X).permute([0, 2, 1])
        X = X.permute([1, 0, 2])
        for i in range(self.tau):
            _, hidden = self.latent_regularizer(X)
            X_pred = hidden[-1]
            X = torch.concat([X, X_pred], dim = 0)

        Y_pred = self.decoder(X[-self.tau:]).permute([1, 2, 0])
        return Y_rec, Y_pred


def my_loss(look_back_wnd, forecasting_horizon_wnd, reconstruction, prediction, lambda_R): 
    loss1 = F.l1_loss(reconstruction, look_back_wnd)
    loss2 = F.mse_loss(prediction, forecasting_horizon_wnd)
    return loss1 + lambda_R * loss2

def train(model, dataloader, test_dataloader, loss_fn, optimizer, num_epochs, lambda_R, writer): 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs): 
        model.train()
        running_loss = 0.0

        for i, (input, target) in enumerate(dataloader):
            input = input.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            rec, pred = model(input)
            loss = loss_fn(input, target, rec, pred, lambda_R)

            loss.backward()
            optimizer.step()

            running_loss = running_loss + loss.item()
            writer.add_scalar('Train Loss', loss.item(), epoch * len(dataloader) + i)

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        model.eval()
        test_loss = 0.0

        with torch.no_grad(): 
            for input, target in test_dataloader:
                input = input.to(device)
                target = target.to(device)

                rec, pred = model(input)
                loss = loss_fn(input, target, rec, pred, lambda_R)

                test_loss += loss.item()

            avg_loss = test_loss / len(test_dataloader)
            print(f"Epoch {epoch}: Testing Loss: {avg_loss}")  
            writer.add_scalar('Test Loss', avg_loss, epoch)

    writer.close()


