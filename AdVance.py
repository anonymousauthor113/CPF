import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from mamba_ssm import Mamba

class TimeStampPositionalEncoding(nn.Module): 

    def __init__(self, dim): 
        super().__init__()
        assert dim // 2 * 2 == dim, "The embedding dimension should be even. "
        self.weight = nn.Parameter(torch.randn(dim/2, 1))

    def forward(self, t, x):
        t = torch.Tensor(t)
        pos = t.unsqueeze(-1).repeat(1, 1, x.size(-1))
        pos[:, :, 0::2] = torch.sin(pos[:, :, 0::2] * self.weight)
        pos[:, :, 1::2] = torch.sin(pos[:, :, 1::2] * self.weight)
        return x + pos
    

class LocalInterestModule(nn.Module): 

    def __init__(self, dim, ssm_state=16, ssm_num_layers=3, ssm_conv=4, ssm_expand=2):
        super().__init__()
        self.dim = dim
        self.pe = TimeStampPositionalEncoding(dim)
        self.local_ssm = nn.Sequential(*[Mamba(d_model=dim + 1, d_state=ssm_state, d_conv=ssm_conv, expand=ssm_expand) for _ in range(ssm_num_layers)])

    def forward(self, history, time_stamp, click_indices): 
        indices = torch.Tensor([[i for i, x in enumerate(l) if x == True] for l in click_indices])
        clicks = torch.gather(history, 1, indices)
        clicks_time_stamp = torch.gather(time_stamp, 1, indices)
        clicks = self.pe(clicks_time_stamp, clicks)
        time_intervals = torch.zeros_like(time_stamp)
        time_intervals[:, 1:] = time_stamp[:, 1:] - time_stamp[:, :-1]
        history = torch.cat([history, time_intervals.unsqueeze_[-1]], dim=-1)
        fatigue_vectors = self.local_ssm(history)
        return fatigue_vectors, clicks, clicks_time_stamp


def causal_mask(time_stamp, clicks_time_stamp, current_time_stamp): 
    history_indices = [[i for i, x in enumerate(l) if x <= current_time_stamp[j]] for j, l in enumerate(time_stamp)]
    click_indices = [[i for i, x in enumerate(l) if x <= current_time_stamp[j]] for j, l in enumerate(clicks_time_stamp)]
    return history_indices[:, -1:], click_indices

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output
    

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_fatigue, expansion, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in + d_fatigue, d_in * expansion) 
        self.w_2 = nn.Linear(d_in * expansion, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, fatigue_vec):
        residual = x
        x = torch.cat([x, fatigue_vec.unsqueeze(1).repeat(1, x.size(1), 1)])
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, dim, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(dim, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, dim, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)


    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q = self.attention(q, k, v)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q
    

class MLP(nn.Module):
    def __init__(
        self,    
        dimensions = [256, 128, 64], 
        activation = "ReLu",
        dropout = 0.5
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
    
class EncoderLayer(nn.Module):

    def __init__(self, dim, d_fatigue, expansion, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_cross_attn = MultiHeadAttention(n_head, dim, d_k, d_v, dropout=dropout)
        self.conditional_MLP = PositionwiseFeedForward(dim, d_fatigue, expansion, dropout=dropout)

    def forward(self, q, k, fatigue_vec):
        enc_output = self.slf_cross_attn(q, k, k)
        enc_output = self.conditional_MLP(enc_output, fatigue_vec)
        return enc_output
    

class AuctionRepresentationModule(nn.Module): 
    def __init__(self, dim, d_fatigue, expansion, n_head, d_k, d_v, dropout, num_layers):
        super().__init__()
        self.enc_stack = nn.ModuleList(
            [EncoderLayer(dim, d_fatigue, expansion, n_head, d_k, d_v, dropout) for _ in range(num_layers)]
        )
        self.win_rate = MLP([dim] + [128, 64, 1])
        self.ctr_model = MLP([dim] + [128, 64, 1])
        self.cvr_model = MLP([dim] + [128, 64, 1])

    def forward(self, candidate_ads, clicks, fatigue_vec, user_feature, context): 
        q = candidate_ads
        k = torch.cat([candidate_ads, clicks, user_feature.unsqueeze(1), context.unsqueeze(1)])
        for layer in self.enc_stack:
            q = layer(q, k, fatigue_vec)

        win_logits = self.win_rate(q)
        ctr_logits = self.ctr_model(q)
        cvr_logits = self.cvr_model(q)

        return q[:, 0, :], win_logits, ctr_logits, cvr_logits
    

class GlobalCampaignModule(nn.Module): 
    def __init__(self, dim, ssm_state, ssm_num_layers=3, ssm_conv=4, ssm_expand=2): 
        self.global_ssm = nn.Sequential(
            *[Mamba(d_model=dim + 1, d_state=ssm_state, d_conv=ssm_conv, expand=ssm_expand) for _ in range(ssm_num_layers)])
        self.cp_model = MLP([dim + 4] + [128, 64, 4])

    def forward(self, auc_seq, time_stamp, accumulation): 
        time_intervals = torch.zeros_like(time_stamp)
        time_intervals[:, 1:] = time_stamp[:, 1:] - time_stamp[:, :-1]
        input = torch.cat([auc_seq, time_intervals.unsqueeze(-1)], dim = -1)
        output = self.global_ssm(input)[:, -1, :]
        output = torch.cat([output, accumulation.unsqueeze(-1)], dim=-1)
        campaign_performance = self.cp_model(output)
        return campaign_performance


class AdVance(nn.Module): 

    def __init__(self, dim, local_ssm_state=16, l_ssm_num_layers=3, l_ssm_conv=4, l_ssm_expand=2, global_ssm_state=16, g_ssm_num_layers=3, g_ssm_conv=4, g_ssm_expand=2, n_heads=4, expansion=4, num_enc_layers=3): 
        super().__init__()
        self.local_interest = LocalInterestModule(dim, local_ssm_state, l_ssm_num_layers, l_ssm_conv, l_ssm_expand)
        self.auction_repr = AuctionRepresentationModule(dim, dim, expansion, n_heads, dim, dim, dropout=0.1, num_layers=num_enc_layers)
        self.global_campaign = GlobalCampaignModule(dim, global_ssm_state, g_ssm_num_layers, g_ssm_conv, g_ssm_expand)

    def forward(self, auc_records): 
        """
        auc_records has the shape [Batch, Length]. The batch dimension corresponds to different campaigns. Each sample in the auc_records is a dictionary containing the current auction's timestamp, candidate ads' embeddings, user feature, context, user history, and the time stamp of the user history. 
        """
        def extract(dict, key): 
            return [[dict[i][j][key] for j in range(len(dict[i]))] for i in range(len(dict))]
        
        history = extract(auc_records, 'his')
        time_stamp = extract(auc_records, 'his_stamp')
        click_indices = extract(auc_records, 'clk')

        fatigue_vectors, clicks, clicks_time_stamp = self.local_interest(history, time_stamp, click_indices)
        
        fatigue_indices, click_indices = causal_mask(time_stamp, clicks_time_stamp, extract(auc_records, 'cur_stamp'))

        fatigue_vec = torch.gather(fatigue_vectors, 1, fatigue_indices)
        clicks = torch.gather(clicks, 1, click_indices)

        auc_seq, win_logits, ctr_logits, cvr_logits = self.auction_repr(
             extract(auc_records, 'ads'), 
             clicks, 
             fatigue_vec, 
             extract(auc_records, 'usr'), 
             extract(auc_records, 'ctx'))
        
        imp_accu = torch.sum(F.sigmoi(win_logits[...,0,:]))
        cost_accu = torch.sum(F.sigmoi(win_logits[...,0,:]) * extract(auc_records, 'ecpm'))
        clk_accu = torch.sum(F.sigmoid(win_logits[...,0,:]) * F.sigmoid(ctr_logits[...,0,:]))
        cvr_accu = torch.sum(F.sigmoid(win_logits[...,0,:]) * F.sigmoid(ctr_logits[...,0,:]) * F.sigmoid(cvr_logits[...,0,:]))

        accumulation = torch.cat([imp_accu, cost_accu, clk_accu, cvr_accu], dim=-1).squeeze()



        campaign_performance = self.global_campaign(auc_seq, extract(auc_records, 'cur_stamp'), accumulation)

        return win_logits, ctr_logits, cvr_logits, campaign_performance
    

def train(model, dataloader, test_dataloader, loss_fn, optimizer, num_epochs, alpha, writer): 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs): 
        model.train()
        running_loss = 0.0

        for i, (auc_records, clk, cvr, win_distr, cp) in enumerate(dataloader):
            auc_records = auc_records.to(device)
            clk = clk.to(device)
            cvr = cvr.to(device)
            win_distr = win_distr.to(device)
            cp = cp.to(device)


            optimizer.zero_grad()

            win_logits, ctr_logits, cvr_logits, campaign_performance = model(auc_records)
            loss = alpha * F.binary_cross_entropy_with_logits(ctr_logits, clk) + (1 - alpha) * F.binary_cross_entropy(F.sigmoid(ctr_logits) * F.sigmoid(cvr_logits), cvr) + F.mse_loss(campaign_performance, cp)

            loss.backward()
            optimizer.step()

            running_loss = running_loss + loss.item()
            writer.add_scalar('Train Loss', loss.item(), epoch * len(dataloader) + i)

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        model.eval()
        test_loss = 0.0

        with torch.no_grad(): 
            for auc_records, clk, cvr, win_distr, cp in test_dataloader:
                auc_records = auc_records.to(device)
                clk = clk.to(device)
                cvr = cvr.to(device)
                win_distr = win_distr.to(device)
                cp = cp.to(device)

                win_logits, ctr_logits, cvr_logits, campaign_performance = model(auc_records)
                loss = alpha * F.binary_cross_entropy_with_logits(ctr_logits, clk) + (1 - alpha) * F.binary_cross_entropy(F.sigmoid(ctr_logits) * F.sigmoid(cvr_logits), cvr) + F.mse_loss(campaign_performance, cp)


                test_loss += loss.item()

            avg_loss = test_loss / len(test_dataloader)
            print(f"Epoch {epoch}: Testing Loss: {avg_loss}")  
            writer.add_scalar('Test Loss', avg_loss, epoch)

    writer.close()