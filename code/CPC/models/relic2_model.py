import torch.nn as nn
import torch
from models.mv2 import cat_net

def SelfAttentionMap(x):
    batch_size, in_channels, h, w = x.size()
    quary = x.view(batch_size, in_channels, -1)
    key = quary
    quary = quary.permute(0, 2, 1)

    sim_map = torch.matmul(quary, key)

    ql2 = torch.norm(quary, dim=2, keepdim=True)
    kl2 = torch.norm(key, dim=1, keepdim=True)
    sim_map = torch.div(sim_map, torch.matmul(ql2, kl2).clamp(min=1e-8))

    return sim_map

class NIMA(nn.Module):
    def __init__(self):
        super(NIMA, self).__init__()
        base_model = cat_net()

        self.base_model = base_model
        for p in self.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(8,64)
        self.relu = nn.Tanh()
        self.fc1 = nn.Linear(64,2)
        # self.sm = nn.Softmax(dim=1)
        self.sm = nn.Sigmoid()

        self.head = nn.Sequential(
            nn.Linear(3681, 50),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(50, 1)
        )

    def forward(self,x):
        x1,x2 = self.base_model(x)
        x1_max = torch.max(x1,dim=1)[0].unsqueeze(1)
        x1_min = torch.min(x1,dim=1)[0].unsqueeze(1)
        x1_mean = torch.mean(x1,dim=1).unsqueeze(1)
        x1_std = torch.std(x1,dim=1).unsqueeze(1)
        x1_in = torch.cat([x1_max,x1_min,x1_mean,x1_std],1)

        x2_max = torch.max(x2,dim=1)[0].unsqueeze(1)
        x2_min = torch.min(x2,dim=1)[0].unsqueeze(1)
        x2_mean = torch.mean(x2,dim=1).unsqueeze(1)
        x2_std = torch.std(x2,dim=1).unsqueeze(1)
        x2_in = torch.cat([x2_max,x2_min,x2_mean,x2_std],1)

        x_in = torch.cat([x1_in,x2_in],1)
        x_out =self.sm(self.fc1(self.relu(self.fc(x_in))))

        x1 = x1 * torch.unsqueeze(x_out[:, 0], 1)
        x2 = x2 * torch.unsqueeze(x_out[:, 1], 1)
        x = torch.cat([x1,x2],1)
        x = self.head(x)

        return x
