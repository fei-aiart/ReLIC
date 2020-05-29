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

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(3681, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        x1, x2 = self.base_model(x)
        x = torch.cat([x1,x2],1)
        x = self.head(x)

        return x

#
# if __name__=='__main__':
#     model = NIMA()
#     x = torch.rand((16,3,224,224))
#     out = model(x)