
import torch
from torch import nn


class AVDNetwork(nn.Module):
    """
    Animation via Disentanglement network
    """

    def __init__(self, num_tps, id_bottle_size=64, pose_bottle_size=64):
        super(AVDNetwork, self).__init__()
        input_size = 5*2 * num_tps
        self.num_tps = num_tps

        self.id_encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, id_bottle_size)
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, pose_bottle_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(pose_bottle_size + id_bottle_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_size)
        )

    def forward(self, kp_source, kp_random):

        bs = kp_source['fg_kp'].shape[0]
        
        pose_emb = self.pose_encoder(kp_random['fg_kp'].view(bs, -1))
        id_emb = self.id_encoder(kp_source['fg_kp'].view(bs, -1))

        rec = self.decoder(torch.cat([pose_emb, id_emb], dim=1))

        rec =  {'fg_kp': rec.view(bs, self.num_tps*5, -1)}
        return rec
