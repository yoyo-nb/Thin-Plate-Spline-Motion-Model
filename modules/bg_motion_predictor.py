from torch import nn
import torch
from torchvision import models

class BGMotionPredictor(nn.Module):
    """
    Module for background estimation, return single transformation, parametrized as 3x3 matrix. The third row is [0 0 1]
    """

    def __init__(self):
        super(BGMotionPredictor, self).__init__()
        self.bg_encoder = models.resnet18(pretrained=False)
        self.bg_encoder.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = self.bg_encoder.fc.in_features
        self.bg_encoder.fc = nn.Linear(num_features, 6)
        self.bg_encoder.fc.weight.data.zero_()
        self.bg_encoder.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, source_image, driving_image):
        bs = source_image.shape[0]
        out = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).type(source_image.type())
        prediction = self.bg_encoder(torch.cat([source_image, driving_image], dim=1))
        out[:, :2, :] = prediction.view(bs, 2, 3)
        return out
