import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import Bottleneck


class HSI_CLS_model(nn.Module):
    def __init__(self, band_num, pretrain=False):
        super().__init__()
        # self.f0 = nn.Sequential(
        #     nn.Conv2d(band_num, 3, kernel_size=7, stride=1, padding=3),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(True),
        # )
        self.base_model = torchvision.models.resnet50(pretrain)
        self.fc = nn.Linear(1000, 1)

    def forward(self, input):
        # out = self.f0(input)
        # out = self.base_model(out)
        out = self.base_model(input)
        out = self.fc(out)
        out = torch.sigmoid(out)

        return out

class HSI_CLS_PCA_model(nn.Module):
    def __init__(self, band_num, pretrain=False):
        super().__init__()
        self.f0 = nn.Sequential(
            nn.Conv2d(band_num, 3, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        self.base_model = torchvision.models.resnet50(pretrain)
        self.fc = nn.Linear(1000, 1)

    def forward(self, input):
        out = self.f0(input)
        out = self.base_model(out)
        # out = self.base_model(input)
        out = self.fc(out)
        out = torch.sigmoid(out)

        return out


# class HybridSN()
#    def __init__(self, band_num, pretrain=False):
#         self.f0 = nn.Sequential(
#             nn.Conv2d(band_num, 3, kernel_size=7, stride=1, padding=3),
#             nn.BatchNorm2d(3),
#             nn.ReLU(True),
#         )

#     def forward(self, input):


if __name__ == '__main__':
    model = HSI_CLS_model(603)
    input = torch.rand([3, 603, 200, 200])
    print(type(input))
    output = model(input)
    print(output.size())
