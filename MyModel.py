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


class HybridSN(nn.Module):
    def __init__(self, band_num, pretrain=False):
        super().__init__()
        # self.layer3d1 = nn.Conv3d(1, 8, kernel_size=(3,3,3), stride=1, padding=1)
        self.layer3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.Conv3d(8, 16, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
        )
        self.layer2d = nn.Sequential(
            nn.Conv2d(band_num*32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((10,10))
        self.layer1d = nn.Sequential(
            nn.Linear(6400, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 1),
        )

    def forward(self, input):
        out = self.layer3d(input)
        # out = self.layer3d1(input)
        # out = self.layer3d(out)
        conv3d_shape = out.shape
        out = out.reshape([conv3d_shape[0],conv3d_shape[1]*conv3d_shape[2],conv3d_shape[3],conv3d_shape[4]])
        out = self.layer2d(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.layer1d(out)
        out = torch.sigmoid(out)
        return out

class multiclass_HybridSN(nn.Module):
    def __init__(self, band_num, cls_num, pretrain=False):
        super().__init__()
        # self.layer3d1 = nn.Conv3d(1, 8, kernel_size=(3,3,3), stride=1, padding=1)
        self.layer3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.Conv3d(8, 16, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
        )
        self.layer2d = nn.Sequential(
            nn.Conv2d(band_num*32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((10,10))
        self.layer1d = nn.Sequential(
            nn.Linear(6400, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, cls_num),
        )

    def forward(self, input):
        out = self.layer3d(input)
        # out = self.layer3d1(input)
        # out = self.layer3d(out)
        conv3d_shape = out.shape
        out = out.reshape([conv3d_shape[0],conv3d_shape[1]*conv3d_shape[2],conv3d_shape[3],conv3d_shape[4]])
        out = self.layer2d(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.layer1d(out)
        out = torch.nn.functional.softmax(out,dim=1)
        return out

if __name__ == '__main__':
    # model = HSI_CLS_model(603)
    # input = torch.rand([3, 603, 200, 200])
    # print(type(input))
    # output = model(input)
    # print(output.size())

    # model = HybridSN(10)
    # input = torch.rand([3, 1, 10, 200, 200])
    # # input = torch.rand([3, 10, 200, 200])
    # print(type(input))
    # output = model(input)
    # print(output.size())

    model = multiclass_HybridSN(10,9)
    input = torch.rand([3, 1, 10, 200, 200])
    # input = torch.rand([3, 10, 200, 200])
    print(type(input))
    output = model(input)
    print(output)
    print(output.size())
