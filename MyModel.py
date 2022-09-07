import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import Bottleneck


class syjAE(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        # Encoder
        self.base_model = torchvision.models.resnet18(pretrain)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(*self.base_layers[0:3])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.avgpool = nn.AvgPool2d(3, stride=1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.Conv2d(3, 3, kernel_size=1, stride=1),
        )

    def forward(self, input):
        e1 = self.layer1(input)  # 64,128,128
        e2 = self.layer2(e1)  # 64,64,64
        e3 = self.layer3(e2)  # 128,32,32
        e4 = self.layer4(e3)  # 256,16,16
        f = self.layer5(e4)  # 512,8,8

        dout = self.decoder(f)

        dout = torch.sigmoid(dout)

        return dout


class vggAE(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        # Encoder
        self.base_model = torchvision.models.vgg16_bn(pretrain)
        self.base_layers = list(self.base_model.children())
        self.encoder = self.base_layers[0]

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.Conv2d(3, 3, kernel_size=1, stride=1),
        )

    def forward(self, input):

        f = self.encoder(input)  # 512,8,8

        dout = self.decoder(f)

        dout = torch.sigmoid(dout)

        return dout


class vgg2AE(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        # Encoder
        self.base_model = torchvision.models.vgg16_bn(pretrain)
        self.base_layers = list(self.base_model.children())
        self.encoder = self.base_layers[0]

        self.decoder = Recon_Head()

    def forward(self, input):

        f = self.encoder(input)  # 512,8,8

        dout = self.decoder(f)

        dout = torch.sigmoid(dout)

        return dout


class cyrAE(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        # Encoder
        self.base_model = torchvision.models.resnet50(pretrain)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(*self.base_layers[0:3])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]

        self.decoder = Recon_Head()

    def forward(self, input):
        e1 = self.layer1(input)  # 64,128,128
        e2 = self.layer2(e1)  # 64,64,64
        e3 = self.layer3(e2)  # 128,32,32
        e4 = self.layer4(e3)  # 256,16,16
        f = self.layer5(e4)  # 512,8,8

        dout = self.decoder(f)

        return dout


class UnetDecoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(UnetDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        # Encoder
        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(*self.base_layers[0:3])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        # UnetDecoder
        self.unetdecode4 = UnetDecoder(512, 256+256, 256)
        self.unetdecode3 = UnetDecoder(256, 256+128, 256)
        self.unetdecode2 = UnetDecoder(256, 128+64, 128)
        self.unetdecode1 = UnetDecoder(128, 64+64, 64)
        self.unetdecode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.unetconv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        e0 = self.layer0(input)
        e1 = self.layer1(e0)  # 64,128,128
        e2 = self.layer2(e1)  # 64,64,64
        e3 = self.layer3(e2)  # 128,32,32
        e4 = self.layer4(e3)  # 256,16,16
        f = self.layer5(e4)  # 512,8,8

        ud4 = self.unetdecode4(f, e4)  # 256,16,16
        ud3 = self.unetdecode3(ud4, e3)  # 256,32,32
        ud2 = self.unetdecode2(ud3, e2)  # 128,64,64
        ud1 = self.unetdecode1(ud2, e1)  # 64,128,128
        ud0 = self.unetdecode0(ud1)  # 64,256,256
        uout = self.unetconv_last(ud0)  # 1,256,256
        uout = torch.sigmoid(uout)

        return uout


class Recon_Head(nn.Module):
    def __init__(self, block=Bottleneck):
        super(Recon_Head, self).__init__()

        self.decoder = nn.Sequential(
            UpBlock(512, 512, upsample=True),
            nn.Conv2d(512, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            UpBlock(512, 256, upsample=True),
            nn.Conv2d(256, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            UpBlock(256, 128, upsample=True),
            UpBlock(128, 64, upsample=True),
            UpBlock(64, 32, upsample=True),
            nn.Conv2d(32, 3, 5, 1, 2),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)


class UpBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.will_ups = upsample

    def forward(self, x):
        if self.will_ups:
            x = nn.functional.interpolate(x,
                                          scale_factor=2, mode="bilinear", align_corners=True)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ReconstructiveSubNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_width=128):
        super(ReconstructiveSubNetwork, self).__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)

    def forward(self, x):
        b5 = self.encoder(x)
        output = self.decoder(b5)
        return output


class DiscriminativeSubNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, out_features=False):
        super(DiscriminativeSubNetwork, self).__init__()
        base_width = base_channels
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)
        self.segment_act = torch.nn.Sigmoid()
        self.out_features = out_features

    def forward(self, x):
        b1, b2, b3, b4, b5, b6 = self.encoder_segment(x)
        output_segment = self.decoder_segment(b1, b2, b3, b4, b5, b6)
        # output_segment = self.segment_act(output_segment)
        if self.out_features:
            return output_segment, b2, b3, b4, b5, b6
        else:
            return output_segment


class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderDiscriminative, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))

        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp3(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1, b2, b3, b4, b5, b6


class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderDiscriminative, self).__init__()

        self.up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                  nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(base_width * 8),
                                  nn.ReLU(inplace=True))
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width*(8+8), base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*(4+8), base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*(2+4), base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*(2+1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1, b2, b3, b4, b5, b6):
        up_b = self.up_b(b6)
        cat_b = torch.cat((up_b, b5), dim=1)
        db_b = self.db_b(cat_b)

        up1 = self.up1(db_b)
        cat1 = torch.cat((up1, b4), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b3), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b2), dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4, b1), dim=1)
        db4 = self.db4(cat4)

        out = self.fin_out(db4)
        return out


class EncoderReconstructive(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderReconstructive, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp3(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        return b5


class DecoderReconstructive(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderReconstructive, self).__init__()

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width*2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width*2),
                                 nn.ReLU(inplace=True))
        # cat with base*1
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*1, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*1),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))
        # self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, b5):
        up1 = self.up1(b5)
        db1 = self.db1(up1)

        up2 = self.up2(db1)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        up4 = self.up4(db3)
        db4 = self.db4(up4)

        out = self.fin_out(db4)
        return out


class ssimAE(nn.Module):
    def __init__(self,midchannel = 1):
        super().__init__()
        base_width = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, 2*base_width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2*base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*base_width, 2*base_width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*base_width, 4*base_width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4*base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*base_width, 2*base_width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*base_width, base_width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, 1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(1, base_width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, 2*base_width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*base_width, 4*base_width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*base_width),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4*base_width, 2*base_width, kernel_size=2, stride=2),
            nn.BatchNorm2d(2*base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*base_width, 2*base_width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*base_width),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*base_width, base_width, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_width, base_width, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_width, base_width, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_width, 3, kernel_size=2, stride=2),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.encoder(input)
        out = self.decoder(out)

        return out


class bottleneck(nn.Module):
    def __init__(self, in_channels):
        self.f = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, in_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.f(input)
