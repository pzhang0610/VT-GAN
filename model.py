import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_chn, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_chn, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, gf_dim=64, cond_dim=11, repeat_num=5):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1 + cond_dim, gf_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(gf_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers
        curr_dim = gf_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers
        for i in range(repeat_num):
            layers.append(ResidualBlock(in_chn=curr_dim, out_chn=curr_dim))

        # Up-sampling layers
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, c), dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, image_size=64, df_dim=64, ag_dim=11, repeat_num=5):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, df_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = df_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.shared_block = nn.Sequential(*layers)
        self.discriminator = nn.Conv2d(curr_dim, 1, kernel_size, stride=1, padding=1, bias=False)
        self.cls_angle = nn.Conv2d(curr_dim, ag_dim, kernel_size=kernel_size, bias=False)
        # self.cls_state = nn.Conv2d(curr_dim, st_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        middle = self.shared_block(x)
        dis_logit = self.discriminator(middle)
        cls_angle_logit = self.cls_angle(middle)
        # cls_state_logit = self.cls_state(middle)
        return dis_logit, cls_angle_logit.view(cls_angle_logit.size(0), cls_angle_logit.size(1))#, cls_state_logit.view(cls_state_logit.size(0), cls_state_logit.size(1))


class SiaNet(nn.Module):
    def __init__(self, df_dim=64, ed_dim=64):
        super(SiaNet, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, df_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        curr_dim = df_dim
        for i in range(1, 3):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            curr_dim = curr_dim * 2
        self.linear = nn.Linear(curr_dim, out_features=ed_dim, bias=None)

        self.embedding = nn.Sequential(*layers)

    def forward(self, x):
        features =  self.embedding(x)
        return self.linear(features.squeeze())

if __name__ == "__main__":
    device = torch.device("cuda")
    # G = Generator().to(device)
    # a = torch.rand(100, 1, 64, 64).to(device)
    # c = torch.ones(100,11).to(device)
    # logit = G(a, c)

    S = SiaNet().to(device)
    print(S)
