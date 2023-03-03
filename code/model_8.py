import torch
import torch.nn as nn

from acon import MetaAconC,AconC


class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,  # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.0625):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        if out_channel < 128:
            self.left = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=True),
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.1),
                nn.GELU(),
                nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.1),
                
                MetaAconC(out_channel),
                # AconC(out_channel),
                # nn.GELU(),

                nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.1)
            )
        else:
            self.left = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=True),
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.1),
                nn.GELU(),
                nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.1),

                MetaAconC(out_channel),
                # AconC(out_channel),
                # nn.GELU(),

                nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.1),
                SqueezeExcite(in_channel, out_channel)
            )
        self.shortcut = None
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.1),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.1),
            )
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.left(x)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        out = self.gelu(out)
        return out


class resnetv1(nn.Module):
    def __init__(self):
        super(resnetv1, self).__init__()
        self.inputblock = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.1),

            MetaAconC(64),
            # AconC(64),
            # nn.GELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.1),

            MetaAconC(64),
            # AconC(64),
            # nn.GELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = ResidualBlock(64, 64)  # 32*112*112
        self.layer2 = ResidualBlock(64, 64)  # 32*112*112
        self.layer3 = ResidualBlock(64, 128,2)  # 64*56*56
        self.layer4 = ResidualBlock(128, 128)  # 64*56*56
        self.layer5 = ResidualBlock(128, 128)  # 128*28*28
        self.layer6 = ResidualBlock(128, 256,2)  # 128*28*28
        self.layer7 = ResidualBlock(256, 256)  # 256*14*14
        self.layer8 = ResidualBlock(256, 256)  # 256*14*14
        self.layer9 = ResidualBlock(256, 512,2)  # 512*7*7
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.inputblock(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# from torchstat import stat
# from torchsummary import summary
# net = resnetv1().cuda()
# summary(net,(3,224,224),device="cuda")
