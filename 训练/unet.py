import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode='3d'):
        super(ConvBlock, self).__init__()
        inter_channels = in_channels//2 if in_channels>out_channels else out_channels//2
        if mode == '3d':
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, inter_channels, (1, 3, 3), 1, (0, 1, 1)),
                nn.BatchNorm3d(inter_channels),
                nn.ReLU(True),

                nn.Conv3d(inter_channels, out_channels, (2, 3, 3), 1, (0, 1, 1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 3, 1, 1),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(True),

                nn.Conv2d(inter_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

    def forward(self, x):
        skip = self.conv(x)
        b, c, t, h, w = skip.size()
        out = self.pool(skip)
        return skip.view(b, c*t, h, w), out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels, mode='2d')

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        super(UNet, self).__init__()
        self.down1 = Down(in_channels, out_channels_list[0])
        self.down2 = Down(out_channels_list[0], out_channels_list[1])
        self.down3 = Down(out_channels_list[1], out_channels_list[2])

        self.bridge_conv = ConvBlock(out_channels_list[2], out_channels_list[3])

        self.up1 = Up(out_channels_list[3]+out_channels_list[2]*2, out_channels_list[2])
        self.up2 = Up(out_channels_list[2]+out_channels_list[1]*3, out_channels_list[1])
        self.up3 = Up(out_channels_list[1]+out_channels_list[0]*4, out_channels_list[0])

        self.end_conv = nn.Conv2d(out_channels_list[0], 2, 3, 1, 1)


    def forward(self, x):
        # down
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)

        x = self.bridge_conv(x).squeeze(2)

        # up
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)

        return self.end_conv(x)

if __name__ == '__main__':
    input = torch.rand(6, 22, 5, 256, 256).cuda()
    net = UNet(22, [32, 64, 128, 256]).cuda()
    output = net(input)
    print(output.size())