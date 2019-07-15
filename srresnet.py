import torch
import torch.nn as nn
import math

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        # output = self.relu(self.in1(self.conv1(x)))
        output = self.relu(self.conv1(x))
        # output = self.in2(self.conv2(output))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output

class _NetG(nn.Module):
    def __init__(self):
        super(_NetG, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                torch.nn.init.kaiming_normal_(m.weight * 0.1)
                # m.weight.data.normal_(0, math.sqrt(0.2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        # out = self.bn_mid(self.conv_mid(out))
        out = self.conv_mid(out)
        out = torch.add(out,residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out

class _NetD(nn.Module):
    def __init__(self):
        super(_NetD, self).__init__()

        # input is (3) x 96 x 96
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        # state size. (64) x 96 x 96
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)            
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        # state size. (64) x 96 x 96
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        
        # state size. (64) x 48 x 48
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        # state size. (128) x 48 x 48
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        # state size. (256) x 24 x 24
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)

        # state size. (256) x 12 x 12
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm7 = nn.BatchNorm2d(512)
        self.relu7 = nn.LeakyReLU(0.2, inplace=True)

        # state size. (512) x 12 x 12
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm8 = nn.BatchNorm2d(512)
        self.relu8 = nn.LeakyReLU(0.2, inplace=True)

        self.relu9 = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):

        out1_noactiv = self.conv1(input)
        out1 = self.relu1(out1_noactiv)

        out2_noactiv = self.batchnorm2(self.conv2(out1))
        out2 = self.relu2(out2_noactiv)

        out3_noactiv = self.batchnorm3(self.conv3(out2))
        out3 = self.relu3(out3_noactiv)

        out4_noactiv = self.batchnorm4(self.conv4(out3))
        out4 = self.relu4(out4_noactiv)

        out5_noactiv = self.batchnorm5(self.conv5(out4))
        out5 = self.relu5(out5_noactiv)

        out6_noactiv = self.batchnorm6(self.conv6(out5))
        out6 = self.relu6(out6_noactiv)

        out7_noactiv = self.batchnorm7(self.conv7(out6))
        out7 = self.relu7(out7_noactiv)

        out8 = self.relu8(self.batchnorm8(self.conv8(out7)))
        
        out1_noactiv = out1_noactiv.view(out1_noactiv.size(0), -1)
        out2_noactiv = out2_noactiv.view(out2_noactiv.size(0), -1)
        out3_noactiv = out3_noactiv.view(out3_noactiv.size(0), -1)
        out4_noactiv = out4_noactiv.view(out4_noactiv.size(0), -1)
        out5_noactiv = out5_noactiv.view(out5_noactiv.size(0), -1)
        out6_noactiv = out6_noactiv.view(out6_noactiv.size(0), -1)
        out7_noactiv = out7_noactiv.view(out7_noactiv.size(0), -1)

        # state size. (512) x 6 x 6
        out8 = out8.view(out8.size(0), -1)

        # state size. (512 x 6 x 6)
        out_fc1 = self.fc1(out8)

        # state size. (1024)
        out_fc1 = self.relu9(out_fc1)

        out_fc2 = self.fc2(out_fc1)
        out_final = self.sigmoid(out_fc2)
        out_final = out_final.view(-1, 1).squeeze(1)
        
        return out1_noactiv, out2_noactiv, out3_noactiv, out4_noactiv, out5_noactiv, out6_noactiv, out7_noactiv, out_final