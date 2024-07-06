import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    

class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, in_channels=64, n_filters=16, normalization='none', branches=3):
        super(UNet, self).__init__()
        self.branches = branches
        convBlock = ConvBlock
        self.block_one = convBlock(1, in_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(4, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(8, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(8, n_filters * 16, n_filters * 16, normalization=normalization)

        self.block_five_up_list = torch.nn.ModuleList([UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization) for _ in range(self.branches)])

        self.block_six_list = torch.nn.ModuleList([convBlock(8, n_filters * 8*2, n_filters * 8, normalization=normalization) for _ in range(self.branches)])
        self.block_six_up_list = torch.nn.ModuleList([UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization) for _ in range(self.branches)])

        self.block_seven_list = torch.nn.ModuleList([convBlock(4, n_filters * 4*2, n_filters * 4, normalization=normalization) for _ in range(self.branches)])
        self.block_seven_up_list = torch.nn.ModuleList([UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization) for _ in range(self.branches)])

        self.block_eight_list = torch.nn.ModuleList([convBlock(2, n_filters * 2*2, n_filters * 2, normalization=normalization) for _ in range(self.branches)])
        self.block_eight_up_list = torch.nn.ModuleList([UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization) for _ in range(self.branches)])

        self.block_nine_list = torch.nn.ModuleList([convBlock(1, n_filters*2, n_filters, normalization=normalization) for _ in range(self.branches)])

        self.out_conv2d_list = torch.nn.ModuleList([nn.Conv2d(n_filters, 1, 1, padding=0) for _ in range(self.branches)])

        self.map_out_activation = torch.nn.Sigmoid()

    def encoder(self, input):

        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        res = [x1, x2, x3, x4, x5]
        return res

    def decoder(self, features, branch_idx):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up_list[branch_idx](x5)
        x5_up = torch.cat([x5_up, x4], 1)

        x6 = self.block_six_list[branch_idx](x5_up)
        x6_up = self.block_six_up_list[branch_idx](x6)
        x6_up = torch.cat([x6_up, x3], 1)

        x7 = self.block_seven_list[branch_idx](x6_up)
        x7_up = self.block_seven_up_list[branch_idx](x7)
        x7_up = torch.cat([x7_up, x2], 1)

        x8 = self.block_eight_list[branch_idx](x7_up)
        x8_up = self.block_eight_up_list[branch_idx](x8)
        x8_up = torch.cat([x8_up, x1], 1)
        x9 = self.block_nine_list[branch_idx](x8_up)

        out = self.out_conv2d_list[branch_idx](x9)
        return out

    def forward(self, input):
        features = self.encoder(input)
        # features = intermediate layer features of encoder E

        assert self.branches == 3

        tmax = self.map_out_activation(self.decoder(features, 0))
        cbv = self.map_out_activation(self.decoder(features, 1))
        cbf = self.map_out_activation(self.decoder(features, 2))

        map_pred_dict = {
            'tmax': tmax,
            'cbv': cbv,
            'cbf': cbf
        }

        return map_pred_dict, features, features[-1]

    def initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    model = UNet(
        iin_channels=32,
        n_filters=32,
        normalization='batchnorm',
        branches=3)
    x = torch.randn(1, 32, 128, 128).cpu()
    output, _, features = model(x)
    print(output['tmax'].shape)
    print(output['cbv'].shape)
    print(output['cbf'].shape)
