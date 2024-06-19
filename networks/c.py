import numpy as np
import torch.nn as nn


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone().float()
    return fun1

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

class AdversarialNetwork(nn.Module):
    def __init__(self, max_iter):
        super(AdversarialNetwork, self).__init__()
    

        self.conv1 = nn.Conv2d(256, 128, 1)
        self.conv2 = nn.Conv2d(128, 64, 1)

        self.ad_layer1 = nn.Linear(8*8*64, 512)
        self.ad_layer2 = nn.Linear(512, 32)
        self.ad_layer3 = nn.Linear(32, 3)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        # self.sigmoid = nn.Sigmoid()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter * 2

    def forward(self, bottleneck_feature):
        if self.training:
            self.iter_num += 1

        # reduce to channel with 1x1 convs
        # to 128
        # to 32

        x = self.conv1(bottleneck_feature)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)

        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0

        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        # y = self.sigmoid(y)
        return y

    def output_num(self):
        return 3

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



def DANN(features, ad_net, DA_target_array):
    ad_out = ad_net(features)
    dc_target = DA_target_array
    return nn.CrossEntropyLoss().cuda()(ad_out, dc_target)




# ad_net = AdversarialNetwork(max_iter)

# loss += DANN(bottleneck_feature, ad_net, DA_target_array)
