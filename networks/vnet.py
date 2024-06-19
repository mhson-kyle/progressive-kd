import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, n_filters, kernel_size, stride, padding),
            nn.BatchNorm3d(n_filters),
            nn.PReLU()
        )
        
    def forward(self, x):
        return self.net(x)


class BigBlock(nn.Module):
    def __init__(self, depth, in_channels, n_filters):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(ConvBlock(in_channels, n_filters))
            in_channels = n_filters
            
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class VNet(nn.Module):
    def __init__(self, in_channels=1, n_filters=16, decoder_branches=3):
        super().__init__()
        self.branches = decoder_branches
        
        self.enc1 = BigBlock(depth=1, in_channels=in_channels, n_filters=n_filters)
        self.expand_ch1 = torch.nn.ModuleList([nn.Conv3d(n_filters, n_filters * 2, kernel_size=3, stride=1, padding=1) for _ in range(self.branches)])
        self.down1 = nn.Conv3d(n_filters, n_filters * 2, kernel_size=2, stride=2) 
        
        self.enc2 = BigBlock(depth=2, in_channels=n_filters * 2, n_filters=n_filters * 2)
        self.expand_ch2 = torch.nn.ModuleList([nn.Conv3d(n_filters * 2, n_filters * 4, kernel_size=3, stride=1, padding=1) for _ in range(self.branches)])
        self.down2 = nn.Conv3d(n_filters * 2, n_filters * 4, kernel_size=2, stride=2) 
        
        self.enc3 = BigBlock(depth=3, in_channels=n_filters * 4, n_filters=n_filters * 4)
        self.expand_ch3 = torch.nn.ModuleList([nn.Conv3d(n_filters * 4, n_filters * 8, kernel_size=3, stride=1, padding=1) for _ in range(self.branches)])
        self.down3 = nn.Conv3d(n_filters * 4, n_filters * 8, kernel_size=2, stride=2) 

        self.enc4 = BigBlock(depth=3, in_channels=n_filters * 8, n_filters=n_filters * 8)
        self.expand_ch4 = torch.nn.ModuleList([nn.Conv3d(n_filters * 8, n_filters * 16, kernel_size=3, stride=1, padding=1) for _ in range(self.branches)])
        self.down4 = nn.Conv3d(n_filters * 8, n_filters * 16, kernel_size=2, stride=2) 

        self.enc5 = BigBlock(depth=3, in_channels=n_filters * 16, n_filters=n_filters * 16)
        self.up5 = torch.nn.ModuleList([nn.ConvTranspose3d(n_filters * 16, n_filters * 16, kernel_size=2, stride=2) for _ in range(self.branches)])

        self.dec4 = torch.nn.ModuleList([BigBlock(depth=3, in_channels=n_filters * 16, n_filters=n_filters * 16) for _ in range(self.branches)])
        self.up4 = torch.nn.ModuleList([nn.ConvTranspose3d(n_filters * 16, n_filters * 8, kernel_size=2, stride=2) for _ in range(self.branches)])
        
        self.dec3 = torch.nn.ModuleList([BigBlock(depth=3, in_channels=n_filters * 8, n_filters=n_filters * 8) for _ in range(self.branches)])
        self.up3 = torch.nn.ModuleList([nn.ConvTranspose3d(n_filters * 8, n_filters * 4, kernel_size=2, stride=2) for _ in range(self.branches)])

        self.dec2 = torch.nn.ModuleList([BigBlock(depth=2, in_channels=n_filters * 4, n_filters=n_filters * 4) for _ in range(self.branches)])
        self.up2 = torch.nn.ModuleList([nn.ConvTranspose3d(n_filters * 4, n_filters * 2, kernel_size=2, stride=2) for _ in range(self.branches)])

        self.dec1 = torch.nn.ModuleList([BigBlock(depth=1, in_channels=n_filters * 2, n_filters=n_filters * 2) for _ in range(self.branches)])

        self.conv = torch.nn.ModuleList([nn.Conv3d(n_filters * 2, in_channels, 1, 1) for _ in range(self.branches)])
        self.conv2 = torch.nn.ModuleList([nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0) for _ in range(self.branches)])

    def encoder(self, x):
        enc1_res = x
        enc1 = self.enc1(x)
        enc1 += enc1_res
        
        enc2_res = self.down1(enc1)
        enc2 = self.enc2(enc2_res)
        enc2 += enc2_res
        
        enc3_res = self.down2(enc2)
        enc3 = self.enc3(enc3_res)
        enc3 += enc3_res

        enc4_res = self.down3(enc3)
        enc4 = self.enc4(enc4_res)
        enc4 += enc4_res
        
        enc5_res = self.down4(enc4)
        enc5 = self.enc5(enc5_res)
        enc5 += enc5_res
        
        return [enc1, enc2, enc3, enc4, enc5]    
    
    def decoder(self, features, branch_idx):
        enc1, enc2, enc3, enc4, enc5 = features[0], features[1], features[2], features[3], features[4]
        
        dec4_res = self.up5[branch_idx](enc5)
        enc4 = self.expand_ch4[branch_idx](enc4)
        dec4 = dec4_res + enc4
        dec4 = self.dec4[branch_idx](dec4)
        dec4 += dec4_res
        
        dec3_res = self.up4[branch_idx](enc4)
        enc3 = self.expand_ch3[branch_idx](enc3)
        dec3 = dec3_res + enc3
        dec3 = self.dec3[branch_idx](dec3)
        dec3 += dec3_res
        
        dec2_res = self.up3[branch_idx](enc3)
        enc2 = self.expand_ch2[branch_idx](enc2)
        dec2 = dec2_res + enc2
        dec2 = self.dec2[branch_idx](dec2)
        dec2 += dec2_res
        
        dec1_res = self.up2[branch_idx](enc2)
        enc1 = self.expand_ch1[branch_idx](enc1)
        dec1 = dec1_res + enc1
        dec1 = self.dec1[branch_idx](dec1)
        dec1 += dec1_res
        
        outputs = self.conv[branch_idx](dec1)
        outputs = outputs.view(-1, 32, 128, 128)
        outputs = self.conv2[branch_idx](outputs)
        
        return outputs
    
    def forward(self, x):
        x = x.unsqueeze(1)
        assert x.ndim == 5
        features = self.encoder(x)
        
        assert self.branches == 3

        out_map_tmax = self.decoder(features, 0)
        out_map_cbv = self.decoder(features, 1)
        out_map_cbf = self.decoder(features, 2)

        map_pred_dict = {
            'out_map_tmax' : out_map_tmax,
            'out_map_cbv' : out_map_cbv,
            'out_map_cbf' : out_map_cbf
        }

        return map_pred_dict, features
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

    
if __name__ == "__main__":
    model = VNet(1, 16, 3).cpu()
    from torchsummary import summary
    # print(summary(model, (1, 64, 128, 128), device='cpu'))
    x = torch.randn(16, 32, 128, 128).cpu()
    output,features = model(x)
    print(output['out_map_tmax'].shape)
    print(output['out_map_cbv'].shape)
    print(output['out_map_cbf'].shape)