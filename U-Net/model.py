import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self , in_channels , out_channels):
        super().__init__()
        
        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_channels , out_channels, kernel_size=3 , stride= 1 , padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels , out_channels=out_channels , kernel_size = 3, stride=1 , padding = 1 , bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self , x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self , in_channels=3 , out_channels=1 , features = [64,128,256,512]):
        super().__init__()
        self.in_channels = in_channels
        self.donws = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.maxpoll = nn.MaxPool2d(kernel_size=2)
        
        
        # Down sampling part 
        for feature in features:
            self.donws.append(DoubleConv(self.in_channels , feature))
            self.in_channels = feature
        
        # Up samping part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2 , feature , kernel_size=2 ,stride = 2))
            self.ups.append(DoubleConv(feature*2 , feature))
        
        # Bottelneck
        self.bottleneck = DoubleConv(features[-1] , features[-1]*2)
        
        # Final conv layer(1x1)
        self.fianl_conv = nn.Conv2d(features[0], out_channels, kernel_size=1 , stride=1)
        
        
    def forward(self , x):
        skip_connections = []
        # Down sapling part
        for down in self.donws:
            x = down(x)
            skip_connections.append(x)
            x = self.maxpoll(x)
        # Bottelencek
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        # up sampling 
        for idx in range(0,len(self.ups),2): # 짝수 번째에서 up samping , 홀수 번째에 DoubleCOnv
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, skip_connection.shape[2:] , antialias=True)
            # (B,C,H,W)에서 채널에 대하여 concat연산
            concat_x = torch.cat((skip_connection , x) ,dim=1)
            x = self.ups[idx+1](concat_x)
        
        return self.fianl_conv(x)
