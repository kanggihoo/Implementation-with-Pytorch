import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthSeperableConv2d(nn.Module):
  def __init__(self , in_channels , out_channels , kernel_size , **kwargs):
    super().__init__()
    self.depthwise = nn.Sequential(
        nn.Conv2d(in_channels , in_channels , kernel_size , groups=in_channels,padding=1,bias=False, **kwargs ),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True)
    )
    self.pointwise = nn.Sequential(
        nn.Conv2d(in_channels, out_channels , kernel_size = 1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
  def forward(self , x):
    x = self.depthwise(x)
    x = self.pointwise(x)
    return x

class BaseConv2d(nn.Module):
  def __init__(self , in_channels , out_channels , kernel_size , **kwargs):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels , out_channels , kernel_size ,bias=False, **kwargs),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )
  def forward(self , x):
    return self.conv(x)

class MobileNet(nn.Module):
  '''
  width_factor : 이 값에 따라 입력, 출력 채널의 depth 값에 width_factor만큼 모든 layer에 곱해짐
  '''
  def __init__(self , in_channels:int = 3 ,
               width_factor:int = 1 ,
               num_classes:int = 1000):
    super().__init__()
    self.alpha = width_factor

    self.base = BaseConv2d(in_channels , int(32*self.alpha) , 3 , stride = 2,padding=1)

    self.conv1 = nn.Sequential(
        DepthSeperableConv2d(int(32*self.alpha) , int(64*self.alpha) , 3 , stride = 1),
        DepthSeperableConv2d(int(64*self.alpha) , int(128*self.alpha) , 3 , stride = 2),
    )

    self.conv2 = nn.Sequential(
        DepthSeperableConv2d(int(128*self.alpha) , int(256*self.alpha) , 3 , stride = 1),
        DepthSeperableConv2d(int(256*self.alpha) , int(256*self.alpha) , 3 , stride = 2),
    )

    self.conv3 = nn.Sequential(
        DepthSeperableConv2d(int(256*self.alpha) , int(256*self.alpha) , 3 , stride = 1),
        DepthSeperableConv2d(int(256*self.alpha) , int(512*self.alpha) , 3 , stride = 2),
    )

    self.conv4 = nn.Sequential(
        *nn.ModuleList([DepthSeperableConv2d(int(512*self.alpha) , int(512*self.alpha) ,3 , stride=1)  for _ in range(5)])
    )

    self.conv5 = nn.Sequential(
        DepthSeperableConv2d(int(512*self.alpha) , int(1024*self.alpha) , 3 , stride = 2),
        DepthSeperableConv2d(int(1024*self.alpha) , int(1024*self.alpha) , 3 , stride = 1),
    )

    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(int(1024*self.alpha) , num_classes)

  def forward(self , x):
    x = self.base(x) # (B,3,224,224) => (B,32,112,112)
    x = self.conv1(x) # (B,32,112,122) => (B,64,56,56)
    x = self.conv2(x) # (B,64,56,56) => (B,128,28,28)
    x = self.conv3(x) # (B,128,28,28) => (B,256,14,14)
    x = self.conv4(x) # (B,256,14,14) => (B,512,14,14)
    x = self.conv5(x) # (B,512,14,14) => (B,1024,7,7)
    x = self.avgpool(x) # (B,1024,7,7) => (B,1024,1,1)
    x = x.view(x.size(0) , -1) # (B,1024,1,1) => (B,1024)
    x = self.fc(x) # (B,1024) => (B,num_class)
    return x