import torch
import torch.nn as nn
import torch.nn.functional as F
class InvertResidual(nn.Module):
  def __init__(self,in_channels, out_channels, stride ,t =6 ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride

  
    self.residual = nn.Sequential(
        # 1x1 conv (point wise)
          nn.Conv2d(in_channels , in_channels*t,kernel_size = 1 , stride=1 , bias = False),
          nn.BatchNorm2d(in_channels*t),
          nn.ReLU6(inplace = True),
        # 3x3 conv (depth wise)
          nn.Conv2d(in_channels*t , in_channels*t , kernel_size = 3 , stride = stride , padding=1, groups=in_channels*t , bias= False),
          nn.BatchNorm2d(in_channels*t),
          nn.ReLU6(inplace = True),
        # 1x1 conv (point wise)
          nn.Conv2d(in_channels*t , out_channels , 1),
          nn.BatchNorm2d(out_channels)
      )

  def forward(self , x):
    if self.stride == 1 and self.in_channels == self.out_channels:
      return x + self.residual(x)
    return self.residual(x)

class MobileNetV2(nn.Module):
  def __init__(self , class_num = 100):
    super().__init__()

    self.base = nn.Sequential(
        nn.Conv2d(3, 32 ,3,2,1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU6(inplace = True)
    )
  
    self.in_channels = 32
    cfgs = [
        # t, c, n, s
        [1,16,1,1],
        [6,24,2,2],
        [6,32,3,2],
        [6,64,4,2],
        [6,96,3,1],
        [6,160,3,2],
        [6,320,1,1],
    ]

    self.layers =  self._make_layer(cfgs)

    
    self.conv1 = nn.Sequential(
        nn.Conv2d(320, 1280 ,1, bias=False),
        nn.BatchNorm2d(1280),
        nn.ReLU6(inplace = True)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(1280, class_num ,1, bias=False),
    )

        
  def forward(self , x):
    x = self.base(x)
    x = self.layers(x)
    x = self.conv1(x)
    x = F.avg_pool2d(x,7)
    x = self.conv2(x)
    x = x.view(x.size(0),-1)
    return x

  def _make_layer(self, cfg):
    layers = []
    for t,c,n,s in cfg:
      for i in range(n):
        layers.append(InvertResidual(self.in_channels , c , s , t))
        self.in_channels = c
        s = 1
      
    return nn.Sequential(*layers)
    # 

def mobilenetv2(num_class):
    return MobileNetV2(num_class)
      





