import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding ,bias = False, active:bool=True):
    super().__init__()
    module = nn.ModuleList(
        [
            nn.Conv2d(in_channels , out_channels , kernel_size = kernel_size , stride= stride , padding = padding,bias = bias),
            nn.BatchNorm2d(out_channels)
        ]
    )
    if active:
      module.append(nn.ReLU(inplace = True))
    self.sequence = nn.Sequential(*module)

  def forward(self , x):
    return self.sequence(x)

class SEBlock(nn.Module):
  def __init__(self , out_channels , r=16 ):
    super().__init__()
    self.squeeze = nn.AdaptiveAvgPool2d(1)
    self.excitation = nn.Sequential(
        nn.Linear(out_channels , out_channels//r),
        nn.ReLU(inplace = True),
        nn.Linear(out_channels//r , out_channels),
        nn.Sigmoid()
    )

  def forward(self , x):
    squeeze = self.squeeze(x)
    flatten = squeeze.view(squeeze.size(0),-1)
    excitation = self.excitation(flatten)
    return excitation

class BaseResidualSEBlock(nn.Module):
  expansion = 1
  def __init__(self , in_channels, out_channels , kernel_size = 3 , stride =1 , padding =1 , r=16 ):
    super().__init__()

    self.residual = nn.Sequential(
        ConvBlock(in_channels , out_channels, kernel_size , stride , padding , active=True),
        ConvBlock(out_channels , out_channels*BaseResidualSEBlock.expansion , kernel_size , stride=1, padding=padding, active = True)
    )
    
    self.shortcut = nn.Sequential()
    if in_channels != out_channels*BaseResidualSEBlock.expansion or stride !=1:
      self.shortcut = nn.Sequential(
          ConvBlock(in_channels , out_channels*BaseResidualSEBlock.expansion , kernel_size = 1 , stride =stride , padding=0, active=False)
      )

    self.se = SEBlock(out_channels*BaseResidualSEBlock.expansion ,r )

  def forward(self , x):
    shortcut = self.shortcut(x)
    residual = self.residual(x)
    se = self.se(residual) # B C H W => (B,C)
    se = se.view(se.size(0),se.size(1) , 1,1).repeat(1,1,residual.size(2) , residual.size(3)) # (B,C) => (B,C,1,1) => (B,C,H,W)
    return F.relu(residual*se + shortcut)

class BottelneckResidualSEBlock(nn.Module):
  expansion = 4
  def __init__(self , in_channels, out_channels , kernel_size = 3 , stride =2 , padding =1 , r=16 ):
    super().__init__()

    self.residual = nn.Sequential(
        ConvBlock(in_channels , out_channels, kernel_size=1 , stride=1 , padding=0 , active=True),
        ConvBlock(out_channels , out_channels, kernel_size=3 , stride=stride, padding=1, active = True),
        ConvBlock(out_channels , out_channels*self.expansion, kernel_size=1 , stride=1, padding=0, active = True),
    )

    self.se = SEBlock(out_channels*self.expansion ,r )
    self.shortcut = nn.Sequential()
    if stride !=1 or in_channels != out_channels*self.expansion:
      self.shortcut = nn.Sequential(
        ConvBlock(in_channels , out_channels*self.expansion , kernel_size = 1 , stride =stride , padding=0, active=False)
    )
  def forward(self , x):
    shortcut = self.shortcut(x)
    residual = self.residual(x)
    se = self.se(residual)
    se = se.view(se.size(0),se.size(1) , 1,1).repeat(1,1,residual.size(2) , residual.size(3))
    return F.relu(residual*se + shortcut)
    
class SEResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes=10):
    super(SEResNet, self).__init__()
    self.expansion = block.expansion
    self.in_channels = 64
    self.num_classes = num_classes
    self.base = ConvBlock(3 , 64, kernel_size = 7 , stride =2 , padding =3 , active = True)
    self.stage1 = self._make_layer(block , 64,num_blocks[0] , stride=2)
    self.stage2 = self._make_layer(block , 128,num_blocks[1] , stride=2)
    self.stage3 = self._make_layer(block , 256,num_blocks[2] , stride=2)
    self.stage4 = self._make_layer(block , 512,num_blocks[3] , stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512*block.expansion , num_classes)

  def forward(self, x):
    assert x.shape ==(x.size(0) , 3 , 224,224) , f"input_image shape :{x.shape}"
    x = self.base(x)
    assert x.shape ==(x.size(0) , 64 , 112,112) , f"base conv output shape :{x.shape}"
    x = self.stage1(x)
    assert x.shape ==(x.size(0) , 64*self.expansion , 56,56) , f"stage1 output shape :{x.shape}"
    x = self.stage2(x)
    assert x.shape ==(x.size(0) , 128*self.expansion , 28,28) , f"stage1 output shape :{x.shape}"
    x = self.stage3(x)
    assert x.shape ==(x.size(0) , 256*self.expansion , 14,14) , f"stage1 output shape :{x.shape}"
    x = self.stage4(x)
    assert x.shape ==(x.size(0) , 512*self.expansion , 7,7) , f"stage1 output shape :{x.shape}"
    x = self.avgpool(x)
    x = x.view(x.size(0) , -1)
    x = self.fc(x)
    assert x.shape ==(x.size(0) , self.num_classes) , f"final output shape :{x.shape}"
    return x

  def _make_layer(self,
                  block,
                  out_channels,
                  num_blocks,
                  stride):
    strides = [stride] + [1]*(num_blocks-1)
    module = []
    for i in range(num_blocks):
      module.append(block(self.in_channels , out_channels , stride = strides[i] , r=16))
      self.in_channels = block.expansion*out_channels
    return nn.Sequential(*module)
   

def resnet18(num_classes=10):
  return SEResNet(BaseResidualSEBlock , [2,2,2,2] , num_classes = num_classes)
def resnet34(num_classes=10):
  return SEResNet(BaseResidualSEBlock , [3,4,6,3] , num_classes = num_classes)
def resnet50(num_classes=10):
  return SEResNet(BottelneckResidualSEBlock , [3,4,6,3] , num_classes = num_classes)
def resnet101(num_classes=10):
  return SEResNet(BottelneckResidualSEBlock , [3,4,23,3] , num_classes = num_classes)
def resnet152(num_classes=10):
  return SEResNet(BottelneckResidualSEBlock , [3,8,36,3] , num_classes = num_classes)




  