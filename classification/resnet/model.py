import torch 
import torch.nn as nn
import torch.nn.functional as F

from typing import Union  


class ConvBlock(nn.Module):
  '''
  base conv block : Conv => BN => ReLU
  '''
  def __init__(self, in_channels , out_channels , kernel_size , stride , padding = 0, bias = False, active = True , **kwargs) -> None:
    super().__init__()
    module = nn.ModuleList(
        [
          nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.BatchNorm2d(out_channels),    
        ]
    )
    if active:
      module.append(nn.ReLU(inplace=True))

    self.sequence = nn.Sequential(*module)

  def forward(self ,x):
    return self.sequence(x)

class BaseBlock(nn.Module):
  '''
  base block for resnet 18,34 layers
  '''
  expansion = 1
  def __init__(self,in_channels , out_channels, stride = 1) -> None:
    super(BaseBlock,self).__init__()

    self.residual = nn.Sequential(
        ConvBlock(in_channels, out_channels , kernel_size = 3 , stride=stride , padding = 1 , bias = False),
        ConvBlock(out_channels, out_channels * BaseBlock.expansion , kernel_size = 3 , stride=1 , padding = 1 , bias = False , active=False),
    )

    # to prevent differences in input dimention and output dimention, use 1*1 conv layer 
    self.shortcut = nn.Sequential()

    if stride != 1 or in_channels != out_channels*BaseBlock.expansion:
      self.shortcut = nn.Sequential(
          ConvBlock(in_channels, out_channels*BaseBlock.expansion , stride = stride , kernel_size=1 , bias = False , active=False),
      )
  def forward(self , x):
    return F.relu(self.residual(x) +self.shortcut(x))


class Bottleneck(nn.Module):
  '''
  Residual block for resnet over 50 layers
  '''
  expansion = 4
  def __init__(self, in_channels:int , out_channels:int , stride:int = 1) -> None:
    super(Bottleneck,self).__init__()

    self.residual = nn.Sequential(
        ConvBlock(in_channels , out_channels , kernel_size= 1, stride = 1 , padding = 0 , bias = False ),
        ConvBlock(out_channels , out_channels , kernel_size =3 , stride = stride , padding = 1 ,  bais=False ),
        ConvBlock(out_channels , out_channels * Bottleneck.expansion , kernel_size = 1 , stride = 1 , padding = 0 , bias = False , active = False)
    )

    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels *  Bottleneck.expansion:
      self.shortcut = nn.Sequential(
          ConvBlock(in_channels , out_channels * Bottleneck.expansion , kernel_size = 1 , stride = stride , padding = 0 , bias = False , active = False)
      )

  def forward(self, x):
    return F.relu(self.residual(x) + self.shortcut(x))
  

class ResNet(nn.Module):
  """
  두번째 conv2 연산을 제외한 나머지 conv 연산에서는 downsample 이 적용되어 총 4번의 downsample을 통해 입력이미지의 size / 2^16 으로 출력 feature map size를 줄임
  마지막에 AdaptiveAvgPool2d 연산을 통해 feature map size를 (1,1)으로 만든 후 flatten 시킨 뒤 클래스 개수에 맞게 linear layer를 통해 분류
  """
  def __init__(self, block , num_block , num_classes=100) -> None:
    super().__init__()
    # input image size = 224
    self.in_channels = 64
    self.conv1 = ConvBlock(3, self.in_channels, kernel_size=7, stride=2, padding=1, bias=False) # (3,224,224) = > #(64,112,112)
    self.conv2 = self._make_layers(block, 64, num_block[0], stride=1) 
    self.conv3 = self._make_layers(block, 128, num_block[1], stride=2)
    self.conv4 = self._make_layers(block, 256, num_block[2], stride=2)
    self.conv5 = self._make_layers(block, 512, num_block[3], stride=2)
    self.avg_pool = nn.AdaptiveAvgPool2d((1,1)) # ()
    self.fc = nn.Linear(512 * block.expansion, num_classes)
  
  def _make_layers(self,block:Union[BaseBlock , Bottleneck],
                  out_channels : int,
                  num_blocks : int,
                  stride : int
                  )->nn.Sequential:
      strides = [stride] + [1]*(num_blocks-1) # 각각의 
      layers = []
      for stride in strides:
        layers.append(block(self.in_channels, out_channels, stride)) 
        self.in_channels = block.expansion * out_channels
      
      return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.avg_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
    
def resnet18(num_class=100):
  return ResNet(BaseBlock , [2,2,2,2] , num_class)

def resnet34(num_class=100):
  return ResNet(BaseBlock , [3,4,6,3],num_class)

def resnet50(num_class=100):
  return ResNet(Bottleneck , [3,4,6,3],num_class)

def resnet101(num_class=100):
  return ResNet(Bottleneck , [3,4,23,3],num_class)

def resnet152(num_class=100):
  return ResNet(Bottleneck , [3,8,36,3],num_class)
