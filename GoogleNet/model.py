import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self , in_channels , ch1x1 , ch3x3_reduct , ch3x3 , ch5x5_reduct , ch5x5 , pool ):
        super(InceptionModule , self).__init__()
        
        self.conv1x1 = nn.Conv2d(in_channels , ch1x1 , kernel_size=1)
        self.conv3x3 = nn.Sequential(BaseConv2d(in_channels , ch3x3_reduct , kernel_size=1),
                                     BaseConv2d(ch3x3_reduct,ch3x3 , kernel_size=3,padding=1))
        self.conv5x5 = nn.Sequential(BaseConv2d(in_channels , ch5x5_reduct , kernel_size=1),
                                     BaseConv2d(ch5x5_reduct , ch5x5 , kernel_size=5 , padding=2))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3,padding=1,stride=1),
                                  nn.Conv2d(in_channels , pool , kernel_size=1))
    def forward(self ,x ):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x4 = self.pool(x)
        
        return torch.cat([x1,x2,x3,x4] , dim=1)
    

class BaseConv2d(nn.Module):
    def __init__(self , in_channels , out_channels  , **kwards):
        super(BaseConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels , out_channels  , **kwards)
        self.relu = nn.ReLU(inplace=True)
    def forward(self , x):
        return self.relu(self.conv2d(x))

class AuxMoudle(nn.Module):
    def __init__(self , in_channels , num_classes):
        super(AuxMoudle,self).__init__()
        self.aux = nn.Sequential(nn.AvgPool2d(kernel_size=5 , stride=3),
                                 BaseConv2d(in_channels,128,kernel_size = 1),
                                 nn.Flatten(start_dim=1),
                                 nn.Linear(4*4*128,1024),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.7),
                                 nn.Linear(1024,num_classes))
    def forward(self, x):
        return self.aux(x)

# image size =(3x224x224) 
class GoogleNet(nn.Module):
    def __init__(self, in_channels , num_classes):
        super().__init__()
        self.training =True
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(BaseConv2d(3,64,kernel_size = 7,stride=2,padding=3),
                                 nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
                                 nn.LocalResponseNorm(2))
        self.conv2 = nn.Sequential(BaseConv2d(64,192,kernel_size=1),
                                 BaseConv2d(192,192,kernel_size=3,padding=1),
                                 nn.LocalResponseNorm(2),
                                 nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        self.inception3_a = InceptionModule(192,64,96,128,16,32,32)
        self.inception3_b = InceptionModule(256,128,128,192,32,96,64)
        self.aux1 = AuxMoudle(512,self.num_classes)
        self.inception4_a = InceptionModule(480,192,96,208,16,48,64)
        self.inception4_b = InceptionModule(512,160,112,224,24,64,64)
        self.inception4_c = InceptionModule(512,128,128,256,24,64,64)
        self.aux2 = AuxMoudle(528,self.num_classes)
        self.inception4_d = InceptionModule(512,112,144,288,32,64,64)
        self.inception4_e = InceptionModule(528,256,160,320,32,128,128)
        self.inception5_a = InceptionModule(832,256,160,320,32,128,128)
        self.inception5_b = InceptionModule(832,384,192,384,48,128,128)

        self.maxpool = nn.MaxPool2d(3,stride=2,padding=1)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Sequential(nn.Flatten(start_dim=1),
                                nn.Dropout(p=0.4),
                                nn.Linear(1024,num_classes))
    
    def forward(self , x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3_a(x)
        x = self.inception3_b(x)
        x = self.maxpool(x)
        x = self.inception4_a(x)
        x = self.inception4_b(x)
        
        if self.training:
            out1 = self.aux1(x)
        x = self.inception4_c(x)
        x = self.inception4_d(x)
        if self.training:
            out2 = self.aux2(x)
        x = self.inception4_e(x)
        x = self.maxpool(x)
        x = self.inception5_a(x)
        x = self.inception5_b(x)
        x = self.avgpool(x)
        x = self.fc(x)
        if self.training:
            return [x , out1 , out2]
        else:
            return x
    
    def set_train(self):
        self.training = True
    def set_eval(self):
        self.training = False



    

def test_model():
    model = GoogleNet(3,100)
    model.set_train()
    test_data = torch.randn(16,3,224,224)
    out1 , out2 , out3 = model(test_data)
    print(out1.shape ,out2.shape , out3.shape)
    model.set_val()
    pred = model(test_data)
    print(pred.shape)
    
    
    
if __name__ =="__main__":
    test_model()