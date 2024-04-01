import unittest
from model import *
import torch

class test_model(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 1
        self.input_shape = (self.batch_size,3,224,224)
        self.data = torch.randn(self.input_shape)
        self.num_class = 30
        self.output_shape = (self.batch_size,self.num_class)
    def test_resnet18(self):
        res18 = resnet18(self.num_class)
        self.assertEqual(res18(self.data).shape , self.output_shape)
    
    def test_resnet34(self):
        res34 = resnet34(self.num_class)
        self.assertEqual(res34(self.data).shape , self.output_shape)
        
    def test_resnet50(self):
        res50 = resnet50(self.num_class)
        self.assertEqual(res50(self.data).shape , self.output_shape)
    
    def test_resnet101(self):
        res101 = resnet101(self.num_class)
        self.assertEqual(res101(self.data).shape , self.output_shape)
    def test_resnet152(self):
        res152 = resnet152(self.num_class)
        self.assertEqual(res152(self.data).shape , self.output_shape)
    
if __name__ =="__main__":
    unittest.main()

