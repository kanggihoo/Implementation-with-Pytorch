from model import *
import torch
import unittest

class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.input_shape = (32,3,224,224)
        self.data = torch.randn(self.input_shape)
        self.num_classes = 10
        self.output_shape = (self.data.size(0) , self.num_classes)
        
    def test_resnet18(self):
        model = resnet18(self.num_classes)
        output = model(self.data)
        self.assertEqual(output.shape , self.output_shape)
    
    def test_resnet34(self):
        model = resnet34(self.num_classes)
        output = model(self.data)
        self.assertEqual(output.shape , self.output_shape)
    
    def test_resnet50(self):
        model = resnet50(self.num_classes)
        output = model(self.data)
        self.assertEqual(output.shape , self.output_shape)
        
    def test_resnet101(self):
        model = resnet101(self.num_classes)
        output = model(self.data)
        self.assertEqual(output.shape , self.output_shape)
    
    def test_resnet152(self):
        model = resnet152(self.num_classes)
        output = model(self.data)
        self.assertEqual(output.shape , self.output_shape)

if __name__ =="__main__":
    unittest.main()