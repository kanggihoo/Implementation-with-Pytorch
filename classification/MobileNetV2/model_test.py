from model import mobilenetv2
import unittest
import torch 


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_shape = (self.batch_size,3,224,224)
        self.data = torch.randn(self.input_shape).to(self.device)
        self.num_class = 30
        self.output_shape = (self.batch_size,self.num_class)
        
    def test_model(self):
        model = mobilenetv2(self.num_class).to(self.device)
        output = model(self.data)
        self.assertEqual(output.shape , self.output_shape)

if __name__ =="__main__":
    unittest.main()