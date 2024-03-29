import unittest
from model import VAE
import torch

class TestVAE(unittest.TestCase):
  def setUp(self):
    self.model = VAE(3,128)
    self.latent_dim = 128
    self.batch_size = 8
    self.channels = 3
    self.img_size = (self.batch_size , self.channels ,64,64)

  def test_encode(self):
    mu,logvar = self.model.encode(torch.randn(self.img_size))
    self.assertEqual(mu.shape ,(self.batch_size , self.latent_dim))
    self.assertEqual(logvar.shape , (self.batch_size , self.latent_dim))

  def test_decode(self):
    output = self.model.decode(torch.randn(self.batch_size,self.latent_dim))
    self.assertEqual(output.shape , (self.batch_size , 3, 64, 64))

  def test_reparam(self):
    z = self.model.reparameterize(torch.randn(self.batch_size,self.latent_dim),torch.randn(self.batch_size,self.latent_dim))
    self.assertEqual(z.shape , (self.batch_size , self.latent_dim))

  def test_forward(self):
    x = torch.randn(self.img_size)
    output,input , mu,log_var = self.model(x)
    self.assertEqual(output.shape , x.shape)
    self.assertEqual(input.shape , x.shape)
    self.assertEqual(mu.shape, (self.batch_size , self.latent_dim))
    self.assertEqual(log_var.shape, (self.batch_size , self.latent_dim))

  def test_loss(self):
    x = torch.randn(self.img_size)
    result = self.model(x)
    loss = self.model.loss_function(*result , M_N = 0.005)
    print(loss)

if __name__ =="__main__":
  unittest.main()


