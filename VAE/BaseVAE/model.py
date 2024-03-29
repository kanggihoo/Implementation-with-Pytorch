import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List , Any, Union
from abc import abstractmethod

class BaseVAE(nn.Module):
  def __init__(self) ->None:
    super().__init__()
  def encode(self , input:torch.Tensor) -> List[torch.Tensor]:
    raise NotImplementedError
  def decode(self , input:torch.Tensor) -> List[torch.Tensor]:
    raise NotImplementedError
  def sample(self , batch_size:int , device:int , **kwargs) ->torch.Tensor:
    raise NotImplementedError
  def generate(self , x:torch.Tensor , **kwargs) -> torch.Tensor:
    raise NotImplementedError
  @abstractmethod
  def forward(self ,*inputs:torch.Tensor)->torch.Tensor:
    pass
  @abstractmethod
  def loss_function(self ,*inputs:Any , **kwargs)->torch.Tensor:
    pass

class VAE(BaseVAE):
  def __init__(self ,
               in_channels :int,
               latent_dim : int,
               hidden_dims : Union[List, None]=None,
               **kwargs) -> None:
      super(VAE, self).__init__()
      self.latent_dim = latent_dim
      modules = []
      if hidden_dims is None:
        hidden_dims = [32,64,128,256,512]
      # ----------
      # encoder
      # ----------
      for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels , h_dim , 3,2,1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
            )
        )
        in_channels = h_dim

      self.encoder = nn.Sequential(*modules)
      self.fc_mu = nn.Linear(hidden_dims[-1]*4 , latent_dim)
      self.fc_logvar = nn.Linear(hidden_dims[-1]*4 , latent_dim)

      # ----------
      # decoder
      # ----------

      modules = []
      self.decoder_input = nn.Linear(latent_dim , hidden_dims[-1]*4)

      hidden_dims = hidden_dims[::-1]
      for idx in range(len(hidden_dims)-1):
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[idx] , hidden_dims[idx+1],3,2,1,output_padding=1),
                nn.BatchNorm2d(hidden_dims[idx+1]),
                nn.LeakyReLU(),
            )
        )
      self.decoder = nn.Sequential(*modules)

      self.final_layer = nn.Sequential(
          nn.ConvTranspose2d(hidden_dims[-1] , hidden_dims[-1] , 3,2,1,output_padding=1),
          nn.BatchNorm2d(hidden_dims[-1]),
          nn.LeakyReLU(),
          nn.Conv2d(hidden_dims[-1] , out_channels=3 , kernel_size=3,stride=1,padding=1),
          nn.Tanh()
      )

  def encode(self , input:torch.Tensor) -> List[torch.Tensor]:
    """
    encode the input tensor into latent codes
    input : (B,C,H,W)
    """
    assert input.shape[1] == 3 , f"{input.shape[1]} , {input.shape}"
    output = self.encoder(input) # (B,C,H,W) => (B,last_hidden_dim , 2,2)
    assert output.shape[1] == 512 and output.shape[2] == 2 , f"{output.shape[1]} , {output.shape}"
    output = torch.flatten(output , start_dim=1) # (B,last_hidden_dim , 2,2) => (B,last_hidden_dim*4)
    assert output.shape[1] == 512*4 , f"{output.shape[1]} , {output.shape}"
    mu = self.fc_mu(output)  # (B,last_hidden_dim*4) => (B,last_hidden_dim)
    log_var = self.fc_logvar(output) # (B,last_hidden_dim*4) => (B,last_hidden_dim)
    assert mu.shape == log_var.shape == (input.shape[0] , self.latent_dim) , f"{mu.shape} , {log_var.shape} , {input.shape} , {output.shape}"
    return [mu,log_var]

  def decode(self, z):
    """
    decode the latent codes into images
    input : (B,latent_dim)
    output : (B,C,H,W)
    """
    z = self.decoder_input(z) # (B,latent_dim) => (B,last_hidden_dim*4)
    z = z.view(-1,512,2,2) # (B,last_hidden_dim*4) => (B,512,2,2)
    output = self.decoder(z) # (B,512,2,2) => (B,32,H//2,W//2)
    output = self.final_layer(output) #  (B,32,H//2,W//2) => (B,3,H,W)
    return output

  def reparameterize(self , mu:torch.Tensor , logvar:torch.Tensor) -> torch.Tensor:
    """
    mu = (B,latent_dim)
    logvar = (B,latent_dim)
    output : z =(B,latent_dim)
    """
    noise = torch.randn(mu.size() , device = mu.device)
    assert noise.shape == mu.shape , f"{noise.shape} , {mu.shape}"
    std = torch.exp(logvar) **0.5 # => variance = e^log_var , std = variance**0.5
    assert std.shape == mu.shape , f"{std.shape} , {mu.shape}"
    z = mu+(std*noise)
    assert z.shape == mu.shape , f"{z.shape} , {mu.shape}"
    return z

  def forward(self , input:torch.Tensor) -> List[torch.Tensor]:
    mu,log_var = self.encode(input)
    z = self.reparameterize(mu,log_var)
    return [self.decode(z) , input , mu, log_var]

  def loss_function(self, *args , **kwargs ):
    """
    Calulate VAE loss
    """
    recons = args[0] # generated_image
    input = args[1] # input_image
    mu = args[2] # mu
    log_var = args[3] # log_var

    KLD_weight = kwargs["M_N"]
    recons_loss = F.mse_loss(recons , input)
    KLD_loss = torch.mean(-0.5*torch.sum(1 + log_var - mu**2 - torch.exp(log_var),dim=1),dim=0)
    loss = recons_loss + KLD_weight*KLD_loss
    return {"loss":loss , "Reconstruction_Loss" : recons_loss.item() , "KLD":-KLD_loss.item()}

  def sample(self ,
            n_samples:int,
            device,
            **kwargs
            )->torch.Tensor:
    """
    sample from the latent space and return image
    """
    z = torch.randn(n_samples,self.latent_dim)
    z = z.to(device)
    return self.decode(z)

  def generate(self , x:torch.Tensor , **kwargs) -> torch.Tensor:
    """
    Given an input image x , and then return generate image

    """
    return self.forward(x)[0]


