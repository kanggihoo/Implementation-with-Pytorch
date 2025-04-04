import torch
import torch.nn as nn
class BatchNorm(nn.Module):
    def __init__(self , channels:int ,
                 eps: float = 1e-5,
                 momentum : float = 0.1,
                 affine:bool = True,
                 track_running_stats : bool = True
                 ):
        super(BatchNorm,self).__init__()
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.scale = nn.Parameter(torch.ones(self.channels))
            self.shift = nn.Parameter(torch.ones(self.channels))
        if self.track_running_stats:
            self.register_buffer("exp_mean" , torch.zeros(self.channels))
            self.register_buffer("exp_var" , torch.ones(self.channels))    
        
    def forward(self , x:torch.Tensor):
        x_shape = x.shape
        batch_size = x.size(0)
        assert self.channels == x_shape[1]
        
        x = x.view(batch_size , self.channels , -1) # (B,C,*)
        
        if self.training or not self.track_running_stats:
            mean = x.mean(dim = [0,2]) # (B,C,*) => (C)
            mean_2 = (x**2).mean(dim=[0,2])
            var = mean_2 - mean**2 # (B,C,*) => (C)

            if self.training and self.track_running_stats:
                self.exp_mean = (1-self.momentum)*self.exp_mean + self.momentum*mean
                self.exp_var = (1-self.momentum)*self.exp_var + self.momentum*var
        
        else:
            mean = self.exp_mean
            var = self.exp_var
        
        x_norm = (x - mean.view(1,-1,1)) / torch.sqrt(var + self.eps).view(1,-1,1)
        
        
        if self.affine:
            x_norm = x_norm*self.scale + self.shift
        
        return x_norm.view(x_shape)
    
            
            
        

        
        