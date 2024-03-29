# %%writefile model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from typing import Union

## BERT
class BERT(nn.Module):
  def __init__(self ,
               vocab_size:int,
               embed_dim:int=768,
               num_layers:int=12,
               num_heads:int=12,
               dropout:float=0.1,
               ):
    super(BERT , self).__init__()
    self.num_layers = num_layers
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.d_ff = 4*embed_dim

    self.embedding = BertEmbedding(vocab_size , embed_dim , dropout)
    self.transformer_blocks = nn.ModuleList([
        TransformerBlock(self.embed_dim , self.num_heads , self.d_ff,dropout) for _ in range(self.num_layers)
        ])

  def forward(self , input , segment_label):
    """
    input : (B,S)
    segment_label = (B,S)
    output : (B,S,E)
    """

    mask = (input>0).unsqueeze(1).repeat(1,input.size(1),1).unsqueeze(1) # (B,S) => (B,1,S) =>(B,S,S) => (B,1,S,S)
    x = self.embedding(input ,segment_label)
    for transformer in self.transformer_blocks:
      x = transformer(x,mask)
    return x

## Transformer Block
class TransformerBlock(nn.Module):
  def __init__(self ,
               embed_dim:int,
               num_heads:int,
               d_ff : int,
               dropout:float=0.1,
               ):
    super(TransformerBlock , self).__init__()
    self.mha = MultiHeadAttention(num_heads , embed_dim)
    self.ffn = FeedForwardBlock(embed_dim , d_ff)
    self.residual1 = ResidualBlock(embed_dim,dropout)
    self.residual2 = ResidualBlock(embed_dim,dropout)
    self.dropout = nn.Dropout(dropout)
  def forward(self , x,mask):
    """
    input
      x: (B,S,E)
      mask : (B,1,S,S)
    output : (B,S,E)

    """
    x = self.residual1(x , lambda _x : self.mha(_x, mask))
    x = self.residual2(x , self.ffn)
    return self.dropout(x)

class Attention(nn.Module):
  def __init__(self):
    super(Attention , self).__init__()

  def forward(self , q,k,v, mask=None , dropout:Union[torch.nn.Module,None]=None):
    # q,k,v = (B,h,seq_len , num_head)
    attn = q@k.transpose(-1,-2) / np.sqrt(q.size(-1)) # (B,h,seq_len , seq_len)
    if mask is not None:
      attn.masked_fill_(mask ==0 , -torch.inf)

    attn = F.softmax(attn , dim=-1)

    if dropout is not None:
      attn = dropout(attn)
    return attn@v , attn # (B,h,S,S) , (B,h,S,S)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads:int , embed_dim:int, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.linear_layer = nn.Linear(embed_dim , 3*embed_dim)
        self.output_linear = nn.Linear(embed_dim , embed_dim)
        self.attention = Attention()
        self.dropout = nn.Dropout(dropout)

    def forward(self , x, mask):
      """
      input:
        x : (B,S,E)
        mask : (B,1,S,S)

      output: (B,S,E)
      """
      batch_size , seq_len , embed_dim = x.size()

      q,k,v = self.linear_layer(x).chunk(3,dim=-1) # B,S,E
      q,k,v = map(lambda t: rearrange(t, 'b s (h d_k) -> b h s d_k', h =self.num_heads , d_k =self.head_dim ), [q,k,v]) # (B,S,E) => (B,num_head , S , head_dim)
      assert q.size() == k.size() == v.size() == (batch_size , self.num_heads , seq_len , self.head_dim) , f"Shape error! {q.shape} must (B,h,S,head_dim)"
      x , attn = self.attention(q,k,v,mask=mask,dropout=self.dropout)
      x = rearrange(x, 'b h s d -> b s (h d)')
      x = self.output_linear(x)
      return x

class FeedForwardBlock(nn.Module):
  def __init__(self , d_model , d_ff , dropout =0.1):
    super(FeedForwardBlock , self).__init__()
    self.linear1 = nn.Linear(d_model , d_ff)
    self.linear2 = nn.Linear(d_ff , d_model)
    self.dropout = nn.Dropout(dropout)
  def forward(self , x):
    """
    input : (B,S,E)
    outut : (B,S,E)
    """
    x = self.dropout(F.gelu(self.linear1(x)))
    x = self.linear2(x)
    return x

# class LayerNorm(nn.Module):
#   def __init__(self , d_model , eps=1e-5):
#     super(LayerNorm , self).__init__()
#     self.alpha = nn.Parameter(torch.ones(d_model))
#     self.beta = nn.Parameter(torch.zeros(d_model))
#     self.eps = eps
  # def forward(self , x):
  #   # x = (B,S,E)
  #   mean = x.mean(dim=-1 , keepdim= True) # (B,S,1)
  #   std = x.std(dim=-1 , keepdim= True) # # (B,S,1)
  #   x = (x - mean) / (std+self.eps)
  #   return x*self.alpha + self.beta

class ResidualBlock(nn.Module):
  def __init__(self , d_model , dropout = 0.1):
    super(ResidualBlock , self).__init__()
    self.layer_norm = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)
  def forward(self , x , sublayer):
    """
    input :
      x : (B,S,E)
      sublayer : MultiHeadAttention or FeedForwardBlock
    output : (B,S,E)
    """
    return x + self.dropout(sublayer(self.layer_norm(x)))


# Embedding Module
###############################################################################################################
class BertEmbedding(nn.Module):
  """
  1. PositionalEmbedding
  2. SegmentEmbedding : adding sentence segment info ([SEG])
  3. TokenEmbedding : normal embedding matrix
  sumation of all these embeddings are output of BertEmbedding
  """
  def __init__(self , vocab_size , embed_dim:int = 768 , dropout=0.1):
    super(BertEmbedding , self).__init__()
    self.token = TokenEmbedding(vocab_size , embed_dim)
    self.segment = SegmentEmbedding(embed_dim)
    self.position = PositionalEmbedding(embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self , input , segment_label):
    output= self.token(input) + self.position(input) + self.segment(segment_label)
    return self.dropout(output)


class PositionalEmbedding(nn.Module):
  def __init__(self , embedding_dim=768 , max_len = 64):
    super(PositionalEmbedding , self).__init__()
    pe = torch.zeros(max_len , embedding_dim).float().requires_grad_(False) # (max_len , embedding_dim)
    pos = torch.arange(0,max_len).float().unsqueeze(1) # (max_len , 1)
    div_term = (torch.arange(0,embedding_dim,2) *(-np.log(10000)/embedding_dim)).exp()
    pe[:,::2] = torch.sin(pos*div_term)
    pe[:,1::2] = torch.cos(pos*div_term)
    pe = pe.unsqueeze(0) # (1,max_len, embedding_dim)
    self.register_buffer("pe" , pe)
  def forward(self , x):
    return self.pe[:,:x.size(1)]

# segment
class SegmentEmbedding(nn.Embedding): # segment embedding은 첫번째 문장은 1로 , 두번째 문장은 2로 임베딩 되므로 vocab size =3
  def __init__(self , embed_dim:int = 768):
    super(SegmentEmbedding , self).__init__(num_embeddings=3 , embedding_dim=embed_dim , padding_idx=0)

# token
class TokenEmbedding(nn.Embedding):
  def __init__(self ,vocab_size,  embed_dim:int = 768):
    super(TokenEmbedding , self).__init__(num_embeddings=vocab_size , embedding_dim=embed_dim , padding_idx=0)


#############################################################################################################    



