import torch 
import torch.nn as nn
from model import BERT

class BERTLM(nn.Module):
  def __init__(self, bert:BERT, vocab_size ) -> None:
    super(BERTLM , self).__init__()
    self.bert = bert
    self.next_sentence = NextSentencePrediction(bert.embed_dim)
    self.mask_language = MaskedLanguageModeling(bert.embed_dim , vocab_size)

  def forward(self , x, segment_label):
    output = self.bert(x, segment_label)
    return self.next_sentence(output) , self.mask_language(output)


class NextSentencePrediction(torch.nn.Module):
  # Using Only [CLS] token to predict the sentence is next or not next (if next : predict 1 , if not : predict 0)
  def __init__(self, hidden) -> None:
     super(NextSentencePrediction , self).__init__()
     self.linear = nn.Linear(hidden,2)
     self.softmax = nn.LogSoftmax(dim = -1)
  def forward(self , x):
    # each [CLS] is located in first dimention of S
    CLS = x[:,0,:] # (B,E)
    return self.softmax(self.linear(CLS)) # (B,E) = > (B,2)

class MaskedLanguageModeling(torch.nn.Module):
  """
  predicting origin token from maked input sentence
  """
  def __init__(self, hidden, vocab_size) -> None:
    super(MaskedLanguageModeling , self).__init__()
    self.liear = nn.Linear(hidden , vocab_size)
    self.softmax = nn.LogSoftmax(dim=-1)
  def forward(self , x):
    # (B,S,E) => (B,S,Vocab_Size)
    return self.softmax(self.liear(x))




