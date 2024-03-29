from torch.utils.data import DataLoader
from dataset import BERTDataset 
import torch
from language_model import BERTLM
from model import BERT
from train import BERTTrainer
from dataset import GetDataset
from tokenizer import Tokenizer
import yaml

import yaml

with open("config.yaml" , "r") as f:
    config = yaml.safe_load(f)


g = GetDataset(config["data_params"]["MAX_LEN"])
pairs = g.build()
t = Tokenizer(pairs)
tokenizer = t.train_tokenizer()


print("vocab_size : ",len(tokenizer.vocab)) 
print("number of train_data : ",len(pairs))

train_data = BERTDataset(pairs , seq_len = config["data_params"]["MAX_LEN"] , tokenizer = tokenizer)
train_loader = DataLoader(
    train_data , batch_size = config["data_params"]["batch_size"] , shuffle = True , pin_memory = True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model = BERT(
    vocab_size = len(tokenizer.vocab),
    embed_dim = 768,
    num_layers = 2,
    num_heads = 12,
    dropout = 0.1
)


bert_lm = BERTLM(
    bert = bert_model,
    vocab_size = len(tokenizer.vocab),
).to(device)


bert_trainer = BERTTrainer(
    model = bert_lm,
    train_dataloader = train_loader,
    device = device
)

epochs = 20
for epoch in range(epochs):
  bert_trainer.train(epoch)