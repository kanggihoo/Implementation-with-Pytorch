from dataset import VAEDataset as VAEDataset
from model import VAE
from utils import download_data

import os
import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--config" , "-c" ,
                    dest = "filename",
                    default = "config.yaml")

args = parser.parse_args()

download_data("https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz" , remove=True)

with open(args.filename,"r") as f:
  config = yaml.safe_load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(config["model_params"]["in_channels"] , config["model_params"]["latent_dim"]).to(device)
data = VAEDataset(**config["data_params"])

data.setup()
train_dataloader = data.train_dataloader()
val_dataloader = data.val_dataloader()

## Train
optimizer = optim.Adam(model.parameters() , lr = config["exp_params"]["LR"])
for epoch in range(config["trainer_params"]["max_epochs"]):
  with tqdm(train_dataloader,total=len(train_dataloader) ,desc="training") as pbar:
    for img,_ in train_dataloader:
      img = img.to(device)
      # label = label.to(device)
      result = model(img)
      losses = model.loss_function(*result , M_N = 0.00025)
      loss = losses["loss"]
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      pbar.set_postfix(loss=loss.item())
      pbar.update(1)
    print(
        f"[Epoch {epoch+1}/{config['trainer_params']['max_epochs']}] "
        f"[loss : {loss.item():.4f}]"
        f"[RC loss : {losses['Reconstruction_Loss']:.4f}] "
        f"[KL loss : {losses['KLD']:.4f}] "
    )
  with torch.no_grad():
    test_img , _ = next(iter(val_dataloader))
    output = model.generate(test_img.to(device))
    if not Path("output").exists():
      Path(r"output").mkdir(parents=True, exist_ok=True)

    save_image(torch.cat((test_img[:25].data , output[:25].cpu().data ) , axis = 0)  ,
               f"output/Reconstruction_{epoch}.png" , nrow=5, normalize = True)

    samples = model.sample(25 ,device)
    save_image(samples , f"output/samples_{epoch}.png" , nrow=5, normalize = True)



