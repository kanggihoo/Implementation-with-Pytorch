import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNet
from dataset import UNetDataset
import os
from pathlib import Path
from tqdm import tqdm
from utils import (
    get_loader,
    
)
#======================#
#    Hyper Parmeter    #
#======================#
LEANING_RATE = 1e-4
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 10
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
TRAIN_IMG_DIR  =  "data/train"
TRAIN_MASK_DIR  = "data/train_masks"
VAL_IMG_DIR = "data/val"
VAL_MASK_DIR = "data/val_masks"

LOAD_MODEL = False


def train_fn(Loader , model , loss_fn , optimizer , device ):
    
    for batch , (img , mask) in tqdm(enumerate(Loader)):
        img = img.to(device)
        mask = mask.to(device)
        
        pred = model(img)
        loss = loss_fn(pred , loss)
        optimizer.zero_grad()
        loss.backward()
        loss.step()

def main():
    train_transform = A.Compose(
        [
            A.Resize(IMAGE_HEIGHT , IMAGE_WIDTH),
            A.Rotate(limit= 35 , p=0.2),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(
                mean = [0.0,0.0,0.0],
                std = [1.0,1.0,1.0],
                max_pixel_value=255
            ),
            ToTensorV2()
        ]
    )
    
    val_transform =  A.Compose(
        [
            A.Resize(IMAGE_HEIGHT , IMAGE_WIDTH),
            A.Normalize(
                mean = [0.0,0.0,0.0],
                std = [1.0,1.0,1.0],
                max_pixel_value=255
            ),
            ToTensorV2()
        ]
    )
    
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters() , lr = LEANING_RATE)
    
    train_loader , val_loader = get_loader(TRAIN_IMG_DIR ,
                                           TRAIN_MASK_DIR,
                                           VAL_IMG_DIR,
                                           VAL_MASK_DIR,
                                           BATCH_SIZE,
                                           train_transform,
                                           val_transform)
        
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader , model , loss_fn , optimizer , DEVICE)



A.Rotate()