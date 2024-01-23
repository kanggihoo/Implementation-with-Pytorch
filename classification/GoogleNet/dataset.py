import torch 
from torch.utils.data import Dataset , DataLoader
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

class CatDogDataset(Dataset):
    def __init__(self ,home_dir, images , transform = None):
        self.images = images
        self.transform = transform
        self.class_to_index = {"cat" : 0 ,"dog" : 1}
        self.home_dir = Path(home_dir)
    def __len__(self):
        return len(self.images)
    def __getitem__(self , idx):
        img = Image.open(self.home_dir.joinpath(self.images[idx]))
        img = np.array(img)
        
        if self.transform is not None:
            ag = self.transform(image=img)
            img = ag["image"]
        label = self.class_to_index[self.images[idx].split(".")[0]]
        label = torch.tensor(label , dtype = torch.float)
        return img , label
    
def get_dataloader(home_dir , train_image , val_image , batch_size , train_transform = None , val_transform = None):
    train_dataset = CatDogDataset(home_dir , train_image , train_transform)
    val_dataset = CatDogDataset(home_dir , val_image , val_transform)
    train_loader = DataLoader(train_dataset , batch_size = batch_size , shuffle=True )
    test_loader = DataLoader(val_dataset , batch_size=batch_size , shuffle=False)
    return train_loader , test_loader
