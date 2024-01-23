import torch
from dataset import UNetDataset
from torch.utils.data import DataLoader
from pathlib import Path

def get_loader(train_img_dir,
               train_mask_dir,
               val_img_dir,
               val_mask_dir,
               batch_size,
               train_transform,
               val_transform,
               ):
    """return train , val dataloder
    Args:
        train_img_dir (_type_): 
        train_mask_dir (_type_):
        val_image_dir (_type_): 
        val_mask_dir (_type_): 
        batch_size (_type_): 
        train_transform (_type_): 
        val_transform (_type_): 
    """
    HOME_DIR = Path(__file__).parent
    train_dataset = UNetDataset(train_img_dir , train_mask_dir, train_transform)
    val_dataset = UNetDataset(val_img_dir , val_mask_dir , val_transform)
    train_loader = DataLoader(train_dataset , batch_size , shuffle=True )
    val_loader = DataLoader(val_dataset , batch_size , shuffle=False )
    return train_loader , val_loader

def save_checkpoint(state , filename):
    print("=> Saving checkpoint")
    torch.save(state , filename)
def load_checkpoint(checkpoint , model):
    model.load_state_dict(checkpoint["state_dict"])
    
def check_accuracy(loader , model , device):
    num_correct = 0
    num_pixels = 0
    model.eval()
    with torch.inference_mode():
        for x ,y in loader:
            x = x.to(device) 
            y = y.to(device)
            pred = torch.sigmoid(model(x))
            
            torch.equal(torch.round(pred) , y)
            