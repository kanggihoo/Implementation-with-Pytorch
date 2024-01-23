import torch
from pathlib import Path
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
class UNetDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir="data/train" , mask_dir="data/train_masks", transforms=None):
        self.img_dir = Path(__file__).parent.joinpath(img_dir)
        self.mask_dir = Path(__file__).parent.joinpath(mask_dir)
        self.transforms = transforms
        self.imgs = os.listdir(self.img_dir)
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self , idx):        
        img_file = self.img_dir.joinpath(self.imgs[idx])
        mask_file = self.mask_dir.joinpath(self.imgs[idx].replace(".jpg" , "_mask.gif"))
        img = np.array(Image.open(img_file).convert("RGB"))
        mask = np.array(Image.open(mask_file.convert("L")) , dtype=np.float32) # => gray sclae convert and then convert np.ndarray => 0~255.0 
        
        # mask값이 0또는 1의 값을 가지도록 변경
        mask[mask==255.0] = 1.0
        
        if self.transforms is not None: # use albumentations
            albumentations = self.transforms(image = img , mask = mask)
            img,mask = albumentations['image'] , albumentations['mask']
            
            
        return img , mask


def test_creatdataset():
    home_dir = Path(__file__).parent
    img_dir = home_dir / "data" / "train"
    mask_dir = home_dir / "data" / "train_masks"
    a = UNetDataset()
    return a

def plot_test(img , mask):
    fig , ax = plt.subplots(1,2)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    ax[0].set_title("img")
    ax[1].set_title("mask img")
    plt.show()
    

if __name__ == "__main__":
    dataset = test_creatdataset()
    img , mask = dataset[0]
    print(img.shape , mask.shape)
    plot_test(img , mask)
    print(Path(__file__).parent.joinpath("./data/train"))
    


