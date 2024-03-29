import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import torch

class OxfordPets(Dataset):
  def __init__(self ,
               data_path : str,
               transform = None,
               split :str = "train",
               **kwargs):
    self.data_path = Path(data_path)
    self.transform = transform
    imges = [img for img in self.data_path.rglob("*.jpg")]
    self.imges = imges[:int(len(imges)*0.75)] if split =="train" else imges[int(len(imges)*0.75):]
    # self.imges = sorted(imges , key = lambda x : (x.stem , int(x.stem.split("_")[-1])))
    # imges = sorted([img for img in self.data_path.iterdir() if img.suffix ==".jpg"] , key = lambda x : (x.stem , int(x.stem.split("_")[-1])))
    self.labels = set(["_".join(i.stem.split("_")[:-1]) for i in self.imges])
    self.class_to_idx = {c : idx for idx, c in enumerate(self.labels)}
    self.idx_to_class = {v:k for k , v  in self.class_to_idx.items()}
  def __len__(self):
    return len(self.imges)
  def __getitem__(self, idx) :
    file_name = self.imges[idx]
    label = "_".join(file_name.stem.split("_")[:-1])
    img = Image.open(file_name)
    img = img.convert("RGB")
    label = torch.tensor(self.class_to_idx[label], dtype= torch.long)
    if self.transform is not None:
      img = self.transform(img)

    return img , label

class VAEDataset():
  def __init__(self ,
               data_path : str,
               train_batch_size:int,
               val_batch_size:int,
               img_size : int,
               **kwargs ):
    self.data_path = data_path
    self.train_batch_size = train_batch_size
    self.val_batch_size = val_batch_size
    self.img_size = img_size
  def setup(self):
    transform = transforms.Compose([
          transforms.Resize((self.img_size,self.img_size)),
          transforms.ToTensor(),
          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    self.train_dataset = OxfordPets(self.data_path , transform=transform , split="train")
    self.val_dataset = OxfordPets(self.data_path , transform=transform , split="val")

  def train_dataloader(self) -> DataLoader:
    return DataLoader(self.train_dataset , batch_size=self.train_batch_size , shuffle=True)
  def val_dataloader(self) -> DataLoader:
    return DataLoader(self.val_dataset , batch_size=self.val_batch_size , shuffle=False)
