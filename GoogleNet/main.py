import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from dataset import get_dataloader
from model import GoogleNet
from utils import load_checkpoint
# Parameter setting
CLASS_NUM = 1
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 6
BATCH_SIZE = 64
LOAD_MODEL = True

# Setting transform
train_transform = A.Compose( [ 
    A.Resize(224,224),
    A.OneOf([
        A.HorizontalFlip(p=0.3),
        A.Rotate(limit = 20 ,p=0.5 , value = 0)
    ],p=0.5),
    A.Sharpen(p=0.3, lightness=(0.9,1.0)),
    A.Normalize(mean =[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
val_transform = A.Compose( [ 
    A.Resize(240,240),
    A.Normalize(mean =[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# train_loader , val_loader = get_dataloader("./train" , train_img , val_img , BATCH_SIZE , train_transform , val_transform)

# model define
model = GoogleNet(3,CLASS_NUM).to(DEVICE)
# optimizer , loss_fn define
optimizer = torch.optim.Adam(model.parameters() , lr = LEARNING_RATE)
if LOAD_MODEL:
    load_checkpoint("/kaggle/working/model/best.pth",model, optimizer)
loss_fn = torch.nn.BCEWithLogitsLoss()