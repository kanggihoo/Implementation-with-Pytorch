from tqdm.notebook import tqdm
from pathlib import Path
import torch
from utils import save_checkpoint , load_checkpoint

def loss_epoch(loss_fn , preds , y):
    assert len(preds) == 3, f"set model to training mode!"
    loss1 = loss_fn(preds[0].squeeze() , y)
    loss2 = loss_fn(preds[1].squeeze(), y)
    loss3 = loss_fn(preds[2].squeeze() , y)
    return loss1 + 0.3*loss2 + 0.3 *loss3
    
def train_fn(model , loader , optimizer , loss_fn , device):
    model.train()
    model.set_train()
    train_loss = 0
    train_acc = 0
    loop = tqdm(loader ,desc = "Traning" , unit = "iter")
    for x,y in loader :
        x =x.to(device)
        y = y.to(device)
        preds = model(x)    
        loss = loss_epoch(loss_fn , preds , y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # caculate metric(train_acc , loss)
        train_loss+=loss.item()
        train_acc += ((torch.round(torch.sigmoid(preds[0])).squeeze()==y).sum().item() / len(y))
        # loop update
        loop.update(1)
        
    train_loss /= len(loader)
    train_acc /= len(loader)
    loop.close()
    return train_loss , train_acc

def test_fn(model , loader , loss_fn , device):
    model.eval()
    model.set_eval()
    assert model.training == False , f"set model to val_mode the value of model.training : {model.training}"
    test_loss =0
    test_acc = 0
    with torch.inference_mode():
        loop = tqdm(loader ,desc = "Val" , unit = "iter")        
        for x ,y in loader:
            x =x.to(device)
            y = y.to(device)
            pred = model(x)    
            test_loss += loss_fn(pred.squeeze() , y).item()
            test_acc += (torch.round(torch.sigmoid(pred)).squeeze()==y).sum() / len(y)

            # loop update
            loop.update(1)
        loop.close()
        test_loss /= len(loader)
        test_acc /= len(loader)
    return test_loss , test_acc

def train(model,
          train_loader,
          val_loader,
          optimizer,
          loss_fn,
          device,
          epochs):
    best_loss = None
    train_loss_his = []
    train_acc_his = []
    val_loss_his = []
    val_acc_his = []
    for epoch in range(epochs):
        train_loss , train_acc = train_fn(model, train_loader , optimizer , loss_fn , device)
        val_loss , val_acc = test_fn(model , val_loader , loss_fn , device)
        train_loss_his.append(train_loss)
        train_acc_his.append(train_acc)
        val_loss_his.append(val_loss)
        val_acc_his.append(val_acc)
        
        # print result
        print(f"[{epoch+1}/{epochs}] | Train_loss: {train_loss:.5f} , Train_acc : {train_acc*100:.2f}% \
                | Val_loss: {val_loss:.5f} , Val_acc : {val_acc*100:.2f}%")
        # model save
        path_dir = "/kaggle/working/model"
        if best_loss is None:
            best_loss = train_loss
            save_checkpoint(model , optimizer , "best" , path_dir , epoch = epoch , loss = train_loss , acc = train_acc)
        elif best_loss > train_loss:
            print("Best!!")
            best_loss = train_loss
            save_checkpoint(model , optimizer , "best" , path_dir , epoch = epoch , loss = train_loss , acc = train_acc)
        save_checkpoint(model , optimizer , "last" , path_dir , epoch = epoch , loss = train_loss , acc = train_acc)
    history = {"train_acc" : train_acc_his,
              "train_loss" : train_loss_his,
              "val_acc" : val_acc_his,
              "val_loss" : val_loss_his}
    return history
            
            