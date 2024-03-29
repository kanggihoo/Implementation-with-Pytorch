import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from optim_schedule import ScheduledOptim

class BERTTrainer():
  def __init__(self,
               model,
               train_dataloader,
               test_dataloader=None,
               lr=1e-4,
               weight_decay=0.01,
               betas = (0.9,0.999),
               warmup_steps = 10000,
               log_freq = 10,
               device = "cuda"):
    self.device = device
    self.model = model
    self.train_data = train_dataloader
    self.test_data = test_dataloader
    # setting optimizer
    self.optim = optim.Adam(self.model.parameters() , lr = lr, betas = betas , weight_decay = weight_decay)

    self.optim_schedule = ScheduledOptim(self.optim , self.model.bert.embed_dim , n_warmup_steps=warmup_steps)

    # define Loss function (Negative Log Likelihood loss)
    self.criterion = nn.NLLLoss(ignore_index = 0).to(device)
    self.log_freq = log_freq
    print("Total Parameters:" , sum([param.nelement() for param in self.model.parameters()]))

  def train(self , epoch):
    self.iteration(epoch , self.train_data , train=True)

  def test(self , epoch):
    self.iteration(epoch , self.test_data , train=False)

  def iteration(self , epoch , data_loader, train:bool=True):
    avg_loss = 0
    total_correct = 0
    total_element = 0
    mode = "train" if train else "test"

    data_iter = tqdm.tqdm(enumerate(data_loader) ,
                          desc=f"EP_{mode}:{epoch}",
                          total=len(data_loader),
                          bar_format = "{l_bar}{r_bar}"
                          )
    for i, data in data_iter:
      # Each Batch Data sent to current device
      data = {key:value.to(self.device) for key ,  value in data.items()}
      # proceed forward method in our model
      next_sent_output , mask_language_output = self.model(data["bert_input"] , data["segment_label"])


      print(next_sent_output.shape , data["is_next"].shape)
      # calculate NLL loss about classification result of is_next
      next_loss = self.criterion(next_sent_output , data["is_next"])

      # calculate NLL loss How the model well predicted about masked token????
      mask_loss = self.criterion(mask_language_output.transpose(1,2) , data["bert_label"])

      loss = next_loss + mask_loss

      if mode == "train":
        self.optim_schedule.zero_grad()
        loss.backward()
        self.optim_schedule.step_and_update_lr()
      # measure sentence prediction accuary

      correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()

      avg_loss += loss.item()
      total_correct += correct
      total_element += data["bert_input"].size(0)

      result = {
          "epoch" : epoch,
          "iter" : i,
          "avg_loss" : avg_loss / (i+1),
          "avg_acc" : total_correct / total_element * 100,
          "loss" : loss.item()
      }

      if i % self.log_freq ==0:
        data_iter.write(str(result))

    print(
        f"Epoch :{epoch}\
        Mode : {mode}\
        avg_loss = {avg_loss/len(data_loader)}\
        total_acc = {total_correct / total_element * 100}"
    )
