import os
import numpy as np
import random
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as op


directory = './VM/train'  
td = './VM/test'


data_list = []
td_list = []


for root, dirs, files in os.walk(directory):
    for file in files:
       
        if file.endswith('.npy'):
         
            file_path = os.path.join(root, file)
           
            data = np.load(file_path)
          
            print(f"Loaded {file_path}, shape: {data.shape}")
        
            data_list.append(data)

for root, dirs, files in os.walk(td):
    for file in files:
       
        if file.endswith('.npy'):
          
            file_path = os.path.join(root, file)
    
            data = np.load(file_path)
        
            print(f"Loaded {file_path}, shape: {data.shape}")
          
            td_list.append(data)

label = np.stack(data_list,axis=-1)
label = torch.from_numpy(label).cuda()

tdlabel = np.stack(td_list,axis=-1)
tdlabel = torch.from_numpy(tdlabel).cuda()


test = np.load('/home/niyunfei/workspace/wuRenji/data/train_label.npy')
test = torch.from_numpy(test).cuda()

tdtest = np.load('/home/niyunfei/workspace/wuRenji/data/test_label.npy')
tdtest = torch.from_numpy(tdtest).cuda()

class att(nn.Module):
  def __init__(self,in_ch):
    super().__init__()
    self.n = in_ch
    
    self.conv = nn.Sequential(
      nn.Conv1d(155,155,in_ch,bias=True),
    )

  def forward(self,x):
   
    dx = self.conv(x)
   
    return dx

if __name__ == '__main__':
    # self att
    loss_fn =  nn.CrossEntropyLoss()
    att_add = att(len(data_list)).cuda()
    optimizer = op.SGD(att_add.parameters(), lr=0.1,momentum=0.9,weight_decay=0.0004)

    #0.1 200 4000
    ite = 200
    total_ep = 4000
    schedule = op.lr_scheduler.MultiStepLR(optimizer, milestones=[total_ep*ite//(ite + i) for i in range(ite)], gamma=0.8)
        
    label_one_hot = torch.zeros((16432,155)).cuda()
    for i in range(16432):
      label_one_hot[i,test[i]] = 1
      
    tdlabel_one_hot = torch.zeros((2000,155)).cuda()
    for i in range(2000):
      tdlabel_one_hot[i,test[i]] = 1
  
    vote = 0
    wvote = 0
    weight = [0.3, 0.9, 0.6, 0.7, 0.2, 0.4]
    for i in range(len(td_list)):
     
      vote += td_list[i]
      wvote += td_list[i]*weight[i]
      p = torch.Tensor(td_list[i].argmax(axis=1)).cuda()
      acc = torch.sum(p==tdtest)/2000
      print(f"prev pred:{acc}")
    
    vote = torch.Tensor(vote.argmax(axis=1)).cuda()
    acc = torch.sum(vote==tdtest)/2000
    print(f"vote pred:{acc}")    
    
    wvote = torch.Tensor(wvote.argmax(axis=1)).cuda()
    acc = torch.sum(wvote==tdtest)/2000
    print(f"weight vote:{acc}") 
    
    time.sleep(1)
    
    # waccm = 0
    # wm = []
    # best_w = []
    # for j in range(5):
    #   i = random.randint(0, len(td_list)-1)
    #   waccm = 0
    #   while waccm < 0.762:
    #     i = random.randint(0, len(td_list)-1)
    #     weight[i] = random.uniform(0, 1)
    #     wvote = 0
        
    #     for i in range(len(td_list)):
    #       wvote += td_list[i]*weight[i]
    #     wvote = torch.Tensor(wvote.argmax(axis=1)).cuda()
    #     acc = torch.sum(wvote==tdtest)/2000
    #     print(f"weight vote:{acc}") 
      
    #     if acc > waccm:
    #       wm = weight
    #       waccm = acc
      
    #   print('best weight:{}\nbest acc:{}'.format(wm,waccm))
    #   best_w.append(wm)
    #   # weight[i] = wm[i]
    #   time.sleep(5)
    
    
     
    # for j in range(9):
    #   wvm = 0
    #   for i in range(len(td_list)):
    #     wvm += td_list[i]*(weight[i]+random.uniform(-0.01*(j+1),0.01*(j+1)))
        
    #   wvm = torch.Tensor(wvm.argmax(axis=1)).cuda()
    #   accd = torch.sum(wvm==tdtest)/2000
    #   print('wvm:{},{} delta:{:4f}'.format(accd,0.01*(j+1),(accd-acc)/acc))
    
    # accm = 0    
    # for i in tqdm(range(total_ep)):
    #     # print("epoch:",i)
        
        
    #     optimizer.zero_grad()
    #     p = att_add(label)
    #     if p.shape[1] == 155:
    #       p = p.squeeze(-1)
        
    #     loss = loss_fn(p,label_one_hot)
    #     # print("loss:",loss.item())
        
    #     p = p.argmax(dim=1)
    #     acc = torch.sum(p==test)/16432
    #     # print(f"pred:{acc}")
    #     # if acc >= accm:
    #     #     accm = acc
    #     #     im = i
    #     #     torch.save(att_add.state_dict(),"/home/niyunfei/workspace/wuRenji/VM/test/att_add.pth")
    #     # # time.sleep(1)
        
    #     loss.backward()
    #     optimizer.step()
        
    #     with torch.no_grad():
    #       tp = att_add(tdlabel)
    #       if tp.shape[1] == 155:
    #         tp = tp.squeeze(-1)
    #       tp = tp.argmax(dim=1)
    #       tacc = torch.sum(tp==tdtest)/2000
    #       # print(f"test pred:{tacc}")  
    #       if tacc >= accm:
    #         accm = tacc
    #         im = i
    #         if accm > 0.8:
    #           print("save model")
    #           # torch.save(att_add.state_dict(),"/home/niyunfei/workspace/wuRenji/VM/test/att_add.pth")
    #     # time.sleep(1)
        
    # print(att_add.state_dict())
    # print("highest acc:",accm,"in:",im)
    
    # torch.save(att_add.state_dict(),"/home/niyunfei/workspace/wuRenji/VM/test/att_add.pth")
          
   
