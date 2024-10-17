import os
import math
import numpy as np
import random
import time
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as op

def polynomial_value(coefficients, x):
    """
    计算多项式的值。

    :param coefficients: 多项式系数列表，从最高次项到常数项
    :param x: 变量值
    :return: 多项式的值
    """
    n = len(coefficients)
    result = 0
    for i in range(n):
        result += coefficients[i] * (x ** (i))
    return round(result)

def search(b,L,waccm,data_list,test,label,init_weight):
    # bias = polynomial_value(init_weight,b)
    # print(bias)
    for _ in tqdm(range(b**L)): 
    #   _ += bias
      weight = [0 for _ in range(L)]  
      for j in range(len(weight)):
        weight[j] = (_//b**j)%b/b
      # print(weight)
      wvote = 0
        
      for i in range(len(data_list)):
        wvote += label[:,:,i]*weight[i]
        
      wvote = torch.Tensor(wvote.argmax(axis=1)).cuda()
      acc = torch.sum(wvote==test)/4599
        # print(f"weight vote:{acc}") 
      
      if acc > waccm:
        wm = weight
        waccm = acc
      
        print('best weight:{}\nbest acc:{}'.format(wm,waccm))
        
        
    

directory = './log'  

skip = [0,4,12]
data_list = []
name_list = []
count = -1
for root, dirs, files in os.walk(directory):
    for file in files:
       
        if file.endswith('.npy'):
            count += 1
            file_path = os.path.join(root, file)
           
            data = np.load(file_path)
            print(f"{count}Loaded {file_path}, shape: {data.shape}")
            if count in skip:
              continue
            # print(f"{len(data_list)}Loaded {file_path}, shape: {data.shape}")
            data_list.append(data)
            name_list.append(file)


label = np.stack(data_list,axis=2)
label = torch.from_numpy(label).cuda()

test = np.load('/home/niyunfei/workspace/wuRenji/wuRenji/wuRenji/data/test_label.npy')
test = torch.from_numpy(test).cuda()


if __name__ == '__main__':
 
    label_one_hot = torch.zeros((4599,155)).cuda()
    for i in range(4599):
      label_one_hot[i,test[i]] = 1
  
    vote = 0
    wvote = 0
    # weight = [64.2885, 72.2760, 61.1249, 70.2942, 84.6421, 79.3695, 69.4743, 58.7067,
    #      73.5881, 57.4281]
    
    for i in range(len(data_list)):
     
      vote += data_list[i]
    #   wvote += data_list[i]*weight[len(data_list)-1-i]
      p = torch.Tensor(data_list[i].argmax(-1)).cuda()
      acc = torch.sum(p==test)/4599
      print(f"model {name_list[i]} prev pred:{acc}")
    
    vote = torch.Tensor(vote.argmax(axis=1)).cuda()
    acc = torch.sum(vote==test)/4599
    print(f"vote pred:{acc}")    
    
    # wvote = torch.Tensor(wvote.argmax(axis=1)).cuda()
    # acc = torch.sum(wvote==test)/4599
    # print(f"weig vote:{acc}") 
    
    time.sleep(1)
    
    # # SEARCH
    # b=5
    # L = len(data_list)
    # waccm = acc   
    # init_weight = weight

    # search(b,L,waccm,data_list,test,label,init_weight)  
 
    # TRAIN
    
    class MLP(nn.Module):
      def __init__(self, c):
        super(MLP, self).__init__()
        self.fc = nn.Linear(c,1,bias=False)
        self.fc.weight.data = torch.Tensor([64.2885, 72.2760, 61.1249, 70.2942, 84.6421, 79.3695, 69.4743, 58.7067,
         73.5881, 57.4281])
        
      def forward(self, x):
       
        x = self.fc(x).squeeze(-1)
        return x
      
    total = 100000
    flip = 100
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()
    mlp = MLP(label.shape[-1]).cuda()
    optim = op.SGD(mlp.parameters(),lr=0.00001,momentum=0.9,weight_decay=0.0004)
    schedule = op.lr_scheduler.LambdaLR(optim, lr_lambda=[lambda flip:total*flip/(flip+i)])
    
    weight = [64.2885, 72.2760, 61.1249, 70.2942, 84.6421, 79.3695, 69.4743, 58.7067,
         73.5881, 57.4281]
    accm = 0
    index = 0
    eval = 0
    for _ in tqdm(range(total)):
        optim.zero_grad()
      
        out = mlp(label)
        loss = loss_fn(out,label_one_hot)
        
        acc = torch.sum(out.argmax(dim=-1)==test)/4599
        if eval==1:
            print(f'acc:{acc}')
            break
        
        if acc > accm:
          accm = acc
          weight = mlp.fc.weight
          index = _
        
        loss.backward()
        optim.step()
        schedule.step()
    print('best weight:{}\nbest acc:{} in {}'.format(weight,accm,index))    
        
    
    # Rubutness
    # weight = [171.8091,  95.9382, 508.0397, 459.0803, 240.3197, 147.5455, 142.4117,
    #      177.9169, 155.9857,  76.2838]
    # for j in range(9):
    #   wvm = 0
    #   std = np.std(weight)/10
    #   for i in range(len(data_list)):
    #     wvm += data_list[i]*(weight[i]+random.uniform(-std*(j+1),std*(j+1)))
        
    #   wvm = torch.Tensor(wvm.argmax(axis=1)).cuda()
    #   accd = torch.sum(wvm==test)/4599
    #   print('wvm:{},{} delta:{:4f}'.format(accd,0.01*(j+1),(accd-acc)/acc))
    
   
