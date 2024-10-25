import os
import math
import numpy as np
import random
import time
import pdb 
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as op
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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

def search(b,L,waccm,data_list,test,label,init_weight,num = 16432):
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
      acc = torch.sum(wvote==test)/num
        # print(f"weight vote:{acc}") 
      
      if acc > waccm:
        wm = weight
        waccm = acc
      
        print('best weight:{}\nbest acc:{}'.format(wm,waccm))
        
        
    

directory = './train'  

skip = [0,5]
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

test = np.load('/home/niyunfei/workspace/wuRenji/wuRenji/wuRenji/data/train_label.npy')
test = torch.from_numpy(test).cuda()


if __name__ == '__main__':
    num = 16432
 
    label_one_hot = torch.zeros((num,155)).cuda()
    for i in range(num):
      label_one_hot[i,test[i]] = 1
  
    vote = 0
    wvote = 0
    # weight = [ 0.1419,  0.1419,  0.2274, -0.0611,  0.1439,  0.1262,  0.0693,  0.0842,
    #       0.0767,  0.1104]
    
    for i in range(len(data_list)):
     
      vote += data_list[i]
      # wvote += data_list[i]*weight[i]
      p = torch.Tensor(data_list[i].argmax(-1)).cuda()
      acc = torch.sum(p==test)/num
      print(f"model {name_list[i]} prev pred:{acc}")
    
    vote = torch.Tensor(vote.argmax(axis=1)).cuda()
    acc = torch.sum(vote==test)/num
    print(f"voted pred:{acc}")    
    
    # np.save('pred.npy',wvote)
    # print(wvote.shape)
    # wvote = torch.Tensor(wvote.argmax(axis=1)).cuda()
    # acc = torch.sum(wvote==test)/num
    # print(f"weigb vote:{acc}") 
 

    # time.sleep(1000)
    
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
        self.fc = nn.Sequential(
          nn.Linear(c, 256, bias=False),
          nn.ReLU(),
          nn.Linear(256, 1,bias=False),
        )
        # self.fc = nn.Linear(c,1,bias=False)
    
        
      def forward(self, x):
    
        x = self.fc(x).squeeze(-1)
        return x
      
    total = 2000
    flip = 100
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()
    mlp = MLP(label.shape[-1]).cuda()
    optim = op.SGD(mlp.parameters(),lr=0.0001,momentum=0.9,weight_decay=0.0004)
    schedule = op.lr_scheduler.LambdaLR(optim, lr_lambda=[lambda flip:total*flip/(flip+i)])
    
    weigh = [1 for _ in range(label.shape[-1])]
    accm = 0
    index = 0
    eval = 1
    mlp.load_state_dict(torch.load('vote.pt'))
    for _ in tqdm(range(total)):
        optim.zero_grad()
      
        out = mlp(label)
        loss = loss_fn(out,label_one_hot)
        
        acc = torch.sum(out.argmax(dim=-1)==test)/num
        
        if eval==1:
            mlp.load_state_dict(torch.load('vote.pt'))
            mlp.eval()
            out = mlp(label)
            acc = torch.sum(out.argmax(dim=-1)==test)/num
            print(f'acc:{acc}')
            break
        
        if acc > accm:
            print('best acc:{}'.format(acc))
            accm = acc
            torch.save(mlp.state_dict(),'vote.pt')
            index = _
        
        loss.backward()
        optim.step()
        schedule.step()
    print('best acc:{} in {}'.format(accm,index))    
    # pdb.set_trace()
    
    
    
    
    
    
    
    
    
    # Rubutness
    # weight = torch.load('74646.pt')
    # import pdb
    # pdb.set_trace()
    # weight = [ 0.1419,  0.1419,  0.2274, -0.0611,  0.1439,  0.1262,  0.0693,  0.0842,
    #       0.0767,  0.1104]
    # # weight = torch.Tensor(weight).cuda().softmax(dim=0)
    # for j in range(9):
    #   wvm = 0
    #   std = np.std(weight)/10
    #   mean = np.mean(weight)
    #   for i in range(len(data_list)):
    #     wvm += data_list[i]*(weight[i]+random.uniform(-std*(j+1),std*(j+1)))
        
    #   wvm = torch.Tensor(wvm.argmax(axis=1)).cuda()
    #   accd = torch.sum(wvm==test)/num
    #   print('wvm:{},delta weight: {:f}% delta acc: {:f}%\n'.format(accd,100*std*(j+1)/mean,(accd-acc).abs()*100/acc))
    
   
