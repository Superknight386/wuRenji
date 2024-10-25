import math
import pdb
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from einops import rearrange

# B, C, T, V, M
# use fft in T and V*M, viewing it as H and W. 
# than we use transformer to learn the relation between H and W

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


def build_position_encoding(h, w):
    
    temperature = 10000

    # populate all possible relative distances
    x_embed = torch.linspace(0, w, w, dtype=torch.float32).cuda()

    # scale distance if there is down sample

    dim_y = torch.arange(h, dtype=torch.float32)
    y_embed = temperature ** (2 * (dim_y // 2) / h).cuda()

    pos = x_embed[:, None] / y_embed
    # interleave cos and sin instead of concatenate
    for i in range(h):
        pos[:, i] = pos[:, i].sin() if i % 2 == 0 else pos[:, i].cos()
    pos_embed = pos.transpose(0,1).contiguous()
    # plt.imsave("/home/niyunfei/o/pe.png", pos_embed, cmap="jet")
    return pos_embed

def Get_fft(x):
    '''
    return: amplitude and phase
    '''
    x = torch.fft.fft(x, dim=-1)
    # x = torch.view_as_real(x)[:,:, :-1]
    # x = rearrange(x,'n c t v d -> n c (t d) v')
    # return x
    x_real = x.real
    x_imag = x.imag
    
    amp = torch.sqrt(x_real**2 + x_imag**2)
    phase = torch.atan2(x_imag, x_real)
    return amp, phase
    
class Multi(nn.Module):
    def __init__(self,ker,indim,dim):
        super(Multi, self).__init__()
        if ker == 2:
            self.out1 = nn.Sequential(
            nn.Conv2d(indim, dim, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            )
            self.out3 = nn.Sequential(
            nn.Conv2d(indim, dim, kernel_size=(3,1), stride=(1,1), padding=(1,0)),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            )
            self.out5 = nn.Sequential(
            nn.Conv2d(indim, dim, kernel_size=(5,1), stride=(1,1), padding=(2,0)),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            )
        elif ker == 3:
            self.out1 = nn.Sequential(
            nn.Conv3d(indim, dim, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
            )
            self.out3 = nn.Sequential(
            nn.Conv3d(indim, dim, kernel_size=(3,3,1), stride=(1,1,1), padding=(1,1,0)),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
            )
            self.out5 = nn.Sequential(
            nn.Conv3d(indim, dim, kernel_size=(5,5,1), stride=(1,1,1), padding=(2,2,0)),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
            )
        
    def forward(self, x):
        return (self.out1(x)+self.out3(x)+self.out5(x))/3

class VAtt(nn.Module):
    def __init__(self):
        super(VAtt, self).__init__()
        self.vf = nn.MultiheadAttention(embed_dim=300, num_heads=30)
        self.vt = nn.MultiheadAttention(embed_dim=300, num_heads=30)
        
        self.QKVf = nn.Linear(102,306)
        self.QKVt = nn.Linear(102,306)
        self.norm = nn.Sequential(
            nn.Sigmoid(),
            nn.LayerNorm(300),
            nn.Dropout(0.2),
        )
    def forward(self, xf, xt):
        # B C L -> L B C
        # B 102 300 -> 102 B 300
        # xf = xf.permute(1,0,2)
        # xt = xt.permute(1,0,2)
        
        xfq,xfk,xfv = torch.chunk(self.QKVf(xf.permute(0,2,1)).permute(2,0,1),3,dim=0)
        xtq,xtk,xtv = torch.chunk(self.QKVt(xt.permute(0,2,1)).permute(2,0,1),3,dim=0)

        x_vf, _ = self.vf(xfq,xtk,xtv)
        x_vt, _ = self.vt(xtq,xfk,xfv)
        
        x_vf = x_vf.permute(1,0,2) + xf
        x_vt = x_vt.permute(1,0,2) + xt
        
        return self.norm(x_vf), self.norm(x_vt)

class Att(nn.Module):
    def __init__(self,hidden_dim,edim):
        super(Att, self).__init__()
        # self.Att = nn.MultiheadAttention(embed_dim=edim, num_heads=8)
        self.Atttt = nn.MultiheadAttention(embed_dim=102, num_heads=6)
        self.Attff = nn.MultiheadAttention(embed_dim=102, num_heads=6)
        self.Atttf = nn.MultiheadAttention(embed_dim=102, num_heads=6)
        self.Attft = nn.MultiheadAttention(embed_dim=102, num_heads=6)
        
        # self.Atttt = nn.MultiheadAttention(embed_dim=hidden_dim*17, num_heads=10)
        # self.Attff = nn.MultiheadAttention(embed_dim=hidden_dim*17, num_heads=10)
        
        # self.QKVff = nn.Linear(hidden_dim*17,hidden_dim*17*3)
        # self.QKVtt = nn.Linear(hidden_dim*17,hidden_dim*17*3)
        
        self.QKVf = nn.Linear(300,900)
        self.QKVt = nn.Linear(300,900)
        self.norm = nn.Sequential(
            nn.Sigmoid(),
            nn.LayerNorm(300),
            nn.Dropout(0.2),
        )
        #L B C
    def forward(self, xf, xt):
        # dx = self.QKV(x)
        # q, k, v = torch.chunk(dx,3,dim=-1)
        # pdb.set_trace()
        x_tt_q, x_tt_k, x_tt_v = torch.chunk(self.QKVt(xt).permute(2,0,1),3,dim=0)
        # x_tf_q, x_tf_k, x_tf_v = torch.chunk(self.QKVtf(xt),3,dim=-1)
        
        x_ff_q, x_ff_k, x_ff_v = torch.chunk(self.QKVf(xf).permute(2,0,1),3,dim=0)
        # x_ft_q, x_ft_k, x_ft_v = torch.chunk(self.QKVft(xf),3,dim=-1)
        # pdb.set_trace()
        x_t_t= self.Atttt(x_tt_q, x_tt_k, x_tt_v)[0].permute(1,2,0)
        x_f_f= self.Attff(x_ff_q, x_ff_k, x_ff_v)[0].permute(1,2,0)
        x_t_f= self.Atttf(x_ff_q, x_tt_k, x_tt_v)[0].permute(1,2,0)
        x_f_t= self.Attft(x_tt_q, x_ff_k, x_ff_v)[0].permute(1,2,0)
       
        # x_t_f = torch.einsum('l b c,b l w -> b c w',x_tt_v,wf)
        # x_f_t = torch.einsum('l b c,b l w -> b c w',x_ff_v,wt)
        
        # x_t_t = x_t_t.permute(1,2,0)
        # x_f_f = x_f_f.permute(1,2,0)
        
        x_t = x_t_t + x_t_f 
        x_f = x_f_f + x_f_t 
        return self.norm(x_t), self.norm(x_f)

class msstMoudel(nn.Module):
    def __init__(self, inchannel, outchannel1, outchannel3, outchannel5, outchannel7, outchannel11, stride=(1, 1)):
        super(msstMoudel, self).__init__()
        inplace = True

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel1, kernel_size=(1, 1), stride=stride,
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel1, affine=True),
            nn.ReLU(inplace))
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel3, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel3, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel3, out_channels=outchannel3, kernel_size=(3, 3), stride=stride,
                      padding=(1, 1)),
            nn.BatchNorm2d(outchannel3, affine=True),
            nn.ReLU(inplace))

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel5, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel5, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel5, out_channels=outchannel5, kernel_size=(5, 1), stride=stride,
                      padding=(2, 0)),
            nn.BatchNorm2d(outchannel5, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel5, out_channels=outchannel5, kernel_size=(1, 5), stride=(1, 1),
                      padding=(0, 2)),
            nn.BatchNorm2d(outchannel5, affine=True),
            nn.ReLU(inplace))

        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel7, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel7, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel7, out_channels=outchannel7, kernel_size=(7, 1), stride=stride,
                      padding=(3, 0)),
            nn.BatchNorm2d(outchannel7, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel7, out_channels=outchannel7, kernel_size=(1, 7), stride=(1, 1),
                      padding=(0, 3)),
            nn.BatchNorm2d(outchannel7, affine=True),
            nn.ReLU(inplace))

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel11, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel11, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel11, out_channels=outchannel11, kernel_size=(11, 1), stride=stride,
                      padding=(5, 0)),
            nn.BatchNorm2d(outchannel11, affine=True),
            nn.ReLU(inplace))
            # nn.Conv2d(in_channels=outchannel11, out_channels=outchannel11, kernel_size=(1, 11), stride=stride,
            #           padding=(0, 5)),
            # nn.BatchNorm2d(outchannel11, affine=True),
            # nn.ReLU(inplace))

    def forward(self, input):
        output1 = self.conv1x1(input)
        # print('o1',output1.size())
        output3 = self.conv3x3(input)
        # print('o3', output3.size())
        output5 = self.conv5x5(input)
        # print('o5', output5.size())
        output7 = self.conv7x7(input)

        # print('o7', output7.size())
        output11 = self.conv11(input)
        output = torch.cat([output1, output3, output5, output7, output11], 1)
        return output

class MSSTNet(nn.Module):

    def __init__(self, num_class=1000, dropout=0.8):
        super(MSSTNet, self).__init__()
        inplace = True
        self.dropout = dropout

        self.avgpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), ceil_mode=False, count_include_pad=True),
            nn.BatchNorm2d(416, affine=True),
            nn.ReLU(inplace))
        self.avgpool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), ceil_mode=False, count_include_pad=True),
            nn.BatchNorm2d(672, affine=True),
            nn.ReLU(inplace))

        self.m1 = msstMoudel(inchannel=3, outchannel1=32, outchannel3=48, outchannel5=48, outchannel7=48,
                              outchannel11=48, stride=(2, 1))
        self.m2 = msstMoudel(inchannel=224, outchannel1=32, outchannel3=64, outchannel5=64, outchannel7=64,
                              outchannel11=64, stride=(2, 1))
        self.m3 = msstMoudel(inchannel=288, outchannel1=32, outchannel3=96, outchannel5=96, outchannel7=96,
                              outchannel11=96, stride=(2, 1))
        self.m4 = msstMoudel(inchannel=416, outchannel1=128, outchannel3=128, outchannel5=128, outchannel7=128,
                              outchannel11=128, stride=(2, 1))
        self.m5 = msstMoudel(inchannel=640, outchannel1=32, outchannel3=160, outchannel5=160, outchannel7=160,
                              outchannel11=160, stride=(2, 1))
        self.m6 = msstMoudel(inchannel=672, outchannel1=192, outchannel3=192, outchannel5=192, outchannel7=192,
                              outchannel11=192, stride=(1, 1))
        self.m7 = msstMoudel(inchannel=960, outchannel1=256, outchannel3=256, outchannel5=256, outchannel7=256,
                              outchannel11=256, stride=(1, 1))
        self.dropout = nn.Dropout(p=self.dropout)
        self.last_linear = nn.Linear(1280, num_class)

    def features(self, input):
        m1out = self.m1(input)
        m2out = self.m2(m1out)
        m3out = self.m3(m2out)
        m3poolout = self.avgpool(m3out)
        m4out = self.m4(m3poolout)
        m5out = self.m5(m4out)
        m5poolout = self.avgpool2(m5out)
        m6out = self.m6(m5poolout)
        m7out = self.m7(m6out)
        # print(m7out.size())
        return m7out

    def logits(self, features):
        fea_base = features
        adaptiveAvgPoolWidth = (features.shape[2], features.shape[3])
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x_base = x.view(x.size(0), -1)
        x = self.dropout(x_base)
        x = self.last_linear(x)
        return fea_base, x

    def forward(self, input):
        x = self.features(input)
        x_base, x = self.logits(x)
        return x_base, x

class Model(nn.Module):
    def __init__(self, num_class=155, dropout=0.8):
        super(Model, self).__init__()
        self.dropout = dropout
        self.num_class = num_class
        self.model = MSSTNet(num_class=self.num_class, dropout=self.dropout)
        self.model_p = MSSTNet(num_class=self.num_class, dropout=self.dropout)
        self.model_a = MSSTNet(num_class=self.num_class, dropout=self.dropout)
        
        self.MLP = nn.Sequential(
            nn.Linear(num_class*3, num_class*6),
            nn.ReLU(),
            nn.Linear(num_class*6, self.num_class)
        )
        

    def forward(self, input):
        #N C T V M
        input = input.permute(0,4,1,2,3).contiguous()
        B, O, C, H, W = input.size()
        input = input.view(B*O,C,H,W)
        x_f_a,x_f_p = Get_fft(input)
        
        
        out_base, output = self.model(input)
        _,out_a = self.model_a(x_f_a)
        _,out_p = self.model_p(x_f_p)
        
        
        out_a = out_a.view(B,O,-1).mean(dim=1)
        out_p = out_p.view(B,O,-1).mean(dim=1)
        output = output.view(B,O,-1).mean(dim=1)
        out = torch.cat([out_a,out_p,output],dim=1)
        out = self.MLP(out)
        
        
        return output
    # def forward(self, x):
    #     N, C, T, V, M = x.size() 
    #     x = rearrange(x,'n c t v m -> (n m) c v t')
        
    #     # if torch.isnan(x).any() or torch.isinf(x).any():
    #     # pdb.set_trace()
    #     _,_,h,w = x.shape
        
    #     pe = build_position_encoding(h,w)
        
    #     #N, 3*2*17, 300 (time or freq)
    #     # x_f = torch.einsum('n c t,t f -> n c t f',x,fpe).sum(dim=-2)*pe
    #     # pdb.set_trace()
    #     x_f_a,x_f_p = Get_fft(x)
        
        
# summary(Model(),(1,3,300,17,2))        