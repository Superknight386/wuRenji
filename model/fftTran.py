import math
import pdb

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
    x = torch.fft.fft(x, dim=2)
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


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model,self).__init__()
        self.hidden_dim = 256
        
        self.Patchf = nn.Sequential(
            nn.Conv2d(6, self.hidden_dim, kernel_size=(15,1), stride=(15,1)),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
        )
        self.Patcht = nn.Sequential(
            nn.Conv2d(6, self.hidden_dim, kernel_size=(15,1), stride=(15,1)),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
        )
        
        # self.Proj = nn.Sequential(
        #     nn.Conv2d(12, 12, kernel_size=1, stride=1,groups=2),
        #     nn.BatchNorm2d(12),
        #     nn.ReLU(),
        #     nn.Conv2d(12, 12, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(12),
        #     nn.ReLU(),
        # )
        # self.Multi = Multi(3,6,6)
        
        # self.Temp= nn.Sequential(
        #     nn.Conv2d(24, self.hidden_dim, kernel_size=(6,1), stride=(6,1), padding=(0,0)),
        #     nn.BatchNorm2d(self.hidden_dim),
        #     nn.ReLU(),
        # )
        self.Trans = nn.ModuleList([Att(20,self.hidden_dim) for _ in range(12)])
        self.VTrans = nn.ModuleList([VAtt() for _ in range(12)])
        self.MLP = nn.Sequential(
            nn.Linear(num_point*6*600, 300),
            nn.Tanh(),
            nn.Linear(300, num_class),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.Conv3d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.BatchNorm3d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)
        
    def forward(self, x):
        N, C, T, V, M = x.size() 
        x = rearrange(x,'n c t v m -> n (c m v) t')
        
        # if torch.isnan(x).any() or torch.isinf(x).any():
        # pdb.set_trace()
        _,c,t = x.shape
        fpe = build_position_encoding(t,t)
        pe = build_position_encoding(c,t)
        
        #N, 3*2*17, 300 (time or freq)
        # x_f = torch.einsum('n c t,t f -> n c t f',x,fpe).sum(dim=-2)*pe
        # pdb.set_trace()
        x_f_a,x_f_p = Get_fft(x)
        x_t = x
        pdb.set_trace()
        
        # x_f = self.Patchf(x_f)
        # x_t = self.Patcht(x_t)
        
        # # N, 96, 20*17
        # x_f = rearrange(x_f,'n c t v -> n c (t v)')
        # x_t = rearrange(x_t,'n c t v -> n c (t v)')
       
        
        # x_f = x_f*pe
        # x_t = x_t*pe
 
        for i in range(12):
            # x = x.detach()
            x_f, x_t = self.Trans[i](x_f, x_t)
            x_f, x_t = self.VTrans[i](x_f, x_t)
        pdb.set_trace()
        # B, 96, 17*20
        # N, 6*17, 300
        x = torch.cat([x_f, x_t], dim=-1).flatten(1,2)
        # x = (x ).sum(dim=1)
        x = self.MLP(x)
        
        # pdb.set_trace()
        
        # pdb.set_trace()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        #         pa = m.state_dict()
        #         pa1 = pa['bias']
        #         pa2 = pa['weight']
        #         if torch.isnan(pa1).any() or torch.isnan(pa2).any():
        #             pdb.set_trace()
        return x
# summary(Model(),(1,3,25,17,2))        