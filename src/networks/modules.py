import torch
import torch.nn as nn
import torch.nn.functional as F

def printt(x, t=False):
    if t: print(x)

def sigmoid(x, alpha=10000):
    return 1/(1+torch.exp(-x*alpha))

class xshape(nn.Module):
    def __init__(self, name='name'):
        super().__init__()
        self.name = name
        
    def forward(self, x):
        print(self.name, ': ', x.shape)
        
        return x
    
class NoneBlock(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x
    
class Res_block(nn.Module):
    def __init__(self, dim_in, dim_out, k=3, dr=1, down=1, act=nn.SiLU()):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.down = down
        self.act = act
        self.dr = dr
        dim_down = dim_out//dr
        stride = 2 if self.down==2 else 1

        if dr>1: # Bottle Neck
            self.encoder = nn.Sequential(
                nn.Conv2d(dim_in, dim_down, (1, 1), stride=1, bias=False), 
                nn.BatchNorm2d(dim_down), 
                self.act, 
                
                nn.Conv2d(dim_down, dim_down, (k, k), stride=stride, padding=k//2, bias=False), 
                nn.BatchNorm2d(dim_down), 
                self.act, 
                
                nn.Conv2d(dim_down, dim_out, (1, 1), stride=1, bias=False), 
                nn.BatchNorm2d(dim_out))
        else: # Res Block
            self.encoder = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, (k, k), stride=stride, padding=k//2, bias=False), 
                nn.BatchNorm2d(dim_out), 
                self.act, 
                
                nn.Conv2d(dim_out, dim_out, (k, k), stride=1, padding=k//2, bias=False), 
                nn.BatchNorm2d(dim_out))
        
        self.eq_channel = nn.Conv2d(dim_in, dim_out, (1, 1), stride=1)
        self.eq_size_up = nn.Upsample(scale_factor=1/down, mode='bilinear', align_corners=True)
        self.eq_size_down = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        y = self.encoder(x)
        if y.shape[1]!=x.shape[1]: 
            x = self.eq_channel(x)
        if self.down==2: 
            x = self.eq_size_down(x)
        if self.down==0.5: 
            x = self.eq_size_up(x)
            y = self.eq_size_up(y)
        y = self.act(y+x)
        
        return y