import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.modules import *

class ResNet(nn.Module):
    def __init__(self, dim_in=1, dim_out=1, dim_base=64, shape=False):
        super().__init__()
        self.shape = shape
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_base*1, (7, 7), stride=2, padding=7//2, bias=False), 
            nn.BatchNorm2d(dim_base*1), 
            nn.ReLU(), 
            # Res_block(dim_base*1, dim_base*1), 
            # Res_block(dim_base*1, dim_base*1), 
            Res_block(dim_base*1, dim_base*1, down=1))
            
        self.encoder_2 = nn.Sequential(
            Res_block(dim_base*1, dim_base*2), 
            # Res_block(dim_base*2, dim_base*2), 
            # Res_block(dim_base*2, dim_base*2), 
            Res_block(dim_base*2, dim_base*2, down=1))
        
        self.encoder_3 = nn.Sequential(
            Res_block(dim_base*2, dim_base*4),
            # Res_block(dim_base*4, dim_base*4), 
            # Res_block(dim_base*4, dim_base*4), 
            # Res_block(dim_base*4, dim_base*4), 
            # Res_block(dim_base*4, dim_base*4), 
            Res_block(dim_base*4, dim_base*4, down=2))
        
        self.encoder_4 = nn.Sequential(
            Res_block(dim_base*4, dim_base*8), 
            # Res_block(dim_base*8, dim_base*8),
            Res_block(dim_base*8, dim_base*8, down=1))
            
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(dim_base*8, dim_out), 
            nn.Sigmoid())
        
    def forward(self, x):
        B, S = x.shape[0], x.shape[1] - 1
        x = self.encoder_1(x)
        printt(f'{"encoder_1":<20}: {x.shape}', t=self.shape)
        x = self.encoder_2(x)
        printt(f'{"encoder_2":<20}: {x.shape}', t=self.shape)
        x = self.encoder_3(x)
        printt(f'{"encoder_3":<20}: {x.shape}', t=self.shape)
        x = self.encoder_4(x)
        printt(f'{"encoder_4":<20}: {x.shape}', t=self.shape)
        x = self.pool(x)
        printt(f'{"pool":<20}: {x.shape}', t=self.shape)
        x = x.reshape(B, x.shape[1])
        x = self.fc(x)
        printt(f'{"fc":<20}: {x.shape}', t=self.shape)
        
        return x
