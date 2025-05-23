import torch
import torch.nn as nn

from .sphereconv.src.spherenet import SphereConv2d

class CNNBlock(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=2, dropout=0.05):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        hidden_dim = int(inp * expansion)
        
        self.pointwise1 = nn.Sequential(
            nn.Conv2d(
                in_channels=inp,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=False,
                groups=inp
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU()
        )
        
        self.depthwise = nn.Sequential(
            SphereConv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                groups=hidden_dim,
                bias=False,                
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),
        )
        
        self.pointwise2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=oup,
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=False, 
            ),
            nn.BatchNorm2d(oup),
        )

        self.pointwise_res = nn.Sequential(
            nn.Conv2d(
                in_channels=inp,
                out_channels=oup,
                kernel_size=1,
                stride=stride,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(oup),            
        )
        
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
    
            
    def forward(self, x):
        identity = x
        
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.pointwise2(x)
        
        identity = self.pointwise_res(identity)
        x = x + identity
        x = self.act(x)
        x = self.dropout(x)
        
        return x