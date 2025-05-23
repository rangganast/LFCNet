import torch
import torch.nn as nn
from einops import rearrange

from .deformable_att import DAttentionBaseline
# from models.deformable_att import DAttentionBaseline

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU()
    )

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')


class LayerScale(nn.Module):

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1, 1))
        else:
            return x * self.weight.view(-1, 1, 1)
        
class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim2, 1, 1, 0),
            # nn.GELU(),
            # nn.BatchNorm2d(self.dim2, eps=1e-5)
        )
        self.drop1 = nn.Dropout(drop, inplace=False)
        self.act = nn.GELU()
        # self.bn = nn.BatchNorm2d(self.dim2, eps=1e-5)
        self.linear2 = nn.Sequential(
            nn.Conv2d(self.dim2, self.dim1, 1, 1, 0),
            # nn.BatchNorm2d(self.dim1, eps=1e-5)
        )
        self.drop2 = nn.Dropout(drop, inplace=False)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
    
    def forward(self, x):
        
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        # x = self.bn(x)
        x = self.linear2(x)
        x = self.drop2(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, fmap_size, dim, depth, heads, dim_head, expansion, n_groups, stride, dropout=0.):
        super().__init__()
        self.depth = depth
        
        # attn
        self.norm_attn_layers = nn.ModuleList([
            LayerNormProxy(dim) for _ in range(depth)
        ])
        self.attn_layers = nn.ModuleList([
            DAttentionBaseline(
                q_size=fmap_size,
                n_heads=heads,
                n_head_channels=dim_head,
                n_groups=n_groups,
                proj_drop=dropout,
                stride=stride,
                offset_range_factor=16,
                use_pe=False,
                dwc_pe=False,
                no_off=False,
                fixed_pe=False,
                ksize=16,
                log_cpb=False,
                attn_drop=dropout,
            ) for _ in range(depth)
        ])
        self.scale_attn_layers = nn.ModuleList([
            LayerScale(dim, False) for _ in range(depth)
        ])
        
        # ffn (mlp)
        self.norm_mlp_layers = nn.ModuleList([
            LayerNormProxy(dim) for _ in range(depth)
        ])
        self.mlp_layers = nn.ModuleList([
            TransformerMLPWithConv(dim, expansion=expansion, drop=0.0) for _ in range(depth)
        ])
        self.scale_mlp_layers = nn.ModuleList([
            LayerScale(dim, False) for _ in range(depth)
        ])
        
        self.drop_path = nn.ModuleList([
            DropPath(dropout) if dropout > 0.0 else nn.Identity() for _ in range(depth)
        ])
    
    def forward(self, x):
        for i in range(self.depth):
            x0 = x
            x = self.norm_attn_layers[i](x)
            x = self.attn_layers[i](x)
            x = self.scale_attn_layers[i](x)
            x = self.drop_path[i](x) + x0
            
            x0 = x
            x = self.norm_mlp_layers[i](x)
            x = self.mlp_layers[i](x)
            x = self.scale_mlp_layers[i](x)
            x = self.drop_path[i](x) + x0
            
        return x.contiguous()
    
class TransformerBlock(nn.Module):
    def __init__(self, fmap_size, dim, depth, channel, heads, dim_head, expansion, n_groups, stride, kernel_size, patch_size, dropout=0.05):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(fmap_size=fmap_size,
                                       dim=dim,
                                       depth=depth,
                                       heads=heads,
                                       dim_head=dim_head,
                                       expansion=expansion,
                                       n_groups=n_groups,
                                       stride=stride,
                                       dropout=dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        # x = rearrange(x, 'b d (h ph) (w pw) -> b d (ph pw) (h w)', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        # x = rearrange(x, 'b d (ph pw) (h w) -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x
    
if __name__ == "__main__":
    import time
    
    DEVICE = "cpu"
    
    input_rgb = torch.randn([1, 32, 20, 20]).to(DEVICE)
    input_depth = torch.randn([1, 32, 20, 20]).to(DEVICE)
    
    block = TransformerBlock(
        fmap_size=(20, 20),
        dim=32,
        depth=2,
        channel=32,
        heads=8,
        dim_head=4,
        expansion=4,
        n_groups=32,
        stride=2,
        kernel_size=(3,3),
        patch_size=(2,2),
    ).to(DEVICE)
    
    """
    q_size=fmap_size,
    n_heads=8,
    n_head_channels=2,
    n_groups=2,
    proj_drop=dropout,
    stride=1,
    offset_range_factor=2,
    use_pe=False,
    dwc_pe=False,
    no_off=False,
    fixed_pe=False,
    ksize=7,
    log_cpb=False,
    """
    
    start = time.time()
    x = block(input_rgb)
    end = time.time()
    print(f"{end-start}s")
    print(x.shape)