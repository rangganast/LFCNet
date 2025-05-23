import torch
import torch.nn.functional as F
from torch import nn

from .GridGenerator import GridGenerator

class OptimizedSphereConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=bias)
        self.grid_shape = None
        self.register_buffer('precomputed_grid', None)

    def genSamplingPattern(self, h, w, device):
        gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride)
        LonLatSamplingPattern = gridGenerator.createSamplingPattern()

        # Generate grid for F.grid_sample
        lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
        lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

        grid = torch.stack((lon_grid, lat_grid), axis=-1)
        self.precomputed_grid = torch.FloatTensor(grid).to(device)

    def forward(self, x):
        B, C, H, W = x.shape

        # Check if grid needs to be regenerated
        if self.grid_shape is None or self.grid_shape != (H, W):
            self.grid_shape = (H, W)
            self.genSamplingPattern(H, W, x.device)

        # Repeat precomputed grid for the batch
        grid = self.precomputed_grid.expand(B, -1, -1, -1)  # (B, H*Kh, W*Kw, 2)

        # Sample input tensor
        x_sampled = F.grid_sample(x, grid, align_corners=True, mode='nearest')  # (B, C, H*Kh, W*Kw)

        # Apply grouped convolutions
        x = F.conv2d(x_sampled, self.weight, self.bias, stride=self.kernel_size, groups=self.groups)

        return x

class SphereConv2d(nn.Conv2d):
  """
  kernel_size: (H, W)
  """

  def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3),
               stride=1, padding=0, dilation=1,
               groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
    super(SphereConv2d, self).__init__(
      in_channels, out_channels, kernel_size,
      stride, padding, dilation, groups, bias, padding_mode)
    self.grid_shape = None
    self.grid = None

  def genSamplingPattern(self, h, w):
    gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    # generate grid to use `F.grid_sample`
    lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
    lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

    grid = torch.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      self.grid = torch.FloatTensor(grid)
      self.grid.requires_grad = False

  def forward(self, x):
    # Generate Sampling Pattern
    B, C, H, W = x.shape

    if (self.grid_shape is None) or (self.grid_shape != (H, W)):
      self.grid_shape = (H, W)
      self.genSamplingPattern(H, W)

    with torch.no_grad():
      grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2)
      grid.requires_grad = False

    x = F.grid_sample(x, grid, align_corners=True, mode='nearest')  # (B, in_c, H*Kh, W*Kw)

    # self.weight -> (out_c, in_c, Kh, Kw)
    x = F.conv2d(x, self.weight, self.bias, stride=self.kernel_size, groups=self.groups)

    return x  # (B, out_c, H/stride_h, W/stride_w)
