import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from .sphereconv.src.spherenet import SphereMaxPool2d
from .sphereconv.src.spherenet import SphereConv2d

from .conv import CNNBlock
from .transformer_deformable_att import TransformerBlock
from spatial_correlation_sampler import SpatialCorrelationSampler
# from .utils.utils import coords_grid

def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=1, bias=True),
        nn.LeakyReLU(0.2)
    )
    
def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class Net(nn.Module):
    def __init__(self, image_size=(350, 350), expansion=2, patch_size=(2, 2), md=4):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        depth = [2, 3, 1]
        heads = [16, 32, 32]

        self.pool_rgb = nn.Sequential(
            SphereConv2d(3, 32, stride=1, groups=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            SphereMaxPool2d(stride=2)
        )
        
        self.cnn_blocks = nn.ModuleList([])
        self.cnn_blocks.append(CNNBlock(32, 64, stride=2, expansion=expansion, dropout=0.05))
        self.cnn_blocks.append(CNNBlock(64, 128, stride=2, expansion=expansion, dropout=0.05))
        self.cnn_blocks.append(CNNBlock(128, 256, stride=2, expansion=expansion, dropout=0.05))
        
        self.transformer_blocks_rgb = nn.ModuleList([])
        self.transformer_blocks_rgb.append(TransformerBlock(fmap_size=(88, 88), dim=64, depth=depth[0], channel=64, heads=2, dim_head=64//2, expansion=4, n_groups=4, stride=8, kernel_size=(3,3), patch_size=(2,2), dropout=0.05))
        self.transformer_blocks_rgb.append(TransformerBlock(fmap_size=(44, 44), dim=128, depth=depth[1], channel=128, heads=4, dim_head=128//4, expansion=4, n_groups=4, stride=8, kernel_size=(3,3), patch_size=(2,2), dropout=0.05))
        self.transformer_blocks_rgb.append(TransformerBlock(fmap_size=(22, 22), dim=256, depth=depth[2], channel=256, heads=8, dim_head=256//8, expansion=4, n_groups=4, stride=8, kernel_size=(3,3), patch_size=(2,2), dropout=0.05))
        
        self.pool_depth = nn.Sequential(
            SphereConv2d(3, 32, stride=1, groups=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            SphereMaxPool2d(stride=2)
        )
        
        self.cnn_blocks_depth = nn.ModuleList([])
        self.cnn_blocks_depth.append(CNNBlock(32, 64, stride=2, expansion=expansion, dropout=0.05))
        self.cnn_blocks_depth.append(CNNBlock(64, 128, stride=2, expansion=expansion, dropout=0.05))
        self.cnn_blocks_depth.append(CNNBlock(128, 256, stride=2, expansion=expansion, dropout=0.05))
        
        self.transformer_blocks_depth = nn.ModuleList([])
        self.transformer_blocks_depth.append(TransformerBlock(fmap_size=(88, 88), dim=64, depth=depth[0], channel=64, heads=2, dim_head=64//2, expansion=4, n_groups=4, stride=8, kernel_size=(3,3), patch_size=(2,2), dropout=0.05))
        self.transformer_blocks_depth.append(TransformerBlock(fmap_size=(44, 44), dim=128, depth=depth[1], channel=128, heads=4, dim_head=128//4, expansion=4, n_groups=4, stride=8, kernel_size=(3,3), patch_size=(2,2), dropout=0.05))
        self.transformer_blocks_depth.append(TransformerBlock(fmap_size=(22, 22), dim=256, depth=depth[2], channel=256, heads=8, dim_head=256//8, expansion=4, n_groups=4, stride=8, kernel_size=(3,3), patch_size=(2,2), dropout=0.05))

        self.corr = SpatialCorrelationSampler(
                                kernel_size=1,
                                patch_size=9,
                                stride=1,
                                padding=0,
                                dilation=1,
                                dilation_patch=2)
        self.corr_act = nn.LeakyReLU(0.2)
        
        nd = (2 * md + 1) ** 2  #81
        dd = np.cumsum([128, 128, 64, 32])
        add_list = [128, 64]
        
        od = nd
        self.conv3_0 = myconv(od, 128, kernel_size=3, padding=1)
        self.conv3_1 = myconv(od + dd[0], 128, kernel_size=3, padding=1)
        self.conv3_2 = myconv(od + dd[1], 64, kernel_size=3, padding=1)
        self.conv3_3 = myconv(od + dd[2], 32, kernel_size=3, padding=1)
        
        self.predict_flow3 = predict_flow(od + dd[3])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[3], 2, kernel_size=4, stride=2, padding=1)
        
        od = nd + add_list[0] + 4
        self.conv2_0 = myconv(od, 128, kernel_size=3, padding=1)
        self.conv2_1 = myconv(od + dd[0], 128, kernel_size=3, padding=1)
        self.conv2_2 = myconv(od + dd[1], 64, kernel_size=3, padding=1)
        self.conv2_3 = myconv(od + dd[2], 32, kernel_size=3, padding=1)
        
        self.predict_flow2 = predict_flow(od + dd[3])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat2 = deconv(od + dd[3], 2, kernel_size=4, stride=2, padding=1)
        
        od = nd + add_list[1] + 4
        self.conv1_0 = myconv(od, 128, kernel_size=3, padding=1)
        self.conv1_1 = myconv(od + dd[0], 128, kernel_size=3, padding=1)
        self.conv1_2 = myconv(od + dd[1], 64, kernel_size=3, padding=1)
        self.conv1_3 = myconv(od + dd[2], 32, kernel_size=3, padding=1)        
        
        self.calib_conv1 = myconv(od + dd[3], 128, kernel_size=3, stride=1, padding=1, dilation=7)
        self.calib_conv2 = myconv(128, 128, kernel_size=3, stride=1, padding=1, dilation=5)
        self.calib_conv3 = myconv(128, 64, kernel_size=3, stride=1, padding=1, dilation=3)
        self.calib_conv4 = myconv(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d(4)
        self.fcn = nn.Linear(1024, 256)
        self.act_fcn = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.05)
        
        self.pred_q = nn.Linear(256, 4)
        self.pred_t = nn.Linear(256, 3)
        
    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        mask = torch.floor(torch.clamp(mask, 0, 1))

        return output * mask
        
    def forward(self, x_rgb, x_depth):
        x_rgb_pool = self.pool_rgb(x_rgb)
        x_depth_pool = self.pool_depth(x_depth)
        
        x1_rgb = self.cnn_blocks[0](x_rgb_pool)
        x1_rgb = self.transformer_blocks_rgb[0](x1_rgb)
        x1_depth = self.cnn_blocks_depth[0](x_depth_pool)
        x1_depth = self.transformer_blocks_depth[0](x1_depth)
        
        x2_rgb = self.cnn_blocks[1](x1_rgb)
        x2_rgb = self.transformer_blocks_rgb[1](x2_rgb)
        x2_depth = self.cnn_blocks_depth[1](x1_depth)
        x2_depth = self.transformer_blocks_depth[1](x2_depth)
        
        x3_rgb = self.cnn_blocks[2](x2_rgb)
        x3_rgb = self.transformer_blocks_rgb[2](x3_rgb)
        x3_depth = self.cnn_blocks_depth[2](x2_depth)
        x3_depth = self.transformer_blocks_depth[2](x3_depth)

        x3_corr = self.corr(x3_rgb, x3_depth)
        x3_corr = x3_corr.permute(0, 3, 4, 1, 2)
        x3_corr = x3_corr.reshape(x3_corr.shape[0], -1, x3_corr.shape[1], x3_corr.shape[2]) 
        x3_corr = self.corr_act(x3_corr)

        x3 = torch.cat((self.conv3_0(x3_corr), x3_corr), 1)
        x3 = torch.cat((self.conv3_1(x3), x3), 1)
        x3 = torch.cat((self.conv3_2(x3), x3), 1)
        x3 = torch.cat((self.conv3_3(x3), x3), 1)
        
        x3_flow = self.predict_flow3(x3)
        x3_up_flow = self.deconv3(x3_flow)
        x3_up_feat = self.upfeat3(x3)
        
        x2_warp = self.warp(x2_depth, x3_up_flow*1.25)
        x2_corr = self.corr(x2_rgb, x2_warp)
        x2_corr = x2_corr.permute(0, 3, 4, 1, 2)
        x2_corr = x2_corr.reshape(x2_corr.shape[0], -1, x2_corr.shape[1], x2_corr.shape[2]) 
        x2_corr = self.corr_act(x2_corr)
        
        x2 = torch.cat((x2_corr, x2_rgb, x3_up_flow, x3_up_feat), 1)
        x2 = torch.cat((self.conv2_0(x2), x2), 1)
        x2 = torch.cat((self.conv2_1(x2), x2), 1)
        x2 = torch.cat((self.conv2_2(x2), x2), 1)
        x2 = torch.cat((self.conv2_3(x2), x2), 1)
        
        x2_flow = self.predict_flow2(x2)
        x2_up_flow = self.deconv2(x2_flow)
        x2_up_feat = self.upfeat2(x2)
        
        x1_warp = self.warp(x1_depth, x2_up_flow*2.5)
        x1_corr = self.corr(x1_rgb, x1_warp)
        x1_corr = x1_corr.permute(0, 3, 4, 1, 2)
        x1_corr = x1_corr.reshape(x1_corr.shape[0], -1, x1_corr.shape[1], x1_corr.shape[2]) 
        x1_corr = self.corr_act(x1_corr)
        
        x1 = torch.cat((x1_corr, x1_rgb, x2_up_flow, x2_up_feat), 1)
        x1 = torch.cat((self.conv1_0(x1), x1), 1)
        x1 = torch.cat((self.conv1_1(x1), x1), 1)
        x1 = torch.cat((self.conv1_2(x1), x1), 1)
        x1 = torch.cat((self.conv1_3(x1), x1), 1)
        
        x = self.calib_conv1(x1)
        x = self.calib_conv2(x)
        x = self.calib_conv3(x)
        x = self.calib_conv4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fcn(x)
        x = self.act_fcn(x)
        x = self.dropout(x)
        
        pred_q = self.pred_q(x)
        pred_t = self.pred_t(x)
        
        pred_q_norm = pred_q / (torch.sqrt(torch.sum(pred_q * pred_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        
        return pred_q_norm, pred_t