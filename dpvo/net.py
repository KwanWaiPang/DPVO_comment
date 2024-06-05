import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch_scatter
from torch_scatter import scatter_sum

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .extractor import BasicEncoder, BasicEncoder4
from .blocks import GradientClip, GatedResidual, SoftAgg

from .utils import *
from .ba import BA
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt

DIM = 384

class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))
        
        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """

        net = net + inp + self.corr(corr)
        net = self.norm(net)

        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii*12345 + jj)

        net = self.gru(net)

        return net, (self.d(net), self.w(net), None)

# 一个继承自 nn.Module 的类，负责将输入图像分割成小块（patch）并进行处理。
class Patchifier(nn.Module):
    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size #patch_size为3（传入及默认的都为3）
        # 卷积网络的编码器。主要作用是对输入图像进行特征提取，经过多个卷积层和归一化层的处理，最后输出一个指定维度的特征图。
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance') #输出维度为128，norm_fn为归一化形式（进行patch特征的提取）
        self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')#输出维度为384（DIM）
    
    # 计算图像的梯度
    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    # 用于从输入图像中提取特征块。
    def forward(self, images, patches_per_image=80, disps=None, gradient_bias=False, return_color=False):
        """ extract patches from input images """

        # 进行特征提取（将提取的特征图缩放到四分之一大小。）
        fmap = self.fnet(images) / 4.0 #通过fnet对输入图像进行特征提取，然后除以4.0，获取特征图 fmap
        imap = self.inet(images) / 4.0 #通过inet对输入图像进行特征提取，然后除以4.0，获取内部特征图 imap

        # 获取特征图的形状，分别是批次大小 b、图像数量 n、通道数 c、高度 h 和宽度 w。
        b, n, c, h, w = fmap.shape

        # bias patch selection towards regions with high gradient
        if gradient_bias:#参数传入为false
            # 计算图像的梯度
            g = self.__image_gradient(images)

            # 随机生成多个候选坐标 x 和 y。
            x = torch.randint(1, w-1, size=[n, 3*patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, 3*patches_per_image], device="cuda")
            # torch.randint 用于生成范围在 [1, w-1) 和 [1, h-1) 之间的随机整数。
            # n 是图像的数量，patches_per_image 是每张图像的补丁数量。

            # 将 x 和 y 坐标堆叠在一起形成一个新的张量 coords，其形状为 [n, 3*patches_per_image, 2]。
            # dim=-1 表示在最后一个维度上堆叠，使得每个坐标对 (x, y) 成为二维坐标。
            coords = torch.stack([x, y], dim=-1).float()
            # 根据坐标 coords 从梯度图 g 中提取patch。然后展平成一维张量。
            g = altcorr.patchify(g, coords, 0).view(-1)
            
            ix = torch.argsort(g)#进行梯度值的排序
            # 提取梯度值最大的 patches_per_image 个坐标。
            x = x[:, ix[-patches_per_image:]]
            y = y[:, ix[-patches_per_image:]]

        else:#若为false，则直接随机生成坐标
            x = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")
        
        coords = torch.stack([x, y], dim=-1).float() #patch的坐标
        # 获取对应patch坐标的特征图
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords, 1).view(b, -1, 128, 3, 3)

        #如果需要返回颜色信息，则从图像中提取对应的颜色特征块 clr（直接从图像提取，而前面的是从特征图中提取）。
        if return_color: 
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)

        # 如果未提供视差图，则创建一个全为1的视差图。
        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        # 坐标网格生成（用来表示每个像素在图像序列中的空间和时间位置。）
        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        # 从网格中提取特征块 patches。
        patches = altcorr.patchify(grid[0], coords, 1).view(b, -1, 3, 3, 3)

        # 生成特征块的索引 index。（这个索引可以不用管，因为返回了也不用~~~）
        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        # 获取信息：从图像中提取特征图（fmap）、全局特征图（gmap）、内部特征图（imap）和图像块（patches），同时获取颜色信息（clr）
        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)

# 定义的VO网络
class VONet(nn.Module):#一个继承自nn.Module的类，表示一个神经网络模型。
    # 初始化（构造）函数
    def __init__(self, use_viewer=False):
        super(VONet, self).__init__()
        self.P = 3  #patch size为3
        self.patchify = Patchifier(self.P)#Patchifier类的实例化对象，进行特征提取
        self.update = Update(self.P)#Update类的实例化对象，进行更新操作

        self.DIM = DIM  #为384？？？
        self.RES = 4


    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False):
        """ Estimates SE3 or Sim3 between pair of frames """

        images = 2 * (images / 255.0) - 0.5
        intrinsics = intrinsics / 4.0
        disps = disps[:, :, 1::4, 1::4].float()

        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)

        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"))
        ii = ix[kk]

        imap = imap.view(b, -1, DIM)
        net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)
        
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"))
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"))

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))

        return traj

