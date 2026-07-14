import numpy as np
import torch
from model.basic_block import *
from model.mamba2 import BiMamba2Ac2d

def conv5x5_relu(in_channels, out_channels, stride):
    """5x5卷积层+relu激活函数"""
    return conv(in_channels, out_channels, 5, stride, 2, activation_fn=partial(nn.ReLU, inplace=True))


def deconv5x5_relu(in_channels, out_channels, stride, output_padding):
    """5x5转置卷积层+relu激活函数"""
    return deconv(in_channels, out_channels, 5, stride, 2, output_padding=output_padding,
                  activation_fn=partial(nn.ReLU, inplace=True))


def resblock(in_channels):
    """Resblock 使用5x5卷积核,通道数不改变,不使用批量归一化(BN)和 最后的激活函数(the last activation)"""
    return BasicBlock(in_channels, out_channels=in_channels, kernel_size=5, stride=1, use_batchnorm=False,
                      activation_fn=partial(nn.ReLU, inplace=True), last_activation_fn=None)


class EBlock(nn.Module):
    """编码器(EBlock)由一个5x5conv层+三个Resblock组成,和InBlock输入块组成相同"""
    def __init__(self, in_channels, out_channels, stride):
        super(type(self), self).__init__()
        # 5x5conv层
        self.conv = conv5x5_relu(in_channels, out_channels, stride)
        # 3个ResBlock块
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(out_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)

    def forward(self, x):
        x = self.conv(x)
        x = self.resblock_stack(x)
        return x


class DBlock(nn.Module):
    """解码器(DBlock)由三个Resblock+一层deconv层组成"""
    def __init__(self, in_channels, out_channels, stride, output_padding):
        super(type(self), self).__init__()
        # 3个ResBlock块
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        # 5x5deconv层
        self.deconv = deconv5x5_relu(in_channels, out_channels, stride, output_padding)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.deconv(x)
        return x


class OutBlock(nn.Module):
    """输出块(OutBlock)由三个Resblock+一层conv层组成,将通道数in_channels变为3"""
    def __init__(self, in_channels):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.conv = conv(in_channels, 3, 5, 1, 2, activation_fn=None)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.conv(x)
        return x


class OSRNetT(nn.Module):

    def __init__(self, upsample_fn=partial(torch.nn.functional.interpolate, mode='bilinear'), xavier_init_all=True):
        super(type(self), self).__init__()
        self.upsample_fn = upsample_fn          # 下采样方法(upsample_fn)是双线性插值(bilinear)
        self.input_padding = None
        self.input_padding_h = None # 记录上轮的图片输出

        # 输入块
        self.inblock = EBlock(4 + 3, 16, 1)     # 这里的3+3意思是原本输入图像具有3通道,从上一个输出图像具有3通道
        # 编码块(通道c倍增,高h宽w减半)
        self.eblock1 = EBlock(16, 32, 2)
        self.eblock2 = EBlock(32, 64, 2)

        # convlstm单层
        self.bimamba2 = BiMamba2Ac2d(128, 64, 32)

        # 解码块(通道c倍减,高h宽w翻倍)
        self.dblock1 = DBlock(64, 32, 2, 1)
        self.dblock2 = DBlock(32, 16, 2, 1)
        # 输出块
        self.outblock = OutBlock(16)

        # 初始化参数
        if xavier_init_all:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    torch.nn.init.xavier_normal_(m.weight)

    def forward_step(self, x, hidden_state):
        """单步forward
        Args:
            x:      (b,c,h,w),其中c是6通道(3通道+3通道)
        Returns:
            d3:     (b,c,h,w),其中c是3通道
            h,c:    (b,c,h,w),其中c为128通道
        """
        # 输入块+编码块(通道6(3+3)->32->64->128,h和w在两层编码块变为h/4,w/4)
        e32 = self.inblock(x)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)

        h = self.bimamba2(torch.cat([e128, hidden_state],dim=1)) # 解码块+输出块(通道128->64->32->3,h/4和w/4在两层解码块变为h和w)

        d64 = self.dblock1(h)
        d32 = self.dblock2(d64 + e64)   # 含残差块
        d3 = self.outblock(d32 + e32)   # 含残差块
        return e32, e64, e128, d64, d32, d3, h

    def forward(self, b1_o, b2_o, b3_o, b1_mask, b2_mask, b3_mask):
        """三次不同规模的forward
        Arg:
            b1, b2, b3: 原规模,1/2规模,1/4规模的图片
        Return:
            i1, i2, i3: 经过网络后的原规模,1/2规模,1/4规模的图片
        """

        b1 = torch.torch.cat([b1_o, b1_mask], 1)
        b2 = torch.torch.cat([b2_o, b2_mask], 1)
        b3 = torch.torch.cat([b3_o, b3_mask], 1)
        # input_padding是第一次用于填充1/2规模的输入图片
        if self.input_padding is None or self.input_padding.shape != b3.shape:
            self.input_padding = torch.zeros_like(b3_o)

        if self.input_padding_h is None or self.input_padding_h.shape != b3.shape:
            self.input_padding_h = torch.zeros((b1.shape[0], 64, b1.shape[-2]//8, b1.shape[-1]//8), device='cuda:0')

        fea3_1,fea3_2,fea3_3,fea3_4,fea3_5,i3,fea3_6 = self.forward_step(torch.cat([b3, self.input_padding], 1), self.input_padding_h)
        h = self.upsample_fn(fea3_6, scale_factor=1.5)

        fea2_1,fea2_2,fea2_3,fea2_4,fea2_5, i2,fea2_6 = self.forward_step(torch.cat([b2, self.upsample_fn(i3, scale_factor=1.5)], 1), h)

        h = self.upsample_fn(fea2_6, scale_factor=4/3)


        fea1_1, fea1_2, fea1_3, fea1_4, fea1_5, i1, fea1_6= self.forward_step(torch.cat([b1, self.upsample_fn(i2, scale_factor=4 / 3)], 1), h)


        return ([fea3_1,fea2_1,fea1_1], [fea3_2,fea2_2,fea1_2], [fea3_3,fea2_3,fea1_3],
                [fea3_4,fea2_4,fea1_4], [fea3_5,fea2_5,fea1_5], [fea3_6,fea2_6,fea1_6],
                i1, i2, i3)




