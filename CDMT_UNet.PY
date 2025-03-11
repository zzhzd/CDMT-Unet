
import torch.nn as nn
from collections import OrderedDict
import torch
from timm.models.layers import DropPath, trunc_normal_

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),torch.min(x, 1)[0].unsqueeze(1)), dim=1)
from pytorch_wavelets import DWTForward

# GitHub地址 ：https://github.com/apple1986/HWD
# 论文地址：https://www.sciencedirect.com/science/article/pii/S0031320323005174
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()

        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.zp=ZPool()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch+6, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        yL, yH = self.wt(x)
        y_HL = self.zp(yH[0][:, :, 0, ::])
        y_LH = self.zp(yH[0][:, :, 1, ::])
        y_HH = self.zp(yH[0][:, :, 2, ::])
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x*y

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


'''
提取特征约等于注意力机制
'''


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)
        # self.pool=nn.LPPool2d(3,1,1)

    def forward(self, x):
        return self.pool(x) - x


'''

特征提取和特征整合
'''


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, out_features, 1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return x



class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim,out, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0.2, drop_path=0.2,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        # self.token_mixer = Pooling(pool_size=pool_size)
        # self.token_mixer = ZPool()
        self.token_mixer = Down_wt(dim,out)
        self.norm2 = norm_layer(out)
        mlp_hidden_dim = int(out * mlp_ratio)
        self.mlp = Mlp(in_features=out, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()



    def forward(self, x):
        x = self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PoolFormerLayer(nn.Module):
    def __init__(self):
        super(PoolFormerLayer,self).__init__()
        self.layer1=PoolFormerBlock(3,128)
        self.layer2=PoolFormerBlock(128,256)
        self.layer3=PoolFormerBlock(256,512)
        self.layer4=PoolFormerBlock(512,1024)
    # def forward(self,x):
    #     x=self.layer1(x)
    #     out3=self.layer2(x)
    #     out4=self.layer3(out3)
    #     out5=self.layer4(out4)
    #     # print(f"out3.shape: {out3.shape}, out4: {out4.shape}, out5: {out5.shape} ")
    #     return out3,out4,out5

# conv_bn为网络的第一个卷积块，步长为2
def conv_bn(inp, oup, stride=2):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


# conv_dw为深度可分离卷积
def conv_dw(inp, oup, stride=2):
    return nn.Sequential(
        # 3x3卷积提取特征，步长为2
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        # 1x1卷积，步长为1
        nn.Conv2d(inp, oup, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, n_channels):
        super(MobileNet, self).__init__()
        self.layer1 = nn.Sequential(
            # 第一个卷积块，步长为2，压缩一次
            conv_bn(n_channels, 128, 1),  # 416,416,3 -> 208,208,32

            # 第一个深度可分离卷积，步长为1
            conv_dw(128, 128, 1),  # 208,208,32 -> 208,208,64

            # 两个深度可分离卷积块
            conv_dw(128, 128, 2),  # 208,208,64 -> 104,104,128
            conv_dw(128, 128, 1),

            # 104,104,128 -> 52,52,256
            conv_dw(128, 256, 1),
            conv_dw(256, 256, 1),
        )
        # 52,52,256 -> 26,26,512
        self.layer2 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        # 26,26,512 -> 13,13,1024
        self.layer3 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)
    # 定义数据转换


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups,
                           bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))
class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.maxpool=nn.MaxPool2d(3,1,padding=1)
        self.avgpool=nn.AvgPool2d(3,1,padding=1)
    def forward(self, x,y):
        x = self.maxpool(x)
        y = self.avgpool(y)
        return x-y

class mobilenet(nn.Module):
    def __init__(self, n_channels):
        super(mobilenet, self).__init__()
        self.model = MobileNet(n_channels)
        self.model1= PoolFormerLayer()
        self.fussion=Fusion()

    def forward(self, x):
        x=self.model1.layer1(x)
        out3 = self.model.layer1(x)
        out3_1 = self.model1.layer2(x)
        out4 = self.model.layer2(out3_1)
        out4_1 = self.model1.layer3(out3)
        out5 = self.model.layer3(out4_1)
        out5_1 = self.model1.layer4(out4)
        return self.fussion(out3,out3_1), self.fussion(out4,out4_1), self.fussion(out5,out5_1)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CDMT_UNet(nn.Module):
    def __init__(self, n_channels, num_classes):
        super(CDMT_UNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes

        # ---------------------------------------------------#
        #   64,64,256；32,32,512；16,16,1024
        # ---------------------------------------------------#
        self.backbone = mobilenet(n_channels)
        # self.backbone = PoolFormerLayer()

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = DoubleConv(1024, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = DoubleConv(512, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        # nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv4 = DoubleConv(128, 64)

        self.oup = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)
        P5 = self.up1(x0)
        P5 = self.conv1(P5)  # P5: 26x26x512
        # print(P5.shape)
        P4 = x1  # P4: 26x26x512
        P4 = torch.cat([P4, P5], axis=1)  # P4(堆叠后): 26x26x1024

        P4 = self.up2(P4)  # 52x52x1024
        P4 = self.conv2(P4)  # 52x52x256
        P3 = x2  # x2 = 52x52x256
        P3 = torch.cat([P4, P3], axis=1)  # 52x52x512

        P3 = self.up3(P3)
        P3 = self.conv3(P3)

        P3 = self.up4(P3)
        P3 = self.conv4(P3)

        out = self.oup(P3)
        # print(f"out.shape is {out.shape}")

        return out
