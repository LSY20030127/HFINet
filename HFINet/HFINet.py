import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import weight_init, trunc_normal_
import numpy as np
from libs.backbone.pvtv2 import pvt_v2_b2
class ChannelAttention(nn.Module):
    """
    CBAM混合注意力机制的通道注意力
    """

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            # 全连接层
            # nn.Linear(in_planes, in_planes // ratio, bias=False),
            # nn.ReLU(),
            # nn.Linear(in_planes // ratio, in_planes, bias=False)

            # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
            nn.Conv2d(64, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.conv128_64 = nn.Conv2d(in_channels, 64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
       avg_out = self.fc(self.avg_pool(x))
       max_out = self.fc(self.max_pool(x))
       out = avg_out + max_out
       out = self.sigmoid(out)
       out  = self.conv128_64(out)
       return out * x
class SpatialAttention(nn.Module):
    """
    CBAM混合注意力机制的空间注意力
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x
class CBAM(nn.Module):
    """
    CBAM混合注意力机制
    """

    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x
class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, bias=False, bn=True, relu=True):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
            # conv.append(nn.LayerNorm(out_channel, eps=1e-6))
        if relu:
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
class CLFI(nn.Module):
    def __init__(self, hchannel, mchannel, lchannel):
        super(CLFI, self).__init__()
        self.Conv512_320 = basicConv(512, mchannel, k=1)
        self.Conv640_128 = basicConv(hchannel, lchannel, k=1)

    def forward(self, high, middle, low):
        f4 = F.interpolate(high, size=(middle.shape[2], middle.shape[3]), mode='bilinear', align_corners=False)
        fc34 = torch.cat((f4, middle), 1)  # 7, 640, 16, 16

        fc34 = F.interpolate(fc34, size=(low.shape[2], low.shape[3]), mode='bilinear', align_corners=False)
        fc34 = self.Conv640_128(fc34)  # 7, 128, 32, 32
        fc234 = torch.cat((fc34, low), 1)  # 7, 256, 32, 32

        return fc234
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.In = nn.InstanceNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return x

    #####################    LDM   ######################
class LFR(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LFR, self).__init__()
        self.relu = nn.ReLU(True)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv3x3 = BasicConv2d(out_channel, out_channel, 3, padding=0)
        self.conv1x1 = BasicConv2d(in_channel, out_channel, 1)
        self.conv3x3_3 = BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        self.conv3x3_5 = BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        self.conv3x3_7 = BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        self.conv1x3 = BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))
        self.conv3x1 = BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0))
        self.conv1x5 = BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2))
        self.conv5x1 = BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0))
        self.conv1x7 = BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3))
        self.conv7x1 = BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0))
        self.conv5x3 = BasicConv2d(out_channel, out_channel, kernel_size=(5, 3), padding=(2, 1))
        self.conv3x5 = BasicConv2d(out_channel, out_channel, kernel_size=(3, 5), padding=(1, 2))
        self.conv3x7 = BasicConv2d(out_channel, out_channel, kernel_size=(3, 7), padding=(1, 3))
        self.conv7x3 = BasicConv2d(out_channel, out_channel, kernel_size=(7, 3), padding=(3, 1))

        self.conv_cat = BasicConv2d(5 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(out_channel, out_channel, 1)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.branch0 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.eps = 1e-5
        self.IterAtt = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 8, kernel_size=1),
            nn.LayerNorm([out_channel // 8, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel // 8, out_channel, kernel_size=1)
        )
        self.ConvOut = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )


        t = int(abs((2 + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = nn.Conv2d(out_channel, out_channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.cbam = CBAM(in_channel, out_channel)
    def forward(self, x):
        x = self.conv1x1(x)

        x0 = self.conv3x3(x)

        x1 = self.conv3x3_5(self.conv5x1(self.conv3x5(self.conv1x3(self.max_pool(x)))))
        x2 = self.conv3x3_3(self.conv3x1(self.conv5x3(self.conv1x5(self.max_pool(x)))))
        x3 = self.conv3x3_3(self.conv3x1(self.conv7x3(self.conv1x7(self.max_pool(x)))))
        x4 = self.conv3x3_7(self.conv7x1(self.conv3x7(self.conv1x3(self.max_pool(x)))))

        x1 = x1 * x

        x2 = x2 * x

        x3 = x3 * x
        x4 = x4 * x

        x1 = self.branch1(x1)
        x2 = self.branch0(x2)
        x3 = self.branch0(x3)
        x4 = self.branch2(x4)
        x1 = self.cbam(x1)
        x2 = self.cbam(x2)
        x3 = self.cbam(x3)
        x4 = self.cbam(x4)
        x_cat = self.conv_cat(torch.cat((x, x1, x2, x3, x4), 1))
        # print(x_cat.size())
        x_cat = self.conv_res(x_cat)
        x0 = self.cbam(x0)

        x0 = F.interpolate(x0, size=(64,64), mode='bilinear', align_corners=True)
        x_cat = F.interpolate(x_cat, size=(64,64), mode='bilinear', align_corners=True)
        fuse = self.relu(x_cat + x0)
        # can change to the MS-CAM or SE Attention, refer to lib.RCAB.

        context = (fuse.pow(2).sum((2, 3), keepdim=True) + self.eps).pow(0.5)  # [B, C, 1, 1]
        channel_add_term = self.IterAtt(context)
        out = channel_add_term * fuse + fuse

        # ConvOut: 3x3 Conv Layer
        out = self.ConvOut(out)
        # Residual Connection
        out = out +out
        # Conv2D: Further Conv Layer
        out = self.conv2d(out)
        # Adaptive Average Pooling
        wei = self.avg_pool(out)
        # 1D Conv Layer
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Sigmoid Activation
        wei = self.sigmoid(wei)
        # Element-wise Multiplication
        out = out * wei
        return out
class GAM(nn.Module):
    def __init__(self, channel_in=64):
        super(GAM, self).__init__()
        self.query_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)
        self.key_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)

        self.scale = 1.0 / (channel_in ** 0.5)

        self.conv6 = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)

        for layer in [self.query_transform, self.key_transform, self.conv6]:
            c2_msra_fill(layer)

    def forward(self, x5):
        # x: B,C,H,W
        # x_query: B,C,HW
        B, C, H5, W5 = x5.size()
        # print('GAM input:',B, C, H5, W5)

        x_query = self.query_transform(x5).view(B, C, -1)
        # print('x_query1:',x_query.shape)

        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C)  # BHW, C
        # print('x_query2:',x_query.shape)
        # x_key: B,C,HW
        x_key = self.key_transform(x5).view(B, C, -1)
        # print('x_key1:',x_key.shape)

        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1)  # C, BHW
        # print('x_key2:',x_key.shape)

        # W = Q^T K: B,HW,HW
        x_w = torch.matmul(x_query, x_key)  # * self.scale # BHW, BHW
        x_w = x_w.view(B * H5 * W5, B, H5 * W5)
        x_w = torch.max(x_w, -1).values  # BHW, B
        x_w = x_w.mean(-1)
        # x_w = torch.mean(x_w, -1).values # BHW
        x_w = x_w.view(B, -1) * self.scale  # B, HW
        x_w = F.softmax(x_w, dim=-1)  # B, HW
        x_w = x_w.view(B, H5, W5).unsqueeze(1)  # B, 1, H, W

        x5 = x5 * x_w
        x5 = self.conv6(x5)

        return x5
class MSCA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MSCA, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        xl = self.local_att(x)
        try:
            xg = self.global_att(x)
        except:
            xg = x
        xlg = xl + xg
        wei = self.sig(xlg)
        return wei

class DGFF(nn.Module):
    def __init__(self, channel=64):
        super(DGFF, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.h2l = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.h2h = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.mscah = MSCA()
        self.mscal = MSCA()
        self.gam = GAM()
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)

        # Global average pooling and projection to shared feature space
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(channel, 128)  # 128 is the embedding size

    def forward(self, x, y):
        # Self-attention using GAM on high-res feature map
        x = self.gam(x)
        y = self.gam(y)
        # High and low resolution feature processing
        x_h = self.h2h(x)
        x_l = self.h2l(self.h2l_pool(y))
        x_h = x_h * self.mscah(x_h)
        x_l = x_l * self.mscal(x_l)

        # Upsample and combine high and low res features
        x_h = F.interpolate(x_h, size=(62, 62), mode='bilinear', align_corners=True)
        x_l = F.interpolate(x_l, size=(62, 62), mode='bilinear', align_corners=True)
        out = x_h + x_l
        out = self.conv(out)

        return out

def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Module, Tensor]`.
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)
class HFINet(nn.Module):
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(HFINet, self).__init__()
        # backbone
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pth/backbone/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # CLFI模块
        self.clfi1 = CLFI(832, 64, lchannel=64)  # 只保留一个CLFI
        self.clfi2 = CLFI(448, 64, lchannel=64)  # 只保留一个CLFI

        # LFR 模块
        self.LFR1 = LFR(64, 64)
        self.LFR2 = LFR(64, 64)
        self.LFR3 = LFR(128, 64)
        self.LFR4 = LFR(192, 64)

        self.conv64_1 = nn.Conv2d(64, 1, 1)
        self.conv192_64 = nn.Conv2d(192, 64, 1)
        self.conv320_64 = nn.Conv2d(320, 64, 1)
        self.conv128_64 = nn.Conv2d(128, 64, 1)
        self.conv128_64 = nn.Conv2d(128, 64, 1)
        self.conv512_64 = nn.Conv2d(512, 64, 1)
        self.dgff = DGFF(64)
        self.upconv3 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1_s = pvt[0]  # 2, 64, 64, 64
        x2_s = pvt[1]  # 2, 128, 32, 32
        x3_s = pvt[2]  # 2, 320, 16, 16
        x4_s = pvt[3]  # 2, 512, 8, 8
        # 只调用一次 CLFI
        x4 = self.clfi1(x4_s, x3_s, x2_s)  # 2, 192, 32, 32
        x3 = self.clfi2(x3_s, x2_s, x1_s)  # 2, 128, 64, 64
        
        # LFR 和 DGFF 应用
        ldm_output_1 = self.LFR1(x1_s)
        x2_s = self.conv128_64(x2_s)  # 对 x2_s 进行 1x1 卷积
        # DGFF 需要传入 x1_s 和 x2_s，得到的输出经过 upconv3 处理
        out1 = self.upconv3(self.dgff(x1_s, x2_s) + ldm_output_1)  # 输入 x1_s, x2_s 作为 DGCM 的 x 和 y

        ldm_output_2 = self.LFR2(x2_s)
        x3_s = self.conv320_64(x3_s)  # �� x3_s 进行 1x1 卷积
        # DGFF 需要传入 x2_s 和 x3_s，得到的输出经过 upconv3 处理
        out2 = self.upconv3(self.dgff(x2_s, x3_s) + ldm_output_2)  # 输入 x2_s, x3_s 作为 DGCM 的 x 和 y

        ldm_output_3 = self.LFR3(x3)  # 注意，x3 是通过 clfi 处理后的结果
        x4_s = self.conv512_64(x4_s)  # 对 x4_s 进行 1x1 卷积
        # DGFF 需要传入 x3 和 x4_s，得到的输出经过 upconv3 处理
        x3 = self.conv128_64(x3)
        out3 = self.upconv3(self.dgff(x3_s, x4_s) + ldm_output_3)  # 输入 x3, x4_s 作为 DGCM 的 x 和 y

        ldm_output_4 = self.LFR4(x4)  # 注意，x4 是通过 clfi 处理后的结果
        x4 = self.conv192_64(x4)  # 对 x4 进行 1x1 卷积
        # DGFF 需要传入 x4 和 x3_s，得到的输出经过 upconv3 处理
        out4 = self.upconv3(self.dgff(x3, x4) + ldm_output_4)  # 输入 x4, x3_s 作为 DGCM 的 x 和 y

        # 融合所有特征图
        out = out1 + out2 + out3 + out4  # 将所有不同尺度的输出加和
        out = self.conv64_1(out)  # 通过 1x1 卷积调整通道数
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)  # 上采样到 256x256

        return out

if __name__ == '__main__':
    import numpy as np
    from time import time

    net = HFINet(imagenet_pretrained=False)
    net.eval()
    dump_x = torch.randn(2, 3, 256, 256)
    # frame_rate = np.zeros((1000, 1))
    start = time()
    y = net(dump_x)
    print(y.shape)
