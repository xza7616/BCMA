from idlelib.configdialog import changes

import torch
import torch.nn as nn
import torch.nn.functional as F
# from toolbox.models.text1_Net.models.backbone.p2t import p2t_base
from toolbox.models.text1_Net.models.backbone.mix_transformer import mit_b2
from toolbox.models.text1_Net.models.Sparse_Cross_Attention import Sparse_Cross_Attention
from toolbox.models.text1_Net.models.mamba_utils import SS2D
from toolbox.models.text1_Net.models.new_utils import Fused_Fourier_Conv_Mixer
from toolbox.models.text1_Net.models.decoders.MLPDecoder import DecoderHead
from toolbox.models.text1_Net.models.PConv import Pinwheel_shapedConv
# from toolbox.models.text1_Net.engine.logger import get_logger
from collections import OrderedDict
# logger = get_logger()

from timm.models.layers import trunc_normal_
import math

class BasicConv2d(nn.Module):#基本卷积
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BaseFusion(nn.Module):
    def __init__(self, channle):
        super(BaseFusion, self).__init__()
        self.channle = channle
        self.BN = BasicConv2d(2*channle, channle, kernel_size=1, padding=1)

    def forward(self, x1, x2):
        f = torch.cat((x1, x2), dim=1)
        f = self.BN(f)
        return f

class FUsion(nn.Module):
    def __init__(self, channle, num_head, sparse_size):
        super(FUsion, self).__init__()
        self.channle = channle
        self.spare = Sparse_Cross_Attention(dim=channle,
        num_heads=num_head,
        sparse_size=sparse_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.1,
        attn_drop=0.1 ,
        drop_path=0.1 )
        self.BN = BasicConv2d(channle, channle, kernel_size=1)
        self.conv = nn.Conv2d(channle, channle, kernel_size=1, bias=True)
        self.ss2d_r = SS2D(channle, d_state=16, expand=2, dropout=0)
        self.ss2d_d = SS2D(channle, d_state=16, expand=2, dropout=0)
        self.residual = nn.Conv2d(channle, channle, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channle)
        self.pconv = Pinwheel_shapedConv(channle, channle, 3, 1)

    def forward(self, x1, x2):

        x1 = self.ss2d_r(x1.permute(0, 2, 3, 1))
        x2 = self.ss2d_d(x2.permute(0, 2, 3, 1))
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        meg = self.spare(x1, x2)
        residual = self.residual(meg)
        out = self.pconv(meg)
        out = self.BN(out)
        # print("out", out.shape)
        out = self.norm(out + residual)
        return out


class EncoderDecoder(nn.Module):
    def __init__(self,  norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        channels = [64, 128, 320, 512]
        # num_heads = [1, 2, 4, 8]
        input_resolution = [(120, 160), (60, 80), (30, 40), (15, 20)]
        self.norm_layer = norm_layer
        self.rgb = mit_b2()
        self.depth = mit_b2()
        # self.backbone = P2tBackbone(path)

        self.Fusion = nn.ModuleList([
            FUsion(channels[0], num_head=4, sparse_size=4), FUsion(channels[1], num_head=8, sparse_size=4),
            FUsion(channels[2], num_head=8, sparse_size=2), FUsion(channels[3], num_head=16,sparse_size=1)
        ])
        self.aux_head = None
        print('Using MLP Decoder')
        self.decode = DecoderHead(in_channels=channels, num_classes=8, norm_layer=norm_layer, embed_dim=512)


    def forward(self, rgb, depth):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        rgb_list = self.rgb(rgb)
        depth_list = self.depth(depth)

        f1 = self.Fusion[0](rgb_list[0], depth_list[0])
        f2 = self.Fusion[1](rgb_list[1], depth_list[1])
        f3 = self.Fusion[2](rgb_list[2], depth_list[2])
        f4 = self.Fusion[3](rgb_list[3], depth_list[3])

        out = self.decode(f1, f2, f3, f4)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        return out, f1, f2 , f3, f4

    def load_pre(self, pre_model):
        save_model = torch.load(pre_model)
        model_dict_r = self.rgb.state_dict()
        state_dict_r = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.rgb.load_state_dict(model_dict_r)
        print(f"RGB Loading pre_model ${pre_model}")

        save_model = torch.load(pre_model)
        model_dict_d = self.depth.state_dict()
        state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_d.keys()}
        model_dict_d.update(state_dict_d)
        self.depth.load_state_dict(model_dict_d)
        print(f"Depth Loading pre_model ${pre_model}")

if __name__ == "__main__":
    model = EncoderDecoder().cuda()
    in_rgb = torch.randn(2, 3, 480, 640).cuda()
    in_depth = torch.randn(2, 3, 480, 640).cuda()

    output= model(in_rgb, in_depth)
    # print(output[-1].shape)
    print(output.shape)