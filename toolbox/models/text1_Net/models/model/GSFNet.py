from idlelib.configdialog import changes

import torch
import torch.nn as nn
import torch.nn.functional as F
from toolbox.models.text1_Net.models.decoders.MLPDecoder import DecoderHead
from toolbox.models.text1_Net.models.decoders.fcnhead import FCNDecoder
from toolbox.models.text1_Net.models.backbone.convnext import convnext_tiny
from toolbox.models.text1_Net.models.new_utils import Fused_Fourier_Conv_Mixer
from toolbox.models.text1_Net.models.manhattan_attention import VisionRetentionChunk, RetNetRelPos2d
# from toolbox.models.text1_Net.engine.logger import get_logger
from collections import OrderedDict
from toolbox.models.text1_Net.models.net_utils import BasicConv2d, SC

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

# class BaseFusion(nn.Module):
#     def __init__(self, channle):
#         super(BaseFusion, self).__init__()
#         self.channle = channle
#         self.BN = BasicConv2d(2*channle, channle, kernel_size=1, padding=1)
#
#     def forward(self, x1, x2):
#         f = torch.cat((x1, x2), dim=1)
#         f = self.BN(f)
#         return f


class Manhattan_Attention_FUsion(nn.Module):
    def __init__(self, channle):
         super(Manhattan_Attention_FUsion, self).__init__()
         self.channle = channle
         self.pos = RetNetRelPos2d(embed_dim=channle, num_heads=4, initial_value=1, heads_range=3)
         self.retention = VisionRetentionChunk(embed_dim=channle, num_heads=4)

    def forward(self, x1, x2):
         b, c, h, w = x1.shape
         rel_pos = self.pos((h,w), chunkwise_recurrent=True)
         x1 = x1.permute(0, 2, 3, 1)
         x2 = x2.permute(0, 2, 3, 1)
         out = self.retention(x1, x2, rel_pos)
         out = out.permute(0, 3, 1, 2)
         return out

class EncoderDecoder(nn.Module):
    def __init__(self,  norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        channels = [96, 192, 384, 768]
        num_heads = [1, 2, 4, 8]
        self.norm_layer = norm_layer
        self.rgb = convnext_tiny(pretrained=True, drop_path_rate=0.3)
        self.depth = convnext_tiny(pretrained=True, drop_path_rate=0.3)

        # self.BF = nn.ModuleList([
        #     BaseFusion(channels[0]), BaseFusion(channels[1]), BaseFusion(channels[2]), BaseFusion(channels[3])
        # ])

        self.Man = nn.ModuleList([
                    Manhattan_Attention_FUsion(channels[0]), Manhattan_Attention_FUsion(channels[1]),
                    Manhattan_Attention_FUsion(channels[2]), Manhattan_Attention_FUsion(channels[3])
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

        f1 = self.Man[0](rgb_list[0], depth_list[0])
        f2 = self.Man[1](rgb_list[1], depth_list[1])
        f3 = self.Man[2](rgb_list[2], depth_list[2])
        f4 = self.Man[3](rgb_list[3], depth_list[3])



        out = self.decode(f1, f2, f3, f4)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        return out, f1, f2, f3, f4


if __name__ == "__main__":
    model = EncoderDecoder().cuda()
    in_rgb = torch.randn(2, 3, 480, 640).cuda()
    in_depth = torch.randn(2, 3, 480, 640).cuda()

    output= model(in_rgb, in_depth)
    # print(output[-1].shape)
    print(output.shape)