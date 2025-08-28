import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.cnn import kaiming_init
from mmengine.model import constant_init, kaiming_init

class BCMALoss(nn.Module):

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 temp=0.5,
                 gamma=0.5,
                 lambda_=1):
        super().__init__()
        self.temp = temp
        self.gamma = gamma
        self.lambda_ = lambda_


        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
        else:
            self.align = None

        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)

        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))

        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))

        self.reset_parameters()

    def forward(self, preds_S, preds_T):
        assert preds_S.shape[-2:] == preds_T.shape[-2:],

        if self.align is not None:
            preds_S = self.align(preds_S)

        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)

        mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        rela_loss = self.get_rela_loss(preds_S, preds_T)

        total_loss = self.gamma * mask_loss + self.lambda_ * gda_loss
        return total_loss

    def get_attention(self, preds, temp):
        B, C, H, W = preds.shape

        fea_map = torch.abs(preds).mean(dim=1, keepdim=True)  # [B,1,H,W]
        spatial_attention = H * W * F.softmax(fea_map.view(B, -1) / temp, dim=1).view(B, H, W)


        channel_map = torch.abs(preds).mean(dim=[2, 3])  # [B,C]
        channel_attention = C * F.softmax(channel_map / temp, dim=1)
        return spatial_attention, channel_attention

    def get_mask_loss(self, C_s, C_t, S_s, S_t):
        return F.l1_loss(C_s, C_t) + F.l1_loss(S_s, S_t)

    def get_gda_loss(self, preds_S, preds_T):
        context_s = self.spatial_pool(preds_S, 's')
        context_t = self.spatial_pool(preds_T, 't')

        channel_add_s = self.channel_add_conv_s(context_s)
        channel_add_t = self.channel_add_conv_t(context_t)

        return F.mse_loss(preds_S + channel_add_s, preds_T + channel_add_t)

    def spatial_pool(self, x, mode):
        B, C, H, W = x.size()
        if mode == 's':
            context_mask = self.conv_mask_s(x)  # [B,1,H,W]
        else:
            context_mask = self.conv_mask_t(x)

        context_mask = context_mask.view(B, 1, H * W)
        context_mask = F.softmax(context_mask, dim=2)  # [B,1,HW]
        context = torch.bmm(x.view(B, C, H * W), context_mask.permute(0, 2, 1))  # [B,C,1]
        return context.view(B, C, 1, 1)

    def reset_parameters(self):
        kaiming_init(self.conv_mask_s)
        kaiming_init(self.conv_mask_t)
        constant_init(self.channel_add_conv_s[-1], 0)
        constant_init(self.channel_add_conv_t[-1], 0)


