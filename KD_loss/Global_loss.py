# modified_global_distill.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.cnn import kaiming_init
from mmengine.model import constant_init, kaiming_init

class GlobalDistillLoss(nn.Module):
    """全局知识蒸馏损失(无前景背景区分)

    Args:
        student_channels (int): 学生特征图的通道数
        teacher_channels (int): 教师特征图的通道数
        temp (float, optional): 注意力温度系数. Default: 0.5
        gamma (float, optional): 注意力掩码损失权重. Default: 0.001
        lambda_(float, optional): 关系损失权重. Default: 0.000005
    """

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 temp=0.5,
                 gamma=0.001,
                 lambda_=0.000005):
        super().__init__()
        self.temp = temp
        self.gamma = gamma
        self.lambda_ = lambda_

        # 通道对齐
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
        else:
            self.align = None

        # 注意力掩码生成器
        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)

        # 通道特征增强
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
        """输入维度: BxCxHxW"""
        assert preds_S.shape[-2:] == preds_T.shape[-2:], "特征图尺寸不匹配"

        # 通道对齐
        if self.align is not None:
            preds_S = self.align(preds_S)

        # 计算注意力
        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)

        # 计算损失项
        mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        rela_loss = self.get_rela_loss(preds_S, preds_T)

        # 加权总损失
        total_loss = self.gamma * mask_loss + self.lambda_ * rela_loss
        return total_loss

    def get_attention(self, preds, temp):
        """计算空间注意力和通道注意力"""
        B, C, H, W = preds.shape

        # 空间注意力 (基于特征绝对值均值)
        fea_map = torch.abs(preds).mean(dim=1, keepdim=True)  # [B,1,H,W]
        spatial_attention = H * W * F.softmax(fea_map.view(B, -1) / temp, dim=1).view(B, H, W)

        # 通道注意力
        channel_map = torch.abs(preds).mean(dim=[2, 3])  # [B,C]
        channel_attention = C * F.softmax(channel_map / temp, dim=1)

        return spatial_attention, channel_attention

    def get_mask_loss(self, C_s, C_t, S_s, S_t):
        """全局注意力对齐损失"""
        return F.l1_loss(C_s, C_t) + F.l1_loss(S_s, S_t)

    def get_rela_loss(self, preds_S, preds_T):
        """全局关系蒸馏"""
        context_s = self.spatial_pool(preds_S, 'student')
        context_t = self.spatial_pool(preds_T, 'teacher')

        # 通道特征增强
        channel_add_s = self.channel_add_conv_s(context_s)
        channel_add_t = self.channel_add_conv_t(context_t)

        return F.mse_loss(preds_S + channel_add_s, preds_T + channel_add_t)

    def spatial_pool(self, x, mode):
        """空间上下文池化"""
        B, C, H, W = x.size()
        if mode == 'student':
            context_mask = self.conv_mask_s(x)  # [B,1,H,W]
        else:
            context_mask = self.conv_mask_t(x)

        context_mask = context_mask.view(B, 1, H * W)
        context_mask = F.softmax(context_mask, dim=2)  # [B,1,HW]
        context = torch.bmm(x.view(B, C, H * W), context_mask.permute(0, 2, 1))  # [B,C,1]
        return context.view(B, C, 1, 1)

    def reset_parameters(self):
        """参数初始化"""
        kaiming_init(self.conv_mask_s)
        kaiming_init(self.conv_mask_t)
        constant_init(self.channel_add_conv_s[-1], 0)
        constant_init(self.channel_add_conv_t[-1], 0)


if __name__ == "__main__":
    # 测试用例
    B, C_S, C_T, H, W = 2, 64, 256, 32, 32

    # 生成随机输入
    preds_S = torch.randn(B, C_S, H, W)
    preds_T = torch.randn(B, C_T, H, W)

    # 初始化损失函数
    loss_func = GlobalDistillLoss(
        student_channels=C_S,
        teacher_channels=C_T,
        gamma=0.001,
        lambda_=0.000005
    )

    # 计算损失
    loss = loss_func(preds_S, preds_T)
    print(f"Global Distillation Loss: {loss.item():.4f}")