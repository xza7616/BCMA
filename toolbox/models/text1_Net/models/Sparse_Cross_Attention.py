import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath, to_2tuple


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print(x.shape)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        B, N, C = query.shape
        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(key_value).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权值
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def block(x, block_size_h, block_size_w):
    B, H, W, C = x.shape
    pad_h = (block_size_h - H % block_size_h) % block_size_h
    pad_w = (block_size_w - W % block_size_w) % block_size_w

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w

    x = x.reshape(B, Hp // block_size_h, block_size_h, Wp // block_size_w, block_size_w, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    return x, H, Hp, W, Wp, C


def unblock(x, Ho, Wo):
    B, H, W, win_H, win_W, C = x.shape
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H * win_H, W * win_W, C)
    if H * win_H > Ho or W * win_W > Wo:
        x = x[:, :Ho, :Wo, :].contiguous()
    return x


def alter_sparse(x, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    H, W = x.shape[1], x.shape[2]

    assert H % sparse_size == 0 and W % sparse_size == 0, "输入尺寸必须能被稀疏块大小整除"

    grid_size_h = H // sparse_size
    grid_size_w = W // sparse_size
    Hp = sparse_size * grid_size_h
    Wp = sparse_size * grid_size_w

    out, H, Hp, W, Wp, C = block(x, grid_size_h, grid_size_w)
    out = out.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = out.reshape(-1, sparse_size, sparse_size, C)
    out = out.permute(0, 3, 1, 2)


    return out, H, Hp, W, Wp, C


def alter_unsparse(x, Ho, Hp, Wo, Wp, C, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, Hp // sparse_size, Wp // sparse_size, sparse_size, sparse_size, C)

    x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = unblock(x, Ho, Wo)
    out = out.permute(0, 3, 1, 2)

    return out


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Sparse_Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, sparse_size=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.sparse_size = sparse_size

    def forward(self, x_q, x_kv):

        x_q_before = x_q.flatten(2).transpose(1, 2)
        x_kv_before = x_kv.flatten(2).transpose(1, 2)

        x_q_sparse, Hq, Hqp, Wq, Wqp, Cq = alter_sparse(x_q, self.sparse_size)
        x_kv_sparse, Hkv, Hkvp, Wkv, Wkvp, Ckv = alter_sparse(x_kv, self.sparse_size)


        Bq, Nq, Hsq, Wsq = x_q_sparse.shape
        Bkv, Nkv, Hskv, Wskv = x_kv_sparse.shape

        x_q_sparse = x_q_sparse.flatten(2).transpose(1, 2)
        x_kv_sparse = x_kv_sparse.flatten(2).transpose(1, 2)

        x_q_attn = self.attn(self.norm1_q(x_q_sparse), self.norm1_kv(x_kv_sparse))
        x_q_attn = x_q_attn.transpose(1, 2).reshape(Bq, Nq, Hsq, Wsq)

        x_q_unsparse = alter_unsparse(x_q_attn, Hq, Hqp, Wq, Wqp, Cq, self.sparse_size)
        x_q_unsparse = x_q_unsparse.flatten(2).transpose(1, 2)
        B, _, _ = x_q_unsparse.shape

        x_q = x_q_before + self.drop_path(x_q_unsparse)
        x_q = x_q + self.drop_path(self.mlp(self.norm2(x_q), Hq, Wq))

        x_q = x_q.transpose(1, 2).reshape(B, Cq, Hq, Wq)

        return x_q


layer_scale = True
init_value = 1e-6

if __name__ == '__main__':

    B, C, H, W = 1, 64, 120, 160
    input_tensor1 = torch.randn(B, C, H, W)
    input_tensor2 = torch.randn(B, C, H, W)

    dim = C
    num_heads = 4
    sparse_size = 4
    mlp_ratio = 4.0
    qkv_bias = True
    drop = 0.1
    attn_drop = 0.1
    drop_path = 0.1

    sablock = Sparse_Cross_Attention(
        dim=dim,
        num_heads=num_heads,
        sparse_size=sparse_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop=drop,
        attn_drop=attn_drop,
        drop_path=drop_path
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sablock = sablock.to(device)

    input_tensor1 = input_tensor1.to(device)
    input_tensor2 = input_tensor2.to(device)
    output = sablock(input_tensor1, input_tensor2)

    print(f"Output shape: {output.shape}")