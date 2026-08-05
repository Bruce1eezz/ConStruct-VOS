import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Optional, Tuple
import math


class TrajectoryAwareCrossAttention(nn.Module):
    """
    轨迹感知的cross-attention,接口与标准CrossAttention一致
    """

    def __init__(self,
                 dim: int,
                 nhead: int,
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 add_pe_to_qkv: List[bool] = [True, True, False],
                 residual: bool = True,
                 norm: bool = True,
                 num_sampling_points: int = 18,
                 field_dim: int = 3,
                 offset_kernel: int = 3,
                 offset_heads: int = 2):
        """
        Args:
            dim: 特征维度
            nhead: 注意力头数
            dropout: dropout率
            batch_first: 必须为True
            add_pe_to_qkv: 控制是否将位置编码添加到Q/K/V
            residual: 是否使用残差连接
            norm: 是否使用LayerNorm
            num_sampling_points: 采样点数量
            field_dim: field predictor输出维度
            offset_kernel: offset卷积核大小
            offset_heads: offset头数
        """
        super().__init__()

        assert batch_first, "Only batch_first=True is supported"

        # 基础参数
        self.dim = dim
        self.nhead = nhead
        self.num_sampling_points = num_sampling_points
        self.attention_dim = dim // nhead
        self.offset_kernel = offset_kernel
        self.offset_heads = offset_heads
        self.add_pe_to_qkv = add_pe_to_qkv
        self.residual = residual

        # LayerNorm和Dropout
        if norm:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()
        self.dropout = nn.Dropout(dropout)

        # ===== 1. Field Predictor (基于mem特征) =====
        self.field_predictor = self._build_field_predictor(dim, field_dim)

        # ===== 2. Offset Generator =====
        self.offset_generator = self._build_offset_generator(field_dim)

        # ===== 3. Cross-Attention模块 =====
        self.W_Q = nn.Linear(dim, self.attention_dim * nhead)
        self.W_K = nn.Linear(dim, self.attention_dim * nhead)
        self.W_V = nn.Linear(dim, self.attention_dim * nhead)

        # 输出投影
        self.fc_out = nn.Linear(self.attention_dim * nhead, dim)

        # 缓存位置编码
        self.cached_pe_x = None
        self.cached_pe_mem = None
        self.cached_pe_shape_x = None
        self.cached_pe_shape_mem = None

    def _build_field_predictor(self, in_channels, field_dim):
        """构建Field Predictor (UNet结构)"""

        class DoubleConv(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                return self.conv(x)

        class Down(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.mpconv = nn.Sequential(
                    nn.MaxPool2d(2),
                    DoubleConv(in_ch, out_ch)
                )

            def forward(self, x):
                return self.mpconv(x)

        class Up(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
                self.conv = DoubleConv(in_ch, out_ch)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)

        return nn.ModuleDict({
            'in_layer': DoubleConv(in_channels, 16),
            'inc': DoubleConv(16, 16),
            'down1': Down(16, 32),
            'down2': Down(32, 64),
            'up1': Up(64, 32),
            'up2': Up(32, 16),
            'outc': nn.Conv2d(16, field_dim, 1),
            'norm': nn.Sigmoid()
        })

    def _build_offset_generator(self, in_channels):
        """构建Offset Generator"""
        out_channels = 2 * self.offset_kernel * self.offset_kernel * self.offset_heads
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def _infer_spatial_shape(self, num_queries: int):
        """从num_queries推断正方形空间尺寸"""
        sqrt_q = int(math.sqrt(num_queries))
        if sqrt_q * sqrt_q != num_queries:
            raise ValueError(
                f"num_queries ({num_queries}) must be a perfect square. "
                f"Got sqrt={sqrt_q}, but {sqrt_q}^2={sqrt_q * sqrt_q}"
            )
        return sqrt_q, sqrt_q

    def _get_position_encoding(self, H: int, W: int, device, cache_name='x'):
        """获取或创建2D正弦位置编码"""
        cached_pe = self.cached_pe_x if cache_name == 'x' else self.cached_pe_mem
        cached_shape = self.cached_pe_shape_x if cache_name == 'x' else self.cached_pe_shape_mem

        if cached_pe is None or cached_shape != (H, W):
            pe = torch.zeros(1, H, W, self.dim, device=device)

            y_pos = torch.arange(H, device=device).unsqueeze(1).float()
            x_pos = torch.arange(W, device=device).unsqueeze(0).float()

            div_term = torch.exp(torch.arange(0, self.dim, 2, device=device).float() *
                                 (-math.log(10000.0) / self.dim))

            # Y方向位置编码
            y_embed = y_pos * div_term
            pe[0, :, :, 0::2] = torch.sin(y_embed).unsqueeze(1).expand(-1, W, -1)
            pe[0, :, :, 1::2] = torch.cos(y_embed).unsqueeze(1).expand(-1, W, -1)

            # X方向位置编码
            x_embed = x_pos.T * div_term
            pe[0, :, :, 0::2] += torch.sin(x_embed).unsqueeze(0).expand(H, -1, -1)
            pe[0, :, :, 1::2] += torch.cos(x_embed).unsqueeze(0).expand(H, -1, -1)

            if cache_name == 'x':
                self.cached_pe_x = pe
                self.cached_pe_shape_x = (H, W)
            else:
                self.cached_pe_mem = pe
                self.cached_pe_shape_mem = (H, W)

            return pe

        return cached_pe

    def predict_field(self, x):
        """预测特征轨迹场"""
        x = self.field_predictor['in_layer'](x)
        x1 = self.field_predictor['inc'](x)
        x2 = self.field_predictor['down1'](x1)
        x3 = self.field_predictor['down2'](x2)
        x = self.field_predictor['up1'](x3, x2)
        x = self.field_predictor['up2'](x, x1)
        x = self.field_predictor['outc'](x)
        x = self.field_predictor['norm'](x)
        return x

    def generate_offset(self, field):
        """从轨迹场生成偏移量"""
        return self.offset_generator(field)

    def offset_to_indices(self, offset):
        """将偏移量转换为采样索引"""
        B, _, H, W = offset.shape
        kernel = self.offset_kernel

        offset = rearrange(offset, 'b (hd c) h w -> b hd c h w', hd=self.offset_heads)

        anchor_h, anchor_w = torch.meshgrid(
            torch.arange(kernel, device=offset.device),
            torch.arange(kernel, device=offset.device),
            indexing='ij'
        )
        anchor_h = anchor_h - (kernel - 1) / 2
        anchor_w = anchor_w - (kernel - 1) / 2
        anchor = torch.cat((anchor_h[None, ...], anchor_w[None, ...]), dim=0).type_as(offset)
        anchor = rearrange(anchor, 'xy k v -> (xy k v)')

        offset = offset + anchor[None, None, :, None, None]
        offset = rearrange(offset, 'b hd (xy k) h w -> b hd xy k h w', k=kernel * kernel)

        grid_h, grid_w = torch.meshgrid(
            torch.arange(H, device=offset.device),
            torch.arange(W, device=offset.device),
            indexing='ij'
        )
        grid = torch.cat((grid_h[None, ...], grid_w[None, ...]), dim=0).type_as(offset)

        offset = offset + grid[None, None, :, None, :, :]

        offset = rearrange(offset, 'b hd xy kv h w -> b (hd kv) xy h w')
        offset = offset[:, :, 0, ...] * W + offset[:, :, 1, ...]

        offset = rearrange(offset, 'b c h w -> b (h w) c')
        offset = torch.clamp(offset.round().long(), 0, H * W - 1)

        selected_indices = offset[:, :, :self.num_sampling_points]

        return selected_indices

    def trajectory_cross_attention(self, Q_flat, K_flat, V_flat, selected_indices, need_weights=False):
        """
        轨迹感知的cross-attention计算

        Args:
            Q_flat: (B*N, H_q*W_q, C) query特征
            K_flat: (B*N, H_k*W_k, C) key特征
            V_flat: (B*N, H_k*W_k, C) value特征
            selected_indices: (B*N, H_q*W_q, num_sampling_points) 采样索引
            need_weights: 是否返回attention权重
        """
        batch_size, num_q, _ = Q_flat.shape
        _, num_k, _ = K_flat.shape

        # 对 selected_indices 做形状自检与修正:
        # 期望形状 (b, num_q, S)，若为 (b, num_k, S) 或其它，则复制/裁剪到每个 query
        if selected_indices.dim() != 3:
            raise ValueError(f"selected_indices must be 3D, got {selected_indices.shape}")
        if selected_indices.shape[1] != num_q:
            # 常见情形: selected_indices 是基于 mem 的 (b, num_k, S)，复制到每个 query
            if selected_indices.shape[1] == num_k:
                selected_indices = selected_indices[:, :1, :].expand(-1, num_q, -1)
            else:
                # 无法直接匹配，兜底为复制第一行到所有 query
                selected_indices = selected_indices[:, :1, :].expand(-1, num_q, -1)

        # 计算Q, K, V
        Q = self.W_Q(Q_flat)
        K = self.W_K(K_flat)
        V = self.W_V(V_flat)

        # 重塑为多头
        Q = Q.view(batch_size, num_q, self.nhead, self.attention_dim)
        K = K.view(batch_size, num_k, self.nhead, self.attention_dim)
        V = V.view(batch_size, num_k, self.nhead, self.attention_dim)

        # 使用selected_indices从K和V中采样
        indices_expanded = selected_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, self.nhead, self.attention_dim
        )

        # Gather along the key dimension (dim=2) using per-query indices
        # K, V: (b, num_k, nhead, attn_dim) -> (b, 1, num_k, nhead, attn_dim)
        # 为每个 query 扩展一份 K/V，然后在 key 维度 (dim=2) 按索引采样
        K_h = K.unsqueeze(1).expand(-1, num_q, -1, -1, -1)
        V_h = V.unsqueeze(1).expand(-1, num_q, -1, -1, -1)
        # indices_expanded: (b, num_q, num_sampling_points, nhead, attn_dim)
        selected_keys = torch.gather(K_h, 2, indices_expanded)
        selected_values = torch.gather(V_h, 2, indices_expanded)

        # 计算cross-attention分数
        attention_scores = torch.einsum('bihd,bijhd->bijh', Q, selected_keys)
        attention_scores = attention_scores / math.sqrt(self.attention_dim)
        attention_weights = F.softmax(attention_scores, dim=2)

        # 加权求和
        context = torch.einsum('bijh,bijhd->bihd', attention_weights, selected_values)
        context = context.reshape(batch_size, num_q, self.nhead * self.attention_dim)

        # 输出投影
        out = self.fc_out(context)

        if need_weights:
            return out, attention_weights
        else:
            return out, None

    def forward(self,
                x: torch.Tensor,
                mem: torch.Tensor,
                x_pe: torch.Tensor,
                mem_pe: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                *,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播 - 接口与标准CrossAttention一致

        Args:
            x: (B*N, Q_x, C) query特征序列
            mem: (B*N, Q_mem, C) key/value特征序列
            x_pe: (B*N, Q_x, C) query的位置编码
            mem_pe: (B*N, Q_mem, C) memory的位置编码
            attn_mask: 注意力掩码(暂不支持)
            need_weights: 是否返回attention权重

        Returns:
            (output, weights): 输出特征和attention权重(可选)
        """
        # 验证输入
        if x.dim() != 3 or mem.dim() != 3:
            raise ValueError(f"Expected 3D inputs, got x: {x.shape}, mem: {mem.shape}")

        batch_size, num_queries_x, channels = x.shape
        _, num_queries_mem, _ = mem.shape

        # 推断空间尺寸
        H_x, W_x = self._infer_spatial_shape(num_queries_x)
        H_mem, W_mem = self._infer_spatial_shape(num_queries_mem)

        # 保存残差
        r = x

        # LayerNorm
        x = self.norm(x)

        # 处理位置编码 - 根据add_pe_to_qkv控制
        if self.add_pe_to_qkv[0]:
            q = x + x_pe
        else:
            q = x

        if any(self.add_pe_to_qkv[1:]):
            mem_with_pe = mem + mem_pe
            k = mem_with_pe if self.add_pe_to_qkv[1] else mem
            v = mem_with_pe if self.add_pe_to_qkv[2] else mem
        else:
            k = v = mem

            # 1. 转换为2D格式
        q_2d = rearrange(q, 'b (h w) c -> b h w c', h=H_x, w=W_x)
        k_2d = rearrange(k, 'b (h w) c -> b h w c', h=H_mem, w=W_mem)
        v_2d = rearrange(v, 'b (h w) c -> b h w c', h=H_mem, w=W_mem)

        # 2. 只对mem(K/V)进行轨迹预测
        mem_conv = rearrange(k_2d, 'b h w c -> b c h w')

        # 3. 预测轨迹场(基于mem)
        field = self.predict_field(mem_conv)

        # 4. 生成偏移量
        offset = self.generate_offset(field)

        # 5. 转换为采样索引
        selected_indices = self.offset_to_indices(offset)
        # 注意: offset_to_indices 是基于 mem(H_mem*W_mem) 返回 (b, H_mem*W_mem, S)
        # 但后续按每个 query 进行采样，需要形状 (b, Q_x, S)。这里采用同一组采样位置复制到所有 query。
        if selected_indices.shape[1] != num_queries_x:
            selected_indices = selected_indices[:, :1, :].expand(-1, num_queries_x, -1)

        # 6. 展平为序列
        q_flat = rearrange(q_2d, 'b h w c -> b (h w) c')
        k_flat = rearrange(k_2d, 'b h w c -> b (h w) c')
        v_flat = rearrange(v_2d, 'b h w c -> b (h w) c')

        # 7. 执行轨迹感知cross-attention
        out, weights = self.trajectory_cross_attention(
            q_flat, k_flat, v_flat, selected_indices, need_weights
        )

        # 8. 构造与标准MultiheadAttention一致的全尺寸权重 (b, nhead, Q_x, H_mem*W_mem)
        if need_weights and weights is not None:
            # weights: (b, Q_x, S, nhead) -> (b, nhead, Q_x, S)
            weights_hw = weights.permute(0, 3, 1, 2).contiguous()
            weights_full = torch.zeros(
                batch_size, self.nhead, num_queries_x, H_mem * W_mem,
                device=weights.device, dtype=weights.dtype
            )
            # selected_indices: (b, Q_x, S) -> (b, nhead, Q_x, S)
            idx = selected_indices.unsqueeze(1).expand(-1, self.nhead, -1, -1)
            # scatter sampled weights into full spatial map
            weights_full.scatter_(dim=-1, index=idx, src=weights_hw)
            weights_out = weights_full
        else:
            weights_out = None

        # 9. 残差连接
        if self.residual:
            return r + self.dropout(out), weights_out
        else:
            return self.dropout(out), weights_out



if __name__ == '__main__':
    # 初始化模块 - 接口与标准CrossAttention一致
    cross_attn = TrajectoryAwareCrossAttention(
        dim=256,
        nhead=8,
        dropout=0.1,
        add_pe_to_qkv=[True, True, False],  # 控制位置编码
        residual=True,  # 使用残差连接
        norm=True,  # 使用LayerNorm
        num_sampling_points=16,  #这个让他和num_queries 一样大，相当于一个query对应一个采样点
        field_dim=3,
        offset_kernel=3,
        offset_heads=2
    )

    # 前向传播
    batch_size = 4
    num_objects = 10
    H = W = 32
    num_queries = H * W
    embed_dim = 256

    # 输入
    x = torch.randn(batch_size * num_objects, num_queries, embed_dim)  # query
    mem = torch.randn(batch_size * num_objects, num_queries, embed_dim)  # key/value
    x_pe = torch.randn(batch_size * num_objects, num_queries, embed_dim)  # query位置编码
    mem_pe = torch.randn(batch_size * num_objects, num_queries, embed_dim)  # memory位置编码

    # 输出
    output, weights = cross_attn(x, mem, x_pe, mem_pe, need_weights=True)

    print(f"Query shape:  {x.shape}")  # (40, 32*32, 256)
    print(f"Memory shape: {mem.shape}")  # (40, 1024, 256)
    print(f"Output shape: {output.shape}")  # (40, 1024, 256)
    print(f"Weights shape: {weights.shape if weights is not None else None}")
