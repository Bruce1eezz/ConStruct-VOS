#Adaptive Thermal Contextualization (ATC)         HeatFlow Spatio-Temporal Context (HFSTC)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- FeedForward (前馈网络) 模块 ---
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, p=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # 使用 GELU 激活函数
            nn.Dropout(p),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p)
        )

    def forward(self, x):
        return self.net(x)


# --- Heat2D (二维热传导模拟) 模块 ---
class Heat2D(nn.Module):
    def __init__(self, res, dim, hidden_dim):
        super().__init__()
        self.res = res  # 空间分辨率，这里假设是 H (或 W)
        self.dim = dim  # 输入特征维度 C
        self.hidden_dim = hidden_dim  # 内部隐藏维度，通常与 dim 相同

        # 深度可分离卷积，用于初步的空间特征提取
        # Groups=dim 实现了深度可分离，每个通道独立卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # 线性层，用于特征的投影和 SiLU 门控的准备
        self.linear = nn.Linear(dim, hidden_dim * 2)
        self.out_linear = nn.Linear(hidden_dim, dim)

        # LayerNorm 归一化，作用于特征维度
        self.out_norm = nn.LayerNorm(hidden_dim)

        # 这个 to_k 层是核心，它将时间嵌入 (freq_embed) 映射为衰减参数 k
        # to_k 的输出维度与特征维度 dim 相同，以便在每个通道上动态调整衰减
        self.to_k = nn.Sequential(
            nn.Linear(dim, dim, bias=True),  # 输入 freq_embed 维度 dim，输出 dim
            nn.ReLU()  # 使用 ReLU 确保 k 值非负
        )

        # 缓存 DCT/IDCT 矩阵和衰减图，避免重复计算
        self.register_buffer('weight_cosn', self.get_cos_map(res))
        self.register_buffer('weight_cosm', self.get_cos_map(res))  # 假设 H=W=res
        self.register_buffer('weight_exp', self.get_decay_map((res, res)))  # 假设 H=W=res

        self.infer_mode = False  # 推理模式标志，可以用于优化计算

    # 生成 DCT/IDCT 的余弦变换矩阵
    @staticmethod
    def get_cos_map(length, device=None):
        x = torch.linspace(0., length - 1, length, device=device)
        y = torch.linspace(0., length - 1, length, device=device)

        # 创建一个网格，用于计算余弦值
        # 这里的 k 是频率索引，n 是空间索引
        # `(k * pi / L) * (n + 0.5)` 是 DCT 变换中的标准项
        # 注意这里是 1D DCT 的 kernel
        n_grid = x[None, :] + 0.5  # [1, length]
        k_grid = y[:, None] * math.pi / length  # [length, 1]

        weight_cos = torch.cos(n_grid * k_grid)  # [length, length]

        # 归一化因子
        weight_cos[0, :] = weight_cos[0, :] / math.sqrt(length)
        weight_cos[1:, :] = weight_cos[1:, :] / math.sqrt(length / 2)
        return weight_cos

    # 生成频率域的衰减图，模拟热传导的扩散
    @staticmethod
    def get_decay_map(res, device=None):
        # res 是 (H, W)
        H, W = res
        # 创建频率网格
        y_H = torch.linspace(0., H - 1, H, device=device)
        y_W = torch.linspace(0., W - 1, W, device=device)

        # 频率分量，这里是简单的线性衰减，也可以是其他函数
        # 通常频率越高，衰减越强
        # 这里使用 (f/F_max)^2 形式，模拟高频衰减更快
        alpha_H = (y_H / H) ** 2
        alpha_W = (y_W / W) ** 2

        # 2D 衰减图
        # 通过外积得到二维频率衰减，形状 [H, W]
        decay_map = torch.sqrt(alpha_H[:, None] + alpha_W[None, :])
        # 将衰减因子限制在合理范围，并转换为指数形式，因为 k 是指数
        decay_map = torch.exp(-decay_map * 10)  # 乘以一个常数调整衰减速度
        decay_map = torch.clamp(decay_map, 1e-4, 1.)  # 避免极小值或过大值
        return decay_map

    def infer_init_heat2d(self, infer_k_value=None):
        """
        在推理模式下预计算 k_exp，如果 infer_k_value 提供，则固定 k。
        这用于需要固定衰减参数的场景，例如非时间敏感的推理。
        """
        self.infer_mode = True
        if infer_k_value is None:
            # 如果没有提供固定 k 值，则默认使用一个中等衰减
            # 注意：这会失去时间动态性，除非你的任务在推理时不需要时间动态性
            # 或者你可以在此处根据推理的特定时间步动态生成 infer_k_value。
            # 为了时间动态性，infer_mode 下的 k_exp 应该根据输入的 t_indices 动态计算。
            # 这段代码仅为非时间敏感的推理预留。
            default_k = torch.full((self.dim,), 0.05, device=self.weight_exp.device)  # 假设一个默认值
            self.k_exp = torch.pow(self.weight_exp[:, :, None], default_k).detach_()
        else:
            # 如果提供了具体的 k 值，则直接使用它
            self.k_exp = torch.pow(self.weight_exp[:, :, None], infer_k_value).detach_()

    def forward(self, x: torch.Tensor, freq_embed=None):
        # 输入 x: [B_total, C, H_spatial, W_spatial] (对应MTTLayer中的 [B*T, C, H, W])
        # 输入 freq_embed: [B_total, C] (来自MTTLayer的 time_embedding 结果)

        B, C, H, W = x.shape  # B: B_total (即 B*T), C: hidden_dim, H: H_spatial, W: W_spatial

        # 1. 深度可分离卷积
        x = self.dwconv(x)  # 形状: [B, C, H, W]

        # 2. 线性投影和 chunk (用于 SiLU 门控)
        x_permuted = x.permute(0, 2, 3, 1).contiguous()  # 形状: [B, H, W, C]
        x_linear = self.linear(x_permuted)  # 形状: [B, H, W, 2*C]
        x, z = x_linear.chunk(chunks=2, dim=-1)  # x, z 形状: [B, H, W, C]

        # 获取 DCT/IDCT 权重和衰减图 (已在 __init__ 中注册为 buffer)
        weight_cosn = self.weight_cosn
        weight_cosm = self.weight_cosm
        weight_exp = self.weight_exp  # 形状: [H, W]

        N_freq, M_freq = weight_cosn.shape[0], weight_cosm.shape[0]  # N_freq=H, M_freq=W

        # 3. 2D DCT (通过 1D 卷积近似) - 空间域到频率域的转换
        # 沿 H 维度进行 DCT
        x_freq_h = F.conv1d(x.contiguous().view(B, H, -1),  # [B, H, W*C]
                            weight_cosn.contiguous().view(N_freq, H, 1))  # [H, H, 1] 卷积核
        # x_freq_h 形状: [B, N_freq, W*C]

        # 沿 W 维度进行 DCT
        x_freq = F.conv1d(x_freq_h.contiguous().view(-1, W, C),  # [B*N_freq, W, C]
                          weight_cosm.contiguous().view(M_freq, W, 1))  # [W, W, 1] 卷积核
        x_freq = x_freq.contiguous().view(B, N_freq, M_freq, C)  # 形状: [B, H_freq, W_freq, C]

        # 4. 频率域衰减 (模拟热传导扩散) - 时间信息介入空间处理的关键
        if self.infer_mode:
            # 推理模式下，使用预计算的 k_exp (注意：如果需要时间动态性，这里要额外处理)
            x = torch.einsum("bhwc, nmc -> bhwc", x_freq, self.k_exp)  # bhwc -> B, H_freq, W_freq, C
        else:
            # --- 关键差异点：动态生成衰减因子 ---
            # self.to_k(freq_embed) 形状: [B, C]
            to_k_output = self.to_k(freq_embed)  # 这是每个批次元素（时间步）和每个通道的 k 值

            # 将 weight_exp (空间频率衰减基图) 扩展到 [1, H_freq, W_freq, 1]，以便广播到所有批次和通道
            expanded_weight_exp = weight_exp.unsqueeze(0).unsqueeze(-1)  # 形状: [1, H, W, 1]

            # 将 to_k_output (动态 k 值) 扩展到 [B, 1, 1, C]，以便广播到所有频率分量
            expanded_to_k_output = to_k_output.unsqueeze(1).unsqueeze(1)  # 形状: [B, 1, 1, C]

            # 计算最终的衰减因子：[B, H_freq, W_freq, C]
            # 这里的 pow 运算会将每个时间步、每个通道的 k 值，作为对应频率分量的指数。
            # 这是实现“当前帧温度高，其他帧温度低”的关键。
            decay_factor = torch.pow(expanded_weight_exp, expanded_to_k_output)

            # 将频率域特征与衰减因子相乘
            # einsum 确保了 B, H_freq, W_freq, C 维度上的逐元素相乘
            x = torch.einsum("bhwc,bhwc -> bhwc", x_freq, decay_factor)
            # 形状: [B, H_freq, W_freq, C]

        # 5. 2D IDCT (通过 1D 卷积近似) - 频率域回空间域的转换
        # 沿 H 维度进行 IDCT
        x_spatial_h = F.conv1d(x.contiguous().view(B, N_freq, -1),  # [B, N_freq, M_freq*C]
                               weight_cosn.t().contiguous().view(H, N_freq, 1))  # [H, N_freq, 1] 卷积核
        # x_spatial_h 形状: [B, H, M_freq*C]

        # 沿 W 维度进行 IDCT
        x_spatial = F.conv1d(x_spatial_h.contiguous().view(-1, M_freq, C),  # [B*H, M_freq, C]
                             weight_cosm.t().contiguous().view(W, M_freq, 1))  # [W, M_freq, 1] 卷积核
        x_spatial = x_spatial.contiguous().view(B, H, W, C)  # 形状: [B, H_spatial, W_spatial, C]

        # 6. 输出归一化
        x = self.out_norm(x_spatial)  # 形状: [B, H, W, C]

        # 7. SiLU 门控
        x = x * nn.functional.silu(z)  # 形状: [B, H, W, C]

        # 8. 输出线性层
        x = self.out_linear(x)  # 形状: [B, H, W, C]

        # 9. 最终维度重排，回到 [B, C, H, W] 格式
        x = x.permute(0, 3, 1, 2).contiguous()  # 形状: [B, C, H, W]

        return x


# --- MTTLayer (主模块) ---
class MTTLayer(nn.Module):
    def __init__(self, tokensize, hidden=96, dropout=0., sequence_length=5):
        super().__init__()
        self.global_heat_attention = Heat2D(res=tokensize[0], dim=hidden, hidden_dim=hidden)

        self.ffn = FeedForward(hidden, hidden, p=dropout)  # FFN 模块

        # 定义两个 LayerNorm 模块
        self.norm_heat = nn.LayerNorm(hidden)  # 在 Heat2D 之前，以及残差之后
        self.norm_ffn = nn.LayerNorm(hidden)  # 在 FFN 之前，以及残差之后

        self.dropout = nn.Dropout(p=dropout)

        self.sequence_length = sequence_length
        max_time_distance = self.sequence_length // 2

        self.time_embedding_layer = nn.Embedding(
            num_embeddings=max_time_distance + 1,
            embedding_dim=hidden
        )

    def forward(self, input):
        x_input = input['x']  # [B, L, C, H, W]
        H, W = input['h'], input['w']

        B, L, C_orig, H_orig, W_orig = x_input.shape

        if H_orig != H or W_orig != W:
            raise ValueError(f"Input spatial dimensions ({H_orig}, {W_orig}) do not match tokensize ({H}, {W}).")

        N = H * W

        # 将输入从 [B, L, C, H, W] 展平为 [B*L, C, H, W]
        # 并为 LayerNorm 转换成 [B*L, N, C]
        x_current_stage = x_input.view(B * L, C_orig, H_orig, W_orig).contiguous()
        x_current_stage_for_norm = x_current_stage.permute(0, 2, 3, 1).reshape(B * L, N, C_orig).contiguous()

        # --- 第一部分：LayerNorm -> Heat2D -> Residual 1 ---

        # 1. 保存当前阶段输入 x_current_stage_for_norm 用于残差连接
        residual_heat = x_current_stage_for_norm

        # 2. LayerNorm (在 Heat2D 之前)
        x_normed_before_heat = self.norm_heat(x_current_stage_for_norm)

        # 3. 准备 Heat2D 的输入形状：从 [B*L, N, C] 转换回 [B*L, C, H, W]
        x_reshaped_for_heat = x_normed_before_heat.permute(0, 2, 1).reshape(B * L, C_orig, H, W).contiguous()

        # --- 生成时间距离索引用于嵌入 ---
        current_time_steps = torch.arange(self.sequence_length, device=x_input.device)
        current_frame_abs_idx = self.sequence_length // 2
        time_distances_per_sequence = torch.abs(current_time_steps - current_frame_abs_idx)
        freq_embed_indices = time_distances_per_sequence.repeat(B).long()
        freq_embed = self.time_embedding_layer(freq_embed_indices)


        # 4. 调用 Heat2D 模块
        x_heat_output = self.global_heat_attention(x_reshaped_for_heat, freq_embed=freq_embed)

        # 5. 将 Heat2D 的输出转换回 [B*L, N, C]
        x_heat_flat = x_heat_output.reshape(B * L, C_orig, N).permute(0, 2, 1).contiguous()

        # 6. 第一个残差连接： Heat2D 输出 + residual_heat
        # x_current_stage_for_norm 会更新为 Heat2D 模块的输出加上残差连接
        x_current_stage_for_norm = x_heat_flat + residual_heat

        # --- 第二部分：LayerNorm -> FFN -> Residual 2 ---

        # 7. 保存当前阶段输入 x_current_stage_for_norm 用于残差连接
        residual_ffn = x_current_stage_for_norm

        # 8. LayerNorm (在 FFN 之前)
        x_normed_before_ffn = self.norm_ffn(x_current_stage_for_norm)

        # 9. FFN
        x_ffn_output = self.ffn(x_normed_before_ffn)

        # 10. 第二个残差连接： FFN 输出 + residual_ffn
        x_final_output_flat = x_ffn_output + residual_ffn

        # 11. 最终输出，转换回 [B, L, C, H, W] 的原始形式
        x_final_output = x_final_output_flat.reshape(B, L, N, C_orig).permute(0, 1, 3, 2).reshape(B, L, C_orig, H,
                                                                                                  W).contiguous()

        return {'x': x_final_output, 't': freq_embed_indices, 'h': H, 'w': W}


if __name__ == '__main__':
    # 示例使用
    batch_size = 2
    sequence_length = 3  # L 必须是奇数，例如 3, 5, 7 等
    height = 32
    width = 32
    num_tokens = height * width
    channels = 96  # 特征维度 (hidden_dim)

    # 模拟输入特征 x，形状现在是 [B, L, C, H, W]
    dummy_x_input = torch.randn(batch_size, sequence_length, channels, height, width)

    # 创建 MTTLayer 实例
    # tokensize = (height, width)
    # 传入 sequence_length
    mtt_layer = MTTLayer(tokensize=(height, width), hidden=channels, sequence_length=sequence_length)

    # 准备输入字典
    input_data = {
        'x': dummy_x_input,
        'h': height,
        'w': width
    }

    # 执行前向传播
    output = mtt_layer(input_data)

    print(f"输出特征 x 的形状: {output['x'].shape}")  # 预期: [batch_size, sequence_length, channels, height, width]
    print(f"MTTLayer 内部生成的 freq_embed 索引示例 (第一个批次): {output['t'].view(batch_size, sequence_length)[0]}")
    # 预期对于 L=5，第一个批次会是 [2, 1, 0, 1, 2] (对应距离)

