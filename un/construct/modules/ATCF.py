import torch
import torch.nn as nn
from modules.ATC import *


class CurrentFrameEnhancementModule(nn.Module):
    def __init__(self, original_channels, mtt_output_channels, height, width, sequence_length):
        super().__init__()

        # 确定当前帧的索引 (L为奇数，中间帧)
        if sequence_length % 2 == 0:
            raise ValueError("Sequence length L should be an odd number for clear 'current' frame.")
        self.current_frame_idx = sequence_length // 2

        # 实例化你的 MTTLayer
        # 这里假设 MTTLayer 的输入通道数等于 original_channels，
        # 且其 hidden 参数决定了输出通道数 mtt_output_channels
        self.mtt_layer = MTTLayer(
            tokensize=(height, width),
            hidden=mtt_output_channels,
            sequence_length=sequence_length,
            # 可能需要 input_channels=original_channels 参数，取决于你的MTTLayer定义
        )

        # 聚合层：在 L 维度上进行平均池化
        # 这里使用 mean() 函数在 dim=1 (L维度) 上进行平均
        self.temporal_aggregator = lambda x: torch.mean(x, dim=1)
        # 也可以替换为：
        # self.temporal_aggregator = lambda x: torch.max(x, dim=1)[0] # 最大池化

        # 如果 MTTLayer 的输出通道数与原始特征通道数不同，需要一个投影层
        if mtt_output_channels != original_channels:
            self.channel_projection = nn.Conv2d(mtt_output_channels, original_channels, kernel_size=1)
        else:
            self.channel_projection = nn.Identity()

    def forward(self, backbone_features_L_frames):
        """
        Args:
            backbone_features_L_frames (torch.Tensor):
                来自主干网络的多帧特征，形状为 [B, L, C_orig, H, W]。
                L 是序列长度，C_orig 是原始特征的通道数。

        Returns:
            torch.Tensor:
                增强后的当前帧特征，形状为 [B, C_orig, H, W]。
        """
        B, L, C_orig, H, W = backbone_features_L_frames.shape

        # 1. 提取原始当前帧的特征
        original_current_frame_feature = backbone_features_L_frames[:, self.current_frame_idx, :, :, :]
        # 形状: [B, C_orig, H, W]

        # 2. 将多帧特征输入 MTTLayer
        mtt_input_data = {
            'x': backbone_features_L_frames,
            'h': H,
            'w': W
        }
        mtt_output_dict = self.mtt_layer(mtt_input_data)
        mtt_all_frames_enhanced_features = mtt_output_dict['x']
        # 形状: [B, L, C_MTT, H, W]

        # 3. 对 MTTLayer 增强后的所有帧进行时间维度聚合
        aggregated_mtt_feature = self.temporal_aggregator(mtt_all_frames_enhanced_features)
        # 形状: [B, C_MTT, H, W] (L维度被聚合掉了)

        # 4. 对聚合后的特征通道进行对齐投影
        projected_mtt_feature = self.channel_projection(aggregated_mtt_feature)
        # 形状: [B, C_orig, H, W]

        # 5. 残差融合：原始当前帧特征 + 聚合并投影后的 MTTLayer 增强特征
        enhanced_current_frame_feature = original_current_frame_feature + projected_mtt_feature

        return enhanced_current_frame_feature


if __name__ == '__main__':
    # --- 示例使用 ---
    # 假设参数
    batch_size = 2
    sequence_length = 5  # 必须是奇数，例如 3, 5, 7...
    original_channels = 256
    mtt_output_channels = 256  # 假设 MTTLayer 输出与原始通道数相同
    height, width = 32, 32



    # 模拟主干网络输出的多帧特征
    simulated_backbone_features = torch.randn(batch_size, sequence_length, original_channels, height, width)

    # 实例化增强模块
    enhancement_module = CurrentFrameEnhancementModule(
        original_channels=original_channels,
        mtt_output_channels=mtt_output_channels,
        height=height,
        width=width,
        sequence_length=sequence_length
    )

    # 运行模块，获取增强后的当前帧特征
    enhanced_current_frame_output = enhancement_module(simulated_backbone_features)


    print(f"增强后的当前帧特征形状: {enhanced_current_frame_output.shape}")  # 应该输出 [B, C_orig, H, W]



