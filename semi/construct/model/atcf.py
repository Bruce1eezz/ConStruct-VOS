import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, p=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p),
        )

    def forward(self, x):
        return self.net(x)


class Heat2D(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.linear = nn.Linear(dim, hidden_dim * 2)
        self.out_linear = nn.Linear(hidden_dim, dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.to_k = nn.Sequential(nn.Linear(dim, dim, bias=True), nn.ReLU())
        self._cos_cache = {}
        self._decay_cache = {}

    @staticmethod
    def _cache_key(length_or_shape, device, dtype):
        device_key = (device.type, device.index if device.index is not None else -1)
        return length_or_shape, device_key, str(dtype)

    @staticmethod
    def _build_cos_map(length, device, dtype):
        x = torch.linspace(0., length - 1, length, device=device, dtype=dtype)
        y = torch.linspace(0., length - 1, length, device=device, dtype=dtype)
        n_grid = x[None, :] + 0.5
        k_grid = y[:, None] * math.pi / length
        weight_cos = torch.cos(n_grid * k_grid)
        weight_cos[0, :] = weight_cos[0, :] / math.sqrt(length)
        if length > 1:
            weight_cos[1:, :] = weight_cos[1:, :] / math.sqrt(length / 2)
        return weight_cos

    @staticmethod
    def _build_decay_map(height, width, device, dtype):
        y_h = torch.linspace(0., height - 1, height, device=device, dtype=dtype)
        y_w = torch.linspace(0., width - 1, width, device=device, dtype=dtype)
        alpha_h = (y_h / max(height, 1))**2
        alpha_w = (y_w / max(width, 1))**2
        decay = torch.sqrt(alpha_h[:, None] + alpha_w[None, :])
        decay = torch.exp(-decay * 10)
        return torch.clamp(decay, 1e-4, 1.)

    def _get_cos_map(self, length, device, dtype):
        key = self._cache_key(length, device, dtype)
        if key not in self._cos_cache:
            self._cos_cache[key] = self._build_cos_map(length, device, dtype)
        return self._cos_cache[key]

    def _get_decay_map(self, height, width, device, dtype):
        key = self._cache_key((height, width), device, dtype)
        if key not in self._decay_cache:
            self._decay_cache[key] = self._build_decay_map(height, width, device, dtype)
        return self._decay_cache[key]

    def forward(self, x: torch.Tensor, freq_embed: torch.Tensor):
        residual_dtype = x.dtype
        x = x.float()
        freq_embed = freq_embed.float()

        b, c, h, w = x.shape
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x, z = self.linear(x).chunk(chunks=2, dim=-1)

        weight_cos_h = self._get_cos_map(h, x.device, x.dtype)
        weight_cos_w = self._get_cos_map(w, x.device, x.dtype)
        weight_exp = self._get_decay_map(h, w, x.device, x.dtype)

        x_freq_h = F.conv1d(x.view(b, h, -1), weight_cos_h.view(h, h, 1))
        x_freq = F.conv1d(x_freq_h.view(-1, w, c), weight_cos_w.view(w, w, 1))
        x_freq = x_freq.view(b, h, w, c)

        decay_factor = torch.pow(weight_exp.unsqueeze(0).unsqueeze(-1),
                                 self.to_k(freq_embed).unsqueeze(1).unsqueeze(1))
        x = x_freq * decay_factor

        x_spatial_h = F.conv1d(x.view(b, h, -1), weight_cos_h.t().contiguous().view(h, h, 1))
        x_spatial = F.conv1d(x_spatial_h.view(-1, w, c),
                             weight_cos_w.t().contiguous().view(w, w, 1))
        x_spatial = x_spatial.view(b, h, w, c)

        x = self.out_norm(x_spatial)
        x = x * F.silu(z)
        x = self.out_linear(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.to(residual_dtype)


class MTTLayer(nn.Module):
    def __init__(self, hidden=96, dropout=0., sequence_length=3):
        super().__init__()
        self.global_heat_attention = Heat2D(dim=hidden, hidden_dim=hidden)
        self.ffn = FeedForward(hidden, hidden, p=dropout)
        self.norm_heat = nn.LayerNorm(hidden)
        self.norm_ffn = nn.LayerNorm(hidden)
        self.sequence_length = sequence_length
        max_time_distance = self.sequence_length // 2
        self.time_embedding_layer = nn.Embedding(max_time_distance + 1, hidden)

    def forward(self, input_dict):
        x_input = input_dict['x']
        b, l, c, h, w = x_input.shape
        if l != self.sequence_length:
            raise ValueError(f'Expected sequence_length={self.sequence_length}, got {l}')

        n = h * w
        x_current = x_input.view(b * l, c, h, w).contiguous()
        x_current_flat = x_current.permute(0, 2, 3, 1).reshape(b * l, n, c).contiguous()

        residual_heat = x_current_flat
        x_normed = self.norm_heat(x_current_flat)
        x_heat_input = x_normed.permute(0, 2, 1).reshape(b * l, c, h, w).contiguous()

        current_steps = torch.arange(self.sequence_length, device=x_input.device)
        current_idx = self.sequence_length // 2
        distances = torch.abs(current_steps - current_idx)
        freq_embed = self.time_embedding_layer(distances.repeat(b).long())

        x_heat = self.global_heat_attention(x_heat_input, freq_embed=freq_embed)
        x_heat = x_heat.reshape(b * l, c, n).permute(0, 2, 1).contiguous()
        x_current_flat = x_heat + residual_heat

        residual_ffn = x_current_flat
        x_ffn = self.ffn(self.norm_ffn(x_current_flat))
        x_final = x_ffn + residual_ffn
        x_final = x_final.reshape(b, l, n, c).permute(0, 1, 3, 2).reshape(b, l, c, h, w)

        return {'x': x_final, 'h': h, 'w': w}


class CurrentFrameEnhancementModule(nn.Module):
    def __init__(self, original_channels, mtt_output_channels, sequence_length=3):
        super().__init__()
        if sequence_length % 2 == 0:
            raise ValueError('sequence_length must be odd')
        self.current_frame_idx = sequence_length // 2
        self.mtt_layer = MTTLayer(hidden=mtt_output_channels, sequence_length=sequence_length)
        self.temporal_aggregator = lambda x: torch.mean(x, dim=1)
        if mtt_output_channels != original_channels:
            self.channel_projection = nn.Conv2d(mtt_output_channels,
                                                original_channels,
                                                kernel_size=1)
        else:
            self.channel_projection = nn.Identity()

    def forward(self, backbone_features_l_frames):
        original_current = backbone_features_l_frames[:, self.current_frame_idx]
        mtt_output = self.mtt_layer({'x': backbone_features_l_frames})['x']
        aggregated = self.temporal_aggregator(mtt_output)
        projected = self.channel_projection(aggregated)
        return original_current + projected
