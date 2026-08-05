import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from mmcv.ops.carafe import carafe, normal_init, xavier_init
except ImportError:

    def xavier_init(module: nn.Module,
                    gain: float = 1,
                    bias: float = 0,
                    distribution: str = 'normal') -> None:
        assert distribution in ['uniform', 'normal']
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def carafe(x, normed_mask, kernel_size, group=1, up=1):
        b, c, h, w = x.shape
        _, _, out_h, out_w = normed_mask.shape
        assert out_h == up * h
        assert out_w == up * w
        pad = kernel_size // 2
        padded = F.pad(x, pad=[pad] * 4, mode='reflect')
        unfolded = F.unfold(padded, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
        unfolded = unfolded.reshape(b, c * kernel_size * kernel_size, h, w)
        unfolded = F.interpolate(unfolded, scale_factor=up, mode='nearest')
        unfolded = unfolded.reshape(b, c, kernel_size * kernel_size, out_h, out_w)
        normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, out_h, out_w)
        return (unfolded * normed_mask).sum(dim=2)

    def normal_init(module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning and size is not None and align_corners:
        input_h, input_w = tuple(int(x) for x in input.shape[2:])
        output_h, output_w = tuple(int(x) for x in size)
        if output_h > input_h or output_w > input_w:
            if ((output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)):
                warnings.warn(
                    f'When align_corners={align_corners}, input size {(input_h, input_w)} '
                    f'and output size {(output_h, output_w)} are not perfectly aligned.')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def hamming2d(height, width):
    return np.outer(np.hamming(height), np.hamming(width))


def compute_similarity(input_tensor, k=3, dilation=1, sim='cos'):
    b, c, h, w = input_tensor.shape
    unfolded = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation)
    unfolded = unfolded.reshape(b, c, k * k, h, w)
    center = unfolded[:, :, k * k // 2:k * k // 2 + 1]
    if sim == 'cos':
        similarity = F.cosine_similarity(center, unfolded, dim=1)
    elif sim == 'dot':
        similarity = (center * unfolded).sum(dim=1)
    else:
        raise NotImplementedError
    similarity = torch.cat([similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]], dim=1)
    return similarity.view(b, k * k - 1, h, w)


class LocalSimGuidedSampler(nn.Module):
    def __init__(self,
                 in_channels,
                 scale=2,
                 style='lp',
                 groups=4,
                 use_direct_scale=True,
                 kernel_size=1,
                 local_window=3,
                 norm=True,
                 direction_feat='sim_concat'):
        super().__init__()
        assert scale == 2
        assert style == 'lp'
        assert in_channels >= groups and in_channels % groups == 0

        self.scale = scale
        self.style = style
        self.groups = groups
        self.local_window = local_window
        self.direction_feat = direction_feat

        out_channels = 2 * groups * scale * scale
        sim_channels = local_window * local_window - 1
        offset_in_channels = sim_channels if direction_feat == 'sim' else in_channels + sim_channels

        self.offset = nn.Conv2d(offset_in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2)
        normal_init(self.offset, std=0.001)

        self.direct_scale = None
        if use_direct_scale:
            self.direct_scale = nn.Conv2d(offset_in_channels,
                                          out_channels,
                                          kernel_size=kernel_size,
                                          padding=kernel_size // 2)
            constant_init(self.direct_scale, val=0.)

        self.hr_offset = nn.Conv2d(offset_in_channels,
                                   2 * groups,
                                   kernel_size=kernel_size,
                                   padding=kernel_size // 2)
        normal_init(self.hr_offset, std=0.001)

        self.hr_direct_scale = None
        if use_direct_scale:
            self.hr_direct_scale = nn.Conv2d(offset_in_channels,
                                             2 * groups,
                                             kernel_size=kernel_size,
                                             padding=kernel_size // 2)
            constant_init(self.hr_direct_scale, val=0.)

        if norm:
            self.norm_hr = nn.GroupNorm(max(1, in_channels // 8), in_channels)
            self.norm_lr = nn.GroupNorm(max(1, in_channels // 8), in_channels)
        else:
            self.norm_hr = nn.Identity()
            self.norm_lr = nn.Identity()

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        pos = torch.arange((-self.scale + 1) / 2,
                           (self.scale - 1) / 2 + 1) / self.scale
        mesh = torch.stack(torch.meshgrid([pos, pos]))
        return mesh.transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h]))
        coords = coords.transpose(1, 2).unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([w, h], device=x.device, dtype=x.dtype).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
        coords = coords.view(b, 2, -1, self.scale * h, self.scale * w)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        sampled = F.grid_sample(x.reshape(b * self.groups, -1, x.size(-2), x.size(-1)),
                                coords,
                                mode='bilinear',
                                align_corners=False,
                                padding_mode='border')
        return sampled.view(b, -1, self.scale * h, self.scale * w)

    def get_offset_lp(self, hr_x, lr_x):
        if self.direction_feat == 'sim':
            hr_feat = compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')
            lr_feat = compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')
        elif self.direction_feat == 'sim_concat':
            hr_feat = torch.cat(
                [hr_x, compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')], dim=1)
            lr_feat = torch.cat(
                [lr_x, compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')], dim=1)
        else:
            raise NotImplementedError

        if self.direct_scale is None:
            return (self.offset(lr_feat) +
                    F.pixel_unshuffle(self.hr_offset(hr_feat), self.scale)) * 0.25 + self.init_pos

        scale = (self.direct_scale(lr_feat) +
                 F.pixel_unshuffle(self.hr_direct_scale(hr_feat), self.scale)).sigmoid()
        return (self.offset(lr_feat) +
                F.pixel_unshuffle(self.hr_offset(hr_feat), self.scale)) * scale + self.init_pos

    def forward(self, hr_x, lr_x, feat2sample):
        hr_x = self.norm_hr(hr_x)
        lr_x = self.norm_lr(lr_x)
        offset = self.get_offset_lp(hr_x, lr_x)
        return self.sample(feat2sample, offset)


class FreqFusion(nn.Module):
    def __init__(self,
                 hr_channels,
                 lr_channels,
                 scale_factor=1,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64,
                 align_corners=False,
                 upsample_mode='nearest',
                 feature_resample=False,
                 feature_resample_group=4,
                 comp_feat_upsample=True,
                 use_high_pass=True,
                 use_low_pass=True,
                 hr_residual=True,
                 semi_conv=True,
                 hamming_window=True,
                 feature_resample_norm=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.feature_resample = feature_resample
        self.comp_feat_upsample = comp_feat_upsample

        self.hr_channel_compressor = nn.Conv2d(hr_channels, compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, compressed_channels, 1)
        self.content_encoder = nn.Conv2d(
            compressed_channels,
            lowpass_kernel * lowpass_kernel * up_group * scale_factor * scale_factor,
            encoder_kernel,
            padding=int((encoder_kernel - 1) * encoder_dilation / 2),
            dilation=encoder_dilation,
            groups=1)

        if feature_resample:
            self.dysampler = LocalSimGuidedSampler(in_channels=compressed_channels,
                                                   scale=2,
                                                   style='lp',
                                                   groups=feature_resample_group,
                                                   use_direct_scale=True,
                                                   kernel_size=encoder_kernel,
                                                   norm=feature_resample_norm)
        if use_high_pass:
            self.content_encoder2 = nn.Conv2d(
                compressed_channels,
                highpass_kernel * highpass_kernel * up_group * scale_factor * scale_factor,
                encoder_kernel,
                padding=int((encoder_kernel - 1) * encoder_dilation / 2),
                dilation=encoder_dilation,
                groups=1)

        if hamming_window:
            self.register_buffer(
                'hamming_lowpass',
                torch.FloatTensor(hamming2d(lowpass_kernel, lowpass_kernel))[None, None])
            self.register_buffer(
                'hamming_highpass',
                torch.FloatTensor(hamming2d(highpass_kernel, highpass_kernel))[None, None])
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                xavier_init(module, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel * kernel))
        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).reshape(n, -1, kernel, kernel)
        mask = mask * hamming
        mask = mask / mask.sum(dim=(-1, -2), keepdim=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).reshape(n, -1, h, w).contiguous()
        return mask

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)

        if self.semi_conv:
            if not self.comp_feat_upsample or not self.use_high_pass:
                raise NotImplementedError

            mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
            mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat,
                                                  self.highpass_kernel,
                                                  hamming=self.hamming_highpass)
            compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(
                compressed_hr_feat, mask_hr_init, self.highpass_kernel, self.up_group, 1)

            mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
            mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat,
                                                  self.lowpass_kernel,
                                                  hamming=self.hamming_lowpass)

            mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
            mask_lr_lr_feat = F.interpolate(
                carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel, self.up_group, 2),
                size=compressed_hr_feat.shape[-2:],
                mode='nearest')
            mask_lr = mask_lr_hr_feat + mask_lr_lr_feat
            mask_lr_init = self.kernel_normalizer(mask_lr,
                                                  self.lowpass_kernel,
                                                  hamming=self.hamming_lowpass)

            mask_hr_lr_feat = F.interpolate(
                carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel,
                       self.up_group, 2),
                size=compressed_hr_feat.shape[-2:],
                mode='nearest')
            mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
        else:
            compressed_x = F.interpolate(compressed_lr_feat,
                                         size=compressed_hr_feat.shape[-2:],
                                         mode='nearest') + compressed_hr_feat
            mask_lr = self.content_encoder(compressed_x)
            mask_hr = self.content_encoder2(compressed_x) if self.use_high_pass else None

        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)

        if self.semi_conv:
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 2)
        else:
            lr_feat = resize(lr_feat,
                             size=hr_feat.shape[2:],
                             mode=self.upsample_mode,
                             align_corners=None if self.upsample_mode == 'nearest' else
                             self.align_corners)
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 1)

        if self.use_high_pass:
            mask_hr = self.kernel_normalizer(mask_hr,
                                             self.highpass_kernel,
                                             hamming=self.hamming_highpass)
            hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)
            hr_feat = hr_feat_hf + hr_feat if self.hr_residual else hr_feat_hf

        if self.feature_resample:
            lr_feat = self.dysampler(hr_x=compressed_hr_feat,
                                     lr_x=compressed_lr_feat,
                                     feat2sample=lr_feat)

        return mask_lr, hr_feat, lr_feat
