
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from transformers import SegformerModel
import os
from convnextv2 import build
from convnextv2.build import build_convnextv2_model
from modules.FreqFuse import *
from modules.ATCF import *
from modules.MoSEPlus import *


# basic modules
class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(ConvRelu(c, c, 1, 1, 0), Conv(c, c, 1, 1, 0))
        self.conv3 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        c = torch.sigmoid(self.conv2(F.adaptive_avg_pool2d(x, output_size=(1, 1))) + self.conv2(F.adaptive_max_pool2d(x, output_size=(1, 1))))
        x = x * c
        s = torch.sigmoid(self.conv3(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))
        x = x * s
        return x


# encoding module
class Encoder(nn.Module):
    def __init__(self, ver):
        super().__init__()
        self.ver = ver

        # ResNet-101 backbone
        if ver == 'rn101':
            backbone = tv.models.resnet101(pretrained=True)
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # MiT-b2 backbone
        if ver == 'mitb2':
            # 构建指向 mit-b2 的相对路径
            current_dir = os.path.dirname(__file__)  # 获取当前文件所在路径
            mit_b2_path = os.path.abspath(os.path.join(current_dir, "..", "backbone", "mit-b2"))
            self.backbone = SegformerModel.from_pretrained(mit_b2_path)#这里指出了backbone的位置
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ConvNeXtV2_tiny backbone
        if ver == 'convnextv2':
            self.backbone = build_convnextv2_model(model_type='convnextv2_tiny.fcmae_ft_in22k_in1k')
            self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, img):

        # ResNet-101 backbone
        if self.ver == 'rn101':
            x = (img - self.mean) / self.std
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            s4 = x
            x = self.layer2(x)
            s8 = x
            x = self.layer3(x)
            s16 = x
            x = self.layer4(x)
            s32 = x
            return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}

        # MiT-b2 backbone
        if self.ver == 'mitb2':
            x = (img - self.mean) / self.std
            x = self.backbone(x, output_hidden_states=True).hidden_states
            s4 = x[0]
            s8 = x[1]
            s16 = x[2]
            s32 = x[3]
            return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}

        # ConvNeXtV2_tiny backbone
        if self.ver == 'convnextv2':
            x = (img - self.mean) / self.std
            x = self.backbone(x)
            s4 = x[0]
            s8 = x[1]
            s16 = x[2]
            s32 = x[3]
            return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}


# decoding module
class Decoder(nn.Module):
    def __init__(self, ver):
        super().__init__()

        # ResNet-101 backbone
        if ver == 'rn101':
            self.conv1 = ConvRelu(2048, 256, 1, 1, 0)
            self.blend1 = ConvRelu(256, 256, 3, 1, 1)
            self.cbam1 = CBAM(256)
            self.ff1 = FreqFusion(256, 256)
            self.conv2 = ConvRelu(1024, 256, 1, 1, 0)
            self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam2 = CBAM(256)
            self.ff2 = FreqFusion(256, 256)
            self.conv3 = ConvRelu(512, 256, 1, 1, 0)
            self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam3 = CBAM(256)
            self.ff3 = FreqFusion(256, 256)
            self.conv4 = ConvRelu(256, 256, 1, 1, 0)
            self.blend4 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam4 = CBAM(256)
            self.predictor = Conv(256, 2, 3, 1, 1)

        # MiT-b2 backbone
        if ver == 'mitb2':
            self.conv1 = ConvRelu(512, 256, 1, 1, 0)
            self.blend1 = ConvRelu(256, 256, 3, 1, 1)
            self.cbam1 = CBAM(256)
            self.ff1 = FreqFusion(256, 256)
            self.conv2 = ConvRelu(320, 256, 1, 1, 0)
            self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam2 = CBAM(256)
            self.ff2 = FreqFusion(256, 256)
            self.conv3 = ConvRelu(128, 256, 1, 1, 0)
            self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam3 = CBAM(256)
            self.ff3 = FreqFusion(256, 256)
            self.conv4 = ConvRelu(64, 256, 1, 1, 0)
            self.blend4 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam4 = CBAM(256)
            self.predictor = Conv(256, 2, 3, 1, 1)

        # # ConvNeXtV2_tiny backbone
        if ver == 'convnextv2':
            self.conv1 = ConvRelu(768, 256, 1, 1, 0)
            self.blend1 = ConvRelu(256, 256, 3, 1, 1)
            self.cbam1 = CBAM(256)
            self.ff1 = FreqFusion(256, 256)
            self.conv2 = ConvRelu(384, 256, 1, 1, 0)
            self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam2 = CBAM(256)
            self.ff2 = FreqFusion(256, 256)
            self.conv3 = ConvRelu(192, 256, 1, 1, 0)
            self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam3 = CBAM(256)
            self.ff3 = FreqFusion(256, 256)
            self.conv4 = ConvRelu(96, 256, 1, 1, 0)
            self.blend4 = ConvRelu(256 + 256, 256, 3, 1, 1)
            self.cbam4 = CBAM(256)
            self.predictor = Conv(256, 2, 3, 1, 1)

    def forward(self, app_feats, mo_feats):
        x = self.conv1(app_feats['s32'] + mo_feats['s32'])  # s32: 1/32
        x = self.cbam1(self.blend1(x))
        # 使用 FreqFusion 替代上采样
        _, x_hr, x_lr = self.ff1(hr_feat=self.conv2(app_feats['s16'] + mo_feats['s16']), lr_feat=x)  # 上采样到1/16
        x = torch.cat((x_hr, x_lr), dim=1)
        x = self.cbam2(self.blend2(x))
        _, x_hr, x_lr = self.ff2(hr_feat=self.conv3(app_feats['s8'] + mo_feats['s8']), lr_feat=x)  # 上采样到1/8
        x = torch.cat((x_hr, x_lr), dim=1)
        x = self.cbam3(self.blend3(x))
        _, x_hr, x_lr = self.ff3(hr_feat=self.conv4(app_feats['s4'] + mo_feats['s4']), lr_feat=x)  # 上采样到1/4
        x = torch.cat((x_hr, x_lr), dim=1)
        x = self.predictor(self.cbam4(self.blend4(x)))
        score = F.interpolate(x, scale_factor=4, mode='bicubic')
        return score

# time mixing
class TimeMixingLayer(nn.Module):
    def __init__(self, num_frames, ver):
        super(TimeMixingLayer, self).__init__()
        self.num_frames = num_frames
        if ver == 'convnextv2':
            #self.atcf4 = CurrentFrameEnhancementModule(96,96,128,128,num_frames)
            #self.atcf8 = CurrentFrameEnhancementModule(192,192,64,64,num_frames)
            self.atcf16 = CurrentFrameEnhancementModule(384, 384, 32, 32, num_frames)
            self.atcf32 = CurrentFrameEnhancementModule(768, 768, 16, 16, num_frames)

        elif ver == 'mitb2':
            #self.atcf4 = CurrentFrameEnhancementModule(64,64,128,128,num_frames)
            #self.atcf8 = CurrentFrameEnhancementModule(128,128,64,64,num_frames)
            self.atcf16 = CurrentFrameEnhancementModule(320, 320, 32, 32, num_frames)
            self.atcf32 = CurrentFrameEnhancementModule(512, 512, 16, 16, num_frames)

        elif ver == 'rn101':
            #self.atcf4 = CurrentFrameEnhancementModule(256,256,128,128,num_frames)
            #self.atcf8 = CurrentFrameEnhancementModule(512,512,64,64,num_frames)
            self.atcf16 = CurrentFrameEnhancementModule(1024, 1024, 32, 32, num_frames)
            self.atcf32 = CurrentFrameEnhancementModule(2048, 2048, 16, 16, num_frames)



    def forward(self, temp_feats):
        #temp_feats: [B,S,C,H,W] -> outputs: [B,C,H,W]

        #s4=self.atcf4(temp_feats['s4'])
        #s8=self.atcf8(temp_feats['s8'])

        # new! only apply timemix on s16 and s32

        s4=temp_feats['s4']
        s4=s4[:, self.num_frames//2, :, :]
        s8=temp_feats['s8']
        s8=s8[:, self.num_frames//2, :, :]
        s16=self.atcf16(temp_feats['s16'])
        s32=self.atcf32(temp_feats['s32'])

        return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}

# MoSE Layer
class MoSELayer(nn.Module):
    def __init__(self, num_experts, ver):
        super(MoSELayer, self).__init__()
        self.num_experts = num_experts
        if ver == 'convnextv2':


            self.mose16 = VisionMOE(
                channels=384,
                proto_expert=ProtoExpert(hidden_size=384),
                num_experts=self.num_experts,
                moe_2layer_gate=True
                )


            self.mose32 = VisionMOE(
                channels=768,
                proto_expert=ProtoExpert(hidden_size=768),
                num_experts=self.num_experts,
                moe_2layer_gate=True
            )

        elif ver == 'mitb2':



            self.mose16 = VisionMOE(
                channels=320,
                proto_expert=ProtoExpert(hidden_size=320),
                num_experts=self.num_experts,
                moe_2layer_gate=True
            )

            self.mose32 = VisionMOE(
                channels=512,
                proto_expert=ProtoExpert(hidden_size=512),
                num_experts=self.num_experts,
                moe_2layer_gate=True
            )

        elif ver == 'rn101':


            self.mose16 = VisionMOE(
                channels=1024,
                proto_expert=ProtoExpert(hidden_size=1024),
                num_experts=self.num_experts,
                moe_2layer_gate=True
            )

            self.mose32 = VisionMOE(
                channels=2048,
                proto_expert=ProtoExpert(hidden_size=2048),
                num_experts=self.num_experts,
                moe_2layer_gate=True
            )



    def forward(self, feats):
        #feats: [B,C,H,W] -> outputs: [B,C,H,W]

        # new! only apply timemix on s32

        s4=feats['s4']
        s8=feats['s8']
        s16=feats['s16']
        s32=self.mose32(feats['s32'])+feats['s32']

        return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}


# VOS model
class VOS(nn.Module):
    def __init__(self, ver):
        super().__init__()
        self.app_encoder = Encoder(ver)
        self.mo_encoder = Encoder(ver)
        self.timemixlayer = TimeMixingLayer(3, ver)  # set num_frames=3, let num_frames = 1 is ok.
        self.moselayer = MoSELayer(8, ver) # set num_experts=8
        self.decoder = Decoder(ver)


# TMO model
class TMO(nn.Module):
    def __init__(self, ver):
        super().__init__()
        self.vos = VOS( ver)

    def forward(self, ref_imgs, ref_flows):
        # [B,L,S,C,H,W]
        B, L, S, _, H1, W1 = ref_imgs.size()

        _, _, _, _, H2, W2 = ref_flows.size()

        # resize to 512p
        s = 512
        ref_imgs = F.interpolate(ref_imgs.view(B * L * S, -1, H1, W1), size=(s, s), mode='bicubic').view(B, L, S, -1, s, s)
        ref_flows = F.interpolate(ref_flows.view(B * L * S, -1, H2, W2), size=(s, s), mode='bicubic').view(B, L, S, -1, s, s)


        # for each frame
        score_lst = []
        mask_lst = []
        for i in range(L):

            # 获取当前时间步的所有参考帧 [B, S, C, H, W]
            current_imgs_with_s = ref_imgs[:, i]
            current_flows_with_s = ref_flows[:, i]

            # 变形为 [B*S, C, H, W]
            b, s, c, h, w = current_imgs_with_s.size()
            encoder_input_imgs = current_imgs_with_s.view(B * S, c, h, w)
            encoder_input_flows = current_flows_with_s.view(B * S, c, h, w)

            # query frame prediction
            temp_app_feats = self.vos.app_encoder(encoder_input_imgs)
            temp_mo_feats = self.vos.mo_encoder(encoder_input_flows)

            #resize feature after encoder -> [B,S,C,H,W]

            ref_app_feats = {key: tensor.view(B, S, *tensor.shape[1:]) for key, tensor in temp_app_feats.items()}
            ref_mo_feats = {key: tensor.view(B, S, *tensor.shape[1:]) for key, tensor in temp_mo_feats.items()}


            """
            add TimeMixingLayer:
            input:[B,S,C,H,W]
            output:[B,C,H,W]
            """
            current_app_feats = self.vos.timemixlayer(ref_app_feats)
            current_mo_feats = self.vos.timemixlayer(ref_mo_feats)


            score = self.vos.decoder(current_app_feats, current_mo_feats)
            score = F.interpolate(score, size=(H1, W1), mode='bicubic')

            # store soft scores
            if B != 1:
                score_lst.append(score)

            # store hard masks
            if B == 1:
                pred_seg = torch.softmax(score, dim=1)
                pred_mask = torch.max(pred_seg, dim=1, keepdim=True)[1]
                mask_lst.append(pred_mask)

        # generate output
        output = {}
        if B != 1:
            output['scores'] = torch.stack(score_lst, dim=1)
        if B == 1:
            output['masks'] = torch.stack(mask_lst, dim=1)
        return output


# 测试代码
if __name__ == "__main__":
    import torch

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建FakeFlow模型实例
    model = TMO(ver='convnextv2').to(device)

    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"模型参数量 total={total_params:,} ({total_params / 1e6:.2f}M), "
        f"trainable={trainable_params:,} ({trainable_params / 1e6:.2f}M)"
    )

    # 生成模拟输入数据
    batch_size = 1
    sequence_length = 1
    select_frames = 3
    img_size = 512
    flow_size = 512

    imgs = torch.randn(batch_size, sequence_length, select_frames, 3, img_size, img_size).to(device)
    flows = torch.randn(batch_size, sequence_length, select_frames, 3, flow_size, flow_size).to(device)

    print(f"输入图像形状: {imgs.shape}")
    print(f"输入光流形状: {flows.shape}")

    # 运行模型
    model.eval()

    output = model(imgs, flows)

    # 打印输出
    if 'scores' in output:
        print(f"输出分数形状: {output['scores'].shape}")
    if 'masks' in output:
        print(f"输出掩码形状: {output['masks'].shape}")


    print("测试完成!")


"""
resnet-101
resize pictures to s*s = 512*512
(batch_size, channels, height, width)
s4 shape: torch.Size([B, 256, s/4, s/4]), 128=s/4
s8 shape: torch.Size([B, 512, 48, 48]), 64=s/8
s16 shape: torch.Size([B, 1024, s/8, s/8]), 32=s/16
s32 shape: torch.Size([B, 2048, s/32, s/32]), 16=s/32




mit-b1/mit-b2
resize pictures to s*s = 512*512
(batch_size, channels, height, width)
s4 shape: torch.Size([B, 64, s/4, s/4]), 128=s/4
s8 shape: torch.Size([B, 128, 48, 48]), 64=s/8
s16 shape: torch.Size([B, 320, s/8, s/8]), 32=s/16
s32 shape: torch.Size([B, 512, s/32, s/32]), 16=s/32



convnextv2_tiny
resize pictures to s*s = 512*512
(batch_size, channels, height, width)
s4 shape: torch.Size([B, 96, s/4, s/4]), 128=s/4
s8 shape: torch.Size([B, 192, 48, 48]), 64=s/8
s16 shape: torch.Size([B, 384, s/8, s/8]), 32=s/16
s32 shape: torch.Size([B, 768, s/32, s/32]), 16=s/32

"""
