
import typing
from collections.abc import Callable
from collections import defaultdict
from typing import Any, Dict, TYPE_CHECKING, Optional, Tuple, List

import torch
import copy

from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn

# 类型检查时，为了避免循环导入，将Base定义为Module[Tensor]
if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    # 正常运行时，Base就是torch.nn.Module
    Base = Module

# --- 全局常量定义 ---
# 定义在MoE门控机制中，每个token要路由到的Top-K个专家数量
MOE_TOP_K = 2
# 这个常量似乎没有在当前代码片段中使用，可能是为后续功能预留的
Constant = 1


class ProtoExpert(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, inputs):
        return self.net(inputs)


# --- 特殊专家模块定义 ---

class CopyExpert(torch.nn.Module):
    """
    一个特殊的“复制”专家, 类似于residual connect。
    它的作用是按原样返回输入，实现一个恒等变换（identity function）。
    这种专家在MoE中可以作为一个“直通”或“默认”路径，不改变输入信号。
    """

    def __init__(self, expert):

        super(CopyExpert, self).__init__()
        pass

    def forward(self, inputs: Tensor) -> Tensor:
        """
        前向传播函数。
        :param inputs: 输入张量。
        :return: 与输入完全相同的张量。
        """
        return inputs


class ZeroExpert(torch.nn.Module):
    """
    一个特殊的“零”专家，类似于dropout。
    它的作用是返回一个与输入形状相同但所有元素都为零的张量。
    这种专家可以被看作是一个“丢弃”或“沉默”的专家，它不提供任何信息。
    """

    def __init__(self, expert):

        super(ZeroExpert, self).__init__()
        pass

    def forward(self, inputs: Tensor) -> Tensor:
        """
        前向传播函数。
        :param inputs: 输入张量。
        :return: 一个与输入形状、数据类型和设备都相同的零张量。
        """
        return torch.zeros_like(inputs).to(inputs.dtype).to(inputs.device)


class ConstantExpert(torch.nn.Module):
    """
    一个特殊的“常数”专家。
    这个专家学习一个固定的向量（self.constant），并根据输入动态地将输入和这个常数向量进行加权融合。
    这为模型提供了一种在输入和某个学习到的“通用知识”之间进行选择的能力。
    """

    def __init__(self, expert):
        """
        构造函数。
        :param expert: 一个标准的专家模块，主要用来获取其隐藏层大小（hidden_size）以初始化常数向量。
        """
        super(ConstantExpert, self).__init__()
        # 定义一个可学习的常数向量，其维度与专家的隐藏层大小相同
        self.constant = torch.nn.Parameter(
            torch.empty((expert.hidden_size)))
        # 使用正态分布初始化这个常数向量
        torch.nn.init.normal_(self.constant)

        # 定义一个线性层，用于将输入映射到两个权重值，决定输入和常数向量的混合比例
        self.wg = torch.nn.Linear(expert.hidden_size, 2, bias=False)
        # 定义Softmax层，用于归一化权重
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        前向传播函数。
        :param inputs: 输入张量，形状通常是 [批次大小, 隐藏层维度] 或 [token数量, 隐藏层维度]。
        :return: 输入和常数向量的加权和。
        """
        # 1. 计算权重：将输入通过线性层得到两个原始分数
        weight = self.wg(inputs)
        # 2. 归一化权重：使用softmax将分数转换为概率分布
        weight = self.softmax(weight)

        # 3. 加权融合：
        #    - weight[:, 0] 是分配给原始输入 `inputs` 的权重
        #    - weight[:, 1] 是分配给可学习常数 `self.constant` 的权重
        #    - 使用 torch.einsum 进行高效的加权求和：
        #      'b,bd->bd' 表示将每个token的输入权重（标量）与其对应的输入向量（维度d）相乘。
        #      'b,d->bd' 表示将每个token的常数权重（标量）与共享的常数向量（维度d）相乘。
        return torch.einsum('b,bd->bd', [weight[:, 0].type_as(inputs), inputs]) + torch.einsum(
            'b,d->bd', [weight[:, 1].type_as(inputs), self.constant.type_as(inputs)])


class ConstantExpertWithNoise(torch.nn.Module):
    """
    一个加入了高斯噪声注入的“常数”专家。
    在训练阶段，它会向学习到的常数向量中添加噪声，以提升模型的健壮性和泛化能力。
    """

    def __init__(self, expert, noise_std: float = 0.001):
        """
        构造函数。
        :param expert: 一个标准的专家模块，用于获取 hidden_size。
        :param noise_std: 注入的高斯噪声的标准差。如果为0，则不注入噪声。
        """
        super(ConstantExpertWithNoise, self).__init__()

        # --- 新增参数 ---
        # 保存噪声的标准差，这是一个需要调整的超参数
        self.noise_std = noise_std

        # --- 原有代码 ---
        self.constant = torch.nn.Parameter(
            torch.empty((expert.hidden_size)))
        torch.nn.init.normal_(self.constant)

        self.wg = torch.nn.Linear(expert.hidden_size, 2, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        #print(f"--- ConstantExpertWithNoise initialized with noise_std = {self.noise_std} ---")

    def forward(self, inputs: Tensor) -> Tensor:
        """
        前向传播函数。
        """
        # 1. 计算权重 (与原版相同)
        weight = self.wg(inputs)
        weight = self.softmax(weight)

        # --- 核心修改：在训练时注入噪声 ---
        effective_constant = self.constant
        # 检查是否处于训练模式，并且噪声标准差大于0
        if self.training and self.noise_std > 0:
            # 生成与 self.constant 形状相同的高斯噪声
            # torch.randn_like(self.constant) 生成均值为0，方差为1的噪声
            noise = torch.randn_like(self.constant) * self.noise_std
            # 将噪声加到常数向量上，得到“有效的”常数向量
            effective_constant = self.constant + noise

        # 3. 使用可能被注入了噪声的 effective_constant 进行加权融合
        return torch.einsum('b,bd->bd', [weight[:, 0].type_as(inputs), inputs]) + torch.einsum(
            'b,d->bd', [weight[:, 1].type_as(inputs), effective_constant.type_as(inputs)])


# --- 门控函数 ---

def gating(logits: Tensor, moe_use_mixtral_gating: bool = False, moe_use_logits_norm: bool = False,
           moe_gate_norm_std: float = 1.0) -> Dict[int, List[Tuple[int, float]]]:
    """
    自定义的门控函数，用于决定每个token应该被路由到哪些专家。
    这个函数实现了两种不同的门控策略，并通过 `moe_use_mixtral_gating` 参数切换。

    :param logits: 门控网络的原始输出，形状为 [token数量, 专家数量]。每个值代表对应token对相应专家的偏好分数。
    :param moe_use_mixtral_gating:布尔值，是否使用类似Mixtral模型的门控策略。
    :param moe_use_logits_norm: 布尔值，是否在计算softmax之前对logits进行归一化。
    :param moe_gate_norm_std: 归一化时目标标准差。
    :return: 一个字典，键是专家ID，值是一个列表，包含被路由到该专家的 [token_ids, scores]。
    """
    # 获取专家的总数
    num_experts = logits.size(1)

    if moe_use_mixtral_gating:
        # --- 策略一：类似Mixtral的门控 ---
        # 核心思想：先选出Top-K个专家，再对这K个专家的logits做softmax。

        if moe_use_logits_norm:
            # 可选：对logits进行归一化，以稳定训练
            target_std = moe_gate_norm_std
            logits_std = logits.std(dim=1, keepdim=True)
            logits = logits / (logits_std / target_std)

        # 选取分数最高的Top-K个专家
        gates, indices = torch.topk(logits, k=MOE_TOP_K, dim=1)  # gates是分数，indices是专家的ID
        # 仅对这Top-K个专家的分数进行softmax归一化
        gates = F.softmax(gates, dim=1)

    else:
        # --- 策略二：标准的Top-K门控（带有特殊处理） ---
        # 核心思想：先对所有专家的logits做softmax，再从得到的概率中选出Top-K。

        target_std = moe_gate_norm_std
        if moe_use_logits_norm:
            # 可选：对logits进行归一化
            logits_std = logits.std(dim=1, keepdim=True)
            gates = F.softmax(logits / (logits_std / target_std), dim=1)
        else:
            # 对所有专家的logits进行softmax，得到每个token到所有专家的概率分布
            gates = F.softmax(logits, dim=1)

        # 从概率分布中选取最高的Top-K个
        # gates是这Top-K个概率值，indices是对应的专家ID
        gates, indices = torch.topk(gates, k=MOE_TOP_K, dim=1)

        # --- 特殊处理 ---
        # 如果被选中的专家是最后一个专家（ID为 num_experts-1），则将其权重置为0。
        # 这暗示最后一个专家可能是一个特殊的“空”或“丢弃”专家，不应该接收任何token。
        # `torch.where`函数根据条件（indices == num_experts-1）选择性地替换值。
        gates = torch.where(indices == (num_experts - 1), torch.zeros_like(gates).to(gates.dtype).to(gates.device),gates)

        # 由于可能将某些权重置零，需要重新归一化，确保每个token的权重之和为1。
        gates /= torch.sum(gates, dim=1, keepdim=True)

    # --- 整理路由信息 ---
    # 创建一个默认字典，用于存储每个专家的路由信息
    expert_info = defaultdict(list)
    # 遍历所有可能的专家ID
    for expert_id in range(num_experts):
        # 找到所有被路由到当前expert_id的token
        # torch.nonzero返回一个元组，包含满足条件的元素的索引
        token_ids, score_ids = torch.nonzero(indices == expert_id, as_tuple=True)
        # score_ids 在这里实际上是 top-k 维度上的索引（0 或 1），
        # 我们需要用它来从 gates 张量中获取正确的权重分数。

        # 将这个专家的信息存入字典：[分配给它的token的ID列表, 对应的权重分数列表]
        expert_info[expert_id] = [token_ids, gates[token_ids, score_ids]]

    return expert_info


class Router(Module):
    """
    路由器模块。
    负责为每个输入token生成路由决策（即分配给哪些专家的权重）。
    """

    def __init__(self,
                 model_dim: int,  # 输入特征的维度 (d_model)
                 num_experts: int,  # 专家的总数
                 moe_use_mixtral_gating: bool,  # 是否使用类似Mixtral的门控策略
                 moe_2layer_gate: bool,  # 是否使用一个双层MLP作为门控网络
                 moe_use_logits_norm: bool,  # 是否对门控的logits进行归一化
                 moe_gate_norm_std: float,  # 归一化时目标标准差
                 ) -> None:
        super().__init__()

        if moe_2layer_gate:
            # 如果配置为使用双层门控网络，则构建一个简单的MLP
            # (Linear -> Tanh -> Linear)
            # 这种更复杂的门控网络可能提供更强的路由决策能力
            self.wg = torch.nn.Sequential(
                torch.nn.Linear(model_dim, num_experts * 8, bias=False).float(),  # 中间层维度通常是专家数量的倍数
                torch.nn.Tanh(),
                torch.nn.Linear(num_experts * 8, num_experts, bias=False).float()
            ).float()
        else:
            # 否则，只使用一个简单的线性层作为门控网络
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()

        # 这个线性层用于处理可选的“门控残差”，允许将上一层的路由信息传递到当前层
        self.gate_map = torch.nn.Linear(num_experts, num_experts, bias=False)

        # 引用从上一个代码片段中定义的门控函数
        self.gate = gating
        # 保存门控策略的配置参数
        self.moe_use_mixtral_gating = moe_use_mixtral_gating
        self.moe_use_logits_norm = moe_use_logits_norm
        self.moe_gate_norm_std = moe_gate_norm_std

    def forward(self, input: torch.Tensor, gate_residual: Optional[Tensor] = None) -> Tuple[
        Dict[int, List[Tuple[int, float]]], Tensor]:
        """
        前向传播函数。
        :param input: 输入张量，形状为 [token数量, model_dim]。
        :param gate_residual: 可选的门控残差，来自前一个MoE层的logits。
        :return: 一个元组，包含：
                  - expert_info: 路由信息的字典。
                  - logits: 当前层计算出的原始门控分数（可作为下一层的残差）。
        """
        # --- 确保门控网络的权重是float32类型 ---
        # 这通常是为了在混合精度训练中保持路由计算的稳定性
        if isinstance(self.wg, torch.nn.Linear):
            if self.wg.weight.dtype != torch.float32:
                self.wg = self.wg.float()
                setattr(self.wg.weight, 'router', True)  # 添加一个标记，可能用于特定的优化或参数分组
        else:  # 对应双层门控网络的情况
            if self.wg[0].weight.dtype != torch.float32:
                self.wg = self.wg.float()
                setattr(self.wg[0].weight, "router", True)
                setattr(self.wg[2].weight, "router", True)

        # 将输入转换为float32进行logits计算
        input_fp32 = input.float()
        # 通过门控网络计算每个token对每个专家的原始分数（logits）
        logits = self.wg(input_fp32)

        # 如果提供了门控残差 (gate_residual)
        if gate_residual is not None:
            # 将残差通过一个线性映射，并加到当前logits上
            # 这是一种层与层之间传递路由信息的机制，可以帮助稳定和协调连续MoE层的决策
            gate_residual = self.gate_map(gate_residual.to(self.gate_map.weight.dtype))
            logits += gate_residual

        # 调用gating函数，根据logits计算出最终的路由表
        gate_output = self.gate(logits, self.moe_use_mixtral_gating, self.moe_use_logits_norm, self.moe_gate_norm_std)

        # 返回路由表和本层的原始logits（用于下一层）
        return gate_output, logits


class Experts(torch.nn.Module):
    """
    一个容器模块，用于存放所有的专家。
    它根据配置，将普通专家、ConstantExpert、CopyExpert和ZeroExpert组合在一起。
    """

    def __init__(self, expert: Module, num_local_experts: int = 1):
        """
        :param expert: 一个“原型”专家模块，将被深度复制以创建多个普通专家。
        :param num_local_experts: 本地（当前设备上）的专家总数。
        """
        super(Experts, self).__init__()

        # --- 专家列表的构建逻辑 ---
        # 假设 num_local_experts = 8, Constant = 2
        # 1. 普通专家数量 = 8 - 2 - 2 = 4 个
        # 2. ConstantExpert 数量 = 2 个
        # 3. CopyExpert 数量 = 1 个
        # 4. ZeroExpert 数量 = 1 个
        # 总共 = 4 + 2 + 1 + 1 = 8 个专家
        self.experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for _ in range(num_local_experts - 2 - Constant)] +  # 创建N个普通的专家
            [ConstantExpertWithNoise(expert, noise_std=0.001) for _ in range(Constant)] +  # 创建Constant个ConstantExpert
            [CopyExpert(expert), ZeroExpert(expert)]  # 添加一个CopyExpert和一个ZeroExpert
        )

    def forward(self, inputs):
        # 这个forward函数没有被实现，因为它只是一个容器。
        # 具体的专家调用逻辑在MOELayer中完成。
        raise NotImplementedError


class MOELayer(Base):
    """
    MoE核心逻辑层。
    该层接收输入，通过路由器（gate）获取路由决策，然后将输入分发给相应的专家，
    最后将专家的输出加权聚合起来。
    """

    def __init__(self,
                 gate: Module,  # 路由器实例
                 experts: Module,  # 专家容器实例
                 ep_size: int,  # Expert Parallelism size, 专家并行的组大小
                 num_local_experts: int,  # 本地专家数量
                 moe_use_mixtral_gating: bool,  # 是否使用Mixtral门控
                 moe_feature_no_mul_topk: bool  # 是否在聚合前不对输入进行top_k缩放
                 ) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts
        self.moe_use_mixtral_gating = moe_use_mixtral_gating
        self.moe_feature_no_mul_topk = moe_feature_no_mul_topk

    def forward(self, *input: Tensor, gate_residual=None, **kwargs: Any) -> Tuple[Tensor, Tensor]:
        # 获取输入的维度信息
        d_model = input[0].shape[-1]
        # 将输入从 [批次大小, 序列长度, d_model] reshape为 [token数量, d_model]
        reshaped_input = input[0].reshape(-1, d_model)

        # 创建一个与输入形状相同的零张量，用于累积专家的输出
        output = torch.zeros_like(reshaped_input)

        # 调用路由器获取路由信息和门控残差
        expert_info, gate_residual = self.gate(reshaped_input, gate_residual)

        # --- 输入缩放（可选）---
        # 在一些标准的MoE实现中，为了保持总计算量（FLOPs）与非MoE模型大致相当，
        # 会将输入乘以top_k。这里提供了不进行此操作的选项。
        if not (self.moe_use_mixtral_gating or self.moe_feature_no_mul_topk):
            reshaped_input *= MOE_TOP_K

        # --- 核心路由与计算循环 ---
        # 遍历由路由器返回的每个专家的信息
        for expert_id, token_indices_and_gates in expert_info.items():
            indices, gating_scores = token_indices_and_gates  # indices是token ID, gating_scores是对应的权重

            # 如果没有任何token被分配给这个专家，则跳过
            if indices.numel() == 0:
                continue

            # 将权重分数扩展一个维度，以便进行广播乘法 [num_tokens_for_expert, 1]
            gating_scores = gating_scores.unsqueeze(-1)
            # 根据索引从`reshaped_input`中选出需要由该专家处理的tokens
            tokens_for_expert = reshaped_input.index_select(dim=0, index=indices)

            # 将选中的tokens传入对应的专家模块进行计算
            expert_output = self.experts.experts[expert_id](tokens_for_expert)

            # 将专家的输出与门控权重相乘
            expert_output *= gating_scores

            # 使用 index_add_ 将加权后的专家输出“放回”到输出张量的正确位置
            # `index_add_` 是一个原地操作，可以高效地实现稀疏更新，避免冲突
            output.index_add_(dim=0, index=indices, source=expert_output)

        # 将输出张量reshape回原始的输入形状
        output = output.reshape(input[0].shape)

        return output, gate_residual


class MOE(torch.nn.Module):
    """
    顶层MoE封装模块。
    这个模块将Router, Experts, 和 MOELayer组装在一起，构成一个完整的MoE块，
    可以方便地集成到大语言模型中。
    """

    def __init__(self,
                 hidden_size,  # 模型的隐藏层维度
                 expert,  # “原型”专家模块
                 num_experts=1,  # 专家总数
                 ep_size=1,  # 专家并行大小（用于分布式训练）
                 moe_use_mixtral_gating=False,
                 moe_2layer_gate=True,
                 moe_use_logits_norm=False,
                 moe_gate_norm_std=1.0,
                 moe_feature_no_mul_topk=False):
        super(MOE, self).__init__()

        self.ep_size = ep_size
        self.num_experts = num_experts
        # 计算分配到当前设备/进程的本地专家数量
        self.num_local_experts = num_experts // self.ep_size

        # 保存所有配置参数
        self.moe_use_mixtral_gating = moe_use_mixtral_gating
        self.moe_2layer_gate = moe_2layer_gate
        self.moe_use_logits_norm = moe_use_logits_norm
        self.moe_gate_norm_std = moe_gate_norm_std
        self.moe_feature_no_mul_topk = moe_feature_no_mul_topk

        # 实例化专家容器
        experts = Experts(expert, self.num_local_experts)

        # 实例化完整的MoE层，传入所有需要的组件和配置
        self.moe = MOELayer(Router(hidden_size,
                                   num_experts,
                                   self.moe_use_mixtral_gating,
                                   self.moe_2layer_gate,
                                   self.moe_use_logits_norm,
                                   self.moe_gate_norm_std),
                            experts,
                            self.ep_size,
                            self.num_local_experts,
                            self.moe_use_mixtral_gating,
                            self.moe_feature_no_mul_topk,
                            )

    def forward(self, hidden_states, used_token=None, gate_residual=None):
        """
        最终的前向接口。
        :param hidden_states: 来自模型上一层的隐藏状态。
        :param used_token: 此处未使用，可能为未来扩展或兼容性保留。
        :param gate_residual: 来自前一个MoE层的门控残差。
        :return: MoE层的输出和本层的门控残差。
        """
        output, gate_residual = self.moe(hidden_states, used_token, gate_residual=gate_residual)
        return output, gate_residual


class VisionMOE(torch.nn.Module):
    """
    一个将基于Token的MOE模块适配到图像数据的封装层。
    它处理 [B, C, W, H] -> [B, W, H, C] 的维度转换。
    """

    def __init__(self, channels, proto_expert, **moe_kwargs):
        """
        :param channels: 图像的通道数 (C)，等同于 hidden_size 或 d_model。
        :param proto_expert: 一个“原型”专家模块，其输入和输出维度应为 channels。
        :param moe_kwargs: 传递给底层 MOE 模块的其他所有参数
                           (如 num_experts, moe_2layer_gate 等)。
        """
        super().__init__()
        self.moe_layer = MOE(
            hidden_size=channels,
            expert=proto_expert,
            **moe_kwargs
        )

    def forward(self, x: Tensor, gate_residual: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        :param x: 输入的图像张量，形状为 [B, C, W, H]。
        :param gate_residual: 可选的门控残差。
        :return: 一个元组 (output_tensor, new_gate_residual)
                 output_tensor 的形状为 [B, C, W, H]。
        """
        # 检查输入形状是否合法
        if x.ndim != 4:
            raise ValueError(f"期望4D输入 (B, C, W, H)，但得到 {x.ndim}D")

        b, c, w, h = x.shape

        # 1. 维度重排：将通道维度 C 移到最后
        # [B, C, W, H] -> [B, W, H, C]
        x_permuted = x.permute(0, 2, 3, 1)

        # 2. 将重排后的张量送入 MoE 层
        # MoE 内部会将其看作 [B*W*H, C]

        moe_output, new_gate_residual = self.moe_layer(x_permuted, gate_residual=gate_residual)

        # 3. 将输出维度重排回标准图像格式
        # [B, W, H, C] -> [B, C, W, H]
        output_permuted = moe_output.permute(0, 3, 1, 2)

        return output_permuted, new_gate_residual


if __name__ == '__main__':

    # --- 测试 VisionMOE ---
    print("\n\n" + "=" * 20)
    print("  测试 VisionMOE 模块")
    print("=" * 20)

    # 图像超参数
    B, C, W, H = 4, 64, 32, 32  # 批大小, 通道数, 宽高
    num_experts = 4

    # 实例化一个给Vision用的原型专家
    vision_proto_expert = ProtoExpert(hidden_size=C)

    # 实例化VisionMOE封装层
    vision_moe_layer = VisionMOE(
        channels=C,
        proto_expert=vision_proto_expert,
        num_experts=num_experts,
        moe_2layer_gate=True
    )

    print("\n--- 检查已实例化的专家序列 ---")

    # 1. 定位到存储专家模块的 ModuleList
    expert_list = vision_moe_layer.moe_layer.moe.experts.experts

    # 2. 遍历列表并打印每个专家的索引和类型
    for i, expert_module in enumerate(expert_list):
        # expert_module.__class__.__name__ 可以获取到类名字符串
        print(f"  - Expert {i}: {expert_module.__class__.__name__}")



    # 创建假的图像输入
    dummy_image_input = torch.randn(B, C, W, H)
    print(f"输入图像张量形状: {dummy_image_input.shape}")

    # 前向传播
    vision_output, vision_gate_residual = vision_moe_layer(dummy_image_input)


    # 检查输出
    print(f"输出图像张量形状: {vision_output.shape}")
    assert vision_output.shape == dummy_image_input.shape
    print("VisionMOE 输出形状正确！")


