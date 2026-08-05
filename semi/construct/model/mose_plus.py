from collections import defaultdict
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MOE_TOP_K = 2
CONSTANT_EXPERTS = 1


class ProtoExpert(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class CopyExpert(nn.Module):
    def __init__(self, expert: nn.Module):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class ZeroExpert(nn.Module):
    def __init__(self, expert: nn.Module):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(inputs)


class ConstantExpertWithNoise(nn.Module):
    def __init__(self, expert: ProtoExpert, noise_std: float = 0.001):
        super().__init__()
        self.noise_std = noise_std
        self.constant = nn.Parameter(torch.empty((expert.hidden_size, )))
        nn.init.normal_(self.constant)
        self.wg = nn.Linear(expert.hidden_size, 2, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.softmax(self.wg(inputs))
        constant = self.constant
        if self.training and self.noise_std > 0:
            constant = constant + torch.randn_like(constant) * self.noise_std
        return (torch.einsum('b,bd->bd', weight[:, 0].type_as(inputs), inputs) +
                torch.einsum('b,d->bd', weight[:, 1].type_as(inputs), constant.type_as(inputs)))


def gating(logits: torch.Tensor,
           use_mixtral_gating: bool = False,
           use_logits_norm: bool = False,
           gate_norm_std: float = 1.0) -> Dict[int, List[torch.Tensor]]:
    num_experts = logits.size(1)
    if use_mixtral_gating:
        if use_logits_norm:
            logits_std = logits.std(dim=1, keepdim=True)
            logits = logits / (logits_std / gate_norm_std)
        gates, indices = torch.topk(logits, k=MOE_TOP_K, dim=1)
        gates = F.softmax(gates, dim=1)
    else:
        if use_logits_norm:
            logits_std = logits.std(dim=1, keepdim=True)
            gates = F.softmax(logits / (logits_std / gate_norm_std), dim=1)
        else:
            gates = F.softmax(logits, dim=1)
        gates, indices = torch.topk(gates, k=MOE_TOP_K, dim=1)
        gates = torch.where(indices == (num_experts - 1), torch.zeros_like(gates), gates)
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)

    expert_info = defaultdict(list)
    for expert_id in range(num_experts):
        token_ids, score_ids = torch.nonzero(indices == expert_id, as_tuple=True)
        expert_info[expert_id] = [token_ids, gates[token_ids, score_ids]]
    return expert_info


class Router(nn.Module):
    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 use_mixtral_gating: bool,
                 use_2layer_gate: bool,
                 use_logits_norm: bool,
                 gate_norm_std: float):
        super().__init__()
        if use_2layer_gate:
            self.wg = nn.Sequential(
                nn.Linear(model_dim, num_experts * 8, bias=False).float(),
                nn.Tanh(),
                nn.Linear(num_experts * 8, num_experts, bias=False).float(),
            ).float()
        else:
            self.wg = nn.Linear(model_dim, num_experts, bias=False).float()

        self.gate_map = nn.Linear(num_experts, num_experts, bias=False)
        self.use_mixtral_gating = use_mixtral_gating
        self.use_logits_norm = use_logits_norm
        self.gate_norm_std = gate_norm_std

    def forward(self,
                inputs: torch.Tensor,
                gate_residual: Optional[torch.Tensor] = None) -> Tuple[Dict[int, List[torch.Tensor]],
                                                                       torch.Tensor]:
        if isinstance(self.wg, nn.Linear):
            if self.wg.weight.dtype != torch.float32:
                self.wg = self.wg.float()
        elif self.wg[0].weight.dtype != torch.float32:
            self.wg = self.wg.float()

        logits = self.wg(inputs.float())
        if gate_residual is not None:
            logits = logits + self.gate_map(gate_residual.to(self.gate_map.weight.dtype))
        return gating(logits, self.use_mixtral_gating, self.use_logits_norm,
                      self.gate_norm_std), logits


class Experts(nn.Module):
    def __init__(self, expert: nn.Module, num_local_experts: int = 1):
        super().__init__()
        base_experts = max(0, num_local_experts - 2 - CONSTANT_EXPERTS)
        experts = [copy.deepcopy(expert) for _ in range(base_experts)]
        experts += [ConstantExpertWithNoise(expert, noise_std=0.001) for _ in range(CONSTANT_EXPERTS)]
        experts += [CopyExpert(expert), ZeroExpert(expert)]
        self.experts = nn.ModuleList(experts)


class MOELayer(nn.Module):
    def __init__(self,
                 gate: nn.Module,
                 experts: Experts,
                 ep_size: int,
                 num_local_experts: int,
                 use_mixtral_gating: bool,
                 feature_no_mul_topk: bool):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts
        self.use_mixtral_gating = use_mixtral_gating
        self.feature_no_mul_topk = feature_no_mul_topk

    def forward(self,
                inputs: torch.Tensor,
                used_token=None,
                gate_residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        d_model = inputs.shape[-1]
        reshaped = inputs.reshape(-1, d_model)
        output = torch.zeros_like(reshaped)

        expert_info, gate_residual = self.gate(reshaped, gate_residual)
        if not (self.use_mixtral_gating or self.feature_no_mul_topk):
            reshaped = reshaped * MOE_TOP_K

        for expert_id, (indices, scores) in expert_info.items():
            if indices.numel() == 0:
                continue
            tokens = reshaped.index_select(dim=0, index=indices)
            expert_output = self.experts.experts[expert_id](tokens)
            output.index_add_(dim=0,
                              index=indices,
                              source=expert_output * scores.unsqueeze(-1))

        return output.reshape_as(inputs), gate_residual


class MOE(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 expert: nn.Module,
                 num_experts=1,
                 ep_size=1,
                 moe_use_mixtral_gating=False,
                 moe_2layer_gate=True,
                 moe_use_logits_norm=False,
                 moe_gate_norm_std=1.0,
                 moe_feature_no_mul_topk=False):
        super().__init__()
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size

        experts = Experts(expert, self.num_local_experts)
        self.moe = MOELayer(
            Router(hidden_size, num_experts, moe_use_mixtral_gating, moe_2layer_gate,
                   moe_use_logits_norm, moe_gate_norm_std),
            experts,
            ep_size,
            self.num_local_experts,
            moe_use_mixtral_gating,
            moe_feature_no_mul_topk,
        )

    def forward(self,
                hidden_states: torch.Tensor,
                used_token=None,
                gate_residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.moe(hidden_states, used_token, gate_residual=gate_residual)


class VisionMOE(nn.Module):
    def __init__(self, channels: int, proto_expert: ProtoExpert, **moe_kwargs):
        super().__init__()
        self.moe_layer = MOE(hidden_size=channels, expert=proto_expert, **moe_kwargs)

    def forward(self,
                x: torch.Tensor,
                gate_residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f'Expected 4D input, got {x.ndim}D')
        x = x.permute(0, 2, 3, 1)
        output, gate_residual = self.moe_layer(x, gate_residual=gate_residual)
        return output.permute(0, 3, 1, 2), gate_residual
