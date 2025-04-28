#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : Azure-Tang, Boxin Zhang, chenht2022
Date         : 2024-07-25 11:25:24
Version      : 0.1.0
LastEditors  : Azure 
LastEditTime : 2024-08-29 09:41:10
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''

import json
from typing import Any, Union
import numpy as np
import numpy.typing as npt
from torch import Tensor, nn
import torch.nn.functional as F
import torch
import sys, os
from ktransformers.operators.base_operator import BaseInjectedModule
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Release"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Debug"))
import cpuinfer_ext
from cpuinfer_ext.moe import MOEConfig, MOE
import ctypes
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.util.utils import InferenceState
from ktransformers.server.config.config import Config
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from abc import ABC, abstractmethod
from ktransformers.operators.linear import KLinearMarlin, KLinearTorch, KTransformersLinear
import time
from ktransformers.operators.cpuinfer import CPUInfer


# class Base(BaseInjectedModule, ABC):
class KExpertsBase(ABC):
    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, device: str = "cuda", **kwargs):
        # super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.key = key
        self.gguf_loader = gguf_loader
        self.config = config
        self.device = device
    
    @abstractmethod
    def forward(self, input_tensor, expert_ids, weights):
        pass

    @abstractmethod
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str = "cpu", warmup: bool = False):
        pass
    
    @abstractmethod
    def unload():
        pass

    def load_weights(self, override_key: str | None = None, device: str = "cpu"):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                targets = [".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight" ]
                tensors = self.load_multi(key, targets, device=device)
                gate = tensors[".ffn_gate_exps.weight"]
                up = tensors[".ffn_up_exps.weight"]
                down = tensors[".ffn_down_exps.weight"]
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
            elif key + ".ffn_down.0.weight" in self.gguf_loader.tensor_info:
                # for supporting  Mixtral-8x7B-Instuct  
                gate = []
                up = []
                down = []
                for i in range(8):
                    gatei, upi, downi = f".ffn_gate.{i}.weight", f".ffn_up.{i}.weight", f".ffn_down.{i}.weight"
                    targets = [gatei, upi, downi]
                    tensors = self.load_multi(key, targets, device=device)
                    gate_it, up_it, down_it = tensors[gatei], tensors[upi], tensors[downi]
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = torch.stack(gate)
                up = torch.stack(up)
                down = torch.stack(down)
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate.0.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up.0.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down.0.weight"]["ggml_type"]
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")
            res = {key:{"gate": gate, "up": up, "down": down, "gate_type": gate_type, "up_type": up_type, "down_type": down_type}}
        return res
    
    def load_multi(self, key: str, keys: list[str], device: str = "cpu"):
        tensors = {}
        for k in keys:
            tensors[k] = self.gguf_loader.load_gguf_tensor(key + k, device=device)
        return tensors

class KExpertsCPU(KExpertsBase):
    input_tensor_cpu: Tensor = None
    size_tensor_cpu: Tensor = None
    expert_ids_cpu: Tensor = None
    weights_cpu: Tensor = None
    output_cpu: Tensor = None
    output_gpu_map: dict = {} # Manage output tensor buffer on different gpu
    #stream_map:dict = {} # Manage cuda stream on different gpu
    #gguf_loader:GGUFLoader = None
    CPU_INFER = None
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        out_device: str = "cuda", # this device mean which device the output should on. TODO: support cpu.
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        if KExpertsCPU.CPU_INFER is None:
            KExpertsCPU.CPU_INFER = CPUInfer(Config().cpu_infer)
        #if KExpertsCPU.gguf_loader is None:
        #    KExpertsCPU.gguf_loader = GGUFLoader("/mnt/data/model/DeepseekV3-q4km-gguf")
        self.gguf_loader = gguf_loader
        assert device.lower() == "cpu", "KExpertsCPU can only be loaded on CPU"
        self.n_routed_experts = n_routed_experts
        self.out_device = out_device

    def load(self, w: dict | nn.Parameter | tuple | None = None, device:str|None = None, warmup:bool = False):
        if device:
            assert device.lower() == "cpu", "KExpertsCPU can only be loaded on CPU, Parameter \"device\" can be cpu or None."
        if w is None: w = self.load_weights()[self.key]
        self.gate = w["gate"]
        self.up = w["up"]
        self.down = w["down"]
        self.gate_type = w["gate_type"]
        self.up_type = w["up_type"]
        self.down_type = w["down_type"]
        gate_ptr = ctypes.addressof(
            ctypes.cast(self.gate.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        up_ptr = ctypes.addressof(
            ctypes.cast(self.up.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        down_ptr = ctypes.addressof(
            ctypes.cast(self.down.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        #print(self.gate_type, self.up_type, self.down_type)
        n_routed_experts = self.n_routed_experts
        # n_routed_experts = len(self.orig_module)
        moe_config = MOEConfig(
            n_routed_experts,
            self.config.num_experts_per_tok,
            self.config.hidden_size,
            self.config.moe_intermediate_size,
            64,
            10,
            1024,
            gate_ptr,
            up_ptr,
            down_ptr,
            self.gate_type,
            self.up_type,
            self.down_type,
            30, # TODO: get from model.dtype
        )
        # print(n_routed_experts, hidden_size, moe_intermediate_size)
        num_experts_per_tok = self.config.num_experts_per_tok
        self.moe = MOE(moe_config)
        self.cpu_infer = KExpertsCPU.CPU_INFER
        if warmup:
            self.cpu_infer.submit(self.moe.warm_up())
            self.cpu_infer.sync()
        if self.out_device not in KExpertsCPU.output_gpu_map:
            KExpertsCPU.output_gpu_map[self.out_device] = torch.zeros((self.config.hidden_size), device=self.out_device)
        if KExpertsCPU.input_tensor_cpu == None:
            KExpertsCPU.input_tensor_cpu = torch.zeros((self.config.hidden_size), device="cpu", pin_memory=True)
            KExpertsCPU.size_tensor_cpu = torch.zeros((1), device="cpu", dtype=torch.int64, pin_memory=True)
            KExpertsCPU.expert_ids_cpu = torch.zeros((num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True)
            KExpertsCPU.weights_cpu = torch.zeros((num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True)
            KExpertsCPU.output_cpu = torch.zeros((self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16)
            
    def submit_for_one_decode(self, input_tensor, expert_ids, weights, size=None):
        if not size:
            size = expert_ids.size(0)
        KExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
        KExpertsCPU.size_tensor_cpu.copy_(torch.tensor([size], dtype=torch.int64), non_blocking=True)
        KExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
        KExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
        self.cpu_infer.submit_with_cuda_stream(
            torch.cuda.current_stream(self.out_device).cuda_stream,
            self.moe.forward(
                1,
                size,
                KExpertsCPU.size_tensor_cpu.data_ptr(),
                KExpertsCPU.expert_ids_cpu.data_ptr(),
                KExpertsCPU.weights_cpu.data_ptr(),
                KExpertsCPU.input_tensor_cpu.data_ptr(),
                KExpertsCPU.output_cpu.data_ptr(),
            ),
        )
        
    def sync_for_one_decode(self):
        self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream)
        KExpertsCPU.output_gpu_map[self.out_device].copy_(KExpertsCPU.output_cpu, non_blocking=True)
        return KExpertsCPU.output_gpu_map[self.out_device]

    def submit(self, input_tensor, expert_ids, weights, shared_experts=None, size_per_token=None):
        self.input_tensor = input_tensor.contiguous().cpu()
        self.expert_ids = expert_ids.contiguous().cpu()
        self.weights = weights.contiguous().to(torch.float32).cpu()
        self.output = torch.empty_like(self.input_tensor).contiguous().cpu()
        if size_per_token is not None:
            self.size_per_token = size_per_token.contiguous().cpu()
        else:
            self.size_per_token = [self.expert_ids.size(1)] * self.expert_ids.size(0)
            self.size_per_token = torch.tensor(self.size_per_token, dtype=self.expert_ids.dtype)
            self.size_per_token = self.size_per_token.contiguous().cpu()
        self.shared_experts = shared_experts

        self.cpu_infer.submit(
            self.moe.forward(
                self.expert_ids.size(0),
                self.expert_ids.size(1),
                self.size_per_token.data_ptr(),
                self.expert_ids.data_ptr(),
                self.weights.data_ptr(),
                self.input_tensor.data_ptr(),
                self.output.data_ptr(),
            )
        )

    def sync(self):
        if self.shared_experts is not None:
            shared_expert_out = self.shared_experts()
            shared_expert_out.resize_(self.output.size())
        self.cpu_infer.sync()
        if self.shared_experts is not None:
            self.output += shared_expert_out.to(self.output.device)
        return self.output.to(device=object.__getattribute__(self, "out_device"))

    def forward(self, input_tensor, expert_ids, weights, shared_experts=None, size_per_token=None):
        self.submit(input_tensor, expert_ids, weights, shared_experts, size_per_token)
        return self.sync()
    
    def unload(self):
        return

    def load_weights(self, override_key: str | None = None, device: str = "cpu"):
        # TODO: support Bias
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if self.gguf_loader.safetensor_loader is not None:
                # using a temp ugly way to temprary load the tensor
                gate = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_gate_exps.weight").numpy()
                up = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_up_exps.weight").numpy()
                down = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_down_exps.weight").numpy()
                gate_type = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_gate_exps.ggml_type").item()
                up_type = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_up_exps.ggml_type").item()
                down_type = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_down_exps.ggml_type").item()
            
            elif key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
            elif key + ".ffn_down.0.weight" in self.gguf_loader.tensor_info:
                # for supporting  Mixtral-8x7B-Instuct  
                gate = []
                up = []
                down = []
                for i in range(8):
                    gate_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_gate.{i}.weight")
                    up_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_up.{i}.weight")
                    down_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_down.{i}.weight")
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = np.stack(gate)
                up = np.stack(up)
                down = np.stack(down)
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate.0.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up.0.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down.0.weight"]["ggml_type"]
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")
            res = {key:{"gate": gate, "up": up, "down": down, "gate_type": gate_type, "up_type": up_type, "down_type": down_type}}
        return res
    
class KExpertsMarlin(KExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.load_size = config.load_size
        self.prefetch_size = config.prefetch_size
        self.device_usage = config.device_usage
        self.expert_num = n_routed_experts
        self.layer_idx = int(key.split(".")[1])
        self.loaded_experts_idx = []
        self.act_fn = ACT2FN[config.hidden_act]
        assert device.lower() != "cpu", "Marlin experts can only be loaded on GPU"
        self.device = device
        self.type = torch.get_default_dtype()
        self.elements_per_tensor = config.moe_intermediate_size * config.hidden_size

        # create empty marlin experts according to the number of experts per token
        # up
        self.up_projs = [KLinearMarlin(key+ "." + "ffn_up_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]
        # gate
        self.gate_projs = [KLinearMarlin(key+ "." + "ffn_gate_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]
        # down
        self.down_projs = [KLinearMarlin(key+ "." + "ffn_down_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]
        self.loading_lock = [[torch.cuda.Event() for _ in range(3)] for _ in range(self.expert_num)]
        # 创建 KExpertsCPU 实例
        self.cpu_experts = KExpertsCPU(key, gguf_loader, config, n_routed_experts, orig_module, device="cpu")

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = None, warmup: bool = False):
        if device is None: device = self.device
        assert device.lower() != "cpu", "Marlin experts can only be loaded on GPU"
        if w is None:
            w = self.load_weights()
        if KExpertsCache._instance is None:
            self.up_projs[0].load(nn.Parameter(w["up"][0, ...]), device=device)
            # weights = self.up_projs[0].move() #这里开始self.up_projs[0].marlin_q_w开始变成None，下面就会出bug
            # self.up_projs[0].unload()
            self.cache = KExpertsCache(
                self.config,
                shape=self.up_projs[0].marlin_q_w_shape,
                dtype=torch.int32,
                load_size=self.load_size,
                prefetch_size=self.prefetch_size,
                devices_usage=self.device_usage,
            )
            weights = None
        else:
            self.cache = KExpertsCache._instance
        if isinstance(w, dict):
            for i in range(self.expert_num):
                self.up_projs[i].load(nn.Parameter(w["up"][i, ...]), device=device)
                self.gate_projs[i].load(nn.Parameter(w["gate"][i, ...]), device=device)
                self.down_projs[i].load(nn.Parameter(w["down"][i, ...]), device=device)
                self.loaded_experts_idx.append(i)
                self.cache.load_weights_to_storage(
                    self.gate_projs[i].move(),
                    self.up_projs[i].move(),
                    self.down_projs[i].move(),
                    i + self.layer_idx * self.expert_num,
                    dtype=torch.int32,
                )
            self.dtype = self.up_projs[0].dtype
        self.cpu_experts.load()
        return 

    def unload(self):
        for i in self.loaded_experts_idx:
            self.up_projs[i].unload()
            self.gate_projs[i].unload()
            self.down_projs[i].unload()
        self.loaded_experts_idx = []

    def load_weights(self, override_key: str | None = None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None

        for key in keys:
            if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.load_gguf_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.load_gguf_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.load_gguf_tensor(key + ".ffn_down_exps.weight")
                # tensors = self.load_multi(key, [".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight"])
            res = {"gate": gate, "up": up, "down": down}
        return res

    def forward(
        self,
        input_tensor: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
        shared_experts: dict = None,
    ) -> torch.Tensor:
        start = time.time()
        org_device = input_tensor.device
        org_dtype = input_tensor.dtype
        input_tensor = input_tensor.to(self.device)
        expert_ids = expert_ids.to(self.device)
        weights = weights.to(self.device)
        generate = input_tensor.size(0) == 1
        final_hidden_states = torch.zeros_like(input_tensor)
        input_tensor = input_tensor.to(self.dtype)
        weights = weights.to(self.dtype)
        expert_mask = torch.nn.functional.one_hot(expert_ids, num_classes=self.expert_num).permute(2, 1, 0)

        id_counts = None
        if not generate:
            unique_selected_experts = torch.unique(expert_ids).tolist()
            id_counts = torch.bincount(expert_ids.view(-1))
        else:
            unique_selected_experts = expert_ids.squeeze().tolist()
            id_counts = torch.tensor([1] * self.expert_num, dtype=torch.int64)
        load_idxs, unload_idxs, org_indexes = self.cache.get_expert_place(unique_selected_experts, self.layer_idx)
        cpu_idxs, gpu_idxs, cpu_indexes = self.cache.calc_experts(
            load_idxs, unload_idxs, org_indexes, generate, id_counts
        )
        self.cache.wait_prefetch()
        if not generate:
            idx_top_x_list = [torch.where(expert_mask[expert_idx]) for expert_idx in range(self.expert_num)]
            expert_weights, sorted_idx = self.cache.get_experts_weights(
                gpu_idxs, self.layer_idx, self.loading_lock, True
            )
            if len(cpu_idxs) != 0:
                cpu_experts_idxs = [[] for _ in range(input_tensor.size(0))]
                cpu_indexes = [[] for _ in range(input_tensor.size(0))]
                for expert_idx in cpu_idxs:
                    idx, top_x = idx_top_x_list[expert_idx]
                    for i, x in enumerate(top_x):
                        cpu_experts_idxs[x].append(expert_idx)
                        cpu_indexes[x].append(idx[i].item())
                size_per_token = torch.tensor(
                    [len(cpu_experts_idxs[i]) for i in range(len(cpu_experts_idxs))], dtype=torch.int64
                )
                for i in range(len(cpu_experts_idxs)):
                    cpu_experts_idxs[i].extend([-1] * (expert_ids.size(1) - len(cpu_experts_idxs[i])))
                    cpu_indexes[i].extend([0] * (expert_ids.size(1) - len(cpu_indexes[i])))
                cpu_experts_idxs = torch.tensor(cpu_experts_idxs, device=expert_ids.device)
                cpu_weights = torch.zeros(
                    (len(cpu_experts_idxs), weights.size(1)), dtype=weights.dtype, device=weights.device
                )
                for i in range(len(weights)):
                    cpu_weights[i] = weights[i][cpu_indexes[i]]
                self.cpu_experts.submit(input_tensor, cpu_experts_idxs, cpu_weights, shared_experts, size_per_token)
            for expert_idx in sorted_idx:
                idx, top_x = idx_top_x_list[expert_idx]
                current_state = input_tensor[None, top_x].squeeze(0)
                down_proj = self.down_projs[expert_idx]
                gate_proj = self.gate_projs[expert_idx]
                up_proj = self.up_projs[expert_idx]
                gate_weight, up_weight, down_weight = expert_weights[expert_idx]
                self.loading_lock[expert_idx][0].wait()
                G = gate_proj(current_state, gate_weight)
                A = self.act_fn(G)
                self.loading_lock[expert_idx][1].wait()
                U = up_proj(current_state, up_weight)
                H = A * U
                self.loading_lock[expert_idx][2].wait()
                D = down_proj(H, down_weight)
                current_hidden_states = D * weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states)
            if len(cpu_idxs) != 0:
                final_hidden_states += self.cpu_experts.sync()
            if shared_experts is not None and len(cpu_idxs) == 0:
                shared_out = shared_experts()
                final_hidden_states += shared_out
            self.cache.reset_buffer(self.layer_idx)
            return final_hidden_states.to(device=org_device, dtype=org_dtype)
        if len(cpu_idxs) != 0:
            num = len(cpu_idxs)
            cpu_idxs.extend(gpu_idxs)
            cpu_weights = weights[0][cpu_indexes]
            cpu_idxs = torch.tensor(cpu_idxs)
            self.cpu_experts.submit_for_one_decode(input_tensor[0], cpu_idxs, cpu_weights, num)
        if shared_experts is not None:
            shared_out = shared_experts()
            final_hidden_states += shared_out
        if len(gpu_idxs) != 0:
            expert_weights, sorted_idx = self.cache.get_experts_weights(gpu_idxs, self.layer_idx, self.loading_lock)
            self.cache.prefetch_expert(layer_idx=self.layer_idx)
            for expert_idx in sorted_idx:
                idx, top_x = torch.where(expert_mask[expert_idx])
                current_state = input_tensor[None, top_x].squeeze(0)
                down_proj = self.down_projs[expert_idx]
                gate_proj = self.gate_projs[expert_idx]
                up_proj = self.up_projs[expert_idx]
                gate_weight, up_weight, down_weight = expert_weights[expert_idx]
                self.loading_lock[expert_idx][0].wait()
                G = gate_proj(current_state, gate_weight)
                A = self.act_fn(G)
                self.loading_lock[expert_idx][1].wait()
                U = up_proj(current_state, up_weight)
                H = A * U
                self.loading_lock[expert_idx][2].wait()
                D = down_proj(H, down_weight)
                current_hidden_states = D * weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states)
        if len(cpu_idxs) != 0:
            final_hidden_states += self.cpu_experts.sync_for_one_decode().unsqueeze(0)
        if generate and self.cache.stop_add == False:
            self.cache.add_time(len(gpu_idxs), time.time() - start)
        return final_hidden_states.to(device=org_device, dtype=org_dtype)
    
# untested, CUDA OOM
class KExpertsTorch(KExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.layer_idx = int(key.split(".")[1])
        self.expert_num = n_routed_experts
        # self.loaded_experts_idx = []
        self.act_fn = ACT2FN[config.hidden_act]
        self.device = device
        self.elements_per_tensor = config.moe_intermediate_size * config.hidden_size
        self.gate = [None for _ in range(self.expert_num)]
        self.up = [None for _ in range(self.expert_num)]
        self.down = [None for _ in range(self.expert_num)]
        self.dtype = torch.get_default_dtype()
        self.cache = KExpertsCache._instance
        self.loading_lock = [[torch.cuda.Event() for _ in range(3)] for _ in range(self.expert_num)]

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = None, warmup: bool = False):
        if device is None:
            device = self.device
        if w is None:
            w = self.load_weights(device=device)[self.key]
        if isinstance(w, dict):
            for expert_idx in range(self.expert_num):
                self.cache.load_weights_to_storage(
                    w["gate"][expert_idx],
                    w["up"][expert_idx],
                    w["down"][expert_idx],
                    expert_idx + self.layer_idx * self.expert_num,
                    dtype=self.dtype,
                )
        w = None
        return

    def unload(self):
        if self.cache is None:
            return
        for expert_idx in range(self.expert_num):
            self.cache.unload_expert_weights(
                expert_uid=expert_idx + self.layer_idx * self.expert_num,
            )

    def load_weights(self, override_key: str | None = None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None

        for key in keys:
            if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
            res = {"gate": gate, "up": up, "down": down}
        return res

    def forward(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:

        org_device = hidden_states_cpu.device
        hidden_states_cpu = hidden_states_cpu.to(self.device)
        selected_experts_cpu = selected_experts_cpu.to(self.device)
        routing_weights_cpu = routing_weights_cpu.to(self.device)
        
        batch_sequence_length, hidden_dim = hidden_states_cpu.size()

        final_hidden_states = torch.zeros(
            (batch_sequence_length, hidden_dim), dtype=self.gate.dtype, device=hidden_states_cpu.device
        )
        org_dtype = hidden_states_cpu.dtype
        hidden_states_cpu = hidden_states_cpu.to(self.gate.dtype)
        routing_weights_cpu = routing_weights_cpu.to(self.gate.dtype)
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.expert_num).permute(2, 1, 0)

        unique_selected_experts = torch.unique(selected_experts_cpu).tolist()
        self.cache.wait_prefetch()
        expert_weights, sorted_idx = self.cache.get_experts_weights(
            unique_selected_experts, self.layer_idx, self.loading_lock
        )
        self.cache.prefetch_expert(layer_idx=self.layer_idx)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in sorted_idx:
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            gate_weight, up_weight, down_weight = expert_weights[expert_idx]
            self.loading_lock[expert_idx][0].wait()
            G = current_state @ self.gate[expert_idx,...].T
            A = self.act_fn(G)
            self.loading_lock[expert_idx][1].wait()
            U = current_state @ self.up[expert_idx,...].T
            H = A * U  # Element-wise multiplication
            self.loading_lock[expert_idx][2].wait()
            current_hidden_states = H @ self.down[expert_idx,...].T * routing_weights_cpu[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states)


        return final_hidden_states.to(dtype=org_dtype, device=org_device)

class KScoreAwareCache:
    # S =α×TopP(s)+(1−α)×S
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(KScoreAwareCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, num_layers, num_experts, alpha=0.5, p=12):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.alpha = alpha
        self.p = p
        self.priority_scores = [0.0] * (num_layers * num_experts)
        self.priority_scores = torch.tensor(self.priority_scores, dtype=torch.float32, device="cuda")

    def update_priority_scores(self, layer_idx, routing_scores):
        top_p_scores, top_p_id = torch.topk(routing_scores, self.p, dim=1)
        top_p_scores = top_p_scores[0]
        top_p_id = top_p_id[0]

        start_idx = layer_idx * self.num_experts
        end_idx = (layer_idx + 1) * self.num_experts
        current_priority_scores = self.priority_scores[start_idx:end_idx]

        mask = torch.zeros(self.num_experts, dtype=torch.bool, device=top_p_id.device)
        mask[top_p_id] = True

        current_priority_scores[mask] = self.alpha * top_p_scores + (1 - self.alpha) * current_priority_scores[mask]
        current_priority_scores[~mask] = (1 - self.alpha) * current_priority_scores[~mask]

        self.priority_scores[start_idx:end_idx] = current_priority_scores

    def get_expert_to_unload(self, experts=None):
        # find the expert with the lowest priority score
        if experts is None:
            experts = torch.arange(self.num_experts, device=self.priority_scores.device)
        else:
            experts = torch.tensor(list(experts.keys()), device=self.priority_scores.device)

        expert_scores = self.priority_scores[experts]
        min_expert_idx = torch.argmin(expert_scores)
        min_expert = experts[min_expert_idx].item()

        return min_expert

    def get_experts_to_prefetch(self, layer_idx, k):
        # get the top k experts with the highest priority scores
        priority_scores = self.priority_scores[layer_idx * self.num_experts : (layer_idx + 1) * self.num_experts]
        top_k_scores, top_k_id = torch.topk(torch.tensor(priority_scores), k)
        return top_k_id

class KExpertsCache:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(KExpertsCache, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        config: PretrainedConfig,
        load_size=None,
        shape=None,
        prefetch_size=None,
        dtype=torch.get_default_dtype(),
        devices_usage=None,
    ):
        if not hasattr(self, "initialized"):
            if load_size is None:
                self.load_size = [16] * (config.num_hidden_layers)
            else:
                assert len(load_size) == config.num_hidden_layers
                self.load_size = load_size
            if prefetch_size is None:
                self.prefetch_size = 0
            else:
                self.prefetch_size = prefetch_size
            self.dtype = dtype
            if hasattr(config, "n_routed_experts"):
                self.model_n = 0
                self.expert_num_per_layer = config.n_routed_experts
                self.total_expert_num = config.n_routed_experts * (config.num_hidden_layers)
                self.hidden_size = config.hidden_size
                self.moe_intermediate_size = config.moe_intermediate_size
            elif hasattr(config, "num_local_experts"):
                self.model_n = 1
                self.expert_num_per_layer = config.num_local_experts
                self.total_expert_num = config.num_local_experts * (config.num_hidden_layers)
                self.hidden_size = config.hidden_size
                self.moe_intermediate_size = config.intermediate_size
            elif hasattr(config, "num_experts"):
                self.model_n = 2
                self.expert_num_per_layer = config.num_experts
                self.total_expert_num = config.num_experts * (config.num_hidden_layers)
                self.hidden_size = config.hidden_size
                self.moe_intermediate_size = config.intermediate_size
            else:
                raise ValueError("config error, n_routed_experts or num_local_experts or num_experts not found")
            if shape is None:
                self.gate_shape = torch.Size([self.moe_intermediate_size, self.hidden_size])
                self.up_shape = torch.Size([self.moe_intermediate_size, self.hidden_size])
                self.down_shape = torch.Size([self.hidden_size, self.moe_intermediate_size])
            else:
                self.gate_shape = shape
                self.up_shape = shape
                self.down_shape = shape
            # default to load experts on cuda:0
            if devices_usage is None:
                devices_usage = {}
                devices_usage["cuda:0"] = [i for i in range(0, config.num_hidden_layers)]
            self.devices_usage = devices_usage
            self.layer2device = {}
            for device, layers in devices_usage.items():
                for layer in layers:
                    self.layer2device[layer] = device

            # DRAM
            size_per_expert = self.dtype.itemsize * self.gate_shape.numel()
            total_size = size_per_expert * self.total_expert_num
            gate_large = torch.UntypedStorage(total_size).pin_memory()
            up_large = torch.UntypedStorage(total_size).pin_memory()
            down_large = torch.UntypedStorage(total_size).pin_memory()

            self.gate_storage = [
                gate_large[i * size_per_expert : (i + 1) * size_per_expert] for i in range(self.total_expert_num)
            ]
            self.up_storage = [
                up_large[i * size_per_expert : (i + 1) * size_per_expert] for i in range(self.total_expert_num)
            ]
            self.down_storage = [
                down_large[i * size_per_expert : (i + 1) * size_per_expert] for i in range(self.total_expert_num)
            ]

            # VRAM
            self.gate_memory = {}
            self.up_memory = {}
            self.down_memory = {}
            self.buffer_gate_memory = {}
            self.buffer_up_memory = {}
            self.buffer_down_memory = {}
            self.gate_views = {}
            self.up_views = {}
            self.down_views = {}
            self.buffer_gate_views = {}
            self.buffer_up_views = {}
            self.buffer_down_views = {}
            for device in devices_usage:
                load_experts_num = sum(self.load_size[layer_idx] for layer_idx in self.devices_usage[device])
                total_size = load_experts_num * size_per_expert
                total_gate_memory = torch.UntypedStorage(total_size, device=device)
                # divide gate_memory into load_experts_num experts
                self.gate_memory[device] = [
                    total_gate_memory[i * size_per_expert : (i + 1) * size_per_expert] for i in range(load_experts_num)
                ]
                self.gate_views[device] = [
                    torch.as_tensor(storage, dtype=dtype, device=device).view(self.gate_shape)
                    for storage in self.gate_memory[device]
                ]
                total_up_memory = torch.UntypedStorage(total_size, device=device)
                self.up_memory[device] = [
                    total_up_memory[i * size_per_expert : (i + 1) * size_per_expert] for i in range(load_experts_num)
                ]
                self.up_views[device] = [
                    torch.as_tensor(storage, dtype=dtype, device=device).view(self.up_shape)
                    for storage in self.up_memory[device]
                ]
                total_down_memory = torch.UntypedStorage(total_size, device=device)
                self.down_memory[device] = [
                    total_down_memory[i * size_per_expert : (i + 1) * size_per_expert] for i in range(load_experts_num)
                ]
                self.down_views[device] = [
                    torch.as_tensor(storage, dtype=dtype, device=device).view(self.down_shape)
                    for storage in self.down_memory[device]
                ]

                buffer_total_size = self.expert_num_per_layer * size_per_expert
                buffer_total_gate_memory = torch.UntypedStorage(buffer_total_size, device=device)
                self.buffer_gate_memory[device] = [
                    buffer_total_gate_memory[i * size_per_expert : (i + 1) * size_per_expert]
                    for i in range(self.expert_num_per_layer)
                ]
                self.buffer_gate_views[device] = [
                    torch.as_tensor(storage, dtype=dtype, device=device).view(self.gate_shape)
                    for storage in self.buffer_gate_memory[device]
                ]
                buffer_total_up_memory = torch.UntypedStorage(buffer_total_size, device=device)
                self.buffer_up_memory[device] = [
                    buffer_total_up_memory[i * size_per_expert : (i + 1) * size_per_expert]
                    for i in range(self.expert_num_per_layer)
                ]
                self.buffer_up_views[device] = [
                    torch.as_tensor(storage, dtype=dtype, device=device).view(self.up_shape)
                    for storage in self.buffer_up_memory[device]
                ]
                buffer_total_down_memory = torch.UntypedStorage(buffer_total_size, device=device)
                self.buffer_down_memory[device] = [
                    buffer_total_down_memory[i * size_per_expert : (i + 1) * size_per_expert]
                    for i in range(self.expert_num_per_layer)
                ]
                self.buffer_down_views[device] = [
                    torch.as_tensor(storage, dtype=dtype, device=device).view(self.down_shape)
                    for storage in self.buffer_down_memory[device]
                ]
                self.buffer_slots = [-1] * self.expert_num_per_layer
                # print(f"device {device} memory: {3 * total_gate_memory.size() / 1024 / 1024 / 1024}GB")
                # print(f"device {device} dtype: {dtype}")

            # current loaded experts index and position in memory
            self.loaded_experts_idx = {}
            for device, layers in self.devices_usage.items():
                self.loaded_experts_idx[device] = {}
                for layer in layers:
                    self.loaded_experts_idx[device][layer] = {}
            self.free_memory_slots, self.slots2layer = self.initialize_free_memory_slots()

            # CUDA Stream
            self.copy_stream = torch.cuda.Stream()
            self.prefetch_stream = torch.cuda.Stream()
            self.prefetch_lock = [[torch.cuda.Event() for _ in range(3)] for _ in range(self.expert_num_per_layer)]
            self.prefetching = False
            self.hits = 0
            self.misses = 0
            self.fetch_hits = 0
            self.fetch_misses = 0
            self.fetchs = {}
            self.initialized = True

            self.time_table = {}
            self.prefill = None
            self.stop_add = False

            self.score_aware_cache = KScoreAwareCache._instance

    def initialize_free_memory_slots(self):
        # each device has continuous indices, allocated according to the number of experts
        free_memory_slots = {}
        slots2layer = [-1] * sum(self.load_size)
        for device, layers in self.devices_usage.items():
            free_memory_slots[device] = {}
            current_index = 0
            for layer in layers:
                next_index = current_index + self.load_size[layer]
                free_memory_slots[device][layer] = list(range(current_index, next_index))
                slots2layer[current_index:next_index] = [layer] * self.load_size[layer]
                current_index = next_index
        return free_memory_slots, slots2layer

    def weight_to_storage(self, weight, dtype=torch.float16, device="cpu"):
        weight = weight.to(dtype)

        storage_size = weight.nbytes
        storage = torch.UntypedStorage(storage_size, device=device)

        a_view = torch.as_tensor(storage, dtype=dtype, device=device).view(weight.shape)
        a_view.copy_(weight)
        assert a_view.data_ptr() == storage.data_ptr()
        return storage

    def storage_to_weight(self, storage, shape, dtype=torch.float16, device="cuda"):
        weight = torch.as_tensor(storage, dtype=dtype, device=device).view(shape)
        return weight

    def load_expert_weights(
        self, expert_uid, init=False, non_blocking=True, loading_lock=None, stream=None, prefill=False
    ):
        layer_idx = expert_uid // self.expert_num_per_layer
        initial_layer_idx = layer_idx
        device = self.layer2device[layer_idx]

        if not self.free_memory_slots[device][layer_idx]:
            if prefill:
                for i, slot in enumerate(self.buffer_slots):
                    if slot == -1:
                        self.buffer_slots[i] = expert_uid
                        with torch.cuda.stream(self.copy_stream):
                            self.buffer_gate_memory[device][i].copy_(
                                self.gate_storage[expert_uid],
                                non_blocking=non_blocking,
                            )
                            loading_lock[0].record()
                            self.buffer_up_memory[device][i].copy_(
                                self.up_storage[expert_uid],
                                non_blocking=non_blocking,
                            )
                            loading_lock[1].record()
                            self.buffer_down_memory[device][i].copy_(
                                self.down_storage[expert_uid],
                                non_blocking=non_blocking,
                            )
                            loading_lock[2].record()
                        return
            if init:
                return
            # Use MRS
            if not self.free_memory_slots[device][layer_idx]:
                # offload_expert = next(iter(self.loaded_experts_idx[device][layer_idx]))
                offload_expert = self.score_aware_cache.get_expert_to_unload(self.loaded_experts_idx[device][layer_idx])
                layer_idx = self.unload_expert_weights(
                    offload_expert,
                    device,
                    layer_idx,
                )

        memory_slot = self.free_memory_slots[device][layer_idx].pop(0)
        # CUDA Stream
        if stream is None:
            stream = self.copy_stream
        with torch.cuda.stream(stream):
            self.gate_memory[device][memory_slot].copy_(
                self.gate_storage[expert_uid],
                non_blocking=non_blocking,
            )
            loading_lock[0].record()
            self.up_memory[device][memory_slot].copy_(
                self.up_storage[expert_uid],
                non_blocking=non_blocking,
            )
            loading_lock[1].record()
            self.down_memory[device][memory_slot].copy_(
                self.down_storage[expert_uid],
                non_blocking=non_blocking,
            )
            loading_lock[2].record()

        self.loaded_experts_idx[device][initial_layer_idx][expert_uid] = memory_slot

    def reset_offload(self):
        for device, layers in self.loaded_experts_idx.items():
            for layer, experts in layers.items():
                experts_copy = list(experts.keys())
                for expert in experts_copy:
                    self.unload_expert_weights(expert, device, layer)

    def unload_expert_weights(self, expert_uid, device=None, layer_idx=None):
        if device is None:
            layer_idx = expert_uid // self.expert_num_per_layer
            device = self.layer2device[layer_idx]
        if expert_uid not in self.loaded_experts_idx[device][layer_idx]:
            return
        memory_slot = self.loaded_experts_idx[device][layer_idx][expert_uid]
        self.free_memory_slots[device][self.slots2layer[memory_slot]].append(memory_slot)
        self.loaded_experts_idx[device][layer_idx].pop(expert_uid)
        return self.slots2layer[memory_slot]

    def get_expert_place(self, expert_idxs, layer_idx):
        load_ids = []
        unload_ids = []
        load_indices = []
        unload_indices = []
        device = self.layer2device[layer_idx]
        loaded_experts = self.loaded_experts_idx[device][layer_idx].keys()
        for idx, expert_idx in enumerate(expert_idxs):
            expert_uid = expert_idx + layer_idx * self.expert_num_per_layer
            if expert_uid in loaded_experts:
                load_ids.append(expert_idx)
                load_indices.append(idx)
            else:
                unload_ids.append(expert_idx)
                unload_indices.append(idx)
        return load_ids, unload_ids, load_indices + unload_indices

    def get_experts_weights(self, expert_idxs, layer_idx, loading_lock, prefill=False):
        experts = {}
        load_idxs = []
        unload_idxs = []
        device = self.layer2device[layer_idx]
        loaded_experts_keys = self.loaded_experts_idx[device][layer_idx].keys()
        for expert_idx in expert_idxs:
            expert_uid = expert_idx + layer_idx * self.expert_num_per_layer
            if expert_uid in loaded_experts_keys:
                load_idxs.append(expert_idx)
                self.fetch_hits += 1
            else:
                unload_idxs.append(expert_idx)
                self.fetch_misses += 1
            if layer_idx in self.fetchs.keys():
                if expert_uid in self.fetchs[layer_idx]:
                    self.hits += 1
                else:
                    self.misses += 1
        if layer_idx in self.fetchs.keys():
            self.fetchs[layer_idx] = []

        sorted_idxs = load_idxs + unload_idxs
        for expert_idx in sorted_idxs:
            experts[expert_idx] = self.get_expert_weights(
                expert_idx + layer_idx * self.expert_num_per_layer,
                loading_lock=loading_lock[expert_idx],
                prefill=prefill,
            )
        return experts, sorted_idxs

    def prefetch_expert(self, layer_idx):
        if layer_idx + 1 < len(self.load_size):
            next_layer = layer_idx + 1
        else:
            return
        self.prefetching = True
        experts = self.score_aware_cache.get_experts_to_prefetch(next_layer, self.prefetch_size)
        for idx, expert_idx in enumerate(experts):
            device = self.layer2device[next_layer]
            expert_uid = expert_idx.item() + next_layer * self.expert_num_per_layer #原版没有item，在prefetch时会出错
            self.fetchs[next_layer] = self.loaded_experts_idx[device][next_layer].keys()
            if not expert_uid in self.loaded_experts_idx[device][next_layer].keys():
                self.load_expert_weights(
                    expert_uid,
                    non_blocking=True,
                    loading_lock=self.prefetch_lock[idx],
                    stream=self.prefetch_stream,
                )

    def wait_prefetch(self):
        for lock in self.prefetch_lock:
            for l in lock:
                l.wait()
        self.prefetching = False

    def get_expert_weights(self, expert_uid, loading_lock=None, prefill=False):
        layer = expert_uid // self.expert_num_per_layer
        device = self.layer2device[layer]
        if expert_uid in self.loaded_experts_idx[device][layer].keys():
            self.loaded_experts_idx[device][layer][expert_uid] = self.loaded_experts_idx[device][layer].pop(expert_uid)
            memory_slot = self.loaded_experts_idx[device][layer][expert_uid]
            expert = (
                self.gate_views[device][memory_slot],
                self.up_views[device][memory_slot],
                self.down_views[device][memory_slot],
            )
            return expert
        if expert_uid in self.buffer_slots:
            memory_slot = self.buffer_slots.index(expert_uid)
            expert = (
                self.buffer_gate_views[device][memory_slot],
                self.buffer_up_views[device][memory_slot],
                self.buffer_down_views[device][memory_slot],
            )
            return expert
        self.load_expert_weights(expert_uid, non_blocking=True, loading_lock=loading_lock, prefill=prefill)
        return self.get_expert_weights(expert_uid, loading_lock=loading_lock)

    def reset_buffer(self, layer_idx):
        for i in range(len(self.buffer_slots)):
            self.buffer_slots[i] = -1
        if layer_idx < len(self.load_size) - 2:
            layer_idx += 1
            for i in range(self.expert_num_per_layer):
                self.load_expert_weights(
                    i + (layer_idx + 1) * self.expert_num_per_layer,
                    prefill=True,
                    loading_lock=self.prefetch_lock[i],
                )

    def load_weights_to_storage(self, gate, up, down, expert_uid, dtype):
        gate = self.weight_to_storage(gate, dtype, torch.device("cpu"))
        up = self.weight_to_storage(up, dtype, torch.device("cpu"))#up is nonetype - ytj
        down = self.weight_to_storage(down, dtype, torch.device("cpu"))
        self.gate_storage[expert_uid].copy_(gate)
        self.up_storage[expert_uid].copy_(up)
        self.down_storage[expert_uid].copy_(down)

        loading_lock = [torch.cuda.Event() for _ in range(3)]
        self.load_expert_weights(expert_uid, init=True, loading_lock=loading_lock)

    def calc_experts(self, load_idxs, unload_idxs, org_indexes, generate, id_counts):
        all_idxs = load_idxs + unload_idxs
        if not generate:
            return [], all_idxs, org_indexes
        else:
            return self.HSS(load_idxs, unload_idxs, id_counts)

    # HybridSchedulingStrategy
    def HSS(self, load_experts, unload_experts, id_counts):
        gpu_total_time = 0
        cpu_total_time = 0
        move_total_time = 0
        gpu_queue = []
        cpu_queue = []
        original_gpu_queue = []
        original_cpu_queue = []

        # sort load_experts by token_num
        load_experts_sorted = sorted(load_experts, key=lambda idx: id_counts[idx], reverse=True)
        # sort unload_experts by token_num
        unload_experts_sorted = sorted(unload_experts, key=lambda idx: id_counts[idx], reverse=True)
        sorted_experts = load_experts_sorted + unload_experts_sorted

        while sorted_experts:
            first_idx = sorted_experts[0]
            last_idx = sorted_experts[-1]
            is_move = first_idx in unload_experts
            gpu_time, intercept = self.model_gpu_time(id_counts[first_idx])
            if len(gpu_queue) == 0:
                gpu_time += intercept
            cpu_time, intercept = self.model_cpu_time(id_counts[last_idx])
            if len(cpu_queue) == 0:
                cpu_time += intercept
            if is_move:
                move_time = self.model_move_time()
                if move_total_time + move_time > gpu_total_time:
                    gpu_time += move_total_time + move_time - gpu_total_time
            if gpu_total_time + gpu_time <= cpu_total_time + cpu_time:
                if is_move:
                    move_total_time += move_time
                gpu_total_time += gpu_time
                gpu_queue.append(first_idx)
                original_gpu_queue.append(
                    load_experts.index(first_idx) if first_idx in load_experts else unload_experts.index(first_idx)
                )
                sorted_experts.pop(0)
            else:
                cpu_total_time += cpu_time
                cpu_queue.append(last_idx)
                original_cpu_queue.append(
                    load_experts.index(last_idx) if last_idx in load_experts else unload_experts.index(last_idx)
                )
                sorted_experts.pop(-1)

        return cpu_queue, gpu_queue, original_cpu_queue + original_gpu_queue

    def model_cpu_time(self, token_num):
        # need to be tested
        # deepseek 1 * token_num
        # mixtral 0.2 * token_num
        # qwen2 0.8 * token_num
        coef = "1 * {token_num}"
        intercept = 0.79
        return eval(coef.format(token_num=token_num)), intercept

    def model_gpu_time(self, token_num):
        # need to be tested
        coef = "0.5"
        intercept = 0  # shared_experts
        return eval(coef.format(token_num=token_num)), intercept

    def model_move_time(self):
        # need to be tested
        return 0.41791011235117914

    # 11.1
    def add_time(self, x, time):
        if x not in self.time_table:
            self.time_table[x] = {}
            self.time_table[x]["count"] = 0
            self.time_table[x]["time"] = 0
        self.time_table[x]["count"] += 1
        self.time_table[x]["time"] += time

    def print_time(self):
        for item in self.time_table:
            if self.time_table[item]["count"] == 0:
                continue
            self.time_table[item]["avg"] = self.time_table[item]["time"] / self.time_table[item]["count"]
        print(json.dumps(self.time_table, indent=4))

    def getTest(self):
        token_nums = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        expertNum = [1, 2]
        t_i = int((self.test / len(expertNum)) % len(token_nums))
        e_i = int(self.test % len(expertNum))
        self.test += 1
        return token_nums[t_i], expertNum[e_i]

EXPERTS_MAP = {
    "KExpertsCPU": KExpertsCPU,
    "KExpertsTorch": KExpertsTorch,
    "KExpertsMarlin": KExpertsMarlin,
}

#finish
class KTransformersExperts(BaseInjectedModule, KExpertsBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
    #  device: str = "cuda",
        prefill_device: str = "cuda",
        prefill_op: str | None = "KExpertsTorch",
        generate_device: str = "cpu",
        generate_op: str | None = "KExpertsCPU",
        **kwargs
    ):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        KExpertsBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        if generate_op is not None:
            self.generate_experts = EXPERTS_MAP[generate_op](key, gguf_loader, config, len(orig_module), device=generate_device, **kwargs)
        else:
            self.generate_experts = None
        if prefill_op is not None:
            self.prefill_experts = EXPERTS_MAP[prefill_op](key, gguf_loader, config, len(orig_module), device=prefill_device, **kwargs)
        else:
            self.prefill_experts = None
        self.gpu_mlp_type = prefill_op
        self.cpu_mlp_type = generate_op
        self.mode = InferenceState.UNLOAD

    def load(self, w: dict = None,  mode: InferenceState = None, warmup: bool = True):
        # TODO support w as input
        if not mode: mode = InferenceState.GENERATE
        if mode == InferenceState.GENERATE:
            self.prefill_experts.unload()
            self.generate_experts.load(w, warmup=warmup)
            self.device = self.generate_experts.device
            self.mode = mode
        elif mode == InferenceState.PREFILL:
            self.generate_experts.unload()
            self.prefill_experts.load(w, warmup=warmup)
            self.device = self.prefill_experts.device
            self.mode = mode
        elif mode == InferenceState.UNLOAD:
            self.unload()
            self.mode = mode
            self.device = self.generate_experts.device
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")

    def unload(self):
        if self.generate_experts is not None:
            self.generate_experts.unload()
        if self.prefill_experts is not None:
            self.prefill_experts.unload()
        self.device = self.generate_experts.device

    def forward(self, input_tensor, expert_ids, weights, shared_experts):
        if self.mode == InferenceState.GENERATE:
            assert self.generate_experts is not None, "generate_experts is None"
            return self.generate_experts.forward(input_tensor, expert_ids, weights, shared_experts)
        elif self.mode == InferenceState.PREFILL:
            assert self.prefill_experts is not None, "prefill_experts is None"
            return self.prefill_experts.forward(input_tensor, expert_ids, weights, shared_experts)
        else:
            raise ValueError("load or set_inference_mode before forward")

    def set_inference_mode(self, mode: InferenceState):
        if mode == InferenceState.GENERATE:
            self.load(mode=InferenceState.GENERATE, warmup=False)
        elif mode == InferenceState.PREFILL:
            self.load(mode=InferenceState.PREFILL, warmup=False)
        elif mode == InferenceState.UNLOAD:
            self.unload()
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")


from ktransformers.models.modeling_deepseek import DeepseekV2MoE
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3MoE
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from ktransformers.models.modeling_mixtral import MixtralSparseMoeBlock


class KQwen2MoeSparseMoeBlock(BaseInjectedModule, Qwen2MoeSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode"):
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], selected_experts[0], routing_weights[0])
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += shared_expert_output
            y.resize_(*orig_shape)
            return y, router_logits
        
        hidden_states_expert = hidden_states.to(self.experts.device)  if isinstance(self.experts, KExpertsBase) else hidden_states_expert.cpu()
        selected_experts_expert = selected_experts.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else selected_experts_expert.cpu()
        routing_weights_expert = routing_weights.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else routing_weights_expert.cpu()

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )

        if isinstance(self.experts, KExpertsBase):
            y = (
                self.moe_kexperts(
                    hidden_states_expert, selected_experts_expert, routing_weights_expert
                )
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states_expert.size(0) > 10:
            y = self.moe_infer(
                hidden_states_expert, selected_experts_expert, routing_weights_expert, orig_shape
            ).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(
                hidden_states_expert, selected_experts_expert, routing_weights_expert
            ).to(device=hidden_states.device)
        y += shared_expert_output
        y.resize_(*orig_shape)
        return y, router_logits
    
    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        '''
        hidden_states_cpu: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        '''
        outs = torch.zeros_like(hidden_states_cpu)
        for token_idx in range(selected_experts_cpu.size(0)):
            for expert_idx in range(selected_experts_cpu.size(1)):
                expert = self.experts[selected_experts_cpu[token_idx, expert_idx]]
                outs[token_idx] += expert.forward(hidden_states_cpu[token_idx]) * routing_weights_cpu[token_idx, expert_idx]
        return outs
    
    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        
        batch_size, sequence_length, hidden_dim = orig_shape

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states_cpu.dtype, device=hidden_states_cpu.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer.forward(current_state) * routing_weights_cpu[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_cpu.dtype))

        return final_hidden_states

#finish
class KDeepseekV2MoE(BaseInjectedModule, DeepseekV2MoE):
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_current_stream_capturing():
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0])
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y
        
        def shared_experts():
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
                return y_
            return None
        
        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity).squeeze(0)
            
        if isinstance(self.experts, KExpertsBase):
            y = self.moe_kexperts(hidden_states, topk_idx, topk_weight, shared_experts).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
                y += y_
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
                y += y_
        return y

    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, shared_experts=None) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight, shared_experts)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KDeepseekV3MoE(BaseInjectedModule, DeepseekV3MoE):
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        # only for generate phase
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_current_stream_capturing():
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0])
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y

        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity).squeeze(0)
            
        if isinstance(self.experts, KExpertsBase):
            y = self.moe_kexperts(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KMistralSparseMoEBlock(BaseInjectedModule, MixtralSparseMoeBlock):
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode"):
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], selected_experts[0], routing_weights[0])
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y.resize_(*orig_shape)
            return y, router_logits
        
        hidden_states_expert = hidden_states.to(self.experts.device)  if isinstance(self.experts, KExpertsBase) else hidden_states_expert.cpu()
        selected_experts_expert = selected_experts.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else selected_experts_expert.cpu()
        routing_weights_expert = routing_weights.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else routing_weights_expert.cpu()

        if isinstance(self.experts, KExpertsBase):
            y = (
                self.moe_kexperts(
                    hidden_states_expert, selected_experts_expert, routing_weights_expert
                )
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states_expert.size(0) > 10:
            y = self.moe_infer(
                hidden_states_expert, selected_experts_expert, routing_weights_expert, orig_shape
            ).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(
                hidden_states_expert, selected_experts_expert, routing_weights_expert
            ).to(device=hidden_states.device)
            
        y.resize_(*orig_shape)
        return y, router_logits
    
    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        '''
        hidden_states_cpu: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        '''
        outs = torch.zeros_like(hidden_states_cpu)
        for token_idx in range(selected_experts_cpu.size(0)):
            for expert_idx in range(selected_experts_cpu.size(1)):
                expert = self.experts[selected_experts_cpu[token_idx, expert_idx]]
                outs[token_idx] += expert.forward(hidden_states_cpu[token_idx]) * routing_weights_cpu[token_idx, expert_idx]
        return outs
    
    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        
        batch_size, sequence_length, hidden_dim = orig_shape

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states_cpu.dtype, device=hidden_states_cpu.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer.forward(current_state) * routing_weights_cpu[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_cpu.dtype))

        return final_hidden_states