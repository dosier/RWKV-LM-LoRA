import math
import random

import unittest
from parameterized import parameterized_class

import numpy
import torch
import torch.utils.cpp_extension

from torch import nn

from enum import Enum
from lora_linear import make_linear_att
from wkv_forward_cpu import rwkv_linear_attention_cpu
from wkv_backward_cpu import rwkv_backward_cpu

class FloatMode(Enum):
    bf16 = 1
    fp16 = 2
    fp32 = 3

    def is_16_bit(self):
        return self == FloatMode.bf16 or self == FloatMode.fp16

    def convert(self, tensor):
        if self == FloatMode.bf16:
            return tensor.bfloat16()
        elif self == FloatMode.fp16:
            return tensor.half()
        else:
            return tensor.float()


@parameterized_class(('float_mode', 'dim_att', 'layer_id', 'n_layer', 'n_embed', 'ctx_len'), [
    (FloatMode.fp16, 2, 1, 12, 768, 1024),
    (FloatMode.fp16, 2, 1, 12, 768, 1024),
])
class TestWKV(unittest.TestCase):

    def prepare(self, device):

        torch.set_default_device("cpu")

        random.seed(69)
        numpy.random.seed(69)
        torch.manual_seed(69)

        if self.float_mode == FloatMode.bf16 and device == "mps":
            raise NotImplementedError("MPS does not support bf16")

        time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        decay_speed = torch.ones(self.dim_att)
        time_decay = nn.Parameter(decay_speed)
        zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(self.dim_att)]) * 0.5
        time_first = nn.Parameter(torch.ones(self.dim_att) * math.log(0.3) + zigzag)

        with torch.no_grad():
            ratio_0_to_1 = self.layer_id / (self.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, self.n_embed)
            for i in range(self.n_embed):
                ddd[0, 0, i] = i / self.n_embed
            time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        key = make_linear_att(self.n_embed, self.dim_att, bias=False)
        value = make_linear_att(self.n_embed, self.dim_att, bias=False)

        x = torch.randn(1, 1, 1)
        xx = time_shift(x)  # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)

        w = time_decay
        u = time_first
        k = key(xk)
        v = value(xv)

        if self.float_mode.is_16_bit():
            w = w.float()
            u = u.float()
            k = k.float()
            v = v.float()

        w = -torch.exp(w)

        T_MAX = int(self.ctx_len)

        B = 1
        T = 1
        C = self.dim_att

        assert T <= T_MAX
        assert B * C % min(C, 32) == 0

        if device != "cpu":
            w = w.contiguous()
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()

            if self.float_mode == FloatMode.bf16:
                y = torch.empty(B, T, C,
                                memory_format=torch.contiguous_format,
                                dtype=torch.bfloat16)
            else:
                y = torch.empty(B, T, C,
                                memory_format=torch.contiguous_format)
        else:
            if self.float_mode == FloatMode.bf16:
                y = torch.empty(B, T, C, dtype=torch.bfloat16)
            else:
                y = torch.empty(B, T, C)

        w = w.to(device)
        u = u.to(device)
        k = k.to(device)
        v = v.to(device)
        y = y.to(device)

        print("--------------------------")
        print(device, ":")
        print("w:", w)
        print("u:", u)
        print("k:", k)
        print("v:", v)
        print("y:", y)
        return T_MAX, B, T, C, w, u, k, v, y

    def test_cpu(self):

        T_MAX, B, T, C, w, u, k, v, y = self.prepare(device="cpu")

        print("FORWARD PASS...")
        y = rwkv_linear_attention_cpu(seq_length=T, time_decay=w, time_first=u, key=k, value=v, output=y)[0]
        print("y:", y)
        print("BACKWARD PASS...")
        gy = torch.rand_like(y).contiguous()
        gw = torch.empty((B, C), device=gy.device)
        gu = torch.empty((B, C), device=gy.device)
        gk = torch.empty((B, T, C), device=gy.device)
        gv = torch.empty((B, T, C), device=gy.device)
        rwkv_backward_cpu(T_MAX, B, T, C, w, u, k, v, y, gy, gw, gu, gk, gv)

        print("gy:", gy)
        print("gw:", gw)
        print("gu:", gu)
        print("gk:", gk)
        print("gv:", gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        print("gw_sum:", gw)
        print("gu_sum:", gu)

    @unittest.skipIf(not torch.backends.mps.is_available(), "-----------------\nMPS not available")
    def test_metal(self):

        T_MAX, B, T, C, w, u, k, v, y = self.prepare("mps")

        compiled_lib = torch.utils.cpp_extension.load(
            name='CustomWKV',
            sources=['metal/wkv_metal.mm'],
            extra_cflags=['-std=c++17'],
        )

        print("FORWARD PASS...")
        compiled_lib.mps_forward_kernel(B, T, C, w, u, k, v, y)
        print("y:", y)
        print("BACKWARD PASS...")
        gy = torch.rand_like(y).contiguous()
        gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
        gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
        gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
        gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
        compiled_lib.mps_backward_kernel(T_MAX, B, T, C, w, u, k, v, y, gy, gw, gu, gk, gv)
        print("gy:", gy)
        print("gw:", gw)
        print("gu:", gu)
        print("gk:", gk)
        print("gv:", gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        print("gw_sum:", gw)
        print("gu_sum:", gu)

    @unittest.skipIf(not torch.cuda.is_available(), "-----------------\nCUDA not available")
    def test_cuda(self):

        T_MAX, B, T, C, w, u, k, v, y = self.prepare("cuda")

        wkv_cuda = torch.utils.cpp_extension.load(
            name=
            f"wkv_{T_MAX}_bf16" if self.float_mode == FloatMode.bf16 else
            f"wkv_{T_MAX}",
            sources=
            ["cuda/wkv_op_bf16.cpp", "cuda/wkv_cuda_bf16.cu"] if self.float_mode == FloatMode.bf16 else
            ["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
            verbose=True,
            extra_cuda_cflags=
            (["-t 4", "-std=c++17"] if self.float_mode == FloatMode.bf16 else []) +
            [
                "-res-usage",
                "--maxrregcount 60",
                "--use_fast_math", "-O3",
                "-Xptxas -O3",
                "--extra-device-vectorization",
                f"-DTmax={T_MAX}"
            ]
        )

        print("FORWARD PASS...")
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        print("y:", y)
        print("BACKWARD PASS...")
        gy = torch.rand_like(y).contiguous()
        gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
        gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
        gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
        gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
        wkv_cuda.backward(B, T, C, w, u, k, v, y, gy, gw, gu, gk, gv)
        print("gy:", gy)
        print("gw:", gw)
        print("gu:", gu)
        print("gk:", gk)
        print("gv:", gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        print("gw_sum:", gw)
        print("gu_sum:", gu)

