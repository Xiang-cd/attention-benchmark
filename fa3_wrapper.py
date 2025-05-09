import torch
import torch.nn.functional as F
from typing import Any, List, Literal, Optional, Tuple, Union

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    FA3_ENABLED = True
except:
    FA3_ENABLED = False

@torch.compiler.disable
def fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    **kwargs: Any,
):
    
    dtype = q.dtype
    assert FA3_ENABLED, "FA3 not available"
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    if tensor_layout == "HND":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    
    o = flash_attn_func_v3(q, k, v, causal=is_causal, softmax_scale=sm_scale)[0]

    if tensor_layout == "HND":
        o = o.transpose(1, 2)

    return o

@torch.compiler.disable
def fa3_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    **kwargs: Any,
):
    
    dtype = q.dtype
    assert FA3_ENABLED, "FA3 not available"
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
    Bq, Hq, Lq, Dq = q.shape
    Bk, Hk, Lk, Dk = k.shape
    Bv, Hv, Lv, Dv = v.shape
    if tensor_layout == "HND":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    q_scale = (q.abs().max().to(torch.float32) / 448.0)
    k_scale = (k.abs().max().to(torch.float32) / 448.0)
    v_scale = (v.abs().max().to(torch.float32) / 448.0)

    q_f8 = (q / q_scale).to(torch.float8_e4m3fn)
    k_f8 = (k / k_scale).to(torch.float8_e4m3fn)
    v_f8 = (v / v_scale).to(torch.float8_e4m3fn)
    q_scale = q_scale.repeat((Bq, Hq))
    k_scale = k_scale.repeat((Bk, Hk))
    v_scale = v_scale.repeat((Bv, Hv))

    o = flash_attn_func_v3(q_f8, k_f8, v_f8, q_descale=q_scale, k_descale=k_scale, v_descale=v_scale, causal=is_causal, softmax_scale=sm_scale)[0].to(dtype)
    
    if tensor_layout == "HND":
        o = o.transpose(1, 2)

    return o



from flash_attn.flash_attn_interface import flash_attn_func
@torch.compiler.disable
def fa2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    **kwargs: Any,
):
    
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    if tensor_layout == "HND":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    
    o = flash_attn_func(q, k, v, causal=is_causal, softmax_scale=sm_scale)[0]

    if tensor_layout == "HND":
        o = o.transpose(1, 2)

    return o