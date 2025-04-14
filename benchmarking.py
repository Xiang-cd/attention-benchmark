from tqdm import tqdm
from torch.nn.functional import scaled_dot_product_attention as sdpa
import torch
import itertools
import math
from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined
from fa3_wrapper import fa3, fa3_fp8, fa2
from sageattention import sageattn_qk_int8_pv_fp16_cuda, sageattn_qk_int8_pv_fp8_cuda_sm90
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', help='output file')
args = parser.parse_args()

head_dim_ls = [64, 128]
num_head_ls = [32]
batch_size_ls = [1, 4]
seq_len_ls = [1024*2**i for i in range(2, 9)]
device = 'cuda'
device_name = torch.cuda.get_device_name(device)
sm = torch.cuda.get_device_capability()
sm = 10*sm[0] + sm[1]
if sm >= 89:
    method_ls = [fa2, fa3, fa3_fp8, sdpa, sageattn_qk_int8_pv_fp16_cuda, sageattn_qk_int8_pv_fp8_cuda_sm90]
else:
    method_ls = [fa2, sdpa, sageattn_qk_int8_pv_fp16_cuda]
    

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0



is_causal = False
print(f"is_causal: {is_causal}")
mode = 'fwd'
res_ls = []

for headdim, head, batch, seq_len, method in tqdm(list(itertools.product(head_dim_ls, num_head_ls, batch_size_ls, seq_len_ls, method_ls))):
    q = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device=device)
    k = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device=device)
    v = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device=device)
    try:
        if mode == 'fwd':
            _, time = benchmark_forward(method, q, k, v, is_causal=is_causal, repeats=3, verbose=False, desc='forward')
            torch.cuda.synchronize()
            _, time = benchmark_forward(method, q, k, v, is_causal=is_causal, repeats=6, verbose=False, desc='forward')
        elif mode == 'bwd':
            _, time = benchmark_backward(method, q, k, v, is_causal=is_causal, repeats=3, verbose=False, desc='backward')
            torch.cuda.synchronize()
            _, time = benchmark_backward(method, q, k, v, is_causal=is_causal, repeats=6, verbose=False, desc='backward')
        elif mode == 'fwd_bwd':
            _, time = benchmark_combined(method, q, k, v, is_causal=is_causal, repeats=3, verbose=False, desc='combine')
            torch.cuda.synchronize()
            _, time = benchmark_combined(method, q, k, v, is_causal=is_causal, repeats=6, verbose=False, desc='combine')
        
        res_ls.append({
            'algorithm': method.__name__,
            'batch_size': batch,
            'headdim': headdim,
            'numhead': head,
            'seq_len': seq_len,
            'time': time.mean,
            'flops': efficiency(flops(batch, seq_len, headdim, head, is_causal, mode), time.mean),
            'mode': mode
        })
    except RuntimeError as e:
        print(method.__name__)
        print(e)
        continue

with open(args.output, 'w') as f:
    res_dict = {
        'device_name': device_name,
        'res_ls': res_ls
    }
    json.dump(res_dict, f, indent=4)
    

# is_causal = True
# print(f"is_causal: {is_causal}")
# for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
#     flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
#     q = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
#     k = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
#     v = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
#     for i in range(5): sdpa(q, k, v, is_causal=is_causal)
#     torch.cuda.synchronize()
#     _, time = benchmark_forward(sdpa, q, k, v, is_causal=is_causal, repeats=100, verbose=False, desc='Triton')
#     print(f'{seq_len} flops:{flops/time.mean*1e-12}')