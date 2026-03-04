"""Problem definitions — test cases for each coding challenge."""

from __future__ import annotations
from typing import Any

TASKS: dict[str, dict[str, Any]] = {
    "relu": {
        "title": "Implement ReLU",
        "difficulty": "Easy",
        "function_name": "relu",
        "hint": "ReLU(x) = max(0, x). Think about element-wise comparison with zero.",
        "tests": [
            {
                "name": "Basic values",
                "code": """
import torch
x = torch.tensor([-2., -1., 0., 1., 2.])
out = {fn}(x)
expected = torch.tensor([0., 0., 0., 1., 2.])
assert out.shape == expected.shape, f'Shape mismatch: {out.shape} vs {expected.shape}'
assert torch.allclose(out, expected), f'Wrong Answer: {out} vs {expected}'
""",
            },
            {
                "name": "2-D tensor",
                "code": """
import torch
x = torch.randn(4, 8)
out = {fn}(x)
assert out.shape == x.shape, f'Shape mismatch on 2-D input'
assert (out >= 0).all(), 'ReLU output must be non-negative'
assert torch.allclose(out, x.clamp(min=0)), 'Value mismatch on random input'
""",
            },
            {
                "name": "Gradient check",
                "code": """
import torch
x = torch.tensor([-1., 0., 1., 2.], requires_grad=True)
out = {fn}(x)
out.sum().backward()
assert x.grad is not None, 'Gradient not computed'
assert x.grad[0] == 0., f'grad at x=-1 should be 0, got {x.grad[0]}'
assert x.grad[2] == 1., f'grad at x=1 should be 1, got {x.grad[2]}'
assert x.grad[3] == 1., f'grad at x=2 should be 1, got {x.grad[3]}'
assert x.grad[1] in (0., 1.), f'grad at x=0 should be 0 or 1, got {x.grad[1]}'
""",
            },
            {
                "name": "Performance",
                "code": """
import torch, time
big = torch.randn(1024, 1024)
t0 = time.perf_counter()
for _ in range(100):
    {fn}(big)
elapsed = time.perf_counter() - t0
assert elapsed < 5.0, f'Too slow: {elapsed:.2f}s for 100 iterations'
""",
            },
        ],
    },
    "linear": {
        "title": "Simple Linear Layer",
        "difficulty": "Medium",
        "function_name": "SimpleLinear",
        "hint": "y = x @ W^T + b. Initialize weight with Kaiming scaling: randn * (1/sqrt(in_features)).",
        "tests": [
            {
                "name": "Weight & bias shape",
                "code": """
import torch
layer = {fn}(8, 4)
assert layer.weight.shape == (4, 8), f'Weight shape: {layer.weight.shape}'
assert layer.bias.shape == (4,), f'Bias shape: {layer.bias.shape}'
assert layer.weight.requires_grad, 'weight must require grad'
assert layer.bias.requires_grad, 'bias must require grad'
""",
            },
            {
                "name": "Forward pass",
                "code": """
import torch
layer = {fn}(8, 4)
x = torch.randn(2, 8)
y = layer.forward(x)
assert y.shape == (2, 4), f'Output shape: {y.shape}'
expected = x @ layer.weight.T + layer.bias
assert torch.allclose(y, expected, atol=1e-5), 'Forward != x @ W^T + b'
""",
            },
            {
                "name": "Gradient flow",
                "code": """
import torch
layer = {fn}(8, 4)
x = torch.randn(2, 8)
y = layer.forward(x)
y.sum().backward()
assert layer.weight.grad is not None, 'weight.grad is None'
assert layer.bias.grad is not None, 'bias.grad is None'
""",
            },
        ],
    },
    "attention": {
        "title": "Softmax Attention",
        "difficulty": "Hard",
        "function_name": "scaled_dot_product_attention",
        "hint": "scores = Q @ K^T / sqrt(d_k), then softmax(scores, dim=-1) @ V. Use torch.bmm for batched matmul.",
        "tests": [
            {
                "name": "Output shape",
                "code": """
import torch, math
torch.manual_seed(42)
B, S, D = 2, 4, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V)
assert out.shape == (B, S, D), f'Shape mismatch: {out.shape} vs {(B, S, D)}'
""",
            },
            {
                "name": "Numerical correctness",
                "code": """
import torch, math
torch.manual_seed(42)
B, S, D = 2, 4, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)
weights = torch.softmax(scores, dim=-1)
ref = torch.bmm(weights, V)
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch vs reference'
""",
            },
            {
                "name": "Gradient check",
                "code": """
import torch, math
torch.manual_seed(42)
Q = torch.randn(2, 4, 8, requires_grad=True)
K = torch.randn(2, 4, 8, requires_grad=True)
V = torch.randn(2, 4, 8, requires_grad=True)
out = {fn}(Q, K, V)
out.sum().backward()
assert Q.grad is not None, 'Q.grad is None'
assert K.grad is not None, 'K.grad is None'
assert V.grad is not None, 'V.grad is None'
""",
            },
            {
                "name": "Cross-attention (seq_q != seq_k)",
                "code": """
import torch
Q = torch.randn(1, 3, 16)
K = torch.randn(1, 5, 16)
V = torch.randn(1, 5, 32)
out = {fn}(Q, K, V)
assert out.shape == (1, 3, 32), f'Cross-attention shape: {out.shape}'
""",
            },
        ],
    },
    "softmax": {
        "title": "Implement Softmax",
        "difficulty": "Easy",
        "function_name": "my_softmax",
        "hint": "softmax(x)_i = exp(x_i) / sum(exp(x_j)). Subtract max(x) first for numerical stability.",
        "tests": [
            {
                "name": "Basic 1-D",
                "code": """
import torch
x = torch.tensor([1.0, 2.0, 3.0])
out = {fn}(x, dim=-1)
expected = torch.softmax(x, dim=-1)
assert torch.allclose(out, expected, atol=1e-5), f'{out} vs {expected}'
""",
            },
            {
                "name": "2-D along dim=-1",
                "code": """
import torch
x = torch.randn(4, 8)
out = {fn}(x, dim=-1)
expected = torch.softmax(x, dim=-1)
assert out.shape == expected.shape, f'Shape mismatch'
assert torch.allclose(out, expected, atol=1e-5), 'Values differ'
assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5), 'Rows must sum to 1'
""",
            },
            {
                "name": "Numerical stability",
                "code": """
import torch
x = torch.tensor([1000., 1001., 1002.])
out = {fn}(x, dim=-1)
assert not torch.isnan(out).any(), 'NaN in output — not numerically stable'
assert not torch.isinf(out).any(), 'Inf in output — not numerically stable'
expected = torch.softmax(x, dim=-1)
assert torch.allclose(out, expected, atol=1e-5), 'Values differ on large input'
""",
            },
        ],
    },
    "layernorm": {
        "title": "Implement LayerNorm",
        "difficulty": "Medium",
        "function_name": "my_layer_norm",
        "hint": "Normalize over the last dim: (x - mean) / sqrt(var + eps), then scale by gamma and shift by beta.",
        "tests": [
            {
                "name": "Shape and basic behavior",
                "code": """
import torch
x = torch.randn(2, 3, 8)
gamma = torch.ones(8)
beta = torch.zeros(8)
out = {fn}(x, gamma, beta)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch vs F.layer_norm'
""",
            },
            {
                "name": "With learned parameters",
                "code": """
import torch
x = torch.randn(4, 16)
gamma = torch.randn(16)
beta = torch.randn(16)
out = {fn}(x, gamma, beta)
ref = torch.nn.functional.layer_norm(x, [16], gamma, beta)
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch with non-trivial gamma/beta'
""",
            },
            {
                "name": "Gradient flow",
                "code": """
import torch
x = torch.randn(2, 8, requires_grad=True)
gamma = torch.ones(8, requires_grad=True)
beta = torch.zeros(8, requires_grad=True)
out = {fn}(x, gamma, beta)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert gamma.grad is not None, 'gamma.grad is None'
""",
            },
        ],
    },
    "mha": {
        "title": "Multi-Head Attention",
        "difficulty": "Hard",
        "function_name": "MultiHeadAttention",
        "hint": (
            "Use nn.Linear for Q/K/V/O projections. d_k = d_model // num_heads. "
            "Reshape to (B, heads, S, d_k), SDPA per head, concat, output projection."
        ),
        "tests": [
            {
                "name": "Output shape",
                "code": """
import torch
torch.manual_seed(0)
B, S, D, H = 2, 6, 32, 4
mha = {fn}(d_model=D, num_heads=H)
x = torch.randn(B, S, D)
out = mha.forward(x, x, x)
assert out.shape == (B, S, D), f'Shape mismatch: {out.shape} vs {(B, S, D)}'
""",
            },
            {
                "name": "Uses nn.Linear with correct shapes",
                "code": """
import torch, torch.nn as nn
mha = {fn}(d_model=32, num_heads=4)
for name in ['W_q', 'W_k', 'W_v', 'W_o']:
    layer = getattr(mha, name)
    assert isinstance(layer, nn.Linear), f'{name} should be nn.Linear, got {type(layer)}'
    assert layer.weight.shape == (32, 32), f'{name}.weight shape: {layer.weight.shape}'
    assert layer.weight.requires_grad, f'{name}.weight must require grad'
""",
            },
            {
                "name": "Numerical correctness vs reference",
                "code": """
import torch, torch.nn as nn, math
torch.manual_seed(0)
D, H = 16, 2
d_k = D // H
mha = {fn}(d_model=D, num_heads=H)
Q = torch.randn(1, 4, D)
K = torch.randn(1, 4, D)
V = torch.randn(1, 4, D)
out = mha.forward(Q, K, V)
q = mha.W_q(Q).view(1, 4, H, d_k).transpose(1, 2)
k = mha.W_k(K).view(1, 4, H, d_k).transpose(1, 2)
v = mha.W_v(V).view(1, 4, H, d_k).transpose(1, 2)
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
ref = mha.W_o(attn.transpose(1, 2).contiguous().view(1, 4, D))
assert torch.allclose(out, ref, atol=1e-5), 'Output does not match reference'
""",
            },
            {
                "name": "Gradient flow",
                "code": """
import torch
torch.manual_seed(0)
mha = {fn}(d_model=16, num_heads=2)
x = torch.randn(1, 4, 16, requires_grad=True)
out = mha.forward(x, x, x)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert mha.W_q.weight.grad is not None, 'W_q.weight.grad is None'
assert mha.W_o.weight.grad is not None, 'W_o.weight.grad is None'
""",
            },
            {
                "name": "Cross-attention (seq_q != seq_k)",
                "code": """
import torch
mha = {fn}(d_model=32, num_heads=4)
Q = torch.randn(1, 3, 32)
K = torch.randn(1, 7, 32)
V = torch.randn(1, 7, 32)
out = mha.forward(Q, K, V)
assert out.shape == (1, 3, 32), f'Cross-attention shape: {out.shape}'
""",
            },
            {
                "name": "Different heads give different outputs",
                "code": """
import torch
torch.manual_seed(42)
D, H = 16, 4
d_k = D // H
mha = {fn}(d_model=D, num_heads=H)
x = torch.randn(1, 4, D)
q = mha.W_q(x).view(1, 4, H, d_k).transpose(1, 2)
assert not torch.allclose(q[:, 0], q[:, 1], atol=1e-3), 'Heads produce identical queries'
""",
            },
        ],
    },
    "batchnorm": {
        "title": "Implement BatchNorm",
        "difficulty": "Medium",
        "function_name": "my_batch_norm",
        "hint": "Normalize each feature across the batch (dim=0): (x - mean) / sqrt(var + eps) * gamma + beta. Use unbiased=False for variance.",
        "tests": [
            {
                "name": "Basic behavior — zero mean per feature",
                "code": """
import torch
x = torch.randn(8, 4)
gamma = torch.ones(4)
beta = torch.zeros(4)
out = {fn}(x, gamma, beta)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
col_means = out.mean(dim=0)
assert torch.allclose(col_means, torch.zeros(4), atol=1e-5), f'Column means not zero: {col_means}'
""",
            },
            {
                "name": "Numerical correctness",
                "code": """
import torch
x = torch.randn(16, 8)
gamma = torch.randn(8)
beta = torch.randn(8)
out = {fn}(x, gamma, beta)
mean = x.mean(dim=0)
var = x.var(dim=0, unbiased=False)
ref = gamma * (x - mean) / torch.sqrt(var + 1e-5) + beta
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch'
""",
            },
            {
                "name": "Gradient flow",
                "code": """
import torch
x = torch.randn(4, 8, requires_grad=True)
gamma = torch.ones(8, requires_grad=True)
beta = torch.zeros(8, requires_grad=True)
out = {fn}(x, gamma, beta)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert gamma.grad is not None, 'gamma.grad is None'
""",
            },
        ],
    },
    "rmsnorm": {
        "title": "Implement RMSNorm",
        "difficulty": "Medium",
        "function_name": "rms_norm",
        "hint": "RMS(x) = sqrt(mean(x^2) + eps). RMSNorm(x) = x / RMS(x) * weight. Simpler than LayerNorm — no mean subtraction.",
        "tests": [
            {
                "name": "Basic behavior",
                "code": """
import torch
x = torch.randn(2, 8)
weight = torch.ones(8)
out = {fn}(x, weight)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch'
""",
            },
            {
                "name": "With learned weight",
                "code": """
import torch
x = torch.randn(4, 16)
weight = torch.randn(16)
out = {fn}(x, weight)
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch with non-trivial weight'
""",
            },
            {
                "name": "3-D input",
                "code": """
import torch
x = torch.randn(2, 4, 32)
weight = torch.ones(32)
out = {fn}(x, weight)
assert out.shape == x.shape, f'Shape mismatch on 3-D: {out.shape}'
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch on 3-D'
""",
            },
            {
                "name": "Gradient flow",
                "code": """
import torch
x = torch.randn(2, 8, requires_grad=True)
weight = torch.ones(8, requires_grad=True)
out = {fn}(x, weight)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert weight.grad is not None, 'weight.grad is None'
""",
            },
        ],
    },
    "causal_attention": {
        "title": "Causal Self-Attention",
        "difficulty": "Hard",
        "function_name": "causal_attention",
        "hint": "Same as softmax attention but mask future positions with -inf before softmax. torch.triu(..., diagonal=1) gives the upper triangle.",
        "tests": [
            {
                "name": "Output shape",
                "code": """
import torch
out = {fn}(torch.randn(2, 6, 16), torch.randn(2, 6, 16), torch.randn(2, 6, 16))
assert out.shape == (2, 6, 16), f'Shape mismatch: {out.shape}'
""",
            },
            {
                "name": "Future tokens don't affect past",
                "code": """
import torch
torch.manual_seed(0)
B, S, D = 1, 8, 16
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out1 = {fn}(Q, K, V)
K2, V2 = K.clone(), V.clone()
K2[:, 4:] = torch.randn(B, 4, D)
V2[:, 4:] = torch.randn(B, 4, D)
out2 = {fn}(Q, K2, V2)
assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5), 'Changing future K/V affected past outputs'
""",
            },
            {
                "name": "First position only sees itself",
                "code": """
import torch
torch.manual_seed(0)
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = {fn}(Q, K, V)
assert torch.allclose(out[:, 0], V[:, 0], atol=1e-5), 'Position 0 should output V[0]'
""",
            },
            {
                "name": "Gradient flow",
                "code": """
import torch
Q = torch.randn(2, 4, 8, requires_grad=True)
K = torch.randn(2, 4, 8, requires_grad=True)
V = torch.randn(2, 4, 8, requires_grad=True)
out = {fn}(Q, K, V)
out.sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None, 'Missing gradients'
""",
            },
        ],
    },
    "sliding_window": {
        "title": "Sliding Window Attention",
        "difficulty": "Hard",
        "function_name": "sliding_window_attention",
        "hint": "Like softmax attention but position i only attends to positions j where |i-j| <= window_size. Mask the rest with -inf.",
        "tests": [
            {
                "name": "Output shape",
                "code": """
import torch
out = {fn}(torch.randn(2, 8, 16), torch.randn(2, 8, 16), torch.randn(2, 8, 16), window_size=2)
assert out.shape == (2, 8, 16), f'Shape mismatch: {out.shape}'
""",
            },
            {
                "name": "window_size=0 — only sees itself",
                "code": """
import torch
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = {fn}(Q, K, V, window_size=0)
assert torch.allclose(out, V, atol=1e-5), 'window=0: each position should output V[i]'
""",
            },
            {
                "name": "Large window equals full attention",
                "code": """
import torch, math
torch.manual_seed(0)
B, S, D = 2, 6, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out_win = {fn}(Q, K, V, window_size=S)
d_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
ref = torch.bmm(torch.softmax(scores, dim=-1), V)
assert torch.allclose(out_win, ref, atol=1e-5), 'Large window should equal full attention'
""",
            },
            {
                "name": "Distant tokens don't affect output",
                "code": """
import torch
torch.manual_seed(0)
B, S, D = 1, 10, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out1 = {fn}(Q, K, V, window_size=1)
K2, V2 = K.clone(), V.clone()
K2[:, 5:] = torch.randn(B, 5, D)
V2[:, 5:] = torch.randn(B, 5, D)
out2 = {fn}(Q, K2, V2, window_size=1)
assert torch.allclose(out1[:, 0], out2[:, 0], atol=1e-5), 'Distant tokens should not affect output'
""",
            },
            {
                "name": "Gradient flow",
                "code": """
import torch
Q = torch.randn(2, 4, 8, requires_grad=True)
K = torch.randn(2, 4, 8, requires_grad=True)
V = torch.randn(2, 4, 8, requires_grad=True)
{fn}(Q, K, V, window_size=1).sum().backward()
assert Q.grad is not None, 'Q.grad is None'
""",
            },
        ],
    },
    "linear_attention": {
        "title": "Linear Self-Attention",
        "difficulty": "Hard",
        "function_name": "linear_attention",
        "hint": "Feature map: phi(x) = elu(x) + 1. Compute phi(Q) @ (phi(K)^T @ V) instead of softmax(Q @ K^T) @ V. Normalize by phi(Q) @ sum(phi(K)).",
        "tests": [
            {
                "name": "Output shape",
                "code": """
import torch
out = {fn}(torch.randn(2, 8, 16), torch.randn(2, 8, 16), torch.randn(2, 8, 32))
assert out.shape == (2, 8, 32), f'Shape mismatch: {out.shape}'
""",
            },
            {
                "name": "No NaN or Inf",
                "code": """
import torch
torch.manual_seed(0)
out = {fn}(torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.randn(2, 16, 8))
assert not torch.isnan(out).any(), 'NaN in output'
assert not torch.isinf(out).any(), 'Inf in output'
""",
            },
            {
                "name": "Gradient flow",
                "code": """
import torch
Q = torch.randn(1, 4, 8, requires_grad=True)
K = torch.randn(1, 4, 8, requires_grad=True)
V = torch.randn(1, 4, 8, requires_grad=True)
{fn}(Q, K, V).sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None, 'Missing gradients'
""",
            },
            {
                "name": "Runs fast on long sequences (linear complexity)",
                "code": """
import torch, time
torch.manual_seed(0)
Q = torch.randn(1, 2048, 64)
K = torch.randn(1, 2048, 64)
V = torch.randn(1, 2048, 64)
t0 = time.perf_counter()
for _ in range(10):
    {fn}(Q, K, V)
elapsed = time.perf_counter() - t0
assert elapsed < 5.0, f'Too slow: {elapsed:.2f}s — should be O(S*D^2) not O(S^2*D)'
""",
            },
        ],
    },
    "gqa": {
        "title": "Grouped Query Attention",
        "difficulty": "Hard",
        "function_name": "GroupQueryAttention",
        "hint": "Like MHA but fewer KV heads. W_k/W_v project to num_kv_heads * d_k dims. Use repeat_interleave to expand KV heads to match Q heads.",
        "tests": [
            {
                "name": "Output shape",
                "code": """
import torch
torch.manual_seed(0)
gqa = {fn}(d_model=32, num_heads=8, num_kv_heads=2)
out = gqa.forward(torch.randn(2, 6, 32))
assert out.shape == (2, 6, 32), f'Shape mismatch: {out.shape}'
""",
            },
            {
                "name": "nn.Linear with correct shapes",
                "code": """
import torch, torch.nn as nn
gqa = {fn}(d_model=32, num_heads=8, num_kv_heads=2)
d_k = 32 // 8
assert isinstance(gqa.W_q, nn.Linear) and gqa.W_q.weight.shape == (32, 32), f'W_q wrong'
assert isinstance(gqa.W_k, nn.Linear) and gqa.W_k.weight.shape == (2 * d_k, 32), f'W_k shape: {gqa.W_k.weight.shape}'
assert isinstance(gqa.W_v, nn.Linear) and gqa.W_v.weight.shape == (2 * d_k, 32), f'W_v shape: {gqa.W_v.weight.shape}'
assert isinstance(gqa.W_o, nn.Linear), 'W_o should be nn.Linear'
""",
            },
            {
                "name": "Degenerates to MHA when kv_heads == heads",
                "code": """
import torch
torch.manual_seed(42)
gqa = {fn}(d_model=16, num_heads=4, num_kv_heads=4)
out = gqa.forward(torch.randn(1, 4, 16))
assert out.shape == (1, 4, 16)
assert gqa.W_k.weight.shape == (16, 16), 'Full KV when kv_heads == heads'
""",
            },
            {
                "name": "KV heads are shared correctly",
                "code": """
import torch
torch.manual_seed(0)
D, H, KV = 16, 4, 2
d_k = D // H
gqa = {fn}(d_model=D, num_heads=H, num_kv_heads=KV)
x = torch.randn(1, 4, D)
k = gqa.W_k(x).view(1, 4, KV, d_k).transpose(1, 2)
k_exp = k.repeat_interleave(H // KV, dim=1)
assert torch.equal(k_exp[:, 0], k_exp[:, 1]), 'Heads 0,1 should share same K'
assert not torch.equal(k_exp[:, 0], k_exp[:, 2]), 'Different groups need different K'
""",
            },
            {
                "name": "Gradient flow",
                "code": """
import torch
torch.manual_seed(0)
gqa = {fn}(d_model=16, num_heads=4, num_kv_heads=2)
x = torch.randn(1, 4, 16, requires_grad=True)
gqa.forward(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
assert gqa.W_q.weight.grad is not None and gqa.W_k.weight.grad is not None, 'Missing weight gradients'
""",
            },
        ],
    },
    "gpt2_block": {
        "title": "GPT-2 Transformer Block",
        "difficulty": "Hard",
        "function_name": "GPT2Block",
        "hint": "Pre-norm: x = x + attn(ln1(x)), x = x + mlp(ln2(x)). MLP: Linear(d, 4d) -> GELU -> Linear(4d, d). Attention must be causal. Inherit from nn.Module.",
        "tests": [
            {
                "name": "Output shape",
                "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
block = {fn}(d_model=64, num_heads=4)
assert isinstance(block, nn.Module), 'GPT2Block should inherit from nn.Module'
out = block(torch.randn(2, 8, 64))
assert out.shape == (2, 8, 64), f'Shape mismatch: {out.shape}'
""",
            },
            {
                "name": "Has LayerNorm (pre-norm architecture)",
                "code": """
import torch, torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert hasattr(block, 'ln1') and isinstance(block.ln1, nn.LayerNorm), 'Need self.ln1 = nn.LayerNorm'
assert hasattr(block, 'ln2') and isinstance(block.ln2, nn.LayerNorm), 'Need self.ln2 = nn.LayerNorm'
""",
            },
            {
                "name": "MLP has 4x expansion with GELU",
                "code": """
import torch, torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert hasattr(block, 'mlp'), 'Need self.mlp'
linears = [m for m in block.mlp.modules() if isinstance(m, nn.Linear)]
assert len(linears) >= 2, f'MLP needs >= 2 Linear layers, got {len(linears)}'
assert linears[0].weight.shape == (128, 32), f'MLP first layer: {linears[0].weight.shape}, expected (128, 32)'
assert linears[-1].weight.shape == (32, 128), f'MLP last layer: {linears[-1].weight.shape}, expected (32, 128)'
""",
            },
            {
                "name": "Causal masking — future doesn't affect past",
                "code": """
import torch
torch.manual_seed(0)
block = {fn}(d_model=32, num_heads=4)
x = torch.randn(1, 8, 32)
out1 = block(x)
x2 = x.clone()
x2[:, 4:] = torch.randn(1, 4, 32)
out2 = block(x2)
assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5), 'Future tokens affected past — not causal'
""",
            },
            {
                "name": "Gradient flow to all parameters",
                "code": """
import torch
torch.manual_seed(0)
block = {fn}(d_model=32, num_heads=4)
x = torch.randn(1, 4, 32, requires_grad=True)
block(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
n_total = sum(1 for p in block.parameters())
n_grad = sum(1 for p in block.parameters() if p.grad is not None)
assert n_grad == n_total, f'Only {n_grad}/{n_total} params got gradients'
""",
            },
        ],
    },
}

DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}


def get_task(task_id: str) -> dict[str, Any] | None:
    return TASKS.get(task_id)


def list_tasks() -> list[tuple[str, dict[str, Any]]]:
    return sorted(TASKS.items(), key=lambda t: DIFFICULTY_ORDER.get(t[1]["difficulty"], 9))
