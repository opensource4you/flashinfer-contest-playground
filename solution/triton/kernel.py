"""
Triton Kernel Template for FlashInfer Competition.

Implement your kernel logic here. The entry point function name should match
the `entry_point` setting in config.toml.

See the track definition for required function signature and semantics.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel():
    """
    Your Triton kernel implementation.

    TODO: Implement your kernel according to the track definition.
    The function signature should match the track requirements.
    """
    pass


# TODO: You should replace this with your implementation
def my_kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    """
    • FP8 block-scale dequantization: float ≈ fp8 * scale
    • DeepSeek-V3 no-aux routing:
        s = sigmoid(logits)
        s_with_bias = s + bias
        group by n_group=8; per group take top-2 sum → pick topk_group=4 groups
        on the kept groups, take global top_k=8 experts
        combine with weights derived from s (without bias), normalized and
        scaled by routed_scaling_factor
    • Local computation:
        only experts in [local_expert_offset, local_expert_offset + E_local) are
        computed on this rank (GEMM1 → SwiGLU → GEMM2), then per-token weighted
        accumulation.

    Reference:
    https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
    """
    # Fixed DeepSeek-V3/R1 geometry
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]

    BLOCK = 128
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    assert H == 7168, "hidden_size must be 7168"
    assert I == 2048, "intermediate_size must be 2048"
    assert E_global == 256, "num_experts must be 256"
    assert E_local == 32, "num_local_experts must be 32"

    # Routing constants
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    # Block counts
    num_hidden_blocks = H // BLOCK  # 56
    num_intermediate_blocks = I // BLOCK  # 16
    num_gemm1_out_blocks = (2 * I) // BLOCK  # 32

    # Shape checks
    assert hidden_states.shape == (T, H)
    assert hidden_states_scale.shape == (num_hidden_blocks, T)
    assert gemm1_weights.shape == (E_local, 2 * I, H)
    assert gemm1_weights_scale.shape == (
        E_local,
        num_gemm1_out_blocks,
        num_hidden_blocks,
    )
    assert gemm2_weights.shape == (E_local, H, I)
    assert gemm2_weights_scale.shape == (
        E_local,
        num_hidden_blocks,
        num_intermediate_blocks,
    )
    assert routing_bias.shape[-1] == E_global

    device = hidden_states.device

    # 1) FP8 block-scale dequantization
    # hidden_states: [T, H], scale: [H/128, T] (transposed layout)
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)  # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()  # [T, H/128]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)  # [T, H/128, 128]
        .reshape(T, H)  # [T, H]
        .contiguous()
    )
    A = A_fp32 * A_scale_expanded  # [T, H] float32

    # W13: [E_local, 2I, H], scale: [E_local, (2I)/128, H/128]
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_expanded = torch.repeat_interleave(S13, BLOCK, dim=1)  # [E, 2I, H/128]
    S13_expanded = torch.repeat_interleave(S13_expanded, BLOCK, dim=2)  # [E, 2I, H]
    W13 = W13_fp32 * S13_expanded  # [E, 2I, H] float32

    # W2: [E_local, H, I], scale: [E_local, H/128, I/128]
    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2_expanded = torch.repeat_interleave(S2, BLOCK, dim=1)  # [E, H, I/128]
    S2_expanded = torch.repeat_interleave(S2_expanded, BLOCK, dim=2)  # [E, H, I]
    W2 = W2_fp32 * S2_expanded  # [E, H, I] float32

    # 2) No-aux routing
    logits = routing_logits.to(torch.float32)  # [T, E_global]
    bias = routing_bias.to(torch.float32).reshape(-1)  # [E_global]

    # Sigmoid
    s = 1.0 / (1.0 + torch.exp(-logits))  # [T, E]
    s_with_bias = s + bias  # [T, E] (broadcast)

    # Grouping
    group_size = E_global // N_GROUP  # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)  # [T, 8, 32]

    # Group scores = sum of top-2 values within each group
    top2_vals, _ = torch.topk(
        s_wb_grouped, k=2, dim=2, largest=True, sorted=False
    )  # [T, 8, 2]
    group_scores = top2_vals.sum(dim=2)  # [T, 8]

    # Select topk_group groups → group mask
    _, group_idx = torch.topk(
        group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False
    )  # [T, 4]
    group_mask = torch.zeros_like(group_scores)  # [T, 8]
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)
    )  # [T, E]

    # Global top-k (within kept groups), based on s_with_bias
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)  # [T, E]
    _, topk_idx = torch.topk(
        scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False
    )  # [T, 8]

    # Combination weights: use s (without bias) for normalization
    M = torch.zeros_like(s)  # [T, E]
    M.scatter_(1, topk_idx, 1.0)  # 0/1 mask
    weights = s * M  # [T, E]
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor  # [T, E]

    # 3) Local expert compute and accumulation
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    local_start = int(local_expert_offset)

    # For each local expert: find selected tokens, run GEMM1→SwiGLU→GEMM2, accumulate by weights
    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        # Tokens that selected this global expert ge in their top-k
        sel_mask_per_token = (topk_idx == ge).any(dim=1)  # [T] bool
        if not sel_mask_per_token.any():
            continue

        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)  # [Tk]
        Tk = token_idx.numel()

        # Gather inputs and weights for this expert
        A_e = A.index_select(0, token_idx)  # [Tk, H]
        W13_e = W13[le]  # [2I, H]
        W2_e = W2[le]  # [H, I]

        # GEMM1: [Tk, H] @ [H, 2I] = [Tk, 2I]
        G1 = A_e.matmul(W13_e.t())  # [Tk, 2I]

        # SwiGLU: split and apply silu(x) = x / (1 + exp(-x))
        X1 = G1[:, :I]  # [Tk, I]
        X2 = G1[:, I:]  # [Tk, I]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))  # [Tk, I]
        C = silu_X2 * X1  # [Tk, I]

        # GEMM2: [Tk, I] @ [I, H] = [Tk, H]
        O = C.matmul(W2_e.t())  # [Tk, H]

        # Accumulate with per-token routing weights for this expert
        w_tok = weights.index_select(0, token_idx)[:, ge]  # [Tk]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))  # [Tk,H] * [Tk,1]

    return output.to(torch.bfloat16)
