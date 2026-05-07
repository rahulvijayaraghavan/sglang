import torch
import triton
import triton.language as tl


@triton.jit
def _fused_norm_rope_kernel_mode2(
    input_ptr,
    weight_ptr,
    positions_ptr,
    freqs_cis_ptr,
    N,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    freqs_stride_0,
    input_stride_0,
    eps: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
):
    """Mode 2 (DefaultForward): row[i] = i, position[i] = handle[i]."""
    pid = tl.program_id(0)
    if pid >= N:
        return

    row_idx = pid
    pos_idx = tl.load(positions_ptr + pid).to(tl.int64)

    row_base = input_ptr + row_idx * input_stride_0
    freq_base = freqs_cis_ptr + pos_idx * freqs_stride_0

    offs = tl.arange(0, BLOCK_HD)
    mask = offs < head_dim
    x = tl.load(row_base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm
    var = tl.sum(x * x, axis=0) / head_dim
    x = x * tl.rsqrt(var + eps) * w

    # Store normalized result
    tl.store(row_base + offs, x, mask=mask)

    # RoPE on trailing rope_dim
    rope_start = head_dim - rope_dim
    rope_offs = tl.arange(0, BLOCK_ROPE)
    rope_mask = rope_offs < (rope_dim // 2)

    real_offs = rope_start + rope_offs * 2
    imag_offs = rope_start + rope_offs * 2 + 1

    xr = tl.load(row_base + real_offs, mask=rope_mask, other=0.0).to(tl.float32)
    xi = tl.load(row_base + imag_offs, mask=rope_mask, other=0.0).to(tl.float32)

    fr = tl.load(freq_base + rope_offs * 2, mask=rope_mask, other=0.0).to(tl.float32)
    fi = tl.load(freq_base + rope_offs * 2 + 1, mask=rope_mask, other=0.0).to(tl.float32)

    out_r = xr * fr - xi * fi
    out_i = xr * fi + xi * fr

    tl.store(row_base + real_offs, out_r, mask=rope_mask)
    tl.store(row_base + imag_offs, out_i, mask=rope_mask)


@triton.jit
def _fused_norm_rope_kernel_mode1(
    input_ptr,
    weight_ptr,
    seq_lens_ptr,
    freqs_cis_ptr,
    N,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    compress_ratio: tl.constexpr,
    freqs_stride_0,
    input_stride_0,
    eps: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
):
    """Mode 1 (CompressDecode): validity check inside kernel.
    Row i is valid if seq_lens[i] % compress_ratio == 0.
    Position = seq_lens[i] - compress_ratio.
    """
    pid = tl.program_id(0)
    if pid >= N:
        return

    seq_len = tl.load(seq_lens_ptr + pid).to(tl.int64)

    # Validity check: skip if seq_len % compress_ratio != 0
    if seq_len % compress_ratio != 0:
        return

    row_idx = pid
    pos_idx = seq_len - compress_ratio

    row_base = input_ptr + row_idx * input_stride_0
    freq_base = freqs_cis_ptr + pos_idx * freqs_stride_0

    offs = tl.arange(0, BLOCK_HD)
    mask = offs < head_dim
    x = tl.load(row_base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / head_dim
    x = x * tl.rsqrt(var + eps) * w

    tl.store(row_base + offs, x, mask=mask)

    rope_start = head_dim - rope_dim
    rope_offs = tl.arange(0, BLOCK_ROPE)
    rope_mask = rope_offs < (rope_dim // 2)

    real_offs = rope_start + rope_offs * 2
    imag_offs = rope_start + rope_offs * 2 + 1

    xr = tl.load(row_base + real_offs, mask=rope_mask, other=0.0).to(tl.float32)
    xi = tl.load(row_base + imag_offs, mask=rope_mask, other=0.0).to(tl.float32)

    fr = tl.load(freq_base + rope_offs * 2, mask=rope_mask, other=0.0).to(tl.float32)
    fi = tl.load(freq_base + rope_offs * 2 + 1, mask=rope_mask, other=0.0).to(tl.float32)

    out_r = xr * fr - xi * fi
    out_i = xr * fi + xi * fr

    tl.store(row_base + real_offs, out_r, mask=rope_mask)
    tl.store(row_base + imag_offs, out_i, mask=rope_mask)


@triton.jit
def _fused_norm_rope_kernel_mode0(
    input_ptr,
    weight_ptr,
    plan_ptr,            # [N, 3+] int64 — decoded prefill plan already on device
    freqs_cis_ptr,
    N,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    compress_ratio: tl.constexpr,
    invalid_marker: tl.constexpr,
    plan_stride_0,
    freqs_stride_0,
    input_stride_0,
    eps: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
):
    """Mode 0 (CompressExtend): validity check inside kernel.
    plan[i, 0] == INVALID_PLAN means skip.
    row = plan[i, 0], position = plan[i, 2] + 1 - compress_ratio.
    """
    pid = tl.program_id(0)
    if pid >= N:
        return

    plan_base = plan_ptr + pid * plan_stride_0
    row_idx = tl.load(plan_base + 0).to(tl.int64)

    # Validity: skip if row marker is invalid
    if row_idx == invalid_marker:
        return

    pos_raw = tl.load(plan_base + 2).to(tl.int64)
    pos_idx = pos_raw + 1 - compress_ratio

    row_base = input_ptr + row_idx * input_stride_0
    freq_base = freqs_cis_ptr + pos_idx * freqs_stride_0

    offs = tl.arange(0, BLOCK_HD)
    mask = offs < head_dim
    x = tl.load(row_base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / head_dim
    x = x * tl.rsqrt(var + eps) * w

    tl.store(row_base + offs, x, mask=mask)

    rope_start = head_dim - rope_dim
    rope_offs = tl.arange(0, BLOCK_ROPE)
    rope_mask = rope_offs < (rope_dim // 2)

    real_offs = rope_start + rope_offs * 2
    imag_offs = rope_start + rope_offs * 2 + 1

    xr = tl.load(row_base + real_offs, mask=rope_mask, other=0.0).to(tl.float32)
    xi = tl.load(row_base + imag_offs, mask=rope_mask, other=0.0).to(tl.float32)

    fr = tl.load(freq_base + rope_offs * 2, mask=rope_mask, other=0.0).to(tl.float32)
    fi = tl.load(freq_base + rope_offs * 2 + 1, mask=rope_mask, other=0.0).to(tl.float32)

    out_r = xr * fr - xi * fi
    out_i = xr * fi + xi * fr

    tl.store(row_base + real_offs, out_r, mask=rope_mask)
    tl.store(row_base + imag_offs, out_i, mask=rope_mask)


def _triton_fused_norm_rope(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    handle: torch.Tensor,
    freqs_cis: torch.Tensor,
    mode: int,
    eps: float,
    compress_ratio: int,
) -> None:
    """Triton fused RMSNorm + RoPE in-place.

    No host-blocking ops (no .any(), no nonzero(), no .to(device) copies).
    Validity checks are pushed into the kernel — invalid rows early-return.
    """
    head_dim = input_tensor.shape[-1]
    rope_dim = freqs_cis.shape[-1]

    BLOCK_HD = triton.next_power_of_2(head_dim)
    BLOCK_ROPE = triton.next_power_of_2(rope_dim // 2)

    if mode == 2:
        # DefaultForward: handle = positions [N], row[i] = i
        N = handle.shape[0]
        _fused_norm_rope_kernel_mode2[(N,)](
            input_tensor,
            weight,
            handle,
            freqs_cis,
            N,
            head_dim=head_dim,
            rope_dim=rope_dim,
            freqs_stride_0=freqs_cis.stride(0),
            input_stride_0=input_tensor.stride(0),
            eps=eps,
            BLOCK_HD=BLOCK_HD,
            BLOCK_ROPE=BLOCK_ROPE,
        )
    elif mode == 1:
        # CompressDecode: handle = seq_lens [N], valid if seq_lens[i] % compress_ratio == 0
        N = handle.shape[0]
        _fused_norm_rope_kernel_mode1[(N,)](
            input_tensor,
            weight,
            handle,
            freqs_cis,
            N,
            head_dim=head_dim,
            rope_dim=rope_dim,
            compress_ratio=compress_ratio,
            freqs_stride_0=freqs_cis.stride(0),
            input_stride_0=input_tensor.stride(0),
            eps=eps,
            BLOCK_HD=BLOCK_HD,
            BLOCK_ROPE=BLOCK_ROPE,
        )
    elif mode == 0:
        # CompressExtend: handle = packed PrefillPlan
        from sglang.jit_kernel.deepseek_v4 import _decode_prefill_plan, _INVALID_PLAN
        # _decode_prefill_plan returns a CPU tensor; move to device ONCE
        # and cache on the handle if possible. This is unavoidable but happens
        # only for prefill (not the hot decode path).
        plan = _decode_prefill_plan(handle).to(input_tensor.device)
        N = plan.shape[0]
        _fused_norm_rope_kernel_mode0[(N,)](
            input_tensor,
            weight,
            plan,
            freqs_cis,
            N,
            head_dim=head_dim,
            rope_dim=rope_dim,
            compress_ratio=compress_ratio,
            invalid_marker=_INVALID_PLAN,
            plan_stride_0=plan.stride(0),
            freqs_stride_0=freqs_cis.stride(0),
            input_stride_0=input_tensor.stride(0),
            eps=eps,
            BLOCK_HD=BLOCK_HD,
            BLOCK_ROPE=BLOCK_ROPE,
        )
    else:
        raise ValueError(f"unsupported fused_norm_rope mode: {mode}")
