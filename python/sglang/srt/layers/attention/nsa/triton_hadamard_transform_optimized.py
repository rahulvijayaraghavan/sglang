import torch
import triton
import triton.language as tl


@triton.jit
def _hadamard_kernel_generic(
    input_ptr,
    output_ptr,
    stride_row,
    scale,
    n: tl.constexpr,
    LOG2N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    row_in = input_ptr + pid * stride_row
    row_out = output_ptr + pid * stride_row

    x = tl.load(row_in + offs, mask=mask, other=0.0).to(tl.float32)

    for k in tl.static_range(LOG2N):
        h = 1 << k
        partner_offs = offs ^ h
        tl.store(row_out + offs, x, mask=mask)
        partner = tl.load(row_out + partner_offs, mask=mask, other=0.0)
        is_top = (offs & h) == 0
        x = tl.where(is_top, x + partner, partner - x)

    x = x * scale
    tl.store(row_out + offs, x, mask=mask)


def _triton_hadamard_transform_optimized(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Triton Fast Walsh-Hadamard Transform (drop-in replacement).

    Last dim must be a power of two. Same contract as the fused
    hadamard_transform op: operates on last dim, multiplies by scale.
    """
    n = x.size(-1)
    assert n > 0 and (n & (n - 1)) == 0, "last dim must be a power of two"

    leading = x.shape[:-1]
    num_rows = x[..., 0].numel()
    log2n = int(n).bit_length() - 1

    x_flat = x.reshape(num_rows, n).contiguous()
    output = torch.empty_like(x_flat)

    BLOCK_SIZE = triton.next_power_of_2(n)

    grid = (num_rows,)
    _hadamard_kernel_generic[grid](
        x_flat,
        output,
        stride_row=x_flat.stride(0),
        scale=scale,
        n=n,
        LOG2N=log2n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view(*leading, n)
