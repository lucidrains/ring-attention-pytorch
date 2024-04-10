import click

import torch

from ring_attention_pytorch import (
    default_attention,
    ring_flash_attn
)

# variables

@click.command()
@click.option('--causal', is_flag = True)
@click.option('--seq-len', default = 62)
@click.option('--bucket_size', default = 4)
@click.option('--flash-cuda-kernel', is_flag = True)
def test(
    causal: bool,
    seq_len: int,
    bucket_size: int,
    flash_cuda_kernel: bool
):
    # base qkv

    q = torch.randn(2, seq_len, 2, 16)
    k = torch.randn(2, seq_len, 2, 16)
    v = torch.randn(2, seq_len, 2, 16)

    # flash and regular qkv's

    fq = q.clone().requires_grad_()
    fk = k.clone().requires_grad_()
    fv = v.clone().requires_grad_()

    rq = q.clone().requires_grad_()
    rk = k.clone().requires_grad_()
    rv = v.clone().requires_grad_()

    if flash_cuda_kernel:
        assert torch.cuda.is_available()

        fcq = q.clone().cuda().requires_grad_()
        fck = k.clone().cuda().requires_grad_()
        fcv = v.clone().cuda().requires_grad_()

    # forward

    o = default_attention(rq, rk, rv, causal = causal)
    fo = ring_flash_attn(fq, fk, fv, bucket_size = bucket_size, causal = causal)

    assert torch.allclose(o, fo, atol = 1e-6)

    if flash_cuda_kernel:
        from ring_attention_pytorch.ring_flash_attention_cuda import ring_flash_attn_cuda

        fco = ring_flash_attn_cuda(fcq, fck, fcv, None, causal)
        fco.sum().backward()

        assert torch.allclose(o, fco.cpu(), atol = 1e-2)

    # backwards

    o.sum().backward()
    fo.sum().backward()

    assert torch.allclose(rq.grad, fq.grad, atol = 1e-6)
    assert torch.allclose(rk.grad, fk.grad, atol = 1e-6)
    assert torch.allclose(rv.grad, fv.grad, atol = 1e-6)

    if flash_cuda_kernel:
        assert torch.allclose(rq.grad, fcq.grad.cpu(), atol = 1e-2)
        assert torch.allclose(rk.grad, fck.grad.cpu(), atol = 1e-2)
        assert torch.allclose(rv.grad, fcv.grad.cpu(), atol = 1e-2)

    print('✅ outputs and gradients are same between regular attention and naive flash attention')

if __name__ == '__main__':
    test()
