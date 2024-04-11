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
@click.option('--dim-head', default = 16)
@click.option('--heads', default = 2)
@click.option('--rand-key-pad-mask', is_flag = True)
@click.option('--bucket_size', default = 4)
@click.option('--cuda-kernel', is_flag = True)
def test(
    causal: bool,
    seq_len: int,
    dim_head: int,
    heads: int,
    rand_key_pad_mask: bool,
    bucket_size: int,
    cuda_kernel: bool
):
    # base qkv

    q = torch.randn(2, seq_len, heads, dim_head)
    k = torch.randn(2, seq_len, heads, dim_head)
    v = torch.randn(2, seq_len, heads, dim_head)

    # key padding mask

    mask = None
    if rand_key_pad_mask:
        assert not causal
        mask = torch.randint(0, 2, (2, seq_len)).bool()

    # flash and regular qkv's

    fq = q.clone().requires_grad_()
    fk = k.clone().requires_grad_()
    fv = v.clone().requires_grad_()

    rq = q.clone().requires_grad_()
    rk = k.clone().requires_grad_()
    rv = v.clone().requires_grad_()

    if cuda_kernel:
        assert torch.cuda.is_available()

        fcq = q.clone().cuda().requires_grad_()
        fck = k.clone().cuda().requires_grad_()
        fcv = v.clone().cuda().requires_grad_()

    # forward

    o = default_attention(rq, rk, rv, causal = causal, mask = mask)
    fo = ring_flash_attn(fq, fk, fv, bucket_size = bucket_size, causal = causal, mask = mask)

    assert torch.allclose(o, fo, atol = 1e-6)

    if cuda_kernel:
        from ring_attention_pytorch.ring_flash_attention_cuda import ring_flash_attn_cuda

        if mask is not None:
            mask = mask.cuda()

        fco = ring_flash_attn_cuda(fcq, fck, fcv, mask, causal)
        fco.sum().backward()

        assert torch.allclose(o, fco.cpu(), atol = 1e-2)

    # backwards

    o.sum().backward()
    fo.sum().backward()

    assert torch.allclose(rq.grad, fq.grad, atol = 1e-6)
    assert torch.allclose(rk.grad, fk.grad, atol = 1e-6)
    assert torch.allclose(rv.grad, fv.grad, atol = 1e-6)

    if cuda_kernel:
        assert torch.allclose(rq.grad, fcq.grad.cpu(), atol = 1e-2)
        assert torch.allclose(rk.grad, fck.grad.cpu(), atol = 1e-2)
        assert torch.allclose(rv.grad, fcv.grad.cpu(), atol = 1e-2)

    print('✅ outputs and gradients are same between regular attention and naive flash attention')

if __name__ == '__main__':
    test()
