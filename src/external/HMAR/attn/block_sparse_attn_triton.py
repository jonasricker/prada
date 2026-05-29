# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

# Adapted from: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
# Extended to:
# 1) Support arbitary sequence lengths for both the fwd and the bwd pass
# 2) Support block diagonal attention (used in HMAR) for fwd and bwd pass
# 3) Support block causal attention (used in VAR) for fwd and bwd pass
# 4) Leverage the sparsity in block diagonal & block causal attention masks to speedup the computation

import os

import torch
import triton
import triton.language as tl

RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

AUTO_TUNING = os.environ.get("TRITON_AUTO_TUNING", "0") == "1"

# sparsity pattern for block sparse attention
BLOCK_DIAGONAL: tl.constexpr = 0
BLOCK_CAUSAL: tl.constexpr = 1

sparsity_patterns = {"block_diagonal": 0, "block_causal": 1}


@triton.jit
def next_multiple(number, base):
    return ((number + base - 1) // base) * base


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    qk_scale,
    rs_i: tl.constexpr,
    re_i: tl.constexpr,
    lo: tl.constexpr,
    hi: tl.constexpr,
    MASK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    offs_n: tl.constexpr,
):
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q, k)

        if MASK:
            mask = rs_i[:, None] <= (start_n + offs_n[None, :]) and re_i[:, None] > (
                start_n + offs_n[None, :]
            )

            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        else:
            qk = qk * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # -- update output accumulator --
        acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        p = p.to(tl.float16)

        acc = tl.dot(p, v, acc)

        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i


configs_fwd = (
    [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
        for BM in [32, 64, 128]
        for BN in [32, 64]
        for s in [3, 4]
        for w in [4, 8]
    ]
    if AUTO_TUNING
    else [triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=4, num_warps=4)]
)


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs_fwd)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    row_starts,
    row_ends,
    stride_rs,
    stride_re,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qzh_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    kzh_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    vzh_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    ozh_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qzh_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + vzh_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + kzh_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + ozh_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # load scales
    qk_scale = sm_scale
    qk_scale *= RCP_LN2

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # mask for loading row starts and ends
    mask = offs_m < N_CTX

    # load the row starts and row ends within this block for the block causal,
    # the row starts are all zeros and we could potentially skips this step and use
    # zeros directly, but really does not affect performance in a meaningful way.
    # Also cleaner to use the same logic for both block diagonal and block causal without branching.
    rs_i = tl.load(row_starts + offs_m * stride_rs, mask=mask, other=N_CTX)
    re_i = tl.load(row_ends + offs_m * stride_re, mask=mask, other=0)

    # min/max row starts and ends tell us what should be masked or not for each block
    rs_i_min = tl.min(rs_i)
    rs_i_max = tl.max(rs_i)
    re_i_min = tl.min(re_i)
    re_i_max = tl.max(re_i)

    # For block diagonal attention, if we have a single block diagonal assigned to the current
    # triton block, we can compute certain parts without masking. We can also do the same for
    # block causal attention.
    if rs_i_min == rs_i_max:
        lo = rs_i_min.to(tl.int32)
        hi = (lo + ((re_i_min - lo) // BLOCK_N) * BLOCK_N).to(tl.int32)

        # non masked blocks
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            qk_scale,  #
            rs_i,
            re_i,
            lo,
            hi,
            False,
            BLOCK_N,  #
            offs_n,
        )

        # After computing the non-masked blocks, update the start and end locations
        lo = hi.to(tl.int32)
        hi = re_i_max.to(tl.int32)
    else:

        # For block diagonal attention if we have different block diagonals assigned to the current triton
        # block, then we need to do all the computations using masks.
        lo = rs_i_min.to(tl.int32)
        hi = re_i_max.to(tl.int32)

    # masked blocks
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,
        qk_scale,
        rs_i,
        re_i,
        lo,
        hi,
        True,
        BLOCK_N,
        offs_n,
    )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)

    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))


configs_pre = (
    [
        triton.Config({"PRE_BLOCK": BM}, num_stages=s, num_warps=w)
        for BM in [32, 64, 128]
        for s in [3, 4]
        for w in [4, 8]
    ]
    if AUTO_TUNING
    else [triton.Config({"PRE_BLOCK": 128}, num_stages=4, num_warps=4)]
)


@triton.autotune(configs_pre, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,
    o_stride_z,
    o_stride_h,
    o_stride_m,
    o_stride_k,
    do_stride_z,
    do_stride_h,
    do_stride_m,
    do_stride_k,
    Delta,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    PRE_BLOCK: tl.constexpr,
):
    off_m = tl.program_id(0) * PRE_BLOCK + tl.arange(0, PRE_BLOCK)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H

    off_n = tl.arange(0, HEAD_DIM)

    mask = off_m[:, None] < N_CTX

    # compute ptrs
    o_ptr = (
        O
        + off_z * o_stride_z
        + off_h * o_stride_h
        + off_m[:, None] * o_stride_m
        + off_n[None, :] * o_stride_k
    )
    do_ptr = (
        DO
        + off_z * do_stride_z
        + off_h * do_stride_h
        + off_m[:, None] * do_stride_m
        + off_n[None, :] * do_stride_k
    )

    # load
    o = tl.load(o_ptr, mask=mask).to(tl.float32)
    do = tl.load(do_ptr, mask=mask).to(tl.float32)
    delta = tl.sum(o * do, axis=1)

    # write-back
    out_mask = off_m < N_CTX
    tl.store(Delta + off_hz * N_CTX + off_m, delta, mask=out_mask)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,
    Q,
    k,
    v,
    DO,
    M,
    D,
    stride_tokq,
    stride_dq,
    stride_tokdo,
    stride_ddo,
    rs_i,
    re_i,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_m,
    num_steps,
    MASK: tl.constexpr,
):

    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_k = tl.arange(0, HEAD_DIM)

    step_m = BLOCK_M1

    for _ in range(num_steps):
        q_ptrs = Q + offs_m[:, None] * stride_tokq + offs_k[None, :] * stride_dq
        do_ptrs = DO + offs_m[:, None] * stride_tokdo + offs_k[None, :] * stride_ddo

        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        # Load m and do before computing qk to reduce pipeline stall.
        m = tl.load(M + offs_m, mask=offs_m < N_CTX, other=0.0)
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        qkT = tl.dot(k, q.trans())
        pT = tl.math.exp2(qkT - m[None, :])

        if MASK:
            mask = rs_i[:, None] <= (offs_m[None, :]) and re_i[:, None] > (
                offs_m[None, :]
            )
            pT = tl.where(mask, pT, 0.0)

        # Compute dV.
        ppT = pT
        dv += tl.dot(ppT.to(tl.float16), do.to(tl.float16))

        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m, mask=offs_m < N_CTX, other=0.0)
        dpT = tl.dot(v.to(tl.float16), tl.trans(do).to(tl.float16)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])

        dk += tl.dot(dsT.to(tl.float16), q.to(tl.float16))

        # Increment pointers.
        offs_m += step_m
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,
    do,
    m,
    D,
    stride_tokk,
    stride_dk,
    stride_tokv,
    stride_dv,
    rs_i,
    re_i,
    N_CTX,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_m,
    start_n,
    num_steps,
    MASK: tl.constexpr,
):

    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)

    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m, mask=offs_m < N_CTX, other=0.0)

    step_n = BLOCK_N2

    for _ in range(num_steps):
        kT_ptrs = K + offs_n[None, :] * stride_tokk + offs_k[:, None] * stride_dk
        vT_ptrs = V + offs_n[None, :] * stride_tokv + offs_k[:, None] * stride_dv

        kT = tl.load(kT_ptrs, mask=offs_n[None, :] < N_CTX, other=0.0)
        vT = tl.load(vT_ptrs, mask=offs_n[None, :] < N_CTX, other=0.0)

        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)

        if MASK:
            mask = rs_i[:, None] <= (offs_n[None, :]) and re_i[:, None] > (
                offs_n[None, :]
            )
            p = tl.where(mask, p, 0.0)

        # Compute dP and dS.
        dp = tl.dot(do.to(tl.float16), vT.to(tl.float16)).to(tl.float32)
        ds = p * (dp - Di[:, None])

        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds.to(tl.float16), tl.trans(kT).to(tl.float16))

        # Increment pointers.
        offs_n += step_n
    return dq


configs_bwd = (
    [
        triton.Config(
            {"BLOCK_M1": BM_1, "BLOCK_M2": BM_2, "BLOCK_N1": BN_1, "BLOCK_N2": BN_2},
            num_stages=s,
            num_warps=w,
        )
        for BM_1 in [16, 32]
        for BM_2 in [64, 128]
        for BN_1 in [32, 64]
        for BN_2 in [16, 32]
        for s in [3, 4]
        for w in [4, 8]
    ]
    if AUTO_TUNING
    else [
        triton.Config(
            {"BLOCK_M1": 16, "BLOCK_M2": 128, "BLOCK_N1": 16, "BLOCK_N2": 32},
            num_stages=4,
            num_warps=4,
        )
    ]
)


@triton.autotune(configs_bwd, key=["N_CTX", "HEAD_DIM", "H"])
@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    M,
    D,
    stride_zq,
    stride_hq,
    stride_tokq,
    stride_dq,
    stride_zk,
    stride_hk,
    stride_tokk,
    stride_dk,
    stride_zv,
    stride_hv,
    stride_tokv,
    stride_dv,
    stride_zdq,
    stride_hdq,
    stride_tokdq,
    stride_ddq,
    stride_zdk,
    stride_hdk,
    stride_tokdk,
    stride_ddk,
    stride_zdv,
    stride_hdv,
    stride_tokdv,
    stride_ddv,
    stride_zdo,
    stride_hdo,
    stride_tokdo,
    stride_ddo,
    r_s,
    r_e,
    c_s,
    sparsity_pattern,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
):

    bhid = tl.program_id(1)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj_q = (stride_hq * (bhid % H) + stride_zq * (bhid // H)).to(tl.int64)
    adj_k = (stride_hk * (bhid % H) + stride_zk * (bhid // H)).to(tl.int64)
    adj_v = (stride_hv * (bhid % H) + stride_zv * (bhid // H)).to(tl.int64)
    adj_do = (stride_hdo * (bhid % H) + stride_zdo * (bhid // H)).to(tl.int64)
    adj_dq = (stride_hdq * (bhid % H) + stride_zdq * (bhid // H)).to(tl.int64)
    adj_dk = (stride_hdk * (bhid % H) + stride_zdk * (bhid // H)).to(tl.int64)
    adj_dv = (stride_hdv * (bhid % H) + stride_zdv * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj_q
    K += adj_k
    V += adj_v
    DO += adj_do
    DQ += adj_dq
    DK += adj_dk
    DV += adj_dv
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1

    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(
        K + offs_n[:, None] * stride_tokk + offs_k[None, :] * stride_dk,
        mask=offs_n[:, None] < N_CTX,
        other=0.0,
    )
    v = tl.load(
        V + offs_n[:, None] * stride_tokv + offs_k[None, :] * stride_dv,
        mask=offs_n[:, None] < N_CTX,
        other=0.0,
    )

    if sparsity_pattern == BLOCK_DIAGONAL:
        # For block diagonal, p and p.T have the same patterns so
        # we can still use the row_starts and row_ends to know what to mask.
        rs_i = tl.load(r_s + offs_n, mask=offs_n < N_CTX, other=N_CTX)
        re_i = tl.load(r_e + offs_n, mask=offs_n < N_CTX, other=0)
    else:
        # For block causal, p and p.T have different patterns, so we need to
        # load the cummulative sums, which helps to know what to mask in p.T
        rs_i = tl.load(c_s + offs_n, mask=offs_n < N_CTX, other=N_CTX)
        re_i = tl.full((BLOCK_N1,), N_CTX, dtype=tl.int32)

    rs_i_min = tl.min(rs_i)
    rs_i_max = tl.max(rs_i)
    re_i_min = tl.min(re_i)
    re_i_max = tl.max(re_i)

    if (
        sparsity_pattern == BLOCK_DIAGONAL and rs_i_min == rs_i_max
    ) or sparsity_pattern == BLOCK_CAUSAL:

        if sparsity_pattern == BLOCK_DIAGONAL:
            num_steps = (re_i_min - rs_i_min) // BLOCK_M1
            start_m = rs_i_min
        else:
            num_steps = tl.cdiv((N_CTX - next_multiple(rs_i_max, BLOCK_M1)), BLOCK_M1)
            start_m = next_multiple(rs_i_max, BLOCK_M1)

        # non-masked blocks
        dk, dv = _attn_bwd_dkdv(
            dk,
            dv,  #
            Q,
            k,
            v,  #
            DO,  #
            M,
            D,  #
            stride_tokq,
            stride_dq,  #
            stride_tokdo,
            stride_ddo,
            rs_i,
            re_i,
            N_CTX,  #
            BLOCK_M1,
            HEAD_DIM,  #
            start_m,
            num_steps,
            False,
        )

        if sparsity_pattern == BLOCK_DIAGONAL:
            start_m = num_steps * BLOCK_M1 + rs_i_min
            num_steps = tl.cdiv((re_i_max - start_m), BLOCK_M1)
        else:
            num_steps = tl.cdiv((start_m - (rs_i_min // BLOCK_M1) * BLOCK_M1), BLOCK_M1)
            start_m = (rs_i_min // BLOCK_M1) * BLOCK_M1
    else:
        num_steps = tl.cdiv((re_i_max - rs_i_min), BLOCK_M1)
        start_m = rs_i_min

    # masked blocks

    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,
        Q,
        k,
        v,
        DO,
        M,
        D,
        stride_tokq,
        stride_dq,
        stride_tokdo,
        stride_ddo,
        rs_i,
        re_i,
        N_CTX,
        BLOCK_M1,
        HEAD_DIM,
        start_m,
        num_steps,
        True,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tokdv + offs_k[None, :] * stride_ddv
    tl.store(dv_ptrs, dv, mask=offs_n[:, None] < N_CTX)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tokdk + offs_k[None, :] * stride_ddk
    tl.store(dk_ptrs, dk, mask=offs_n[:, None] < N_CTX)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2

    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(
        Q + offs_m[:, None] * stride_tokq + offs_k[None, :] * stride_dq,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(
        DO + offs_m[:, None] * stride_tokdo + offs_k[None, :] * stride_ddo,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )

    m = tl.load(M + offs_m, mask=offs_m < N_CTX, other=0.0)[:, None]

    rs_i = tl.load(r_s + offs_m, mask=offs_m < N_CTX, other=N_CTX)
    re_i = tl.load(r_e + offs_m, mask=offs_m < N_CTX, other=0)

    rs_i_min = tl.min(rs_i)
    rs_i_max = tl.max(rs_i)
    re_i_min = tl.min(re_i)
    re_i_max = tl.max(re_i)

    if rs_i_min == rs_i_max:
        num_steps = (re_i_min - rs_i_min) // BLOCK_N2
        start_n = rs_i_min.to(tl.int32)

        # unmasked blocks
        dq = _attn_bwd_dq(
            dq,
            q,
            K,
            V,
            do,
            m,
            D,
            stride_tokk,
            stride_dk,
            stride_tokv,
            stride_dv,
            rs_i,
            re_i,
            N_CTX,
            BLOCK_M2,
            BLOCK_N2,
            HEAD_DIM,
            start_m,
            start_n,
            num_steps,
            False,
        )

        start_n = (num_steps * BLOCK_N2 + rs_i_min).to(tl.int32)
        num_steps = tl.cdiv((re_i_max - rs_i_min), BLOCK_N2) - num_steps
    else:
        num_steps = tl.cdiv((re_i_max - rs_i_min), BLOCK_N2)
        start_n = rs_i_min.to(tl.int32)

    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,
        do,
        m,
        D,
        stride_tokk,
        stride_dk,
        stride_tokv,
        stride_dv,
        rs_i,
        re_i,
        N_CTX,
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,
        start_m,
        start_n,
        num_steps,
        True,
    )

    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tokdq + offs_k[None, :] * stride_ddq
    dq *= LN2
    tl.store(dq_ptrs, dq, mask=offs_m[:, None] < N_CTX)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, q, k, v, row_starts, row_ends, cum_sums, sm_scale, sparsity_pattern
    ):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]

        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        assert sparsity_pattern in sparsity_patterns.keys()

        o = torch.empty_like(q)

        grid = lambda args: (
            triton.cdiv(q.shape[2], args["BLOCK_M"]),
            q.shape[0] * q.shape[1],
            1,
        )
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        _attn_fwd[grid](
            q,
            k,
            v,
            sm_scale,
            M,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            row_starts,
            row_ends,
            row_starts.stride(0),
            row_ends.stride(0),
            q.shape[1],
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
        )

        ctx.save_for_backward(q, k, v, o, M, row_starts, row_ends, cum_sums)
        ctx.sparsity_pattern = sparsity_pattern
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M, r_s, r_e, c_s = ctx.saved_tensors

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        pre_grid = lambda args: (triton.cdiv(N_CTX, args["PRE_BLOCK"]), BATCH * N_HEAD)

        delta = torch.empty_like(M)

        _attn_bwd_preprocess[pre_grid](
            o,
            do,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            delta,
            N_HEAD,
            N_CTX,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        grid = lambda args: (triton.cdiv(N_CTX, args["BLOCK_N1"]), BATCH * N_HEAD)

        _attn_bwd[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,
            M,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            r_s,
            r_e,
            c_s,
            sparsity_patterns[ctx.sparsity_pattern],
            N_HEAD,
            N_CTX,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        return dq, dk, dv, None, None, None, None, None


block_sparse_attn_func = _attention.apply
