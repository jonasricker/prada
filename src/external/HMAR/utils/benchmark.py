# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

# Adapted from https://github.com/HazyResearch/flash-fft-conv/blob/main/benchmarks/benchmark.py

import time
import torch
import torch.utils.benchmark as benchmark

def benchmark_runtime(func, repeats=10):
    torch.cuda.synchronize()
    # Warm-up run
    for _ in range(10):
        func()
    torch.cuda.synchronize()

    
    run_times = []

    for _ in range(repeats):
        torch.cuda.synchronize()
        start_time = time.time()
        func()
        torch.cuda.synchronize()
        end_time = time.time()
        run_times.append(end_time - start_time)

    average_time = sum(run_times) / repeats

    return average_time

def benchmark_memory_usage(func):
    #warm-up run
    func()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    func()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (2**30)

def benchmark_forward(fn, *inputs, repeats = 10, desc='', verbose=True, amp=False,
                      amp_dtype=torch.float16, **kwinputs):
    """ Use Pytorch Benchmark on the forward pass of an arbitrary function. """
    if verbose:
        print(desc, '- Forward pass')
    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)
    t = benchmark.Timer(
            stmt='fn_amp(*inputs, **kwinputs)',
            globals={'fn_amp': amp_wrapper, 'inputs': inputs, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_backward(fn, *inputs, grad=None, repeats=10, desc='', verbose=True, amp=False,
                       amp_dtype=torch.float16, **kwinputs):
    """ Use Pytorch Benchmark on the backward pass of an arbitrary function. """
    if verbose:
        print(desc, '- Backward pass')
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError('Grad shape does not match output shape')
    t = benchmark.Timer(
            stmt='y.backward(grad, retain_graph=True)',
            globals={'y': y, 'grad': grad},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_combined(fn, *inputs, grad=None, repeats=10, desc='', verbose=True, amp=False,
                       amp_dtype=torch.float16, **kwinputs):
    """ Use Pytorch Benchmark on the forward+backward pass of an arbitrary function. """
    if verbose:
        print(desc, '- Forward + Backward pass')
    # y = fn(*inputs, **kwinputs)
    # if grad is None:
    #     grad = torch.randn_like(y)
    # else:
    #     if grad.shape != y.shape:
    #         raise RuntimeError('Grad shape does not match output shape')
    # del y
    def f(grad, *inputs, **kwinputs):
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            y = fn(*inputs, **kwinputs)
            if type(y) is tuple:
                y = y[0]
        if grad is None:
            grad = torch.randn_like(y)
        else:
            if grad.shape != y.shape:
                raise RuntimeError('Grad shape does not match output shape')
        y.backward(grad, retain_graph=True)
    t = benchmark.Timer(
            stmt='f(grad, *inputs, **kwinputs)',
            globals={'f': f, 'fn': fn, 'inputs': inputs, 'grad': grad, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_all(fn, *inputs, grad=None, repeats=10, desc='', verbose=True, amp=False,
                  amp_dtype=torch.float16, **kwinputs):
    """ Use Pytorch Benchmark on the forward+backward pass of an arbitrary function. """
    return (
        benchmark_forward(fn, *inputs, repeats=repeats, desc=desc, verbose=verbose,
                          amp=amp, amp_dtype=amp_dtype, **kwinputs),
        benchmark_backward(fn, *inputs, grad=grad, repeats=repeats, desc=desc, verbose=verbose,
                           amp=amp, amp_dtype=amp_dtype, **kwinputs),
        benchmark_combined(fn, *inputs, grad=grad, repeats=repeats, desc=desc, verbose=verbose,
                           amp=amp, amp_dtype=amp_dtype, **kwinputs),
    )


def pytorch_profiler(fn, *inputs, trace_filename=None, backward=False, amp=False,
                     amp_dtype=torch.float16, cpu=False, verbose=True, **kwinputs):
    """ Wrap benchmark functions in Pytorch profiler to see CUDA information. """
    if backward:
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            g = torch.randn_like(fn(*inputs, **kwinputs))
    for _ in range(30):   # Warm up
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            if backward:
                for x in inputs:
                    if isinstance(x, torch.Tensor):
                        x.grad = None
            # fn(*inputs, **kwinputs) if not backward else fn(*inputs, **kwinputs).backward(g)
            out = fn(*inputs, **kwinputs)
        # Backward should be done outside autocast
        if backward:
            out.backward(g)
    activities = ([torch.profiler.ProfilerActivity.CPU] if cpu else []) + [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        # profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            if backward:
                for x in inputs:
                    if isinstance(x, torch.Tensor):
                        x.grad = None
            out = fn(*inputs, **kwinputs)
        if backward: out.backward(g)
    if verbose:
        # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
        print(prof.key_averages().table(row_limit=50))
    if trace_filename is not None:
        prof.export_chrome_trace(trace_filename)