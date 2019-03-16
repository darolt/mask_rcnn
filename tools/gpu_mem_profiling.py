"""
This module is used for profiling the use of GPU memory.

Licensed under The MIT License
Written by Jean Da Rolt
"""
import datetime
import gc
import logging
import os
import sys

import torch
from py3nvml import py3nvml


PRINT_TENSOR_SIZES = True
# clears GPU cache frequently, showing only actual memory usage
EMPTY_CACHE = True
GPU_PROFILE_FN = (f"{datetime.datetime.now():%d-%b-%y-%H:%M:%S}"
                  f"-gpu_mem_prof.txt")
if 'GPU_DEBUG' in os.environ:
    logging.info(f"profiling gpu usage to {GPU_PROFILE_FN}")


def init_profiler(device, debug_function):
    os.environ['GPU_DEBUG'] = str(device)
    os.environ['TRACE_INTO'] = debug_function
    sys.settrace(trace_calls)


def trace_calls(frame, event, arg):  # pylint: disable=W0613
    """Tracer used by sys. Requires TRACE_INTO environment
    variable to be defined. Will only debug line-by-line inside
    functions defined inside TRACE_INTO."""
    if event != 'call':
        return
    code = frame.f_code
    func_name = code.co_name

    try:
        trace_into = str(os.environ['TRACE_INTO'])
    except:
        logging.info('TRACE_INTO environment variable not defined.')
        print(os.environ)
        exit()
    if func_name in trace_into.split(' '):
        return _trace_lines
    return


def _trace_lines(frame, event, arg):  # pylint: disable=W0613
    if event != 'line':
        return
    if EMPTY_CACHE:
        torch.cuda.empty_cache()
    code = frame.f_code
    func_name = code.co_name
    line_no = frame.f_lineno
    filename = code.co_filename
    py3nvml.nvmlInit()
    mem_used = _get_gpu_mem_used()
    where_str = f"{func_name} in {filename}:{line_no}"
    with open(GPU_PROFILE_FN, 'a+') as file:
        file.write(f"{where_str} --> {mem_used:<7.1f}Mb\n")
        if PRINT_TENSOR_SIZES:
            _print_tensors(file, where_str)

    py3nvml.nvmlShutdown()


def _get_gpu_mem_used():
    handle = py3nvml.nvmlDeviceGetHandleByIndex(
        int(os.environ['GPU_DEBUG']))
    meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.used/1024**2


_last_tensor_sizes = set()  # pylint: disable=C0103


def _print_tensors(file, where_str):
    global _last_tensor_sizes  # pylint: disable=W0603, C0103
    for tensor in _get_tensors():
        if not hasattr(tensor, 'dbg_alloc_where'):
            tensor.dbg_alloc_where = where_str
    new_tensor_sizes = {(x.type(), tuple(x.shape), x.dbg_alloc_where)
                        for x in _get_tensors()}
    for ttype, shape, loc in new_tensor_sizes - _last_tensor_sizes:
        file.write(f"+ {loc:<50} {str(shape):<20} {str(ttype):<10}\n")
    for ttype, shape, loc in _last_tensor_sizes - new_tensor_sizes:
        file.write(f"- {loc:<50} {str(shape):<20} {str(ttype):<10}\n")
    _last_tensor_sizes = new_tensor_sizes


def _get_tensors():
    gc.collect()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except:
            pass
