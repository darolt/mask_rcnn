"""
This module is used for profiling CPU and GPU times.

Licensed under The MIT License
Written by Jean Da Rolt
"""
import os
from timeit import default_timer as timer

import torch


def profilable(func):
    """To be used as a decorator in functions that should be
    time-profiled.
    """
    if 'TIME_PROF' not in os.environ:
        return func

    def wrapper(*args, **kwargs):  # pylint: disable=C0111
        # before
        start_time = timer()
        start_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

        ret = func(*args, **kwargs)
        torch.cuda.synchronize()

        # after
        end_time = timer()
        end_evt = torch.cuda.Event(enable_timing=True)
        end_evt.record()
        torch.cuda.synchronize()
        print(f"{func.__name__}: Timeit: {end_time-start_time}, "
              f"CUDA Event time: {start_evt.elapsed_time(end_evt)}")

        return ret
    return wrapper
