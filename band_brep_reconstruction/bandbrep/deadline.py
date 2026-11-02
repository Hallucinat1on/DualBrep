"""One cooperative wall-clock budget per sample, including nested field calls.

Native calls may delay Python checkpoints; the batch runner also supervises each
sample in a separate process to enforce the same deadline externally.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import math
import time
from .schemas import ReconstructionError

_current = ContextVar('bandbrep_sample_deadline', default=None)


@contextmanager
def sample_budget(seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('sample timeout must be finite and positive')
    started = time.monotonic()
    parent = _current.get()
    budget = (started, started + seconds, float(seconds))
    if parent is not None and parent[1] < budget[1]:budget = parent
    token = _current.set(budget)
    try:yield
    finally:_current.reset(token)


def checkpoint(stage=None):
    budget = _current.get()
    if budget is None:return
    now = time.monotonic()
    if now >= budget[1]:
        raise ReconstructionError('sample_timeout', 'Sample processing time limit exceeded',
            dict(timeout_seconds=budget[2],elapsed_seconds=now-budget[0],stage=stage))


def timed_reconstruction(fn):
    @wraps(fn)
    def wrapped(mesh,udf,cfg,output_dir,resume=False):
        with sample_budget(cfg.sample_timeout_seconds):
            return fn(mesh,udf,cfg,output_dir,resume=resume)
    return wrapped
