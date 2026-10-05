"""
The decorator of sdim's numba kernels.

Every cached kernel in the package is declared with `_kernel`, so all of them behave the same
when numba has nowhere to write its cache and when its JIT is switched off.
"""

import functools
import threading

import numpy as np
from numba import njit
from numba.core.dispatcher import Dispatcher

# Whether a kernel without the JIT is running on this thread, with overflow warnings off.
_quiet = threading.local()


def _kernel(func=None, **options):
    """
    `njit(cache=True, **options)`, written `@_kernel` or `@_kernel(nogil=True)`.  The kernel
    compiles on its first call.

    numba picks the cache directory when the decorator runs (NUMBA_CACHE_DIR, the package's
    __pycache__ or the user-wide cache directory) and raises RuntimeError if none of them is
    writable, for example for a read-only install with a read-only home directory.  The package
    must still import then, so the kernel goes without the cache and compiles again in each
    process.  Any other error is raised.

    With the JIT switched off (NUMBA_DISABLE_JIT=1) the kernel runs as plain Python on NumPy
    scalars.  Its uint64 arithmetic (the random number generators) wraps around as in compiled
    code, but NumPy warns about every such overflow, so it runs with overflow warnings off.
    """
    if func is None:
        return functools.partial(_kernel, **options)
    try:
        kernel = njit(cache=True, **options)(func)
    except RuntimeError as error:
        if "no locator available" not in str(error):
            raise
        kernel = njit(**options)(func)
    if isinstance(kernel, Dispatcher):
        return kernel

    @functools.wraps(func)
    def without_overflow_warnings(*args, **kwargs):
        # Kernels call each other for every random draw, and np.errstate costs more than most of
        # them, so only the outermost call sets it.
        if getattr(_quiet, "on", False):
            return func(*args, **kwargs)
        _quiet.on = True
        try:
            with np.errstate(over="ignore"):
                return func(*args, **kwargs)
        finally:
            _quiet.on = False

    return without_overflow_warnings
