"""Compat shim: the system python3.11 is an incomplete pre-release build
(reports itself as 3.11.0rc1) that predates PEP-related sys.set_int_max_str_digits
/ sys.get_int_max_str_digits landing in the final 3.11.0. torch._dynamo's polyfill
loader references sys.get_int_max_str_digits unconditionally at import time and
crashes with AttributeError without it. Stub it with the real CPython 3.11+
default (4300) before torch is imported anywhere.

Import this module first, before torch/ultralytics, in any script that trains
or runs a model.
"""

import sys

if not hasattr(sys, "get_int_max_str_digits"):
    sys._int_max_str_digits = 4300

    def set_int_max_str_digits(maxdigits: int) -> None:
        sys._int_max_str_digits = maxdigits

    def get_int_max_str_digits() -> int:
        return sys._int_max_str_digits

    sys.set_int_max_str_digits = set_int_max_str_digits
    sys.get_int_max_str_digits = get_int_max_str_digits
