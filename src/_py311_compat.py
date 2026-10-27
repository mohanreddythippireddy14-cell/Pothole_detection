"""Compat shim for broken pre-release Python 3.11 builds (e.g. 3.11.0rc1).

Some pre-release 3.11 builds lack sys.get_int_max_str_digits /
sys.set_int_max_str_digits, which torch._dynamo references at import time. On such a
build this stubs both with CPython's real default (4300). On any normal Python
(3.10, or 3.11.0+) the functions already exist and this module does nothing.

Import it before torch/ultralytics in scripts that train or run a model.
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
