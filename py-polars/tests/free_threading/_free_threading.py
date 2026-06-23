from __future__ import annotations

import sys
import sysconfig


def is_free_threaded_python() -> bool:
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def assert_gil_disabled() -> None:
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    assert callable(is_gil_enabled)
    assert not is_gil_enabled()
