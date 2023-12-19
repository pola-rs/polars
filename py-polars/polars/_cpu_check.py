# Vendored parts of the code from https://github.com/flababah/cpuid.py,
# so we replicate its copyright license.

# Copyright (c) 2014 Anders Høst
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from __future__ import annotations

import ctypes
import os
import platform
from ctypes import CFUNCTYPE, POINTER, c_long, c_size_t, c_uint32, c_ulong, c_void_p
from typing import ClassVar

"""
Determine whether Polars can be run on the current CPU.

This must be done in pure Python, before the Polars binary is imported. If we
were to try it on the Rust side the compiler could emit illegal instructions
before/during the CPU feature check code.
"""


# Replaced during the build process with our list of required feature flags
# enabled at compile time.
_POLARS_FEATURE_FLAGS = ""

_IS_WINDOWS = os.name == "nt"
_IS_64BIT = ctypes.sizeof(ctypes.c_void_p) == 8
_IS_X86 = platform.machine() in ("AMD64", "x86_64", "x86", "i686")

# Posix x86_64:
# Three first call registers : RDI, RSI, RDX
# Volatile registers         : RAX, RCX, RDX, RSI, RDI, R8-11

# Windows x86_64:
# Three first call registers : RCX, RDX, R8
# Volatile registers         : RAX, RCX, RDX, R8-11

# cdecl 32 bit:
# Three first call registers : Stack (%esp)
# Volatile registers         : EAX, ECX, EDX

# fmt: off
_POSIX_64_OPC = [
        0x53,                    # push   %rbx
        0x89, 0xf0,              # mov    %esi,%eax
        0x89, 0xd1,              # mov    %edx,%ecx
        0x0f, 0xa2,              # cpuid
        0x89, 0x07,              # mov    %eax,(%rdi)
        0x89, 0x5f, 0x04,        # mov    %ebx,0x4(%rdi)
        0x89, 0x4f, 0x08,        # mov    %ecx,0x8(%rdi)
        0x89, 0x57, 0x0c,        # mov    %edx,0xc(%rdi)
        0x5b,                    # pop    %rbx
        0xc3                     # retq
]

_WINDOWS_64_OPC = [
        0x53,                    # push   %rbx
        0x89, 0xd0,              # mov    %edx,%eax
        0x49, 0x89, 0xc9,        # mov    %rcx,%r9
        0x44, 0x89, 0xc1,        # mov    %r8d,%ecx
        0x0f, 0xa2,              # cpuid
        0x41, 0x89, 0x01,        # mov    %eax,(%r9)
        0x41, 0x89, 0x59, 0x04,  # mov    %ebx,0x4(%r9)
        0x41, 0x89, 0x49, 0x08,  # mov    %ecx,0x8(%r9)
        0x41, 0x89, 0x51, 0x0c,  # mov    %edx,0xc(%r9)
        0x5b,                    # pop    %rbx
        0xc3                     # retq
]

_CDECL_32_OPC = [
        0x53,                    # push   %ebx
        0x57,                    # push   %edi
        0x8b, 0x7c, 0x24, 0x0c,  # mov    0xc(%esp),%edi
        0x8b, 0x44, 0x24, 0x10,  # mov    0x10(%esp),%eax
        0x8b, 0x4c, 0x24, 0x14,  # mov    0x14(%esp),%ecx
        0x0f, 0xa2,              # cpuid
        0x89, 0x07,              # mov    %eax,(%edi)
        0x89, 0x5f, 0x04,        # mov    %ebx,0x4(%edi)
        0x89, 0x4f, 0x08,        # mov    %ecx,0x8(%edi)
        0x89, 0x57, 0x0c,        # mov    %edx,0xc(%edi)
        0x5f,                    # pop    %edi
        0x5b,                    # pop    %ebx
        0xc3                     # ret
]
# fmt: on

# From memoryapi.h
_MEM_COMMIT = 0x1000
_MEM_RESERVE = 0x2000
_MEM_RELEASE = 0x8000
_PAGE_EXECUTE_READWRITE = 0x40


class CPUID_struct(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, type]]] = [
        (r, c_uint32) for r in ("eax", "ebx", "ecx", "edx")
    ]


class CPUID:
    def __init__(self) -> None:
        if not _IS_X86:
            raise SystemError("CPUID is only available for x86")

        if _IS_WINDOWS:
            if _IS_64BIT:
                # VirtualAlloc seems to fail under some weird
                # circumstances when ctypes.windll.kernel32 is
                # used under 64 bit Python. CDLL fixes this.
                self.win = ctypes.CDLL("kernel32.dll")
                opc = _WINDOWS_64_OPC
            else:
                # Here ctypes.windll.kernel32 is needed to get the
                # right DLL. Otherwise it will fail when running
                # 32 bit Python on 64 bit Windows.
                self.win = ctypes.windll.kernel32  # type: ignore[attr-defined]
                opc = _CDECL_32_OPC
        else:
            opc = _POSIX_64_OPC if _IS_64BIT else _CDECL_32_OPC

        size = len(opc)
        code = (ctypes.c_ubyte * size)(*opc)

        if _IS_WINDOWS:
            self.win.VirtualAlloc.restype = c_void_p
            self.win.VirtualAlloc.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_ulong,
                ctypes.c_ulong,
            ]
            self.addr = self.win.VirtualAlloc(
                None, size, _MEM_COMMIT | _MEM_RESERVE, _PAGE_EXECUTE_READWRITE
            )
            if not self.addr:
                raise MemoryError("could not allocate memory for CPUID check")
            ctypes.memmove(self.addr, code, size)
        else:
            import mmap  # Only import if necessary.

            self.mmap = mmap.mmap(
                -1,
                size,
                mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
            )
            self.addr = ctypes.addressof(ctypes.c_void_p.from_buffer(self.mmap))
            self.mmap.write(code)

        func_type = CFUNCTYPE(None, POINTER(CPUID_struct), c_uint32, c_uint32)
        self.func_ptr = func_type(self.addr)

    def __call__(self, eax: int, ecx: int = 0) -> CPUID_struct:
        struct = CPUID_struct()
        self.func_ptr(struct, eax, ecx)
        return struct

    def __del__(self) -> None:
        if _IS_WINDOWS:
            self.win.VirtualFree.restype = c_long
            self.win.VirtualFree.argtypes = [c_void_p, c_size_t, c_ulong]
            self.win.VirtualFree(self.addr, 0, _MEM_RELEASE)


def read_cpu_flags() -> dict[str, bool]:
    # Right now we only enable extra feature flags for x86.
    if not _IS_X86:
        return {}

    # CPU flags from https://en.wikipedia.org/wiki/CPUID
    cpuid = CPUID()
    cpuid1 = cpuid(1, 0)
    cpuid7 = cpuid(7, 0)
    cpuid81h = cpuid(0x80000001, 0)

    return {
        "sse3": bool(cpuid1.ecx & (1 << 0)),
        "ssse3": bool(cpuid1.ecx & (1 << 9)),
        "fma": bool(cpuid1.ecx & (1 << 12)),
        "sse4.1": bool(cpuid1.ecx & (1 << 19)),
        "sse4.2": bool(cpuid1.ecx & (1 << 20)),
        "popcnt": bool(cpuid1.ecx & (1 << 23)),
        "avx": bool(cpuid1.ecx & (1 << 28)),
        "bmi1": bool(cpuid7.ebx & (1 << 3)),
        "bmi2": bool(cpuid7.ebx & (1 << 8)),
        "avx2": bool(cpuid7.ebx & (1 << 5)),
        "lzcnt": bool(cpuid81h.ecx & (1 << 5)),
    }


def check_cpu_flags() -> None:
    expected_cpu_flags = [f.lstrip("+") for f in _POLARS_FEATURE_FLAGS.split(",")]
    supported_cpu_flags = read_cpu_flags()

    missing_features = []
    for f in expected_cpu_flags:
        if f not in supported_cpu_flags:
            raise RuntimeError(f"unknown feature flag {f}")

        if not supported_cpu_flags[f]:
            missing_features.append(f)

    if missing_features:
        import warnings  # Only import if necessary.

        warnings.warn(
            f"""Missing required CPU features.

This version of Polars requires the following CPU features your processor appears to be missing:
    {", ".join(missing_features)}
Continuing to use this version of Polars on this processor will likely result in a crash.

If you are on an Apple ARM machine (e.g. M1) this is likely due to running Python under Rosetta. It is recommended to install a native version of Python that does not run under Rosetta x86-64 emulation.

Otherwise, please try the `polars-lts-cpu` package instead of `polars`. This package is compiled with fewer CPU flags to support older / less featureful processors.
""",
            RuntimeWarning,
            stacklevel=1,
        )
