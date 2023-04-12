# ===================================================================
# this is a vendored copy of python's "tempfile.TemporaryDirectory"
# that sets 'ignore_cleanup_errors=True' flag on windows by default
# (we can't use this param otherwise, as it requires python >= 3.10)
# ===================================================================
from __future__ import annotations

import os as _os
import shutil as _shutil
import sys as _sys
import warnings as _warnings
import weakref as _weakref
from tempfile import mkdtemp

if _sys.version_info >= (3, 10):
    from tempfile import TemporaryDirectory
else:
    if _sys.version_info > (3, 9):
        import types as _types

        GenericAlias = _types.GenericAlias
    else:
        import typing as _typing

        GenericAlias = _typing._GenericAlias

    class TemporaryDirectory:
        """
        Create and return a temporary directory.

        This has the same behavior as mkdtemp but can be used as a context manager.
        For example:

            with TemporaryDirectory() as tmpdir:
                ...

        Upon exiting the context, the directory and everything contained
        in it are removed.

        """

        def __init__(
            self, suffix=None, prefix=None, dir=None, ignore_cleanup_errors=False
        ) -> None:
            self.name = mkdtemp(suffix, prefix, dir)
            self._ignore_cleanup_errors = ignore_cleanup_errors
            self._finalizer = _weakref.finalize(
                self,
                self._cleanup,
                self.name,
                warn_message=f"Implicitly cleaning up {self!r}",
                ignore_errors=self._ignore_cleanup_errors,
            )

        @classmethod
        def _rmtree(cls, name, ignore_errors=False) -> None:
            def onerror(func, path, exc_info):
                if issubclass(exc_info[0], PermissionError):

                    def resetperms(path):
                        try:  # noqa: SIM105
                            _os.chflags(path, 0)
                        except AttributeError:
                            pass
                        _os.chmod(path, 0o700)

                    try:
                        if path != name:
                            resetperms(_os.path.dirname(path))
                        resetperms(path)

                        try:
                            _os.unlink(path)
                        # PermissionError is raised on FreeBSD for directories
                        except (IsADirectoryError, PermissionError):
                            cls._rmtree(path, ignore_errors=ignore_errors)
                    except FileNotFoundError:
                        pass
                elif issubclass(exc_info[0], FileNotFoundError):
                    pass
                else:
                    if not ignore_errors:
                        raise

            _shutil.rmtree(name, onerror=onerror)

        @classmethod
        def _cleanup(cls, name, warn_message, ignore_errors=False) -> None:
            cls._rmtree(name, ignore_errors=ignore_errors)
            _warnings.warn(warn_message, ResourceWarning, stacklevel=1)

        def __repr__(self) -> str:
            return f"<{self.__class__.__name__} {self.name!r}>"

        def __enter__(self) -> str:
            return self.name

        def __exit__(self, exc, value, tb) -> None:
            self.cleanup()

        def cleanup(self) -> None:
            if self._finalizer.detach() or _os.path.exists(self.name):
                self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)

        __class_getitem__ = classmethod(GenericAlias)


__all__ = ["TemporaryDirectory"]
