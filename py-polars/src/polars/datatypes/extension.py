from __future__ import annotations

import contextlib

from polars import datatypes as dt
from polars._utils.unstable import unstable

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import _register_extension_type, _unregister_extension_type

_REGISTRY: dict[str, str | type[dt.BaseExtension]] = {}


@unstable()
def register_extension_type(
    ext_name: str,
    ext_class: type[dt.BaseExtension] | None = None,
    *,
    as_storage: bool = False,
) -> None:
    """
    Register the extension type for the given extension name.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.
    """
    if "ext_name" in _REGISTRY:
        msg = f"extension type '{ext_name}' is already registered"
        raise ValueError(msg)

    if as_storage:
        assert ext_class is None, "cannot specify ext_class when as_storage is True"
        _REGISTRY[ext_name] = "storage"
        with contextlib.suppress(NameError):  # _plr module may be unavailable
            _register_extension_type(ext_name, None)
    else:
        assert not as_storage, "as_storage must be False when ext_class is provided"
        assert isinstance(ext_class, type)
        assert issubclass(ext_class, dt.BaseExtension)
        _REGISTRY[ext_name] = ext_class
        with contextlib.suppress(NameError):  # _plr module may be unavailable
            _register_extension_type(ext_name, ext_class)


@unstable()
def unregister_extension_type(ext_name: str) -> None:
    """
    Unregister the extension type for the given extension name.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.
    """
    _REGISTRY.pop(ext_name)
    _unregister_extension_type(ext_name)


@unstable()
def get_extension_type(ext_name: str) -> type[dt.BaseExtension] | str | None:
    """
    Get the extension type class for the given extension name.

    If an extension is registered to be passed through as storage, this returns
    the string "storage".

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.
    """
    return _REGISTRY.get(ext_name)
