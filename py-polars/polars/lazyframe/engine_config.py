from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rmm.mr import DeviceMemoryResource  # type: ignore[import-not-found]


class GPUEngine:
    """
    Configuration options for the GPU execution engine.

    Use this if you want control over details of the execution.

    Supported options

    - `device`: Select the device to run the query on.
    - `memory_resource`: Set an RMM memory resource for
      device-side allocations.
    """

    device: int | None
    """Device on which to run query."""
    memory_resource: DeviceMemoryResource | None
    """Memory resource to use for device allocations."""
    config: Mapping[str, Any]
    """Additional configuration options for the engine."""

    def __init__(
        self,
        *,
        device: int | None = None,
        memory_resource: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.device = device
        self.memory_resource = memory_resource
        self.config = kwargs
