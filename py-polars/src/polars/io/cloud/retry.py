from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class BackoffConfig:
    """
    Configuration for exponential backoff with jitter.

    All durations are in seconds.
    """

    init_backoff: float = 0.1
    max_backoff: float = 15.0
    base: float = 2.0


@dataclass(kw_only=True)
class RetryConfig:
    """
    Retry policy for cloud requests.

    All durations are in seconds.
    """

    max_retries: int = 2
    retry_timeout: float = 10.0
    backoff: BackoffConfig = field(default_factory=BackoffConfig)
