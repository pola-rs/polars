from __future__ import annotations

import abc
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Literal, Union

import polars._utils.logging
from polars._utils.logging import eprint, verbose
from polars._utils.unstable import issue_unstable_warning
from polars.io.cloud._utils import NoPickleOption, ZeroHashWrap
from polars.io.cloud.credential_provider._providers import (
    CredentialProvider,
    CredentialProviderAWS,
    CredentialProviderAzure,
    CredentialProviderFunction,
    CredentialProviderGCP,
    UserProvidedGCPToken,
)

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

# https://docs.rs/object_store/latest/object_store/enum.ClientConfigKey.html
OBJECT_STORE_CLIENT_OPTIONS: frozenset[str] = frozenset(
    [
        "allow_http",
        "allow_invalid_certificates",
        "connect_timeout",
        "default_content_type",
        "http1_only",
        "http2_only",
        "http2_keep_alive_interval",
        "http2_keep_alive_timeout",
        "http2_keep_alive_while_idle",
        "http2_max_frame_size",
        "pool_idle_timeout",
        "pool_max_idle_per_host",
        "proxy_url",
        "proxy_ca_certificate",
        "proxy_excludes",
        "timeout",
        "user_agent",
    ]
)

CredentialProviderBuilderReturn: TypeAlias = Union[
    CredentialProvider, CredentialProviderFunction, None
]


class CredentialProviderBuilder:
    """
    Builds credential providers.

    This is used to defer credential provider initialization to happen at
    `collect()` rather than immediately during query construction. This makes
    the behavior predictable when queries are sent to another environment for
    execution.
    """

    def __init__(
        self,
        credential_provider_init: CredentialProviderBuilderImpl,
    ) -> None:
        """
        Initialize configuration for building a credential provider.

        Parameters
        ----------
        credential_provider_init
            Initializer function that returns a credential provider.
        """
        self.credential_provider_init = credential_provider_init

    # Note: The rust-side expects this exact function name.
    def build_credential_provider(
        self,
    ) -> CredentialProviderBuilderReturn:
        """Instantiate a credential provider from configuration."""
        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(
                "[CredentialProviderBuilder]: Begin initialize "
                f"{self.credential_provider_init!r}"
            )

        v = self.credential_provider_init()

        if verbose:
            if v is not None:
                eprint(
                    f"[CredentialProviderBuilder]: Initialized {v!r} "
                    f"from {self.credential_provider_init!r}"
                )
            else:
                eprint(
                    f"[CredentialProviderBuilder]: No provider initialized "
                    f"from {self.credential_provider_init!r}"
                )

        return v

    @classmethod
    def from_initialized_provider(
        cls, credential_provider: CredentialProviderFunction
    ) -> CredentialProviderBuilder:
        """Initialize with an already constructed provider."""
        return cls(InitializedCredentialProvider(credential_provider))

    def __getstate__(self) -> Any:
        state = self.credential_provider_init

        if verbose():
            eprint(f"[CredentialProviderBuilder]: __getstate__(): {state = !r} ")

        return state

    def __setstate__(self, state: Any) -> None:
        self.credential_provider_init = state

        if verbose():
            eprint(f"[CredentialProviderBuilder]: __setstate__(): {self = !r}")

    def __repr__(self) -> str:
        return f"CredentialProviderBuilder({self.credential_provider_init!r})"


class CredentialProviderBuilderImpl(abc.ABC):
    @abc.abstractmethod
    def __call__(self) -> CredentialProviderFunction | None:
        pass

    @property
    @abc.abstractmethod
    def provider_repr(self) -> str:
        """Used for logging."""

    def __repr__(self) -> str:
        provider_repr = self.provider_repr
        builder_name = type(self).__name__

        return f"{provider_repr} @ {builder_name}"


# Wraps an already ininitialized credential provider into the builder interface.
# Used for e.g. user-provided credential providers.
class InitializedCredentialProvider(CredentialProviderBuilderImpl):
    """Wraps an already initialized credential provider."""

    def __init__(self, credential_provider: CredentialProviderFunction | None) -> None:
        self.credential_provider = credential_provider

    def __call__(self) -> CredentialProviderFunction | None:
        return self.credential_provider

    @property
    def provider_repr(self) -> str:
        return repr(self.credential_provider)


AUTO_INIT_LRU_CACHE: (
    Callable[
        [bytes, ZeroHashWrap[Callable[[], CredentialProviderBuilderReturn]]],
        CredentialProviderBuilderReturn,
    ]
    | None
) = None


def _auto_init_with_cache(
    get_cache_key_func: Callable[[], bytes],
    build_provider_func: ZeroHashWrap[Callable[[], CredentialProviderBuilderReturn]],
) -> CredentialProviderBuilderReturn:
    global AUTO_INIT_LRU_CACHE

    if (
        maxsize := int(os.getenv("POLARS_CREDENTIAL_PROVIDER_BUILDER_CACHE_SIZE", 8))
    ) <= 0:
        AUTO_INIT_LRU_CACHE = None

        return build_provider_func.get()()

    if AUTO_INIT_LRU_CACHE is None:
        if verbose():
            eprint(f"Create credential provider AutoInit LRU cache ({maxsize = })")

        @lru_cache(maxsize=maxsize)
        def cache(
            _cache_key: bytes,
            build_provider_func: ZeroHashWrap[
                Callable[[], CredentialProviderBuilderReturn]
            ],
        ) -> CredentialProviderBuilderReturn:
            return build_provider_func.get()()

        AUTO_INIT_LRU_CACHE = cache

    return AUTO_INIT_LRU_CACHE(
        get_cache_key_func(),
        build_provider_func,
    )


# Represents an automatic initialization configuration. This is created for
# credential_provider="auto".
class AutoInit(CredentialProviderBuilderImpl):
    def __init__(self, cls: Any, **kw: Any) -> None:
        self.cls = cls
        self.kw = kw
        self._cache_key: NoPickleOption[bytes] = NoPickleOption()

    def __call__(self) -> CredentialProviderFunction | None:
        # This is used for credential_provider="auto", which allows for
        # ImportErrors.
        try:
            return _auto_init_with_cache(
                self.get_or_init_cache_key,
                ZeroHashWrap(lambda: self.cls(**self.kw)),
            )
        except ImportError as e:
            if verbose():
                eprint(f"failed to auto-initialize {self.provider_repr}: {e!r}")

        return None

    def get_or_init_cache_key(self) -> bytes:
        cache_key = self._cache_key.get()

        if cache_key is None:
            import hashlib
            import pickle

            hash = hashlib.sha256(pickle.dumps(self))
            self._cache_key.set(hash.digest())
            cache_key = self._cache_key.get()
            assert isinstance(cache_key, bytes)

            if verbose():
                eprint(f"{self!r}: AutoInit cache key: {hash.hexdigest()}")

        return cache_key

    @property
    def provider_repr(self) -> str:
        return self.cls.__name__


def _init_credential_provider_builder(
    credential_provider: CredentialProviderFunction
    | CredentialProviderBuilder
    | Literal["auto"]
    | None,
    source: Any,
    storage_options: dict[str, Any] | None,
    caller_name: str,
) -> CredentialProviderBuilder | None:
    def f() -> CredentialProviderBuilder | None:
        # Note: The behavior of this function should depend only on the function
        # parameters. Any environment-specific behavior should take place inside
        # instantiated credential providers.

        from polars.io.cloud._utils import (
            _first_scan_path,
            _get_path_scheme,
            _is_aws_cloud,
            _is_azure_cloud,
            _is_gcp_cloud,
        )

        if credential_provider is None:
            return None

        if isinstance(credential_provider, CredentialProviderBuilder):
            # This happens when the catalog client auto-inits and passes it to
            # scan/write_delta, which calls us again.
            return credential_provider

        if credential_provider != "auto":
            msg = f"the `credential_provider` parameter of `{caller_name}` is considered unstable."
            issue_unstable_warning(msg)

            return CredentialProviderBuilder.from_initialized_provider(
                credential_provider
            )

        if (path := _first_scan_path(source)) is None:
            return None

        if (scheme := _get_path_scheme(path)) is None:
            return None

        if _is_azure_cloud(scheme):
            tenant_id = None
            storage_account = None

            if storage_options is not None:
                for k, v in storage_options.items():
                    k = k.lower()

                    # https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html
                    if k in {
                        "azure_storage_tenant_id",
                        "azure_storage_authority_id",
                        "azure_tenant_id",
                        "azure_authority_id",
                        "tenant_id",
                        "authority_id",
                    }:
                        tenant_id = v
                    elif k in {"azure_storage_account_name", "account_name"}:
                        storage_account = v
                    elif k in {"azure_use_azure_cli", "use_azure_cli"}:
                        continue
                    elif k in OBJECT_STORE_CLIENT_OPTIONS:
                        continue
                    else:
                        # We assume some sort of access key was given, so we
                        # just dispatch to the rust side.
                        return None

            storage_account = (
                # Prefer the one embedded in the path
                CredentialProviderAzure._extract_adls_uri_storage_account(str(path))
                or storage_account
            )

            return CredentialProviderBuilder(
                AutoInit(
                    CredentialProviderAzure,
                    tenant_id=tenant_id,
                    _storage_account=storage_account,
                )
            )

        elif _is_aws_cloud(scheme):
            region = None
            profile = None
            default_region = None
            unhandled_key = None
            has_endpoint_url = False

            if storage_options is not None:
                for k, v in storage_options.items():
                    k = k.lower()

                    # https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html
                    if k in {"aws_region", "region"}:
                        region = v
                    elif k in {"aws_default_region", "default_region"}:
                        default_region = v
                    elif k in {"aws_profile", "profile"}:
                        profile = v
                    elif k in {
                        "aws_endpoint",
                        "aws_endpoint_url",
                        "endpoint",
                        "endpoint_url",
                    }:
                        has_endpoint_url = True
                    elif k in OBJECT_STORE_CLIENT_OPTIONS:
                        continue
                    else:
                        # We assume this is some sort of access key
                        unhandled_key = k

            if unhandled_key is not None:
                if profile is not None:
                    msg = (
                        "unsupported: cannot combine aws_profile with "
                        f"{unhandled_key} in storage_options"
                    )
                    raise ValueError(msg)

            return CredentialProviderBuilder(
                AutoInit(
                    CredentialProviderAWS,
                    profile_name=profile,
                    region_name=region or default_region,
                    _auto_init_unhandled_key=unhandled_key,
                    _storage_options_has_endpoint_url=has_endpoint_url,
                )
            )

        elif _is_gcp_cloud(scheme):
            token = None
            unhandled_key = None

            if storage_options is not None:
                for k, v in storage_options.items():
                    k = k.lower()

                    # https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html
                    if k in {"token", "bearer_token"}:
                        token = v
                    elif k in {
                        "google_bucket",
                        "google_bucket_name",
                        "bucket",
                        "bucket_name",
                    }:
                        continue
                    elif k in OBJECT_STORE_CLIENT_OPTIONS:
                        continue
                    else:
                        # We assume some sort of access key was given, so we
                        # just dispatch to the rust side.
                        unhandled_key = k

            if unhandled_key is not None:
                if token is not None:
                    msg = (
                        "unsupported: cannot combine token with "
                        f"{unhandled_key} in storage_options"
                    )
                    raise ValueError(msg)

                return None

            if token is not None:
                return CredentialProviderBuilder(
                    InitializedCredentialProvider(UserProvidedGCPToken(token))
                )

            return CredentialProviderBuilder(AutoInit(CredentialProviderGCP))

        return None

    credential_provider_init = f()

    if verbose():
        eprint(f"_init_credential_provider_builder(): {credential_provider_init = !r}")

    return credential_provider_init
