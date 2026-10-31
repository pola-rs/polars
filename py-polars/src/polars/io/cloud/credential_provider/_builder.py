from __future__ import annotations

import abc
import os
import threading
from copy import copy
from typing import TYPE_CHECKING, Any, Final, Literal

import polars._utils.logging
from polars._utils.cache import LRUCache
from polars._utils.logging import eprint, verbose
from polars._utils.unstable import issue_unstable_warning
from polars.io.cloud._utils import POLARS_STORAGE_CONFIG_KEYS, NoPickleOption
from polars.io.cloud.credential_provider._providers import (
    CachedCredentialProvider,
    CachingCredentialProvider,
    CredentialProvider,
    CredentialProviderAWS,
    CredentialProviderAzure,
    CredentialProviderFunction,
    CredentialProviderGCP,
    UserProvidedGCPToken,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from polars._typing import StorageOptionsDict


# `storage_options` keys that are ignored when auto-initializing a credential provider.
AUTOINIT_IGNORED_KEYS: Final[frozenset[str]] = frozenset(
    [
        # Object store client options
        # https://docs.rs/object_store/latest/object_store/enum.ClientConfigKey.html
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
        *POLARS_STORAGE_CONFIG_KEYS,
        # Azure
        "azure_use_azure_cli",
        "use_azure_cli",
        # AWS
        "aws_request_payer",
        "request_payer",
        # GCS
        "google_bucket",
        "google_bucket_name",
        "bucket",
        "bucket_name",
    ]
)


CredentialProviderBuilderReturn: TypeAlias = (
    CredentialProvider | CredentialProviderFunction | None
)


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
        clear_cached_credentials: bool = False,  # noqa: FBT001
    ) -> CredentialProviderBuilderReturn:
        """
        Instantiate a credential provider from configuration.

        Parameters
        ----------
        clear_cached_credentials
            If the built provider is an instance of `CachingCredentialProvider`,
            clears any cached credentials on that object.
        """
        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(
                "[CredentialProviderBuilder]: Begin initialize "
                f"{self.credential_provider_init!r} "
                f"{clear_cached_credentials = }"
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

        if clear_cached_credentials and isinstance(v, CachingCredentialProvider):
            v.clear_cached_credentials()

            if verbose:
                eprint(
                    f"[CredentialProviderBuilder]: Clear cached credentials for {v!r}"
                )

        return v

    @classmethod
    def from_initialized_provider(
        cls, credential_provider: CredentialProviderFunction
    ) -> CredentialProviderBuilder:
        """Initialize with an already constructed provider."""
        return cls(InitializedCredentialProvider(credential_provider))

    def stable_cache_key(self) -> bytes:
        return self.credential_provider_init.stable_cache_key()

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

    @abc.abstractmethod
    def stable_cache_key(self) -> bytes:
        """Content-based key that survives pickle round-trips."""

    def __repr__(self) -> str:
        provider_repr = self.provider_repr
        builder_name = type(self).__name__

        return f"{provider_repr} @ {builder_name}"


# Wraps an already initialized credential provider into the builder interface.
# Used for e.g. user-provided credential providers.
class InitializedCredentialProvider(CredentialProviderBuilderImpl):
    """Wraps an already initialized credential provider."""

    def __init__(self, credential_provider: CredentialProviderFunction) -> None:
        self.credential_provider = credential_provider

    def __call__(self) -> CredentialProviderBuilderReturn:
        if isinstance(self.credential_provider, CachingCredentialProvider):
            return self.credential_provider

        # We use the cache by keying the entry as the address of the object
        # provided by the user.
        return _build_with_cache(
            lambda: id(self.credential_provider),
            lambda: CachedCredentialProvider(self.credential_provider),
        )

    def stable_cache_key(self) -> bytes:
        import hashlib
        import pickle

        verbose = polars._utils.logging.verbose()
        try:
            return hashlib.sha256(pickle.dumps(self.credential_provider)).digest()[:16]
        except Exception as e:
            if verbose:
                print(f"CredentialProvider stable_cache_key() failed: {e = }")
            # If we cannot pickle, there is no need for the cache key to be
            # globally stable. Instead, a locally stable cache key is sufficient.
            return id(self.credential_provider).to_bytes(8, byteorder="little")

    @property
    def provider_repr(self) -> str:
        return repr(self.credential_provider)


# The keys of this can be:
# * int: Object address of a user-passed credential provider
# * bytes: Hash of an AutoInit configuration
BUILT_PROVIDERS_LRU_CACHE: (
    LRUCache[int | bytes, CredentialProviderBuilderReturn] | None
) = None
BUILT_PROVIDERS_LRU_CACHE_LOCK: threading.RLock = threading.RLock()


def _build_with_cache(
    get_cache_key_func: Callable[[], int | bytes],
    build_provider_func: Callable[[], CredentialProviderBuilderReturn],
) -> CredentialProviderBuilderReturn:
    global BUILT_PROVIDERS_LRU_CACHE

    if (
        max_items := int(
            os.getenv(
                "POLARS_CREDENTIAL_PROVIDER_BUILDER_CACHE_SIZE",
                8,
            )
        )
    ) <= 0:
        if BUILT_PROVIDERS_LRU_CACHE_LOCK.acquire(blocking=False):
            BUILT_PROVIDERS_LRU_CACHE = None
            BUILT_PROVIDERS_LRU_CACHE_LOCK.release()

        return build_provider_func()

    verbose = polars._utils.logging.verbose()

    with BUILT_PROVIDERS_LRU_CACHE_LOCK:
        if BUILT_PROVIDERS_LRU_CACHE is None:
            if verbose:
                eprint(f"Create built credential providers LRU cache ({max_items = })")

            BUILT_PROVIDERS_LRU_CACHE = LRUCache(max_items)

        cache_key = get_cache_key_func()

        try:
            provider = BUILT_PROVIDERS_LRU_CACHE[cache_key]

            if verbose:
                eprint(
                    f"Loaded credential provider from cache: {provider!r} {cache_key = }"
                )
        except KeyError:
            provider = build_provider_func()
            BUILT_PROVIDERS_LRU_CACHE[cache_key] = provider

            if verbose:
                eprint(
                    f"Added new credential provider to cache: {provider!r} {cache_key = }"
                )

        return provider


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
            return _build_with_cache(
                self.get_or_init_cache_key,
                lambda: self.cls(**self.kw),
            )
        except ImportError as e:
            if verbose():
                eprint(f"failed to auto-initialize {self.provider_repr}: {e!r}")

        return None

    def get_or_init_cache_key(self) -> bytes:
        cache_key = self._cache_key.get()

        if cache_key is None:
            cache_key = self.get_cache_key_impl()
            self._cache_key.set(cache_key)

            if verbose():
                eprint(f"{self!r}: AutoInit cache key: {cache_key.hex()}")

        return cache_key

    def get_cache_key_impl(self) -> bytes:
        import hashlib
        import pickle

        hash = hashlib.sha256(pickle.dumps(self))
        return hash.digest()[:16]

    def stable_cache_key(self) -> bytes:
        return self.get_or_init_cache_key()

    @property
    def provider_repr(self) -> str:
        return self.cls.__name__


DEFAULT_CREDENTIAL_PROVIDER: CredentialProviderFunction | Literal["auto"] | None = (
    "auto"
)


def _init_credential_provider_builder(
    credential_provider: CredentialProviderFunction
    | CredentialProviderBuilder
    | Literal["auto"]
    | None,
    source: Any,
    storage_options: StorageOptionsDict | None,
    caller_name: str,
) -> CredentialProviderBuilder | None:
    aws_storage_options: dict[str, str | None] = {}
    for key, value in (storage_options or {}).items():
        key = key.lower().removeprefix("aws_")
        if key in {"region", "default_region", "virtual_hosted_style_request"}:
            aws_storage_options[key] = value
        elif key in {"endpoint", "endpoint_url", "endpoint_url_s3"}:
            aws_storage_options["endpoint"] = None

    def bind_aws_storage_options(
        provider: CredentialProviderFunction,
    ) -> CredentialProviderFunction:
        if not aws_storage_options or not isinstance(provider, CredentialProviderAWS):
            return provider

        provider = copy(provider)
        provider._storage_options = aws_storage_options
        return provider

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
                bind_aws_storage_options(credential_provider)
            )

        if DEFAULT_CREDENTIAL_PROVIDER is None:
            return None

        if (first_scan_path := _first_scan_path(source)) is None:
            return None

        if (scheme := _get_path_scheme(first_scan_path)) is None:
            return None

        def get_default_credential_provider(
            *, bind_aws: bool = False
        ) -> CredentialProviderBuilder | None:
            return (
                CredentialProviderBuilder.from_initialized_provider(
                    bind_aws_storage_options(DEFAULT_CREDENTIAL_PROVIDER)
                    if bind_aws
                    else DEFAULT_CREDENTIAL_PROVIDER
                )
                if DEFAULT_CREDENTIAL_PROVIDER != "auto"
                else None
            )

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
                    elif k in AUTOINIT_IGNORED_KEYS:
                        continue
                    else:
                        # We assume some sort of access key was given, so we
                        # just dispatch to the rust side.
                        return None

            storage_account = (
                # Prefer the one embedded in the path
                CredentialProviderAzure._extract_adls_uri_storage_account(
                    str(first_scan_path)
                )
                or storage_account
            )

            if (default := get_default_credential_provider()) is not None:
                return default

            return CredentialProviderBuilder(
                AutoInit(
                    CredentialProviderAzure,
                    tenant_id=tenant_id,
                    _storage_account=storage_account,
                )
            )

        elif _is_aws_cloud(scheme=scheme, first_scan_path=str(first_scan_path)):
            profile = None
            region = aws_storage_options.get("region")
            default_region = aws_storage_options.get("default_region")
            unhandled_key = None

            if storage_options is not None:
                for k, v in storage_options.items():
                    k = k.lower()

                    # https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html
                    if k in {"aws_profile", "profile"}:
                        profile = v
                    elif k in AUTOINIT_IGNORED_KEYS or k.removeprefix("aws_") in {
                        "region",
                        "default_region",
                        "endpoint",
                        "endpoint_url",
                        "endpoint_url_s3",
                        "virtual_hosted_style_request",
                    }:
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

            if (
                unhandled_key is None
                and (default := get_default_credential_provider(bind_aws=True))
                is not None
            ):
                return default

            return CredentialProviderBuilder(
                AutoInit(
                    CredentialProviderAWS,
                    profile_name=profile,
                    region_name=region or default_region,
                    _auto_init_unhandled_key=unhandled_key,
                    _storage_options={**aws_storage_options, "region": region},
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
                    elif k in AUTOINIT_IGNORED_KEYS:
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

            if (default := get_default_credential_provider()) is not None:
                return default

            return CredentialProviderBuilder(AutoInit(CredentialProviderGCP))

        return None

    credential_provider_init = f()

    if verbose():
        eprint(f"_init_credential_provider_builder(): {credential_provider_init = !r}")

    return credential_provider_init
