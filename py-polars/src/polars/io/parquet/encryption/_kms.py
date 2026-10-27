from __future__ import annotations

import contextlib
from collections.abc import Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from polars._utils.various import qualified_type_name

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import (
        PyCryptoFactory,
        PyDecryptionConfiguration,
        PyEncryptionConfiguration,
        PyKmsConnectionConfig,
        PyParquetDecryption,
        PyParquetEncryption,
    )

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_DEFAULT_CACHE_LIFETIME = timedelta(minutes=10)


class KmsClient:
    """
    Base class for clients of a Key Management Service (KMS).

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    A KMS client wraps (encrypts) and unwraps (decrypts) data encryption keys
    with master keys that are stored in the KMS. Implement this to integrate
    with your organization's KMS.

    Subclassing this is optional: any object with `wrap_key` and `unwrap_key`
    methods can be used, including subclasses of PyArrow's
    `pyarrow.parquet.encryption.KmsClient`.

    Notes
    -----
    These methods may be called from multiple threads, by any Polars query that
    reads or writes encrypted Parquet files.
    """

    def wrap_key(self, key_bytes: bytes, master_key_identifier: str) -> str:
        """
        Wrap a key with the master key in the KMS.

        Parameters
        ----------
        key_bytes
            The key to wrap.
        master_key_identifier
            The ID of the master key in the KMS.

        Returns
        -------
        str
            The wrapped key. This is stored in the Parquet file's key metadata.
        """
        raise NotImplementedError

    def unwrap_key(self, wrapped_key: str, master_key_identifier: str) -> bytes:
        """
        Unwrap a key with the master key in the KMS.

        Parameters
        ----------
        wrapped_key
            A key returned by :meth:`wrap_key`.
        master_key_identifier
            The ID of the master key in the KMS.

        Returns
        -------
        bytes
            The unwrapped key. Other bytes-like objects such as `bytearray` are
            also accepted.
        """
        raise NotImplementedError


class KmsConnectionConfig:
    """
    Configuration for connecting to a Key Management Service (KMS).

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    kms_instance_id
        The ID of the KMS instance to use. If this is not set, the KMS client
        factory is given `"DEFAULT"`.
    kms_instance_url
        The URL of the KMS instance to use. If this is not set, the KMS client
        factory is given `"DEFAULT"`, or the URL stored in the file being read if
        :class:`DecryptionConfiguration` sets `read_kms_url`.
    key_access_token
        The authorization token passed to the KMS client factory. Use
        :meth:`refresh_key_access_token` to change it.
    custom_kms_conf
        Any KMS-specific configuration, passed to the KMS client factory.

    Notes
    -----
    Unlike PyArrow's `KmsConnectionConfig`, the properties are read-only, apart
    from refreshing the key access token. The object passed to the KMS client
    factory is a read-only snapshot: its token can't be refreshed, and it can't
    be used to read or write files.
    """

    _pyconfig: PyKmsConnectionConfig

    def __init__(
        self,
        kms_instance_id: str = "",
        *,
        kms_instance_url: str = "",
        key_access_token: str = "DEFAULT",
        custom_kms_conf: Mapping[str, str] | None = None,
    ) -> None:
        self._pyconfig = PyKmsConnectionConfig(
            kms_instance_id,
            kms_instance_url,
            key_access_token,
            list(custom_kms_conf.items()) if custom_kms_conf is not None else [],
        )

    @classmethod
    def _from_pyconfig(cls, pyconfig: PyKmsConnectionConfig) -> KmsConnectionConfig:
        config = cls.__new__(cls)
        config._pyconfig = pyconfig
        return config

    @property
    def kms_instance_id(self) -> str:
        """The ID of the KMS instance."""
        return self._pyconfig.kms_instance_id

    @property
    def kms_instance_url(self) -> str:
        """The URL of the KMS instance."""
        return self._pyconfig.kms_instance_url

    @property
    def key_access_token(self) -> str:
        """The authorization token passed to the KMS client factory."""
        return self._pyconfig.key_access_token

    @property
    def custom_kms_conf(self) -> dict[str, str]:
        """Any KMS-specific configuration."""
        return self._pyconfig.custom_kms_conf

    def refresh_key_access_token(self, value: str) -> None:
        """
        Set a new key access token.

        The new token is used for KMS clients created after this call, including
        by queries that are already running with this configuration.

        Parameters
        ----------
        value
            The new token.
        """
        self._pyconfig.refresh_key_access_token(value)


class CryptoFactory:
    """
    Generates and unwraps encryption keys using a Key Management Service (KMS).

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    The KMS clients created by the factory, and unwrapped key encryption keys,
    are cached by the factory and shared by every query that uses it, for the
    cache lifetime of the :class:`EncryptionConfiguration` or
    :class:`DecryptionConfiguration`.

    Parameters
    ----------
    kms_client_factory
        Called with a :class:`KmsConnectionConfig` to create a KMS client: an
        object with the methods of :class:`KmsClient`.
    """

    def __init__(
        self, kms_client_factory: Callable[[KmsConnectionConfig], Any]
    ) -> None:
        if not callable(kms_client_factory):
            msg = (
                "`kms_client_factory` must be callable, "
                f"got {qualified_type_name(kms_client_factory)!r}"
            )
            raise TypeError(msg)

        def create_client(pyconfig: PyKmsConnectionConfig) -> Any:
            return kms_client_factory(KmsConnectionConfig._from_pyconfig(pyconfig))

        self._pycrypto_factory = PyCryptoFactory(create_client)


class EncryptionConfiguration:
    """
    Configuration for writing Parquet files with keys managed by a KMS.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Exactly one of `column_keys` or `uniform_encryption` must be set.

    Parameters
    ----------
    footer_key
        The ID of the master key used to encrypt the file footer, or to sign the
        footer if `plaintext_footer` is set.
    column_keys
        Which columns to encrypt, as a mapping from the ID of a master key to the
        names of the columns encrypted with it. Only these columns are encrypted.
        For nested columns, the column name is the dot-separated path in the
        Parquet schema, e.g. `a.b.c`.
    uniform_encryption
        Encrypt all columns with the footer key.
    plaintext_footer
        Write the file footer in plaintext rather than encrypting it, so that
        readers without the footer key can read the schema and any unencrypted
        columns. The footer is still signed with the footer key.
    double_wrapping
        Wrap data encryption keys with key encryption keys, which are wrapped by
        the master keys in the KMS. This reduces the number of KMS calls. If false,
        data encryption keys are wrapped by the KMS directly.
    cache_lifetime
        How long to cache KMS clients and key encryption keys for.
    """

    def __init__(
        self,
        footer_key: str,
        *,
        column_keys: Mapping[str, Sequence[str]] | None = None,
        uniform_encryption: bool = False,
        plaintext_footer: bool = False,
        double_wrapping: bool = True,
        cache_lifetime: timedelta = _DEFAULT_CACHE_LIFETIME,
    ) -> None:
        column_keys_dict = _check_column_keys(column_keys)
        if column_keys_dict and uniform_encryption:
            msg = "cannot set both `column_keys` and `uniform_encryption`"
            raise ValueError(msg)
        if not column_keys_dict and not uniform_encryption:
            msg = "either `column_keys` or `uniform_encryption` must be set"
            raise ValueError(msg)
        _check_cache_lifetime(cache_lifetime)

        self._footer_key = footer_key
        self._column_keys = column_keys_dict
        self._uniform_encryption = uniform_encryption
        self._plaintext_footer = plaintext_footer
        self._double_wrapping = double_wrapping
        self._cache_lifetime = cache_lifetime
        self._pyconfig = PyEncryptionConfiguration(
            footer_key,
            list(column_keys_dict.items()),
            plaintext_footer,
            double_wrapping,
            cache_lifetime,
        )

    @property
    def footer_key(self) -> str:
        """The ID of the master key used for the footer."""
        return self._footer_key

    @property
    def column_keys(self) -> dict[str, list[str]]:
        """The columns encrypted with each master key."""
        return {key: list(columns) for key, columns in self._column_keys.items()}

    @property
    def uniform_encryption(self) -> bool:
        """Whether all columns are encrypted with the footer key."""
        return self._uniform_encryption

    @property
    def plaintext_footer(self) -> bool:
        """Whether the footer is written in plaintext."""
        return self._plaintext_footer

    @property
    def double_wrapping(self) -> bool:
        """Whether data encryption keys are wrapped with key encryption keys."""
        return self._double_wrapping

    @property
    def cache_lifetime(self) -> timedelta:
        """How long KMS clients and key encryption keys are cached for."""
        return self._cache_lifetime


class DecryptionConfiguration:
    """
    Configuration for reading Parquet files with keys managed by a KMS.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    cache_lifetime
        How long to cache KMS clients and key encryption keys for.
    read_kms_url
        Whether the KMS instance URL may be read from the key metadata of the
        file being read, when it is not set in the :class:`KmsConnectionConfig`.
        The KMS client factory receives this URL along with the key access
        token, so this should only be enabled when the KMS client validates the
        URL it receives. Otherwise, a file could send the token to any server.
    """

    def __init__(
        self,
        *,
        cache_lifetime: timedelta = _DEFAULT_CACHE_LIFETIME,
        read_kms_url: bool = False,
    ) -> None:
        _check_cache_lifetime(cache_lifetime)
        self._cache_lifetime = cache_lifetime
        self._read_kms_url = read_kms_url
        self._pyconfig = PyDecryptionConfiguration(cache_lifetime, read_kms_url)

    @property
    def cache_lifetime(self) -> timedelta:
        """How long KMS clients and key encryption keys are cached for."""
        return self._cache_lifetime

    @property
    def read_kms_url(self) -> bool:
        """Whether the KMS instance URL may be read from the file."""
        return self._read_kms_url


class ParquetEncryptionConfig:
    """
    Write Parquet files encrypted with keys managed by a KMS.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Pass this as the `encryption` parameter of
    :meth:`DataFrame.write_parquet <polars.DataFrame.write_parquet>` or
    :meth:`LazyFrame.sink_parquet <polars.LazyFrame.sink_parquet>`. New data
    encryption keys are generated for every file written.

    Parameters
    ----------
    crypto_factory
        Generates the keys, and wraps them with the KMS.
    kms_connection_config
        How to connect to the KMS.
    encryption_config
        How to encrypt the files.

    Examples
    --------
    >>> from polars.io.parquet.encryption import (
    ...     CryptoFactory,
    ...     EncryptionConfiguration,
    ...     KmsClient,
    ...     KmsConnectionConfig,
    ...     ParquetEncryptionConfig,
    ... )
    >>> class MyKmsClient(KmsClient):
    ...     def __init__(
    ...         self, config: KmsConnectionConfig
    ...     ) -> None: ...  # Connect to the KMS
    ...
    ...     def wrap_key(
    ...         self, key_bytes: bytes, master_key_identifier: str
    ...     ) -> str: ...  # Wrap the key using the KMS
    ...
    ...     def unwrap_key(
    ...         self, wrapped_key: str, master_key_identifier: str
    ...     ) -> bytes: ...  # Unwrap the key using the KMS
    >>> crypto_factory = CryptoFactory(MyKmsClient)
    >>> kms_connection_config = KmsConnectionConfig(
    ...     kms_instance_url="https://kms.example.com"
    ... )
    >>> encryption = ParquetEncryptionConfig(
    ...     crypto_factory,
    ...     kms_connection_config,
    ...     EncryptionConfiguration(
    ...         "footer_key_id", column_keys={"column_key_id": ["x"]}
    ...     ),
    ... )
    >>> df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    >>> df.write_parquet("encrypted.parquet", encryption=encryption)  # doctest: +SKIP
    """

    def __init__(
        self,
        crypto_factory: CryptoFactory,
        kms_connection_config: KmsConnectionConfig,
        encryption_config: EncryptionConfiguration,
    ) -> None:
        _check_type(crypto_factory, CryptoFactory, "crypto_factory")
        _check_type(kms_connection_config, KmsConnectionConfig, "kms_connection_config")
        _check_type(encryption_config, EncryptionConfiguration, "encryption_config")
        self._pyencryption = PyParquetEncryption.from_kms(
            crypto_factory._pycrypto_factory,
            kms_connection_config._pyconfig,
            encryption_config._pyconfig,
        )


class ParquetDecryptionConfig:
    """
    Read Parquet files encrypted with keys managed by a KMS.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Pass this as the `decryption` parameter of
    :func:`read_parquet <polars.read_parquet>` or
    :func:`scan_parquet <polars.scan_parquet>`.

    Parameters
    ----------
    crypto_factory
        Unwraps the keys with the KMS.
    kms_connection_config
        How to connect to the KMS.
    decryption_config
        How to decrypt the files.

    Examples
    --------
    Using `MyKmsClient` from the :class:`ParquetEncryptionConfig` example:

    >>> from polars.io.parquet.encryption import (
    ...     CryptoFactory,
    ...     DecryptionConfiguration,
    ...     KmsConnectionConfig,
    ...     ParquetDecryptionConfig,
    ... )
    >>> crypto_factory = CryptoFactory(MyKmsClient)  # doctest: +SKIP
    >>> decryption = ParquetDecryptionConfig(  # doctest: +SKIP
    ...     crypto_factory,
    ...     KmsConnectionConfig(kms_instance_url="https://kms.example.com"),
    ...     DecryptionConfiguration(),
    ... )
    >>> pl.read_parquet("encrypted.parquet", decryption=decryption)  # doctest: +SKIP
    """

    def __init__(
        self,
        crypto_factory: CryptoFactory,
        kms_connection_config: KmsConnectionConfig,
        decryption_config: DecryptionConfiguration,
    ) -> None:
        _check_type(crypto_factory, CryptoFactory, "crypto_factory")
        _check_type(kms_connection_config, KmsConnectionConfig, "kms_connection_config")
        _check_type(decryption_config, DecryptionConfiguration, "decryption_config")
        self._pydecryption = PyParquetDecryption.from_kms(
            crypto_factory._pycrypto_factory,
            kms_connection_config._pyconfig,
            decryption_config._pyconfig,
        )


def _check_type(value: Any, expected: type, name: str) -> None:
    if not isinstance(value, expected):
        msg = (
            f"`{name}` must be a `{expected.__name__}`, "
            f"got {qualified_type_name(value)!r}"
        )
        raise TypeError(msg)


def _check_column_keys(
    column_keys: Mapping[str, Sequence[str]] | None,
) -> dict[str, list[str]]:
    if column_keys is None:
        return {}
    if not isinstance(column_keys, Mapping):
        msg = (
            "`column_keys` must be a mapping from master key ID to column names, "
            f"got {qualified_type_name(column_keys)!r}"
        )
        raise TypeError(msg)
    result = {}
    for master_key_id, columns in column_keys.items():
        msg = (
            f"the columns for master key {master_key_id!r} must be a "
            f"sequence of column names, got {columns!r}"
        )
        # A string is a sequence of characters, but is almost certainly a mistake.
        if isinstance(columns, str):
            raise TypeError(msg)
        # Materialize first so that one-shot iterables aren't consumed by the
        # type check.
        try:
            column_list = list(columns)
        except TypeError:
            raise TypeError(msg) from None
        if not all(isinstance(c, str) for c in column_list):
            raise TypeError(msg)
        if not column_list:
            msg = f"no columns given for master key {master_key_id!r}"
            raise ValueError(msg)
        result[master_key_id] = column_list
    return result


def _check_cache_lifetime(cache_lifetime: timedelta) -> None:
    if not isinstance(cache_lifetime, timedelta):
        msg = (
            "`cache_lifetime` must be a `timedelta`, "
            f"got {qualified_type_name(cache_lifetime)!r}"
        )
        raise TypeError(msg)
    if cache_lifetime < timedelta(0):
        msg = f"`cache_lifetime` must not be negative, got {cache_lifetime!r}"
        raise ValueError(msg)
