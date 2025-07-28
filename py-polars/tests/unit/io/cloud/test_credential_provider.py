import io
import pickle
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

import polars as pl
import polars.io.cloud.credential_provider
from polars.io.cloud._utils import NoPickleOption, ZeroHashWrap
from polars.io.cloud.credential_provider._builder import (
    _init_credential_provider_builder,
)


@pytest.mark.parametrize(
    "io_func",
    [
        *[pl.scan_parquet, pl.read_parquet],
        pl.scan_csv,
        *[pl.scan_ndjson, pl.read_ndjson],
        pl.scan_ipc,
    ],
)
def test_credential_provider_scan(
    io_func: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    err_magic = "err_magic_3"

    def raises(*_: None, **__: None) -> None:
        raise AssertionError(err_magic)

    from polars.io.cloud.credential_provider._builder import CredentialProviderBuilder

    monkeypatch.setattr(CredentialProviderBuilder, "__init__", raises)

    with pytest.raises(AssertionError, match=err_magic):
        io_func("s3://bucket/path", credential_provider="auto")

    with pytest.raises(AssertionError, match=err_magic):
        io_func(
            "s3://bucket/path",
            credential_provider="auto",
            storage_options={"aws_region": "eu-west-1"},
        )

    # We can't test these with the `read_` functions as they end up executing
    # the query
    if io_func.__name__.startswith("scan_"):
        # Passing `None` should disable the automatic instantiation of
        # `CredentialProviderAWS`
        io_func("s3://bucket/path", credential_provider=None)

    err_magic = "err_magic_7"

    def raises_2() -> pl.CredentialProviderFunctionReturn:
        raise AssertionError(err_magic)

    with pytest.raises(AssertionError, match=err_magic):
        io_func("s3://bucket/path", credential_provider=raises_2).collect()


@pytest.mark.parametrize(
    ("provider_class", "path"),
    [
        (polars.io.cloud.credential_provider.CredentialProviderAWS, "s3://.../..."),
        (polars.io.cloud.credential_provider.CredentialProviderGCP, "gs://.../..."),
        (polars.io.cloud.credential_provider.CredentialProviderAzure, "az://.../..."),
    ],
)
def test_credential_provider_serialization_auto_init(
    provider_class: polars.io.cloud.credential_provider.CredentialProvider,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raises_1(*a: Any, **kw: Any) -> None:
        msg = "err_magic_1"
        raise AssertionError(msg)

    monkeypatch.setattr(provider_class, "__init__", raises_1)

    # If this is not set we will get an error before hitting the credential
    # provider logic when polars attempts to retrieve the region from AWS.
    monkeypatch.setenv("AWS_REGION", "eu-west-1")

    # Credential provider should not be initialized during query plan construction.
    q = pl.scan_parquet(path)

    # Check baseline - query plan is configured to auto-initialize the credential
    # provider.
    with pytest.raises(AssertionError, match="err_magic_1"):
        q.collect()

    q = pickle.loads(pickle.dumps(q))

    def raises_2(*a: Any, **kw: Any) -> None:
        msg = "err_magic_2"
        raise AssertionError(msg)

    monkeypatch.setattr(provider_class, "__init__", raises_2)

    # Check that auto-initialization happens upon executing the deserialized
    # query.
    with pytest.raises(AssertionError, match="err_magic_2"):
        q.collect()


def test_credential_provider_serialization_custom_provider() -> None:
    err_magic = "err_magic_3"

    class ErrCredentialProvider(pl.CredentialProvider):
        def retrieve_credentials_impl(self) -> pl.CredentialProviderFunctionReturn:
            raise AssertionError(err_magic)

    lf = pl.scan_parquet(
        "s3://bucket/path", credential_provider=ErrCredentialProvider()
    )

    serialized = lf.serialize()

    lf = pl.LazyFrame.deserialize(io.BytesIO(serialized))

    with pytest.raises(AssertionError, match=err_magic):
        lf.collect()


def test_credential_provider_gcp_skips_config_autoload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_PATH", "__non_existent")

    with pytest.raises(OSError, match="__non_existent"):
        pl.scan_parquet("gs://.../...", credential_provider=None).collect()

    err_magic = "err_magic_3"

    def raises() -> pl.CredentialProviderFunctionReturn:
        raise AssertionError(err_magic)

    with pytest.raises(AssertionError, match=err_magic):
        pl.scan_parquet("gs://.../...", credential_provider=raises).collect()


def test_credential_provider_aws_import_error_with_requested_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _session(self: Any) -> None:
        msg = "err_magic_3"
        raise ImportError(msg)

    monkeypatch.setattr(pl.CredentialProviderAWS, "_session", _session)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")

    q = pl.scan_parquet(
        "s3://.../...",
        credential_provider=pl.CredentialProviderAWS(profile_name="test_profile"),
    )

    with pytest.raises(
        pl.exceptions.ComputeError,
        match=(
            "cannot load requested aws_profile 'test_profile': ImportError: err_magic_3"
        ),
    ):
        q.collect()

    q = pl.scan_parquet(
        "s3://.../...",
        storage_options={"aws_profile": "test_profile"},
    )

    with pytest.raises(
        pl.exceptions.ComputeError,
        match=(
            "cannot load requested aws_profile 'test_profile': ImportError: err_magic_3"
        ),
    ):
        q.collect()


@pytest.mark.slow
@pytest.mark.write_disk
def test_credential_provider_aws_endpoint_url_scan_no_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    tmp_path.mkdir(exist_ok=True)

    _set_default_credentials(tmp_path, monkeypatch)
    cfg_file_path = tmp_path / "config"

    monkeypatch.setenv("AWS_CONFIG_FILE", str(cfg_file_path))
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    cfg_file_path.write_text("""\
[default]
endpoint_url = http://localhost:333
""")

    # Scan with no parameters should load via CredentialProviderAWS
    q = pl.scan_parquet("s3://.../...")

    capfd.readouterr()

    with pytest.raises(IOError, match=r"Error performing HEAD http://localhost:333"):
        q.collect()

    capture = capfd.readouterr().err
    lines = capture.splitlines()

    assert "[CredentialProviderAWS]: Loaded endpoint_url: http://localhost:333" in lines


@pytest.mark.slow
@pytest.mark.write_disk
def test_credential_provider_aws_endpoint_url_serde(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    tmp_path.mkdir(exist_ok=True)

    _set_default_credentials(tmp_path, monkeypatch)
    cfg_file_path = tmp_path / "config"

    monkeypatch.setenv("AWS_CONFIG_FILE", str(cfg_file_path))
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    cfg_file_path.write_text("""\
[default]
endpoint_url = http://localhost:333
""")

    q = pl.scan_parquet("s3://.../...")
    q = pickle.loads(pickle.dumps(q))

    cfg_file_path.write_text("""\
[default]
endpoint_url = http://localhost:777
""")

    capfd.readouterr()

    with pytest.raises(IOError, match=r"Error performing HEAD http://localhost:777"):
        q.collect()


@pytest.mark.slow
@pytest.mark.write_disk
def test_credential_provider_aws_endpoint_url_with_storage_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    tmp_path.mkdir(exist_ok=True)

    _set_default_credentials(tmp_path, monkeypatch)
    cfg_file_path = tmp_path / "config"

    monkeypatch.setenv("AWS_CONFIG_FILE", str(cfg_file_path))
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    cfg_file_path.write_text("""\
[default]
endpoint_url = http://localhost:333
""")

    # Previously we would not initialize a credential provider at all if secrets
    # were given under `storage_options`. Now we always initialize so that we
    # can load the `endpoint_url`, but we decide at the very last second whether
    # to also retrieve secrets using the credential provider.
    q = pl.scan_parquet(
        "s3://.../...",
        storage_options={
            "aws_access_key_id": "...",
            "aws_secret_access_key": "...",
        },
    )

    with pytest.raises(IOError, match=r"Error performing HEAD http://localhost:333"):
        q.collect()

    capture = capfd.readouterr().err
    lines = capture.splitlines()

    assert (
        "[CredentialProviderAWS]: Will not be used as a provider: unhandled key "
        "in storage_options: 'aws_secret_access_key'"
    ) in lines
    assert "[CredentialProviderAWS]: Loaded endpoint_url: http://localhost:333" in lines


@pytest.mark.parametrize(
    "storage_options",
    [
        {"aws_endpoint_url": "http://localhost:777"},
        {
            "aws_access_key_id": "...",
            "aws_secret_access_key": "...",
            "aws_endpoint_url": "http://localhost:777",
        },
    ],
)
@pytest.mark.slow
@pytest.mark.write_disk
def test_credential_provider_aws_endpoint_url_passed_in_storage_options(
    storage_options: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.mkdir(exist_ok=True)

    _set_default_credentials(tmp_path, monkeypatch)
    cfg_file_path = tmp_path / "config"
    monkeypatch.setenv("AWS_CONFIG_FILE", str(cfg_file_path))

    cfg_file_path.write_text("""\
[default]
endpoint_url = http://localhost:333
""")

    q = pl.scan_parquet("s3://.../...")

    with pytest.raises(IOError, match=r"Error performing HEAD http://localhost:333"):
        q.collect()

    # An endpoint_url passed in `storage_options` should take precedence.
    q = pl.scan_parquet(
        "s3://.../...",
        storage_options=storage_options,
    )

    with pytest.raises(IOError, match=r"Error performing HEAD http://localhost:777"):
        q.collect()


def _set_default_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    creds_file_path = tmp_path / "credentials"
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(creds_file_path))

    creds_file_path.write_text("""\
[default]
aws_access_key_id=Z
aws_secret_access_key=Z
""")


@pytest.mark.slow
def test_credential_provider_python_builder_cache(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # Tests caching of building credential providers.
    def dummy_static_aws_credentials(*a: Any, **kw: Any) -> Any:
        return {
            "aws_access_key_id": "...",
            "aws_secret_access_key": "...",
        }, None

    with monkeypatch.context() as cx:
        init_tracker = TrackCallCount(pl.CredentialProviderAWS.__init__)

        cx.setattr(
            pl.CredentialProviderAWS,
            "__init__",
            init_tracker.get_function(),
        )

        cx.setattr(
            pl.CredentialProviderAWS,
            "retrieve_credentials_impl",
            dummy_static_aws_credentials,
        )

        # Ensure we are building a new query every time.
        def get_q() -> pl.LazyFrame:
            return pl.scan_parquet(
                "s3://.../...",
                storage_options={
                    "aws_profile": "A",
                    "aws_endpoint_url": "http://localhost",
                },
                credential_provider="auto",
            )

        assert init_tracker.count == 0

        with pytest.raises(OSError):
            get_q().collect()

        assert init_tracker.count == 1

        with pytest.raises(OSError):
            get_q().collect()

        assert init_tracker.count == 1

        with pytest.raises(OSError):
            pl.scan_parquet(
                "s3://.../...",
                storage_options={
                    "aws_profile": "B",
                    "aws_endpoint_url": "http://localhost",
                },
                credential_provider="auto",
            ).collect()

        assert init_tracker.count == 2

        with pytest.raises(OSError):
            get_q().collect()

        assert init_tracker.count == 2

        cx.setenv("POLARS_CREDENTIAL_PROVIDER_BUILDER_CACHE_SIZE", "0")

        with pytest.raises(OSError):
            get_q().collect()

        # Note: Increments by 2 due to Rust-side object store rebuilding.

        assert init_tracker.count == 4

        with pytest.raises(OSError):
            get_q().collect()

        assert init_tracker.count == 6

    with monkeypatch.context() as cx:
        cx.setenv("POLARS_VERBOSE", "1")
        builder = _init_credential_provider_builder(
            "auto",
            "s3://.../...",
            None,
            "test",
        )
        assert builder is not None

        capfd.readouterr()

        builder.build_credential_provider()
        builder.build_credential_provider()

        capture = capfd.readouterr().err

        # Ensure cache key is memoized on generation
        assert capture.count("AutoInit cache key") == 1

        pickle.loads(pickle.dumps(builder)).build_credential_provider()

        capture = capfd.readouterr().err

        # Ensure cache key is not serialized
        assert capture.count("AutoInit cache key") == 1


@pytest.mark.slow
def test_credential_provider_python_credentials_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dummy_static_aws_credentials(*a: Any, **kw: Any) -> Any:
        return {
            "aws_access_key_id": "...",
            "aws_secret_access_key": "...",
        }, None

    tracker = TrackCallCount(dummy_static_aws_credentials)

    monkeypatch.setattr(
        pl.CredentialProviderAWS,
        "retrieve_credentials_impl",
        tracker.get_function(),
    )

    assert tracker.count == 0

    provider = pl.CredentialProviderAWS()

    provider()
    assert tracker.count == 1

    provider()
    assert tracker.count == 1

    monkeypatch.setenv("POLARS_DISABLE_PYTHON_CREDENTIAL_CACHING", "1")

    provider()
    assert tracker.count == 2

    provider()
    assert tracker.count == 3

    monkeypatch.delenv("POLARS_DISABLE_PYTHON_CREDENTIAL_CACHING")

    provider()
    assert tracker.count == 4

    provider()
    assert tracker.count == 4

    assert provider._cached_credentials.get() is not None
    assert pickle.loads(pickle.dumps(provider))._cached_credentials.get() is None


class TrackCallCount:  # noqa: D101
    def __init__(self, func: Any) -> None:
        self.func = func
        self.count = 0

    def get_function(self) -> Any:
        def f(*a: Any, **kw: Any) -> Any:
            self.count += 1
            return self.func(*a, **kw)

        return f


def test_no_pickle_option() -> None:
    v = NoPickleOption(3)
    assert v.get() == 3

    out = pickle.loads(pickle.dumps(v))

    assert out.get() is None


def test_zero_hash_wrap() -> None:
    v = ZeroHashWrap(3)
    assert v.get() == 3

    assert ZeroHashWrap(3) == ZeroHashWrap("7")

    @lru_cache
    def cache(value: ZeroHashWrap[Any]) -> int:
        return value.get()  # type: ignore[no-any-return]

    assert cache(ZeroHashWrap(3)) == 3
    assert cache(ZeroHashWrap(7)) == 3
    assert cache(ZeroHashWrap("A")) == 3


@pytest.mark.write_disk
def test_credential_provider_aws_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential_file_path = tmp_path / "credentials.json"

    credential_file_path.write_text(
        """\
{
    "Version": 1,
    "AccessKeyId": "123",
    "SecretAccessKey": "456",
    "SessionToken": "789",
    "Expiration": "2099-01-01T00:00:00+00:00"
}
"""
    )

    cfg_file_path = tmp_path / "config"

    credential_file_path_str = str(credential_file_path).replace("\\", "/")

    cfg_file_path.write_text(f"""\
[profile cred_process]
credential_process = "{sys.executable}" -c "from pathlib import Path; print(Path('{credential_file_path_str}').read_text())"
""")

    monkeypatch.setenv("AWS_CONFIG_FILE", str(cfg_file_path))

    creds, expiry = pl.CredentialProviderAWS(profile_name="cred_process")()

    assert creds == {
        "aws_access_key_id": "123",
        "aws_secret_access_key": "456",
        "aws_session_token": "789",
    }

    assert expiry is not None

    assert datetime.fromtimestamp(expiry, tz=timezone.utc) == datetime.fromisoformat(
        "2099-01-01T00:00:00+00:00"
    )

    credential_file_path.write_text(
        """\
{
    "Version": 1,
    "AccessKeyId": "...",
    "SecretAccessKey": "...",
    "SessionToken": "..."
}
"""
    )

    creds, expiry = pl.CredentialProviderAWS(profile_name="cred_process")()

    assert creds == {
        "aws_access_key_id": "...",
        "aws_secret_access_key": "...",
        "aws_session_token": "...",
    }

    assert expiry is None
