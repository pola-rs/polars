import pytest

import polars as pl
from polars.exceptions import ComputeError


def test_credential_provider_skips_config_autoload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_PATH", "__non_existent")

    with pytest.raises(ComputeError, match="__non_existent"):
        pl.scan_parquet("gs://.../...", credential_provider=None).collect()

    err_magic = "err_magic_3"

    def raises() -> pl.CredentialProviderFunctionReturn:
        raise AssertionError(err_magic)

    # We should get a different error raised by our `raises()` function.
    with pytest.raises(ComputeError, match=err_magic):
        pl.scan_parquet("gs://.../...", credential_provider=raises).collect()
