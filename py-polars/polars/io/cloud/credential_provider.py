from __future__ import annotations

import abc
import importlib.util
import os
import sys
import zoneinfo
from typing import TYPE_CHECKING, Callable, Optional, Union

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

from polars._utils.unstable import issue_unstable_warning

if TYPE_CHECKING:
    from polars._typing import ScanSource


# These typedefs are here to avoid circular import issues, as
# `CredentialProviderFunction` specifies "CredentialProvider"
CredentialProviderFunctionReturn: TypeAlias = tuple[
    dict[str, Optional[str]], Optional[int]
]

CredentialProviderFunction: TypeAlias = Union[
    Callable[[], CredentialProviderFunctionReturn], "CredentialProvider"
]


class CredentialProvider(abc.ABC):
    """
    Base class for credential providers.

    .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
    """

    @abc.abstractmethod
    def __call__(self) -> CredentialProviderFunctionReturn:
        """Fetches the credentials."""


class CredentialProviderAWS(CredentialProvider):
    """
    AWS Credential Provider.

    Using this requires the `boto3` Python package to be installed.

    .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
    """

    def __init__(self, *, profile_name: str = "default") -> None:
        """
        Initialize a credential provider for AWS.

        Parameters
        ----------
        profile_name : str
            Profile name to use from credentials file.
        """
        msg = "`CredentialProviderAWS` functionality is considered unstable"
        issue_unstable_warning(msg)

        self._check_module_availability()
        self.profile_name = profile_name

    def __call__(self) -> CredentialProviderFunctionReturn:
        """Fetch the credentials for the configured profile name."""
        import boto3

        session = boto3.Session(profile_name=self.profile_name)
        creds = session.get_credentials()

        if creds is None:
            msg = "unexpected None value returned from boto3.Session.get_credentials()"
            raise ValueError(msg)

        return {
            "aws_access_key_id": creds.access_key,
            "aws_secret_access_key": creds.secret_key,
            "aws_session_token": creds.token,
        }, None

    @classmethod
    def _check_module_availability(cls) -> None:
        if importlib.util.find_spec("boto3") is None:
            msg = "boto3 must be installed to use `CredentialProviderAWS`"
            raise ImportError(msg)


class CredentialProviderGCP(CredentialProvider):
    """
    GCP Credential Provider.

    Using this requires the `google-auth` Python package to be installed.

    .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
    """

    def __init__(self) -> None:
        """Initialize a credential provider for Google Cloud (GCP)."""
        msg = "`CredentialProviderAWS` functionality is considered unstable"
        issue_unstable_warning(msg)

        self._check_module_availability()

        import google.auth
        import google.auth.credentials

        creds, _ = google.auth.default()  # type: ignore[no-untyped-call]
        self.creds = creds

    def __call__(self) -> CredentialProviderFunctionReturn:
        """Fetch the credentials for the configured profile name."""
        import google.auth.transport.requests

        self.creds.refresh(google.auth.transport.requests.Request())  # type: ignore[no-untyped-call]

        return {"bearer_token": self.creds.token}, (
            int(
                expiry.replace(
                    # Google auth does not set this properly
                    tzinfo=zoneinfo.ZoneInfo("UTC")
                ).timestamp()
            )
            if (expiry := self.creds.expiry) is not None
            else None
        )

    @classmethod
    def _check_module_availability(cls) -> None:
        if importlib.util.find_spec("google.auth") is None:
            msg = "google-auth must be installed to use `CredentialProviderGCP`"
            raise ImportError(msg)


def _auto_select_credential_provider(
    source: ScanSource,
) -> CredentialProvider | None:
    from polars.io.cloud._utils import _infer_cloud_type

    verbose = os.getenv("POLARS_VERBOSE") == "1"
    cloud_type = _infer_cloud_type(source)

    provider = None

    try:
        provider = (
            None
            if cloud_type is None
            else CredentialProviderAWS()
            if cloud_type == "aws"
            else CredentialProviderGCP()
            if cloud_type == "gcp"
            else None
        )
    except ImportError as e:
        if verbose:
            msg = f"Unable to auto-select credential provider: {e}"
            print(msg, file=sys.stderr)

    if provider is not None and verbose:
        msg = f"Auto-selected credential provider: {type(provider).__name__}"
        print(msg, file=sys.stderr)

    return provider
