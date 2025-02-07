from __future__ import annotations

import abc
import importlib.util
import json
import os
import subprocess
import sys
import zoneinfo
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, TypedDict, Union

from polars._utils.various import issue_warning

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

from polars._utils.unstable import issue_unstable_warning

# These typedefs are here to avoid circular import issues, as
# `CredentialProviderFunction` specifies "CredentialProvider"
CredentialProviderFunctionReturn: TypeAlias = tuple[dict[str, str], Optional[int]]

CredentialProviderFunction: TypeAlias = Union[
    Callable[[], CredentialProviderFunctionReturn], "CredentialProvider"
]

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


class AWSAssumeRoleKWArgs(TypedDict):
    """Parameters for [STS.Client.assume_role()](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sts/client/assume_role.html#STS.Client.assume_role)."""

    RoleArn: str
    RoleSessionName: str
    PolicyArns: list[dict[str, str]]
    Policy: str
    DurationSeconds: int
    Tags: list[dict[str, str]]
    TransitiveTagKeys: list[str]
    ExternalId: str
    SerialNumber: str
    TokenCode: str
    SourceIdentity: str
    ProvidedContexts: list[dict[str, str]]


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

    def __init__(
        self,
        *,
        profile_name: str | None = None,
        region_name: str | None = None,
        assume_role: AWSAssumeRoleKWArgs | None = None,
    ) -> None:
        """
        Initialize a credential provider for AWS.

        Parameters
        ----------
        profile_name : str
            Profile name to use from credentials file.
        assume_role : AWSAssumeRoleKWArgs | None
            Configure a role to assume. These are passed as kwarg parameters to
            [STS.client.assume_role()](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sts/client/assume_role.html#STS.Client.assume_role)
        """
        msg = "`CredentialProviderAWS` functionality is considered unstable"
        issue_unstable_warning(msg)

        self._ensure_module_availability()
        self.profile_name = profile_name
        self.region_name = region_name
        self.assume_role = assume_role

    def __call__(self) -> CredentialProviderFunctionReturn:
        """Fetch the credentials for the configured profile name."""
        import boto3

        # Note: boto3 automatically sources the AWS_PROFILE env var
        session = boto3.Session(
            profile_name=self.profile_name,
            region_name=self.region_name,
        )

        if self.assume_role is not None:
            return self._finish_assume_role(session)

        creds = session.get_credentials()

        if creds is None:
            msg = "CredentialProviderAWS: unexpected None value returned from boto3.Session.get_credentials()"
            raise ValueError(msg)

        return {
            "aws_access_key_id": creds.access_key,
            "aws_secret_access_key": creds.secret_key,
            **({"aws_session_token": creds.token} if creds.token is not None else {}),
        }, None

    def _finish_assume_role(self, session: Any) -> CredentialProviderFunctionReturn:
        client = session.client("sts")

        sts_response = client.assume_role(**self.assume_role)
        creds = sts_response["Credentials"]

        expiry = creds["Expiration"]

        if expiry.tzinfo is None:
            msg = "expiration time in STS response did not contain timezone information"
            raise ValueError(msg)

        return {
            "aws_access_key_id": creds["AccessKeyId"],
            "aws_secret_access_key": creds["SecretAccessKey"],
            "aws_session_token": creds["SessionToken"],
        }, int(expiry.timestamp())

    @classmethod
    def _ensure_module_availability(cls) -> None:
        if importlib.util.find_spec("boto3") is None:
            msg = "boto3 must be installed to use `CredentialProviderAWS`"
            raise ImportError(msg)


class CredentialProviderAzure(CredentialProvider):
    """
    Azure Credential Provider.

    Using this requires the `azure-identity` Python package to be installed.

    .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
    """

    def __init__(
        self,
        *,
        scopes: list[str] | None = None,
        tenant_id: str | None = None,
        credentials: Any | None = None,
        _storage_account: str | None = None,
    ) -> None:
        """
        Initialize a credential provider for Microsoft Azure.

        By default, this uses `azure.identity.DefaultAzureCredential()`.

        Parameters
        ----------
        scopes
            Scopes to pass to `get_token`
        tenant_id
            Azure tenant ID.
        credentials
            Optionally pass an instantiated Azure credential class to use (e.g.
            `azure.identity.DefaultAzureCredential`). The credential class must
            have a `get_token()` method.
        """
        msg = "`CredentialProviderAzure` functionality is considered unstable"
        issue_unstable_warning(msg)

        self.account_name = _storage_account
        self.scopes = (
            scopes if scopes is not None else ["https://storage.azure.com/.default"]
        )
        self.tenant_id = tenant_id
        self.credentials = credentials

        if credentials is not None:
            # If the user passes a credentials class, we just need to ensure it
            # has a `get_token()` method.
            if not hasattr(credentials, "get_token"):
                msg = (
                    f"the provided `credentials` object {credentials!r} does "
                    "not have a `get_token()` method."
                )
                raise ValueError(msg)

        # We don't need the module if we are permitted and able to retrieve the
        # account key from the Azure CLI.
        elif self._try_get_azure_storage_account_credentials_if_permitted() is None:
            self._ensure_module_availability()

        if os.getenv("POLARS_VERBOSE") == "1":
            print(
                (
                    "[CredentialProviderAzure]: "
                    f"{self.account_name = } "
                    f"{self.tenant_id = } "
                    f"{self.scopes = } "
                ),
                file=sys.stderr,
            )

    def __call__(self) -> CredentialProviderFunctionReturn:
        """Fetch the credentials."""
        if (
            v := self._try_get_azure_storage_account_credentials_if_permitted()
        ) is not None:
            return v

        # Done like this to bypass mypy, we don't have stubs for azure.identity
        credential = (
            self.credentials
            or importlib.import_module("azure.identity").__dict__[
                "DefaultAzureCredential"
            ]()
        )
        token = credential.get_token(*self.scopes, tenant_id=self.tenant_id)

        return {
            "bearer_token": token.token,
        }, token.expires_on

    def _try_get_azure_storage_account_credentials_if_permitted(
        self,
    ) -> CredentialProviderFunctionReturn | None:
        POLARS_AUTO_USE_AZURE_STORAGE_ACCOUNT_KEY = os.getenv(
            "POLARS_AUTO_USE_AZURE_STORAGE_ACCOUNT_KEY"
        )

        verbose = os.getenv("POLARS_VERBOSE") == "1"

        if verbose:
            print(
                "[CredentialProviderAzure]: "
                f"{self.account_name = } "
                f"{POLARS_AUTO_USE_AZURE_STORAGE_ACCOUNT_KEY = }",
                file=sys.stderr,
            )

        if (
            self.account_name is not None
            and POLARS_AUTO_USE_AZURE_STORAGE_ACCOUNT_KEY == "1"
        ):
            try:
                creds = {
                    "account_key": self._get_azure_storage_account_key_az_cli(
                        self.account_name
                    )
                }

                if verbose:
                    print(
                        "[CredentialProviderAzure]: Retrieved account key from Azure CLI",
                        file=sys.stderr,
                    )
            except Exception as e:
                if verbose:
                    print(
                        f"[CredentialProviderAzure]: Could not retrieve account key from Azure CLI: {e}",
                        file=sys.stderr,
                    )
            else:
                return creds, None

        return None

    @classmethod
    def _ensure_module_availability(cls) -> None:
        if importlib.util.find_spec("azure.identity") is None:
            msg = "azure-identity must be installed to use `CredentialProviderAzure`"
            raise ImportError(msg)

    @staticmethod
    def _extract_adls_uri_storage_account(uri: str) -> str | None:
        # "abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net/"
        #                      ^^^^^^^^^^^^^^^^^
        try:
            return (
                uri.split("://", 1)[1]
                .split("/", 1)[0]
                .split("@", 1)[1]
                .split(".dfs.core.windows.net", 1)[0]
            )

        except IndexError:
            return None

    @classmethod
    def _get_azure_storage_account_key_az_cli(cls, account_name: str) -> str:
        # [
        #     {
        #         "creationTime": "1970-01-01T00:00:00.000000+00:00",
        #         "keyName": "key1",
        #         "permissions": "FULL",
        #         "value": "..."
        #     },
        #     {
        #         "creationTime": "1970-01-01T00:00:00.000000+00:00",
        #         "keyName": "key2",
        #         "permissions": "FULL",
        #         "value": "..."
        #     }
        # ]

        return json.loads(
            cls._azcli(
                "storage",
                "account",
                "keys",
                "list",
                "--output",
                "json",
                "--account-name",
                account_name,
            )
        )[0]["value"]

    @classmethod
    def _azcli_version(cls) -> str | None:
        try:
            return json.loads(cls._azcli("version"))["azure-cli"]
        except Exception:
            return None

    @staticmethod
    def _azcli(*args: str) -> bytes:
        return subprocess.check_output(
            ["az", *args] if sys.platform != "win32" else ["cmd", "/C", "az", *args]
        )


class CredentialProviderGCP(CredentialProvider):
    """
    GCP Credential Provider.

    Using this requires the `google-auth` Python package to be installed.

    .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
    """

    def __init__(
        self,
        *,
        scopes: Any | None = None,
        request: Any | None = None,
        quota_project_id: Any | None = None,
        default_scopes: Any | None = None,
    ) -> None:
        """
        Initialize a credential provider for Google Cloud (GCP).

        Parameters
        ----------
        Parameters are passed to `google.auth.default()`
        """
        msg = "`CredentialProviderGCP` functionality is considered unstable"
        issue_unstable_warning(msg)

        self._ensure_module_availability()

        import google.auth
        import google.auth.credentials

        # CI runs with both `mypy` and `mypy --allow-untyped-calls` depending on
        # Python version. If we add a `type: ignore[no-untyped-call]`, then the
        # check that runs with `--allow-untyped-calls` will complain about an
        # unused "type: ignore" comment. And if we don't add the ignore, then
        # he check that runs `mypy` will complain.
        #
        # So we just bypass it with a __dict__[] (because ruff complains about
        # getattr) :|
        creds, _ = google.auth.__dict__["default"](
            scopes=(
                scopes
                if scopes is not None
                else ["https://www.googleapis.com/auth/cloud-platform"]
            ),
            request=request,
            quota_project_id=quota_project_id,
            default_scopes=default_scopes,
        )
        self.creds = creds

    def __call__(self) -> CredentialProviderFunctionReturn:
        """Fetch the credentials."""
        import google.auth.transport.requests

        self.creds.refresh(google.auth.transport.requests.__dict__["Request"]())

        return {"bearer_token": self.creds.token}, (
            int(
                (
                    expiry.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
                    if expiry.tzinfo is None
                    else expiry
                ).timestamp()
            )
            if (expiry := self.creds.expiry) is not None
            else None
        )

    @classmethod
    def _ensure_module_availability(cls) -> None:
        if importlib.util.find_spec("google.auth") is None:
            msg = "google-auth must be installed to use `CredentialProviderGCP`"
            raise ImportError(msg)


def _maybe_init_credential_provider(
    credential_provider: CredentialProviderFunction | Literal["auto"] | None,
    source: Any,
    storage_options: dict[str, Any] | None,
    caller_name: str,
) -> CredentialProviderFunction | CredentialProvider | None:
    from polars.io.cloud._utils import (
        _first_scan_path,
        _get_path_scheme,
        _is_aws_cloud,
        _is_azure_cloud,
        _is_gcp_cloud,
    )

    if credential_provider is not None:
        msg = f"The `credential_provider` parameter of `{caller_name}` is considered unstable."
        issue_unstable_warning(msg)

    if credential_provider != "auto":
        return credential_provider

    verbose = os.getenv("POLARS_VERBOSE") == "1"

    if (path := _first_scan_path(source)) is None:
        return None

    if (scheme := _get_path_scheme(path)) is None:
        return None

    provider: (
        CredentialProviderAWS | CredentialProviderAzure | CredentialProviderGCP | None
    ) = None

    try:
        # For Azure we dispatch to `azure.identity` as much as possible
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

            provider = CredentialProviderAzure(
                tenant_id=tenant_id,
                _storage_account=storage_account,
            )
        elif _is_aws_cloud(scheme):
            region = None
            profile = None
            default_region = None
            unhandled_key = None

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
                    elif k in OBJECT_STORE_CLIENT_OPTIONS:
                        continue
                    else:
                        # We assume some sort of access key was given, so we
                        # just dispatch to the rust side.
                        unhandled_key = k

            to_silence_this_warning = (
                "To silence this warning, pass 'aws_profile': None in storage_options."
            )

            if unhandled_key is not None:
                if profile is not None:
                    msg = (
                        f"the configured AWS profile '{profile}' may be ignored "
                        "as it is not compatible with the provided "
                        f"storage_option key '{unhandled_key}'. "
                        f"{to_silence_this_warning}"
                    )
                    issue_warning(msg, UserWarning)

                return None

            try:
                provider = CredentialProviderAWS(
                    profile_name=profile, region_name=region or default_region
                )
            except ImportError:
                if profile is not None:
                    msg = (
                        f"the configured AWS profile '{profile}' may not "
                        "be used as boto3 is not installed. "
                        f"{to_silence_this_warning}"
                    )
                    # Conservatively warn instead of hard error. It could just be
                    # set as a default environment flag.
                    issue_warning(msg, UserWarning)
                # Note: Enclosing scope will catch ImportErrors
                raise

        elif storage_options is not None and any(
            key.lower() not in OBJECT_STORE_CLIENT_OPTIONS for key in storage_options
        ):
            return None
        elif _is_gcp_cloud(scheme):
            provider = CredentialProviderGCP()

    except ImportError as e:
        if verbose:
            msg = f"unable to auto-select credential provider: {e!r}"
            print(msg, file=sys.stderr)

    if provider is not None:
        # CredentialProviderAWS raises an error in some cases when
        # `get_credentials()` returns None (e.g. the environment may not
        # have / require credentials). We check this here and avoid
        # using it if that is the case.
        try:
            provider()
        except Exception as e:
            provider = None

            if verbose:
                msg = f"unable to auto-select credential provider: {e!r}"
                print(msg, file=sys.stderr)

    if provider is not None and verbose:
        msg = f"auto-selected credential provider: {type(provider).__name__}"
        print(msg, file=sys.stderr)

    return provider


def _get_credentials_from_provider_expiry_aware(
    credential_provider: CredentialProviderFunction,
) -> dict[str, str]:
    creds, opt_expiry = credential_provider()

    if (
        opt_expiry is not None
        and (expires_in := opt_expiry - int(datetime.now().timestamp())) < 7
    ):
        import os
        import sys
        from time import sleep

        if os.getenv("POLARS_VERBOSE") == "1":
            print(
                f"waiting for {expires_in} seconds for refreshed credentials",
                file=sys.stderr,
            )

        sleep(1 + expires_in)
        creds, _ = credential_provider()

    return creds
