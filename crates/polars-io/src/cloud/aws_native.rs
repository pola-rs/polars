//! Credentials of AWS profiles with a role, web identity, SSO or `credential_process`.

use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

use aws_config::Region;
use aws_config::profile::ProfileFileCredentialsProvider;
use aws_config::provider_config::ProviderConfig;
use aws_credential_types::provider::error::CredentialsError;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_runtime::env_config::file::{EnvConfigFileKind, EnvConfigFiles};
use polars_core::runtime::ASYNC;
use polars_error::{PolarsError, PolarsResult, polars_err};

use super::aws_profile::{AwsProfile, CredentialSource};
use super::credential_provider::{ObjectStoreCredential, PlCredentialProvider};
use crate::path_utils::resolve_homedir;

pub(crate) fn resolves(source: CredentialSource) -> bool {
    !matches!(
        source,
        CredentialSource::Login | CredentialSource::RoleWithMfa
    )
}

type Fetched = (ObjectStoreCredential, u64);

/// Fetches once now, so that an unreadable profile fails here, not in the first request.
pub(crate) async fn profile_provider(
    profile: &AwsProfile,
    source: CredentialSource,
    region: Option<String>,
) -> PolarsResult<PlCredentialProvider> {
    // The files Polars read.
    let [config, credentials] = &profile.files;
    let files = EnvConfigFiles::builder()
        .with_file(EnvConfigFileKind::Config, resolve_homedir(config))
        .with_file(EnvConfigFileKind::Credentials, resolve_homedir(credentials))
        .build();
    // For STS; the AWS CLI's default, the global endpoint, is in us-east-1.
    let region = Region::new(region.unwrap_or_else(|| "us-east-1".to_string()));
    let provider = SharedCredentialsProvider::new(
        ProfileFileCredentialsProvider::builder()
            .configure(&ProviderConfig::default().with_region(Some(region)))
            .profile_files(files)
            .profile_name(&profile.name)
            .build(),
    );
    let sso = source == CredentialSource::Sso;
    let profile = profile.name.clone();
    let first = fetch(provider.clone(), profile.clone(), sso).await?;
    let first = Arc::new(Mutex::new(Some(first)));
    Ok(PlCredentialProvider::from_func(move || {
        let (provider, profile, first) = (provider.clone(), profile.clone(), first.clone());
        Box::pin(async move {
            if let Some(fetched) = first.lock().unwrap().take() {
                return Ok(fetched);
            }
            fetch(provider, profile, sso).await
        })
    }))
}

async fn fetch(
    provider: SharedCredentialsProvider,
    profile: String,
    sso: bool,
) -> PolarsResult<Fetched> {
    // The provider's future is not `Sync`.
    let c = ASYNC
        .spawn(async move { provider.provide_credentials().await })
        .await
        .map_err(|e| polars_err!(ComputeError: "loading AWS credentials failed: {e}"))?
        .map_err(|e| to_polars_error(&e, &profile, sso))?;
    let expiry = c
        .expiry()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(u64::MAX, |d| d.as_secs());
    let credential = object_store::aws::AwsCredential {
        key_id: c.access_key_id().to_string(),
        secret_key: c.secret_access_key().to_string(),
        token: c.session_token().map(str::to_string),
    };
    Ok((ObjectStoreCredential::Aws(Arc::new(credential)), expiry))
}

/// With its causes, such as a failing `credential_process`'s stderr, as in the AWS CLI.
fn to_polars_error(err: &CredentialsError, profile: &str, sso: bool) -> PolarsError {
    let mut causes = vec![err.to_string()];
    let mut source = std::error::Error::source(err);
    while let Some(e) = source {
        let cause = e.to_string();
        if causes.last() != Some(&cause) {
            causes.push(cause);
        }
        source = e.source();
    }
    // Said of every failure.
    while causes.len() > 1 && causes[0] == "an error occurred while loading credentials" {
        causes.remove(0);
    }
    let message = causes.join(": ");
    let hint = match sso {
        true => format!("; run `aws sso login --profile {profile}`"),
        false => String::new(),
    };
    polars_err!(ComputeError: "cannot load AWS profile '{profile}': {message}{hint}")
}
