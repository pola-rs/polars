//! The selected profile's keys, region and endpoint from `~/.aws/config` and
//! `~/.aws/credentials`, for when no credential provider gives credentials.

use std::path::Path;

use object_store::aws::AwsCredential;
use polars_error::{PolarsError, polars_err};
use polars_utils::aliases::PlHashMap;

use crate::path_utils::resolve_homedir;

/// The selected profile and its values.
pub(crate) struct AwsProfile {
    /// `default` if no profile was named.
    pub(crate) name: String,
    pub(crate) named: bool,
    /// The files read, as configured.
    pub(crate) files: [String; 2],
    pub(crate) values: AwsProfileValues,
}

/// The selected profile (`requested`, else `AWS_PROFILE`, else `AWS_DEFAULT_PROFILE`, else
/// `default`) and its values from the files the AWS CLI reads.
pub(crate) fn load_aws_profile(requested: Option<&str>) -> AwsProfile {
    let env = |var: &str| std::env::var(var).ok().filter(|v| !v.is_empty());
    let named = requested
        .map(str::to_string)
        .or_else(|| env("AWS_PROFILE"))
        .or_else(|| env("AWS_DEFAULT_PROFILE"));
    let files = [
        env("AWS_CONFIG_FILE").unwrap_or_else(|| "~/.aws/config".to_string()),
        env("AWS_SHARED_CREDENTIALS_FILE").unwrap_or_else(|| "~/.aws/credentials".to_string()),
    ];
    let read = |path: &str| {
        let bytes = std::fs::read(resolve_homedir(Path::new(path))).ok()?;
        Some(String::from_utf8_lossy(&bytes).into_owned())
    };
    let name = named.clone().unwrap_or_else(|| "default".to_string());
    let values = aws_profile_values(
        &name,
        read(&files[0]).as_deref(),
        read(&files[1]).as_deref(),
    );
    AwsProfile {
        name,
        named: named.is_some(),
        files,
        values,
    }
}

/// Values of one profile.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct AwsProfileValues {
    /// Whether either file has a section for the profile.
    pub(crate) defined: bool,
    pub(crate) access_key_id: Option<String>,
    pub(crate) secret_access_key: Option<String>,
    pub(crate) session_token: Option<String>,
    pub(crate) region: Option<String>,
    pub(crate) endpoint_url: Option<String>,
    pub(crate) credential_source: Option<CredentialSource>,
}

/// A source of credentials in a profile, other than keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CredentialSource {
    Role,
    RoleWithMfa,
    WebIdentity,
    Sso,
    Login,
    Process,
}

impl CredentialSource {
    fn describe(self) -> &'static str {
        match self {
            Self::Role => "`role_arn`",
            Self::RoleWithMfa => "a role that needs an MFA code",
            Self::WebIdentity => "`web_identity_token_file`",
            Self::Sso => "SSO",
            Self::Login => "`login_session`",
            Self::Process => "`credential_process`",
        }
    }
}

impl AwsProfileValues {
    pub(crate) fn credential(&self) -> Option<AwsCredential> {
        Some(AwsCredential {
            key_id: self.access_key_id.clone()?,
            secret_key: self.secret_access_key.clone()?,
            token: self.session_token.clone(),
        })
    }
}

/// The values of `profile`; the credentials file overrides the config file. Credentials come
/// from the first source in botocore's order: assume-role, web identity, SSO, keys in the
/// credentials file, `login_session`, `credential_process`, keys in the config file.
pub(crate) fn aws_profile_values(
    profile: &str,
    config: Option<&str>,
    credentials: Option<&str>,
) -> AwsProfileValues {
    let config = config.and_then(|c| section(c, profile, true));
    let credentials = credentials.and_then(|c| section(c, profile, false));
    let defined = config.is_some() || credentials.is_some();
    let mut settings = config.clone().unwrap_or_default();
    settings.extend(credentials.clone().unwrap_or_default());

    let has = |k: &str| settings.contains_key(k);
    let in_credentials = credentials.filter(|c| c.contains_key("aws_access_key_id"));
    let credential_source = if has("role_arn") && has("web_identity_token_file") {
        Some(CredentialSource::WebIdentity)
    } else if has("role_arn") && has("mfa_serial") {
        Some(CredentialSource::RoleWithMfa)
    } else if has("role_arn") {
        Some(CredentialSource::Role)
    } else if has("sso_account_id") || has("sso_role_name") {
        Some(CredentialSource::Sso)
    } else if in_credentials.is_some() {
        None
    } else if has("login_session") {
        Some(CredentialSource::Login)
    } else if has("credential_process") {
        Some(CredentialSource::Process)
    } else {
        None
    };
    // The keys and token come from one file.
    let keys = in_credentials
        .or(config)
        .filter(|_| credential_source.is_none())
        .unwrap_or_default();
    let get = |s: &PlHashMap<String, String>, k: &str| s.get(k).filter(|v| !v.is_empty()).cloned();
    AwsProfileValues {
        defined,
        access_key_id: get(&keys, "aws_access_key_id"),
        secret_access_key: get(&keys, "aws_secret_access_key"),
        session_token: get(&keys, "aws_security_token").or_else(|| get(&keys, "aws_session_token")),
        region: get(&settings, "region"),
        endpoint_url: get(&settings, "endpoint_url"),
        credential_source,
    }
}

impl AwsProfile {
    /// The profile's credential source, unless web identity from the environment comes first.
    pub(crate) fn source(&self, web_identity: bool) -> Option<CredentialSource> {
        (self.values.credential_source).filter(|s| {
            !web_identity || matches!(s, CredentialSource::Role | CredentialSource::RoleWithMfa)
        })
    }

    /// Why the profile's credentials cannot be read here, if so.
    pub(crate) fn error(&self, web_identity: bool) -> Option<PolarsError> {
        let profile = &self.name;
        let values = &self.values;
        if self.named && !values.defined {
            let [config, credentials] = &self.files;
            return Some(polars_err!(
                ComputeError: "AWS profile '{profile}' is not defined in {config} or {credentials}"
            ));
        }
        if let Some(source) = self.source(web_identity) {
            let source = source.describe();
            return Some(polars_err!(
                ComputeError: "AWS profile '{profile}' gets its credentials through {source}, \
                which Polars reads only through boto3 in Python; set the credentials in the \
                environment instead: \
                eval \"$(aws configure export-credentials --profile {profile} --format env)\""
            ));
        }
        if web_identity {
            return None;
        }
        match (&values.access_key_id, &values.secret_access_key) {
            (Some(_), None) => Some(polars_err!(
                ComputeError: "AWS profile '{profile}' has aws_access_key_id but no \
                aws_secret_access_key"
            )),
            _ => None,
        }
    }
}

/// The settings of the profile's section: `[profile name]` or `[default]` in the config file,
/// `[name]` in the credentials file. The last such section wins. Keys are case-insensitive and
/// end at `=` or `:`; a line indented deeper than the key before it continues that key's value or
/// nests under it, and is skipped.
fn section(content: &str, profile: &str, config_file: bool) -> Option<PlHashMap<String, String>> {
    let wanted = |header: &str| match header.split_once(char::is_whitespace) {
        Some(("profile", name)) if config_file => name.trim().trim_matches(['"', '\'']) == profile,
        _ => (!config_file || header == "default") && header == profile,
    };
    let mut out = None;
    let mut inside = false;
    let mut key_indent: Option<usize> = None;
    for line in content.trim_start_matches('\u{feff}').lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if key_indent.is_some_and(|key_indent| indent > key_indent) {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('[') {
            // A comment may follow the header.
            inside = rest.split_once(']').is_some_and(|(h, _)| wanted(h.trim()));
            if inside {
                out = Some(PlHashMap::default());
            }
            key_indent = None;
            continue;
        }
        let Some(at) = trimmed.find(['=', ':']) else {
            continue;
        };
        key_indent = Some(indent);
        if let (true, Some(out)) = (inside, out.as_mut()) {
            let (key, value) = trimmed.split_at(at);
            out.insert(
                key.trim().to_ascii_lowercase(),
                value[1..].trim().to_string(),
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = "
[default]
region = eu-west-1

[profile prod]
region = eu-central-1
s3 =
  region = us-west-1

[profile sso]
sso_session = corp
sso_account_id = 1
sso_role_name = reader
region = us-east-2
";
    const CREDENTIALS: &str = "
[dev] # the dev account
aws_access_key_id = DEV
aws_secret_access_key = dev-secret
aws_session_token = dev-token

[default]
aws_access_key_id = DEFAULT
aws_secret_access_key = default-secret

# comment
[prod]
aws_access_key_id = PROD
aws_secret_access_key = prod-secret
";

    fn keys(v: &AwsProfileValues) -> (Option<&str>, Option<&str>, Option<&str>, Option<&str>) {
        (
            v.access_key_id.as_deref(),
            v.secret_access_key.as_deref(),
            v.session_token.as_deref(),
            v.region.as_deref(),
        )
    }

    #[test]
    fn reads_only_the_selected_profile() {
        let read = |p| aws_profile_values(p, Some(CONFIG), Some(CREDENTIALS));
        let expected = [
            (
                "default",
                (
                    Some("DEFAULT"),
                    Some("default-secret"),
                    None,
                    Some("eu-west-1"),
                ),
            ),
            (
                "prod",
                (
                    Some("PROD"),
                    Some("prod-secret"),
                    None,
                    Some("eu-central-1"),
                ),
            ),
            (
                "dev",
                (Some("DEV"), Some("dev-secret"), Some("dev-token"), None),
            ),
        ];
        for (profile, values) in expected {
            assert_eq!(keys(&read(profile)), values, "{profile}");
        }
    }

    #[test]
    fn profiles_that_cannot_be_read() {
        let profile = |name: &str, named| AwsProfile {
            name: name.to_string(),
            named,
            files: Default::default(),
            values: aws_profile_values(name, Some(CONFIG), Some(CREDENTIALS)),
        };
        assert!(profile("missing", true).error(false).is_some());
        assert!(profile("missing", false).error(false).is_none());
        let sso = profile("sso", true);
        assert_eq!(keys(&sso.values), (None, None, None, Some("us-east-2")));
        assert!(sso.error(false).is_some());
        // Web identity from the environment comes first.
        assert!(sso.error(true).is_none());

        // `credential_process` comes after keys in the credentials file, before those in the
        // config file.
        let config = "[profile p]\ncredential_process = x\naws_access_key_id = C\naws_secret_access_key = s\n";
        let credentials = "[p]\naws_access_key_id = K\naws_secret_access_key = s\n";
        let source =
            |credentials| aws_profile_values("p", Some(config), credentials).credential_source;
        assert_eq!(source(None), Some(CredentialSource::Process));
        assert_eq!(source(Some(credentials)), None);
        // `sso_session` alone configures a bearer token, not credentials.
        let config =
            "[profile b]\nsso_session = corp\naws_access_key_id = K\naws_secret_access_key = s\n";
        let v = aws_profile_values("b", Some(config), None);
        assert_eq!(v.access_key_id.as_deref(), Some("K"));
    }

    #[test]
    fn file_layouts() {
        let config = "[profile \"cfg\"]\n  aws_access_key_id = CFG\n  aws_secret_access_key = s\n";
        let v = aws_profile_values("cfg", Some(config), None);
        assert_eq!(v.access_key_id.as_deref(), Some("CFG"));
        // The later section for the same profile replaces the earlier one.
        let config =
            "[default]\naws_session_token = OLD\n[profile default]\naws_access_key_id = NEW\n";
        let v = aws_profile_values("default", Some(config), None);
        assert_eq!(
            (v.access_key_id.as_deref(), v.session_token),
            (Some("NEW"), None)
        );
        // The credentials file overrides the config file.
        let v = aws_profile_values(
            "x",
            Some("[profile x]\nregion = a\n"),
            Some("[x]\nregion = b\n"),
        );
        assert_eq!(v.region.as_deref(), Some("b"));
        // Any case, `:`, and a deeper line continues the value.
        let v = aws_profile_values(
            "x",
            None,
            Some("[x]\nAWS_ACCESS_KEY_ID : K\n  aws_access_key_id = L\n"),
        );
        assert_eq!(v.access_key_id.as_deref(), Some("K"));
        // Even when it looks like a header.
        let config = "[profile x]\naws_access_key_id = A\n[profile o]\nnote = n\n  [profile x]\n  aws_access_key_id = B\n";
        let v = aws_profile_values("x", Some(config), None);
        assert_eq!(v.access_key_id.as_deref(), Some("A"));
        // The keys and token come from one file.
        let v = aws_profile_values(
            "x",
            Some(
                "[profile x]\naws_access_key_id = A\naws_secret_access_key = a\naws_session_token = T\n",
            ),
            Some("[x]\naws_access_key_id = B\naws_secret_access_key = b\n"),
        );
        assert_eq!(
            (v.access_key_id.as_deref(), v.session_token),
            (Some("B"), None)
        );
    }
}
