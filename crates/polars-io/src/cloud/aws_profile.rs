//! The selected profile's keys and region from `~/.aws/config` and `~/.aws/credentials`, for when
//! no credential provider gives credentials.

use std::path::Path;

use polars_error::{PolarsError, polars_err};
use polars_utils::aliases::PlHashMap;

use crate::path_utils::resolve_homedir;

/// The selected profile and its values.
pub(crate) struct AwsProfile {
    /// `default` if no profile was named.
    pub(crate) name: String,
    pub(crate) named: bool,
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
    let read = |var: &str, default: &str| {
        let path = env(var);
        let path = path.as_deref().unwrap_or(default);
        std::fs::read_to_string(resolve_homedir(Path::new(path))).ok()
    };
    let name = named.clone().unwrap_or_else(|| "default".to_string());
    let values = aws_profile_values(
        &name,
        read("AWS_CONFIG_FILE", "~/.aws/config").as_deref(),
        read("AWS_SHARED_CREDENTIALS_FILE", "~/.aws/credentials").as_deref(),
    );
    AwsProfile {
        name,
        named: named.is_some(),
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
    /// Where the profile gets credentials that are not read here, such as `sso_session`.
    pub(crate) credential_source: Option<&'static str>,
}

/// Resolved before keys in the same profile.
const BEFORE_KEYS: [&str; 4] = ["role_arn", "sso_session", "sso_start_url", "login_session"];
/// Resolved when the profile has no keys.
const WITHOUT_KEYS: [&str; 1] = ["credential_process"];

/// The values of `profile`; the credentials file overrides the config file.
pub(crate) fn aws_profile_values(
    profile: &str,
    config: Option<&str>,
    credentials: Option<&str>,
) -> AwsProfileValues {
    let config = config.and_then(|c| section(c, profile, true));
    let credentials = credentials.and_then(|c| section(c, profile, false));
    let defined = config.is_some() || credentials.is_some();
    let mut settings = config.unwrap_or_default();
    settings.extend(credentials.unwrap_or_default());

    let has_keys = settings.contains_key("aws_access_key_id");
    let credential_source = BEFORE_KEYS
        .into_iter()
        .chain(WITHOUT_KEYS.into_iter().filter(|_| !has_keys))
        .find(|k| settings.contains_key(*k));
    let key = |k: &str| match credential_source {
        Some(_) => None,
        None => settings.get(k).cloned(),
    };
    AwsProfileValues {
        defined,
        access_key_id: key("aws_access_key_id"),
        secret_access_key: key("aws_secret_access_key"),
        session_token: key("aws_session_token"),
        region: settings.get("region").filter(|r| !r.is_empty()).cloned(),
        credential_source,
    }
}

impl AwsProfile {
    /// Why the profile's credentials cannot be read here, if so.
    pub(crate) fn error(&self) -> Option<PolarsError> {
        let profile = &self.name;
        if self.named && !self.values.defined {
            return Some(polars_err!(
                ComputeError: "AWS profile '{profile}' is not defined in the AWS config or credentials file"
            ));
        }
        self.values.credential_source.map(|source| {
            polars_err!(
                ComputeError: "AWS profile '{profile}' gets its credentials through `{source}`, \
                which Polars reads only with boto3 in Python; install boto3, or set the \
                credentials in the environment: \
                eval \"$(aws configure export-credentials --profile {profile} --format env)\""
            )
        })
    }
}

/// The settings of the profile's section: `[profile name]` or `[default]` in the config file,
/// `[name]` in the credentials file. The last such section wins; nested settings are skipped.
fn section(content: &str, profile: &str, config_file: bool) -> Option<PlHashMap<String, String>> {
    let wanted = |header: &str| {
        let name = match header.split_once(char::is_whitespace) {
            Some(("profile", name)) if config_file => name,
            _ if config_file && header != "default" => return false,
            _ => header,
        };
        name.trim().trim_matches('"') == profile
    };
    let mut out = None;
    let mut inside = false;
    let mut nested_below: Option<usize> = None;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('[') {
            // A comment may follow the header.
            inside = rest.split_once(']').is_some_and(|(h, _)| wanted(h.trim()));
            if inside {
                out = Some(PlHashMap::default());
            }
            nested_below = None;
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if nested_below.is_some_and(|key_indent| indent > key_indent) {
            continue;
        }
        nested_below = None;
        if let (true, Some(out), Some((key, value))) =
            (inside, out.as_mut(), trimmed.split_once('='))
        {
            let (key, value) = (key.trim(), value.trim());
            if value.is_empty() {
                nested_below = Some(indent);
            }
            out.insert(key.to_string(), value.to_string());
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
            values: aws_profile_values(name, Some(CONFIG), Some(CREDENTIALS)),
        };
        assert!(profile("missing", true).error().is_some());
        assert!(profile("missing", false).error().is_none());
        let sso = profile("sso", true);
        assert_eq!(keys(&sso.values), (None, None, None, Some("us-east-2")));
        assert!(sso.error().is_some());
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
    }
}
