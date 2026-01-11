use std::path::Path;
use std::sync::Arc;

use crate::pl_path::PlRefPath;

/// A Path or URI
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub(super) enum PlPath {
    Local(Arc<Path>),
    Cloud(PlCloudPath),
}

impl From<PlRefPath> for PlPath {
    fn from(path: PlRefPath) -> Self {
        if let Some(scheme) = path.scheme() {
            Self::Cloud(PlCloudPath {
                scheme: CloudScheme::from_scheme_str(scheme.as_str()).unwrap(),
                uri: path.into_ref_str().into_arc_str(),
            })
        } else {
            Self::Local(Arc::<Path>::from(path.as_std_path()))
        }
    }
}

impl From<PlPath> for PlRefPath {
    fn from(path: PlPath) -> Self {
        match path {
            PlPath::Local(path) => PlRefPath::try_from_path(path.as_ref()).unwrap(),
            PlPath::Cloud(PlCloudPath { scheme: _, uri }) => PlRefPath::new(uri),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub(super) struct PlCloudPath {
    /// The scheme used in cloud e.g. `s3://` or `file://`.
    scheme: CloudScheme,
    /// The full URI e.g. `s3://path/to/bucket`.
    uri: Arc<str>,
}

macro_rules! impl_cloud_scheme {
    ($($t:ident = $n:literal,)+) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
        pub(super) enum CloudScheme {
            $($t,)+
        }

        impl CloudScheme {
            #[expect(unreachable_patterns)]
            fn from_scheme_str(s: &str) -> Option<Self> {
                Some(match s {
                    $($n => Self::$t,)+
                    _ => return None,
                })
            }
        }
    };
}

impl_cloud_scheme! {
    Abfs = "abfs",
    Abfss = "abfss",
    Adl = "adl",
    Az = "az",
    Azure = "azure",
    File = "file",
    FileNoHostname = "file",
    Gcs = "gcs",
    Gs = "gs",
    Hf = "hf",
    Http = "http",
    Https = "https",
    S3 = "s3",
    S3a = "s3a",
}
