use std::sync::Arc;
use std::time::Duration;

use object_store::{Certificate, ClientOptions};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PlClientOptions {
    timeout: Option<Duration>,
    connect_timeout: Option<Duration>,
    allow_http: bool,
    root_certificates: RootCertificates,
}

#[derive(Debug, Clone)]
pub struct RootCertificates(Arc<Vec<Certificate>>);

impl RootCertificates {
    fn new() -> Self {
        Self(Arc::new(Vec::new()))
    }
}

impl PartialEq for RootCertificates {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for RootCertificates {}

impl std::hash::Hash for RootCertificates {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(Arc::as_ptr(&self.0) as *const () as usize)
    }
}

// Implement basic Serialize and Deserialize methods for RootCertificates
// by simply serializing and deserializing an empty RootCertificates object
#[cfg(feature = "serde")]
impl Serialize for RootCertificates {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if !self.0.is_empty() {
            panic!("RootCertificates cannot be serialized if it contains certificates");
        }
        serializer.serialize_none()
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for RootCertificates {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // we simply return an empty RootCertificates
        Ok(RootCertificates::new())
    }
}

impl Default for PlClientOptions {
    fn default() -> Self {
        Self {
            // We set request timeout super high as the timeout isn't reset at ACK,
            // but starts from the moment we start downloading a body.
            // https://docs.rs/reqwest/latest/reqwest/struct.ClientBuilder.html#method.timeout
            timeout: None,
            // Concurrency can increase connection latency, so set to None, similar to default.
            connect_timeout: None,
            allow_http: true,
            root_certificates: RootCertificates::new(),
        }
    }
}

impl From<PlClientOptions> for ClientOptions {
    fn from(
        PlClientOptions {
            timeout,
            connect_timeout,
            allow_http,
            root_certificates,
        }: PlClientOptions,
    ) -> Self {
        let mut opts = ClientOptions::new();

        if let Some(timeout) = timeout {
            opts = opts.with_timeout(timeout);
        } else {
            opts = opts.with_timeout_disabled();
        }
        if let Some(connect_timeout) = connect_timeout {
            opts = opts.with_connect_timeout(connect_timeout);
        } else {
            opts = opts.with_connect_timeout_disabled();
        }
        opts = opts.with_allow_http(allow_http);
        for certificate in root_certificates.0.iter() {
            opts = opts.with_root_certificate(certificate.clone());
        }

        opts
    }
}
