use std::sync::Arc;
use std::time::Duration;

use object_store::{Certificate, ClientOptions};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    fn from(pl_opts: PlClientOptions) -> Self {
        let mut opts = ClientOptions::new();

        if let Some(timeout) = pl_opts.timeout {
            opts = opts.with_timeout(timeout);
        } else {
            opts = opts.with_timeout_disabled();
        }
        if let Some(connect_timeout) = pl_opts.connect_timeout {
            opts = opts.with_connect_timeout(connect_timeout);
        } else {
            opts = opts.with_connect_timeout_disabled();
        }
        opts = opts.with_allow_http(pl_opts.allow_http);
        for certificate in pl_opts.root_certificates.0.iter() {
            opts = opts.with_root_certificate(certificate.clone());
        }

        opts
    }
}
