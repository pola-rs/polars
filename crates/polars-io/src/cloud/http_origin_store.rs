use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use object_store::client::{HttpClient, HttpConnector, ReqwestConnector};
use object_store::http::{HttpBuilder, HttpStore};
use object_store::path::Path;
use object_store::{
    ClientOptions, CopyOptions, Error, GetOptions, GetResult, GetResultPayload, ListResult,
    MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload,
    PutResult, Result,
};

/// [`ObjectStore`] for all URLs on one HTTP(S) origin, sharing one connection pool.
/// Locations are the raw URL path and query; each op uses a per-URL [`HttpStore`].
#[derive(Debug)]
pub(crate) struct HttpOriginStore {
    /// `scheme://authority`, no trailing slash.
    base_url: String,
    client_options: ClientOptions,
    client: HttpClient,
}

impl HttpOriginStore {
    pub(crate) fn new(base_url: &str, client_options: ClientOptions) -> Result<Self> {
        let client = ReqwestConnector::default().connect(&client_options)?;

        Ok(Self {
            base_url: base_url.to_string(),
            client_options,
            client,
        })
    }

    fn store_for(&self, location: &Path) -> Result<HttpStore> {
        HttpBuilder::new()
            .with_url(format!("{}/{}", self.base_url, location))
            .with_client_options(self.client_options.clone())
            .with_http_connector(SharedHttpConnector(self.client.clone()))
            .build()
    }

    fn not_implemented(&self, operation: &str) -> Error {
        Error::NotImplemented {
            operation: operation.to_string(),
            implementer: self.to_string(),
        }
    }
}

/// Sets the location on an error of a per-URL store, which only knows an empty one.
fn with_location(err: Error, location: &Path) -> Error {
    let wrap = |source| {
        Box::new(LocationError {
            location: location.to_string(),
            source,
        }) as _
    };

    match err {
        Error::Generic { store, source } => Error::Generic {
            store,
            source: wrap(source),
        },
        Error::NotSupported { source } => Error::NotSupported {
            source: wrap(source),
        },
        mut err => {
            if let Error::NotFound { path, .. }
            | Error::AlreadyExists { path, .. }
            | Error::Precondition { path, .. }
            | Error::NotModified { path, .. }
            | Error::PermissionDenied { path, .. }
            | Error::Unauthenticated { path, .. } = &mut err
            {
                *path = location.to_string();
            }
            err
        },
    }
}

#[derive(Debug)]
struct LocationError {
    location: String,
    source: Box<dyn std::error::Error + Send + Sync>,
}

impl std::fmt::Display for LocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (location: {})", self.source, self.location)
    }
}

impl std::error::Error for LocationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

#[derive(Debug)]
struct SharedHttpConnector(HttpClient);

impl HttpConnector for SharedHttpConnector {
    fn connect(&self, _options: &ClientOptions) -> Result<HttpClient> {
        Ok(self.0.clone())
    }
}

impl std::fmt::Display for HttpOriginStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Not the URL: it may carry credentials.
        write!(f, "HttpOriginStore")
    }
}

#[async_trait]
impl ObjectStore for HttpOriginStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        self.store_for(location)?
            .put_opts(&Path::default(), payload, opts)
            .await
            .map_err(|e| with_location(e, location))
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.store_for(location)?
            .put_multipart_opts(&Path::default(), opts)
            .await
            .map_err(|e| with_location(e, location))
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        let mut out = self
            .store_for(location)?
            .get_opts(&Path::default(), options)
            .await
            .map_err(|e| with_location(e, location))?;
        out.meta.location = location.clone();
        out.payload = match out.payload {
            GetResultPayload::Stream(stream) => {
                let location = location.clone();
                let stream = stream.map_err(move |e| with_location(e, &location));
                GetResultPayload::Stream(stream.boxed())
            },
            payload => payload,
        };
        Ok(out)
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        self.store_for(location)?
            .get_ranges(&Path::default(), ranges)
            .await
            .map_err(|e| with_location(e, location))
    }

    fn delete_stream(
        &self,
        _locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        futures::stream::once(std::future::ready(Err(self.not_implemented("delete")))).boxed()
    }

    fn list(&self, _prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        futures::stream::once(std::future::ready(Err(self.not_implemented("list")))).boxed()
    }

    async fn list_with_delimiter(&self, _prefix: Option<&Path>) -> Result<ListResult> {
        Err(self.not_implemented("list_with_delimiter"))
    }

    async fn copy_opts(&self, _from: &Path, _to: &Path, _options: CopyOptions) -> Result<()> {
        Err(self.not_implemented("copy"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_location() {
        let location = Path::from("dir/a.parquet");

        let err = Error::NotFound {
            path: String::new(),
            source: "404".into(),
        };
        let err = with_location(err, &location);
        assert!(matches!(&err, Error::NotFound { path, .. } if path == "dir/a.parquet"));

        let err = Error::Generic {
            store: "HTTP",
            source: "reset".into(),
        };
        let err = with_location(err, &location);
        assert!(matches!(err, Error::Generic { .. }));
        assert!(err.to_string().ends_with("reset (location: dir/a.parquet)"));
    }
}
