use bytes::Bytes;
use polars_error::{to_compute_err, PolarsResult};
use reqwest::RequestBuilder;

/// Support for traversing paginated response values that look like:
/// ```text
/// {
///     $key_name: [$T, $T, ...],
///     next_page_token: "token" or null,
/// }
/// ```
#[macro_export]
macro_rules! impl_page_walk {
    ($S:ty, $T:ty, key_name = $key_name:tt) => {
        impl $S {
            pub async fn next(&mut self) -> PolarsResult<Option<Vec<$T>>> {
                return self
                    .0
                    .next(|bytes| {
                        let Response {
                            $key_name: out,
                            next_page_token,
                        } = decode_json_response(bytes)?;

                        Ok((out, next_page_token))
                    })
                    .await;

                #[derive(serde::Deserialize)]
                struct Response {
                    #[serde(default = "Vec::new")]
                    $key_name: Vec<$T>,
                    #[serde(default)]
                    next_page_token: Option<String>,
                }
            }

            pub async fn read_all_pages(mut self) -> PolarsResult<Vec<$T>> {
                let Some(mut out) = self.next().await? else {
                    return Ok(vec![]);
                };

                while let Some(v) = self.next().await? {
                    out.extend(v);
                }

                Ok(out)
            }
        }
    };
}

pub(crate) struct PageWalker {
    request: RequestBuilder,
    next_page_token: Option<String>,
    has_run: bool,
}

impl PageWalker {
    pub(crate) fn new(request: RequestBuilder) -> Self {
        Self {
            request,
            next_page_token: None,
            has_run: false,
        }
    }

    pub(crate) async fn next<F, T>(&mut self, deserializer: F) -> PolarsResult<Option<T>>
    where
        F: Fn(&[u8]) -> PolarsResult<(T, Option<String>)>,
    {
        let Some(resp_bytes) = self.next_bytes().await? else {
            return Ok(None);
        };

        let (value, next_page_token) = deserializer(&resp_bytes)?;
        self.next_page_token = next_page_token;

        Ok(Some(value))
    }

    pub(crate) async fn next_bytes(&mut self) -> PolarsResult<Option<Bytes>> {
        if self.has_run && self.next_page_token.is_none() {
            return Ok(None);
        }

        self.has_run = true;

        let request = self.request.try_clone().unwrap();

        let request = if let Some(page_token) = self.next_page_token.take() {
            request.query(&[("page_token", page_token)])
        } else {
            request
        };

        async { request.send().await?.bytes().await }
            .await
            .map(Some)
            .map_err(to_compute_err)
    }
}
