// Hugging Face path resolution support

use std::collections::VecDeque;
use std::path::PathBuf;

use polars_error::{polars_bail, polars_err, to_compute_err, PolarsResult};

use crate::cloud::{
    extract_prefix_expansion, try_build_http_header_map_from_items_slice, CloudConfig,
    CloudOptions, Matcher,
};
use crate::path_utils::HiveIdxTracker;
use crate::pl_async::with_concurrency_budget;
use crate::prelude::URL_ENCODE_CHAR_SET;

#[derive(Debug, PartialEq)]
struct HFPathParts {
    bucket: String,
    repository: String,
    revision: String,
    /// Path relative to the repository root.
    path: String,
}

struct HFRepoLocation {
    api_base_path: String,
    download_base_path: String,
}

impl HFRepoLocation {
    fn new(bucket: &str, repository: &str, revision: &str) -> Self {
        let bucket = percent_encode(bucket.as_bytes());
        let repository = percent_encode(repository.as_bytes());

        // "https://huggingface.co/api/ [datasets | spaces] / {username} / {reponame} / tree / {revision} / {path from root}"
        let api_base_path = format!(
            "{}{}{}{}{}{}{}",
            "https://huggingface.co/api/", bucket, "/", repository, "/tree/", revision, "/"
        );
        let download_base_path = format!(
            "{}{}{}{}{}{}{}",
            "https://huggingface.co/", bucket, "/", repository, "/resolve/", revision, "/"
        );

        Self {
            api_base_path,
            download_base_path,
        }
    }

    fn get_file_uri(&self, rel_path: &str) -> String {
        format!(
            "{}{}",
            self.download_base_path,
            percent_encode(rel_path.as_bytes())
        )
    }

    fn get_api_uri(&self, rel_path: &str) -> String {
        format!(
            "{}{}",
            self.api_base_path,
            percent_encode(rel_path.as_bytes())
        )
    }
}

impl HFPathParts {
    /// Extracts path components from a hugging face path:
    /// `hf:// [datasets | spaces] / {username} / {reponame} @ {revision} / {path from root}`
    fn try_from_uri(uri: &str) -> PolarsResult<Self> {
        let Some(this) = (|| {
            // hf:// [datasets | spaces] / {username} / {reponame} @ {revision} / {path from root}
            //       !>
            if !uri.starts_with("hf://") {
                return None;
            }
            let uri = &uri[5..];

            // [datasets | spaces] / {username} / {reponame} @ {revision} / {path from root}
            // ^-----------------^   !>
            let i = memchr::memchr(b'/', uri.as_bytes())?;
            let bucket = uri.get(..i)?.to_string();
            let uri = uri.get(1 + i..)?;

            // {username} / {reponame} @ {revision} / {path from root}
            // ^----------------------------------^   !>
            let i = memchr::memchr(b'/', uri.as_bytes())?;
            let i = {
                // Also handle if they just give the repository, i.e.:
                // hf:// [datasets | spaces] / {username} / {reponame} @ {revision}
                let uri = uri.get(1 + i..)?;
                if uri.is_empty() {
                    return None;
                }
                1 + i + memchr::memchr(b'/', uri.as_bytes()).unwrap_or(uri.len())
            };
            let repository = uri.get(..i)?;
            let uri = uri.get(1 + i..).unwrap_or("");

            let (repository, revision) =
                if let Some(i) = memchr::memchr(b'@', repository.as_bytes()) {
                    (repository[..i].to_string(), repository[1 + i..].to_string())
                } else {
                    // No @revision in uri, default to `main`
                    (repository.to_string(), "main".to_string())
                };

            // {path from root}
            // ^--------------^
            let path = uri.to_string();

            Some(HFPathParts {
                bucket,
                repository,
                revision,
                path,
            })
        })() else {
            polars_bail!(ComputeError: "invalid Hugging Face path: {}", uri);
        };

        const BUCKETS: [&str; 2] = ["datasets", "spaces"];
        if !BUCKETS.contains(&this.bucket.as_str()) {
            polars_bail!(ComputeError: "hugging face uri bucket must be one of {:?}, got {} instead.", BUCKETS, this.bucket);
        }

        Ok(this)
    }
}

#[derive(Debug, serde::Deserialize)]
struct HFAPIResponse {
    #[serde(rename = "type")]
    type_: String,
    path: String,
    size: u64,
}

impl HFAPIResponse {
    fn is_file(&self) -> bool {
        self.type_ == "file"
    }

    fn is_directory(&self) -> bool {
        self.type_ == "directory"
    }
}

/// API response is paginated with a `link` header.
/// * https://huggingface.co/docs/hub/en/api#get-apidatasets
/// * https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api?apiVersion=2022-11-28#using-link-headers
struct GetPages<'a> {
    client: &'a reqwest::Client,
    uri: Option<String>,
}

impl<'a> GetPages<'a> {
    async fn next(&mut self) -> Option<PolarsResult<bytes::Bytes>> {
        let uri = self.uri.take()?;

        Some(
            async {
                let resp = with_concurrency_budget(1, || async {
                    self.client.get(uri).send().await.map_err(to_compute_err)
                })
                .await?;

                self.uri = resp
                    .headers()
                    .get("link")
                    .and_then(|x| Self::find_link(x.as_bytes(), "next".as_bytes()))
                    .transpose()?;

                let resp_bytes = resp.bytes().await.map_err(to_compute_err)?;

                Ok(resp_bytes)
            }
            .await,
        )
    }

    fn find_link(mut link: &[u8], rel: &[u8]) -> Option<PolarsResult<String>> {
        // "<https://...>; rel=\"next\", <https://...>; rel=\"last\""
        while !link.is_empty() {
            let i = memchr::memchr(b'<', link)?;
            link = link.get(1 + i..)?;
            let i = memchr::memchr(b'>', link)?;
            let uri = &link[..i];
            link = link.get(1 + i..)?;

            while !link.starts_with("rel=\"".as_bytes()) {
                link = link.get(1..)?
            }

            // rel="next"
            link = link.get(5..)?;
            let i = memchr::memchr(b'"', link)?;

            if &link[..i] == rel {
                return Some(
                    std::str::from_utf8(uri)
                        .map_err(to_compute_err)
                        .map(ToString::to_string),
                );
            }
        }

        None
    }
}

pub(super) async fn expand_paths_hf(
    paths: &[PathBuf],
    check_directory_level: bool,
    cloud_options: Option<&CloudOptions>,
    glob: bool,
) -> PolarsResult<(usize, Vec<PathBuf>)> {
    assert!(!paths.is_empty());

    let client = reqwest::ClientBuilder::new().http1_only().https_only(true);

    let client = if let Some(CloudOptions {
        config: Some(CloudConfig::Http { headers }),
        ..
    }) = cloud_options
    {
        client.default_headers(try_build_http_header_map_from_items_slice(
            headers.as_slice(),
        )?)
    } else {
        client
    };

    let client = &client.build().unwrap();

    let mut out_paths = vec![];
    let mut stack = VecDeque::new();
    let mut entries = vec![];
    let mut hive_idx_tracker = HiveIdxTracker {
        idx: usize::MAX,
        paths,
        check_directory_level,
    };

    for (path_idx, path) in paths.iter().enumerate() {
        let path_parts = &HFPathParts::try_from_uri(path.to_str().unwrap())?;
        let repo_location = &HFRepoLocation::new(
            &path_parts.bucket,
            &path_parts.repository,
            &path_parts.revision,
        );
        let rel_path = path_parts.path.as_str();

        let (prefix, expansion) = if glob {
            extract_prefix_expansion(rel_path)?
        } else {
            (path_parts.path.clone(), None)
        };
        let expansion_matcher = &if expansion.is_some() {
            Some(Matcher::new(prefix.clone(), expansion.as_deref())?)
        } else {
            None
        };

        if !path_parts.path.ends_with("/") && expansion.is_none() {
            hive_idx_tracker.update(0, path_idx)?;
            let file_uri = repo_location.get_file_uri(rel_path);
            let file_uri = file_uri.as_str();

            if with_concurrency_budget(1, || async {
                client.head(file_uri).send().await.map_err(to_compute_err)
            })
            .await?
            .status()
                == 200
            {
                out_paths.push(PathBuf::from(file_uri));
                continue;
            }
        }

        hive_idx_tracker.update(repo_location.get_file_uri(rel_path).len(), path_idx)?;

        assert!(stack.is_empty());
        stack.push_back(prefix.to_string());

        while let Some(rel_path) = stack.pop_front() {
            assert!(entries.is_empty());

            let uri = repo_location.get_api_uri(rel_path.as_str());
            let mut gp = GetPages {
                uri: Some(uri),
                client,
            };

            fn try_parse_api_response(bytes: &[u8]) -> PolarsResult<Vec<HFAPIResponse>> {
                serde_json::from_slice::<Vec<HFAPIResponse>>(bytes).map_err(
                    |e| polars_err!(ComputeError: "failed to parse API response as JSON: error: {}, value: {}", e, std::str::from_utf8(bytes).unwrap()),
                )
            }

            if let Some(matcher) = expansion_matcher {
                while let Some(bytes) = gp.next().await {
                    let bytes = bytes?;
                    let bytes = bytes.as_ref();
                    entries.extend(try_parse_api_response(bytes)?.into_iter().filter(|x| {
                        !x.is_file() || (x.size > 0 && matcher.is_matching(x.path.as_str()))
                    }));
                }
            } else {
                while let Some(bytes) = gp.next().await {
                    let bytes = bytes?;
                    let bytes = bytes.as_ref();
                    entries.extend(
                        try_parse_api_response(bytes)?
                            .into_iter()
                            .filter(|x| !x.is_file() || x.size > 0),
                    );
                }
            }

            entries.sort_unstable_by(|a, b| a.path.as_str().partial_cmp(b.path.as_str()).unwrap());

            for e in entries.drain(..) {
                if e.is_file() {
                    out_paths.push(PathBuf::from(repo_location.get_file_uri(&e.path)));
                } else if e.is_directory() {
                    stack.push_back(e.path);
                }
            }
        }
    }

    Ok((hive_idx_tracker.idx, out_paths))
}

fn percent_encode(bytes: &[u8]) -> percent_encoding::PercentEncode {
    percent_encoding::percent_encode(bytes, URL_ENCODE_CHAR_SET)
}

mod tests {

    #[test]
    fn test_hf_path_from_uri() {
        use super::HFPathParts;

        let uri = "hf://datasets/pola-rs/polars/README.md";
        let expect = HFPathParts {
            bucket: "datasets".into(),
            repository: "pola-rs/polars".into(),
            revision: "main".into(),
            path: "README.md".into(),
        };

        assert_eq!(HFPathParts::try_from_uri(uri).unwrap(), expect);

        let uri = "hf://spaces/pola-rs/polars@~parquet/";
        let expect = HFPathParts {
            bucket: "spaces".into(),
            repository: "pola-rs/polars".into(),
            revision: "~parquet".into(),
            path: "".into(),
        };

        assert_eq!(HFPathParts::try_from_uri(uri).unwrap(), expect);

        let uri = "hf://spaces/pola-rs/polars@~parquet";
        let expect = HFPathParts {
            bucket: "spaces".into(),
            repository: "pola-rs/polars".into(),
            revision: "~parquet".into(),
            path: "".into(),
        };

        assert_eq!(HFPathParts::try_from_uri(uri).unwrap(), expect);

        for uri in [
            "://",
            "s3://",
            "https://",
            "hf://",
            "hf:///",
            "hf:////",
            "hf://datasets/a",
            "hf://datasets/a/",
            "hf://bucket/a/b/c", // Invalid bucket name
        ] {
            let out = HFPathParts::try_from_uri(uri);
            if out.is_err() {
                continue;
            }
            panic!("expected err result for uri {} instead of {:?}", uri, out);
        }
    }

    #[test]
    fn test_get_pages_find_next_link() {
        use super::GetPages;
        let link = r#"<https://api.github.com/repositories/263727855/issues?page=3>; rel="next", <https://api.github.com/repositories/263727855/issues?page=7>; rel="last""#.as_bytes();

        assert_eq!(
            GetPages::find_link(link, "next".as_bytes()).map(Result::unwrap),
            Some("https://api.github.com/repositories/263727855/issues?page=3".into()),
        );

        assert_eq!(
            GetPages::find_link(link, "last".as_bytes()).map(Result::unwrap),
            Some("https://api.github.com/repositories/263727855/issues?page=7".into()),
        );

        assert_eq!(
            GetPages::find_link(link, "non-existent".as_bytes()).map(Result::unwrap),
            None,
        );
    }
}
