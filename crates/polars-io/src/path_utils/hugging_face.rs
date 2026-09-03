// Hugging Face path resolution support

use std::borrow::Cow;

use polars_error::{PolarsResult, polars_bail, to_compute_err};
use polars_utils::pl_path::PlRefPath;

use crate::cloud::{
    CloudConfig, CloudOptions, Matcher, USER_AGENT, extract_prefix_expansion,
    try_build_http_header_map_from_items_slice,
};
use crate::path_utils::HiveIdxTracker;
use crate::pl_async::with_concurrency_budget;
use crate::utils::{URL_ENCODE_CHARSET, decode_json_response};

/// Percent-encoding character set for HF Hub paths.
///
/// This is URL_ENCODE_CHARSET with slashes preserved - by not encoding slashes,
/// the API request will be counted under a higher "resolvers" ratelimit of (3000/5min)
/// compared to the default "pages" limit of (100/5min limit).
///
/// ref <https://github.com/pola-rs/polars/issues/25389>
const HF_PATH_ENCODE_CHARSET: &percent_encoding::AsciiSet = &URL_ENCODE_CHARSET.remove(b'/');

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
        // * Don't percent-encode bucket/repository - they are path segments where
        //   slashes are separators. E.g. "HuggingFaceFW/fineweb-2" must stay as-is.
        // * DO encode revision - slashes in revisions like "refs/convert/parquet"
        //   are part of the revision name, not path separators.
        //   See: https://github.com/pola-rs/polars/issues/25389
        let encoded_revision =
            percent_encoding::percent_encode(revision.as_bytes(), URL_ENCODE_CHARSET);
        let api_base_path = format!(
            "https://huggingface.co/api/{}/{}/tree/{}/",
            bucket, repository, encoded_revision
        );
        let download_base_path = format!(
            "https://huggingface.co/{}/{}/resolve/{}/",
            bucket, repository, encoded_revision
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
            percent_encoding::percent_encode(rel_path.as_bytes(), HF_PATH_ENCODE_CHARSET)
        )
    }

    fn get_api_uri(&self, rel_path: &str) -> String {
        format!(
            "{}{}",
            self.api_base_path,
            percent_encoding::percent_encode(rel_path.as_bytes(), HF_PATH_ENCODE_CHARSET)
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
}

/// API response is paginated with a `link` header.
/// * https://huggingface.co/docs/hub/en/api#get-apidatasets
/// * https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api?apiVersion=2022-11-28#using-link-headers
struct GetPages<'a> {
    client: &'a reqwest::Client,
    uri: Option<String>,
}

impl GetPages<'_> {
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
    paths: &[PlRefPath],
    check_directory_level: bool,
    cloud_options: &Option<CloudOptions>,
    glob: bool,
) -> PolarsResult<(usize, Vec<PlRefPath>)> {
    assert!(!paths.is_empty());

    let client = reqwest::ClientBuilder::new()
        .user_agent(USER_AGENT)
        .http1_only()
        .https_only(true);

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
    let mut hive_idx_tracker = HiveIdxTracker {
        idx: usize::MAX,
        paths,
        check_directory_level,
    };

    for (path_idx, path) in paths.iter().enumerate() {
        let path_parts = &HFPathParts::try_from_uri(path.as_str())?;
        let repo_location = &HFRepoLocation::new(
            &path_parts.bucket,
            &path_parts.repository,
            &path_parts.revision,
        );
        let rel_path = path_parts.path.as_str();

        let (prefix, expansion) = if glob {
            extract_prefix_expansion(rel_path)?
        } else {
            (Cow::Owned(path_parts.path.clone()), None)
        };
        let expansion_matcher = &if expansion.is_some() {
            Some(Matcher::new(prefix.to_string(), expansion.as_deref())?)
        } else {
            None
        };

        let file_uri = repo_location.get_file_uri(rel_path);

        if !path_parts.path.ends_with("/") && expansion.is_none() {
            // Confirm that this is a file using a HEAD request.
            if with_concurrency_budget(1, || async {
                client.head(&file_uri).send().await.map_err(to_compute_err)
            })
            .await?
            .status()
                == 200
            {
                hive_idx_tracker.update(0, path_idx)?;
                out_paths.push(PlRefPath::new(file_uri));
                continue;
            }
        }

        hive_idx_tracker.update(file_uri.len(), path_idx)?;

        let uri = format!("{}?recursive=true", repo_location.get_api_uri(&prefix));
        let mut gp = GetPages {
            uri: Some(uri),
            client,
        };

        let sort_start_idx = out_paths.len();

        while let Some(bytes) = gp.next().await {
            let bytes = bytes?;
            let response: Vec<HFAPIResponse> = decode_json_response(bytes.as_ref())?;

            for entry in response {
                // Only include files with size > 0
                if entry.is_file() && entry.size > 0 {
                    // If we have a glob pattern, filter by it; otherwise include all files
                    let matches = if let Some(matcher) = expansion_matcher {
                        matcher.is_matching(entry.path.as_str())
                    } else {
                        true
                    };

                    if matches {
                        out_paths.push(PlRefPath::new(repo_location.get_file_uri(&entry.path)));
                    }
                }
            }
        }

        if let Some(mut_slice) = out_paths.get_mut(sort_start_idx..) {
            <[PlRefPath]>::sort_unstable(mut_slice);
        }
    }

    Ok((hive_idx_tracker.idx, out_paths))
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
            panic!("expected err result for uri {uri} instead of {out:?}");
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

    #[test]
    fn test_hf_url_encoding() {
        // Verify URLs preserve slashes (don't encode as %2F) but encode special chars.
        // Slashes must remain for correct rate limit classification by HF Hub.
        // Special chars (spaces, colons) must be encoded for file downloads to work.
        // See: https://github.com/pola-rs/polars/issues/25389
        use super::HFRepoLocation;

        let loc = HFRepoLocation::new("datasets", "HuggingFaceFW/fineweb-2", "main");

        // Check base paths don't encode slashes
        assert_eq!(
            loc.api_base_path,
            "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-2/tree/main/"
        );
        assert_eq!(
            loc.download_base_path,
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/"
        );

        // Check file URIs preserve slashes in paths
        let file_uri = loc.get_file_uri("data/aai_Latn/train/000_00000.parquet");
        assert_eq!(
            file_uri,
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/aai_Latn/train/000_00000.parquet"
        );

        // Check that special characters ARE encoded (spaces -> %20, colons -> %3A)
        // This is needed for hive-partitioned paths like "date2=2023-01-01 00:00:00.000000"
        let file_uri = loc.get_file_uri(
            "hive_dates/date1=2024-01-01/date2=2023-01-01 00:00:00.000000/00000000.parquet",
        );
        assert_eq!(
            file_uri,
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/hive_dates/date1%3D2024-01-01/date2%3D2023-01-01%2000%3A00%3A00.000000/00000000.parquet"
        );

        // Check that brackets are encoded ([ -> %5B, ] -> %5D)
        let file_uri = loc.get_file_uri("special-chars/[*.parquet");
        assert_eq!(
            file_uri,
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/special-chars/%5B%2A.parquet"
        );

        // Check that revision slashes ARE encoded (they're part of the revision name)
        // e.g. "refs/convert/parquet" -> "refs%2Fconvert%2Fparquet"
        let loc = HFRepoLocation::new("datasets", "user/repo", "refs/convert/parquet");
        assert_eq!(
            loc.api_base_path,
            "https://huggingface.co/api/datasets/user/repo/tree/refs%2Fconvert%2Fparquet/"
        );
        assert_eq!(
            loc.download_base_path,
            "https://huggingface.co/datasets/user/repo/resolve/refs%2Fconvert%2Fparquet/"
        );
    }
}
