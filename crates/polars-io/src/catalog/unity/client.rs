use polars_error::{polars_bail, to_compute_err, PolarsResult};

use super::models::{CatalogInfo, SchemaInfo, TableInfo};
use super::utils::PageWalker;
use crate::impl_page_walk;
use crate::utils::decode_json_response;

/// Unity catalog client.
pub struct CatalogClient {
    workspace_url: String,
    http_client: reqwest::Client,
}

impl CatalogClient {
    pub async fn list_catalogs(&self) -> PolarsResult<Vec<CatalogInfo>> {
        ListCatalogs(PageWalker::new(self.http_client.get(format!(
            "{}{}",
            &self.workspace_url, "/api/2.1/unity-catalog/catalogs"
        ))))
        .read_all_pages()
        .await
    }

    pub async fn list_schemas(&self, catalog_name: &str) -> PolarsResult<Vec<SchemaInfo>> {
        ListSchemas(PageWalker::new(
            self.http_client
                .get(format!(
                    "{}{}",
                    &self.workspace_url, "/api/2.1/unity-catalog/schemas"
                ))
                .query(&[("catalog_name", catalog_name)]),
        ))
        .read_all_pages()
        .await
    }

    pub async fn list_tables(
        &self,
        catalog_name: &str,
        schema_name: &str,
    ) -> PolarsResult<Vec<TableInfo>> {
        ListTables(PageWalker::new(
            self.http_client
                .get(format!(
                    "{}{}",
                    &self.workspace_url, "/api/2.1/unity-catalog/tables"
                ))
                .query(&[("catalog_name", catalog_name), ("schema_name", schema_name)]),
        ))
        .read_all_pages()
        .await
    }

    pub async fn get_table_info(
        &self,
        catalog_name: &str,
        schema_name: &str,
        table_name: &str,
    ) -> PolarsResult<TableInfo> {
        let full_table_name = format!(
            "{}.{}.{}",
            catalog_name.replace('/', "%2F"),
            schema_name.replace('/', "%2F"),
            table_name.replace('/', "%2F")
        );

        let bytes = async {
            self.http_client
                .get(format!(
                    "{}{}{}",
                    &self.workspace_url, "/api/2.1/unity-catalog/tables/", full_table_name
                ))
                .query(&[("full_name", full_table_name)])
                .send()
                .await?
                .bytes()
                .await
        }
        .await
        .map_err(to_compute_err)?;

        let out: TableInfo = decode_json_response(&bytes)?;

        Ok(out)
    }
}

pub struct CatalogClientBuilder {
    workspace_url: Option<String>,
    bearer_token: Option<String>,
}

#[allow(clippy::derivable_impls)]
impl Default for CatalogClientBuilder {
    fn default() -> Self {
        Self {
            workspace_url: None,
            bearer_token: None,
        }
    }
}

impl CatalogClientBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_workspace_url(mut self, workspace_url: impl Into<String>) -> Self {
        self.workspace_url = Some(workspace_url.into());
        self
    }

    pub fn with_bearer_token(mut self, bearer_token: impl Into<String>) -> Self {
        self.bearer_token = Some(bearer_token.into());
        self
    }

    pub fn build(self) -> PolarsResult<CatalogClient> {
        let Some(workspace_url) = self.workspace_url else {
            polars_bail!(ComputeError: "expected Some(_) for workspace_url")
        };

        Ok(CatalogClient {
            workspace_url,
            http_client: {
                let builder = reqwest::ClientBuilder::new().user_agent("polars");

                let builder = if let Some(bearer_token) = self.bearer_token {
                    use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT};

                    let mut headers = HeaderMap::new();

                    let mut auth_value =
                        HeaderValue::from_str(format!("Bearer {}", bearer_token).as_str()).unwrap();
                    auth_value.set_sensitive(true);

                    headers.insert(AUTHORIZATION, auth_value);
                    headers.insert(USER_AGENT, "polars".try_into().unwrap());

                    builder.default_headers(headers)
                } else {
                    builder
                };

                builder.build().map_err(to_compute_err)?
            },
        })
    }
}

pub struct ListCatalogs(pub(crate) PageWalker);
impl_page_walk!(ListCatalogs, CatalogInfo, key_name = catalogs);

pub struct ListSchemas(pub(crate) PageWalker);
impl_page_walk!(ListSchemas, SchemaInfo, key_name = schemas);

pub struct ListTables(pub(crate) PageWalker);
impl_page_walk!(ListTables, TableInfo, key_name = tables);
