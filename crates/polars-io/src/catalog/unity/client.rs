use polars_core::prelude::PlHashMap;
use polars_core::schema::Schema;
use polars_error::{polars_bail, to_compute_err, PolarsResult};

use super::models::{CatalogInfo, NamespaceInfo, TableCredentials, TableInfo};
use super::utils::{do_request, PageWalker};
use crate::catalog::schema::schema_to_column_info_list;
use crate::catalog::unity::models::{ColumnInfo, DataSourceFormat, TableType};
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

    pub async fn list_namespaces(&self, catalog_name: &str) -> PolarsResult<Vec<NamespaceInfo>> {
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
        namespace: &str,
    ) -> PolarsResult<Vec<TableInfo>> {
        ListTables(PageWalker::new(
            self.http_client
                .get(format!(
                    "{}{}",
                    &self.workspace_url, "/api/2.1/unity-catalog/tables"
                ))
                .query(&[("catalog_name", catalog_name), ("schema_name", namespace)]),
        ))
        .read_all_pages()
        .await
    }

    pub async fn get_table_info(
        &self,
        catalog_name: &str,
        namespace: &str,
        table_name: &str,
    ) -> PolarsResult<TableInfo> {
        let full_table_name = format!(
            "{}.{}.{}",
            catalog_name.replace('/', "%2F"),
            namespace.replace('/', "%2F"),
            table_name.replace('/', "%2F")
        );

        let bytes = do_request(
            self.http_client
                .get(format!(
                    "{}{}{}",
                    &self.workspace_url, "/api/2.1/unity-catalog/tables/", full_table_name
                ))
                .query(&[("full_name", full_table_name)]),
        )
        .await?;

        let out: TableInfo = decode_json_response(&bytes)?;

        Ok(out)
    }

    pub async fn get_table_credentials(
        &self,
        table_id: &str,
        write: bool,
    ) -> PolarsResult<TableCredentials> {
        let bytes = do_request(
            self.http_client
                .post(format!(
                    "{}{}",
                    &self.workspace_url, "/api/2.1/unity-catalog/temporary-table-credentials"
                ))
                .query(&[
                    ("table_id", table_id),
                    ("operation", if write { "READ_WRITE" } else { "READ" }),
                ]),
        )
        .await?;

        let out: TableCredentials = decode_json_response(&bytes)?;

        Ok(out)
    }

    pub async fn create_catalog(
        &self,
        catalog_name: &str,
        comment: Option<&str>,
        storage_root: Option<&str>,
    ) -> PolarsResult<CatalogInfo> {
        let resp = do_request(
            self.http_client
                .post(format!(
                    "{}{}",
                    &self.workspace_url, "/api/2.1/unity-catalog/catalogs"
                ))
                .json(&Body {
                    name: catalog_name,
                    comment,
                    storage_root,
                }),
        )
        .await?;

        return decode_json_response(&resp);

        #[derive(serde::Serialize)]
        struct Body<'a> {
            name: &'a str,
            comment: Option<&'a str>,
            storage_root: Option<&'a str>,
        }
    }

    pub async fn delete_catalog(&self, catalog_name: &str, force: bool) -> PolarsResult<()> {
        let catalog_name = catalog_name.replace('/', "%2F");

        do_request(
            self.http_client
                .delete(format!(
                    "{}{}{}",
                    &self.workspace_url, "/api/2.1/unity-catalog/catalogs/", catalog_name
                ))
                .query(&[("force", force)]),
        )
        .await?;

        Ok(())
    }

    pub async fn create_namespace(
        &self,
        catalog_name: &str,
        namespace: &str,
        comment: Option<&str>,
        storage_root: Option<&str>,
    ) -> PolarsResult<NamespaceInfo> {
        let resp = do_request(
            self.http_client
                .post(format!(
                    "{}{}",
                    &self.workspace_url, "/api/2.1/unity-catalog/schemas"
                ))
                .json(&Body {
                    name: namespace,
                    catalog_name,
                    comment,
                    storage_root,
                }),
        )
        .await?;

        return decode_json_response(&resp);

        #[derive(serde::Serialize)]
        struct Body<'a> {
            name: &'a str,
            catalog_name: &'a str,
            comment: Option<&'a str>,
            storage_root: Option<&'a str>,
        }
    }

    pub async fn delete_namespace(
        &self,
        catalog_name: &str,
        namespace: &str,
        force: bool,
    ) -> PolarsResult<()> {
        let full_name = format!(
            "{}.{}",
            catalog_name.replace('/', "%2F"),
            namespace.replace('/', "%2F"),
        );

        do_request(
            self.http_client
                .delete(format!(
                    "{}{}{}",
                    &self.workspace_url, "/api/2.1/unity-catalog/schemas/", full_name
                ))
                .query(&[("force", force)]),
        )
        .await?;

        Ok(())
    }

    /// Note, `data_source_format` can be None for some `table_type`s.
    #[allow(clippy::too_many_arguments)]
    pub async fn create_table(
        &self,
        catalog_name: &str,
        namespace: &str,
        table_name: &str,
        schema: Option<&Schema>,
        table_type: &TableType,
        data_source_format: Option<&DataSourceFormat>,
        comment: Option<&str>,
        storage_location: Option<&str>,
        properties: &mut (dyn Iterator<Item = (&str, &str)> + Send + Sync),
    ) -> PolarsResult<TableInfo> {
        let columns = schema.map(schema_to_column_info_list).transpose()?;
        let columns = columns.as_deref();

        let resp = do_request(
            self.http_client
                .post(format!(
                    "{}{}",
                    &self.workspace_url, "/api/2.1/unity-catalog/tables"
                ))
                .json(&Body {
                    name: table_name,
                    catalog_name,
                    schema_name: namespace,
                    table_type,
                    data_source_format,
                    comment,
                    columns,
                    storage_location,
                    properties: properties.collect(),
                }),
        )
        .await?;

        return decode_json_response(&resp);

        #[derive(serde::Serialize)]
        struct Body<'a> {
            name: &'a str,
            catalog_name: &'a str,
            schema_name: &'a str,
            comment: Option<&'a str>,
            table_type: &'a TableType,
            #[serde(skip_serializing_if = "Option::is_none")]
            data_source_format: Option<&'a DataSourceFormat>,
            columns: Option<&'a [ColumnInfo]>,
            storage_location: Option<&'a str>,
            properties: PlHashMap<&'a str, &'a str>,
        }
    }

    pub async fn delete_table(
        &self,
        catalog_name: &str,
        namespace: &str,
        table_name: &str,
    ) -> PolarsResult<()> {
        let full_name = format!(
            "{}.{}.{}",
            catalog_name.replace('/', "%2F"),
            namespace.replace('/', "%2F"),
            table_name.replace('/', "%2F"),
        );

        do_request(self.http_client.delete(format!(
            "{}{}{}",
            &self.workspace_url, "/api/2.1/unity-catalog/tables/", full_name
        )))
        .await?;

        Ok(())
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
impl_page_walk!(ListSchemas, NamespaceInfo, key_name = schemas);

pub struct ListTables(pub(crate) PageWalker);
impl_page_walk!(ListTables, TableInfo, key_name = tables);
