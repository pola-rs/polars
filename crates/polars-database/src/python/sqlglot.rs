//! The sqlglot SQL renderer: a [`SqlRenderer`] that hides a Python module.
//!
//! The `SqlQuery` crosses as JSON; `polars_database._sqlglot` builds sqlglot expressions from
//! it and generates the dialect's SQL. Only the planning machine renders SQL.

use polars_error::{PolarsResult, polars_err};
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::database::dialect::DialectName;
use crate::database::sql::{QueryFacts, SqlQuery, SqlRenderer};

pub struct SqlglotRenderer {
    module: Py<PyModule>,
    id: PlSmallStr,
}

impl SqlglotRenderer {
    pub fn new() -> PolarsResult<Self> {
        Python::attach(|py| {
            let module = py.import("polars_database._sqlglot")?;
            let version: String = module.call_method0("version")?.extract()?;
            PyResult::Ok(Self {
                module: module.unbind(),
                id: format!("sqlglot {version}").into(),
            })
        })
        .map_err(|e| {
            polars_err!(
                ComputeError:
                "the sqlglot SQL renderer is not available: {e}\n  \
                install it with: pip install sqlglot"
            )
        })
    }
}

fn dialect_name(d: DialectName) -> Option<&'static str> {
    d.renderer_dialect()
}

impl SqlRenderer for SqlglotRenderer {
    fn id(&self) -> PlSmallStr {
        self.id.clone()
    }

    fn render(&self, query: &SqlQuery, dialect: DialectName) -> PolarsResult<String> {
        let json = serde_json::to_string(query)
            .map_err(|e| polars_err!(ComputeError: "could not encode SQL query: {e}"))?;
        Python::attach(|py| {
            self.module
                .call_method1(py, "render", (json, dialect_name(dialect)))?
                .extract::<String>(py)
        })
        .map_err(|e| polars_err!(ComputeError: "sqlglot could not render SQL for {dialect}: {e}"))
    }

    fn inspect_query(&self, sql: &str, dialect: DialectName) -> PolarsResult<QueryFacts> {
        self.facts("inspect_query", sql, dialect)
            .map_err(|e| polars_err!(ComputeError: "sqlglot could not inspect the query: {e}"))
    }

    fn inspect_view(&self, sql: &str, dialect: DialectName) -> PolarsResult<QueryFacts> {
        self.facts("inspect_view", sql, dialect)
            .map_err(|e| polars_err!(ComputeError: "sqlglot could not inspect the view: {e}"))
    }
}

impl SqlglotRenderer {
    fn facts(&self, function: &str, sql: &str, dialect: DialectName) -> PyResult<QueryFacts> {
        Python::attach(|py| {
            let d = self
                .module
                .call_method1(py, function, (sql, dialect_name(dialect)))?;
            let d = d.bind(py).cast::<PyDict>()?;
            let get = |k: &str| -> PyResult<bool> {
                Ok(d.get_item(k)?
                    .is_some_and(|v| v.is_truthy().unwrap_or(false)))
            };
            let output_columns: Option<Vec<String>> = match d.get_item("output_columns")? {
                Some(v) if !v.is_none() => Some(v.extract()?),
                _ => None,
            };
            let volatile_columns: Vec<String> = match d.get_item("volatile_columns")? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => vec![],
            };
            let body: Option<String> = match d.get_item("body")? {
                Some(v) if !v.is_none() => Some(v.extract()?),
                _ => None,
            };
            let unknown_functions: Vec<String> = match d.get_item("unknown_functions")? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => vec![],
            };
            let known_functions: Vec<String> = match d.get_item("known_functions")? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => vec![],
            };
            let tables: Vec<Vec<String>> = match d.get_item("tables")? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => vec![],
            };
            PyResult::Ok(QueryFacts {
                parses: get("parses")?,
                single_select: get("single_select")?,
                has_cte: get("has_cte")?,
                top_level_order_by_without_limit: get("top_level_order_by_without_limit")?,
                output_columns: output_columns.map(|v| v.into_iter().map(Into::into).collect()),
                read_only: get("read_only")?,
                repeatable: get("repeatable")?,
                top_level_limit: get("top_level_limit")?,
                nested_queries: get("nested_queries")?,
                side_effects: get("side_effects")?,
                side_effects_outside_rows: get("side_effects_outside_rows")?,
                volatile_columns: volatile_columns.into_iter().map(Into::into).collect(),
                all_columns_volatile: get("all_columns_volatile")?,
                unknown_functions: unknown_functions.into_iter().map(Into::into).collect(),
                known_functions: known_functions.into_iter().map(Into::into).collect(),
                tables: tables
                    .into_iter()
                    .map(|t| t.into_iter().map(Into::into).collect())
                    .collect(),
                body: body.map(Into::into),
            })
        })
    }
}
