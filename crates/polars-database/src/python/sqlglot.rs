//! The sqlglot SQL renderer: a [`SqlRenderer`] that hides a Python module.
//!
//! The `SqlQuery` crosses as JSON; `polars_database._sqlglot` builds sqlglot expressions from
//! it and generates the dialect's SQL. Only the planning machine renders SQL.

use polars_error::{PolarsResult, polars_err};
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::database::dialect::DialectName;
use crate::database::sql::{QueryFacts, SqlExpr, SqlQuery, SqlRenderer};

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

    fn render_partitions(
        &self,
        query: &SqlQuery,
        count: &SqlQuery,
        predicates: &[Option<SqlExpr>],
        dialect: DialectName,
    ) -> PolarsResult<Vec<(String, String)>> {
        let encode = |v: &dyn Fn() -> serde_json::Result<String>| {
            v().map_err(|e| polars_err!(ComputeError: "could not encode SQL query: {e}"))
        };
        let query = encode(&|| serde_json::to_string(query))?;
        let count = encode(&|| serde_json::to_string(count))?;
        let predicates = encode(&|| serde_json::to_string(predicates))?;
        Python::attach(|py| {
            self.module
                .call_method1(
                    py,
                    "render_partitions",
                    (query, count, predicates, dialect_name(dialect)),
                )?
                .extract::<Vec<(String, String)>>(py)
        })
        .map_err(|e| polars_err!(ComputeError: "sqlglot could not render SQL for {dialect}: {e}"))
    }

    fn inspect_query(&self, sql: &str, dialect: DialectName) -> PolarsResult<QueryFacts> {
        self.facts("inspect_query", sql, dialect)
            .map_err(|e| polars_err!(ComputeError: "sqlglot could not inspect the query: {e}"))
    }

    fn outer_order_by(
        &self,
        sql: &str,
        dialect: DialectName,
        columns: &[PlSmallStr],
        star_ordinals: bool,
    ) -> PolarsResult<Option<PlSmallStr>> {
        let columns: Vec<&str> = columns.iter().map(|c| c.as_str()).collect();
        Python::attach(|py| {
            self.module
                .call_method1(
                    py,
                    "outer_order_by",
                    (sql, dialect_name(dialect), columns, star_ordinals),
                )?
                .extract::<Option<String>>(py)
        })
        .map(|o| o.map(Into::into))
        .map_err(|e| polars_err!(ComputeError: "sqlglot could not read the query's ORDER BY: {e}"))
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
            let outer_order_by: Option<String> = match d.get_item("outer_order_by")? {
                Some(v) if !v.is_none() => Some(v.extract()?),
                _ => None,
            };
            let known_functions: Vec<String> = match d.get_item("known_functions")? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => vec![],
            };
            let unproven: Vec<String> = match d.get_item("unproven")? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => vec![],
            };
            let tables: Vec<Vec<String>> = match d.get_item("tables")? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => vec![],
            };
            let strings = |k: &str| -> PyResult<Vec<PlSmallStr>> {
                Ok(match d.get_item(k)? {
                    Some(v) if !v.is_none() => v
                        .extract::<Vec<String>>()?
                        .into_iter()
                        .map(Into::into)
                        .collect(),
                    _ => vec![],
                })
            };
            let referenced_columns = match d.get_item("referenced_columns")? {
                Some(v) if !v.is_none() => Some(strings("referenced_columns")?),
                _ => None,
            };
            let final_tables: Vec<Vec<String>> = match d.get_item("final_tables")? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => vec![],
            };
            PyResult::Ok(QueryFacts {
                cast_types: strings("cast_types")?,
                any_operator: get("any_operator")?,
                temporal_cast_columns: strings("temporal_cast_columns")?,
                referenced_columns,
                final_tables: final_tables
                    .into_iter()
                    .map(|t| t.into_iter().map(Into::into).collect())
                    .collect(),
                parses: get("parses")?,
                single_select: get("single_select")?,
                has_cte: get("has_cte")?,
                top_level_order_by_without_limit: get("top_level_order_by_without_limit")?,
                outer_order_by: outer_order_by.map(Into::into),
                output_columns: output_columns.map(|v| v.into_iter().map(Into::into).collect()),
                read_only: get("read_only")?,
                repeatable: get("repeatable")?,
                top_level_limit: get("top_level_limit")?,
                nested_queries: get("nested_queries")?,
                side_effects: get("side_effects")?,
                side_effects_outside_rows: get("side_effects_outside_rows")?,
                volatile_columns: volatile_columns.into_iter().map(Into::into).collect(),
                all_columns_volatile: get("all_columns_volatile")?,
                unproven: unproven.into_iter().map(Into::into).collect(),
                known_functions: known_functions.into_iter().map(Into::into).collect(),
                collapses_rows: get("collapses_rows")?,
                collated: get("collated")?,
                tables: tables
                    .into_iter()
                    .map(|t| t.into_iter().map(Into::into).collect())
                    .collect(),
                body: body.map(Into::into),
            })
        })
    }
}
