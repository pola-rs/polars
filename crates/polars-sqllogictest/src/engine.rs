use std::fmt;
use std::sync::LazyLock;

use polars_core::prelude::{DataFrame, PlHashMap, PlHashSet};
use polars_sql::SQLContext;
use regex::Regex;
use sqllogictest::{DB, DBOutput, DefaultColumnType};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::{output, setup};

/// SQLite's `NOT INDEXED` / `INDEXED BY <name>` table hints are a no-op
/// (they only advise the query planner on index usage); sqlparser's
/// `GenericDialect` doesn't recognize them, so strip them before parsing.
static INDEX_HINT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bNOT\s+INDEXED\b|\bINDEXED\s+BY\s+\w+\b").unwrap()
});

fn strip_index_hints(sql: &str) -> std::borrow::Cow<'_, str> {
    INDEX_HINT_RE.replace_all(sql, "")
}

#[derive(Debug)]
pub struct EngineError(String);

impl EngineError {
    pub fn new(msg: impl Into<String>) -> Self {
        EngineError(msg.into())
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EngineError {}

impl From<polars_core::prelude::PolarsError> for EngineError {
    fn from(err: polars_core::prelude::PolarsError) -> Self {
        EngineError(err.to_string())
    }
}

pub struct PolarsEngine {
    ctx: SQLContext,
    tables: PlHashMap<String, DataFrame>,
    views: PlHashSet<String>,
}

impl PolarsEngine {
    pub fn new() -> Self {
        PolarsEngine {
            ctx: SQLContext::new(),
            tables: PlHashMap::default(),
            views: PlHashSet::default(),
        }
    }
}

impl DB for PolarsEngine {
    type Error = EngineError;
    type ColumnType = DefaultColumnType;

    fn run(&mut self, sql: &str) -> Result<DBOutput<DefaultColumnType>, EngineError> {
        let sql = strip_index_hints(sql);
        let sql: &str = &sql;
        if let Ok(statements) = Parser::parse_sql(&GenericDialect, sql) {
            if !statements.is_empty() && statements.iter().all(setup::is_setup_statement) {
                let mut affected = 0u64;
                for statement in &statements {
                    affected += setup::run_setup_statement(
                        &mut self.tables,
                        &mut self.views,
                        &mut self.ctx,
                        statement,
                    )?;
                }
                return Ok(DBOutput::StatementComplete(affected));
            }
            for statement in &statements {
                if let Some(name) = setup::delete_target_table(statement) {
                    if self.views.contains(&name) {
                        return Err(EngineError::new(format!("cannot DELETE from view: {name}")));
                    }
                }
            }
        }

        let df = self
            .ctx
            .execute(sql)
            .and_then(|lf| lf.collect())
            .map_err(EngineError::from)?;
        Ok(output::dataframe_to_output(&df))
    }

    fn engine_name(&self) -> &str {
        "polars"
    }
}
