use std::fmt;

use polars_core::prelude::{DataFrame, PlHashMap};
use polars_sql::SQLContext;
use sqllogictest::{DB, DBOutput, DefaultColumnType};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::{output, setup};

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
}

impl PolarsEngine {
    pub fn new() -> Self {
        PolarsEngine {
            ctx: SQLContext::new(),
            tables: PlHashMap::default(),
        }
    }
}

impl DB for PolarsEngine {
    type Error = EngineError;
    type ColumnType = DefaultColumnType;

    fn run(&mut self, sql: &str) -> Result<DBOutput<DefaultColumnType>, EngineError> {
        if let Ok(statements) = Parser::parse_sql(&GenericDialect, sql) {
            if statements.len() == 1 && setup::is_setup_statement(&statements[0]) {
                let affected =
                    setup::run_setup_statement(&mut self.tables, &self.ctx, &statements[0])?;
                return Ok(DBOutput::StatementComplete(affected));
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
