use std::borrow::Cow;
use std::sync::Arc;

use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct ColumnExpr {
    name: Arc<str>,
    expr: Expr,
    schema: Option<SchemaRef>,
}

impl ColumnExpr {
    pub fn new(name: Arc<str>, expr: Expr, schema: Option<SchemaRef>) -> Self {
        Self { name, expr, schema }
    }
}

impl ColumnExpr {
    fn check_external_context(
        &self,
        out: PolarsResult<Series>,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        match out {
            Ok(col) => Ok(col),
            Err(e) => {
                if state.ext_contexts.is_empty() {
                    Err(e)
                } else {
                    for df in state.ext_contexts.as_ref() {
                        let out = df.column(&self.name);
                        if out.is_ok() {
                            return out.cloned();
                        }
                    }
                    Err(e)
                }
            }
        }
    }

    fn process_by_idx(
        &self,
        out: &Series,
        _state: &ExecutionState,
        _schema: &Schema,
        df: &DataFrame,
        check_state_schema: bool,
    ) -> PolarsResult<Series> {
        if out.name() != &*self.name {
            if check_state_schema {
                if let Some(schema) = _state.get_schema() {
                    return self.process_from_state_schema(df, _state, &schema);
                }
            }

            // this path should not happen
            #[cfg(feature = "panic_on_schema")]
            {
                if _state.ext_contexts.is_empty()
                    && std::env::var("POLARS_NO_SCHEMA_CHECK").is_err()
                {
                    panic!(
                        "got {} expected: {} from schema: {:?} and DataFrame: {:?}",
                        out.name(),
                        &*self.name,
                        _schema,
                        df
                    )
                }
            }
            // in release we fallback to linear search
            #[allow(unreachable_code)]
            {
                df.column(&self.name).map(|s| s.clone())
            }
        } else {
            Ok(out.clone())
        }
    }
    fn process_by_linear_search(
        &self,
        df: &DataFrame,
        _state: &ExecutionState,
        _panic_during_test: bool,
    ) -> PolarsResult<Series> {
        #[cfg(feature = "panic_on_schema")]
        {
            if _panic_during_test
                && _state.ext_contexts.is_empty()
                && std::env::var("POLARS_NO_SCHEMA_CHECK").is_err()
            {
                panic!("invalid schema")
            }
        }
        // in release we fallback to linear search
        #[allow(unreachable_code)]
        df.column(&self.name).map(|s| s.clone())
    }

    fn process_from_state_schema(
        &self,
        df: &DataFrame,
        state: &ExecutionState,
        schema: &Schema,
    ) -> PolarsResult<Series> {
        match schema.get_full(&self.name) {
            None => self.process_by_linear_search(df, state, true),
            Some((idx, _, _)) => match df.get_columns().get(idx) {
                Some(out) => self.process_by_idx(out, state, schema, df, false),
                None => self.process_by_linear_search(df, state, true),
            },
        }
    }
}

impl PhysicalExpr for ColumnExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let out = match &self.schema {
            None => self.process_by_linear_search(df, state, false),
            Some(schema) => {
                match schema.get_full(&self.name) {
                    Some((idx, _, _)) => {
                        // check if the schema was correct
                        // if not do O(n) search
                        match df.get_columns().get(idx) {
                            Some(out) => self.process_by_idx(out, state, schema, df, true),
                            None => {
                                // partitioned groupby special case
                                if let Some(schema) = state.get_schema() {
                                    self.process_from_state_schema(df, state, &schema)
                                } else {
                                    self.process_by_linear_search(df, state, true)
                                }
                            }
                        }
                    }
                    // in the future we will throw an error here
                    // now we do a linear search first as the lazy reported schema may still be incorrect
                    // in debug builds we panic so that it can be fixed when occurring
                    None => self.process_by_linear_search(df, state, true),
                }
            }
        };
        self.check_external_context(out, state)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let s = self.evaluate(df, state)?;
        Ok(AggregationContext::new(s, Cow::Borrowed(groups), false))
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        let field = input_schema.get_field(&self.name).ok_or_else(|| {
            PolarsError::ColumnNotFound(
                format!(
                    "could not find: {} in schema: {:?}",
                    self.name, &input_schema
                )
                .into(),
            )
        })?;
        Ok(field)
    }
    fn is_valid_aggregation(&self) -> bool {
        false
    }
}

impl PartitionedAggregation for ColumnExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        _groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        self.evaluate(df, state)
    }

    fn finalize(
        &self,
        partitioned: Series,
        _groups: &GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<Series> {
        Ok(partitioned)
    }
}
