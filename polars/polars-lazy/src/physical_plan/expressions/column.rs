use std::borrow::Cow;
use std::sync::Arc;

use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct ColumnExpr(Arc<str>, Expr);

impl ColumnExpr {
    pub fn new(name: Arc<str>, expr: Expr) -> Self {
        Self(name, expr)
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
                        let out = df.column(&self.0);
                        if out.is_ok() {
                            return out.cloned();
                        }
                    }
                    Err(e)
                }
            }
        }
    }
}

impl PhysicalExpr for ColumnExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.1)
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let out = match state.get_schema() {
            None => df.column(&self.0).map(|s| s.clone()),
            Some(schema) => {
                match schema.get_full(&self.0) {
                    Some((idx, _, _)) => {
                        // check if the schema was correct
                        // if not do O(n) search
                        let out = match df.get_columns().get(idx) {
                            Some(out) => out,
                            None => {
                                // this path should not happen
                                #[cfg(feature = "panic_on_schema")]
                                {
                                    if state.ext_contexts.is_empty() {
                                        panic!("invalid schema")
                                    }
                                }
                                // in release we fallback to linear search
                                #[allow(unreachable_code)]
                                {
                                    return self.check_external_context(
                                        df.column(&self.0).map(|s| s.clone()),
                                        state,
                                    );
                                }
                            }
                        };

                        if out.name() != &*self.0 {
                            // this path should not happen
                            #[cfg(feature = "panic_on_schema")]
                            {
                                if state.ext_contexts.is_empty() {
                                    panic!(
                                        "got {} expected: {} from schema: {:?} and DataFrame: {:?}",
                                        out.name(),
                                        &*self.0,
                                        &schema,
                                        &df
                                    )
                                }
                            }
                            // in release we fallback to linear search
                            #[allow(unreachable_code)]
                            {
                                df.column(&self.0).map(|s| s.clone())
                            }
                        } else {
                            Ok(out.clone())
                        }
                    }
                    // in the future we will throw an error here
                    // now we do a linear search first as the lazy reported schema may still be incorrect
                    // in debug builds we panic so that it can be fixed when occurring
                    None => {
                        #[cfg(feature = "panic_on_schema")]
                        {
                            if state.ext_contexts.is_empty() {
                                panic!("invalid schema")
                            }
                        }
                        // in release we fallback to linear search
                        #[allow(unreachable_code)]
                        df.column(&self.0).map(|s| s.clone())
                    }
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
        let field = input_schema.get_field(&self.0).ok_or_else(|| {
            PolarsError::NotFound(
                format!(
                    "could not find column: {} in schema: {:?}",
                    self.0, &input_schema
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
