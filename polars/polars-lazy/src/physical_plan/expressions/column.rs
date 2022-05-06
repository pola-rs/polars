use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

pub struct ColumnExpr(Arc<str>, Expr);

impl ColumnExpr {
    pub fn new(name: Arc<str>, expr: Expr) -> Self {
        Self(name, expr)
    }
}

impl PhysicalExpr for ColumnExpr {
    fn as_expression(&self) -> &Expr {
        &self.1
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        match state.get_schema() {
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
                                    panic!("invalid schema")
                                }
                                // in release we fallback to linear search
                                #[allow(unreachable_code)]
                                {
                                    return df.column(&self.0).map(|s| s.clone());
                                }
                            }
                        };

                        if out.name() != &*self.0 {
                            // this path should not happen
                            #[cfg(feature = "panic_on_schema")]
                            {
                                panic!(
                                    "got {} expected: {} from schema: {:?} and DataFrame: {:?}",
                                    out.name(),
                                    &*self.0,
                                    &schema,
                                    &df
                                )
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
                            panic!("invalid schema")
                        }
                        // in release we fallback to linear search
                        #[allow(unreachable_code)]
                        df.column(&self.0).map(|s| s.clone())
                    }
                }
            }
        }
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let s = self.evaluate(df, state)?;
        Ok(AggregationContext::new(s, Cow::Borrowed(groups), false))
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = input_schema.get_field(&self.0).ok_or_else(|| {
            PolarsError::NotFound(format!(
                "could not find column: {} in schema: {:?}",
                self.0, &input_schema
            ))
        })?;
        Ok(field)
    }
}
