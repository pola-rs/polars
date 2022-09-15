use std::sync::Arc;

use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct CastExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) data_type: DataType,
    pub(crate) expr: Expr,
    pub(crate) strict: bool,
}

impl CastExpr {
    fn finish(&self, input: &Series) -> PolarsResult<Series> {
        // this is quite dirty
        // We use the booleanarray as null series, because we have no null array.
        // in a ternary or binary operation, we then do type coercion to matching supertype.
        // here we create a null array for the types we cannot cast to from a booleanarray

        // todo! check if the expression is really null
        if input.bool().is_ok() && input.null_count() == input.len() {
            match &self.data_type {
                DataType::List(inner) => {
                    return Ok(
                        ListChunked::full_null_with_dtype(input.name(), input.len(), inner)
                            .into_series(),
                    )
                }
                #[cfg(feature = "dtype-date")]
                DataType::Date => {
                    return Ok(Int32Chunked::full_null(input.name(), input.len())
                        .into_date()
                        .into_series())
                }
                #[cfg(feature = "dtype-datetime")]
                DataType::Datetime(tu, tz) => {
                    return Ok(Int64Chunked::full_null(input.name(), input.len())
                        .into_datetime(*tu, tz.clone())
                        .into_series())
                }
                #[cfg(feature = "dtype-duration")]
                DataType::Duration(tu) => {
                    return Ok(Int64Chunked::full_null(input.name(), input.len())
                        .into_duration(*tu)
                        .into_series())
                }
                #[cfg(feature = "dtype-struct")]
                DataType::Struct(fields) => {
                    let fields = fields
                        .iter()
                        .map(|field| {
                            Series::full_null(field.name(), input.len(), field.data_type())
                        })
                        .collect::<Vec<_>>();
                    return Ok(StructChunked::new(input.name(), &fields)
                        .unwrap()
                        .into_series());
                }
                #[cfg(feature = "dtype-categorical")]
                DataType::Categorical(_) => {
                    return Ok(
                        CategoricalChunked::full_null(input.name(), input.len()).into_series()
                    )
                }
                _ => {}
            }
        }

        if self.strict {
            input.strict_cast(&self.data_type)
        } else {
            input.cast(&self.data_type)
        }
    }
}

impl PhysicalExpr for CastExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let series = self.input.evaluate(df, state)?;
        self.finish(&series)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        // before we flatten, make sure that groups are updated
        ac.groups();
        let s = ac.flat_naive();
        let s = self.finish(s.as_ref())?;

        if ac.is_literal() {
            ac.with_literal(s);
        } else {
            ac.with_series(s, false);
        }

        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema).map(|mut fld| {
            fld.coerce(self.data_type.clone());
            fld
        })
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }

    fn is_valid_aggregation(&self) -> bool {
        self.input.is_valid_aggregation()
    }
}

impl PartitionedAggregation for CastExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let e = self.input.as_partitioned_aggregator().unwrap();
        self.finish(&e.evaluate_partitioned(df, groups, state)?)
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
