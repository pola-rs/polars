use crate::logical_plan::Context;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_core::frame::groupby::{fmt_groupby_column, GroupByMethod, GroupTuples};
use polars_core::prelude::*;
use polars_core::utils::{slice_offsets, NoNull};
use rayon::prelude::*;
use std::borrow::Cow;
use std::ops::Deref;
use std::sync::Arc;

pub struct LiteralExpr(pub LiteralValue, Expr);

impl LiteralExpr {
    pub fn new(value: LiteralValue, expr: Expr) -> Self {
        Self(value, expr)
    }
}

impl PhysicalExpr for LiteralExpr {
    fn as_expression(&self) -> &Expr {
        &self.1
    }
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        use LiteralValue::*;
        let s = match &self.0 {
            #[cfg(feature = "dtype-i8")]
            Int8(v) => Int8Chunked::full("literal", *v, 1).into_series(),
            #[cfg(feature = "dtype-i16")]
            Int16(v) => Int16Chunked::full("literal", *v, 1).into_series(),
            Int32(v) => Int32Chunked::full("literal", *v, 1).into_series(),
            Int64(v) => Int64Chunked::full("literal", *v, 1).into_series(),
            #[cfg(feature = "dtype-u8")]
            UInt8(v) => UInt8Chunked::full("literal", *v, 1).into_series(),
            #[cfg(feature = "dtype-u16")]
            UInt16(v) => UInt16Chunked::full("literal", *v, 1).into_series(),
            UInt32(v) => UInt32Chunked::full("literal", *v, 1).into_series(),
            #[cfg(feature = "dtype-u64")]
            UInt64(v) => UInt64Chunked::full("literal", *v, 1).into_series(),
            Float32(v) => Float32Chunked::full("literal", *v, 1).into_series(),
            Float64(v) => Float64Chunked::full("literal", *v, 1).into_series(),
            Boolean(v) => BooleanChunked::full("literal", *v, 1).into_series(),
            Null => BooleanChunked::new_from_opt_slice("literal", &[None]).into_series(),
            Range {
                low,
                high,
                data_type,
            } => match data_type {
                DataType::Int32 => {
                    let low = *low as i32;
                    let high = *high as i32;
                    let ca: NoNull<Int32Chunked> = (low..high).collect();
                    ca.into_inner().into_series()
                }
                DataType::Int64 => {
                    let low = *low as i64;
                    let high = *high as i64;
                    let ca: NoNull<Int64Chunked> = (low..high).collect();
                    ca.into_inner().into_series()
                }
                DataType::UInt32 => {
                    if *low >= 0 || *high <= u32::MAX as i64 {
                        return Err(PolarsError::Other(
                            "range not within bounds of u32 type".into(),
                        ));
                    }
                    let low = *low as u32;
                    let high = *high as u32;
                    let ca: NoNull<UInt32Chunked> = (low..high).collect();
                    ca.into_inner().into_series()
                }
                dt => {
                    return Err(PolarsError::InvalidOperation(
                        format!("datatype {:?} not supported as range", dt).into(),
                    ))
                }
            },
            Utf8(v) => Utf8Chunked::full("literal", v, 1).into_series(),
            #[cfg(all(feature = "temporal", feature = "dtype-date64"))]
            DateTime(ndt) => {
                use polars_core::chunked_array::temporal::conversion::*;
                let timestamp = naive_datetime_to_date64(ndt);
                Date64Chunked::full("literal", timestamp, 1).into_series()
            }
            Series(series) => {
                let s = series.deref().clone();

                if s.len() != df.height() && s.len() != 1 {
                    return Err(PolarsError::ValueError(
                        "Series length should be equal to that of the DataFrame.".into(),
                    ));
                }
                s
            }
        };
        Ok(s)
    }

    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        use LiteralValue::*;
        let name = "literal";
        let field = match &self.0 {
            #[cfg(feature = "dtype-i8")]
            Int8(_) => Field::new(name, DataType::Int8),
            #[cfg(feature = "dtype-i16")]
            Int16(_) => Field::new(name, DataType::Int16),
            Int32(_) => Field::new(name, DataType::Int32),
            Int64(_) => Field::new(name, DataType::Int64),
            #[cfg(feature = "dtype-u8")]
            UInt8(_) => Field::new(name, DataType::UInt8),
            #[cfg(feature = "dtype-u16")]
            UInt16(_) => Field::new(name, DataType::UInt16),
            UInt32(_) => Field::new(name, DataType::UInt32),
            #[cfg(feature = "dtype-u64")]
            UInt64(_) => Field::new(name, DataType::UInt64),
            Float32(_) => Field::new(name, DataType::Float32),
            Float64(_) => Field::new(name, DataType::Float64),
            Boolean(_) => Field::new(name, DataType::Boolean),
            Utf8(_) => Field::new(name, DataType::Utf8),
            Null => Field::new(name, DataType::Null),
            Range { data_type, .. } => Field::new(name, data_type.clone()),
            #[cfg(all(feature = "temporal", feature = "dtype-date64"))]
            DateTime(_) => Field::new(name, DataType::Date64),
            Series(s) => s.field().clone(),
        };
        Ok(field)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct BinaryExpr {
    pub(crate) left: Arc<dyn PhysicalExpr>,
    pub(crate) op: Operator,
    pub(crate) right: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl BinaryExpr {
    pub fn new(
        left: Arc<dyn PhysicalExpr>,
        op: Operator,
        right: Arc<dyn PhysicalExpr>,
        expr: Expr,
    ) -> Self {
        Self {
            left,
            op,
            right,
            expr,
        }
    }
}

pub(crate) fn apply_operator(left: &Series, right: &Series, op: Operator) -> Result<Series> {
    match op {
        Operator::Gt => Ok(ChunkCompare::<&Series>::gt(left, right).into_series()),
        Operator::GtEq => Ok(ChunkCompare::<&Series>::gt_eq(left, right).into_series()),
        Operator::Lt => Ok(ChunkCompare::<&Series>::lt(left, right).into_series()),
        Operator::LtEq => Ok(ChunkCompare::<&Series>::lt_eq(left, right).into_series()),
        Operator::Eq => Ok(ChunkCompare::<&Series>::eq(left, right).into_series()),
        Operator::NotEq => Ok(ChunkCompare::<&Series>::neq(left, right).into_series()),
        Operator::Plus => Ok(left + right),
        Operator::Minus => Ok(left - right),
        Operator::Multiply => Ok(left * right),
        Operator::Divide => Ok(left / right),
        Operator::And => Ok((left.bool()? & right.bool()?).into_series()),
        Operator::Or => Ok((left.bool()? | right.bool()?).into_series()),
        Operator::Modulus => Ok(left % right),
    }
}

impl PhysicalExpr for BinaryExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let lhs = self.left.evaluate(df)?;
        let rhs = self.right.evaluate(df)?;
        apply_operator(&lhs, &rhs, self.op)
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        todo!()
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct ColumnExpr(Arc<String>, Expr);

impl ColumnExpr {
    pub fn new(name: Arc<String>, expr: Expr) -> Self {
        Self(name, expr)
    }
}

impl PhysicalExpr for ColumnExpr {
    fn as_expression(&self) -> &Expr {
        &self.1
    }
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let column = match &**self.0 {
            "" => df.select_at_idx(0).ok_or_else(|| {
                PolarsError::NoData("could not select a column from an empty DataFrame".into())
            })?,
            _ => df.column(&self.0)?,
        };
        Ok(column.clone())
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = input_schema.field_with_name(&self.0).map(|f| f.clone())?;
        Ok(field)
    }
}

pub struct SortExpr {
    pub(crate) physical_expr: Arc<dyn PhysicalExpr>,
    pub(crate) reverse: bool,
    expr: Expr,
}

impl SortExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, reverse: bool, expr: Expr) -> Self {
        Self {
            physical_expr,
            reverse,
            expr,
        }
    }
}

impl PhysicalExpr for SortExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.physical_expr.evaluate(df)?;
        Ok(series.sort(self.reverse))
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        let (series, groups) = self.physical_expr.evaluate_on_groups(df, groups)?;

        let groups = groups
            .iter()
            .map(|(_first, idx)| {
                // Safety:
                // Group tuples are always in bounds
                let group =
                    unsafe { series.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize)) };

                let sorted_idx = group.argsort(self.reverse);

                let new_idx: Vec<_> = sorted_idx
                    .cont_slice()
                    .unwrap()
                    .iter()
                    .map(|&i| {
                        debug_assert!(idx.get(i as usize).is_some());
                        unsafe { *idx.get_unchecked(i as usize) }
                    })
                    .collect();
                (new_idx[0], new_idx)
            })
            .collect();

        Ok((series, Cow::Owned(groups)))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.physical_expr.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct SortByExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) by: Arc<dyn PhysicalExpr>,
    pub(crate) reverse: bool,
    pub(crate) expr: Expr,
}

impl SortByExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        by: Arc<dyn PhysicalExpr>,
        reverse: bool,
        expr: Expr,
    ) -> Self {
        Self {
            input,
            by,
            reverse,
            expr,
        }
    }
}

impl PhysicalExpr for SortByExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.input.evaluate(df)?;
        let series_sort_by = self.by.evaluate(df)?;
        let sorted_idx = series_sort_by.argsort(self.reverse);

        // Safety:
        // sorted index are within bounds
        unsafe { series.take_unchecked(&sorted_idx) }
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        let (series, _) = self.input.evaluate_on_groups(df, groups)?;
        let (series_sort_by, groups) = self.by.evaluate_on_groups(df, groups)?;

        let groups = groups
            .iter()
            .map(|(_first, idx)| {
                // Safety:
                // Group tuples are always in bounds
                let group = unsafe {
                    series_sort_by.take_iter_unchecked(&mut idx.iter().map(|i| *i as usize))
                };

                let sorted_idx = group.argsort(self.reverse);

                let new_idx: Vec<_> = sorted_idx
                    .cont_slice()
                    .unwrap()
                    .iter()
                    .map(|&i| {
                        debug_assert!(idx.get(i as usize).is_some());
                        unsafe { *idx.get_unchecked(i as usize) }
                    })
                    .collect();
                (new_idx[0], new_idx)
            })
            .collect();

        Ok((series, Cow::Owned(groups)))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct NotExpr(Arc<dyn PhysicalExpr>, Expr);

impl NotExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, expr: Expr) -> Self {
        Self(physical_expr, expr)
    }
}
impl PhysicalExpr for NotExpr {
    fn as_expression(&self) -> &Expr {
        &self.1
    }

    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.0.evaluate(df)?;
        if let Ok(ca) = series.bool() {
            Ok((!ca).into_series())
        } else {
            Err(PolarsError::InvalidOperation(
                format!("NotExpr expected a boolean type, got: {:?}", series).into(),
            ))
        }
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        Ok(Field::new("not", DataType::Boolean))
    }
}

pub struct AliasExpr {
    pub(crate) physical_expr: Arc<dyn PhysicalExpr>,
    pub(crate) name: Arc<String>,
    expr: Expr,
}

impl AliasExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, name: Arc<String>, expr: Expr) -> Self {
        Self {
            physical_expr,
            name,
            expr,
        }
    }
}

impl PhysicalExpr for AliasExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let mut series = self.physical_expr.evaluate(df)?;
        series.rename(&self.name);
        Ok(series)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        Ok(Field::new(
            &self.name,
            self.physical_expr
                .to_field(input_schema)?
                .data_type()
                .clone(),
        ))
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct IsNullExpr {
    physical_expr: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl IsNullExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, expr: Expr) -> Self {
        Self {
            physical_expr,
            expr,
        }
    }
}

impl PhysicalExpr for IsNullExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.physical_expr.evaluate(df)?;
        Ok(series.is_null().into_series())
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        Ok(Field::new("is_null", DataType::Boolean))
    }
}

pub struct IsNotNullExpr {
    physical_expr: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl IsNotNullExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, expr: Expr) -> Self {
        Self {
            physical_expr,
            expr,
        }
    }
}

impl PhysicalExpr for IsNotNullExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.physical_expr.evaluate(df)?;
        Ok(series.is_not_null().into_series())
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        Ok(Field::new("is_not_null", DataType::Boolean))
    }
}

pub(crate) struct AggregationExpr {
    pub(crate) expr: Arc<dyn PhysicalExpr>,
    pub(crate) agg_type: GroupByMethod,
}

impl AggregationExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, agg_type: GroupByMethod) -> Self {
        Self { expr, agg_type }
    }
}

impl PhysicalExpr for AggregationExpr {
    fn evaluate(&self, _df: &DataFrame) -> Result<Series> {
        unimplemented!()
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = self.expr.to_field(input_schema)?;
        let new_name = fmt_groupby_column(field.name(), self.agg_type);
        Ok(Field::new(&new_name, field.data_type().clone()))
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct AggQuantileExpr {
    pub(crate) expr: Arc<dyn PhysicalExpr>,
    pub(crate) quantile: f64,
}

impl AggQuantileExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, quantile: f64) -> Self {
        Self { expr, quantile }
    }
}

impl PhysicalExpr for AggQuantileExpr {
    fn evaluate(&self, _df: &DataFrame) -> Result<Series> {
        unimplemented!()
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = self.expr.to_field(input_schema)?;
        let new_name = fmt_groupby_column(field.name(), GroupByMethod::Quantile(self.quantile));
        Ok(Field::new(&new_name, field.data_type().clone()))
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct CastExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) data_type: DataType,
}

impl CastExpr {
    pub fn new(input: Arc<dyn PhysicalExpr>, data_type: DataType) -> Self {
        Self { input, data_type }
    }
}

impl PhysicalExpr for CastExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.input.evaluate(df)?;
        series.cast_with_datatype(&self.data_type)
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct TernaryExpr {
    pub predicate: Arc<dyn PhysicalExpr>,
    pub truthy: Arc<dyn PhysicalExpr>,
    pub falsy: Arc<dyn PhysicalExpr>,
    pub expr: Expr,
}

impl PhysicalExpr for TernaryExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let mask_series = self.predicate.evaluate(df)?;
        let mask = mask_series.bool()?;
        let truthy = self.truthy.evaluate(df)?;
        let falsy = self.falsy.evaluate(df)?;
        truthy.zip_with(&mask, &falsy)
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.truthy.to_field(input_schema)
    }
}

pub struct ApplyExpr {
    pub input: Arc<dyn PhysicalExpr>,
    pub function: NoEq<Arc<dyn SeriesUdf>>,
    pub output_type: Option<DataType>,
    pub expr: Expr,
}

impl ApplyExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        function: NoEq<Arc<dyn SeriesUdf>>,
        output_type: Option<DataType>,
        expr: Expr,
    ) -> Self {
        ApplyExpr {
            input,
            function,
            output_type,
            expr,
        }
    }
}

impl PhysicalExpr for ApplyExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let input = self.input.evaluate(df)?;
        let in_name = input.name().to_string();
        let mut out = self.function.call_udf(input)?;
        if in_name != out.name() {
            out.rename(&in_name);
        }
        Ok(out)
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        match &self.output_type {
            Some(output_type) => {
                let input_field = self.input.to_field(input_schema)?;
                Ok(Field::new(input_field.name(), output_type.clone()))
            }
            None => self.input.to_field(input_schema),
        }
    }
    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct WindowExpr {
    /// the root column that the Function will be applied on.
    /// This will be used to create a smaller DataFrame to prevent taking unneeded columns by index
    pub(crate) group_column: Arc<String>,
    pub(crate) apply_column: Arc<String>,
    pub(crate) out_name: Arc<String>,
    /// A function Expr. i.e. Mean, Median, Max, etc.
    pub(crate) function: Expr,
}

impl PhysicalExpr for WindowExpr {
    // Note: this was first implemented with expression evaluation but this performed really bad.
    // Therefore we choose the groupby -> apply -> self join approach
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let gb = df
            .groupby(self.group_column.as_str())?
            .select(self.apply_column.as_str());

        let out = match &self.function {
            Expr::Udf { function, .. } => {
                let mut df = gb.agg_list()?;
                df.may_apply_at_idx(1, |s| function.call_udf(s.clone()))?;
                Ok(df)
            }
            Expr::Agg(agg) => match agg {
                AggExpr::Median(_) => gb.median(),
                AggExpr::Mean(_) => gb.mean(),
                AggExpr::Max(_) => gb.max(),
                AggExpr::Min(_) => gb.min(),
                AggExpr::Sum(_) => gb.sum(),
                AggExpr::First(_) => gb.first(),
                AggExpr::Last(_) => gb.last(),
                AggExpr::Count(_) => gb.count(),
                AggExpr::NUnique(_) => gb.n_unique(),
                AggExpr::Quantile { quantile, .. } => gb.quantile(*quantile),
                AggExpr::List(_) => gb.agg_list(),
                AggExpr::AggGroups(_) => gb.groups(),
                AggExpr::Std(_) => gb.std(),
                AggExpr::Var(_) => gb.var(),
            },
            _ => Err(PolarsError::Other(
                format!("{:?} function not supported", self.function).into(),
            )),
        }?;

        let mut out = df
            .select(self.group_column.as_str())?
            .left_join(&out, self.group_column.as_str(), &self.group_column)?
            .select_at_idx(1)
            .unwrap_or_else(|| {
                panic!(
                    "the aggregation function did not succeed on {}",
                    self.apply_column
                )
            })
            .clone();
        out.rename(self.out_name.as_str());
        Ok(out)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.function.to_field(input_schema, Context::Default)
    }
}

pub struct SliceExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) offset: i64,
    pub(crate) len: usize,
}

impl PhysicalExpr for SliceExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.input.evaluate(df)?;
        Ok(series.slice(self.offset, self.len))
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        let s = self.input.evaluate(df)?;

        let groups = groups
            .iter()
            .map(|(first, idx)| {
                let (offset, len) = slice_offsets(self.offset, self.len, s.len());
                (*first, idx[offset..offset + len].to_vec())
            })
            .collect();

        Ok((s, Cow::Owned(groups)))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub(crate) struct BinaryFunctionExpr {
    pub(crate) input_a: Arc<dyn PhysicalExpr>,
    pub(crate) input_b: Arc<dyn PhysicalExpr>,
    pub(crate) function: NoEq<Arc<dyn SeriesBinaryUdf>>,
    pub(crate) output_field: NoEq<Arc<dyn BinaryUdfOutputField>>,
}

impl PhysicalExpr for BinaryFunctionExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series_a = self.input_a.evaluate(df)?;
        let series_b = self.input_b.evaluate(df)?;

        self.function.call_udf(series_a, series_b).map(|mut s| {
            s.rename("binary_function");
            s
        })
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field_a = self.input_a.to_field(input_schema)?;
        let field_b = self.input_b.to_field(input_schema)?;
        self.output_field
            .get_field(input_schema, Context::Default, &field_a, &field_b)
            .ok_or_else(|| PolarsError::UnknownSchema("no field found".into()))
    }
    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct FilterExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) by: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl FilterExpr {
    pub fn new(input: Arc<dyn PhysicalExpr>, by: Arc<dyn PhysicalExpr>, expr: Expr) -> Self {
        Self { input, by, expr }
    }
}

impl PhysicalExpr for FilterExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.input.evaluate(df)?;
        let predicate = self.by.evaluate(df)?;
        series.filter(predicate.bool()?)
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        let s = self.input.evaluate(df)?;
        let predicate_s = self.by.evaluate(df)?;
        let predicate = predicate_s.bool()?;

        let groups = groups
            .par_iter()
            .map(|(first, idx)| {
                let idx: Vec<u32> = idx
                    .iter()
                    .filter_map(|i| match predicate.get(*i as usize) {
                        Some(true) => Some(*i),
                        _ => None,
                    })
                    .collect();

                (*idx.get(0).unwrap_or(first), idx)
            })
            .collect();

        Ok((s, Cow::Owned(groups)))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }
}
