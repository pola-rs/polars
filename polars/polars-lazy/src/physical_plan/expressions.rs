use crate::logical_plan::Context;
use crate::physical_plan::AggPhysicalExpr;
use crate::prelude::*;
use polars_arrow::array::ValueSize;
use polars_core::chunked_array::builder::get_list_builder;
use polars_core::frame::group_by::{fmt_groupby_column, GroupByMethod};
use polars_core::prelude::*;
use polars_core::utils::NoNull;
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
    fn evaluate(&self, _df: &DataFrame) -> Result<Series> {
        use LiteralValue::*;
        let s = match &self.0 {
            Int8(v) => Int8Chunked::full("literal", *v, 1).into_series(),
            Int16(v) => Int16Chunked::full("literal", *v, 1).into_series(),
            Int32(v) => Int32Chunked::full("literal", *v, 1).into_series(),
            Int64(v) => Int64Chunked::full("literal", *v, 1).into_series(),
            UInt8(v) => UInt8Chunked::full("literal", *v, 1).into_series(),
            UInt16(v) => UInt16Chunked::full("literal", *v, 1).into_series(),
            UInt32(v) => UInt32Chunked::full("literal", *v, 1).into_series(),
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
            #[cfg(feature = "temporal")]
            DateTime(ndt) => {
                use polars_core::chunked_array::temporal::conversion::*;
                let timestamp = naive_datetime_to_date64(ndt);
                Date64Chunked::full("literal", timestamp, 1).into_series()
            }
        };
        Ok(s)
    }

    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        use LiteralValue::*;
        let name = "literal";
        let field = match &self.0 {
            Int8(_) => Field::new(name, DataType::Int8),
            Int16(_) => Field::new(name, DataType::Int16),
            Int32(_) => Field::new(name, DataType::Int32),
            Int64(_) => Field::new(name, DataType::Int64),
            UInt8(_) => Field::new(name, DataType::UInt8),
            UInt16(_) => Field::new(name, DataType::UInt16),
            UInt32(_) => Field::new(name, DataType::UInt32),
            UInt64(_) => Field::new(name, DataType::UInt64),
            Float32(_) => Field::new(name, DataType::Float32),
            Float64(_) => Field::new(name, DataType::Float64),
            Boolean(_) => Field::new(name, DataType::Boolean),
            Utf8(_) => Field::new(name, DataType::Utf8),
            Null => Field::new(name, DataType::Null),
            Range { data_type, .. } => Field::new(name, data_type.clone()),
            #[cfg(feature = "temporal")]
            DateTime(_) => Field::new(name, DataType::Date64),
        };
        Ok(field)
    }
}

pub struct BinaryExpr {
    left: Arc<dyn PhysicalExpr>,
    op: Operator,
    right: Arc<dyn PhysicalExpr>,
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

impl PhysicalExpr for BinaryExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let lhs = self.left.evaluate(df)?;
        let rhs = self.right.evaluate(df)?;
        let left = &lhs;
        let right = &rhs;

        match self.op {
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
            Operator::Not => Ok(ChunkCompare::<&Series>::eq(left, right).into_series()),
            Operator::Like => todo!(),
            Operator::NotLike => todo!(),
            Operator::Modulus => Ok(left % right),
        }
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        todo!()
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
    physical_expr: Arc<dyn PhysicalExpr>,
    reverse: bool,
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
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.physical_expr.to_field(input_schema)
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
    physical_expr: Arc<dyn PhysicalExpr>,
    name: Arc<String>,
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

    fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
        Ok(self)
    }
}

impl AggPhysicalExpr for AliasExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(u32, Vec<u32>)]) -> Result<Option<Series>> {
        let agg_expr = self.physical_expr.as_agg_expr()?;
        let opt_agg = agg_expr.evaluate(df, groups)?;
        Ok(opt_agg.map(|mut agg| {
            agg.rename(&self.name);
            agg
        }))
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

// TODO: to_field for groups and n_unique is probably wrong as the Datatype changes to Uint32
macro_rules! impl_to_field_for_agg {
    ($self:ident, $input_schema:ident, $groupby_method_variant:expr) => {{
        let field = $self.expr.to_field($input_schema)?;
        let new_name = fmt_groupby_column(field.name(), $groupby_method_variant);
        Ok(Field::new(&new_name, field.data_type().clone()))
    }};
}

pub(crate) struct PhysicalAggExpr {
    expr: Arc<dyn PhysicalExpr>,
    agg_type: GroupByMethod,
}

impl PhysicalAggExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, agg_type: GroupByMethod) -> Self {
        Self { expr, agg_type }
    }
}

impl PhysicalExpr for PhysicalAggExpr {
    fn evaluate(&self, _df: &DataFrame) -> Result<Series> {
        unimplemented!()
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = self.expr.to_field(input_schema)?;
        let new_name = fmt_groupby_column(field.name(), self.agg_type);
        Ok(Field::new(&new_name, field.data_type().clone()))
    }

    fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
        Ok(self)
    }
}

fn rename_option_series(opt: Option<Series>, name: &str) -> Option<Series> {
    opt.map(|mut s| {
        s.rename(name);
        s
    })
}

impl AggPhysicalExpr for PhysicalAggExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(u32, Vec<u32>)]) -> Result<Option<Series>> {
        let series = self.expr.evaluate(df)?;
        let new_name = fmt_groupby_column(series.name(), self.agg_type);

        match self.agg_type {
            GroupByMethod::Min => {
                let agg_s = series.agg_min(groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Max => {
                let agg_s = series.agg_max(groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Median => {
                let agg_s = series.agg_median(groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Mean => {
                let agg_s = series.agg_mean(groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Sum => {
                let agg_s = series.agg_sum(groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Count => {
                let mut ca: NoNull<UInt32Chunked> =
                    groups.iter().map(|(_, g)| g.len() as u32).collect();
                ca.rename(&new_name);
                Ok(Some(ca.into_inner().into_series()))
            }
            GroupByMethod::First => {
                let mut agg_s = series.agg_first(groups);
                agg_s.rename(&new_name);
                Ok(Some(agg_s))
            }
            GroupByMethod::Last => {
                let mut agg_s = series.agg_last(groups);
                agg_s.rename(&new_name);
                Ok(Some(agg_s))
            }
            GroupByMethod::NUnique => {
                let opt_agg = series.agg_n_unique(groups);
                let opt_agg = opt_agg.map(|mut agg| {
                    agg.rename(&new_name);
                    agg.into_series()
                });
                Ok(opt_agg)
            }
            GroupByMethod::List => {
                let opt_agg = series.agg_list(groups);
                Ok(rename_option_series(opt_agg, &new_name))
            }
            GroupByMethod::Groups => {
                let mut column: ListChunked = groups
                    .iter()
                    .map(|(_first, idx)| {
                        let ca: NoNull<UInt32Chunked> = idx.iter().map(|&v| v as u32).collect();
                        ca.into_inner().into_series()
                    })
                    .collect();

                column.rename(&new_name);
                Ok(Some(column.into_series()))
            }
            GroupByMethod::Std => {
                let agg_s = series.agg_std(groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Var => {
                let agg_s = series.agg_var(groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Quantile(_) => {
                unimplemented!()
            }
        }
    }

    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &[(u32, Vec<u32>)],
    ) -> Result<Option<Vec<Series>>> {
        match self.agg_type {
            GroupByMethod::Mean => {
                let series = self.expr.evaluate(df)?;
                let mut new_name = fmt_groupby_column(series.name(), self.agg_type);
                let agg_s = series.agg_sum(groups);

                if let Some(mut agg_s) = agg_s {
                    agg_s.rename(&new_name);
                    new_name.push_str("__POLARS_MEAN_COUNT");
                    let ca: NoNull<UInt32Chunked> =
                        groups.iter().map(|t| t.1.len() as u32).collect();
                    let mut count_s = ca.into_inner().into_series();
                    count_s.rename(&new_name);
                    Ok(Some(vec![agg_s, count_s]))
                } else {
                    Ok(None)
                }
            }
            GroupByMethod::List => {
                let series = self.expr.evaluate(df)?;
                let new_name = fmt_groupby_column(series.name(), self.agg_type);
                let opt_agg = series.agg_list(groups);
                Ok(opt_agg.map(|mut s| {
                    s.rename(&new_name);
                    vec![s]
                }))
            }
            _ => AggPhysicalExpr::evaluate(self, df, groups).map(|opt| opt.map(|s| vec![s])),
        }
    }

    fn evaluate_partitioned_final(
        &self,
        final_df: &DataFrame,
        groups: &[(u32, Vec<u32>)],
    ) -> Result<Option<Series>> {
        match self.agg_type {
            GroupByMethod::Mean => {
                let series = self.expr.evaluate(final_df)?;
                let count_name = format!("{}__POLARS_MEAN_COUNT", series.name());
                let new_name = fmt_groupby_column(series.name(), self.agg_type);
                let count = final_df.column(&count_name).unwrap();
                // divide by the count
                let series = &series / count;
                let agg_s = series.agg_sum(groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::List => {
                let series = self.expr.evaluate(final_df)?;
                let ca = series.list().unwrap();
                let new_name = fmt_groupby_column(ca.name(), self.agg_type);

                let values_type = match ca.dtype() {
                    DataType::List(dt) => DataType::from(dt),
                    _ => unreachable!(),
                };

                let mut builder =
                    get_list_builder(&values_type, ca.get_values_size(), ca.len(), &new_name);
                for (_, idx) in groups {
                    // Safety
                    // The indexes of the groupby operation are never out of bounds
                    let ca = unsafe { ca.take_unchecked(idx.iter().map(|i| *i as usize).into()) };
                    let s = ca.explode_and_offsets()?.0;
                    builder.append_series(&s);
                }
                let out = builder.finish();
                Ok(Some(out.into_series()))
            }
            _ => AggPhysicalExpr::evaluate(self, final_df, groups),
        }
    }
}

pub struct AggQuantileExpr {
    expr: Arc<dyn PhysicalExpr>,
    quantile: f64,
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
        impl_to_field_for_agg!(self, input_schema, GroupByMethod::Quantile(self.quantile))
    }

    fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
        Ok(self)
    }
}

impl AggPhysicalExpr for AggQuantileExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(u32, Vec<u32>)]) -> Result<Option<Series>> {
        let series = self.expr.evaluate(df)?;
        let new_name = fmt_groupby_column(series.name(), GroupByMethod::Quantile(self.quantile));
        let opt_agg = series.agg_quantile(groups, self.quantile);

        let opt_agg = opt_agg.map(|mut agg| {
            agg.rename(&new_name);
            agg.into_series()
        });

        Ok(opt_agg)
    }
}

pub struct CastExpr {
    input: Arc<dyn PhysicalExpr>,
    data_type: DataType,
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
    fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
        Ok(self)
    }
}

impl AggPhysicalExpr for ApplyExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(u32, Vec<u32>)]) -> Result<Option<Series>> {
        match self.input.as_agg_expr() {
            // layer below is also an aggregation expr.
            Ok(expr) => {
                let aggregated = expr.evaluate(df, groups)?;
                let out = aggregated.map(|s| self.function.call_udf(s));
                out.transpose()
            }
            Err(_) => {
                let series = self.input.evaluate(df)?;
                series
                    .agg_list(groups)
                    .map(|s| {
                        let s = self.function.call_udf(s);
                        s.map(|mut s| {
                            s.rename(series.name());
                            s
                        })
                    })
                    .map_or(Ok(None), |v| v.map(Some))
            }
        }
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
        self.function.to_field(input_schema, Context::Other)
    }
}

pub struct SliceExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) offset: isize,
    pub(crate) len: usize,
}

impl SliceExpr {
    fn slice_series(&self, series: &Series) -> Result<Series> {
        let series_len = series.len() as isize;
        let offset = if self.offset >= 0 {
            self.offset as usize
        } else {
            series_len.checked_sub(self.offset).ok_or_else(|| {
                PolarsError::OutOfBounds(
                    format!(
                        "offset {} is larger than Series length of {}",
                        self.offset, series_len
                    )
                    .into(),
                )
            })? as usize
        };
        let len = std::cmp::min(series_len as usize - offset, self.len);
        series.slice(offset, len)
    }
}

impl PhysicalExpr for SliceExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.input.evaluate(df)?;
        self.slice_series(&series)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
        Ok(self)
    }
}

impl AggPhysicalExpr for SliceExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(u32, Vec<u32>)]) -> Result<Option<Series>> {
        let s = self.input.evaluate(df)?;
        let agg_s = s.agg_list(groups);
        let out = agg_s.map(|s| {
            s.list()
                .unwrap()
                .into_iter()
                .map(|opt_s| match opt_s {
                    None => None,
                    Some(s) => {
                        let r = self.slice_series(&s);
                        r.ok()
                    }
                })
                .collect::<ListChunked>()
                .into_series()
        });
        Ok(out)
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
            .get_field(input_schema, Context::Other, &field_a, &field_b)
            .ok_or_else(|| PolarsError::UnknownSchema("no field found".into()))
    }
    fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
        Ok(self)
    }
}

impl AggPhysicalExpr for BinaryFunctionExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(u32, Vec<u32>)]) -> Result<Option<Series>> {
        let a = self.input_a.evaluate(df)?;
        let b = self.input_b.evaluate(df)?;

        let agg_a = a.agg_list(groups).expect("no data?");
        let agg_b = b.agg_list(groups).expect("no data?");

        let mut all_unit_length = true;

        let ca = agg_a
            .list()
            .unwrap()
            .into_iter()
            .zip(agg_b.list().unwrap())
            .map(|(opt_a, opt_b)| match (opt_a, opt_b) {
                (Some(a), Some(b)) => {
                    let out = self.function.call_udf(a, b).ok();

                    if let Some(s) = &out {
                        if s.len() != 1 {
                            all_unit_length = false;
                        }
                    }
                    out
                }
                _ => None,
            })
            .collect::<ListChunked>();

        if all_unit_length {
            return Ok(Some(ca.explode()?));
        }
        Ok(Some(ca.into_series()))
    }
}
