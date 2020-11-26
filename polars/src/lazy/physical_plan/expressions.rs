use crate::frame::group_by::{fmt_groupby_column, GroupByMethod, NumericAggSync};
use crate::lazy::physical_plan::AggPhysicalExpr;
use crate::utils::Xob;
use crate::{
    frame::group_by::{AggFirst, AggLast, AggList, AggNUnique, AggQuantile},
    lazy::prelude::*,
    prelude::*,
};
use std::sync::Arc;

pub struct LiteralExpr(pub ScalarValue, Expr);

impl LiteralExpr {
    pub fn new(value: ScalarValue, expr: Expr) -> Self {
        Self(value, expr)
    }
}

impl PhysicalExpr for LiteralExpr {
    fn as_expression(&self) -> &Expr {
        &self.1
    }
    fn evaluate(&self, _df: &DataFrame) -> Result<Series> {
        use ScalarValue::*;
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
            Utf8(v) => Utf8Chunked::full("literal", v, 1).into_series(),
        };
        Ok(s)
    }

    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        use ScalarValue::*;
        let name = "literal";
        let field = match &self.0 {
            Int8(_) => Field::new(name, ArrowDataType::Int8, true),
            Int16(_) => Field::new(name, ArrowDataType::Int16, true),
            Int32(_) => Field::new(name, ArrowDataType::Int32, true),
            Int64(_) => Field::new(name, ArrowDataType::Int64, true),
            UInt8(_) => Field::new(name, ArrowDataType::UInt8, true),
            UInt16(_) => Field::new(name, ArrowDataType::UInt16, true),
            UInt32(_) => Field::new(name, ArrowDataType::UInt32, true),
            UInt64(_) => Field::new(name, ArrowDataType::UInt64, true),
            Float32(_) => Field::new(name, ArrowDataType::Float32, true),
            Float64(_) => Field::new(name, ArrowDataType::Float64, true),
            Boolean(_) => Field::new(name, ArrowDataType::Boolean, true),
            Utf8(_) => Field::new(name, ArrowDataType::Utf8, true),
            Null => Field::new(name, ArrowDataType::Null, true),
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
        let left = self.left.evaluate(df)?;
        let right = self.right.evaluate(df)?;
        match self.op {
            Operator::Gt => Ok(ChunkCompare::<&Series>::gt(&left, &right).into()),
            Operator::GtEq => Ok(ChunkCompare::<&Series>::gt_eq(&left, &right).into()),
            Operator::Lt => Ok(ChunkCompare::<&Series>::lt(&left, &right).into()),
            Operator::LtEq => Ok(ChunkCompare::<&Series>::lt_eq(&left, &right).into()),
            Operator::Eq => Ok(ChunkCompare::<&Series>::eq(&left, &right).into()),
            Operator::NotEq => Ok(ChunkCompare::<&Series>::neq(&left, &right).into()),
            Operator::Plus => Ok(left + right),
            Operator::Minus => Ok(left - right),
            Operator::Multiply => Ok(left * right),
            Operator::Divide => Ok(left / right),
            Operator::And => Ok((left.bool()? & right.bool()?).into_series()),
            Operator::Or => Ok((left.bool()? | right.bool()?).into_series()),
            Operator::Not => Ok(ChunkCompare::<&Series>::eq(&left, &right).into()),
            Operator::Like => todo!(),
            Operator::NotLike => todo!(),
            Operator::Modulus => {
                apply_method_all_arrow_series!(left, remainder, &right).map(|ca| ca.into_series())
            }
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
        let column = df.column(&self.0)?;
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
        if let Series::Bool(ca) = series {
            Ok((!ca).into_series())
        } else {
            Err(PolarsError::InvalidOperation(
                format!("NotExpr expected a boolean type, got: {:?}", series).into(),
            ))
        }
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        Ok(Field::new("not", ArrowDataType::Boolean, true))
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
            true,
        ))
    }

    fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
        Ok(self)
    }
}

impl AggPhysicalExpr for AliasExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(usize, Vec<usize>)]) -> Result<Option<Series>> {
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
        Ok(Field::new("is_null", ArrowDataType::Boolean, true))
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
        Ok(Field::new("is_not_null", ArrowDataType::Boolean, true))
    }
}

// TODO: to_field for groups and n_unique is probably wrong as the Datatype changes to Uint32
macro_rules! impl_to_field_for_agg {
    ($self:ident, $input_schema:ident, $groupby_method_variant:expr) => {{
        let field = $self.expr.to_field($input_schema)?;
        let new_name = fmt_groupby_column(field.name(), $groupby_method_variant);
        Ok(Field::new(
            &new_name,
            field.data_type().clone(),
            field.is_nullable(),
        ))
    }};
}

pub(crate) struct AggExpr {
    expr: Arc<dyn PhysicalExpr>,
    agg_type: GroupByMethod,
}

impl AggExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, agg_type: GroupByMethod) -> Self {
        Self { expr, agg_type }
    }
}

impl PhysicalExpr for AggExpr {
    fn evaluate(&self, _df: &DataFrame) -> Result<Series> {
        unimplemented!()
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = self.expr.to_field(input_schema)?;
        let new_name = fmt_groupby_column(field.name(), self.agg_type);
        Ok(Field::new(
            &new_name,
            field.data_type().clone(),
            field.is_nullable(),
        ))
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

impl AggPhysicalExpr for AggExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(usize, Vec<usize>)]) -> Result<Option<Series>> {
        let series = self.expr.evaluate(df)?;
        let new_name = fmt_groupby_column(series.name(), self.agg_type);

        match self.agg_type {
            GroupByMethod::Min => {
                let agg_s = apply_method_all_arrow_series!(series, agg_min, groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Max => {
                let agg_s = apply_method_all_arrow_series!(series, agg_max, groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Median => {
                let agg_s = apply_method_all_arrow_series!(series, agg_median, groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Mean => {
                let agg_s = apply_method_all_arrow_series!(series, agg_mean, groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Sum => {
                let agg_s = apply_method_all_arrow_series!(series, agg_sum, groups);
                Ok(rename_option_series(agg_s, &new_name))
            }
            GroupByMethod::Count => {
                let mut ca: Xob<UInt32Chunked> =
                    groups.iter().map(|(_, g)| g.len() as u32).collect();
                ca.rename(&new_name);
                Ok(Some(ca.into_inner().into()))
            }
            GroupByMethod::First => {
                let mut agg_s = apply_method_all_arrow_series!(series, agg_first, groups);
                agg_s.rename(&new_name);
                Ok(Some(agg_s))
            }
            GroupByMethod::Last => {
                let mut agg_s = apply_method_all_arrow_series!(series, agg_last, groups);
                agg_s.rename(&new_name);
                Ok(Some(agg_s))
            }
            GroupByMethod::NUnique => {
                let opt_agg = apply_method_all_arrow_series!(series, agg_n_unique, groups);
                let opt_agg = opt_agg.map(|mut agg| {
                    agg.rename(&new_name);
                    agg.into_series()
                });
                Ok(opt_agg)
            }
            GroupByMethod::List => {
                let opt_agg = apply_method_all_arrow_series!(series, agg_list, groups);
                Ok(rename_option_series(opt_agg, &new_name))
            }
            GroupByMethod::Groups => {
                let mut column: ListChunked = groups
                    .iter()
                    .map(|(_first, idx)| {
                        let ca: Xob<UInt32Chunked> = idx.iter().map(|&v| v as u32).collect();
                        ca.into_inner().into_series()
                    })
                    .collect();

                column.rename(&new_name);
                Ok(Some(column.into_series()))
            }
            _ => unimplemented!(),
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
    fn evaluate(&self, df: &DataFrame, groups: &[(usize, Vec<usize>)]) -> Result<Option<Series>> {
        let series = self.expr.evaluate(df)?;
        let new_name = fmt_groupby_column(series.name(), GroupByMethod::Quantile(self.quantile));
        let opt_agg = apply_method_all_arrow_series!(series, agg_quantile, groups, self.quantile);

        let opt_agg = opt_agg.map(|mut agg| {
            agg.rename(&new_name);
            agg.into_series()
        });

        Ok(opt_agg)
    }
}

pub struct CastExpr {
    expr: Arc<dyn PhysicalExpr>,
    data_type: ArrowDataType,
}

impl CastExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, data_type: ArrowDataType) -> Self {
        Self { expr, data_type }
    }
}

impl PhysicalExpr for CastExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.expr.evaluate(df)?;
        series.cast_with_arrow_datatype(&self.data_type)
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.expr.to_field(input_schema)
    }
}

pub struct TernaryExpr {
    pub predicate: Arc<dyn PhysicalExpr>,
    pub truthy: Arc<dyn PhysicalExpr>,
    pub falsy: Arc<dyn PhysicalExpr>,
}

impl PhysicalExpr for TernaryExpr {
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
    pub function: Arc<dyn Udf>,
    pub output_type: Option<ArrowDataType>,
}

impl ApplyExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        function: Arc<dyn Udf>,
        output_type: Option<ArrowDataType>,
    ) -> Self {
        ApplyExpr {
            input,
            function,
            output_type,
        }
    }
}

impl PhysicalExpr for ApplyExpr {
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
                Ok(Field::new(
                    input_field.name(),
                    output_type.clone(),
                    input_field.is_nullable(),
                ))
            }
            None => self.input.to_field(input_schema),
        }
    }
}
