use crate::frame::group_by::{fmt_groupby_column, GroupByMethod, NumericAggSync};
use crate::lazy::physical_plan::AggPhysicalExpr;
use crate::utils::Xob;
use crate::{
    frame::group_by::{AggFirst, AggLast, AggList, AggNUnique, AggQuantile},
    lazy::prelude::*,
    prelude::*,
};
use std::sync::Arc;

pub struct LiteralExpr(pub ScalarValue);

impl LiteralExpr {
    pub fn new(value: ScalarValue) -> Self {
        Self(value)
    }
}

impl PhysicalExpr for LiteralExpr {
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
}

impl BinaryExpr {
    pub fn new(left: Arc<dyn PhysicalExpr>, op: Operator, right: Arc<dyn PhysicalExpr>) -> Self {
        Self { left, op, right }
    }
}

impl PhysicalExpr for BinaryExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let left = self.left.evaluate(df)?;
        let right = self.right.evaluate(df)?;
        match self.op {
            Operator::Gt => Ok(apply_method_all_series!(left, gt_series, &right).into_series()),
            Operator::GtEq => {
                Ok(apply_method_all_series!(left, gt_eq_series, &right).into_series())
            }
            Operator::Lt => Ok(apply_method_all_series!(left, lt_series, &right).into_series()),
            Operator::LtEq => {
                Ok(apply_method_all_series!(left, lt_eq_series, &right).into_series())
            }
            Operator::Eq => Ok(apply_method_all_series!(left, eq_series, &right).into_series()),
            Operator::NotEq => Ok(apply_method_all_series!(left, neq_series, &right).into_series()),
            Operator::Plus => Ok(left + right),
            Operator::Minus => Ok(left - right),
            Operator::Multiply => Ok(left * right),
            Operator::Divide => Ok(left / right),
            Operator::And => Ok((left.bool()? & right.bool()?).into_series()),
            Operator::Or => Ok((left.bool()? | right.bool()?).into_series()),
            Operator::Not => Ok(apply_method_all_series!(left, eq_series, &right).into_series()),
            Operator::Like => todo!(),
            Operator::NotLike => todo!(),
            Operator::Modulus => {
                apply_method_all_series!(left, remainder, &right).map(|ca| ca.into_series())
            }
        }
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        todo!()
    }
}

pub struct ColumnExpr(Arc<String>);

impl ColumnExpr {
    pub fn new(name: Arc<String>) -> Self {
        Self(name)
    }
}

impl PhysicalExpr for ColumnExpr {
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
    expr: Arc<dyn PhysicalExpr>,
    reverse: bool,
}

impl SortExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, reverse: bool) -> Self {
        Self { expr, reverse }
    }
}

impl PhysicalExpr for SortExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.expr.evaluate(df)?;
        Ok(series.sort(self.reverse))
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.expr.to_field(input_schema)
    }
}

pub struct NotExpr(Arc<dyn PhysicalExpr>);

impl NotExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self(expr)
    }
}
impl PhysicalExpr for NotExpr {
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
    expr: Arc<dyn PhysicalExpr>,
    name: Arc<String>,
}

impl AliasExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, name: Arc<String>) -> Self {
        Self { expr, name }
    }
}

impl PhysicalExpr for AliasExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let mut series = self.expr.evaluate(df)?;
        series.rename(&self.name);
        Ok(series)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        Ok(Field::new(
            &self.name,
            self.expr.to_field(input_schema)?.data_type().clone(),
            true,
        ))
    }

    fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
        Ok(self)
    }
}

impl AggPhysicalExpr for AliasExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(usize, Vec<usize>)]) -> Result<Option<Series>> {
        let agg_expr = self.expr.as_agg_expr()?;
        let opt_agg = agg_expr.evaluate(df, groups)?;
        Ok(opt_agg.map(|mut agg| {
            agg.rename(&self.name);
            agg
        }))
    }
}

pub struct IsNullExpr {
    expr: Arc<dyn PhysicalExpr>,
}

impl IsNullExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self { expr }
    }
}

impl PhysicalExpr for IsNullExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.expr.evaluate(df)?;
        Ok(series.is_null().into_series())
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        Ok(Field::new("is_null", ArrowDataType::Boolean, true))
    }
}

pub struct IsNotNullExpr {
    expr: Arc<dyn PhysicalExpr>,
}

impl IsNotNullExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self { expr }
    }
}

impl PhysicalExpr for IsNotNullExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.expr.evaluate(df)?;
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

macro_rules! impl_aggregation {
    ($expr_struct:ident, $agg_method:ident, $groupby_method_variant:expr, $finish_evaluate:ident) => {
        pub struct $expr_struct {
            expr: Arc<dyn PhysicalExpr>,
        }

        impl $expr_struct {
            pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
                Self { expr }
            }
        }

        impl PhysicalExpr for $expr_struct {
            fn evaluate(&self, _df: &DataFrame) -> Result<Series> {
                unimplemented!()
            }

            fn to_field(&self, input_schema: &Schema) -> Result<Field> {
                impl_to_field_for_agg!(self, input_schema, $groupby_method_variant)
            }

            fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
                Ok(self)
            }
        }

        impl AggPhysicalExpr for $expr_struct {
            fn evaluate(
                &self,
                df: &DataFrame,
                groups: &[(usize, Vec<usize>)],
            ) -> Result<Option<Series>> {
                let series = self.expr.evaluate(df)?;
                let new_name = fmt_groupby_column(series.name(), $groupby_method_variant);
                let opt_agg = apply_method_all_series!(series, $agg_method, groups);
                $finish_evaluate!(opt_agg, new_name)
            }
        }
    };
}

macro_rules! rename_and_cast_to_series {
    ($opt_agg:expr, $new_name:expr) => {{
        let opt_agg = $opt_agg.map(|mut agg| {
            agg.rename(&$new_name);
            agg.into_series()
        });
        Ok(opt_agg)
    }};
}
macro_rules! rename_and_cast_to_option {
    ($agg:expr, $new_name:expr) => {{
        let mut agg = $agg;
        agg.rename(&$new_name);
        Ok(Some(agg))
    }};
}

impl_aggregation!(
    AggMinExpr,
    agg_min,
    GroupByMethod::Min,
    rename_and_cast_to_series
);
impl_aggregation!(
    AggMaxExpr,
    agg_max,
    GroupByMethod::Max,
    rename_and_cast_to_series
);
impl_aggregation!(
    AggFirstExpr,
    agg_first,
    GroupByMethod::First,
    rename_and_cast_to_option
);
impl_aggregation!(
    AggLastExpr,
    agg_last,
    GroupByMethod::Last,
    rename_and_cast_to_option
);
impl_aggregation!(
    AggMedianExpr,
    agg_median,
    GroupByMethod::Median,
    rename_and_cast_to_series
);
impl_aggregation!(
    AggMeanExpr,
    agg_mean,
    GroupByMethod::Mean,
    rename_and_cast_to_series
);
impl_aggregation!(
    AggSumExpr,
    agg_sum,
    GroupByMethod::Sum,
    rename_and_cast_to_series
);

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
        let opt_agg = apply_method_all_series!(series, agg_quantile, groups, self.quantile);

        let opt_agg = opt_agg.map(|mut agg| {
            agg.rename(&new_name);
            agg.into_series()
        });

        Ok(opt_agg)
    }
}

pub struct AggGroupsExpr {
    expr: Arc<dyn PhysicalExpr>,
}

impl AggGroupsExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self { expr }
    }
}

impl PhysicalExpr for AggGroupsExpr {
    fn evaluate(&self, _df: &DataFrame) -> Result<Series> {
        unimplemented!()
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = self.expr.to_field(input_schema)?;
        let new_name = fmt_groupby_column(field.name(), GroupByMethod::Groups);
        let new_field = Field::new(&new_name, ArrowDataType::UInt32, field.is_nullable());
        Ok(new_field)
    }

    fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
        Ok(self)
    }
}

impl AggPhysicalExpr for AggGroupsExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(usize, Vec<usize>)]) -> Result<Option<Series>> {
        let series = self.expr.evaluate(df)?;
        let new_name = fmt_groupby_column(series.name(), GroupByMethod::Groups);

        let mut column: LargeListChunked = groups
            .iter()
            .map(|(_first, idx)| {
                let ca: Xob<UInt32Chunked> = idx.into_iter().map(|&v| v as u32).collect();
                ca.into_inner().into_series()
            })
            .collect();

        column.rename(&new_name);
        Ok(Some(column.into_series()))
    }
}

pub struct AggNUniqueExpr {
    expr: Arc<dyn PhysicalExpr>,
}

impl AggNUniqueExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self { expr }
    }
}

impl PhysicalExpr for AggNUniqueExpr {
    fn evaluate(&self, _df: &DataFrame) -> Result<Series> {
        unimplemented!()
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        impl_to_field_for_agg!(self, input_schema, GroupByMethod::List)
    }

    fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
        Ok(self)
    }
}

impl AggPhysicalExpr for AggListExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(usize, Vec<usize>)]) -> Result<Option<Series>> {
        let series = self.expr.evaluate(df)?;
        let new_name = fmt_groupby_column(series.name(), GroupByMethod::List);
        let opt_agg = apply_method_all_series!(series, agg_list, groups);

        let opt_agg = opt_agg.map(|mut agg| {
            agg.rename(&new_name);
            agg.into_series()
        });

        Ok(opt_agg)
    }
}

pub struct AggListExpr {
    expr: Arc<dyn PhysicalExpr>,
}

impl AggListExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self { expr }
    }
}

impl PhysicalExpr for AggListExpr {
    fn evaluate(&self, _df: &DataFrame) -> Result<Series> {
        unimplemented!()
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = self.expr.to_field(input_schema)?;
        let new_name = fmt_groupby_column(field.name(), GroupByMethod::List);
        let new_field = Field::new(&new_name, ArrowDataType::UInt32, field.is_nullable());
        Ok(new_field)
    }

    fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
        Ok(self)
    }
}

impl AggPhysicalExpr for AggNUniqueExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(usize, Vec<usize>)]) -> Result<Option<Series>> {
        let series = self.expr.evaluate(df)?;
        let new_name = fmt_groupby_column(series.name(), GroupByMethod::NUnique);
        let opt_agg = apply_method_all_series!(series, agg_n_unique, groups);

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
        if &in_name != out.name() {
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
