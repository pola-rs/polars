use crate::frame::group_by::{fmt_groupby_column, GroupByMethod, NumericAggSync};
use crate::lazy::physical_plan::AggPhysicalExpr;
use crate::utils::Xob;
use crate::{
    frame::group_by::{AggFirst, AggLast, AggNUnique, AggQuantile},
    lazy::prelude::*,
    prelude::*,
};
use std::rc::Rc;

#[derive(Debug)]
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

#[derive(Debug)]
pub struct BinaryExpr {
    left: Rc<dyn PhysicalExpr>,
    op: Operator,
    right: Rc<dyn PhysicalExpr>,
}

impl BinaryExpr {
    pub fn new(left: Rc<dyn PhysicalExpr>, op: Operator, right: Rc<dyn PhysicalExpr>) -> Self {
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

#[derive(Debug)]
pub struct ColumnExpr(Rc<String>);

impl ColumnExpr {
    pub fn new(name: Rc<String>) -> Self {
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

#[derive(Debug)]
pub struct SortExpr {
    expr: Rc<dyn PhysicalExpr>,
    reverse: bool,
}

impl SortExpr {
    pub fn new(expr: Rc<dyn PhysicalExpr>, reverse: bool) -> Self {
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

#[derive(Debug)]
pub struct NotExpr(Rc<dyn PhysicalExpr>);

impl NotExpr {
    pub fn new(expr: Rc<dyn PhysicalExpr>) -> Self {
        Self(expr)
    }
}
impl PhysicalExpr for NotExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series> {
        let series = self.0.evaluate(df)?;
        if let Series::Bool(ca) = series {
            Ok((!ca).into_series())
        } else {
            Err(PolarsError::InvalidOperation)
        }
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        Ok(Field::new("not", ArrowDataType::Boolean, true))
    }
}

#[derive(Debug)]
pub struct AliasExpr {
    expr: Rc<dyn PhysicalExpr>,
    name: Rc<String>,
}

impl AliasExpr {
    pub fn new(expr: Rc<dyn PhysicalExpr>, name: Rc<String>) -> Self {
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
    fn evaluate(&self, df: &DataFrame, groups: &[(usize, Vec<usize>)]) -> Result<Series> {
        let agg_expr = self.expr.as_agg_expr()?;
        let mut agg = agg_expr.evaluate(df, groups)?;
        agg.rename(&self.name);
        Ok(agg)
    }
}

#[derive(Debug)]
pub struct IsNullExpr {
    expr: Rc<dyn PhysicalExpr>,
}

impl IsNullExpr {
    pub fn new(expr: Rc<dyn PhysicalExpr>) -> Self {
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

#[derive(Debug)]
pub struct IsNotNullExpr {
    expr: Rc<dyn PhysicalExpr>,
}

impl IsNotNullExpr {
    pub fn new(expr: Rc<dyn PhysicalExpr>) -> Self {
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
    ($expr_struct:ident, $agg_method:ident, $groupby_method_variant:expr, $mayunpack:ident) => {
        #[derive(Debug)]
        pub struct $expr_struct {
            expr: Rc<dyn PhysicalExpr>,
        }

        impl $expr_struct {
            pub fn new(expr: Rc<dyn PhysicalExpr>) -> Self {
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
            fn evaluate(&self, df: &DataFrame, groups: &[(usize, Vec<usize>)]) -> Result<Series> {
                let series = self.expr.evaluate(df)?;
                let new_name = fmt_groupby_column(series.name(), $groupby_method_variant);
                let opt_agg = apply_method_all_series!(series, $agg_method, groups);
                let mut agg = $mayunpack!(opt_agg);
                agg.rename(&new_name);
                Ok(agg.into_series())
            }
        }
    };
}

macro_rules! unpack {
    ($opt_agg:expr) => {{
        let agg = $opt_agg.expect("could not unpack aggregation result");
        agg
    }};
}
macro_rules! identity {
    ($opt_agg:expr) => {{
        $opt_agg
    }};
}

impl_aggregation!(AggMinExpr, agg_min, GroupByMethod::Min, unpack);
impl_aggregation!(AggMaxExpr, agg_max, GroupByMethod::Max, unpack);
impl_aggregation!(AggFirstExpr, agg_first, GroupByMethod::First, identity);
impl_aggregation!(AggLastExpr, agg_last, GroupByMethod::Last, identity);
impl_aggregation!(AggMedianExpr, agg_median, GroupByMethod::Median, unpack);
impl_aggregation!(AggMeanExpr, agg_mean, GroupByMethod::Mean, unpack);
impl_aggregation!(AggSumExpr, agg_sum, GroupByMethod::Sum, unpack);

#[derive(Debug)]
pub struct AggQuantileExpr {
    expr: Rc<dyn PhysicalExpr>,
    quantile: f64,
}

impl AggQuantileExpr {
    pub fn new(expr: Rc<dyn PhysicalExpr>, quantile: f64) -> Self {
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
    fn evaluate(&self, df: &DataFrame, groups: &[(usize, Vec<usize>)]) -> Result<Series> {
        let series = self.expr.evaluate(df)?;
        let new_name = fmt_groupby_column(series.name(), GroupByMethod::Quantile(self.quantile));
        let opt_agg = apply_method_all_series!(series, agg_quantile, groups, self.quantile);
        let mut agg = opt_agg.expect("could not unpack aggregation result");
        agg.rename(&new_name);
        Ok(agg)
    }
}

#[derive(Debug)]
pub struct AggGroupsExpr {
    expr: Rc<dyn PhysicalExpr>,
}

impl AggGroupsExpr {
    pub fn new(expr: Rc<dyn PhysicalExpr>) -> Self {
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
    fn evaluate(&self, df: &DataFrame, groups: &[(usize, Vec<usize>)]) -> Result<Series> {
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
        Ok(column.into_series())
    }
}

#[derive(Debug)]
pub struct AggNUniqueExpr {
    expr: Rc<dyn PhysicalExpr>,
}

impl AggNUniqueExpr {
    pub fn new(expr: Rc<dyn PhysicalExpr>) -> Self {
        Self { expr }
    }
}

impl PhysicalExpr for AggNUniqueExpr {
    fn evaluate(&self, _df: &DataFrame) -> Result<Series> {
        unimplemented!()
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = self.expr.to_field(input_schema)?;
        let new_name = fmt_groupby_column(field.name(), GroupByMethod::NUnique);
        let new_field = Field::new(&new_name, ArrowDataType::UInt32, field.is_nullable());
        Ok(new_field)
    }

    fn as_agg_expr(&self) -> Result<&dyn AggPhysicalExpr> {
        Ok(self)
    }
}

impl AggPhysicalExpr for AggNUniqueExpr {
    fn evaluate(&self, df: &DataFrame, groups: &[(usize, Vec<usize>)]) -> Result<Series> {
        let series = self.expr.evaluate(df)?;
        let new_name = fmt_groupby_column(series.name(), GroupByMethod::NUnique);
        let opt_agg = apply_method_all_series!(series, agg_n_unique, groups);
        let mut agg = unpack!(opt_agg);
        agg.rename(&new_name);
        Ok(agg.into_series())
    }
}
