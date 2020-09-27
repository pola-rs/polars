use crate::{lazy::prelude::*, prelude::*};
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
            Operator::Lt => {
                let a = apply_method_all_series!(left, lt_series, &right);
                Ok(a.into_series())
            }
            op => panic!(format!("Operator {:?} is not implemented", op)),
        }
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
}
