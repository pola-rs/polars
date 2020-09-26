use super::*;
use std::rc::Rc;

#[derive(Debug)]
pub struct LiteralExpr(pub ScalarValue);

impl LiteralExpr {
    pub fn new(value: ScalarValue) -> Self {
        Self(value)
    }
}

impl PhysicalExpr for LiteralExpr {
    fn evaluate(&self, ds: &DataStructure) -> Result<Series> {
        match &self.0 {
            // todo! implement single value chunked_arrays? Or allow comparison and arithemtic with
            //      ca of a single value
            ScalarValue::Int32(v) => Ok(Int32Chunked::full("literal", *v, ds.len()).into_series()),
            sv => panic!(format!("ScalarValue {:?} is not implemented", sv)),
        }
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
    fn evaluate(&self, ds: &DataStructure) -> Result<Series> {
        let left = self.left.evaluate(ds)?;
        let right = self.right.evaluate(ds)?;
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
pub struct ColumnExpr(String);

impl ColumnExpr {
    pub fn new(name: String) -> Self {
        Self(name)
    }
}

impl PhysicalExpr for ColumnExpr {
    fn evaluate(&self, ds: &DataStructure) -> Result<Series> {
        let df = ds.df_ref()?;
        let column = df.column(&self.0)?;
        Ok(column.clone())
    }
}
