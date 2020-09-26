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
    fn data_type(&self, input_schema: &Schema) -> Result<ArrowDataType> {
        unimplemented!()
    }

    fn evaluate(&self, ds: &DataStructure) -> Result<Series> {
        unimplemented!()
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
    fn data_type(&self, input_schema: &Schema) -> Result<ArrowDataType> {
        unimplemented!()
    }

    fn evaluate(&self, ds: &DataStructure) -> Result<Series> {
        unimplemented!()
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
    fn data_type(&self, input_schema: &Schema) -> Result<ArrowDataType> {
        unimplemented!()
    }

    fn evaluate(&self, ds: &DataStructure) -> Result<Series> {
        unimplemented!()
    }
}
