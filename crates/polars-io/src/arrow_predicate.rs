#[derive(Debug, Clone)]
pub enum ArrowPredicate {
    /// Reference to a column by name
    Column(String),
    /// Literal value
    Literal(LiteralValue),
    /// Binary comparison: left OP right
    Comparison {
        left: Box<ArrowPredicate>,
        op: ComparisonOp,
        right: Box<ArrowPredicate>,
    },
    /// Logical AND of two predicates
    And(Box<ArrowPredicate>, Box<ArrowPredicate>),
    /// Logical OR of two predicates
    Or(Box<ArrowPredicate>, Box<ArrowPredicate>),
    /// Logical NOT of a predicate
    Not(Box<ArrowPredicate>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    Eq,
    NotEq,
    Lt,
    Lte,
    Gt,
    Gte,
}

#[derive(Debug, Clone)]
pub enum LiteralValue {
    Null,
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}
