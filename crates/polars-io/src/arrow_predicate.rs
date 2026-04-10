use polars_core::datatypes::{TimeUnit, TimeZone};

/// IR describing a predicate that can be pushed down to the pyarrow / iceberg
/// reader in python.
///
/// This is produced on the Rust side in `polars-plan` and is later converted
/// to a real `pyarrow.compute.Expression` via pyo3 in `polars-mem-engine`,
/// so the Python side does not need to `eval()` a string.
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
    /// `expr.is_null()`
    IsNull(Box<ArrowPredicate>),
    /// `expr.isin(values)`
    IsIn {
        expr: Box<ArrowPredicate>,
        values: Vec<LiteralValue>,
    },
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
    Date(i32),
    Datetime {
        value: i64,
        time_unit: TimeUnit,
        time_zone: Option<TimeZone>,
    },
}
