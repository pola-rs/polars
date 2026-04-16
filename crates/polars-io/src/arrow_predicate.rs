use polars_core::datatypes::{TimeUnit, TimeZone};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


// This is meant to mimic `pyarrow.compute.Expression` API as closely as possible to make it
// easier to convert directly to pyarrow predicates applied at the python/scan level.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ArrowPredicate {
    Column(String),
    Literal(LiteralValue),
    // Binary ops
    Comparison {
        left: Box<ArrowPredicate>,
        op: ComparisonOp,
        right: Box<ArrowPredicate>,
    },
    // Logicals
    And(Box<ArrowPredicate>, Box<ArrowPredicate>),
    Or(Box<ArrowPredicate>, Box<ArrowPredicate>),
    Xor(Box<ArrowPredicate>, Box<ArrowPredicate>),
    Not(Box<ArrowPredicate>),
    // Methods that turn into masks
    IsNull(Box<ArrowPredicate>),
    IsIn {
        expr: Box<ArrowPredicate>,
        values: Vec<LiteralValue>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ComparisonOp {
    Eq,
    NotEq,
    Lt,
    Lte,
    Gt,
    Gte,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LiteralValue {
    Null,
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Date(i32),
    // Should mimic python datetime module semantics
    Datetime {
        value: i64,
        time_unit: TimeUnit,
        time_zone: Option<TimeZone>,
    },
}
