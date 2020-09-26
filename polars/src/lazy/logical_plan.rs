use crate::prelude::*;
use arrow::datatypes::SchemaRef;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum ScalarValue {
    Null,
    /// A binary true or false.
    Boolean(bool),
    /// A UTF8 encoded string type.
    Utf8(String),
    /// An unsigned 8-bit integer number.
    UInt8(u8),
    /// An unsigned 16-bit integer number.
    UInt16(u16),
    /// An unsigned 32-bit integer number.
    UInt32(u32),
    /// An unsigned 64-bit integer number.
    UInt64(u64),
    /// An 8-bit integer number.
    Int8(i8),
    /// A 16-bit integer number.
    Int16(i16),
    /// A 32-bit integer number.
    Int32(i32),
    /// A 64-bit integer number.
    Int64(i64),
    /// A 32-bit floating point number.
    Float32(f32),
    /// A 64-bit floating point number.
    Float64(f64),
    /// A 32-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in days (32 bits).
    Date32(i32),
    /// A 64-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in milliseconds (64 bits).
    Date64(i64),
    /// A 64-bit time representing the elapsed time since midnight in the unit of `TimeUnit`.
    Time64(i64, TimeUnit),
    /// A 32-bit time representing the elapsed time since midnight in the unit of `TimeUnit`.
    Time32(i32, TimeUnit),
    /// Measure of elapsed time in either seconds, milliseconds, microseconds or nanoseconds.
    Duration(i64, TimeUnit),
    /// Naive Time elapsed from the Unix epoch, 00:00:00.000 on 1 January 1970, excluding leap seconds, as a 64-bit integer.
    /// Note that UNIX time does not include leap seconds.
    TimeStamp(i64, TimeUnit),
    /// A "calendar" interval which models types that don't necessarily have a precise duration without the context of a base timestamp
    /// (e.g. days can differ in length during day light savings time transitions).
    IntervalDayTime(i64),
    IntervalYearMonth(i32),
    LargeList(Series),
}

#[derive(Debug, Copy, Clone)]
pub enum Operator {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulus,
    And,
    Or,
    Not,
    Like,
    NotLike,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Alias(Box<Expr>, String),
    Column(String),
    Literal(ScalarValue),
    BinaryExpr {
        left: Box<Expr>,
        op: Operator,
        right: Box<Expr>,
    },
    Nested(Box<Expr>),
    Not(Box<Expr>),
    IsNotNull(Box<Expr>),
    IsNull(Box<Expr>),
    Cast {
        expr: Box<Expr>,
        data_type: ArrowDataType,
    },
    Sort {
        expr: Box<Expr>,
        reverse: bool,
    },
    ScalarFunction {
        name: String,
        args: Vec<Expr>,
        return_type: ArrowDataType,
    },
    AggregateFunction {
        name: String,
        args: Vec<Expr>,
    },
    Wildcard,
}

impl Expr {
    pub fn eq(&self, other: Expr) -> Expr {
        binary_expr(self.clone(), Operator::Eq, other)
    }

    pub fn lt(&self, other: Expr) -> Expr {
        binary_expr(self.clone(), Operator::Lt, other)
    }
}

fn binary_expr(l: Expr, op: Operator, r: Expr) -> Expr {
    Expr::BinaryExpr {
        left: Box::new(l),
        op,
        right: Box::new(r),
    }
}

#[derive(Clone, Debug)]
pub enum LogicalPlan {
    Filter {
        predicate: Expr,
        input: Rc<LogicalPlan>,
    },
    CsvScan {
        path: String,
        schema: Option<SchemaRef>,
        has_header: bool,
        delimiter: Option<u8>,
    },
    DataFrameScan {
        df: Rc<RefCell<DataFrame>>,
    },
}

pub struct LogicalPlanBuilder(LogicalPlan);

impl From<LogicalPlan> for LogicalPlanBuilder {
    fn from(lp: LogicalPlan) -> Self {
        LogicalPlanBuilder(lp)
    }
}

impl LogicalPlanBuilder {
    pub fn scan_csv(
        path: String,
        schema: Option<SchemaRef>,
        has_header: bool,
        delimiter: Option<u8>,
    ) -> Self {
        LogicalPlan::CsvScan {
            path,
            schema,
            has_header,
            delimiter,
        }
        .into()
    }

    /// Apply a filter
    pub fn filter(&self, predicate: Expr) -> Self {
        LogicalPlan::Filter {
            predicate,
            input: Rc::new(self.0.clone()),
        }
        .into()
    }

    pub fn build(self) -> LogicalPlan {
        self.0
    }

    pub fn dataframe(df: DataFrame) -> Self {
        LogicalPlan::DataFrameScan {
            df: Rc::new(RefCell::new(df)),
        }
        .into()
    }
}

/// Create a column expression based on a column name.
pub fn col(name: &str) -> Expr {
    Expr::Column(name.to_owned())
}

pub trait Literal {
    fn lit(self) -> Expr;
}

impl Literal for String {
    fn lit(self) -> Expr {
        Expr::Literal(ScalarValue::Utf8(self))
    }
}

impl<'a> Literal for &'a str {
    fn lit(self) -> Expr {
        Expr::Literal(ScalarValue::Utf8(self.to_owned()))
    }
}

macro_rules! make_literal {
    ($TYPE:ty, $SCALAR:ident) => {
        impl Literal for $TYPE {
            fn lit(self) -> Expr {
                Expr::Literal(ScalarValue::$SCALAR(self))
            }
        }
    };
}

make_literal!(bool, Boolean);
make_literal!(f32, Float32);
make_literal!(f64, Float64);
make_literal!(i8, Int8);
make_literal!(i16, Int16);
make_literal!(i32, Int32);
make_literal!(i64, Int64);
make_literal!(u8, UInt8);
make_literal!(u16, UInt16);
make_literal!(u32, UInt32);
make_literal!(u64, UInt64);

pub fn lit<L: Literal>(t: L) -> Expr {
    t.lit()
}
