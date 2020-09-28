pub(crate) mod optimizer;

use crate::{lazy::prelude::*, prelude::*};
use arrow::datatypes::SchemaRef;
use std::cell::RefCell;
use std::{fmt, rc::Rc};

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

#[derive(Clone)]
pub enum LogicalPlan {
    Filter {
        predicate: Expr,
        input: Box<LogicalPlan>,
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
    // https://stackoverflow.com/questions/1031076/what-are-projection-and-selection
    Projection {
        expr: Vec<Expr>,
        input: Box<LogicalPlan>,
    },
    Sort {
        input: Box<LogicalPlan>,
        expr: Vec<Expr>,
    },
}

impl fmt::Debug for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use LogicalPlan::*;
        match self {
            Filter { predicate, input } => write!(f, "Filter{:?} {:?}", predicate, input),
            CsvScan {
                path,
                schema: _,
                has_header: _,
                delimiter: _,
            } => write!(f, "CSVScan {}", path),
            DataFrameScan { df } => {
                let df = df.borrow();
                write!(f, "{:?}", df.head(Some(1)))
            }
            Projection { expr, input } => write!(f, "SELECT {:?} \nFROM\n{:?}", expr, input),
            Sort { input, expr } => write!(f, "Sort\n\t{:?}\n{:?}", expr, input),
        }
    }
}

pub struct LogicalPlanBuilder(LogicalPlan);

impl LogicalPlan {
    pub fn describe(&self) -> String {
        format!("{:#?}", self)
    }
}

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

    pub fn project(self, expr: Vec<Expr>) -> Self {
        LogicalPlan::Projection {
            expr,
            input: Box::new(self.0),
        }
        .into()
    }

    /// Apply a filter
    pub fn filter(self, predicate: Expr) -> Self {
        LogicalPlan::Filter {
            predicate,
            input: Box::new(self.0),
        }
        .into()
    }

    pub fn build(self) -> LogicalPlan {
        self.0
    }

    pub fn from_existing_df(df: DataFrame) -> Self {
        LogicalPlan::DataFrameScan {
            df: Rc::new(RefCell::new(df)),
        }
        .into()
    }

    pub fn sort(self, expr: Vec<Expr>) -> Self {
        LogicalPlan::Sort {
            input: Box::new(self.0),
            expr,
        }
        .into()
    }
}
