pub(crate) mod optimizer;
use crate::{
    lazy::{prelude::*, utils},
    prelude::*,
};
use arrow::datatypes::DataType;
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

impl ScalarValue {
    /// Getter for the `DataType` of the value
    pub fn get_datatype(&self) -> DataType {
        match *self {
            ScalarValue::Boolean(_) => DataType::Boolean,
            ScalarValue::UInt8(_) => DataType::UInt8,
            ScalarValue::UInt16(_) => DataType::UInt16,
            ScalarValue::UInt32(_) => DataType::UInt32,
            ScalarValue::UInt64(_) => DataType::UInt64,
            ScalarValue::Int8(_) => DataType::Int8,
            ScalarValue::Int16(_) => DataType::Int16,
            ScalarValue::Int32(_) => DataType::Int32,
            ScalarValue::Int64(_) => DataType::Int64,
            ScalarValue::Float32(_) => DataType::Float32,
            ScalarValue::Float64(_) => DataType::Float64,
            ScalarValue::Utf8(_) => DataType::Utf8,
            _ => panic!("Cannot treat {:?} as scalar value", self),
        }
    }
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

// https://stackoverflow.com/questions/1031076/what-are-projection-and-selection
#[derive(Clone)]
pub enum LogicalPlan {
    // filter on a boolean mask
    Selection {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },
    CsvScan {
        path: String,
        schema: Schema,
        has_header: bool,
        delimiter: Option<u8>,
    },
    DataFrameScan {
        df: Rc<RefCell<DataFrame>>,
        schema: Schema,
    },
    // horizontal selection
    Projection {
        expr: Vec<Expr>,
        input: Box<LogicalPlan>,
        schema: Schema,
    },
    Sort {
        input: Box<LogicalPlan>,
        expr: Vec<Expr>,
    },
    Aggregate {
        input: Box<LogicalPlan>,
        keys: Vec<Expr>,
        aggs: Vec<Expr>,
        schema: Schema,
    },
}

impl fmt::Debug for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use LogicalPlan::*;
        match self {
            Selection { predicate, input } => write!(f, "Filter\n\t{:?} {:?}", predicate, input),
            CsvScan { path, .. } => write!(f, "CSVScan {}", path),
            DataFrameScan { schema: schema, .. } => write!(f, "{:?}", schema),
            Projection { expr, input, .. } => write!(f, "SELECT {:?} \nFROM\n{:?}", expr, input),
            Sort { input, expr } => write!(f, "Sort\n\t{:?}\n{:?}", expr, input),
            Aggregate { keys, aggs, .. } => write!(f, "Aggregate\n\t{:?} BY {:?}", aggs, keys),
        }
    }
}

pub struct LogicalPlanBuilder(LogicalPlan);

impl LogicalPlan {
    pub(crate) fn schema(&self) -> &Schema {
        use LogicalPlan::*;
        match self {
            DataFrameScan { schema, .. } => schema,
            Selection { input, .. } => input.schema(),
            CsvScan { schema, .. } => schema,
            Projection { schema, .. } => schema,
            Sort { input, .. } => input.schema(),
            Aggregate { schema, .. } => schema,
        }
    }
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
    pub fn scan_csv() -> Self {
        todo!()
    }

    pub fn project(self, expr: Vec<Expr>) -> Self {
        let schema = utils::expressions_to_schema(&expr, self.0.schema());
        LogicalPlan::Projection {
            expr,
            input: Box::new(self.0),
            schema,
        }
        .into()
    }

    /// Apply a filter
    pub fn filter(self, predicate: Expr) -> Self {
        LogicalPlan::Selection {
            predicate,
            input: Box::new(self.0),
        }
        .into()
    }

    pub fn groupby(self, keys: Vec<Expr>, aggs: Vec<Expr>) -> Self {
        let schema1 = utils::expressions_to_schema(&keys, self.0.schema());
        let schema2 = utils::expressions_to_schema(&aggs, self.0.schema());
        let schema = Schema::try_merge(&[schema1, schema2]).unwrap();

        LogicalPlan::Aggregate {
            input: Box::new(self.0),
            keys,
            aggs,
            schema,
        }
        .into()
    }

    pub fn build(self) -> LogicalPlan {
        self.0
    }

    pub fn from_existing_df(df: DataFrame) -> Self {
        let schema = df.schema();
        LogicalPlan::DataFrameScan {
            df: Rc::new(RefCell::new(df)),
            schema,
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
