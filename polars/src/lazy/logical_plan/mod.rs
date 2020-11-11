pub(crate) mod optimizer;
use crate::lazy::logical_plan::LogicalPlan::CsvScan;
use crate::lazy::utils::expr_to_root_column_expr;
use crate::{
    lazy::{prelude::*, utils},
    prelude::*,
};
use ahash::RandomState;
use arrow::datatypes::DataType;
use std::collections::HashSet;
use std::sync::Mutex;
use std::{fmt, sync::Arc};

#[derive(Clone, Debug, PartialEq)]
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

#[derive(Clone)]
pub enum DataFrameOperation {
    Sort { by_column: String, reverse: bool },
    Reverse,
    Shift { periods: i32 },
    Max,
    Min,
    Sum,
    Mean,
    Median,
    Quantile(f64),
    Explode(String),
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
        df: Arc<Mutex<DataFrame>>,
        schema: Schema,
    },
    // vertical selection
    Projection {
        expr: Vec<Expr>,
        input: Box<LogicalPlan>,
        schema: Schema,
    },
    DataFrameOp {
        input: Box<LogicalPlan>,
        operation: DataFrameOperation,
    },
    Aggregate {
        input: Box<LogicalPlan>,
        keys: Arc<Vec<String>>,
        aggs: Vec<Expr>,
        schema: Schema,
    },
    Join {
        input_left: Box<LogicalPlan>,
        input_right: Box<LogicalPlan>,
        schema: Schema,
        how: JoinType,
        left_on: Expr,
        right_on: Expr,
    },
    HStack {
        input: Box<LogicalPlan>,
        exprs: Vec<Expr>,
        schema: Schema,
    },
}

impl Default for LogicalPlan {
    fn default() -> Self {
        CsvScan {
            path: "".to_string(),
            schema: Schema::new(vec![Field::new("", ArrowDataType::Null, true)]),
            has_header: false,
            delimiter: None,
        }
    }
}

impl fmt::Debug for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use LogicalPlan::*;
        match self {
            Selection { predicate, input } => {
                write!(f, "Filter\n\t{:?}\nFROM\n\t{:?}", predicate, input)
            }
            CsvScan { path, .. } => write!(f, "CSVScan {}", path),
            DataFrameScan { schema, .. } => write!(
                f,
                "TABLE: {:?}",
                schema
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .take(4)
                    .collect::<Vec<_>>()
            ),
            Projection { expr, input, .. } => write!(f, "SELECT {:?} \nFROM\n{:?}", expr, input),
            DataFrameOp {
                input, operation, ..
            } => match operation {
                DataFrameOperation::Sort { .. } => write!(f, "SORT {:?}", input),
                DataFrameOperation::Reverse => write!(f, "REVERSE {:?}", input),
                DataFrameOperation::Shift { periods } => {
                    write!(f, "SHIFT {:?} BY {}", input, periods)
                }
                DataFrameOperation::Max => write!(f, "MAX {:?}", input),
                DataFrameOperation::Min => write!(f, "MIN {:?}", input),
                DataFrameOperation::Sum => write!(f, "SUM {:?}", input),
                DataFrameOperation::Mean => write!(f, "MEAN {:?}", input),
                DataFrameOperation::Median => write!(f, "MEDIAN {:?}", input),
                DataFrameOperation::Quantile(_) => write!(f, "QUANTILE {:?}", input),
                DataFrameOperation::Explode(_) => write!(f, "EXPLODE {:?}", input),
            },
            Aggregate { keys, aggs, .. } => write!(f, "Aggregate\n\t{:?} BY {:?}", aggs, keys),
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                ..
            } => write!(
                f,
                "JOIN\n\t({:?})\nWITH\n\t({:?})\nON (left: {:?} right: {:?})",
                input_left, input_right, left_on, right_on
            ),
            HStack { input, exprs, .. } => {
                write!(f, "\n{:?}\n\tWITH COLUMN(S)\n{:?}\n", input, exprs)
            }
        }
    }
}

fn replace_wildcard_with_column(expr: Expr, column_name: Arc<String>) -> Expr {
    match expr {
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => Expr::Ternary {
            predicate: Box::new(replace_wildcard_with_column(*predicate, column_name)),
            truthy,
            falsy,
        },
        Expr::Apply {
            input,
            function,
            output_type,
        } => Expr::Apply {
            input: Box::new(replace_wildcard_with_column(*input, column_name)),
            function,
            output_type,
        },
        Expr::BinaryExpr { left, op, right } => Expr::BinaryExpr {
            left: Box::new(replace_wildcard_with_column(*left, column_name.clone())),
            op,
            right: Box::new(replace_wildcard_with_column(*right, column_name)),
        },
        Expr::Wildcard => Expr::Column(column_name),
        Expr::IsNotNull(e) => {
            Expr::IsNotNull(Box::new(replace_wildcard_with_column(*e, column_name)))
        }
        Expr::IsNull(e) => Expr::IsNull(Box::new(replace_wildcard_with_column(*e, column_name))),
        Expr::Not(e) => Expr::Not(Box::new(replace_wildcard_with_column(*e, column_name))),
        Expr::Alias(e, name) => Expr::Alias(
            Box::new(replace_wildcard_with_column(*e, column_name)),
            name,
        ),
        Expr::AggMean(e) => Expr::AggMean(Box::new(replace_wildcard_with_column(*e, column_name))),
        Expr::AggMedian(e) => {
            Expr::AggMedian(Box::new(replace_wildcard_with_column(*e, column_name)))
        }
        Expr::AggMax(e) => Expr::AggMax(Box::new(replace_wildcard_with_column(*e, column_name))),
        Expr::AggMin(e) => Expr::AggMin(Box::new(replace_wildcard_with_column(*e, column_name))),
        Expr::AggSum(e) => Expr::AggSum(Box::new(replace_wildcard_with_column(*e, column_name))),
        Expr::AggLast(e) => Expr::AggLast(Box::new(replace_wildcard_with_column(*e, column_name))),
        Expr::AggFirst(e) => {
            Expr::AggFirst(Box::new(replace_wildcard_with_column(*e, column_name)))
        }
        Expr::AggNUnique(e) => {
            Expr::AggNUnique(Box::new(replace_wildcard_with_column(*e, column_name)))
        }
        Expr::AggGroups(e) => {
            Expr::AggGroups(Box::new(replace_wildcard_with_column(*e, column_name)))
        }
        Expr::AggQuantile { expr, quantile } => Expr::AggQuantile {
            expr: Box::new(replace_wildcard_with_column(*expr, column_name)),
            quantile,
        },
        Expr::AggList(e) => Expr::AggList(Box::new(replace_wildcard_with_column(*e, column_name))),
        Expr::Shift { input, periods } => Expr::Shift {
            input: Box::new(replace_wildcard_with_column(*input, column_name)),
            periods,
        },
        Expr::Sort { expr, reverse } => Expr::Sort {
            expr: Box::new(replace_wildcard_with_column(*expr, column_name)),
            reverse,
        },
        Expr::Cast { expr, data_type } => Expr::Cast {
            expr: Box::new(replace_wildcard_with_column(*expr, column_name)),
            data_type,
        },
        Expr::Column(_) => expr,
        Expr::Literal(_) => expr,
    }
}

/// In case of single col(*) -> do nothing, no selection is the same as select all
/// In other cases replace the wildcard with an expression with all columns
fn remove_wildcard_from_exprs(exprs: Vec<Expr>, schema: &Schema) -> Vec<Expr> {
    if exprs.len() == 1 && exprs[0] == Expr::Wildcard {
        // no projection needed
        return vec![];
    };
    let mut result = Vec::with_capacity(exprs.len() + schema.fields().len());
    for expr in exprs {
        if expr_to_root_column_expr(&expr).expect("should have a root column") == &Expr::Wildcard {
            for field in schema.fields() {
                let name = field.name();
                let new_expr = replace_wildcard_with_column(expr.clone(), Arc::new(name.clone()));
                result.push(new_expr)
            }
        } else {
            result.push(expr)
        };
    }
    result
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
            DataFrameOp { input, .. } => input.schema(),
            Aggregate { schema, .. } => schema,
            Join { schema, .. } => schema,
            HStack { schema, .. } => schema,
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

    pub fn project(self, exprs: Vec<Expr>) -> Self {
        let exprs = remove_wildcard_from_exprs(exprs, &self.0.schema());
        let schema = utils::expressions_to_schema(&exprs, self.0.schema());

        // if len == 0, no projection has to be done. This is a select all operation.
        if exprs.len() > 0 {
            LogicalPlan::Projection {
                expr: exprs,
                input: Box::new(self.0),
                schema,
            }
            .into()
        } else {
            self
        }
    }

    pub fn with_columns(self, exprs: Vec<Expr>) -> Self {
        // current schema
        let schema = self.0.schema();
        let added_schema = utils::expressions_to_schema(&exprs, schema);
        let new_schema = Schema::try_merge(&[schema.clone(), added_schema]).unwrap();

        LogicalPlan::HStack {
            input: Box::new(self.0),
            exprs,
            schema: new_schema,
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

    pub fn groupby(self, keys: Arc<Vec<String>>, aggs: Vec<Expr>) -> Self {
        let current_schema = self.0.schema();
        let aggs = remove_wildcard_from_exprs(aggs, current_schema);

        let fields = keys
            .iter()
            .map(|name| current_schema.field_with_name(name).unwrap().clone())
            .collect::<Vec<_>>();

        let schema1 = Schema::new(fields);

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
            df: Arc::new(Mutex::new(df)),
            schema,
        }
        .into()
    }

    pub fn sort(self, by_column: String, reverse: bool) -> Self {
        LogicalPlan::DataFrameOp {
            input: Box::new(self.0),
            operation: DataFrameOperation::Sort { by_column, reverse },
        }
        .into()
    }

    pub fn reverse(self) -> Self {
        LogicalPlan::DataFrameOp {
            input: Box::new(self.0),
            operation: DataFrameOperation::Reverse,
        }
        .into()
    }

    pub fn shift(self, periods: i32) -> Self {
        LogicalPlan::DataFrameOp {
            input: Box::new(self.0),
            operation: DataFrameOperation::Shift { periods },
        }
        .into()
    }

    pub fn explode(self, column: &str) -> Self {
        LogicalPlan::DataFrameOp {
            input: Box::new(self.0),
            operation: DataFrameOperation::Explode(column.to_owned()),
        }
        .into()
    }

    pub fn min(self) -> Self {
        LogicalPlan::DataFrameOp {
            input: Box::new(self.0),
            operation: DataFrameOperation::Min,
        }
        .into()
    }

    pub fn max(self) -> Self {
        LogicalPlan::DataFrameOp {
            input: Box::new(self.0),
            operation: DataFrameOperation::Max,
        }
        .into()
    }

    pub fn sum(self) -> Self {
        LogicalPlan::DataFrameOp {
            input: Box::new(self.0),
            operation: DataFrameOperation::Sum,
        }
        .into()
    }

    pub fn mean(self) -> Self {
        LogicalPlan::DataFrameOp {
            input: Box::new(self.0),
            operation: DataFrameOperation::Mean,
        }
        .into()
    }

    pub fn median(self) -> Self {
        LogicalPlan::DataFrameOp {
            input: Box::new(self.0),
            operation: DataFrameOperation::Median,
        }
        .into()
    }

    pub fn quantile(self, quantile: f64) -> Self {
        LogicalPlan::DataFrameOp {
            input: Box::new(self.0),
            operation: DataFrameOperation::Quantile(quantile),
        }
        .into()
    }

    pub fn join(self, other: LogicalPlan, how: JoinType, left_on: Expr, right_on: Expr) -> Self {
        let schema_left = self.0.schema();
        let schema_right = other.schema();

        // column names of left table
        let mut names: HashSet<&String, RandomState> = HashSet::default();
        // fields of new schema
        let mut fields = vec![];

        for f in schema_left.fields() {
            names.insert(f.name());
            fields.push(f.clone());
        }

        let right_name = utils::output_name(&right_on).expect("could not find name");

        for f in schema_right.fields() {
            let name = f.name();

            if name != &*right_name {
                if names.contains(name) {
                    let new_name = format!("{}_right", name);
                    let field = Field::new(&new_name, f.data_type().clone(), f.is_nullable());
                    fields.push(field)
                } else {
                    fields.push(f.clone())
                }
            }
        }
        let schema = Schema::new(fields);

        LogicalPlan::Join {
            input_left: Box::new(self.0),
            input_right: Box::new(other),
            how,
            schema,
            left_on,
            right_on,
        }
        .into()
    }
}

#[derive(Clone, Debug)]
pub enum JoinType {
    Left,
    Inner,
    Outer,
}

#[cfg(test)]
mod test {
    use crate::lazy::prelude::*;
    use crate::lazy::tests::get_df;
    use crate::prelude::*;

    fn print_plans(lf: &LazyFrame) {
        println!("LOGICAL PLAN\n\n{}\n", lf.describe_plan());
        println!(
            "OPTIMIZED LOGICAL PLAN\n\n{}\n",
            lf.describe_optimized_plan().unwrap()
        );
    }

    #[test]
    fn test_lazy_arithmetic() {
        let df = get_df();
        let lf = df
            .lazy()
            .select(&[((col("sepal.width") * lit(100)).alias("super_wide"))])
            .sort("super_wide", false);

        print_plans(&lf);

        let new = lf.collect().unwrap();
        println!("{:?}", new);
        assert_eq!(new.height(), 7);
        assert_eq!(
            new.column("super_wide").unwrap().f64().unwrap().get(0),
            Some(300.0)
        );
    }

    #[test]
    fn test_lazy_logical_plan_filter_and_alias_combined() {
        let df = get_df();
        let lf = df
            .clone()
            .lazy()
            .filter(col("sepal.width").lt(lit(3.5)))
            .select(&[col("variety").alias("foo")]);

        print_plans(&lf);
        let df = lf.collect().unwrap();
        println!("{:?}", df);
    }

    #[test]
    fn test_lazy_logical_plan_schema() {
        let df = get_df();
        let lp = df
            .clone()
            .lazy()
            .select(&[col("variety").alias("foo")])
            .logical_plan;

        println!("{:#?}", lp.schema().fields());
        assert!(lp.schema().field_with_name("foo").is_ok());

        let lp = df
            .lazy()
            .groupby("variety")
            .agg(vec![col("sepal.width").agg_min()])
            .logical_plan;
        println!("{:#?}", lp.schema().fields());
        assert!(lp.schema().field_with_name("sepal.width_min").is_ok());
    }

    #[test]
    fn test_lazy_logical_plan_join() {
        let left = df!("days" => &[0, 1, 2, 3, 4],
        "temp" => [22.1, 19.9, 7., 2., 3.],
        "rain" => &[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        .unwrap();

        let right = df!(
        "days" => &[1, 2],
        "rain" => &[0.1, 0.2]
        )
        .unwrap();

        // check if optimizations succeeds without selection
        {
            let lf = left
                .clone()
                .lazy()
                .left_join(right.clone().lazy(), col("days"), col("days"));

            print_plans(&lf);
            // implicitly checks logical plan == optimized logical plan
            let df = lf.collect().unwrap();
            println!("{:?}", df);
        }

        // check if optimization succeeds with selection
        {
            let lf = left
                .clone()
                .lazy()
                .left_join(right.clone().lazy(), col("days"), col("days"))
                .select(&[col("temp")]);

            print_plans(&lf);
            let df = lf.collect().unwrap();
            println!("{:?}", df);
        }

        // check if optimization succeeds with selection of a renamed column due to the join
        {
            let lf = left
                .clone()
                .lazy()
                .left_join(right.clone().lazy(), col("days"), col("days"))
                .select(&[col("temp"), col("rain_right")]);

            print_plans(&lf);
            let df = lf.collect().unwrap();
            println!("{:?}", df);
        }

        // check if optimization succeeds with selection of the left and the right (renamed)
        // column due to the join
        {
            let lf = left
                .clone()
                .lazy()
                .left_join(right.clone().lazy(), col("days"), col("days"))
                .select(&[col("temp"), col("rain"), col("rain_right")]);

            print_plans(&lf);
            let df = lf.collect().unwrap();
            println!("{:?}", df);
        }

        // check if optimization succeeds with selection of the left and the right (renamed)
        // column due to the join and an extra alias
        {
            let lf = left
                .clone()
                .lazy()
                .left_join(right.clone().lazy(), col("days"), col("days"))
                .select(&[col("temp"), col("rain").alias("foo"), col("rain_right")]);

            print_plans(&lf);
            let df = lf.collect().unwrap();
            println!("{:?}", df);
        }

        // check if optimization succeeds with selection of the left and the right (renamed)
        // column due to the join and an extra alias
        {
            let lf = left
                .clone()
                .lazy()
                .left_join(right.clone().lazy(), col("days"), col("days"))
                .select(&[col("temp"), col("rain").alias("foo"), col("rain_right")])
                .filter(col("foo").lt(lit(0.3)));

            print_plans(&lf);
            let df = lf.collect().unwrap();
            println!("{:?}", df);
        }
    }
}
