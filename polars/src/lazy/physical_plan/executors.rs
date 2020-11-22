use super::*;
use crate::lazy::logical_plan::DataFrameOperation;
use itertools::Itertools;
use rayon::prelude::*;
use std::sync::Mutex;

pub struct CsvExec {
    path: String,
    schema: Schema,
    has_header: bool,
    delimiter: Option<u8>,
}

impl CsvExec {
    pub fn new(path: String, schema: Schema, has_header: bool, delimiter: Option<u8>) -> Self {
        CsvExec {
            path,
            schema,
            has_header,
            delimiter,
        }
    }
}

impl Executor for CsvExec {
    fn execute(&self) -> Result<DataFrame> {
        let file = std::fs::File::open(&self.path).unwrap();

        let df = CsvReader::new(file)
            .has_header(self.has_header)
            .with_batch_size(10000)
            .finish()?;
        Ok(df)
    }
}

pub struct FilterExec {
    predicate: Arc<dyn PhysicalExpr>,
    input: Arc<dyn Executor>,
}

impl FilterExec {
    pub fn new(predicate: Arc<dyn PhysicalExpr>, input: Arc<dyn Executor>) -> Self {
        Self { predicate, input }
    }
}

impl Executor for FilterExec {
    fn execute(&self) -> Result<DataFrame> {
        let df = self.input.execute()?;
        let s = self.predicate.evaluate(&df)?;
        let mask = s.bool().expect("filter predicate wasn't of type boolean");

        Ok(df.filter(mask)?)
    }
}

pub struct DataFrameExec {
    df: Arc<Mutex<DataFrame>>,
}

impl DataFrameExec {
    pub(crate) fn new(df: Arc<Mutex<DataFrame>>) -> Self {
        DataFrameExec { df }
    }
}

impl Executor for DataFrameExec {
    fn execute(&self) -> Result<DataFrame> {
        let guard = self.df.lock().unwrap();
        let out = guard.clone();
        Ok(out)
    }
}

/// Take an input Executor (creates the input DataFrame)
/// and a multiple PhysicalExpressions (create the output Series)
pub struct StandardExec {
    /// i.e. sort, projection
    operation: &'static str,
    input: Arc<dyn Executor>,
    expr: Vec<Arc<dyn PhysicalExpr>>,
}

impl StandardExec {
    pub(crate) fn new(
        operation: &'static str,
        input: Arc<dyn Executor>,
        expr: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self {
            operation,
            input,
            expr,
        }
    }
}

impl Executor for StandardExec {
    fn execute(&self) -> Result<DataFrame> {
        let df = self.input.execute()?;
        let height = df.height();

        let mut selected_columns = self
            .expr
            .par_iter()
            .map(|expr| expr.evaluate(&df))
            .collect::<Result<Vec<Series>>>()?;

        // If all series are the same length it is ok. If not we can broadcast Series of length one.
        if selected_columns.len() > 1 {
            let all_equal_len = selected_columns.iter().map(|s| s.len()).all_equal();
            if !all_equal_len {
                selected_columns = selected_columns
                    .into_iter()
                    .map(|series| {
                        if series.len() == 1 && height > 1 {
                            series.expand_at_index(0, height)
                        } else {
                            series
                        }
                    })
                    .collect()
            }
        }

        Ok(DataFrame::new_no_checks(selected_columns))
    }
}

pub struct DataFrameOpsExec {
    input: Arc<dyn Executor>,
    operation: DataFrameOperation,
}

impl DataFrameOpsExec {
    pub(crate) fn new(input: Arc<dyn Executor>, operation: DataFrameOperation) -> Self {
        Self { input, operation }
    }
}

impl Executor for DataFrameOpsExec {
    fn execute(&self) -> Result<DataFrame> {
        let df = self.input.execute()?;
        match &self.operation {
            DataFrameOperation::Sort { by_column, reverse } => df.sort(&by_column, *reverse),
            DataFrameOperation::Explode(column) => df.explode(column),
            DataFrameOperation::DropDuplicates {
                maintain_order,
                subset,
            } => df.drop_duplicates(*maintain_order, subset.as_ref().map(|v| v.as_ref())),
        }
    }
}

/// Take an input Executor and a multiple expressions
pub struct GroupByExec {
    input: Arc<dyn Executor>,
    keys: Arc<Vec<String>>,
    aggs: Vec<Arc<dyn PhysicalExpr>>,
}

impl GroupByExec {
    pub(crate) fn new(
        input: Arc<dyn Executor>,
        keys: Arc<Vec<String>>,
        aggs: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self { input, keys, aggs }
    }
}

impl Executor for GroupByExec {
    fn execute(&self) -> Result<DataFrame> {
        let df = self.input.execute()?;
        let gb = df.groupby(&*self.keys)?;
        let groups = gb.get_groups();

        let mut columns = gb.keys();

        for expr in &self.aggs {
            let agg_expr = expr.as_agg_expr()?;
            let opt_agg = agg_expr.evaluate(&df, groups)?;
            if let Some(agg) = opt_agg {
                columns.push(agg)
            }
        }
        Ok(DataFrame::new_no_checks(columns))
    }
}

pub struct JoinExec {
    input_left: Arc<dyn Executor>,
    input_right: Arc<dyn Executor>,
    how: JoinType,
    left_on: Arc<dyn PhysicalExpr>,
    right_on: Arc<dyn PhysicalExpr>,
}

impl JoinExec {
    pub(crate) fn new(
        input_left: Arc<dyn Executor>,
        input_right: Arc<dyn Executor>,
        how: JoinType,
        left_on: Arc<dyn PhysicalExpr>,
        right_on: Arc<dyn PhysicalExpr>,
    ) -> Self {
        JoinExec {
            input_left,
            input_right,
            how,
            left_on,
            right_on,
        }
    }
}

impl Executor for JoinExec {
    fn execute(&self) -> Result<DataFrame> {
        // rayon::join was dropped because that resulted in a deadlock when there were dependencies
        // between the DataFrames. Like joining a specific computation of a DF back to itself.
        let df_left = self.input_left.execute();
        let df_right = self.input_right.execute();
        let df_left = df_left?;
        let df_right = df_right?;

        let s_left = self.left_on.evaluate(&df_left)?;
        let s_right = self.right_on.evaluate(&df_right)?;

        use JoinType::*;
        match self.how {
            Left => df_left.left_join_from_series(&df_right, &s_left, &s_right),
            Inner => df_left.inner_join_from_series(&df_right, &s_left, &s_right),
            Outer => df_left.outer_join_from_series(&df_right, &s_left, &s_right),
        }
    }
}
pub struct StackExec {
    input: Arc<dyn Executor>,
    expr: Vec<Arc<dyn PhysicalExpr>>,
}

impl StackExec {
    pub(crate) fn new(input: Arc<dyn Executor>, expr: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        Self { input, expr }
    }
}

impl Executor for StackExec {
    fn execute(&self) -> Result<DataFrame> {
        let mut df = self.input.execute()?;
        let height = df.height();

        let added_columns = self
            .expr
            .iter()
            .map(|expr| {
                expr.evaluate(&df).map(|series| {
                    // literal series. Should be whole column size
                    if series.len() == 1 && height > 1 {
                        series.expand_at_index(0, height)
                    } else {
                        series
                    }
                })
            })
            .collect::<Result<Vec<Series>>>()?;
        df.hstack_mut(&added_columns)?;
        Ok(df)
    }
}
