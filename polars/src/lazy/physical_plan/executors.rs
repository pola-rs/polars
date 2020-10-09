use super::*;
use rayon;
use std::mem;
use std::sync::Mutex;

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
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
        let mut guard = self.df.lock().unwrap();
        let out = mem::take(&mut *guard);
        Ok(out)
    }
}

/// Take an input Executor and a multiple expressions
#[derive(Debug)]
pub struct PipeExec {
    /// i.e. sort, projection
    operation: &'static str,
    input: Arc<dyn Executor>,
    expr: Vec<Arc<dyn PhysicalExpr>>,
}

impl PipeExec {
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

impl Executor for PipeExec {
    fn execute(&self) -> Result<DataFrame> {
        let df = self.input.execute()?;
        let height = df.height();

        let selected_columns = self
            .expr
            .iter()
            .map(|expr| {
                expr.evaluate(&df).map(|series| {
                    // literal series. Should be whole column size
                    if series.len() == 1 && height > 1 {
                        series.expand_at_index(height, 0)
                    } else {
                        series
                    }
                })
            })
            .collect::<Result<Vec<Series>>>()?;
        Ok(DataFrame::new_no_checks(selected_columns))
    }
}

#[derive(Debug)]
pub struct SortExec {
    input: Arc<dyn Executor>,
    by_column: String,
    reverse: bool,
}

impl SortExec {
    pub(crate) fn new(input: Arc<dyn Executor>, by_column: String, reverse: bool) -> Self {
        Self {
            input,
            by_column,
            reverse,
        }
    }
}

impl Executor for SortExec {
    fn execute(&self) -> Result<DataFrame> {
        let df = self.input.execute()?;
        df.sort(&self.by_column, self.reverse)
    }
}

/// Take an input Executor and a multiple expressions
#[derive(Debug)]
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

#[derive(Debug)]
pub struct JoinExec {
    input_left: Arc<dyn Executor>,
    input_right: Arc<dyn Executor>,
    how: JoinType,
    left_on: Arc<String>,
    right_on: Arc<String>,
}

impl JoinExec {
    pub(crate) fn new(
        input_left: Arc<dyn Executor>,
        input_right: Arc<dyn Executor>,
        how: JoinType,
        left_on: Arc<String>,
        right_on: Arc<String>,
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
        let (df_left, df_right) =
            rayon::join(|| self.input_left.execute(), || self.input_right.execute());

        use JoinType::*;
        match self.how {
            Left => df_left?.left_join(&df_right?, &self.left_on, &self.right_on),
            Inner => df_left?.inner_join(&df_right?, &self.left_on, &self.right_on),
            Outer => df_left?.outer_join(&df_right?, &self.left_on, &self.right_on),
        }
    }
}
#[derive(Debug)]
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
                        series.expand_at_index(height, 0)
                    } else {
                        series
                    }
                })
            })
            .collect::<Result<Vec<Series>>>()?;
        df.hstack(&added_columns)?;
        Ok(df)
    }
}
