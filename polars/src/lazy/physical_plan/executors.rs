use super::*;
use crate::frame::ser::csv::CsvEncoding;
use crate::lazy::logical_plan::{DataFrameOperation, FETCH_ROWS};
use itertools::Itertools;
use rayon::prelude::*;
use std::mem;

const POLARS_VERBOSE: &str = "POLARS_VERBOSE";

fn set_n_rows(stop_after_n_rows: usize) -> usize {
    let fetch_rows = FETCH_ROWS.with(|fetch_rows| fetch_rows.get());
    match fetch_rows {
        None => stop_after_n_rows,
        Some(n) => n,
    }
}

pub struct CacheExec {
    pub key: String,
    pub input: Box<dyn Executor>,
}

impl Executor for CacheExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let guard = cache.lock().unwrap();

        // cache hit
        if let Some(df) = guard.get(&self.key) {
            return Ok(df.clone());
        }
        drop(guard);

        // cache miss
        let df = self.input.execute(cache)?;

        let mut guard = cache.lock().unwrap();
        let key = std::mem::take(&mut self.key);
        guard.insert(key, df.clone());

        if std::env::var(POLARS_VERBOSE).is_ok() {
            println!("cache set {:?}", self.key);
        }
        Ok(df)
    }
}

#[cfg(feature = "parquet")]
pub struct ParquetExec {
    path: String,
    schema: Schema,
    with_columns: Option<Vec<String>>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    stop_after_n_rows: Option<usize>,
    cache: bool,
}

#[cfg(feature = "parquet")]
impl ParquetExec {
    pub fn new(
        path: String,
        schema: Schema,
        with_columns: Option<Vec<String>>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        stop_after_n_rows: Option<usize>,
        cache: bool,
    ) -> Self {
        ParquetExec {
            path,
            schema,
            with_columns,
            predicate,
            stop_after_n_rows,
            cache,
        }
    }
}

#[cfg(feature = "parquet")]
impl Executor for ParquetExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let cache_key = match &self.predicate {
            Some(predicate) => format!("{}{:?}", self.path, predicate.as_expression()),
            None => self.path.to_string(),
        };
        if self.cache {
            let guard = cache.lock().unwrap();
            // cache hit
            if let Some(df) = guard.get(&cache_key) {
                return Ok(df.clone());
            }
            drop(guard);
        }

        // cache miss
        let file = std::fs::File::open(&self.path).unwrap();

        let with_columns = mem::take(&mut self.with_columns);
        let mut schema = Schema::new(vec![]);
        mem::swap(&mut self.schema, &mut schema);

        let projection: Option<Vec<_>> = if let Some(with_columns) = with_columns {
            Some(
                with_columns
                    .iter()
                    .map(|name| schema.column_with_name(name).unwrap().0)
                    .collect(),
            )
        } else {
            None
        };

        let stop_after_n_rows = self.stop_after_n_rows.map(set_n_rows);

        let df = ParquetReader::new(file)
            .with_stop_after_n_rows(stop_after_n_rows)
            .finish_with_predicate(
                self.predicate.clone(),
                projection.as_ref().map(|v| v.as_ref()),
            )?;

        if self.cache {
            let mut guard = cache.lock().unwrap();
            guard.insert(cache_key, df.clone());
        }
        if std::env::var(POLARS_VERBOSE).is_ok() {
            println!("parquet {:?} read", self.path);
        }

        Ok(df)
    }
}

pub struct CsvExec {
    path: String,
    schema: Schema,
    has_header: bool,
    delimiter: u8,
    ignore_errors: bool,
    skip_rows: usize,
    stop_after_n_rows: Option<usize>,
    with_columns: Option<Vec<String>>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    cache: bool,
}

impl CsvExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: String,
        schema: Schema,
        has_header: bool,
        delimiter: u8,
        ignore_errors: bool,
        skip_rows: usize,
        stop_after_n_rows: Option<usize>,
        with_columns: Option<Vec<String>>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        cache: bool,
    ) -> Self {
        CsvExec {
            path,
            schema,
            has_header,
            delimiter,
            ignore_errors,
            skip_rows,
            stop_after_n_rows,
            with_columns,
            predicate,
            cache,
        }
    }
}

impl Executor for CsvExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let cache_key = match &self.predicate {
            Some(predicate) => format!("{}{:?}", self.path, predicate.as_expression()),
            None => self.path.to_string(),
        };
        if self.cache {
            let guard = cache.lock().unwrap();
            // cache hit
            if let Some(df) = guard.get(&cache_key) {
                return Ok(df.clone());
            }
            drop(guard);
        }

        // cache miss
        let file = std::fs::File::open(&self.path).unwrap();

        let mut with_columns = mem::take(&mut self.with_columns);
        let mut projected_len = 0;
        with_columns.as_ref().map(|columns| {
            projected_len = columns.len();
            columns
        });

        if projected_len == 0 {
            with_columns = None;
        }
        let mut schema = Schema::new(vec![]);
        mem::swap(&mut self.schema, &mut schema);
        let stop_after_n_rows = self.stop_after_n_rows.map(set_n_rows);

        let reader = CsvReader::new(file)
            .has_header(self.has_header)
            .with_schema(Arc::new(schema))
            .with_delimiter(self.delimiter)
            .with_ignore_parser_errors(self.ignore_errors)
            .with_skip_rows(self.skip_rows)
            .with_stop_after_n_rows(stop_after_n_rows)
            .with_columns(with_columns)
            .with_encoding(CsvEncoding::LossyUtf8);

        let df = match &self.predicate {
            Some(predicate) => reader.finish_with_predicate(predicate.clone()),
            None => reader.finish(),
        }?;

        if self.cache {
            let mut guard = cache.lock().unwrap();
            guard.insert(cache_key, df.clone());
        }
        if std::env::var(POLARS_VERBOSE).is_ok() {
            println!("csv {:?} read", self.path);
        }

        Ok(df)
    }
}

pub struct FilterExec {
    predicate: Arc<dyn PhysicalExpr>,
    input: Box<dyn Executor>,
}

impl FilterExec {
    pub fn new(predicate: Arc<dyn PhysicalExpr>, input: Box<dyn Executor>) -> Self {
        Self { predicate, input }
    }
}

impl Executor for FilterExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let df = self.input.execute(cache)?;
        let s = self.predicate.evaluate(&df)?;
        let mask = s.bool().expect("filter predicate wasn't of type boolean");
        let df = df.filter(mask)?;
        if std::env::var(POLARS_VERBOSE).is_ok() {
            println!("dataframe filtered");
        }
        Ok(df)
    }
}

pub struct DataFrameExec {
    df: Arc<DataFrame>,
    projection: Option<Vec<Arc<dyn PhysicalExpr>>>,
    selection: Option<Arc<dyn PhysicalExpr>>,
}

impl DataFrameExec {
    pub(crate) fn new(
        df: Arc<DataFrame>,
        projection: Option<Vec<Arc<dyn PhysicalExpr>>>,
        selection: Option<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        DataFrameExec {
            df,
            projection,
            selection,
        }
    }
}

impl Executor for DataFrameExec {
    fn execute(&mut self, _: &Cache) -> Result<DataFrame> {
        let df = mem::take(&mut self.df);
        let mut df = Arc::try_unwrap(df).unwrap_or_else(|df| (*df).clone());

        // projection should be before selection as those are free
        if let Some(projection) = &self.projection {
            df = evaluate_physical_expressions(&df, projection)?;
        }

        if let Some(selection) = &self.selection {
            let s = selection.evaluate(&df)?;
            let mask = s.bool().map_err(|_| {
                PolarsError::Other("filter predicate was not of type boolean".into())
            })?;
            df = df.filter(mask)?;
        }
        Ok(df)
    }
}

/// Take an input Executor (creates the input DataFrame)
/// and a multiple PhysicalExpressions (create the output Series)
pub struct StandardExec {
    /// i.e. sort, projection
    operation: &'static str,
    input: Box<dyn Executor>,
    expr: Vec<Arc<dyn PhysicalExpr>>,
    // make sure that we are not called twice
    valid: bool,
}

impl StandardExec {
    pub(crate) fn new(
        operation: &'static str,
        input: Box<dyn Executor>,
        expr: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self {
            operation,
            input,
            expr,
            valid: true,
        }
    }
}

fn evaluate_physical_expressions(
    df: &DataFrame,
    exprs: &[Arc<dyn PhysicalExpr>],
) -> Result<DataFrame> {
    let height = df.height();
    let mut selected_columns = exprs
        .par_iter()
        .map(|expr| expr.evaluate(df))
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

impl Executor for StandardExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let df = self.input.execute(cache)?;

        let df = evaluate_physical_expressions(&df, &self.expr);
        if std::env::var(POLARS_VERBOSE).is_ok() {
            println!("operation {} on dataframe finished", self.operation);
        }
        df
    }
}

pub struct DataFrameOpsExec {
    input: Box<dyn Executor>,
    operation: DataFrameOperation,
}

impl DataFrameOpsExec {
    pub(crate) fn new(input: Box<dyn Executor>, operation: DataFrameOperation) -> Self {
        Self { input, operation }
    }
}

impl Executor for DataFrameOpsExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let df = self.input.execute(cache)?;
        let df = match &self.operation {
            DataFrameOperation::Sort { by_column, reverse } => df.sort(&by_column, *reverse),
            DataFrameOperation::Explode(column) => df.explode(column),
            DataFrameOperation::DropDuplicates {
                maintain_order,
                subset,
            } => df.drop_duplicates(*maintain_order, subset.as_ref().map(|v| v.as_ref())),
        };
        if std::env::var(POLARS_VERBOSE).is_ok() {
            println!("{:?} on dataframe finished", self.operation);
        }
        df
    }
}

/// Take an input Executor and a multiple expressions
pub struct GroupByExec {
    input: Box<dyn Executor>,
    keys: Arc<Vec<String>>,
    aggs: Vec<Arc<dyn PhysicalExpr>>,
}

impl GroupByExec {
    pub(crate) fn new(
        input: Box<dyn Executor>,
        keys: Arc<Vec<String>>,
        aggs: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self { input, keys, aggs }
    }
}

impl Executor for GroupByExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let df = self.input.execute(cache)?;
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
        let df = DataFrame::new_no_checks(columns);
        if std::env::var(POLARS_VERBOSE).is_ok() {
            println!("groupby {:?} on dataframe finished", self.keys);
        };
        Ok(df)
    }
}

pub struct JoinExec {
    input_left: Box<dyn Executor>,
    input_right: Box<dyn Executor>,
    how: JoinType,
    left_on: Arc<dyn PhysicalExpr>,
    right_on: Arc<dyn PhysicalExpr>,
}

impl JoinExec {
    pub(crate) fn new(
        input_left: Box<dyn Executor>,
        input_right: Box<dyn Executor>,
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
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        // rayon::join was dropped because that resulted in a deadlock when there were dependencies
        // between the DataFrames. Like joining a specific computation of a DF back to itself.
        let df_left = self.input_left.execute(cache);
        let df_right = self.input_right.execute(cache);

        let df_left = df_left?;
        let df_right = df_right?;

        let s_left = self.left_on.evaluate(&df_left)?;
        let s_right = self.right_on.evaluate(&df_right)?;

        use JoinType::*;
        let df = match self.how {
            Left => df_left.left_join_from_series(&df_right, &s_left, &s_right),
            Inner => df_left.inner_join_from_series(&df_right, &s_left, &s_right),
            Outer => df_left.outer_join_from_series(&df_right, &s_left, &s_right),
        };
        if std::env::var(POLARS_VERBOSE).is_ok() {
            println!("{:?} join dataframes finished", self.how);
        };
        df
    }
}
pub struct StackExec {
    input: Box<dyn Executor>,
    expr: Vec<Arc<dyn PhysicalExpr>>,
}

impl StackExec {
    pub(crate) fn new(input: Box<dyn Executor>, expr: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        Self { input, expr }
    }
}

impl Executor for StackExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let mut df = self.input.execute(cache)?;
        let height = df.height();

        let res: Result<_> = self.expr.iter().try_for_each(|expr| {
            let s = expr.evaluate(&df).map(|series| {
                // literal series. Should be whole column size
                if series.len() == 1 && height > 1 {
                    series.expand_at_index(0, height)
                } else {
                    series
                }
            })?;

            let name = s.name().to_string();
            df.replace_or_add(&name, s)?;
            if std::env::var(POLARS_VERBOSE).is_ok() {
                println!("added column {} to dataframe", name);
            }
            Ok(())
        });
        let _ = res?;
        Ok(df)
    }
}
