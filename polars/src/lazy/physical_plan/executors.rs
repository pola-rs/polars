use super::*;
use crate::frame::hash_join::JoinType;
use crate::frame::ser::{csv::CsvEncoding, ScanAggregation};
use crate::lazy::logical_plan::FETCH_ROWS;
use arrow::datatypes::SchemaRef;
use itertools::Itertools;
use rayon::prelude::*;
use std::mem;

const POLARS_VERBOSE: &str = "POLARS_VERBOSE";

fn set_n_rows(stop_after_n_rows: Option<usize>) -> Option<usize> {
    let fetch_rows = FETCH_ROWS.with(|fetch_rows| fetch_rows.get());
    match fetch_rows {
        None => stop_after_n_rows,
        Some(n) => Some(n),
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
    aggregate: Vec<ScanAggregation>,
    stop_after_n_rows: Option<usize>,
    cache: bool,
}

#[cfg(feature = "parquet")]
impl ParquetExec {
    pub(crate) fn new(
        path: String,
        schema: Schema,
        with_columns: Option<Vec<String>>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        aggregate: Vec<ScanAggregation>,
        stop_after_n_rows: Option<usize>,
        cache: bool,
    ) -> Self {
        ParquetExec {
            path,
            schema,
            with_columns,
            predicate,
            aggregate,
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

        let stop_after_n_rows = set_n_rows(self.stop_after_n_rows);
        let aggregate = if self.aggregate.is_empty() {
            None
        } else {
            Some(self.aggregate.as_slice())
        };

        let df = ParquetReader::new(file)
            .with_stop_after_n_rows(stop_after_n_rows)
            .finish_with_can_ops(
                self.predicate.clone(),
                aggregate,
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
    schema: SchemaRef,
    has_header: bool,
    delimiter: u8,
    ignore_errors: bool,
    skip_rows: usize,
    stop_after_n_rows: Option<usize>,
    with_columns: Option<Vec<String>>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    aggregate: Vec<ScanAggregation>,
    cache: bool,
}

impl CsvExec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        path: String,
        schema: SchemaRef,
        has_header: bool,
        delimiter: u8,
        ignore_errors: bool,
        skip_rows: usize,
        stop_after_n_rows: Option<usize>,
        with_columns: Option<Vec<String>>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        aggregate: Vec<ScanAggregation>,
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
            aggregate,
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

        let mut with_columns = mem::take(&mut self.with_columns);
        let mut projected_len = 0;
        with_columns.as_ref().map(|columns| {
            projected_len = columns.len();
            columns
        });

        if projected_len == 0 {
            with_columns = None;
        }
        let stop_after_n_rows = set_n_rows(self.stop_after_n_rows);

        let reader = CsvReader::from_path(&self.path)
            .unwrap()
            .has_header(self.has_header)
            .with_schema(self.schema.clone())
            .with_delimiter(self.delimiter)
            .with_ignore_parser_errors(self.ignore_errors)
            .with_skip_rows(self.skip_rows)
            .with_stop_after_n_rows(stop_after_n_rows)
            .with_columns(with_columns)
            .with_encoding(CsvEncoding::LossyUtf8);

        let aggregate = if self.aggregate.is_empty() {
            None
        } else {
            Some(self.aggregate.as_slice())
        };

        let df = reader.finish_with_scan_ops(self.predicate.clone(), aggregate)?;

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

pub(crate) fn evaluate_physical_expressions(
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

pub(crate) struct ExplodeExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) columns: Vec<String>,
}

impl Executor for ExplodeExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let df = self.input.execute(cache)?;
        df.explode(&self.columns)
    }
}

pub(crate) struct SortExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) by_column: String,
    pub(crate) reverse: bool,
}

impl Executor for SortExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let df = self.input.execute(cache)?;
        df.sort(&self.by_column, self.reverse)
    }
}

pub(crate) struct DropDuplicatesExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) maintain_order: bool,
    pub(crate) subset: Option<Vec<String>>,
}

impl Executor for DropDuplicatesExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let df = self.input.execute(cache)?;
        df.drop_duplicates(
            self.maintain_order,
            self.subset.as_ref().map(|v| v.as_ref()),
        )
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
                if agg.len() != groups.len() {
                    panic!(format!(
                        "returned aggregation is a different length: {} than the group lengths: {}",
                        agg.len(),
                        groups.len()
                    ))
                }
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
    input_left: Option<Box<dyn Executor>>,
    input_right: Option<Box<dyn Executor>>,
    how: JoinType,
    left_on: Vec<Arc<dyn PhysicalExpr>>,
    right_on: Vec<Arc<dyn PhysicalExpr>>,
    parallel: bool,
}

impl JoinExec {
    pub(crate) fn new(
        input_left: Box<dyn Executor>,
        input_right: Box<dyn Executor>,
        how: JoinType,
        left_on: Vec<Arc<dyn PhysicalExpr>>,
        right_on: Vec<Arc<dyn PhysicalExpr>>,
        parallel: bool,
    ) -> Self {
        JoinExec {
            input_left: Some(input_left),
            input_right: Some(input_right),
            how,
            left_on,
            right_on,
            parallel,
        }
    }
}

impl Executor for JoinExec {
    fn execute<'a>(&'a mut self, cache: &'a Cache) -> Result<DataFrame> {
        let mut input_left = self.input_left.take().unwrap();
        let mut input_right = self.input_right.take().unwrap();

        let (df_left, df_right) = if self.parallel {
            let cache_left = cache.clone();
            let cache_right = cache.clone();
            // propagate the fetch_rows static value to the spawning threads.
            let fetch_rows = FETCH_ROWS.with(|fetch_rows| fetch_rows.get());

            let h_left = std::thread::spawn(move || {
                FETCH_ROWS.with(|fr| fr.set(fetch_rows));
                input_left.execute(&cache_left)
            });
            let h_right = std::thread::spawn(move || {
                FETCH_ROWS.with(|fr| fr.set(fetch_rows));
                input_right.execute(&cache_right)
            });
            (h_left.join().unwrap(), h_right.join().unwrap())
        } else {
            (input_left.execute(&cache), input_right.execute(&cache))
        };

        let df_left = df_left?;
        let df_right = df_right?;

        let left_names = self
            .left_on
            .iter()
            .map(|e| e.evaluate(&df_left).map(|s| s.name().to_string()))
            .collect::<Result<Vec<_>>>()?;

        let right_names = self
            .right_on
            .iter()
            .map(|e| e.evaluate(&df_right).map(|s| s.name().to_string()))
            .collect::<Result<Vec<_>>>()?;

        let df = df_left.join(&df_right, &left_names, &right_names, self.how);
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

pub struct SliceExec {
    pub input: Box<dyn Executor>,
    pub offset: usize,
    pub len: usize,
}

impl Executor for SliceExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let df = self.input.execute(cache)?;
        df.slice(self.offset, self.len)
    }
}
pub struct MeltExec {
    pub input: Box<dyn Executor>,
    pub id_vars: Arc<Vec<String>>,
    pub value_vars: Arc<Vec<String>>,
}

impl Executor for MeltExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let df = self.input.execute(cache)?;
        df.melt(&self.id_vars.as_slice(), &self.value_vars.as_slice())
    }
}
