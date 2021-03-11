use super::*;
use crate::logical_plan::{Context, FETCH_ROWS};
use crate::utils::rename_expr_root_name;
use itertools::Itertools;
use polars_core::utils::{accumulate_dataframes_vertical, num_cpus, split_df};
use polars_core::{frame::hash_join::JoinType, POOL};
use polars_io::prelude::*;
use polars_io::{csv::CsvEncoding, ScanAggregation};
use rayon::prelude::*;
use std::io::{Read, Seek};
use std::mem;

trait FinishScanOps {
    /// Read the file and create the DataFrame. Used from lazy execution
    fn finish_with_scan_ops(
        self,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        aggregate: Option<&[ScanAggregation]>,
    ) -> Result<DataFrame>;
}

impl<'a, R: 'static + Read + Seek + Sync + Send> FinishScanOps for CsvReader<'a, R> {
    fn finish_with_scan_ops(
        self,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        aggregate: Option<&[ScanAggregation]>,
    ) -> Result<DataFrame> {
        let predicate =
            predicate.map(|expr| Arc::new(PhysicalIoHelper::new(expr)) as Arc<dyn PhysicalIoExpr>);

        let rechunk = self.rechunk;
        let mut csv_reader = self.build_inner_reader()?;
        let df = csv_reader.as_df(predicate, aggregate)?;
        match rechunk {
            true => Ok(df.agg_chunks()),
            false => Ok(df),
        }
    }
}

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
    schema: SchemaRef,
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
        schema: SchemaRef,
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
        let schema = mem::take(&mut self.schema);

        let projection: Option<Vec<_>> = with_columns.map(|with_columns| {
            with_columns
                .iter()
                .map(|name| schema.column_with_name(name).unwrap().0)
                .collect()
        });

        let stop_after_n_rows = set_n_rows(self.stop_after_n_rows);
        let aggregate = if self.aggregate.is_empty() {
            None
        } else {
            Some(self.aggregate.as_slice())
        };
        let predicate = self
            .predicate
            .clone()
            .map(|expr| Arc::new(PhysicalIoHelper::new(expr)) as Arc<dyn PhysicalIoExpr>);

        let df = ParquetReader::new(file)
            .with_stop_after_n_rows(stop_after_n_rows)
            .finish_with_scan_ops(
                predicate,
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

        if let Some(limit) = set_n_rows(None) {
            Ok(df.head(Some(limit)))
        } else {
            Ok(df)
        }
    }
}

/// Take an input Executor (creates the input DataFrame)
/// and a multiple PhysicalExpressions (create the output Series)
pub struct StandardExec {
    /// i.e. sort, projection
    operation: &'static str,
    input: Box<dyn Executor>,
    expr: Vec<Arc<dyn PhysicalExpr>>,
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
    keys: Vec<Arc<dyn PhysicalExpr>>,
    aggs: Vec<Arc<dyn PhysicalExpr>>,
    apply: Option<Arc<dyn DataFrameUdf>>,
}

impl GroupByExec {
    pub(crate) fn new(
        input: Box<dyn Executor>,
        keys: Vec<Arc<dyn PhysicalExpr>>,
        aggs: Vec<Arc<dyn PhysicalExpr>>,
        apply: Option<Arc<dyn DataFrameUdf>>,
    ) -> Self {
        Self {
            input,
            keys,
            aggs,
            apply,
        }
    }
}

fn groupby_helper(
    df: DataFrame,
    keys: Vec<Series>,
    aggs: &[Arc<dyn PhysicalExpr>],
    apply: Option<&Arc<dyn DataFrameUdf>>,
) -> Result<DataFrame> {
    let gb = df.groupby_with_series(keys, true)?;
    if let Some(f) = apply {
        return gb.apply(|df| f.call_udf(df));
    }

    let groups = gb.get_groups();

    let mut columns = gb.keys();

    let agg_columns = POOL.install(|| {
       aggs
           // benchmarked that using iter was 5% faster than par_iter on db-benchmark q4
           // probably less congestion.
            .iter()
            .map(|expr| {
                let agg_expr = expr.as_agg_expr()?;
                let opt_agg = agg_expr.evaluate(&df, groups)?;
                if let Some(agg) = &opt_agg {
                    if agg.len() != groups.len() {
                        panic!(
                            "returned aggregation is a different length: {} than the group lengths: {}",
                            agg.len(),
                            groups.len()
                        )
                    }
                };
                Ok(opt_agg)
            })
            .collect::<Result<Vec<_>>>()
    })?;

    columns.extend(agg_columns.into_iter().flatten());

    let df = DataFrame::new_no_checks(columns);
    Ok(df)
}

impl Executor for GroupByExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let df = self.input.execute(cache)?;
        let keys = self
            .keys
            .iter()
            .map(|e| e.evaluate(&df))
            .collect::<Result<_>>()?;
        groupby_helper(df, keys, &self.aggs, self.apply.as_ref())
    }
}

/// Take an input Executor and a multiple expressions
pub struct PartitionGroupByExec {
    input: Box<dyn Executor>,
    keys: Vec<Arc<dyn PhysicalExpr>>,
    phys_aggs: Vec<Arc<dyn PhysicalExpr>>,
    aggs: Vec<Expr>,
}

impl PartitionGroupByExec {
    pub(crate) fn new(
        input: Box<dyn Executor>,
        keys: Vec<Arc<dyn PhysicalExpr>>,
        phys_aggs: Vec<Arc<dyn PhysicalExpr>>,
        aggs: Vec<Expr>,
    ) -> Self {
        Self {
            input,
            keys,
            phys_aggs,
            aggs,
        }
    }
}

impl Executor for PartitionGroupByExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let original_df = self.input.execute(cache)?;

        // already get the keys. This is the very last minute decision which groupby method we choose.
        // If the column is a categorical, we know the number of groups we have and can decide to continue
        // partitioned or go for the standard groupby. The partitioned is likely to be faster on a small number
        // of groups.
        let keys = self
            .keys
            .iter()
            .map(|e| e.evaluate(&original_df))
            .collect::<Result<Vec<_>>>()?;

        debug_assert_eq!(keys.len(), 1);
        let s = &keys[0];
        if let Ok(ca) = s.categorical() {
            let cat_map = ca
                .get_categorical_map()
                .expect("categorical type has categorical_map");
            let frac = cat_map.len() as f32 / ca.len() as f32;
            // TODO! proper benchmark which boundary should be chosen.
            if frac > 0.3 {
                return groupby_helper(original_df, keys, &self.phys_aggs, None);
            }
        }
        let mut expr_arena = Arena::with_capacity(64);

        // This will be the aggregation on the partition results. Due to the groupby
        // operation the column names have changed. This makes sure we can select the columns with
        // the new names. We also keep a hold on the names to make sure that we don't get a double
        // new name due to the double aggregation. These output_names will be used to rename the final
        // output
        let schema = original_df.schema();
        let aggs_and_names = self
            .aggs
            .iter()
            .map(|e| {
                let out_field = e.to_field(&schema, Context::Aggregation)?;
                let out_name = Arc::new(out_field.name().clone());
                let e = rename_expr_root_name(e, out_name.clone())?;
                let node = to_aexpr(e, &mut expr_arena);
                Ok((node, out_name))
            })
            .collect::<Result<Vec<_>>>()?;

        let planner = DefaultPlanner {};
        let outer_phys_aggs = aggs_and_names
            .iter()
            .map(|(e, _)| planner.create_physical_expr(*e, Context::Aggregation, &mut expr_arena))
            .collect::<Result<Vec<_>>>()?;

        let n_threads = num_cpus::get();
        // We do a partitioned groupby. Meaning that we first do the groupby operation arbitrarily
        // splitted on several threads. Than the final result we apply the same groupby again.
        let dfs = split_df(&original_df, n_threads)?;

        let dfs = POOL.install(|| {
            dfs.into_par_iter()
                .map(|df| {
                    let keys = self
                        .keys
                        .iter()
                        .map(|e| e.evaluate(&df))
                        .collect::<Result<Vec<_>>>()?;
                    let phys_aggs = &self.phys_aggs;
                    let gb = df.groupby_with_series(keys, false)?;
                    let groups = gb.get_groups();

                    let mut columns = gb.keys();
                    let agg_columns = phys_aggs
                        .iter()
                        .map(|expr| {
                            let agg_expr = expr.as_agg_expr()?;
                            let opt_agg = agg_expr.evaluate_partitioned(&df, groups)?;
                            if let Some(agg) = &opt_agg {
                                if agg[0].len() != groups.len() {
                                    panic!(
                                        "returned aggregation is a different length: {} than the group lengths: {}",
                                        agg.len(),
                                        groups.len()
                                    )
                                }
                            };
                            Ok(opt_agg)
                        }).collect::<Result<Vec<_>>>()?;

                    for agg in agg_columns.into_iter().flatten() {
                            for agg in agg {
                                columns.push(agg)
                            }
                    }

                    let df = DataFrame::new_no_checks(columns);
                    Ok(df)
                })
        }).collect::<Result<Vec<_>>>()?;

        let df = accumulate_dataframes_vertical(dfs)?;

        let keys = self
            .keys
            .iter()
            .map(|e| e.evaluate(&df))
            .collect::<Result<Vec<_>>>()?;

        // do the same on the outer results
        let gb = df.groupby_with_series(keys, true)?;
        let groups = gb.get_groups();

        let mut columns = gb.keys();
        let agg_columns = outer_phys_aggs
            .iter()
            .zip(aggs_and_names.iter().map(|(_, name)| name))
            .filter_map(|(expr, name)| {
                let agg_expr = expr.as_agg_expr().unwrap();
                // If None the column doesn't exist anymore.
                // For instance when summing a string this column will not be in the aggregation result
                let opt_agg = agg_expr.evaluate_partitioned_final(&df, groups).ok();
                opt_agg.map(|opt_s| {
                    opt_s.map(|mut s| {
                        s.rename(name);
                        s
                    })
                })
            });

        columns.extend(agg_columns.flatten());

        let df = DataFrame::new_no_checks(columns);
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

            POOL.join(
                move || {
                    FETCH_ROWS.with(|fr| fr.set(fetch_rows));
                    input_left.execute(&cache_left)
                },
                move || {
                    FETCH_ROWS.with(|fr| fr.set(fetch_rows));
                    input_right.execute(&cache_right)
                },
            )
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

pub(crate) struct UdfExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) function: Arc<dyn DataFrameUdf>,
}

impl Executor for UdfExec {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame> {
        let df = self.input.execute(cache)?;
        self.function.call_udf(df)
    }
}
