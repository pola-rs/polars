use super::dsl::*;
use crate::dataframe::JsDataFrame;
use crate::prelude::*;
use napi::{Env, Task};
use polars::io::RowCount;
use polars::lazy::frame::{LazyCsvReader, LazyFrame, LazyGroupBy};

#[napi]
#[repr(transparent)]
pub struct JsLazyGroupBy {
    // option because we cannot get a self by value in pyo3
    lgb: Option<LazyGroupBy>,
}

#[napi]
#[repr(transparent)]
#[derive(Clone)]
pub struct JsLazyFrame {
    ldf: LazyFrame,
}
impl From<LazyFrame> for JsLazyFrame {
    fn from(ldf: LazyFrame) -> Self {
        JsLazyFrame { ldf }
    }
}

#[napi]
impl JsLazyGroupBy {
    #[napi]
    pub fn agg(&mut self, aggs: Vec<&JsExpr>) -> JsLazyFrame {
        let lgb = self.lgb.take().unwrap();
        lgb.agg(aggs.to_exprs()).into()
    }
    #[napi]
    pub fn head(&mut self, n: i64) -> JsLazyFrame {
        let lgb = self.lgb.take().unwrap();
        lgb.head(Some(n as usize)).into()
    }
    #[napi]
    pub fn tail(&mut self, n: i64) -> JsLazyFrame {
        let lgb = self.lgb.take().unwrap();
        lgb.tail(Some(n as usize)).into()
    }
}

#[napi]
impl JsLazyFrame {
    #[napi]
    pub fn to_js(&self, env: Env) -> napi::Result<napi::JsUnknown> {
        env.to_js_value(&self.ldf.logical_plan)
    }

    #[napi]
    pub fn serialize(&self, format: String) -> napi::Result<Buffer> {
        let buf = match format.as_ref() {
            "bincode" => bincode::serialize(&self.ldf.logical_plan)
                .map_err(|err| napi::Error::from_reason(format!("{:?}", err)))?,
            "json" => serde_json::to_vec(&self.ldf.logical_plan)
                .map_err(|err| napi::Error::from_reason(format!("{:?}", err)))?,
            _ => {
                return Err(napi::Error::from_reason(
                    "unexpected format. \n supported options are 'json', 'bincode'".to_owned(),
                ))
            }
        };
        Ok(Buffer::from(buf))
    }

    #[napi(factory)]
    pub fn deserialize(buf: Buffer, format: String) -> napi::Result<JsLazyFrame> {
        let lp: LogicalPlan = match format.as_ref() {
            "bincode" => bincode::deserialize(&buf)
                .map_err(|err| napi::Error::from_reason(format!("{:?}", err)))?,
            "json" => serde_json::from_slice(&buf)
                .map_err(|err| napi::Error::from_reason(format!("{:?}", err)))?,
            _ => {
                return Err(napi::Error::from_reason(
                    "unexpected format. \n supported options are 'json', 'bincode'".to_owned(),
                ))
            }
        };
        let mut opt_state = OptState::default();
        opt_state.file_caching = true;
        let lf = LazyFrame::from(lp).with_optimizations(opt_state);
        Ok(lf.into())
    }
    #[napi(factory)]
    pub fn clone_external(lf: &JsLazyFrame) -> napi::Result<JsLazyFrame> {
        Ok(lf.clone())
    }
    #[napi]
    pub fn describe_plan(&self) -> String {
        self.ldf.describe_plan()
    }
    #[napi]
    pub fn describe_optimized_plan(&self) -> napi::Result<String> {
        let result = self
            .ldf
            .describe_optimized_plan()
            .map_err(JsPolarsErr::from)?;
        Ok(result)
    }
    #[napi]
    pub fn to_dot(&self, optimized: bool) -> napi::Result<String> {
        let result = self.ldf.to_dot(optimized).map_err(JsPolarsErr::from)?;
        Ok(result)
    }
    #[napi]
    pub fn optimization_toggle(
        &self,
        type_coercion: Option<bool>,
        predicate_pushdown: Option<bool>,
        projection_pushdown: Option<bool>,
        simplify_expr: Option<bool>,
        string_cache: Option<bool>,
        slice_pushdown: Option<bool>,
    ) -> JsLazyFrame {
        let type_coercion = type_coercion.unwrap_or(true);
        let predicate_pushdown = predicate_pushdown.unwrap_or(true);
        let projection_pushdown = projection_pushdown.unwrap_or(true);
        let simplify_expr = simplify_expr.unwrap_or(true);
        let string_cache = string_cache.unwrap_or(true);
        let slice_pushdown = slice_pushdown.unwrap_or(true);

        let ldf = self.ldf.clone();
        let ldf = ldf
            .with_type_coercion(type_coercion)
            .with_predicate_pushdown(predicate_pushdown)
            .with_simplify_expr(simplify_expr)
            .with_string_cache(string_cache)
            .with_slice_pushdown(slice_pushdown)
            .with_projection_pushdown(projection_pushdown);
        ldf.into()
    }
    #[napi]
    pub fn sort(&self, by_column: String, reverse: bool, nulls_last: bool) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.sort(
            &by_column,
            SortOptions {
                descending: reverse,
                nulls_last,
            },
        )
        .into()
    }
    #[napi]
    pub fn sort_by_exprs(
        &self,
        by_column: Vec<&JsExpr>,
        reverse: Vec<bool>,
        nulls_last: bool,
    ) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.sort_by_exprs(by_column.to_exprs(), reverse, nulls_last)
            .into()
    }
    #[napi]
    pub fn cache(&self) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.cache().into()
    }
    #[napi]
    pub fn collect_sync(&self) -> napi::Result<JsDataFrame> {
        let ldf = self.ldf.clone();
        let df = ldf.collect().map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi(ts_return_type = "Promise<JsDataFrame>")]
    pub fn collect(&self) -> AsyncTask<AsyncCollect> {
        let ldf = self.ldf.clone();
        AsyncTask::new(AsyncCollect(ldf))
    }

    #[napi(ts_return_type = "Promise<JsDataFrame>")]
    pub fn fetch(&self, n_rows: i64) -> AsyncTask<AsyncFetch> {
        let ldf = self.ldf.clone();
        AsyncTask::new(AsyncFetch((ldf, n_rows as usize)))
    }

    #[napi]
    pub fn fetch_sync(&self, n_rows: i64) -> napi::Result<JsDataFrame> {
        let ldf = self.ldf.clone();
        let df = ldf.fetch(n_rows as usize).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn filter(&mut self, predicate: &JsExpr) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.filter(predicate.inner.clone()).into()
    }
    #[napi]
    pub fn select(&mut self, exprs: Vec<&JsExpr>) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.select(exprs.to_exprs()).into()
    }
    #[napi]
    pub fn groupby(&mut self, by: Vec<&JsExpr>, maintain_order: bool) -> JsLazyGroupBy {
        let ldf = self.ldf.clone();
        let by = by.to_exprs();
        let lazy_gb = if maintain_order {
            ldf.groupby_stable(by)
        } else {
            ldf.groupby(by)
        };

        JsLazyGroupBy { lgb: Some(lazy_gb) }
    }
    #[napi]
    pub fn groupby_rolling(
        &mut self,
        index_column: String,
        period: String,
        offset: String,
        closed: Wrap<ClosedWindow>,
        by: Vec<&JsExpr>,
    ) -> JsLazyGroupBy {
        let closed_window = closed.0;
        let ldf = self.ldf.clone();
        let by = by.to_exprs();
        let lazy_gb = ldf.groupby_rolling(
            by,
            RollingGroupOptions {
                index_column,
                period: Duration::parse(&period),
                offset: Duration::parse(&offset),
                closed_window,
            },
        );

        JsLazyGroupBy { lgb: Some(lazy_gb) }
    }

    #[allow(clippy::too_many_arguments)]
    #[napi]
    pub fn groupby_dynamic(
        &mut self,
        index_column: String,
        every: String,
        period: String,
        offset: String,
        truncate: bool,
        include_boundaries: bool,
        closed: Wrap<ClosedWindow>,
        by: Vec<&JsExpr>,
    ) -> JsLazyGroupBy {
        let closed_window = closed.0;
        let by = by.to_exprs();
        let ldf = self.ldf.clone();
        let lazy_gb = ldf.groupby_dynamic(
            by,
            DynamicGroupOptions {
                index_column,
                every: Duration::parse(&every),
                period: Duration::parse(&period),
                offset: Duration::parse(&offset),
                truncate,
                include_boundaries,
                closed_window,
            },
        );

        JsLazyGroupBy { lgb: Some(lazy_gb) }
    }
    #[allow(clippy::too_many_arguments)]
    #[napi]
    pub fn join_asof(
        &self,
        other: &JsLazyFrame,
        left_on: &JsExpr,
        right_on: &JsExpr,
        left_by: Option<Vec<String>>,
        right_by: Option<Vec<String>>,
        allow_parallel: bool,
        force_parallel: bool,
        suffix: String,
        strategy: String,
        tolerance: Option<Wrap<AnyValue<'_>>>,
        tolerance_str: Option<String>,
    ) -> JsLazyFrame {
        let strategy = match strategy.as_ref() {
            "forward" => AsofStrategy::Forward,
            "backward" => AsofStrategy::Backward,
            _ => panic!("expected on of {{'forward', 'backward'}}"),
        };

        let ldf = self.ldf.clone();
        let other = other.ldf.clone();
        let left_on = left_on.inner.clone();
        let right_on = right_on.inner.clone();
        ldf.join_builder()
            .with(other)
            .left_on([left_on])
            .right_on([right_on])
            .allow_parallel(allow_parallel)
            .force_parallel(force_parallel)
            .how(JoinType::AsOf(AsOfOptions {
                strategy,
                left_by,
                right_by,
                tolerance: tolerance.map(|t| t.0.into_static().unwrap()),
                tolerance_str,
            }))
            .suffix(suffix)
            .finish()
            .into()
    }
    #[allow(clippy::too_many_arguments)]
    #[napi]
    pub fn join(
        &self,
        other: &JsLazyFrame,
        left_on: Vec<&JsExpr>,
        right_on: Vec<&JsExpr>,
        allow_parallel: bool,
        force_parallel: bool,
        how: String,
        suffix: String,
        asof_by_left: Vec<String>,
        asof_by_right: Vec<String>,
    ) -> JsLazyFrame {
        let how = match how.as_ref() {
            "left" => JoinType::Left,
            "inner" => JoinType::Inner,
            "outer" => JoinType::Outer,
            "semi" => JoinType::Semi,
            "anti" => JoinType::Anti,
            "asof" => JoinType::AsOf(AsOfOptions {
                strategy: AsofStrategy::Backward,
                left_by: if asof_by_left.is_empty() {
                    None
                } else {
                    Some(asof_by_left)
                },
                right_by: if asof_by_right.is_empty() {
                    None
                } else {
                    Some(asof_by_right)
                },
                tolerance: None,
                tolerance_str: None,
            }),
            "cross" => JoinType::Cross,
            _ => panic!("not supported"),
        };

        let ldf = self.ldf.clone();
        let other = other.ldf.clone();
        let left_on = left_on.to_exprs();
        let right_on = right_on.to_exprs();

        ldf.join_builder()
            .with(other)
            .left_on(left_on)
            .right_on(right_on)
            .allow_parallel(allow_parallel)
            .force_parallel(force_parallel)
            .how(how)
            .suffix(suffix)
            .finish()
            .into()
    }
    #[napi]
    pub fn with_column(&mut self, expr: &JsExpr) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.with_column(expr.inner.clone()).into()
    }
    #[napi]
    pub fn with_columns(&mut self, exprs: Vec<&JsExpr>) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.with_columns(exprs.to_exprs()).into()
    }
    #[napi]
    pub fn rename(&mut self, existing: Vec<String>, new_names: Vec<String>) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.rename(existing, new_names).into()
    }
    #[napi]
    pub fn reverse(&self) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.reverse().into()
    }
    #[napi]
    pub fn shift(&self, periods: i64) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.shift(periods).into()
    }
    #[napi]
    pub fn shift_and_fill(&self, periods: i64, fill_value: &JsExpr) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.shift_and_fill(periods, fill_value.inner.clone()).into()
    }

    #[napi]
    pub fn fill_null(&self, fill_value: &JsExpr) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.fill_null(fill_value.inner.clone()).into()
    }

    #[napi]
    pub fn fill_nan(&self, fill_value: &JsExpr) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.fill_nan(fill_value.inner.clone()).into()
    }

    #[napi]
    pub fn min(&self) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.min().into()
    }

    #[napi]
    pub fn max(&self) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.max().into()
    }

    #[napi]
    pub fn sum(&self) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.sum().into()
    }

    #[napi]
    pub fn mean(&self) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.mean().into()
    }

    #[napi]
    pub fn std(&self) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.std().into()
    }

    #[napi]
    pub fn var(&self) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.var().into()
    }

    #[napi]
    pub fn median(&self) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.median().into()
    }
    #[napi]
    pub fn quantile(
        &self,
        quantile: f64,
        interpolation: Wrap<QuantileInterpolOptions>,
    ) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.quantile(quantile, interpolation.0).into()
    }

    #[napi]
    pub fn explode(&self, column: Vec<&JsExpr>) -> JsLazyFrame {
        let ldf = self.ldf.clone();

        ldf.explode(column.to_exprs()).into()
    }
    #[napi]
    pub fn unique(
        &self,
        maintain_order: bool,
        subset: Option<Vec<String>>,
        keep: Wrap<UniqueKeepStrategy>,
    ) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        match maintain_order {
            true => ldf.unique_stable(subset, keep.0),
            false => ldf.unique(subset, keep.0),
        }
        .into()
    }
    #[napi]
    pub fn drop_nulls(&self, subset: Option<Vec<String>>) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.drop_nulls(subset.map(|v| v.into_iter().map(|s| col(&s)).collect()))
            .into()
    }
    #[napi]
    pub fn slice(&self, offset: i64, len: u32) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.slice(offset, len).into()
    }
    #[napi]
    pub fn tail(&self, n: u32) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.tail(n).into()
    }
    #[napi]
    pub fn melt(
        &self,
        id_vars: Vec<String>,
        value_vars: Vec<String>,
        value_name: Option<String>,
        variable_name: Option<String>,
    ) -> JsLazyFrame {
        let args = MeltArgs {
            id_vars,
            value_vars,
            value_name,
            variable_name,
        };

        let ldf = self.ldf.clone();
        ldf.melt(args).into()
    }

    #[napi]
    pub fn with_row_count(&self, name: String, offset: Option<u32>) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.with_row_count(&name, offset).into()
    }

    #[napi]
    pub fn drop_columns(&self, cols: Vec<String>) -> JsLazyFrame {
        let ldf = self.ldf.clone();
        ldf.drop_columns(cols).into()
    }
    #[napi(js_name = "clone")]
    pub fn clone(&self) -> JsLazyFrame {
        self.ldf.clone().into()
    }

    #[napi(getter, js_name = "columns")]
    pub fn columns(&self) -> Vec<String> {
        self.ldf.schema().iter_names().cloned().collect()
    }

    #[napi]
    pub fn unnest(&self, cols: Vec<String>) -> JsLazyFrame {
        self.ldf.clone().unnest(cols).into()
    }
}

#[napi(object)]
pub struct ScanCsvOptions {
    pub infer_schema_length: Option<u32>,
    pub cache: Option<bool>,
    pub has_header: Option<bool>,
    pub ignore_errors: bool,
    pub n_rows: Option<u32>,
    pub skip_rows: Option<u32>,
    pub sep: String,
    pub rechunk: Option<bool>,
    pub columns: Option<Vec<String>>,
    pub encoding: String,
    pub low_memory: Option<bool>,
    pub comment_char: Option<String>,
    pub quote_char: Option<String>,
    pub parse_dates: Option<bool>,
    pub skip_rows_after_header: u32,
    pub row_count: Option<JsRowCount>,
}
#[napi]
pub fn scan_csv(path: String, options: ScanCsvOptions) -> napi::Result<JsLazyFrame> {
    let cache = options.cache.unwrap_or(true);
    let has_header = options.has_header.unwrap_or(true);
    let low_memory = options.low_memory.unwrap_or(false);
    let parse_dates = options.parse_dates.unwrap_or(false);
    let rechunk = options.rechunk.unwrap_or(false);
    let skip_rows = options.skip_rows.unwrap_or(0) as usize;

    let infer_schema_length = options.infer_schema_length.unwrap_or(100) as usize;
    let n_rows = options.n_rows.map(|i| i as usize);
    let comment_char = options.comment_char.map(|s| s.as_bytes()[0]);
    let row_count = options.row_count.map(RowCount::from);
    let quote_char = if let Some(s) = options.quote_char {
        if s.is_empty() {
            None
        } else {
            Some(s.as_bytes()[0])
        }
    } else {
        None
    };

    let encoding = match options.encoding.as_ref() {
        "utf8" => CsvEncoding::Utf8,
        "utf8-lossy" => CsvEncoding::LossyUtf8,
        e => return Err(JsPolarsErr::Other(format!("encoding not {} not implemented.", e)).into()),
    };
    let r = LazyCsvReader::new(path)
        // .with_chunk_size()
        .with_infer_schema_length(Some(infer_schema_length))
        .with_delimiter(options.sep.as_bytes()[0])
        .has_header(has_header)
        .with_ignore_parser_errors(parse_dates)
        .with_skip_rows(skip_rows)
        .with_n_rows(n_rows)
        .with_cache(cache)
        // // .with_dtype_overwrite(overwrite_dtype.as_ref())
        .low_memory(low_memory)
        .with_comment_char(comment_char)
        .with_quote_char(quote_char)
        .with_rechunk(rechunk)
        .with_skip_rows_after_header(skip_rows)
        .with_encoding(encoding)
        .with_row_count(row_count)
        .with_parse_dates(parse_dates)
        .finish()
        .map_err(JsPolarsErr::from)?;
    // .with_null_values(null_values)
    Ok(r.into())
}

#[napi(object)]
pub struct ScanParquetOptions {
    pub n_rows: Option<i64>,
    pub cache: Option<bool>,
    pub rechunk: Option<bool>,
    pub row_count: Option<JsRowCount>,
    pub low_memory: Option<bool>,
}

#[napi]
pub fn scan_parquet(
    path: String,
    options: ScanParquetOptions,
    parallel: Wrap<ParallelStrategy>,
) -> napi::Result<JsLazyFrame> {
    let n_rows = options.n_rows.map(|i| i as usize);
    let cache = options.cache.unwrap_or(true);
    let rechunk = options.rechunk.unwrap_or(false);
    let low_memory = options.low_memory.unwrap_or(false);
    let row_count: Option<RowCount> = options.row_count.map(|rc| rc.into());
    let args = ScanArgsParquet {
        n_rows,
        cache,
        parallel: parallel.0,
        rechunk,
        row_count,
        low_memory,
    };
    let lf = LazyFrame::scan_parquet(path, args).map_err(JsPolarsErr::from)?;
    Ok(lf.into())
}

#[napi(object)]
pub struct ScanIPCOptions {
    pub n_rows: Option<i64>,
    pub cache: Option<bool>,
    pub rechunk: Option<bool>,
    pub row_count: Option<JsRowCount>,
}

#[napi]
pub fn scan_ipc(path: String, options: ScanIPCOptions) -> napi::Result<JsLazyFrame> {
    let n_rows = options.n_rows.map(|i| i as usize);
    let cache = options.cache.unwrap_or(true);
    let rechunk = options.rechunk.unwrap_or(false);
    let row_count: Option<RowCount> = options.row_count.map(|rc| rc.into());
    let args = ScanArgsIpc {
        n_rows,
        cache,
        rechunk,
        row_count,
    };
    let lf = LazyFrame::scan_ipc(path, args).map_err(JsPolarsErr::from)?;
    Ok(lf.into())
}

struct JsonScan {
    path: String,
    batch_size: usize,
}

impl AnonymousScan for JsonScan {
    fn scan(&self, scan_opts: AnonymousScanOptions) -> polars::prelude::Result<DataFrame> {
        let schema = scan_opts.output_schema.unwrap_or(scan_opts.schema);
        JsonLineReader::from_path(&self.path)
            .expect("unable to read file")
            .with_schema(&schema)
            .with_chunk_size(self.batch_size)
            .with_n_rows(scan_opts.n_rows)
            .finish()
    }

    fn schema(&self, infer_schema_length: Option<usize>) -> polars::prelude::Result<Schema> {
        let f = std::fs::File::open(&self.path)?;
        let mut reader = std::io::BufReader::new(f);

        let data_type = ndjson::read::infer(&mut reader, infer_schema_length)
            .map_err(|err| PolarsError::ComputeError(format!("{:#?}", err).into()))?;
        let schema: polars_core::prelude::Schema = StructArray::get_fields(&data_type).into();

        Ok(schema)
    }
}

#[napi(object)]
pub struct JsonScanOptions {
    pub infer_schema_length: i64,
    pub batch_size: i64,
}

#[napi]
pub fn scan_json(path: String, options: JsonScanOptions) -> napi::Result<JsLazyFrame> {
    let f = JsonScan {
        path,
        batch_size: options.batch_size as usize,
    };

    let options = ScanArgsAnonymous {
        name: "JSON SCAN",
        infer_schema_length: Some(options.infer_schema_length as usize),
        ..ScanArgsAnonymous::default()
    };
    let lf =
        LazyFrame::anonymous_scan(std::sync::Arc::new(f), options).map_err(JsPolarsErr::from)?;
    Ok(lf.into())
}

pub struct AsyncFetch((LazyFrame, usize));

impl Task for AsyncFetch {
    type Output = DataFrame;
    type JsValue = JsDataFrame;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let (ldf, n_rows) = &self.0;
        let ldf = ldf.clone();
        let df = ldf.fetch(*n_rows).map_err(JsPolarsErr::from)?;
        Ok(df)
    }

    fn resolve(&mut self, _env: Env, df: DataFrame) -> napi::Result<Self::JsValue> {
        Ok(df.into())
    }
}
pub struct AsyncCollect(LazyFrame);

impl Task for AsyncCollect {
    type Output = DataFrame;
    type JsValue = JsDataFrame;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let ldf = self.0.clone();
        let df = ldf.collect().map_err(JsPolarsErr::from)?;
        Ok(df)
    }

    fn resolve(&mut self, _env: Env, df: DataFrame) -> napi::Result<Self::JsValue> {
        Ok(df.into())
    }
}
