use crate::file::ThreadsafeWriteable;
use crate::prelude::*;
use crate::series::PySeries;
use napi::threadsafe_function::*;
use polars::frame::row::{infer_schema, rows_to_schema, Row};
use polars::io::RowCount;
use polars_core::series::ops::NullBehavior;
use std::io::{BufReader, BufWriter, Cursor, Read};
#[napi]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyDataFrame {
  df: DataFrame,
}

impl PyDataFrame {
  pub(crate) fn new(df: DataFrame) -> PyDataFrame {
    PyDataFrame { df }
  }
}
impl From<DataFrame> for PyDataFrame {
  fn from(s: DataFrame) -> PyDataFrame {
    PyDataFrame::new(s)
  }
}
use polars_core::utils::CustomIterTools;

type PyPolarsErr = JsPolarsErr;

pub(crate) fn to_series_collection(ps: Array) -> Vec<Series> {
  let len = ps.len();
  (0..len)
    .map(|idx| {
      let item: &PySeries = ps.get(idx).unwrap().unwrap();
      item.series.clone()
    })
    .collect()
}
pub(crate) fn to_pyseries_collection(s: Vec<Series>) -> Vec<PySeries> {
  let mut s = std::mem::ManuallyDrop::new(s);

  let p = s.as_mut_ptr() as *mut PySeries;
  let len = s.len();
  let cap = s.capacity();

  unsafe { Vec::from_raw_parts(p, len, cap) }
}

#[napi]
pub fn read_csv(
  path_or_buffer: Either<String, Buffer>,
  options: ReadCsvOptions,
) -> napi::Result<PyDataFrame> {
  let infer_schema_length = options.infer_schema_length.map(|i| i as usize);
  let n_threads = options.n_threads.map(|i| i as usize);
  let n_rows = options.n_rows.map(|i| i as usize);
  let skip_rows_after_header = options.skip_rows_after_header as usize;
  let skip_rows = options.skip_rows as usize;
  let chunk_size = options.chunk_size as usize;

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
    e => return Err(PyPolarsErr::Other(format!("encoding not {} not implemented.", e)).into()),
  };
  let df = match path_or_buffer {
    Either::A(path) => {
      CsvReader::from_path(path)
        .expect("unable to read file")
        .infer_schema(infer_schema_length)
        .has_header(options.has_header)
        .with_n_rows(n_rows)
        .with_delimiter(options.sep.as_bytes()[0])
        .with_skip_rows(skip_rows)
        .with_ignore_parser_errors(options.ignore_errors)
        .with_rechunk(options.rechunk)
        .with_chunk_size(chunk_size)
        .with_encoding(encoding)
        .with_columns(options.columns)
        .with_n_threads(n_threads)
        .low_memory(options.low_memory)
        .with_comment_char(comment_char)
        // .with_null_values(null_values)
        .with_parse_dates(options.parse_dates)
        .with_quote_char(quote_char)
        .with_row_count(row_count)
        .finish()
        .map_err(PyPolarsErr::from)?
    }
    Either::B(buffer) => {
      let cursor = Cursor::new(buffer.as_ref());
      CsvReader::new(cursor)
        .infer_schema(infer_schema_length)
        .has_header(options.has_header)
        .with_n_rows(n_rows)
        .with_delimiter(options.sep.as_bytes()[0])
        .with_skip_rows(skip_rows)
        .with_ignore_parser_errors(options.ignore_errors)
        .with_rechunk(options.rechunk)
        .with_chunk_size(chunk_size)
        .with_encoding(encoding)
        .with_columns(options.columns)
        .with_n_threads(n_threads)
        .low_memory(options.low_memory)
        .with_comment_char(comment_char)
        // .with_null_values(null_values)
        .with_parse_dates(options.parse_dates)
        .with_quote_char(quote_char)
        .with_row_count(row_count)
        .finish()
        .map_err(PyPolarsErr::from)?
    }
  };
  Ok(df.into())
}

#[napi]
pub fn from_rows(
  rows: Array,
  schema: Option<Wrap<Schema>>,
  infer_schema_length: Option<u32>,
) -> napi::Result<PyDataFrame> {
  let schema = match schema {
    Some(s) => s.0,
    None => {
      let infer_schema_length = infer_schema_length.unwrap_or(100) as usize;
      let pairs = obj_to_pairs(&rows, infer_schema_length);
      infer_schema(pairs, infer_schema_length)
    }
  };
  let len = rows.len();
  let it: Vec<Row> = (0..len)
    .into_iter()
    .map(|idx| {
      let obj = rows.get::<Object>(idx as u32).unwrap().unwrap();
      Row(
        schema
          .iter_fields()
          .map(|fld| {
            let dtype = fld.data_type().clone();
            let key = fld.name();
            let av: Wrap<AnyValue> = obj.get(key).unwrap().unwrap_or(Wrap(AnyValue::Null));
            let av = av.0;
            // todo -- add coerce
            let other_dtype: DataType = (&av).into();
            if other_dtype == dtype {
              av
            } else {
              AnyValue::Null
            }
          })
          .collect(),
      )
    })
    .collect();
  let df = DataFrame::from_rows_and_schema(&it, &schema).map_err(JsPolarsErr::from)?;
  Ok(df.into())
}

#[napi]
impl PyDataFrame {
  // #[napi(constructor)]
  // pub fn from_columns(columns: Array) -> napi::Result<PyDataFrame> {
  //   let len = columns.len();
  //   let cols: Vec<Series> = (0..len).map(|idx| {
  //     let item: External<Series> = columns.get(idx).unwrap().unwrap();
  //     item.clone()
  //   }).collect();
  //   todo!()

  //   // let cols: Vec<Series> = columns.iter().map(|c| c.series).collect();

  //   // let df = DataFrame::new(cols).map_err(PyPolarsErr::from)?;
  //   // Ok(PyDataFrame::new(df))
  // }
  #[napi(constructor)]
  pub fn from_columns(columns: Array) -> napi::Result<PyDataFrame> {
    let len = columns.len();
    let cols: Vec<Series> = (0..len)
      .map(|idx| {
        let item: &PySeries = columns.get(idx).unwrap().unwrap();
        item.series.clone()
      })
      .collect();

    let df = DataFrame::new(cols).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame::new(df))
  }

  #[napi]
  pub fn estimated_size(&self) -> u32 {
    self.df.estimated_size() as u32
  }

  #[napi]
  pub fn to_string(&self) -> String {
    format!("{:?}", self.df)
  }

  #[napi]
  pub fn add(&self, s: &PySeries) -> napi::Result<PyDataFrame> {
    let df = (&self.df + &s.series).map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn sub(&self, s: &PySeries) -> napi::Result<PyDataFrame> {
    let df = (&self.df - &s.series).map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn div(&self, s: &PySeries) -> napi::Result<PyDataFrame> {
    let df = (&self.df / &s.series).map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn mul(&self, s: &PySeries) -> napi::Result<PyDataFrame> {
    let df = (&self.df * &s.series).map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn rem(&self, s: &PySeries) -> napi::Result<PyDataFrame> {
    let df = (&self.df % &s.series).map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn add_df(&self, s: &PyDataFrame) -> napi::Result<PyDataFrame> {
    let df = (&self.df + &s.df).map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn sub_df(&self, s: &PyDataFrame) -> napi::Result<PyDataFrame> {
    let df = (&self.df - &s.df).map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn div_df(&self, s: &PyDataFrame) -> napi::Result<PyDataFrame> {
    let df = (&self.df / &s.df).map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn mul_df(&self, s: &PyDataFrame) -> napi::Result<PyDataFrame> {
    let df = (&self.df * &s.df).map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn rem_df(&self, s: &PyDataFrame) -> napi::Result<PyDataFrame> {
    let df = (&self.df % &s.df).map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn rechunk(&mut self) -> PyDataFrame {
    self.df.agg_chunks().into()
  }
  #[napi]
  pub fn fill_null(&self, strategy: Wrap<FillNullStrategy>) -> napi::Result<PyDataFrame> {
    let df = self.df.fill_null(strategy.0).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame::new(df))
  }
  #[napi]
  pub fn join(
    &self,
    other: &PyDataFrame,
    left_on: Vec<&str>,
    right_on: Vec<&str>,
    how: String,
    suffix: String,
) -> napi::Result<PyDataFrame> {
    let how = match how.as_ref() {
        "left" => JoinType::Left,
        "inner" => JoinType::Inner,
        "outer" => JoinType::Outer,
        "semi" => JoinType::Semi,
        "anti" => JoinType::Anti,
        "asof" => JoinType::AsOf(AsOfOptions {
            strategy: AsofStrategy::Backward,
            left_by: None,
            right_by: None,
            tolerance: None,
            tolerance_str: None,
        }),
        "cross" => JoinType::Cross,
        _ => panic!("not supported"),
    };

    let df = self
        .df
        .join(&other.df, left_on, right_on, how, Some(suffix))
        .map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame::new(df))
}
  #[napi]
  pub fn get_columns(&self) -> Vec<PySeries> {
    let cols = self.df.get_columns().clone();
    to_pyseries_collection(cols)
  }

  /// Get column names
  #[napi(getter)]
  pub fn columns(&self) -> Vec<&str> {
    self.df.get_column_names()
  }

  #[napi(setter, js_name = "columns")]
  pub fn set_columns(&mut self, names: Vec<&str>) -> napi::Result<()> {
    self
      .df
      .set_column_names(&names)
      .map_err(PyPolarsErr::from)?;
    Ok(())
  }

  #[napi]
  pub fn with_column(&mut self, s: &PySeries) -> napi::Result<PyDataFrame> {
    let mut df = self.df.clone();
    df.with_column(s.series.clone())
      .map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  /// Get datatypes
  #[napi]
  pub fn dtypes(&self) -> Vec<JsDataType> {
    self.df.iter().map(|s| s.dtype().into()).collect()
  }
  #[napi]
  pub fn n_chunks(&self) -> napi::Result<u32> {
    let n = self.df.n_chunks().map_err(PyPolarsErr::from)?;
    Ok(n as u32)
  }

  #[napi(getter)]
  pub fn shape(&self) -> Shape {
    self.df.shape().into()
  }
  #[napi(getter)]
  pub fn height(&self) -> i64 {
    self.df.height() as i64
  }
  #[napi(getter)]
  pub fn width(&self) -> i64 {
    self.df.width() as i64
  }
  #[napi]
  pub fn hstack_mut(&mut self, columns: Array) -> napi::Result<()> {
    let columns = to_series_collection(columns);
    self.df.hstack_mut(&columns).map_err(PyPolarsErr::from)?;
    Ok(())
  }
  #[napi]
  pub fn hstack(&self, columns: Array) -> napi::Result<PyDataFrame> {
    let columns = to_series_collection(columns);
    let df = self.df.hstack(&columns).map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }
  #[napi]
  pub fn extend(&mut self, df: &PyDataFrame) -> napi::Result<()> {
    self.df.extend(&df.df).map_err(PyPolarsErr::from)?;
    Ok(())
  }
  #[napi]
  pub fn vstack_mut(&mut self, df: &PyDataFrame) -> napi::Result<()> {
    self.df.vstack_mut(&df.df).map_err(PyPolarsErr::from)?;
    Ok(())
  }
  #[napi]
  pub fn vstack(&mut self, df: &PyDataFrame) -> napi::Result<PyDataFrame> {
    let df = self.df.vstack(&df.df).map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }
  #[napi]
  pub fn drop_in_place(&mut self, name: String) -> napi::Result<PySeries> {
    let s = self.df.drop_in_place(&name).map_err(PyPolarsErr::from)?;
    Ok(PySeries { series: s })
  }
  #[napi]
  pub fn drop_nulls(&self, subset: Option<Vec<String>>) -> napi::Result<PyDataFrame> {
    let df = self
      .df
      .drop_nulls(subset.as_ref().map(|s| s.as_ref()))
      .map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn drop(&self, name: String) -> napi::Result<PyDataFrame> {
    let df = self.df.drop(&name).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame::new(df))
  }
  #[napi]
  pub fn select_at_idx(&self, idx: i64) -> Option<PySeries> {
    self
      .df
      .select_at_idx(idx as usize)
      .map(|s| PySeries::new(s.clone()))
  }

  #[napi]
  pub fn find_idx_by_name(&self, name: String) -> Option<i64> {
    self.df.find_idx_by_name(&name).map(|i| i as i64)
  }
  #[napi]
  pub fn column(&self, name: String) -> napi::Result<PySeries> {
    let series = self
      .df
      .column(&name)
      .map(|s| PySeries::new(s.clone()))
      .map_err(PyPolarsErr::from)?;
    Ok(series)
  }
  #[napi]
  pub fn select(&self, selection: Vec<&str>) -> napi::Result<PyDataFrame> {
    let df = self.df.select(&selection).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame::new(df))
  }
  #[napi]
  pub fn filter(&self, mask: &PySeries) -> napi::Result<PyDataFrame> {
    let filter_series = &mask.series;
    if let Ok(ca) = filter_series.bool() {
      let df = self.df.filter(ca).map_err(PyPolarsErr::from)?;
      Ok(PyDataFrame::new(df))
    } else {
      Err(napi::Error::from_reason(
        "Expected a boolean mask".to_owned(),
      ))
    }
  }
  #[napi]
  pub fn take(&self, indices: Vec<u32>) -> napi::Result<PyDataFrame> {
    let indices = UInt32Chunked::from_vec("", indices);
    let df = self.df.take(&indices).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame::new(df))
  }
  #[napi]
  pub fn take_with_series(&self, indices: &PySeries) -> napi::Result<PyDataFrame> {
    let idx = indices.series.u32().map_err(PyPolarsErr::from)?;
    let df = self.df.take(idx).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame::new(df))
  }
  #[napi]
  pub fn sort(
    &self,
    by_column: String,
    reverse: bool,
    nulls_last: bool,
  ) -> napi::Result<PyDataFrame> {
    let df = self
      .df
      .sort_with_options(
        &by_column,
        SortOptions {
          descending: reverse,
          nulls_last,
        },
      )
      .map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame::new(df))
  }
  #[napi]
  pub fn sort_in_place(&mut self, by_column: String, reverse: bool) -> napi::Result<()> {
    self
      .df
      .sort_in_place([&by_column], reverse)
      .map_err(PyPolarsErr::from)?;
    Ok(())
  }
  #[napi]
  pub fn replace(&mut self, column: String, new_col: &PySeries) -> napi::Result<()> {
    self
      .df
      .replace(&column, new_col.series.clone())
      .map_err(PyPolarsErr::from)?;
    Ok(())
  }

  #[napi]
  pub fn rename(&mut self, column: String, new_col: String) -> napi::Result<()> {
    self
      .df
      .rename(&column, &new_col)
      .map_err(PyPolarsErr::from)?;
    Ok(())
  }

  #[napi]
  pub fn replace_at_idx(&mut self, index: i64, new_col: &PySeries) -> napi::Result<()> {
    self
      .df
      .replace_at_idx(index as usize, new_col.series.clone())
      .map_err(PyPolarsErr::from)?;
    Ok(())
  }

  #[napi]
  pub fn insert_at_idx(&mut self, index: i64, new_col: &PySeries) -> napi::Result<()> {
    self
      .df
      .insert_at_idx(index as usize, new_col.series.clone())
      .map_err(PyPolarsErr::from)?;
    Ok(())
  }

  #[napi]
  pub fn slice(&self, offset: i64, length: i64) -> PyDataFrame {
    let df = self.df.slice(offset as i64, length as usize);
    df.into()
  }

  #[napi]
  pub fn head(&self, length: Option<i64>) -> PyDataFrame {
    let length = length.map(|l| l as usize);
    let df = self.df.head(length);
    PyDataFrame::new(df)
  }
  #[napi]
  pub fn tail(&self, length: Option<i64>) -> PyDataFrame {
    let length = length.map(|l| l as usize);
    let df = self.df.tail(length);
    PyDataFrame::new(df)
  }
  #[napi]
  pub fn is_unique(&self) -> napi::Result<PySeries> {
    let mask = self.df.is_unique().map_err(PyPolarsErr::from)?;
    Ok(mask.into_series().into())
  }
  #[napi]
  pub fn is_duplicated(&self) -> napi::Result<PySeries> {
    let mask = self.df.is_duplicated().map_err(PyPolarsErr::from)?;
    Ok(mask.into_series().into())
  }
  #[napi]
  pub fn frame_equal(&self, other: &PyDataFrame, null_equal: bool) -> bool {
    if null_equal {
      self.df.frame_equal_missing(&other.df)
    } else {
      self.df.frame_equal(&other.df)
    }
  }
  #[napi]
  pub fn with_row_count(&self, name: String, offset: Option<u32>) -> napi::Result<PyDataFrame> {
    let df = self
      .df
      .with_row_count(&name, offset)
      .map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  pub fn groupby(
    &self,
    by: Vec<&str>,
    select: Option<Vec<String>>,
    agg: String,
  ) -> napi::Result<PyDataFrame> {
    let gb = self.df.groupby(&by).map_err(PyPolarsErr::from)?;
    let selection = match select.as_ref() {
      Some(s) => gb.select(s),
      None => gb,
    };
    finish_groupby(selection, &agg)
  }
  pub fn pivot(
    &self,
    by: Vec<String>,
    pivot_column: Vec<String>,
    values_column: Vec<String>,
    agg: String,
  ) -> napi::Result<PyDataFrame> {
    let mut gb = self.df.groupby(&by).map_err(PyPolarsErr::from)?;
    let pivot = gb.pivot(pivot_column, values_column);
    let df = match agg.as_ref() {
      "first" => pivot.first(),
      "min" => pivot.min(),
      "max" => pivot.max(),
      "mean" => pivot.mean(),
      "median" => pivot.median(),
      "sum" => pivot.sum(),
      "count" => pivot.count(),
      "last" => pivot.last(),
      a => Err(PolarsError::ComputeError(
        format!("agg fn {} does not exists", a).into(),
      )),
    };
    let df = df.map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame::new(df))
  }
  #[napi]
  pub fn clone(&self) -> PyDataFrame {
    PyDataFrame::new(self.df.clone())
  }
  #[napi]
  pub fn melt(
    &self,
    id_vars: Vec<String>,
    value_vars: Vec<String>,
    value_name: Option<String>,
    variable_name: Option<String>,
  ) -> napi::Result<PyDataFrame> {
    let args = MeltArgs {
      id_vars,
      value_vars,
      value_name,
      variable_name,
    };

    let df = self.df.melt2(args).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame::new(df))
  }

  #[napi]
  pub fn partition_by(&self, groups: Vec<String>, stable: bool) -> napi::Result<Vec<PyDataFrame>> {
    let out = if stable {
      self.df.partition_by_stable(groups)
    } else {
      self.df.partition_by(groups)
    }
    .map_err(PyPolarsErr::from)?;
    // Safety:
    // Repr mem layout
    Ok(unsafe { std::mem::transmute::<Vec<DataFrame>, Vec<PyDataFrame>>(out) })
  }

  #[napi]
  pub fn shift(&self, periods: i64) -> PyDataFrame {
    self.df.shift(periods).into()
  }
  #[napi]
  pub fn unique(
    &self,
    maintain_order: bool,
    subset: Option<Vec<String>>,
    keep: Wrap<UniqueKeepStrategy>,
  ) -> napi::Result<PyDataFrame> {
    let subset = subset.as_ref().map(|v| v.as_ref());
    let df = match maintain_order {
      true => self.df.unique_stable(subset, keep.0),
      false => self.df.unique(subset, keep.0),
    }
    .map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn lazy(&self) -> Object {
    todo!()
    // self.df.clone().lazy().into()
  }

  #[napi]
  pub fn max(&self) -> PyDataFrame {
    self.df.max().into()
  }
  #[napi]
  pub fn min(&self) -> PyDataFrame {
    self.df.min().into()
  }
  #[napi]
  pub fn sum(&self) -> PyDataFrame {
    self.df.sum().into()
  }
  #[napi]
  pub fn mean(&self) -> PyDataFrame {
    self.df.mean().into()
  }
  #[napi]
  pub fn std(&self) -> PyDataFrame {
    self.df.std().into()
  }
  #[napi]
  pub fn var(&self) -> PyDataFrame {
    self.df.var().into()
  }
  #[napi]
  pub fn median(&self) -> PyDataFrame {
    self.df.median().into()
  }

  #[napi]
  pub fn hmean(&self, null_strategy: Wrap<NullStrategy>) -> napi::Result<Option<PySeries>> {
    let s = self.df.hmean(null_strategy.0).map_err(PyPolarsErr::from)?;
    Ok(s.map(|s| s.into()))
  }
  #[napi]
  pub fn hmax(&self) -> napi::Result<Option<PySeries>> {
    let s = self.df.hmax().map_err(PyPolarsErr::from)?;
    Ok(s.map(|s| s.into()))
  }

  #[napi]
  pub fn hmin(&self) -> napi::Result<Option<PySeries>> {
    let s = self.df.hmin().map_err(PyPolarsErr::from)?;
    Ok(s.map(|s| s.into()))
  }

  #[napi]
  pub fn hsum(&self, null_strategy: Wrap<NullStrategy>) -> napi::Result<Option<PySeries>> {
    let s = self.df.hsum(null_strategy.0).map_err(PyPolarsErr::from)?;
    Ok(s.map(|s| s.into()))
  }
  #[napi]
  pub fn quantile(
    &self,
    quantile: f64,
    interpolation: Wrap<QuantileInterpolOptions>,
  ) -> napi::Result<PyDataFrame> {
    let df = self
      .df
      .quantile(quantile, interpolation.0)
      .map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn to_dummies(&self) -> napi::Result<PyDataFrame> {
    let df = self.df.to_dummies().map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }

  #[napi]
  pub fn null_count(&self) -> PyDataFrame {
    let df = self.df.null_count();
    df.into()
  }
  #[napi]
  pub fn shrink_to_fit(&mut self) {
    self.df.shrink_to_fit();
  }
  #[napi]
  pub fn hash_rows(
    &self,
    k0: Wrap<u64>,
    k1: Wrap<u64>,
    k2: Wrap<u64>,
    k3: Wrap<u64>,
  ) -> napi::Result<PySeries> {
    let hb = ahash::RandomState::with_seeds(k0.0, k1.0, k2.0, k3.0);
    let hash = self.df.hash_rows(Some(hb)).map_err(PyPolarsErr::from)?;
    Ok(hash.into_series().into())
  }

  #[napi]
  pub fn transpose(&self, include_header: bool, names: String) -> napi::Result<PyDataFrame> {
    let mut df = self.df.transpose().map_err(PyPolarsErr::from)?;
    if include_header {
      let s = Utf8Chunked::from_iter_values(&names, self.df.get_columns().iter().map(|s| s.name()))
        .into_series();
      df.insert_at_idx(0, s).unwrap();
    }
    Ok(df.into())
  }
  #[napi]
  pub fn upsample(
    &self,
    by: Vec<String>,
    index_column: String,
    every: String,
    offset: String,
    stable: bool,
  ) -> napi::Result<PyDataFrame> {
    let out = if stable {
      self.df.upsample_stable(
        by,
        &index_column,
        Duration::parse(&every),
        Duration::parse(&offset),
      )
    } else {
      self.df.upsample(
        by,
        &index_column,
        Duration::parse(&every),
        Duration::parse(&offset),
      )
    };
    let out = out.map_err(PyPolarsErr::from)?;
    Ok(out.into())
  }
  #[napi]
  pub fn to_struct(&self, name: String) -> PySeries {
    let s = self.df.clone().into_struct(&name);
    s.into_series().into()
  }
  #[napi]
  pub fn unnest(&self, names: Vec<String>) -> napi::Result<PyDataFrame> {
    let df = self.df.unnest(names).map_err(PyPolarsErr::from)?;
    Ok(df.into())
  }
  #[napi(factory)]
  pub fn from_bincode(buf: Buffer) -> napi::Result<PyDataFrame> {
    let df: DataFrame = bincode::deserialize(&buf).unwrap();

    Ok(df.into())
  }
  #[napi]
  pub fn to_bincode(&self) -> napi::Result<Buffer> {
    let buf = bincode::serialize(&self.df).unwrap();
    Ok(Buffer::from(buf))
  }
  #[napi]
  pub fn to_json(&self, pretty: Option<bool>) -> napi::Result<Buffer> {
    let pretty = pretty.unwrap_or(false);
    if pretty {
      let bytes = serde_json::to_vec_pretty(&self.df)?;
      Ok(bytes.into())
    }
    else {
      let bytes = serde_json::to_vec(&self.df)?;
      Ok(bytes.into())
    }
  }
  #[napi]
  pub fn to_row(&self, idx: f64, env: Env) -> napi::Result<Array> {
    let idx = idx as i64;
    
    let idx = if idx < 0 {
      (self.df.height() as i64 + idx) as usize
    } else {
      idx as usize
    };

    let width = self.df.width();
    let mut row = env.create_array(width as u32)?;

    for (i, col) in self.df.get_columns().iter().enumerate() {
      let val = col.get(idx);
      row.set(i as u32, Wrap(val))?;
    }
    Ok(row)
  }

  #[napi]
  pub fn to_rows(&self, env: Env) -> napi::Result<Array> {
    let (height, width) = self.df.shape();

    let mut rows = env.create_array(height as u32)?;
    for idx in 0..height {
      let mut row = env.create_array(width as u32)?;
      for (i, col) in self.df.get_columns().iter().enumerate() {
        let val = col.get(idx);
        row.set(i as u32, Wrap(val))?;
      }
      rows.set(idx as u32, row)?;
    }
    Ok(rows)
  }
  #[napi]
  pub fn to_rows_cb(&self, callback: napi::JsFunction, env: Env) -> napi::Result<()> {
    use napi::threadsafe_function::*;
    use polars_core::utils::rayon::prelude::*;
    let (height, width) = self.df.shape();
    let tsfn: ThreadsafeFunction<
      Either<Vec<JsAnyValue>, napi::JsNull>,
      ErrorStrategy::CalleeHandled,
    > = callback.create_threadsafe_function(
      0,
      |ctx: ThreadSafeCallContext<Either<Vec<JsAnyValue>, napi::JsNull>>| Ok(vec![ctx.value]),
    )?;

    polars_core::POOL.install(|| {
      (0..height).into_par_iter().for_each(|idx| {
        let tsfn = tsfn.clone();
        let values = self
          .df
          .get_columns()
          .iter()
          .map(|s| {
            let av: JsAnyValue = s.get(idx).into();
            av
          })
          .collect::<Vec<_>>();

        tsfn.call(
          Ok(Either::A(values)),
          ThreadsafeFunctionCallMode::NonBlocking,
        );
      });
    });
    tsfn.call(
      Ok(Either::B(env.get_null().unwrap())),
      ThreadsafeFunctionCallMode::NonBlocking,
    );

    Ok(())
  }
  #[napi]
  pub fn to_row_obj(&self, idx: Either<i64, f64>, env: Env) -> napi::Result<Object> {
    let idx = match idx {
      Either::A(a) => a,
      Either::B(b) => b as i64,
    };

    let idx = if idx < 0 {
      (self.df.height() as i64 + idx) as usize
    } else {
      idx as usize
    };

    let mut row = env.create_object()?;

    for col in self.df.get_columns() {
      let key = col.name();
      let val = col.get(idx);
      row.set(key, Wrap(val))?;
    }
    Ok(row)
  }
  #[napi]
  pub fn to_objects(&self, env: Env) -> napi::Result<Array> {
    let (height, width) = self.df.shape();

    let mut rows = env.create_array(height as u32)?;
    for idx in 0..height {
      let mut row = env.create_object()?;
      for col in self.df.get_columns() {
        let key = col.name();
        let val = col.get(idx);
        row.set(key, Wrap(val))?;
      }
      rows.set(idx as u32, row)?;
    }
    Ok(rows)
  }
  #[napi]
  pub fn to_objects_cb(&self, callback: napi::JsFunction, env: Env) -> napi::Result<()> {
    use napi::threadsafe_function::*;
    use polars_core::utils::rayon::prelude::*;
    use std::collections::HashMap;
    let (height, width) = self.df.shape();
    let tsfn: ThreadsafeFunction<
      Either<HashMap<String, JsAnyValue>, napi::JsNull>,
      ErrorStrategy::CalleeHandled,
    > = callback.create_threadsafe_function(
      0,
      |ctx: ThreadSafeCallContext<Either<HashMap<String, JsAnyValue>, napi::JsNull>>| {
        Ok(vec![ctx.value])
      },
    )?;

    polars_core::POOL.install(|| {
      (0..height).into_par_iter().for_each(|idx| {
        let tsfn = tsfn.clone();
        let values = self
          .df
          .get_columns()
          .iter()
          .map(|s| {
            let key = s.name().to_owned();
            let av: JsAnyValue = s.get(idx).into();
            (key, av)
          })
          .collect::<HashMap<_, _>>();

        tsfn.call(
          Ok(Either::A(values)),
          ThreadsafeFunctionCallMode::NonBlocking,
        );
      });
    });
    tsfn.call(
      Ok(Either::B(env.get_null().unwrap())),
      ThreadsafeFunctionCallMode::NonBlocking,
    );

    Ok(())
  }

  #[napi]
  pub fn write_csv(
    &mut self,
    path_or_buffer: Either3<String, Buffer, JsFunction>,
    options: WriteCsvOptions,
  ) -> napi::Result<()> {
    let has_header = options.has_header.unwrap_or(false);
    let sep = options.sep.unwrap_or(",".to_owned());
    let sep = sep.as_bytes()[0];
    let quote = options.quote.unwrap_or(",".to_owned());
    let quote = quote.as_bytes()[0];

    match path_or_buffer {
      Either3::A(path) => {
        let f = std::fs::File::create(path).unwrap();
        let f = BufWriter::new(f);
        CsvWriter::new(f)
          .has_header(has_header)
          .with_delimiter(sep)
          .with_quoting_char(quote)
          .finish(&mut self.df)
          .map_err(PyPolarsErr::from)?;
      }
      Either3::B(mut buf) => {
        let mut b = buf.as_mut();
        CsvWriter::new(&mut b)
          .has_header(has_header)
          .with_delimiter(sep)
          .with_quoting_char(quote)
          .finish(&mut self.df)
          .map_err(PyPolarsErr::from)?;
      }
      Either3::C(func) => {
        let tsfn: ThreadsafeFunction<Buffer, ErrorStrategy::CalleeHandled> =
          func.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;
        let tswriteable = ThreadsafeWriteable { inner: tsfn };

        CsvWriter::new(tswriteable)
          .has_header(has_header)
          .with_delimiter(sep)
          .with_quoting_char(quote)
          .finish(&mut self.df)
          .map_err(PyPolarsErr::from)?;
      }
    };
    Ok(())
  }
  #[napi]
  pub fn write_parquet(
    &mut self,
    path_or_buffer: Either3<String, Buffer, JsFunction>,
    compression: Wrap<ParquetCompression>,
  ) -> napi::Result<()> {
    let compression = compression.0;

    match path_or_buffer {
      Either3::A(path) => {
        let f = std::fs::File::create(path).unwrap();
        let f = BufWriter::new(f);
        ParquetWriter::new(f)
          .with_compression(compression)
          .finish(&mut self.df)
          .map_err(PyPolarsErr::from)?;
      }
      Either3::B(mut buf) => {
        let mut b = buf.as_mut();

        ParquetWriter::new(&mut b)
          .with_compression(compression)
          .finish(&mut self.df)
          .map_err(PyPolarsErr::from)?;
      }
      Either3::C(func) => {
        let tsfn: ThreadsafeFunction<Buffer, ErrorStrategy::CalleeHandled> =
          func.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;
        let tswriteable = ThreadsafeWriteable { inner: tsfn };

        ParquetWriter::new(tswriteable)
          .with_compression(compression)
          .finish(&mut self.df)
          .map_err(PyPolarsErr::from)?;
      }
    };
    Ok(())
  }
  #[napi]
  pub fn write_ipc(
    &mut self,
    path_or_buffer: Either3<String, Buffer, JsFunction>,
    compression: Wrap<Option<IpcCompression>>,
  ) -> napi::Result<()> {
    let compression = compression.0;

    match path_or_buffer {
      Either3::A(path) => {
        let f = std::fs::File::create(path).unwrap();
        let f = BufWriter::new(f);
        IpcWriter::new(f)
          .with_compression(compression)
          .finish(&mut self.df)
          .map_err(PyPolarsErr::from)?;
      }
      Either3::B(mut buf) => {
        let mut b = buf.as_mut();

        IpcWriter::new(&mut b)
          .with_compression(compression)
          .finish(&mut self.df)
          .map_err(PyPolarsErr::from)?;
      }
      Either3::C(func) => {
        let tsfn: ThreadsafeFunction<Buffer, ErrorStrategy::CalleeHandled> =
          func.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;
        let tswriteable = ThreadsafeWriteable { inner: tsfn };

        IpcWriter::new(tswriteable)
          .with_compression(compression)
          .finish(&mut self.df)
          .map_err(PyPolarsErr::from)?;
      }
    };
    Ok(())
  }
  #[napi]
  pub fn write_json(
    &mut self,
    path_or_buffer: Either3<String, Buffer, JsFunction>,
    json_format: String,
  ) -> napi::Result<()> {
    let json_format = match json_format.as_ref() {
      "json" => JsonFormat::Json,
      "lines" => JsonFormat::JsonLines,
      _ => {
        return Err(napi::Error::from_reason(
          "format must be 'json' or `lines'".to_owned(),
        ))
      }
    };

    match path_or_buffer {
      Either3::A(path) => {
        let f = std::fs::File::create(path).unwrap();
        let f = BufWriter::new(f);
        JsonWriter::new(f)
          .with_json_format(json_format)
          .finish(&mut self.df)
          .map_err(PyPolarsErr::from)?;
      }
      Either3::B(mut buf) => {
        let mut b = buf.as_mut();

        JsonWriter::new(&mut b)
          .with_json_format(json_format)
          .finish(&mut self.df)
          .map_err(PyPolarsErr::from)?;
      }
      Either3::C(func) => {
        let tsfn: ThreadsafeFunction<Buffer, ErrorStrategy::CalleeHandled> =
          func.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;
        let tswriteable = ThreadsafeWriteable { inner: tsfn };

        JsonWriter::new(tswriteable)
          .with_json_format(json_format)
          .finish(&mut self.df)
          .map_err(PyPolarsErr::from)?;
      }
    };
    Ok(())
  }
}

fn finish_groupby(gb: GroupBy, agg: &str) -> napi::Result<PyDataFrame> {
  let df = match agg {
    "min" => gb.min(),
    "max" => gb.max(),
    "mean" => gb.mean(),
    "first" => gb.first(),
    "last" => gb.last(),
    "sum" => gb.sum(),
    "count" => gb.count(),
    "n_unique" => gb.n_unique(),
    "median" => gb.median(),
    "agg_list" => gb.agg_list(),
    "groups" => gb.groups(),
    "std" => gb.std(),
    "var" => gb.var(),
    a => Err(PolarsError::ComputeError(
      format!("agg fn {} does not exists", a).into(),
    )),
  };

  let df = df.map_err(PyPolarsErr::from)?;
  Ok(PyDataFrame::new(df))
}

fn obj_to_pairs(rows: &Array, len: usize) -> impl '_ + Iterator<Item = Vec<(String, DataType)>> {
  let len = std::cmp::min(len, rows.len() as usize);

  (0..len).map(move |idx| {
    let obj = rows.get::<Object>(idx as u32).unwrap().unwrap();

    let keys = Object::keys(&obj).unwrap();
    keys
      .iter()
      .map(|key| {
        let value = obj.get::<_, napi::JsUnknown>(&key).unwrap().unwrap();
        let ty = value.get_type().unwrap();
        let dtype = match ty {
          ValueType::Boolean => DataType::Boolean,
          ValueType::Number => DataType::Float64,
          ValueType::String => DataType::Utf8,
          ValueType::Object => DataType::Struct(vec![]),
          ValueType::BigInt => DataType::UInt64,
          _ => DataType::Null,
        };
        (key.to_owned(), dtype)
      })
      .collect()
  })
}

// fn coerce_js_anyvalue<'a>(val: &'a AnyValue, dtype: DataType) -> JsResult<AnyValue<'a>> {
//   use DataType::*;
//   let vtype: DataType = val.into();

//   match (vtype, dtype) {
//     (Null, _) => Ok(AnyValue::Null),
//     (Utf8, Utf8) => val,
//     (_, Utf8) => {
//       let s = val.coerce_to_string()?.into_unknown();
//       AnyValue::from_js(s)
//     }
//     (ValueType::Boolean, Boolean) => bool::from_js(val).map(AnyValue::Boolean),
//     (_, Boolean) => val.coerce_to_bool().map(|b| {
//       let b: bool = b.try_into().unwrap();
//       AnyValue::Boolean(b)
//     }),
//     (ValueType::Bigint | ValueType::Number, UInt64) => u64::from_js(val).map(AnyValue::UInt64),
//     (_, UInt64) => val.coerce_to_number().map(|js_num| {
//       let n = js_num.get_int64().unwrap();
//       AnyValue::UInt64(n as u64)
//     }),
//     (ValueType::Bigint | ValueType::Number, Int64) => i64::from_js(val).map(AnyValue::Int64),
//     (_, Int64) => val.coerce_to_number().map(|js_num| {
//       let n = js_num.get_int64().unwrap();
//       AnyValue::Int64(n)
//     }),
//     (ValueType::Number, Float64) => f64::from_js(val).map(AnyValue::Float64),
//     (_, Float64) => val.coerce_to_number().map(|js_num| {
//       let n = js_num.get_double().unwrap();
//       AnyValue::Float64(n)
//     }),
//     (ValueType::Number, Float32) => f32::from_js(val).map(AnyValue::Float32),
//     (_, Float32) => val.coerce_to_number().map(|js_num| {
//       let n = js_num.get_double().unwrap();
//       AnyValue::Float32(n as f32)
//     }),
//     (ValueType::Number, Int32) => i32::from_js(val).map(AnyValue::Int32),
//     (_, Int32) => val.coerce_to_number().map(|js_num| {
//       let n = js_num.get_int32().unwrap();
//       AnyValue::Int32(n)
//     }),
//     (ValueType::Number, UInt32) => u32::from_js(val).map(AnyValue::UInt32),
//     (_, UInt32) => val.coerce_to_number().map(|js_num| {
//       let n = js_num.get_uint32().unwrap();
//       AnyValue::UInt32(n)
//     }),
//     (ValueType::Number, Int16) => i16::from_js(val).map(AnyValue::Int16),
//     (_, Int16) => val.coerce_to_number().map(|js_num| {
//       let n = js_num.get_int32().unwrap();
//       AnyValue::Int16(n as i16)
//     }),
//     (ValueType::Number, UInt16) => u16::from_js(val).map(AnyValue::UInt16),
//     (_, UInt16) => val.coerce_to_number().map(|js_num| {
//       let n = js_num.get_uint32().unwrap();
//       AnyValue::UInt16(n as u16)
//     }),
//     (ValueType::Number, Int8) => i8::from_js(val).map(AnyValue::Int8),
//     (_, Int8) => val.coerce_to_number().map(|js_num| {
//       let n = js_num.get_int32().unwrap();
//       AnyValue::Int8(n as i8)
//     }),
//     (ValueType::Number, UInt8) => u8::from_js(val).map(AnyValue::UInt8),
//     (_, UInt8) => val.coerce_to_number().map(|js_num| {
//       let n = js_num.get_uint32().unwrap();
//       AnyValue::UInt8(n as u8)
//     }),
//     (ValueType::Number, Date) => i32::from_js(val).map(AnyValue::Date),
//     (_, Date) => val.coerce_to_number().map(|js_num| {
//       let n = js_num.get_int32().unwrap();
//       AnyValue::Date(n)
//     }),
//     (ValueType::Bigint | ValueType::Number, Datetime(_, _)) => {
//       i64::from_js(val).map(|d| AnyValue::Datetime(d, TimeUnit::Milliseconds, &None))
//     }
//     (ValueType::Object, DataType::Datetime(_, _)) => {
//       if val.is_date()? {
//         let d: napi::JsDate = unsafe { val.cast() };
//         let d = d.value_of()?;
//         Ok(AnyValue::Datetime(d as i64, TimeUnit::Milliseconds, &None))
//       } else {
//         Ok(AnyValue::Null)
//       }
//     }
//     _ => Ok(AnyValue::Null),
//   }
// }
