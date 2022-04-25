use crate::datatypes::JsDataType;
use polars::prelude::*;
use wasm_bindgen::convert::FromWasmAbi;
use wasm_bindgen::JsCast;

use super::{error::JsPolarsErr, series::*, JsResult};

use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name=DataFrame)]
pub struct JsDataFrame {
  df: DataFrame,
}

#[wasm_bindgen]
pub struct JsDataFramePromise {
  promise: js_sys::Promise,
}

impl JsDataFrame {
  pub(crate) fn new(df: DataFrame) -> Self {
    JsDataFrame { df }
  }
}
impl From<DataFrame> for JsDataFrame {
  fn from(df: DataFrame) -> Self {
    Self { df }
  }
}

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(typescript_type = "DataFrame")]
  pub type ExternDataFrame;

  #[wasm_bindgen(typescript_type = "any")]
  pub type ExternAnyValue;

  #[wasm_bindgen(method, getter = ptr)]
  fn ptr(this: &ExternDataFrame) -> f64;
  #[wasm_bindgen(typescript_type = "DataFrame[]")]
  pub type DataFrameArray;

}

#[wasm_bindgen(js_class=DataFrame)]
impl JsDataFrame {
  #[wasm_bindgen(constructor)]
  pub fn new0() -> Self {
    DataFrame::new_no_checks(vec![]).into()
  }

  pub fn read_columns(columns: js_sys::Iterator) -> JsResult<JsDataFrame> {
    let cols = to_series_collection(columns);

    let df = DataFrame::new(cols).map_err(JsPolarsErr::from)?;
    Ok(JsDataFrame::from(df))
  }
  pub fn add(&self, s: &JsSeries) -> JsResult<JsDataFrame> {
    let df = (&self.df + &s.series).map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }

  pub fn sub(&self, s: &JsSeries) -> JsResult<JsDataFrame> {
    let df = (&self.df - &s.series).map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }

  pub fn div(&self, s: &JsSeries) -> JsResult<JsDataFrame> {
    let df = (&self.df / &s.series).map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }

  pub fn mul(&self, s: &JsSeries) -> JsResult<JsDataFrame> {
    let df = (&self.df * &s.series).map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }

  pub fn rem(&self, s: &JsSeries) -> JsResult<JsDataFrame> {
    let df = (&self.df % &s.series).map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }

  pub fn add_df(&self, s: &JsDataFrame) -> JsResult<JsDataFrame> {
    let df = (&self.df + &s.df).map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }

  pub fn sub_df(&self, s: &JsDataFrame) -> JsResult<JsDataFrame> {
    let df = (&self.df - &s.df).map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }

  pub fn div_df(&self, s: &JsDataFrame) -> JsResult<JsDataFrame> {
    let df = (&self.df / &s.df).map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }

  pub fn mul_df(&self, s: &JsDataFrame) -> JsResult<JsDataFrame> {
    let df = (&self.df * &s.df).map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }

  pub fn rem_df(&self, s: &JsDataFrame) -> JsResult<JsDataFrame> {
    let df = (&self.df % &s.df).map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }
  pub fn rechunk(&mut self) -> Self {
    self.df.agg_chunks().into()
  }

  /// Format `DataFrame` as String
  pub fn as_str(&self) -> String {
    format!("{:?}", self.df)
  }
  pub fn fill_null(&self, strategy: &str) -> JsResult<JsDataFrame> {
    let strat = match strategy {
      "backward" => FillNullStrategy::Backward,
      "forward" => FillNullStrategy::Forward,
      "min" => FillNullStrategy::Min,
      "max" => FillNullStrategy::Max,
      "mean" => FillNullStrategy::Mean,
      "one" => FillNullStrategy::One,
      "zero" => FillNullStrategy::Zero,
      s => return Err(JsPolarsErr::Other(format!("Strategy {} not supported", s)).into()),
    };
    let df = self.df.fill_null(strat).map_err(JsPolarsErr::from)?;
    Ok(JsDataFrame::from(df))
  }
  pub fn join(
    &self,
    other: &JsDataFrame,
    left_on: js_sys::Array,
    right_on: js_sys::Array,
    how: &str,
    suffix: String,
  ) -> JsResult<JsDataFrame> {
    let how = match how {
      "left" => JoinType::Left,
      "inner" => JoinType::Inner,
      "outer" => JoinType::Outer,
      // "asof" => JoinType::AsOf(AsOfOptions {
      //   strategy: AsofStrategy::Backward,
      //   left_by: None,
      //   right_by: None,
      //   tolerance: None,
      //   tolerance_str: None,
      // }),
      "cross" => JoinType::Cross,
      _ => panic!("not supported"),
    };
    let left_on: Vec<String> = left_on.iter().map(|v| v.as_string().unwrap()).collect();
    let right_on: Vec<String> = right_on.iter().map(|v| v.as_string().unwrap()).collect();

    let df = self
      .df
      .join(&other.df, left_on, right_on, how, Some(suffix))
      .map_err(JsPolarsErr::from)?;
    Ok(JsDataFrame::new(df))
  }

  pub fn get_columns(&self) -> Vec<u32> {
    use wasm_bindgen::convert::IntoWasmAbi;
    self
      .df
      .get_columns()
      .clone()
      .into_iter()
      .map(|s| JsSeries::new(s).into_abi())
      .collect()
  }

  /// Get column names
  #[wasm_bindgen(getter)]
  pub fn columns(&self) -> js_sys::Array {
    self
      .df
      .iter()
      .map(|s| JsValue::from_str(s.name()))
      .collect()
  }
  /// set column names
  #[wasm_bindgen(setter, js_name=columns)]
  pub fn set_column_names(&mut self, names: js_sys::Array) -> JsResult<()> {
    let names: Vec<String> = names.iter().map(|v| v.as_string().unwrap()).collect();

    self
      .df
      .set_column_names(&names)
      .map_err(JsPolarsErr::from)?;
    Ok(())
  }
  pub fn with_column(&mut self, s: JsSeries) -> JsResult<JsDataFrame> {
    let mut df = self.df.clone();
    df.with_column(s.series).map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }
  pub fn dtypes(&self) -> js_sys::Array {
    self
      .df
      .dtypes()
      .iter()
      .map(|arrow_dtype| {
        let dt: JsDataType = arrow_dtype.into();
        let jsv: JsValue = dt.to_string().into();
        jsv
      })
      .collect()
  }
  pub fn n_chunks(&self) -> JsResult<usize> {
    let n = self.df.n_chunks().map_err(JsPolarsErr::from)?;
    Ok(n)
  }
  pub fn shape(&self) -> js_sys::Array {
    let (height, width) = self.df.shape();
    let height: JsValue = height.into();
    let width: JsValue = width.into();
    js_sys::Array::of2(&height, &width)
  }
  pub fn height(&self) -> usize {
    self.df.height()
  }

  pub fn width(&self) -> usize {
    self.df.width()
  }
  pub fn hstack_mut(&mut self, columns: js_sys::Iterator) -> JsResult<()> {
    let cols = to_series_collection(columns);

    self.df.hstack_mut(&cols).map_err(JsPolarsErr::from)?;
    Ok(())
  }

  pub fn hstack(&self, columns: js_sys::Iterator) -> JsResult<JsDataFrame> {
    let columns = to_series_collection(columns);
    let df = self.df.hstack(&columns).map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }
  pub fn extend(&mut self, df: &JsDataFrame) -> JsResult<()> {
    self.df.extend(&df.df).map_err(JsPolarsErr::from)?;
    Ok(())
  }
  pub fn vstack_mut(&mut self, df: &JsDataFrame) -> JsResult<()> {
    self.df.vstack_mut(&df.df).map_err(JsPolarsErr::from)?;
    Ok(())
  }

  pub fn vstack(&mut self, df: &JsDataFrame) -> JsResult<JsDataFrame> {
    let df = self.df.vstack(&df.df).map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }
  pub fn drop_in_place(&mut self, name: &str) -> JsResult<JsSeries> {
    let s = self.df.drop_in_place(name).map_err(JsPolarsErr::from)?;
    Ok(JsSeries { series: s })
  }
  pub fn drop_nulls(&self, _subset: Option<js_sys::Array>) -> JsResult<JsDataFrame> {
    todo!()
  }
  pub fn drop(&self, name: &str) -> JsResult<JsDataFrame> {
    let df = self.df.drop(name).map_err(JsPolarsErr::from)?;
    Ok(JsDataFrame::new(df))
  }
  pub fn select_at_idx(&self, idx: usize) -> Option<JsSeries> {
    self.df.select_at_idx(idx).map(|s| JsSeries::new(s.clone()))
  }
  pub fn find_idx_by_name(&self, name: &str) -> Option<usize> {
    self.df.find_idx_by_name(name)
  }
  pub fn column(&self, name: &str) -> JsResult<JsSeries> {
    let series = self
      .df
      .column(name)
      .map(|s| JsSeries::new(s.clone()))
      .map_err(JsPolarsErr::from)?;
    Ok(series)
  }
  pub fn select(&self, selection: js_sys::Array) -> JsResult<JsDataFrame> {
    let selection: Vec<String> = selection.iter().map(|v| v.as_string().unwrap()).collect();

    let df = self.df.select(&selection).map_err(JsPolarsErr::from)?;
    Ok(JsDataFrame::new(df))
  }

  pub fn filter(&self, mask: &JsSeries) -> JsResult<JsDataFrame> {
    let filter_series = &mask.series;
    if let Ok(ca) = filter_series.bool() {
      let df = self.df.filter(ca).map_err(JsPolarsErr::from)?;
      Ok(JsDataFrame::new(df))
    } else {
      Err(js_sys::TypeError::new("Expected a boolean mask").into())
    }
  }
  pub fn take(&self, indices: js_sys::Array) -> JsResult<JsDataFrame> {
    let indices: Vec<u32> = indices.iter().map(|v| v.as_f64().unwrap() as u32).collect();

    let indices = UInt32Chunked::from_vec("", indices);
    let df = self.df.take(&indices).map_err(JsPolarsErr::from)?;
    Ok(JsDataFrame::new(df))
  }

  pub fn take_with_series(&self, indices: &JsSeries) -> JsResult<JsDataFrame> {
    let idx = indices.series.u32().map_err(JsPolarsErr::from)?;
    let df = self.df.take(idx).map_err(JsPolarsErr::from)?;
    Ok(JsDataFrame::new(df))
  }
  pub fn sort(&self, by_column: &str, reverse: bool) -> JsResult<JsDataFrame> {
    let df = self
      .df
      .sort([by_column], reverse)
      .map_err(JsPolarsErr::from)?;
    Ok(JsDataFrame::new(df))
  }

  pub fn sort_in_place(&mut self, by_column: &str, reverse: bool) -> JsResult<()> {
    self
      .df
      .sort_in_place([by_column], reverse)
      .map_err(JsPolarsErr::from)?;
    Ok(())
  }

  pub fn replace(&mut self, column: &str, new_col: JsSeries) -> JsResult<()> {
    self
      .df
      .replace(column, new_col.series)
      .map_err(JsPolarsErr::from)?;
    Ok(())
  }

  pub fn rename(&mut self, column: &str, new_col: &str) -> JsResult<()> {
    self.df.rename(column, new_col).map_err(JsPolarsErr::from)?;
    Ok(())
  }
  pub fn replace_at_idx(&mut self, index: usize, new_col: JsSeries) -> JsResult<()> {
    self
      .df
      .replace_at_idx(index, new_col.series)
      .map_err(JsPolarsErr::from)?;
    Ok(())
  }

  pub fn insert_at_idx(&mut self, index: usize, new_col: JsSeries) -> JsResult<()> {
    self
      .df
      .insert_at_idx(index, new_col.series)
      .map_err(JsPolarsErr::from)?;
    Ok(())
  }
  pub fn slice(&self, offset: usize, length: usize) -> Self {
    let df = self.df.slice(offset as i64, length);
    df.into()
  }

  pub fn head(&self, length: Option<usize>) -> Self {
    let df = self.df.head(length);
    JsDataFrame::new(df)
  }

  pub fn tail(&self, length: Option<usize>) -> Self {
    let df = self.df.tail(length);
    JsDataFrame::new(df)
  }
  pub fn is_unique(&self) -> JsResult<JsSeries> {
    let mask = self.df.is_unique().map_err(JsPolarsErr::from)?;
    Ok(mask.into_series().into())
  }
  pub fn is_duplicated(&self) -> JsResult<JsSeries> {
    let mask = self.df.is_duplicated().map_err(JsPolarsErr::from)?;
    Ok(mask.into_series().into())
  }

  pub fn frame_equal(&self, other: &JsDataFrame, null_equal: bool) -> bool {
    if null_equal {
      self.df.frame_equal_missing(&other.df)
    } else {
      self.df.frame_equal(&other.df)
    }
  }
  pub fn with_row_count(&self, name: &str, offset: Option<u32>) -> JsResult<JsDataFrame> {
    let df = self
      .df
      .with_row_count(name, offset)
      .map_err(JsPolarsErr::from)?;
    Ok(df.into())
  }
  pub fn clone(&self) -> Self {
    JsDataFrame::new(self.df.clone())
  }
  pub fn melt(&self, id_vars: js_sys::Array, value_vars: js_sys::Array) -> JsResult<JsDataFrame> {
    let df = self
      .df
      .melt(
        id_vars.iter().map(|v| v.as_string().unwrap()),
        value_vars.iter().map(|v| v.as_string().unwrap()),
      )
      .map_err(JsPolarsErr::from)?;
    Ok(JsDataFrame::new(df))
  }

  pub fn shift(&self, periods: f64) -> Self {
    self.df.shift(periods as i64).into()
  }
  pub fn unique(
    &self,
    maintain_order: bool,
    subset: Option<js_sys::Array>,
    keep: &str,
  ) -> JsResult<JsDataFrame> {
    // todo!()
    let keep = match keep {
      "first" => UniqueKeepStrategy::First,
      "last" => UniqueKeepStrategy::Last,
      s => panic!("keep strategy {} is not supported", s),
    };
    let subset: Option<Vec<String>> = subset.map(|v| {
      let s: Vec<String> = v.iter().map(|item| item.as_string().unwrap()).collect();
      s
    });

    let subset = subset.as_ref().map(|v| v.as_ref());

    crate::console_log!("df={:#?}", self.df);

    let df = match maintain_order {
        true => self.df.unique_stable(subset, keep),
        false => self.df.unique(subset, keep),
      }
      .map_err(JsPolarsErr::from)
      .unwrap();
    Ok(JsDataFrame::new(df))
  }
  pub fn lazy(&self) -> Self {
    todo!()
  }
  pub fn max(&self) -> Self {
    self.df.max().into()
  }

  pub fn min(&self) -> Self {
    self.df.min().into()
  }

  pub fn sum(&self) -> Self {
    self.df.sum().into()
  }

  pub fn mean(&self) -> Self {
    self.df.mean().into()
  }

  pub fn std(&self) -> Self {
    self.df.std().into()
  }

  pub fn var(&self) -> Self {
    self.df.var().into()
  }

  pub fn median(&self) -> Self {
    self.df.median().into()
  }
  pub fn null_count(&self) -> Self {
    let df = self.df.null_count();
    df.into()
  }
  pub fn hash_rows(&self, k0: u64, k1: u64, k2: u64, k3: u64) -> JsResult<JsSeries> {
    let hb = ahash::RandomState::with_seeds(k0, k1, k2, k3);
    let hash = self.df.hash_rows(Some(hb)).map_err(JsPolarsErr::from)?;
    Ok(hash.into_series().into())
  }
}
