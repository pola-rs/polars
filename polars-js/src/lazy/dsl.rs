use crate::prelude::*;
use crate::utils::reinterpret;
use polars::lazy::dsl;
use polars::lazy::dsl::Operator;
use polars_core::series::ops::NullBehavior;
use std::borrow::Cow;

#[napi]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyExpr {
  pub(crate) inner: dsl::Expr,
}

pub(crate) trait ToExprs {
  fn to_exprs(self) -> Vec<Expr>;
}
impl PyExpr {
  pub(crate) fn new(inner: dsl::Expr) -> PyExpr {
    PyExpr { inner }
  }
}
impl From<dsl::Expr> for PyExpr {
  fn from(s: dsl::Expr) -> PyExpr {
    PyExpr::new(s)
  }
}
impl ToExprs for Vec<PyExpr> {
  fn to_exprs(self) -> Vec<Expr> {
    // Safety
    // repr is transparent
    // and has only got one inner field`
    unsafe { std::mem::transmute(self) }
  }
}

impl ToExprs for Vec<&PyExpr> {
  fn to_exprs(self) -> Vec<Expr> {
    self
      .into_iter()
      .map(|e| e.inner.clone())
      .collect::<Vec<Expr>>()
  }
}

#[napi]
impl PyExpr {
  #[napi]
  pub fn __add__(&self, rhs: &PyExpr) -> napi::Result<PyExpr> {
    Ok(dsl::binary_expr(self.inner.clone(), Operator::Plus, rhs.inner.clone()).into())
  }

  #[napi]
  pub fn __sub__(&self, rhs: &PyExpr) -> napi::Result<PyExpr> {
    Ok(dsl::binary_expr(self.inner.clone(), Operator::Minus, rhs.inner.clone()).into())
  }

  #[napi]
  pub fn __mul__(&self, rhs: &PyExpr) -> napi::Result<PyExpr> {
    Ok(dsl::binary_expr(self.inner.clone(), Operator::Multiply, rhs.inner.clone()).into())
  }

  #[napi]
  pub fn __truediv__(&self, rhs: &PyExpr) -> napi::Result<PyExpr> {
    Ok(dsl::binary_expr(self.inner.clone(), Operator::TrueDivide, rhs.inner.clone()).into())
  }

  #[napi]
  pub fn __mod__(&self, rhs: &PyExpr) -> napi::Result<PyExpr> {
    Ok(dsl::binary_expr(self.inner.clone(), Operator::Modulus, rhs.inner.clone()).into())
  }

  #[napi]
  pub fn __floordiv__(&self, rhs: &PyExpr) -> napi::Result<PyExpr> {
    Ok(dsl::binary_expr(self.inner.clone(), Operator::Divide, rhs.inner.clone()).into())
  }

  #[napi]
  pub fn to_string(&self) -> String {
    format!("{:?}", self.inner)
  }

  #[napi]
  pub fn eq(&self, other: &PyExpr) -> PyExpr {
    self.clone().inner.eq(other.inner.clone()).into()
  }

  #[napi]
  pub fn neq(&self, other: &PyExpr) -> PyExpr {
    self.clone().inner.neq(other.inner.clone()).into()
  }

  #[napi]
  pub fn gt(&self, other: &PyExpr) -> PyExpr {
    self.clone().inner.gt(other.inner.clone()).into()
  }

  #[napi]
  pub fn gt_eq(&self, other: &PyExpr) -> PyExpr {
    self.clone().inner.gt_eq(other.inner.clone()).into()
  }

  #[napi]
  pub fn lt_eq(&self, other: &PyExpr) -> PyExpr {
    self.clone().inner.lt_eq(other.inner.clone()).into()
  }

  #[napi]
  pub fn lt(&self, other: &PyExpr) -> PyExpr {
    self.clone().inner.lt(other.inner.clone()).into()
  }

  #[napi]
  pub fn alias(&self, name: String) -> PyExpr {
    self.clone().inner.alias(&name).into()
  }

  #[napi]
  pub fn is_not(&self) -> PyExpr {
    self.clone().inner.not().into()
  }

  #[napi]
  pub fn is_null(&self) -> PyExpr {
    self.clone().inner.is_null().into()
  }

  #[napi]
  pub fn is_not_null(&self) -> PyExpr {
    self.clone().inner.is_not_null().into()
  }

  #[napi]
  pub fn is_infinite(&self) -> PyExpr {
    self.clone().inner.is_infinite().into()
  }

  #[napi]
  pub fn is_finite(&self) -> PyExpr {
    self.clone().inner.is_finite().into()
  }

  #[napi]
  pub fn is_nan(&self) -> PyExpr {
    self.clone().inner.is_nan().into()
  }

  #[napi]
  pub fn is_not_nan(&self) -> PyExpr {
    self.clone().inner.is_not_nan().into()
  }

  #[napi]
  pub fn min(&self) -> PyExpr {
    self.clone().inner.min().into()
  }

  #[napi]
  pub fn max(&self) -> PyExpr {
    self.clone().inner.max().into()
  }

  #[napi]
  pub fn mean(&self) -> PyExpr {
    self.clone().inner.mean().into()
  }

  #[napi]
  pub fn median(&self) -> PyExpr {
    self.clone().inner.median().into()
  }

  #[napi]
  pub fn sum(&self) -> PyExpr {
    self.clone().inner.sum().into()
  }

  #[napi]
  pub fn n_unique(&self) -> PyExpr {
    self.clone().inner.n_unique().into()
  }

  #[napi]
  pub fn arg_unique(&self) -> PyExpr {
    self.clone().inner.arg_unique().into()
  }

  #[napi]
  pub fn unique(&self) -> PyExpr {
    self.clone().inner.unique().into()
  }

  #[napi]
  pub fn unique_stable(&self) -> PyExpr {
    self.clone().inner.unique_stable().into()
  }

  #[napi]
  pub fn first(&self) -> PyExpr {
    self.clone().inner.first().into()
  }

  #[napi]
  pub fn last(&self) -> PyExpr {
    self.clone().inner.last().into()
  }

  #[napi]
  pub fn list(&self) -> PyExpr {
    self.clone().inner.list().into()
  }

  #[napi]
  pub fn quantile(&self, quantile: f64, interpolation: Wrap<QuantileInterpolOptions>) -> PyExpr {
    self
      .clone()
      .inner
      .quantile(quantile, interpolation.0)
      .into()
  }

  #[napi]
  pub fn agg_groups(&self) -> PyExpr {
    self.clone().inner.agg_groups().into()
  }

  #[napi]
  pub fn count(&self) -> PyExpr {
    self.clone().inner.count().into()
  }

  #[napi]
  pub fn value_counts(&self, multithreaded: bool) -> PyExpr {
    self.inner.clone().value_counts(multithreaded).into()
  }

  #[napi]
  pub fn unique_counts(&self) -> PyExpr {
    self.inner.clone().unique_counts().into()
  }

  #[napi]
  pub fn cast(&self, data_type: Wrap<DataType>, strict: bool) -> PyExpr {
    let dt = data_type.0;
    let expr = if strict {
      self.inner.clone().strict_cast(dt)
    } else {
      self.inner.clone().cast(dt)
    };
    expr.into()
  }

  #[napi]
  pub fn sort_with(&self, descending: bool, nulls_last: bool) -> PyExpr {
    self
      .clone()
      .inner
      .sort_with(SortOptions {
        descending,
        nulls_last,
      })
      .into()
  }

  #[napi]
  pub fn arg_sort(&self, reverse: bool) -> PyExpr {
    self.clone().inner.arg_sort(reverse).into()
  }
  #[napi]
  pub fn arg_max(&self) -> PyExpr {
    self.clone().inner.arg_max().into()
  }
  #[napi]
  pub fn arg_min(&self) -> PyExpr {
    self.clone().inner.arg_min().into()
  }
  #[napi]
  pub fn take(&self, idx: &PyExpr) -> PyExpr {
    self.clone().inner.take(idx.inner.clone()).into()
  }

  #[napi]
  pub fn sort_by(&self, by: Vec<&PyExpr>, reverse: Vec<bool>) -> PyExpr {
    let by = by.into_iter().map(|e| e.inner.clone()).collect::<Vec<_>>();
    self.clone().inner.sort_by(by, reverse).into()
  }
  #[napi]
  pub fn backward_fill(&self) -> PyExpr {
    self.clone().inner.backward_fill().into()
  }

  #[napi]
  pub fn forward_fill(&self) -> PyExpr {
    self.clone().inner.forward_fill().into()
  }

  #[napi]
  pub fn shift(&self, periods: i64) -> PyExpr {
    self.clone().inner.shift(periods).into()
  }

  #[napi]
  pub fn shift_and_fill(&self, periods: i64, fill_value: &PyExpr) -> PyExpr {
    self
      .clone()
      .inner
      .shift_and_fill(periods, fill_value.inner.clone())
      .into()
  }

  #[napi]
  pub fn fill_null(&self, expr: &PyExpr) -> PyExpr {
    self.clone().inner.fill_null(expr.inner.clone()).into()
  }

  #[napi]
  pub fn fill_null_with_strategy(&self, strategy: Wrap<FillNullStrategy>) -> PyExpr {
    self
      .inner
      .clone()
      .apply(move |s| s.fill_null(strategy.0), GetOutput::same_type())
      .with_fmt("fill_null")
      .into()
  }
  #[napi]
  pub fn fill_nan(&self, expr: &PyExpr) -> PyExpr {
    self.inner.clone().fill_nan(expr.inner.clone()).into()
  }

  #[napi]
  pub fn drop_nulls(&self) -> PyExpr {
    self.inner.clone().drop_nulls().into()
  }

  #[napi]
  pub fn drop_nans(&self) -> PyExpr {
    self.inner.clone().drop_nans().into()
  }

  #[napi]
  pub fn filter(&self, predicate: &PyExpr) -> PyExpr {
    self.clone().inner.filter(predicate.inner.clone()).into()
  }

  #[napi]
  pub fn reverse(&self) -> PyExpr {
    self.clone().inner.reverse().into()
  }

  #[napi]
  pub fn std(&self) -> PyExpr {
    self.clone().inner.std().into()
  }

  #[napi]
  pub fn var(&self) -> PyExpr {
    self.clone().inner.var().into()
  }
  #[napi]
  pub fn is_unique(&self) -> PyExpr {
    self.clone().inner.is_unique().into()
  }

  #[napi]
  pub fn is_first(&self) -> PyExpr {
    self.clone().inner.is_first().into()
  }

  #[napi]
  pub fn explode(&self) -> PyExpr {
    self.clone().inner.explode().into()
  }
  #[napi]
  pub fn take_every(&self, n: i64) -> PyExpr {
    self
      .clone()
      .inner
      .map(
        move |s: Series| Ok(s.take_every(n as usize)),
        GetOutput::same_type(),
      )
      .with_fmt("take_every")
      .into()
  }
  #[napi]
  pub fn tail(&self, n: Option<i64>) -> PyExpr {
    let n = n.map(|v| v as usize);
    self.clone().inner.tail(n).into()
  }

  #[napi]
  pub fn head(&self, n: Option<i64>) -> PyExpr {
    let n = n.map(|v| v as usize);
    self.clone().inner.head(n).into()
  }
  #[napi]
  pub fn slice(&self, offset: &PyExpr, length: &PyExpr) -> PyExpr {
    self
      .inner
      .clone()
      .slice(offset.inner.clone(), length.inner.clone())
      .into()
  }
  #[napi]
  pub fn round(&self, decimals: u32) -> PyExpr {
    self.clone().inner.round(decimals).into()
  }

  #[napi]
  pub fn floor(&self) -> PyExpr {
    self.clone().inner.floor().into()
  }

  #[napi]
  pub fn ceil(&self) -> PyExpr {
    self.clone().inner.ceil().into()
  }
  #[napi]
  pub fn clip(&self, min: f64, max: f64) -> PyExpr {
    self.clone().inner.clip(min, max).into()
  }

  #[napi]
  pub fn abs(&self) -> PyExpr {
    self.clone().inner.abs().into()
  }
  #[napi]
  pub fn is_duplicated(&self) -> PyExpr {
    self.clone().inner.is_duplicated().into()
  }
  #[napi]
  pub fn over(&self, partition_by: Vec<&PyExpr>) -> PyExpr {
    self.clone().inner.over(partition_by.to_exprs()).into()
  }
  #[napi]
  pub fn _and(&self, expr: &PyExpr) -> PyExpr {
    self.clone().inner.and(expr.inner.clone()).into()
  }

  #[napi]
  pub fn _xor(&self, expr: &PyExpr) -> PyExpr {
    self.clone().inner.xor(expr.inner.clone()).into()
  }

  #[napi]
  pub fn _or(&self, expr: &PyExpr) -> PyExpr {
    self.clone().inner.or(expr.inner.clone()).into()
  }
  #[napi]
  pub fn is_in(&self, expr: &PyExpr) -> PyExpr {
    self.clone().inner.is_in(expr.inner.clone()).into()
  }
  #[napi]
  pub fn repeat_by(&self, by: &PyExpr) -> PyExpr {
    self.clone().inner.repeat_by(by.inner.clone()).into()
  }

  #[napi]
  pub fn pow(&self, exponent: f64) -> PyExpr {
    self.clone().inner.pow(exponent).into()
  }
  #[napi]
  pub fn cumsum(&self, reverse: bool) -> PyExpr {
    self.clone().inner.cumsum(reverse).into()
  }
  #[napi]
  pub fn cummax(&self, reverse: bool) -> PyExpr {
    self.clone().inner.cummax(reverse).into()
  }
  #[napi]
  pub fn cummin(&self, reverse: bool) -> PyExpr {
    self.clone().inner.cummin(reverse).into()
  }
  #[napi]
  pub fn cumprod(&self, reverse: bool) -> PyExpr {
    self.clone().inner.cumprod(reverse).into()
  }

  #[napi]
  pub fn product(&self) -> PyExpr {
    self.clone().inner.product().into()
  }

  #[napi]
  pub fn str_parse_date(&self, fmt: Option<String>, strict: bool, exact: bool) -> PyExpr {
    self
      .inner
      .clone()
      .str()
      .strptime(StrpTimeOptions {
        date_dtype: DataType::Date,
        fmt,
        strict,
        exact,
      })
      .into()
  }
  #[napi]
  pub fn str_parse_datetime(&self, fmt: Option<String>, strict: bool, exact: bool) -> PyExpr {
    self
      .inner
      .clone()
      .str()
      .strptime(StrpTimeOptions {
        date_dtype: DataType::Datetime(TimeUnit::Microseconds, None),
        fmt,
        strict,
        exact,
      })
      .into()
  }

  #[napi]
  pub fn str_strip(&self) -> PyExpr {
    let function = |s: Series| {
      let ca = s.utf8()?;

      Ok(ca.apply(|s| Cow::Borrowed(s.trim())).into_series())
    };
    self
      .clone()
      .inner
      .map(function, GetOutput::same_type())
      .with_fmt("str.strip")
      .into()
  }

  #[napi]
  pub fn str_rstrip(&self) -> PyExpr {
    let function = |s: Series| {
      let ca = s.utf8()?;

      Ok(ca.apply(|s| Cow::Borrowed(s.trim_end())).into_series())
    };
    self
      .clone()
      .inner
      .map(function, GetOutput::same_type())
      .with_fmt("str.rstrip")
      .into()
  }

  #[napi]
  pub fn str_lstrip(&self) -> PyExpr {
    let function = |s: Series| {
      let ca = s.utf8()?;

      Ok(ca.apply(|s| Cow::Borrowed(s.trim_start())).into_series())
    };
    self
      .clone()
      .inner
      .map(function, GetOutput::same_type())
      .with_fmt("str.lstrip")
      .into()
  }

  #[napi]
  pub fn str_to_uppercase(&self) -> PyExpr {
    let function = |s: Series| {
      let ca = s.utf8()?;
      Ok(ca.to_uppercase().into_series())
    };
    self
      .clone()
      .inner
      .map(function, GetOutput::from_type(DataType::UInt32))
      .with_fmt("str.to_uppercase")
      .into()
  }
  #[napi]
  pub fn str_slice(&self, start: i64, length: Option<i64>) -> PyExpr {
    let function = move |s: Series| {
      let length = length.map(|l| l as u64);
      let ca = s.utf8()?;
      Ok(ca.str_slice(start, length)?.into_series())
    };
    self
      .clone()
      .inner
      .map(function, GetOutput::from_type(DataType::Utf8))
      .with_fmt("str.slice")
      .into()
  }

  #[napi]
  pub fn str_to_lowercase(&self) -> PyExpr {
    let function = |s: Series| {
      let ca = s.utf8()?;
      Ok(ca.to_lowercase().into_series())
    };
    self
      .clone()
      .inner
      .map(function, GetOutput::from_type(DataType::UInt32))
      .with_fmt("str.to_lowercase")
      .into()
  }

  #[napi]
  pub fn str_lengths(&self) -> PyExpr {
    let function = |s: Series| {
      let ca = s.utf8()?;
      Ok(ca.str_lengths().into_series())
    };
    self
      .clone()
      .inner
      .map(function, GetOutput::from_type(DataType::UInt32))
      .with_fmt("str.len")
      .into()
  }

  #[napi]
  pub fn str_replace(&self, pat: String, val: String) -> PyExpr {
    let function = move |s: Series| {
      let ca = s.utf8()?;
      match ca.replace(&pat, &val) {
        Ok(ca) => Ok(ca.into_series()),
        Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
      }
    };
    self
      .clone()
      .inner
      .map(function, GetOutput::same_type())
      .with_fmt("str.replace")
      .into()
  }

  #[napi]
  pub fn str_replace_all(&self, pat: String, val: String) -> PyExpr {
    let function = move |s: Series| {
      let ca = s.utf8()?;
      match ca.replace_all(&pat, &val) {
        Ok(ca) => Ok(ca.into_series()),
        Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
      }
    };
    self
      .clone()
      .inner
      .map(function, GetOutput::same_type())
      .with_fmt("str.replace_all")
      .into()
  }

  #[napi]
  pub fn str_contains(&self, pat: String) -> PyExpr {
    let function = move |s: Series| {
      let ca = s.utf8()?;
      match ca.contains(&pat) {
        Ok(ca) => Ok(ca.into_series()),
        Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
      }
    };
    self
      .clone()
      .inner
      .map(function, GetOutput::from_type(DataType::Boolean))
      .with_fmt("str.contains")
      .into()
  }
  #[napi]
  pub fn str_hex_encode(&self) -> PyExpr {
    self
      .clone()
      .inner
      .map(
        move |s| s.utf8().map(|s| s.hex_encode().into_series()),
        GetOutput::same_type(),
      )
      .with_fmt("str.hex_encode")
      .into()
  }
  #[napi]
  pub fn str_hex_decode(&self, strict: Option<bool>) -> PyExpr {
    self
      .clone()
      .inner
      .map(
        move |s| s.utf8()?.hex_decode(strict).map(|s| s.into_series()),
        GetOutput::same_type(),
      )
      .with_fmt("str.hex_decode")
      .into()
  }
  #[napi]
  pub fn str_base64_encode(&self) -> PyExpr {
    self
      .clone()
      .inner
      .map(
        move |s| s.utf8().map(|s| s.base64_encode().into_series()),
        GetOutput::same_type(),
      )
      .with_fmt("str.base64_encode")
      .into()
  }

  #[napi]
  pub fn str_base64_decode(&self, strict: Option<bool>) -> PyExpr {
    self
      .clone()
      .inner
      .map(
        move |s| s.utf8()?.base64_decode(strict).map(|s| s.into_series()),
        GetOutput::same_type(),
      )
      .with_fmt("str.base64_decode")
      .into()
  }
  #[napi]
  pub fn str_json_path_match(&self, pat: String) -> PyExpr {
    let function = move |s: Series| {
      let ca = s.utf8()?;
      match ca.json_path_match(&pat) {
        Ok(ca) => Ok(ca.into_series()),
        Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
      }
    };
    self
      .clone()
      .inner
      .map(function, GetOutput::from_type(DataType::Boolean))
      .with_fmt("str.json_path_match")
      .into()
  }
  #[napi]
  pub fn str_extract(&self, pat: String, group_index: i64) -> PyExpr {
    self
      .inner
      .clone()
      .str()
      .extract(&pat, group_index as usize)
      .into()
  }
  #[napi]
  pub fn strftime(&self, fmt: String) -> PyExpr {
    self.inner.clone().dt().strftime(&fmt).into()
  }
  #[napi]
  pub fn str_split(&self, by: String) -> PyExpr {
    self.inner.clone().str().split(&by).into()
  }
  #[napi]
  pub fn str_split_inclusive(&self, by: String) -> PyExpr {
    self.inner.clone().str().split_inclusive(&by).into()
  }
  #[napi]
  pub fn str_split_exact(&self, by: String, n: i64) -> PyExpr {
    self.inner.clone().str().split_exact(&by, n as usize).into()
  }
  #[napi]
  pub fn str_split_exact_inclusive(&self, by: String, n: i64) -> PyExpr {
    self
      .inner
      .clone()
      .str()
      .split_exact_inclusive(&by, n as usize)
      .into()
  }

  #[napi]
  pub fn arr_lengths(&self) -> PyExpr {
    self.inner.clone().arr().lengths().into()
  }

  #[napi]
  pub fn year(&self) -> PyExpr {
    self.clone().inner.dt().year().into()
  }
  #[napi]
  pub fn month(&self) -> PyExpr {
    self.clone().inner.dt().month().into()
  }
  #[napi]
  pub fn week(&self) -> PyExpr {
    self.clone().inner.dt().week().into()
  }
  #[napi]
  pub fn weekday(&self) -> PyExpr {
    self.clone().inner.dt().weekday().into()
  }
  #[napi]
  pub fn day(&self) -> PyExpr {
    self.clone().inner.dt().day().into()
  }
  #[napi]
  pub fn ordinal_day(&self) -> PyExpr {
    self.clone().inner.dt().ordinal_day().into()
  }
  #[napi]
  pub fn hour(&self) -> PyExpr {
    self.clone().inner.dt().hour().into()
  }
  #[napi]
  pub fn minute(&self) -> PyExpr {
    self.clone().inner.dt().minute().into()
  }
  #[napi]
  pub fn second(&self) -> PyExpr {
    self.clone().inner.dt().second().into()
  }
  #[napi]
  pub fn nanosecond(&self) -> PyExpr {
    self.clone().inner.dt().nanosecond().into()
  }
  #[napi]
  pub fn duration_days(&self) -> PyExpr {
    self
      .inner
      .clone()
      .map(
        |s| Ok(s.duration()?.days().into_series()),
        GetOutput::from_type(DataType::Int64),
      )
      .into()
  }
  #[napi]
  pub fn duration_hours(&self) -> PyExpr {
    self
      .inner
      .clone()
      .map(
        |s| Ok(s.duration()?.hours().into_series()),
        GetOutput::from_type(DataType::Int64),
      )
      .into()
  }
  #[napi]
  pub fn duration_seconds(&self) -> PyExpr {
    self
      .inner
      .clone()
      .map(
        |s| Ok(s.duration()?.seconds().into_series()),
        GetOutput::from_type(DataType::Int64),
      )
      .into()
  }
  #[napi]
  pub fn duration_nanoseconds(&self) -> PyExpr {
    self
      .inner
      .clone()
      .map(
        |s| Ok(s.duration()?.nanoseconds().into_series()),
        GetOutput::from_type(DataType::Int64),
      )
      .into()
  }
  #[napi]
  pub fn duration_milliseconds(&self) -> PyExpr {
    self
      .inner
      .clone()
      .map(
        |s| Ok(s.duration()?.milliseconds().into_series()),
        GetOutput::from_type(DataType::Int64),
      )
      .into()
  }
  #[napi]
  pub fn timestamp(&self) -> PyExpr {
    self
      .inner
      .clone()
      .dt()
      .timestamp(TimeUnit::Milliseconds)
      .into()
  }
  #[napi]
  pub fn dt_epoch_seconds(&self) -> PyExpr {
    self
      .clone()
      .inner
      .map(
        |s| {
          s.timestamp(TimeUnit::Milliseconds)
            .map(|ca| (ca / 1000).into_series())
        },
        GetOutput::from_type(DataType::Int64),
      )
      .into()
  }
  #[napi]
  pub fn dot(&self, other: &PyExpr) -> PyExpr {
    self.inner.clone().dot(other.inner.clone()).into()
  }
  #[napi]
  pub fn hash(&self, k0: Wrap<u64>, k1: Wrap<u64>, k2: Wrap<u64>, k3: Wrap<u64>) -> PyExpr {
    let function = move |s: Series| {
      let hb = ahash::RandomState::with_seeds(k0.0, k1.0, k2.0, k3.0);
      Ok(s.hash(hb).into_series())
    };
    self
      .clone()
      .inner
      .map(function, GetOutput::from_type(DataType::UInt64))
      .into()
  }

  #[napi]
  pub fn reinterpret(&self, signed: bool) -> PyExpr {
    let function = move |s: Series| reinterpret(&s, signed);
    let dt = if signed {
      DataType::Int64
    } else {
      DataType::UInt64
    };
    self
      .clone()
      .inner
      .map(function, GetOutput::from_type(dt))
      .into()
  }
  #[napi]
  pub fn mode(&self) -> PyExpr {
    self.inner.clone().mode().into()
  }
  #[napi]
  pub fn keep_name(&self) -> PyExpr {
    self.inner.clone().keep_name().into()
  }
  #[napi]
  pub fn prefix(&self, prefix: String) -> PyExpr {
    self.inner.clone().prefix(&prefix).into()
  }
  #[napi]
  pub fn suffix(&self, suffix: String) -> PyExpr {
    self.inner.clone().suffix(&suffix).into()
  }

  #[napi]
  pub fn exclude(&self, columns: Vec<String>) -> PyExpr {
    self.inner.clone().exclude(&columns).into()
  }

  #[napi]
  pub fn exclude_dtype(&self, dtypes: Vec<Wrap<DataType>>) -> PyExpr {
    // Safety:
    // Wrap is transparent.
    let dtypes: Vec<DataType> = unsafe { std::mem::transmute(dtypes) };
    self.inner.clone().exclude_dtype(&dtypes).into()
  }
  #[napi]
  pub fn interpolate(&self) -> PyExpr {
    self.inner.clone().interpolate().into()
  }
  #[napi]
  pub fn rolling_sum(&self, options: JsRollingOptions) -> PyExpr {
    self.inner.clone().rolling_sum(options.into()).into()
  }
  #[napi]
  pub fn rolling_min(&self, options: JsRollingOptions) -> PyExpr {
    self.inner.clone().rolling_min(options.into()).into()
  }
  #[napi]
  pub fn rolling_max(&self, options: JsRollingOptions) -> PyExpr {
    self.inner.clone().rolling_max(options.into()).into()
  }
  #[napi]
  pub fn rolling_mean(&self, options: JsRollingOptions) -> PyExpr {
    self.inner.clone().rolling_mean(options.into()).into()
  }
  #[napi]
  pub fn rolling_std(&self, options: JsRollingOptions) -> PyExpr {
    self.inner.clone().rolling_std(options.into()).into()
  }
  #[napi]
  pub fn rolling_var(&self, options: JsRollingOptions) -> PyExpr {
    self.inner.clone().rolling_var(options.into()).into()
  }
  #[napi]
  pub fn rolling_median(&self, options: JsRollingOptions) -> PyExpr {
    self.inner.clone().rolling_median(options.into()).into()
  }
  #[napi]
  pub fn rolling_quantile(
    &self,
    quantile: f64,
    interpolation: Wrap<QuantileInterpolOptions>,
    options: JsRollingOptions,
  ) -> PyExpr {
    self
      .inner
      .clone()
      .rolling_quantile(quantile, interpolation.0, options.into())
      .into()
  }
  #[napi]
  pub fn rolling_skew(&self, window_size: i64, bias: bool) -> PyExpr {
    self
      .inner
      .clone()
      .rolling_apply_float(window_size as usize, move |ca| {
        ca.clone().into_series().skew(bias).unwrap()
      })
      .into()
  }
  #[napi]
  pub fn lower_bound(&self) -> PyExpr {
    self.inner.clone().lower_bound().into()
  }

  #[napi]
  pub fn upper_bound(&self) -> PyExpr {
    self.inner.clone().upper_bound().into()
  }

  #[napi]
  pub fn lst_max(&self) -> PyExpr {
    self.inner.clone().arr().max().into()
  }
  #[napi]
  pub fn lst_min(&self) -> PyExpr {
    self.inner.clone().arr().min().into()
  }

  #[napi]
  pub fn lst_sum(&self) -> PyExpr {
    self.inner.clone().arr().sum().with_fmt("arr.sum").into()
  }

  #[napi]
  pub fn lst_mean(&self) -> PyExpr {
    self.inner.clone().arr().mean().with_fmt("arr.mean").into()
  }

  #[napi]
  pub fn lst_sort(&self, reverse: bool) -> PyExpr {
    self
      .inner
      .clone()
      .arr()
      .sort(reverse)
      .with_fmt("arr.sort")
      .into()
  }

  #[napi]
  pub fn lst_reverse(&self) -> PyExpr {
    self
      .inner
      .clone()
      .arr()
      .reverse()
      .with_fmt("arr.reverse")
      .into()
  }

  #[napi]
  pub fn lst_unique(&self) -> PyExpr {
    self
      .inner
      .clone()
      .arr()
      .unique()
      .with_fmt("arr.unique")
      .into()
  }
  #[napi]
  pub fn lst_get(&self, index: i64) -> PyExpr {
    self.inner.clone().arr().get(index).into()
  }
  #[napi]
  pub fn lst_join(&self, separator: String) -> PyExpr {
    self.inner.clone().arr().join(&separator).into()
  }
  #[napi]
  pub fn lst_arg_min(&self) -> PyExpr {
    self.inner.clone().arr().arg_min().into()
  }

  #[napi]
  pub fn lst_arg_max(&self) -> PyExpr {
    self.inner.clone().arr().arg_max().into()
  }

  #[napi]
  pub fn lst_diff(&self, n: i64, null_behavior: Wrap<NullBehavior>) -> PyExpr {
    self
      .inner
      .clone()
      .arr()
      .diff(n as usize, null_behavior.0)
      .into()
  }

  #[napi]
  pub fn lst_shift(&self, periods: i64) -> PyExpr {
    self.inner.clone().arr().shift(periods).into()
  }
  #[napi]
  pub fn lst_slice(&self, offset: i64, length: i64) -> PyExpr {
    self
      .inner
      .clone()
      .arr()
      .slice(offset, length as usize)
      .into()
  }

  #[napi]
  pub fn rank(&self, method: Wrap<RankMethod>, reverse: bool) -> PyExpr {
    let options = RankOptions {
      method: method.0,
      descending: reverse,
    };
    self.inner.clone().rank(options).into()
  }
  #[napi]
  pub fn diff(&self, n: i64, null_behavior: Wrap<NullBehavior>) -> PyExpr {
    self.inner.clone().diff(n as usize, null_behavior.0).into()
  }
  #[napi]
  pub fn pct_change(&self, n: i64) -> PyExpr {
    self.inner.clone().pct_change(n as usize).into()
  }

  #[napi]
  pub fn skew(&self, bias: bool) -> PyExpr {
    self.inner.clone().skew(bias).into()
  }
  #[napi]
  pub fn kurtosis(&self, fisher: bool, bias: bool) -> PyExpr {
    self.inner.clone().kurtosis(fisher, bias).into()
  }
  #[napi]
  pub fn str_concat(&self, delimiter: String) -> PyExpr {
    self.inner.clone().str().concat(&delimiter).into()
  }
  #[napi]
  pub fn cat_set_ordering(&self, ordering: String) -> PyExpr {
    let ordering = match ordering.as_ref() {
      "physical" => CategoricalOrdering::Physical,
      "lexical" => CategoricalOrdering::Lexical,
      _ => panic!("expected one of {{'physical', 'lexical'}}"),
    };

    self.inner.clone().cat().set_ordering(ordering).into()
  }
  #[napi]
  pub fn reshape(&self, dims: Vec<i64>) -> PyExpr {
    self.inner.clone().reshape(&dims).into()
  }
  #[napi]
  pub fn cumcount(&self, reverse: bool) -> PyExpr {
    self.inner.clone().cumcount(reverse).into()
  }
  #[napi]
  pub fn to_physical(&self) -> PyExpr {
    self
      .inner
      .clone()
      .map(
        |s| Ok(s.to_physical_repr().into_owned()),
        GetOutput::map_dtype(|dt| dt.to_physical()),
      )
      .with_fmt("to_physical")
      .into()
  }

  #[napi]
  pub fn shuffle(&self, seed: Wrap<u64>) -> PyExpr {
    self.inner.clone().shuffle(seed.0).into()
  }

  #[napi]
  pub fn sample_frac(&self, frac: f64, with_replacement: bool, seed: Option<i64>) -> PyExpr {
    let seed = seed.map(|s| s as u64);
    self
      .inner
      .clone()
      .sample_frac(frac, with_replacement, seed)
      .into()
  }
  #[napi]
  pub fn ewm_mean(&self, alpha: f64, adjust: bool, min_periods: i64) -> PyExpr {
    let options = EWMOptions {
      alpha,
      adjust,
      min_periods: min_periods as usize,
    };
    self.inner.clone().ewm_mean(options).into()
  }
  #[napi]
  pub fn ewm_std(&self, alpha: f64, adjust: bool, min_periods: i64) -> Self {
    let options = EWMOptions {
      alpha,
      adjust,
      min_periods: min_periods as usize,
    };
    self.inner.clone().ewm_std(options).into()
  }
  #[napi]
  pub fn ewm_var(&self, alpha: f64, adjust: bool, min_periods: i64) -> Self {
    let options = EWMOptions {
      alpha,
      adjust,
      min_periods: min_periods as usize,
    };
    self.inner.clone().ewm_var(options).into()
  }
  #[napi]
  pub fn extend_constant(&self, value: JsAnyValue, n: i64) -> Self {
    self
      .inner
      .clone()
      .apply(
        move |s| s.extend_constant(value.clone().into(), n as usize),
        GetOutput::same_type(),
      )
      .with_fmt("extend")
      .into()
  }
  #[napi]
  pub fn any(&self) -> PyExpr {
    self.inner.clone().any().into()
  }

  #[napi]
  pub fn all(&self) -> PyExpr {
    self.inner.clone().all().into()
  }
  #[napi]
  pub fn struct_field_by_name(&self, name: String) -> PyExpr {
    self.inner.clone().struct_().field_by_name(&name).into()
  }
  #[napi]
  pub fn struct_rename_fields(&self, names: Vec<String>) -> PyExpr {
    self.inner.clone().struct_().rename_fields(names).into()
  }
  #[napi]
  pub fn log(&self, base: f64) -> PyExpr {
    self.inner.clone().log(base).into()
  }
  #[napi]
  pub fn entropy(&self, base: f64) -> PyExpr {
    self.inner.clone().entropy(base).into()
  }
}

// #[napi]
// #[derive(Clone)]
// pub struct When {
//   pub predicate: PyExpr,
// }
// #[napi]
// #[derive(Clone)]
// pub struct WhenThen {
//   pub predicate: PyExpr,
//   pub then: PyExpr,
// }

// #[napi]
// #[derive(Clone)]
// pub struct WhenThenThen {
//   inner: dsl::WhenThenThen,
// }

// #[napi]
// impl When {
//   #[napi]
//   pub fn then(&self, expr: &PyExpr) -> WhenThen {
//     WhenThen {
//       predicate: self.predicate.clone(),
//       then: expr.clone(),
//     }
//   }
// }
// #[napi]
// impl WhenThen {
//   #[napi]
//   pub fn when(&self, predicate: &PyExpr) -> WhenThenThen {
//     let e = dsl::when(self.predicate.inner.clone())
//       .then(self.then.inner.clone())
//       .when(predicate.inner.clone());
//     WhenThenThen { inner: e }
//   }

//   #[napi]
//   pub fn otherwise(&self, expr: &PyExpr) -> PyExpr {
//     dsl::ternary_expr(
//       self.predicate.inner.clone(),
//       self.then.inner.clone(),
//       expr.inner.clone(),
//     )
//     .into()
//   }
// }

// #[napi]
// impl WhenThenThen {
//   #[napi]
//   pub fn when(&self, predicate: &PyExpr) -> WhenThenThen {
//     Self {
//       inner: self.inner.clone().when(predicate.inner.clone()),
//     }
//   }

//   #[napi]
//   pub fn then(&self, expr: &PyExpr) -> WhenThenThen {
//     Self {
//       inner: self.inner.clone().then(expr.inner.clone()),
//     }
//   }
//   #[napi]
//   pub fn otherwise(&self, expr: &PyExpr) -> PyExpr {
//     self.inner.clone().otherwise(expr.inner.clone()).into()
//   }
// }

// #[napi]
// pub fn when(predicate: &PyExpr) -> When {
//   When {
//     predicate: predicate.clone(),
//   }
// }

#[napi]
pub fn col(name: String) -> PyExpr {
  dsl::col(&name).into()
}

#[napi]
pub fn count() -> PyExpr {
  dsl::count().into()
}

#[napi]
pub fn first() -> PyExpr {
  dsl::first().into()
}

#[napi]
pub fn last() -> PyExpr {
  dsl::last().into()
}

#[napi]
pub fn cols(names: Vec<String>) -> PyExpr {
  dsl::cols(names).into()
}
#[napi]
pub fn lit(value: JsAnyValue) -> PyExpr {
  match value {
    JsAnyValue::Boolean(v) => dsl::lit(v),
    JsAnyValue::Utf8(v) => dsl::lit(v),
    JsAnyValue::UInt8(v) => dsl::lit(v),
    JsAnyValue::UInt16(v) => dsl::lit(v),
    JsAnyValue::UInt32(v) => dsl::lit(v),
    JsAnyValue::UInt64(v) => dsl::lit(v),
    JsAnyValue::Int8(v) => dsl::lit(v),
    JsAnyValue::Int16(v) => dsl::lit(v),
    JsAnyValue::Int32(v) => dsl::lit(v),
    JsAnyValue::Int64(v) => dsl::lit(v),
    JsAnyValue::Float32(v) => dsl::lit(v),
    JsAnyValue::Float64(v) => dsl::lit(v),
    JsAnyValue::Date(v) => dsl::lit(v),
    // JsAnyValue::Datetime(v, w, _) => dsl::lit(v),
    // JsAnyValue::Duration(v, _) => dsl::lit(v),
    JsAnyValue::Time(v) => dsl::lit(v),
    JsAnyValue::List(v) => dsl::lit(v),
    _ => dsl::lit(polars::prelude::Null {}),
  }
  .into()
}

#[napi]
pub fn range(low: i64, high: i64, dtype: Wrap<DataType>) -> PyExpr {
  match dtype.0 {
    DataType::Int32 => dsl::range(low as i32, high as i32).into(),
    DataType::UInt32 => dsl::range(low as u32, high as u32).into(),
    _ => dsl::range(low, high).into(),
  }
}
