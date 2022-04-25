use crate::conversion::str_to_null_behavior;
use crate::conversion::str_to_rankmethod;
use crate::conversion::Wrap;
use crate::utils::str_to_polarstype;
use crate::{console_log, log};
use polars::prelude::*;
use wasm_bindgen::JsCast;

use super::{error::JsPolarsErr, JsResult};
use crate::conversion::FromJsValue;
use crate::{extern_iterator, extern_struct};

use std::ops::Deref;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name=Series)]
#[repr(transparent)]
pub struct JsSeries {
    pub(crate) series: Series,
}

impl JsSeries {
    pub(crate) fn new(series: Series) -> Self {
        JsSeries { series }
    }
}

impl From<Series> for JsSeries {
    fn from(series: Series) -> Self {
        Self { series }
    }
}

// impl wasm_bindgen::convert::FromWasmAbi for JsSeries {

// }
impl Deref for JsSeries {
    type Target = Series;

    fn deref(&self) -> &Self::Target {
        &self.series
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "Series")]
    pub type ExternSeries;

    #[wasm_bindgen(typescript_type = "any")]
    pub type ExternAnyValue;

    #[wasm_bindgen(method, getter = ptr)]
    fn ptr(this: &ExternSeries) -> f64;

    #[wasm_bindgen(static_method_of = ExternSeries)]
    fn wrap(ptr: u32) -> ExternSeries;

    #[wasm_bindgen(typescript_type = "Series[]")]
    pub type SeriesArray;

}

extern_struct!(ExternSeries, JsSeries);
extern_iterator!(SeriesArray, ExternSeries, JsSeries);

#[wasm_bindgen(js_class=Series)]
impl JsSeries {
    pub fn wrap(ptr: u32) -> JsSeries {
        unsafe { JsSeries::from_abi(ptr) }
    }
    pub fn new_str(name: &str, values: &js_sys::Array) -> JsResult<JsSeries> {
        let series = Utf8Chunked::from_iter_options(name, values.iter().map(|v| v.as_string()))
            .into_series();
        Ok(JsSeries { series })
    }
    pub fn new_bool(name: &str, values: &js_sys::Array) -> JsResult<JsSeries> {
        let series = BooleanChunked::from_iter_options(name, values.iter().map(|v| v.as_bool()))
            .into_series();

        Ok(JsSeries { series })
    }
    pub fn new_f64(name: &str, values: &js_sys::Array) -> JsResult<JsSeries> {
        let series =
            Float64Chunked::from_iter_options(name, values.iter().map(|v: JsValue| v.as_f64()))
                .into_series();
        Ok(JsSeries { series })
    }
    pub fn new_i8(name: &str, values: &js_sys::Array) -> JsResult<JsSeries> {
        let series = Int8Chunked::from_iter_options(
            name,
            values.iter().map(|v: JsValue| v.as_f64().map(|n| n as i8)),
        )
        .into_series();
        Ok(JsSeries { series })
    }

    pub fn new_series_list(name: &str, val: SeriesArray, _strict: bool) -> Self {
        let vals = val.into_iter().map(|x| x.series).collect::<Box<[Series]>>();
        Series::new(name, &vals).into()
    }

    pub fn get_fmt(&self, index: usize) -> String {
        format!("{}", self.series.get(index))
    }
    pub fn rechunk(&mut self, in_place: bool) -> Option<JsSeries> {
        let series = self.series.rechunk();
        if in_place {
            self.series = series;
            None
        } else {
            Some(series.into())
        }
    }
    pub fn get_idx(&self, idx: usize) -> JsValue {
        Wrap(self.series.get(idx)).into()
    }
    pub fn bitand(&self, other: &JsSeries) -> JsResult<JsSeries> {
        let out = self
            .series
            .bitand(&other.series)
            .map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }

    pub fn bitor(&self, other: &JsSeries) -> JsResult<JsSeries> {
        let out = self
            .series
            .bitor(&other.series)
            .map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }
    pub fn bitxor(&self, other: &JsSeries) -> JsResult<JsSeries> {
        let out = self
            .series
            .bitxor(&other.series)
            .map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }

    #[wasm_bindgen(js_name = cumSum)]
    pub fn cumsum(&self, reverse: bool) -> Self {
        self.series.cumsum(reverse).into()
    }
    #[wasm_bindgen(js_name = cumMax)]
    pub fn cummax(&self, reverse: bool) -> Self {
        self.series.cummax(reverse).into()
    }
    #[wasm_bindgen(js_name = cumMin)]
    pub fn cummin(&self, reverse: bool) -> Self {
        self.series.cummin(reverse).into()
    }
    #[wasm_bindgen(js_name = cumProd)]
    pub fn cumprod(&self, reverse: bool) -> Self {
        self.series.cumprod(reverse).into()
    }
    #[wasm_bindgen(js_name = chunkLengths)]
    pub fn chunk_lengths(&self) -> Vec<usize> {
        self.series.chunk_lengths().collect()
    }

    pub fn name(&self) -> js_sys::JsString {
        self.series.name().into()
    }

    pub fn rename(&mut self, name: &str) {
        self.series.rename(name);
    }
    pub fn dtype(&self) -> String {
        let dt: crate::datatypes::JsDataType = self.series.dtype().into();
        dt.to_string()
    }
    pub fn inner_dtype(&self) -> Option<String> {
        self.series.dtype().inner_dtype().map(|dt| {
            let dt: crate::datatypes::JsDataType = dt.into();
            dt.to_string()
        })
    }
    pub fn mean(&self) -> Option<f64> {
        match self.series.dtype() {
            DataType::Boolean => {
                let s = self.series.cast(&DataType::UInt8).unwrap();
                s.mean()
            }
            _ => self.series.mean(),
        }
    }

    pub fn max(&self) -> JsValue {
        match self.series.dtype() {
            DataType::Float32 | DataType::Float64 => {
                self.series.max::<f64>().map(JsValue::from).into()
            }
            DataType::Boolean => self
                .series
                .max::<u32>()
                .map(|v| v == 1)
                .map(JsValue::from)
                .into(),
            _ => self.series.max::<i64>().map(JsValue::from).into(),
        }
    }

    pub fn min(&self) -> JsValue {
        match self.series.dtype() {
            DataType::Float32 | DataType::Float64 => {
                self.series.min::<f64>().map(JsValue::from).into()
            }
            DataType::Boolean => self
                .series
                .min::<u32>()
                .map(|v| v == 1)
                .map(JsValue::from)
                .into(),
            _ => self.series.min::<i64>().map(JsValue::from).into(),
        }
    }

    pub fn sum(&self) -> JsValue {
        match self.series.dtype() {
            DataType::Float32 | DataType::Float64 => {
                self.series.sum::<f64>().map(JsValue::from).into()
            }
            DataType::Boolean => self
                .series
                .sum::<u32>()
                .map(|v| v == 1)
                .map(JsValue::from)
                .into(),
            _ => self.series.sum::<i64>().map(JsValue::from).into(),
        }
    }
    #[wasm_bindgen(js_name = nChunks)]
    pub fn n_chunks(&self) -> usize {
        self.series.n_chunks()
    }

    pub fn limit(&self, num_elements: usize) -> Self {
        let series = self.series.limit(num_elements);
        series.into()
    }

    pub fn slice(&self, offset: i64, length: usize) -> Self {
        let series = self.series.slice(offset, length);
        series.into()
    }

    pub fn append(&mut self, other: &JsSeries) -> JsResult<()> {
        self.series
            .append(&other.series)
            .map_err(JsPolarsErr::from)?;
        Ok(())
    }
    pub fn extend(&mut self, other: &JsSeries) -> JsResult<()> {
        self.series
            .extend(&other.series)
            .map_err(JsPolarsErr::from)?;
        Ok(())
    }
    pub fn filter(&self, filter: &JsSeries) -> JsResult<JsSeries> {
        let filter_series = &filter.series;
        if let Ok(ca) = filter_series.bool() {
            let series = self.series.filter(ca).map_err(JsPolarsErr::from)?;
            Ok(JsSeries { series })
        } else {
            Err(JsError::new("Expected a boolean mask").into())
        }
    }
    pub fn add(&self, other: &JsSeries) -> Self {
        (&self.series + &other.series).into()
    }

    pub fn sub(&self, other: &JsSeries) -> Self {
        (&self.series - &other.series).into()
    }

    pub fn mul(&self, other: &JsSeries) -> Self {
        (&self.series * &other.series).into()
    }

    pub fn div(&self, other: &JsSeries) -> Self {
        (&self.series / &other.series).into()
    }

    pub fn rem(&self, other: &JsSeries) -> Self {
        (&self.series % &other.series).into()
    }
    pub fn head(&self, length: Option<usize>) -> Self {
        (self.series.head(length)).into()
    }

    pub fn tail(&self, length: Option<usize>) -> Self {
        (self.series.tail(length)).into()
    }

    pub fn sort(&mut self, reverse: bool) -> Self {
        self.series.sort(reverse).into()
    }

    #[wasm_bindgen(js_name = argSort)]
    pub fn argsort(&self, reverse: bool, nulls_last: bool) -> Self {
        self.series
            .argsort(SortOptions {
                descending: reverse,
                nulls_last,
            })
            .into_series()
            .into()
    }

    pub fn unique(&self) -> JsResult<JsSeries> {
        let unique = self.series.unique().map_err(JsPolarsErr::from)?;
        Ok(unique.into())
    }

    // pub fn value_counts(&self) -> JsResult<JsDataFrame> {
    //     let df = self.series.value_counts(true).map_err(JsPolarsErr::from)?;
    //     Ok(df.into())
    // }
    #[wasm_bindgen(js_name = argUnique)]
    pub fn arg_unique(&self) -> JsResult<JsSeries> {
        let arg_unique = self.series.arg_unique().map_err(JsPolarsErr::from)?;
        Ok(arg_unique.into_series().into())
    }

    #[wasm_bindgen(js_name = argMin)]
    pub fn arg_min(&self) -> Option<usize> {
        self.series.arg_min()
    }
    #[wasm_bindgen(js_name = argMax)]
    pub fn arg_max(&self) -> Option<usize> {
        self.series.arg_max()
    }

    pub fn take(&self, indices: Vec<u32>) -> JsResult<JsSeries> {
        let indices = UInt32Chunked::from_vec("", indices);

        let take = self.series.take(&indices).map_err(JsPolarsErr::from)?;
        Ok(JsSeries::from(take))
    }
    #[wasm_bindgen(js_name = takeWithSeries)]
    pub fn take_with_series(&self, indices: &JsSeries) -> JsResult<JsSeries> {
        let idx = indices.series.u32().map_err(JsPolarsErr::from)?;
        let take = self.series.take(idx).map_err(JsPolarsErr::from)?;
        Ok(JsSeries::new(take))
    }

    //
    //
    //
    //

    #[wasm_bindgen(js_name = nullCount)]
    pub fn null_count(&self) -> JsResult<usize> {
        Ok(self.series.null_count())
    }

    #[wasm_bindgen(js_name = hasValidity)]
    pub fn has_validity(&self) -> bool {
        self.series.has_validity()
    }

    #[wasm_bindgen(js_name = isNull)]
    pub fn is_null(&self) -> JsSeries {
        Self::new(self.series.is_null().into_series())
    }

    #[wasm_bindgen(js_name = isNotNull)]
    ///
    /// __Get mask of non null values.__
    ///
    /// *`undefined` values are treated as null*
    /// ___
    /// @example
    /// ```
    /// > const s = pl.Series("a", [1.0, undefined, 2.0, 3.0, null])
    /// > s.isNotNull()
    /// shape: (5,)
    /// Series: 'a' [bool]
    /// [
    ///         true
    ///         false
    ///         true
    ///         true
    ///         false
    /// ]
    /// ```
    pub fn is_not_null(&self) -> JsSeries {
        Self::new(self.series.is_not_null().into_series())
    }

    #[wasm_bindgen(js_name = isNotNan)]
    pub fn is_not_nan(&self) -> JsResult<JsSeries> {
        let ca = self.series.is_not_nan().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }

    #[wasm_bindgen(js_name = isNan)]
    pub fn is_nan(&self) -> JsResult<JsSeries> {
        let ca = self.series.is_nan().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }

    #[wasm_bindgen(js_name = isFinite)]
    pub fn is_finite(&self) -> JsResult<JsSeries> {
        let ca = self.series.is_finite().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }

    #[wasm_bindgen(js_name = isInfinite)]
    pub fn is_infinite(&self) -> JsResult<JsSeries> {
        let ca = self.series.is_infinite().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }

    #[wasm_bindgen(js_name = isUnique)]
    pub fn is_unique(&self) -> JsResult<JsSeries> {
        let ca = self.series.is_unique().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }

    #[wasm_bindgen(js_name = argTrue)]
    pub fn arg_true(&self) -> JsResult<JsSeries> {
        let ca = self.series.arg_true().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }

    // pub fn sample_n(&self, n: usize, with_replacement: bool, seed: u64) -> JsResult<JsSeries> {
    //     let s = self
    //         .series
    //         .sample_n(n, with_replacement, seed)
    //         .map_err(JsPolarsErr::from)?;
    //     Ok(s.into())
    // }

    // pub fn sample_frac(&self, frac: f64, with_replacement: bool, seed: u64) -> JsResult<JsSeries> {
    //     let s = self
    //         .series
    //         .sample_frac(frac, with_replacement, seed)
    //         .map_err(JsPolarsErr::from)?;
    //     Ok(s.into())
    // }
    pub fn is_duplicated(&self) -> JsResult<JsSeries> {
        let ca = self.series.is_duplicated().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }

    pub fn explode(&self) -> JsResult<JsSeries> {
        let s = self.series.explode().map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }

    pub fn take_every(&self, n: usize) -> Self {
        let s = self.series.take_every(n);
        s.into()
    }
    pub fn series_equal(&self, other: &JsSeries, null_equal: bool, strict: bool) -> bool {
        if strict {
            self.series.eq(&other.series)
        } else if null_equal {
            self.series.series_equal_missing(&other.series)
        } else {
            self.series.series_equal(&other.series)
        }
    }
    pub fn eq(&self, rhs: &JsSeries) -> JsResult<JsSeries> {
        Ok(Self::new(self.series.equal(&rhs.series).into_series()))
    }

    pub fn neq(&self, rhs: &JsSeries) -> JsResult<JsSeries> {
        Ok(Self::new(self.series.not_equal(&rhs.series).into_series()))
    }

    pub fn gt(&self, rhs: &JsSeries) -> JsResult<JsSeries> {
        Ok(Self::new(self.series.gt(&rhs.series).into_series()))
    }

    pub fn gt_eq(&self, rhs: &JsSeries) -> JsResult<JsSeries> {
        Ok(Self::new(self.series.gt_eq(&rhs.series).into_series()))
    }

    pub fn lt(&self, rhs: &JsSeries) -> JsResult<JsSeries> {
        Ok(Self::new(self.series.lt(&rhs.series).into_series()))
    }

    pub fn lt_eq(&self, rhs: &JsSeries) -> JsResult<JsSeries> {
        Ok(Self::new(self.series.lt_eq(&rhs.series).into_series()))
    }

    pub fn _not(&self) -> JsResult<JsSeries> {
        let bool = self.series.bool().map_err(JsPolarsErr::from)?;
        Ok((!bool).into_series().into())
    }

    pub fn as_str(&self) -> JsResult<String> {
        Ok(format!("{:?}", self.series))
    }
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        format!("{:?}", self.series)
    }

    pub fn len(&self) -> usize {
        self.series.len()
    }
    pub fn to_physical(&self) -> Self {
        let s = self.series.to_physical_repr().into_owned();
        s.into()
    }
    pub fn to_list(&self) -> JsValue {
        todo!()
    }
    pub fn median(&self) -> Option<f64> {
        match self.series.dtype() {
            DataType::Boolean => {
                let s = self.series.cast(&DataType::UInt8).unwrap();
                s.median()
            }
            _ => self.series.median(),
        }
    }
    // pub fn quantile(&self, quantile: f64, interpolation: Wrap<QuantileInterpolOptions>) -> JsValue {
    //     Wrap(
    //         self.series
    //             .quantile_as_series(quantile, interpolation.0)
    //             .expect("invalid quantile")
    //             .get(0),
    //     )
    //     .into()
    // }
    pub fn as_single_ptr(&mut self) -> JsResult<usize> {
        let ptr = self.series.as_single_ptr().map_err(JsPolarsErr::from)?;
        Ok(ptr)
    }
    pub fn drop_nulls(&self) -> Self {
        self.series.drop_nulls().into()
    }

    pub fn fill_null(&self, strategy: &str) -> JsResult<JsSeries> {
        let strat = match strategy {
            "backward" => FillNullStrategy::Backward,
            "forward" => FillNullStrategy::Forward,
            "min" => FillNullStrategy::Min,
            "max" => FillNullStrategy::Max,
            "mean" => FillNullStrategy::Mean,
            "zero" => FillNullStrategy::Zero,
            "one" => FillNullStrategy::One,
            s => panic!("Strategy {} not supported", s),
        };
        let series = self.series.fill_null(strat).map_err(JsPolarsErr::from)?;
        Ok(JsSeries::new(series))
    }
    // pub fn is_in(&self, other: &JsSeries) -> JsResult<JsSeries> {
    //     let out = self
    //         .series
    //         .is_in(&other.series)
    //         .map_err(JsPolarsErr::from)?;
    //     Ok(out.into_series().into())
    // }

    pub fn clone(&self) -> Self {
        JsSeries::new(self.series.clone())
    }

    pub fn apply_lambda() -> JsResult<JsSeries> {
        todo!()
    }
    pub fn shift(&self, periods: i64) -> Self {
        let s = self.series.shift(periods);
        JsSeries::new(s)
    }
    pub fn zip_with(&self, mask: &JsSeries, other: &JsSeries) -> JsResult<JsSeries> {
        let mask = mask.series.bool().map_err(JsPolarsErr::from)?;
        let s = self
            .series
            .zip_with(mask, &other.series)
            .map_err(JsPolarsErr::from)?;
        Ok(JsSeries::new(s))
    }

    pub fn str_lengths(&self) -> JsResult<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca.str_lengths().into_series();
        Ok(JsSeries::new(s))
    }

    pub fn str_contains(&self, pat: &str) -> JsResult<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca.contains(pat).map_err(JsPolarsErr::from)?.into_series();
        Ok(s.into())
    }

    // pub fn str_json_path_match(&self, pat: &str) -> JsResult<JsSeries> {
    //     let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
    //     let s = ca
    //         .json_path_match(pat)
    //         .map_err(JsPolarsErr::from)?
    //         .into_series();
    //     Ok(s.into())
    // }

    pub fn str_extract(&self, pat: &str, group_index: usize) -> JsResult<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca
            .extract(pat, group_index)
            .map_err(JsPolarsErr::from)?
            .into_series();
        Ok(s.into())
    }

    pub fn str_replace(&self, pat: &str, val: &str) -> JsResult<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca
            .replace(pat, val)
            .map_err(JsPolarsErr::from)?
            .into_series();
        Ok(s.into())
    }

    pub fn str_replace_all(&self, pat: &str, val: &str) -> JsResult<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca
            .replace_all(pat, val)
            .map_err(JsPolarsErr::from)?
            .into_series();
        Ok(s.into())
    }

    pub fn str_to_uppercase(&self) -> JsResult<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca.to_uppercase().into_series();
        Ok(s.into())
    }

    pub fn str_to_lowercase(&self) -> JsResult<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca.to_lowercase().into_series();
        Ok(s.into())
    }

    pub fn str_slice(&self, start: i64, length: Option<u64>) -> JsResult<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca
            .str_slice(start, length)
            .map_err(JsPolarsErr::from)?
            .into_series();
        Ok(s.into())
    }

    pub fn str_hex_encode(&self) -> JsResult<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca.hex_encode().into_series();
        Ok(s.into())
    }
    pub fn str_hex_decode(&self, strict: Option<bool>) -> JsResult<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca
            .hex_decode(strict)
            .map_err(JsPolarsErr::from)?
            .into_series();

        Ok(s.into())
    }
    pub fn str_base64_encode(&self) -> JsResult<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca.base64_encode().into_series();
        Ok(s.into())
    }

    pub fn str_base64_decode(&self, strict: Option<bool>) -> JsResult<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca
            .base64_decode(strict)
            .map_err(JsPolarsErr::from)?
            .into_series();
        Ok(s.into())
    }

    // pub fn strftime(&self, fmt: &str) -> JsResult<JsSeries> {
    //     let s = self.series.strftime(fmt).map_err(JsPolarsErr::from)?;
    //     Ok(s.into())
    // }

    pub fn arr_lengths(&self) -> JsResult<JsSeries> {
        let ca = self.series.list().map_err(JsPolarsErr::from)?;
        let s = ca.lst_lengths().into_series();
        Ok(JsSeries::new(s))
    }

    // // pub fn timestamp(&self, tu: Wrap<TimeUnit>) -> JsResult<JsSeries> {
    // //     let ca = self.series.timestamp(tu.0).map_err(JsPolarsErr::from)?;
    // //     Ok(ca.into_series().into())
    // // }
    pub fn get_list(&self, index: usize) -> Option<JsSeries> {
        if let Ok(ca) = &self.series.list() {
            let s = ca.get(index);
            s.map(|s| s.into())
        } else {
            None
        }
    }

    // pub fn year(&self) -> JsResult<JsSeries> {
    //     let s = self.series.year().map_err(JsPolarsErr::from)?;
    //     Ok(s.into_series().into())
    // }

    // pub fn month(&self) -> JsResult<JsSeries> {
    //     let s = self.series.month().map_err(JsPolarsErr::from)?;
    //     Ok(s.into_series().into())
    // }

    // pub fn weekday(&self) -> JsResult<JsSeries> {
    //     let s = self.series.weekday().map_err(JsPolarsErr::from)?;
    //     Ok(s.into_series().into())
    // }

    // pub fn week(&self) -> JsResult<JsSeries> {
    //     let s = self.series.week().map_err(JsPolarsErr::from)?;
    //     Ok(s.into_series().into())
    // }

    // pub fn day(&self) -> JsResult<JsSeries> {
    //     let s = self.series.day().map_err(JsPolarsErr::from)?;
    //     Ok(s.into_series().into())
    // }

    // pub fn ordinal_day(&self) -> JsResult<JsSeries> {
    //     let s = self.series.ordinal_day().map_err(JsPolarsErr::from)?;
    //     Ok(s.into_series().into())
    // }

    // pub fn hour(&self) -> JsResult<JsSeries> {
    //     let s = self.series.hour().map_err(JsPolarsErr::from)?;
    //     Ok(s.into_series().into())
    // }

    // pub fn minute(&self) -> JsResult<JsSeries> {
    //     let s = self.series.minute().map_err(JsPolarsErr::from)?;
    //     Ok(s.into_series().into())
    // }

    // pub fn second(&self) -> JsResult<JsSeries> {
    //     let s = self.series.second().map_err(JsPolarsErr::from)?;
    //     Ok(s.into_series().into())
    // }

    // pub fn nanosecond(&self) -> JsResult<JsSeries> {
    //     let s = self.series.nanosecond().map_err(JsPolarsErr::from)?;
    //     Ok(s.into_series().into())
    // }

    // pub fn dt_epoch_seconds(&self) -> JsResult<JsSeries> {
    //     let ms = self
    //         .series
    //         .timestamp(TimeUnit::Milliseconds)
    //         .map_err(JsPolarsErr::from)?;
    //     Ok((ms / 1000).into_series().into())
    // }

    pub fn peak_max(&self) -> Self {
        self.series.peak_max().into_series().into()
    }

    pub fn peak_min(&self) -> Self {
        self.series.peak_min().into_series().into()
    }

    pub fn n_unique(&self) -> JsResult<usize> {
        let n = self.series.n_unique().map_err(JsPolarsErr::from)?;
        Ok(n)
    }

    pub fn is_first(&self) -> JsResult<JsSeries> {
        let out = self
            .series
            .is_first()
            .map_err(JsPolarsErr::from)?
            .into_series();
        Ok(out.into())
    }

    pub fn round(&self, decimals: u32) -> JsResult<JsSeries> {
        let s = self.series.round(decimals).map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }

    pub fn floor(&self) -> JsResult<JsSeries> {
        let s = self.series.floor().map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }

    pub fn shrink_to_fit(&mut self) {
        self.series.shrink_to_fit();
    }

    pub fn dot(&self, other: &JsSeries) -> Option<f64> {
        self.series.dot(&other.series)
    }

    pub fn hash(&self, k0: u64, k1: u64, k2: u64, k3: u64) -> Self {
        let hb = ahash::RandomState::with_seeds(k0, k1, k2, k3);
        self.series.hash(hb).into_series().into()
    }
    pub fn reinterpret(&self, signed: bool) -> JsResult<JsSeries> {
        let s = reinterpret(&self.series, signed).map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }

    pub fn mode(&self) -> JsResult<JsSeries> {
        let s = self.series.mode().map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }

    pub fn interpolate(&self) -> Self {
        let s = self.series.interpolate();
        s.into()
    }
    pub fn rank(&self, method: &str, reverse: bool) -> JsResult<JsSeries> {
        let method = str_to_rankmethod(method).unwrap();
        let options = RankOptions {
            method,
            descending: reverse,
        };
        Ok(self.series.rank(options).into())
    }

    pub fn diff(&self, n: usize, null_behavior: &str) -> JsResult<JsSeries> {
        let null_behavior = str_to_null_behavior(null_behavior)?;
        Ok(self.series.diff(n, null_behavior).into())
    }

    pub fn skew(&self, bias: bool) -> JsResult<Option<f64>> {
        let out = self.series.skew(bias).map_err(JsPolarsErr::from)?;
        Ok(out)
    }
    pub fn kurtosis(&self, fisher: bool, bias: bool) -> JsResult<Option<f64>> {
        let out = self
            .series
            .kurtosis(fisher, bias)
            .map_err(JsPolarsErr::from)?;
        Ok(out)
    }
    pub fn cast(&self, dtype: &str, strict: bool) -> JsResult<JsSeries> {
        let dtype = str_to_polarstype(dtype);
        let out = if strict {
            self.series.strict_cast(&dtype)
        } else {
            self.series.cast(&dtype)
        };
        let out = out.map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }
    pub fn abs(&self) -> JsResult<JsSeries> {
        let out = self.series.abs().map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }

    pub fn reshape(&self, dims: Vec<i64>) -> JsResult<JsSeries> {
        let out = self.series.reshape(&dims).map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }

    pub fn extend_constant(&self, jsv: JsValue, n: usize) -> JsResult<JsSeries> {
        let value = AnyValue::from_js(jsv);
        let out = self
            .series
            .extend_constant(value, n)
            .map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }
    pub fn time_unit(&self) -> JsValue {
        if let DataType::Datetime(tu, _) | DataType::Duration(tu) = self.series.dtype() {
            Some(match tu {
                TimeUnit::Nanoseconds => "ns",
                TimeUnit::Microseconds => "us",
                TimeUnit::Milliseconds => "ms",
            })
        } else {
            None
        }
        .into()
    }
}
macro_rules! impl_set_with_mask {
    ($name:ident, $native:ty, $cast:ident, $variant:ident) => {
        fn $name(series: &Series, filter: &JsSeries, value: Option<$native>) -> Result<Series> {
            let mask = filter.series.bool()?;
            let ca = series.$cast()?;
            let new = ca.set(mask, value)?;
            Ok(new.into_series())
        }

        #[wasm_bindgen(js_class=Series)]
        impl JsSeries {
            pub fn $name(&self, filter: &JsSeries, value: Option<$native>) -> JsResult<JsSeries> {
                let series = $name(&self.series, filter, value).map_err(JsPolarsErr::from)?;
                Ok(JsSeries::new(series))
            }
        }
    };
}

// impl_set_with_mask!(set_with_mask_str, &str, utf8, Utf8); // TODO!
impl_set_with_mask!(set_with_mask_f64, f64, f64, Float64);
impl_set_with_mask!(set_with_mask_f32, f32, f32, Float32);
impl_set_with_mask!(set_with_mask_u8, u8, u8, UInt8);
impl_set_with_mask!(set_with_mask_u16, u16, u16, UInt16);
impl_set_with_mask!(set_with_mask_u32, u32, u32, UInt32);
impl_set_with_mask!(set_with_mask_u64, u64, u64, UInt64);
impl_set_with_mask!(set_with_mask_i8, i8, i8, Int8);
impl_set_with_mask!(set_with_mask_i16, i16, i16, Int16);
impl_set_with_mask!(set_with_mask_i32, i32, i32, Int32);
impl_set_with_mask!(set_with_mask_i64, i64, i64, Int64);
impl_set_with_mask!(set_with_mask_bool, bool, bool, Boolean);

macro_rules! impl_set_at_idx {
    ($name:ident, $native:ty, $cast:ident, $variant:ident) => {
        #[wasm_bindgen(js_class=Series)]
        impl JsSeries {
            pub fn $name(&self, idx: js_sys::Array, value: Option<$native>) -> JsResult<JsSeries> {
                let ca = self.$cast().map_err(JsPolarsErr::from)?;
                let new = ca
                    .set_at_idx(
                        idx.iter().map(|v| {
                            let n: f64 = js_sys::Number::unchecked_from_js(v).into();
                            n as usize
                        }),
                        value,
                    )
                    .map_err(JsPolarsErr::from)?;

                Ok(JsSeries::new(new.into_series()))
            }
        }
    };
}
// impl_set_at_idx!(set_at_idx_str, &str, utf8, Utf8); //TODO!
impl_set_at_idx!(set_at_idx_f64, f64, f64, Float64);
impl_set_at_idx!(set_at_idx_f32, f32, f32, Float32);
impl_set_at_idx!(set_at_idx_u8, u8, u8, UInt8);
impl_set_at_idx!(set_at_idx_u16, u16, u16, UInt16);
impl_set_at_idx!(set_at_idx_u32, u32, u32, UInt32);
impl_set_at_idx!(set_at_idx_u64, u64, u64, UInt64);
impl_set_at_idx!(set_at_idx_i8, i8, i8, Int8);
impl_set_at_idx!(set_at_idx_i16, i16, i16, Int16);
impl_set_at_idx!(set_at_idx_i32, i32, i32, Int32);
impl_set_at_idx!(set_at_idx_i64, i64, i64, Int64);

macro_rules! impl_new_numeric {
    ($name:ident, $builder:ident, $type:ty) => {
        #[wasm_bindgen(js_class=Series)]
        impl JsSeries {
            pub fn $name(name: &str, values: &js_sys::Array) -> JsResult<JsSeries> {
                let series = $builder::from_iter_options(
                    name,
                    values
                        .iter()
                        .map(|v: JsValue| v.as_f64().map(|n| n as $type)),
                )
                .into_series();
                Ok(JsSeries { series })
            }
        }
    };
}
// impl_new_numeric!(new_i8, Int8Chunked, i8);
impl_new_numeric!(new_i16, Int16Chunked, i16);
impl_new_numeric!(new_i32, Int32Chunked, i32);
impl_new_numeric!(new_u8, UInt8Chunked, u8);
impl_new_numeric!(new_u16, UInt16Chunked, u16);
impl_new_numeric!(new_u32, UInt32Chunked, u32);
impl_new_numeric!(new_f32, Float32Chunked, f32);
// impl_new_numeric!(new_f64, Float64Chunked, f64);

macro_rules! impl_get {
    ($name:ident, $series_variant:ident, $type:ty) => {
        #[wasm_bindgen(js_class=Series)]
        impl JsSeries {
            pub fn $name(&self, index: f64) -> Option<$type> {
                if let Ok(ca) = self.series.$series_variant() {
                    let index = if index < 0.0 {
                        (ca.len() as f64 + index) as usize
                    } else {
                        index as usize
                    };
                    ca.get(index)
                } else {
                    None
                }
            }
        }
    };
}

impl_get!(get_datetime, datetime, i64);
impl_get!(get_duration, duration, i64);

macro_rules! impl_eq_num {
    ($name:ident, $variant:ident, $type:ty) => {
        paste::paste! {
            #[wasm_bindgen(js_class=Series)]
            impl JsSeries {
                pub fn [<get $name>](&self, index: f64) -> Option<$type> {
                    if let Ok(ca) = self.series.$type() {
                        let index = if index < 0.0 {
                            (ca.len() as f64 + index) as usize
                        } else {
                            index as usize
                        };
                        ca.get(index)
                    } else {
                        None
                    }
                }

                pub fn [<add$name>](&self, other: $type) -> JsSeries {
                    JsSeries::new(&self.series + other)
                }
                pub fn [<sub$name>](&self, other: $type) -> JsSeries {
                    JsSeries::new(&self.series - other)
                }
                pub fn [<mul$name>](&self, other: $type) -> JsSeries {
                    JsSeries::new(&self.series * other)
                }
                pub fn [<div$name>](&self, other: $type) -> JsSeries {
                    JsSeries::new(&self.series / other)
                }
                pub fn [<rem$name>](&self, other: $type) -> JsSeries {
                    JsSeries::new(&self.series % other)
                }
                pub fn [<add$name _rhs>](&self, other: $type) -> JsSeries {
                    JsSeries::new(other.add(&self.series))
                }
                pub fn [<sub$name _rhs>](&self, other: $type) -> JsSeries {
                    JsSeries::new(other.sub(&self.series))
                }
                pub fn [<mul$name _rhs>](&self, other: $type) -> JsSeries {
                    JsSeries::new(other.mul(&self.series))
                }
                pub fn [<div$name _rhs>](&self, other: $type) -> JsSeries {
                    JsSeries::new(other.div(&self.series))
                }
                pub fn [<rem$name _rhs>](&self, other: $type) -> JsSeries {
                    JsSeries::new(other.rem(&self.series))
                }
                pub fn [<eq$name>](&self, rhs: $type) -> Self {
                    JsSeries::new(self.series.equal(rhs).into_series())
                }
                pub fn [<neq$name>](&self, rhs: $type) -> Self {
                    JsSeries::new(self.series.not_equal(rhs).into_series())
                }
                pub fn [<gt$name>](&self, rhs: $type) -> Self {
                    JsSeries::new(self.series.gt(rhs).into_series())
                }
                pub fn [<lt$name>](&self, rhs: $type) -> Self {
                    JsSeries::new(self.series.lt(rhs).into_series())
                }
                pub fn [<gt_eq$name>](&self, rhs: $type) -> Self {
                    JsSeries::new(self.series.gt_eq(rhs).into_series())
                }
                pub fn [<lt_eq$name>](&self, rhs: $type) -> Self {
                    JsSeries::new(self.series.lt_eq(rhs).into_series())
                }
            }
        }
    };
}

impl_eq_num!(_u8, u8, u8);
impl_eq_num!(_u16, u16, u16);
impl_eq_num!(_u32, u32, u32);
impl_eq_num!(_u64, u64, u64);
impl_eq_num!(_i8, i8, i8);
impl_eq_num!(_i16, i16, i16);
impl_eq_num!(_i32, i32, i32);
impl_eq_num!(_i64, i64, i64);
impl_eq_num!(_f32, f32, f32);
impl_eq_num!(_f64, f64, f64);

pub fn reinterpret(s: &Series, signed: bool) -> polars::prelude::Result<Series> {
    match (s.dtype(), signed) {
        (DataType::UInt64, true) => {
            let ca = s.u64().unwrap();
            Ok(ca.reinterpret_signed().into_series())
        }
        (DataType::UInt64, false) => Ok(s.clone()),
        (DataType::Int64, false) => {
            let ca = s.i64().unwrap();
            Ok(ca.reinterpret_unsigned().into_series())
        }
        (DataType::Int64, true) => Ok(s.clone()),
        _ => Err(PolarsError::ComputeError(
            "reinterpret is only allowed for 64bit integers dtype, use cast otherwise".into(),
        )),
    }
}
pub(crate) fn to_series_collection(iter: js_sys::Iterator) -> Vec<Series> {
    let cols: Vec<Series> = iter
        .into_iter()
        .map(|jsv| {
            let jsv = jsv.unwrap();
            let key = JsValue::from_str("ptr");
            let ptr = js_sys::Reflect::get(&jsv, &key).unwrap();
            let n: f64 = js_sys::Number::unchecked_from_js(ptr).into();
            let ser: JsSeries = unsafe { JsSeries::from_abi(n as u32) };
            ser.series
        })
        .collect();
    cols
}

pub(crate) fn to_jsseries_collection(s: Vec<Series>) -> Vec<u32> {
    use wasm_bindgen::convert::IntoWasmAbi;
    let s: Vec<u32> = s
        .into_iter()
        .map(move |series| {
            let js_ser = JsSeries { series };

            js_ser.into_abi()
        })
        .collect();

    s
    // todo!()
}
