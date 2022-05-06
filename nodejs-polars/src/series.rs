use crate::prelude::*;
use crate::utils::reinterpret;
use polars_core::series::ops::NullBehavior;
use polars_core::utils::CustomIterTools;

#[napi]
#[repr(transparent)]
#[derive(Clone)]
pub struct JsSeries {
    pub(crate) series: Series,
}

impl JsSeries {
    pub(crate) fn new(series: Series) -> Self {
        JsSeries { series }
    }
}
impl From<Series> for JsSeries {
    fn from(s: Series) -> Self {
        JsSeries::new(s)
    }
}

#[napi]
impl JsSeries {
    #[napi]
    pub fn to_js(&self, env: Env) -> napi::Result<napi::JsUnknown> {
        env.to_js_value(&self.series)
    }

    #[napi]
    pub fn serialize(&self, format: String) -> napi::Result<Buffer> {
        let buf = match format.as_ref() {
            "bincode" => bincode::serialize(&self.series)
                .map_err(|err| napi::Error::from_reason(format!("{:?}", err)))?,
            "json" => serde_json::to_vec(&self.series)
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
    pub fn deserialize(buf: Buffer, format: String) -> napi::Result<JsSeries> {
        let series: Series = match format.as_ref() {
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
        Ok(series.into())
    }
    //
    // FACTORIES
    //
    #[napi(factory)]
    pub fn new_int_8_array(name: String, arr: Int8Array) -> JsSeries {
        Series::new(&name, arr).into()
    }
    #[napi(factory)]
    pub fn new_uint8_array(name: String, arr: Uint8Array) -> JsSeries {
        Series::new(&name, arr).into()
    }
    #[napi(factory)]
    pub fn new_uint8_clamped_array(name: String, arr: Uint8ClampedArray) -> JsSeries {
        Series::new(&name, arr).into()
    }
    #[napi(factory)]
    pub fn new_int16_array(name: String, arr: Int16Array) -> JsSeries {
        Series::new(&name, arr).into()
    }
    #[napi(factory)]
    pub fn new_uint16_array(name: String, arr: Uint16Array) -> JsSeries {
        Series::new(&name, arr).into()
    }
    #[napi(factory)]
    pub fn new_int32_array(name: String, arr: Int32Array) -> JsSeries {
        Series::new(&name, arr).into()
    }
    #[napi(factory)]
    pub fn new_uint32_array(name: String, arr: Uint32Array) -> JsSeries {
        Series::new(&name, arr).into()
    }
    #[napi(factory)]
    pub fn new_float32_array(name: String, arr: Float32Array) -> JsSeries {
        Series::new(&name, arr).into()
    }
    #[napi(factory)]
    pub fn new_float64_array(name: String, arr: Float64Array) -> JsSeries {
        Series::new(&name, arr).into()
    }
    #[napi(factory)]
    pub fn new_bigint64_array(name: String, arr: BigInt64Array) -> JsSeries {
        Series::new(&name, arr).into()
    }
    #[napi(factory)]
    pub fn new_biguint64_array(name: String, arr: BigUint64Array) -> JsSeries {
        Series::new(&name, arr).into()
    }
    #[napi(factory)]
    pub fn new_opt_str(name: String, val: Wrap<Utf8Chunked>) -> JsSeries {
        let mut s = val.0.into_series();
        s.rename(&name);
        JsSeries::new(s)
    }
    #[napi(factory)]
    pub fn new_opt_bool(name: String, val: Wrap<BooleanChunked>, _strict: bool) -> JsSeries {
        let mut s = val.0.into_series();
        s.rename(&name);
        JsSeries::new(s)
    }
    #[napi(factory)]
    pub fn new_opt_i32(name: String, val: Wrap<Int32Chunked>, _strict: bool) -> JsSeries {
        let mut s = val.0.into_series();
        s.rename(&name);
        JsSeries::new(s)
    }
    #[napi(factory)]
    pub fn new_opt_i64(name: String, val: Wrap<Int64Chunked>, _strict: bool) -> JsSeries {
        let mut s = val.0.into_series();
        s.rename(&name);
        JsSeries::new(s)
    }
    #[napi(factory)]
    pub fn new_opt_u64(name: String, val: Wrap<UInt64Chunked>, _strict: bool) -> JsSeries {
        let mut s = val.0.into_series();
        s.rename(&name);
        JsSeries::new(s)
    }
    #[napi(factory)]
    pub fn new_opt_u32(name: String, val: Wrap<UInt32Chunked>, _strict: bool) -> JsSeries {
        let mut s = val.0.into_series();
        s.rename(&name);
        JsSeries::new(s)
    }
    #[napi(factory)]
    pub fn new_opt_f32(name: String, val: Wrap<Float32Chunked>, _strict: bool) -> JsSeries {
        let mut s = val.0.into_series();
        s.rename(&name);
        JsSeries::new(s)
    }

    #[napi(factory)]
    pub fn new_opt_f64(name: String, val: Wrap<Float64Chunked>, _strict: bool) -> JsSeries {
        let mut s = val.0.into_series();
        s.rename(&name);
        JsSeries::new(s)
    }

    #[napi(factory)]
    pub fn new_opt_date(
        name: String,
        values: Vec<napi::JsUnknown>,
        strict: Option<bool>,
    ) -> napi::Result<JsSeries> {
        let len = values.len();
        let mut builder = PrimitiveChunkedBuilder::<Int64Type>::new(&name, len);
        for item in values.into_iter() {
            match item.get_type()? {
                ValueType::Object => {
                    let obj: &napi::JsObject = unsafe { &item.cast() };
                    if obj.is_date()? {
                        let d: &napi::JsDate = unsafe { &item.cast() };
                        match d.value_of() {
                            Ok(v) => builder.append_value(v as i64),
                            Err(e) => {
                                if strict.unwrap_or(false) {
                                    return Err(e);
                                }
                                builder.append_null()
                            }
                        }
                    }
                }
                ValueType::Null | ValueType::Undefined => builder.append_null(),
                _ => {
                    return Err(JsPolarsErr::Other("Series must be of date type".to_owned()).into())
                }
            }
        }
        let ca: ChunkedArray<Int64Type> = builder.finish();
        Ok(ca
            .into_datetime(TimeUnit::Milliseconds, None)
            .into_series()
            .into())
    }
    #[napi(factory)]
    pub fn new_list(name: String, values: Array, dtype: Wrap<DataType>) -> napi::Result<JsSeries> {
        use crate::list_construction::js_arr_to_list;
        let s = js_arr_to_list(&name, &values, &dtype.0)?;
        Ok(s.into())
    }

    #[napi(factory)]
    pub fn repeat(
        name: String,
        val: JsAnyValue,
        n: i64,
        dtype: Wrap<DataType>,
    ) -> napi::Result<JsSeries> {
        let s: JsSeries = match dtype.0 {
            DataType::Utf8 => {
                let val: String = val.try_into()?;
                let mut ca: Utf8Chunked = (0..n).map(|_| val.clone()).collect_trusted();
                ca.rename(&name);
                ca.into_series().into()
            }
            DataType::Int64 => {
                let val: i64 = val.try_into()?;
                let mut ca: NoNull<Int64Chunked> = (0..n).map(|_| val).collect_trusted();
                ca.rename(&name);
                ca.into_inner().into_series().into()
            }
            DataType::Float64 => {
                let val: f64 = val.try_into()?;
                let mut ca: NoNull<Float64Chunked> = (0..n).map(|_| val).collect_trusted();
                ca.rename(&name);
                ca.into_inner().into_series().into()
            }
            _ => todo!(),
        };
        Ok(s)
    }
    //
    // GETTERS
    //
    #[napi(getter)]
    pub fn dtype(&self) -> JsDataType {
        self.series.dtype().into()
    }

    #[napi(getter)]
    pub fn __inner__(&self) -> External<Series> {
        External::new(self.series.clone())
    }

    #[napi(getter)]
    pub fn inner_dtype(&self) -> Option<JsDataType> {
        self.series.dtype().inner_dtype().map(|dt| dt.into())
    }
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.series.name().to_owned()
    }
    #[napi]
    pub fn to_string(&self) -> String {
        format!("{}", self.series)
    }
    #[napi]
    pub fn get_fmt(&self, index: f64) -> String {
        format!("{}", self.series.get(index as usize))
    }
    #[napi]
    pub fn estimated_size(&self) -> i64 {
        self.series.estimated_size() as i64
    }

    #[napi]
    pub fn rechunk(&mut self, in_place: bool) -> Option<JsSeries> {
        let series = self.series.rechunk();
        if in_place {
            self.series = series;
            None
        } else {
            Some(series.into())
        }
    }
    #[napi]
    pub fn get_idx(&self, idx: i64) -> Wrap<AnyValue> {
        Wrap(self.series.get(idx as usize))
    }
    #[napi]
    pub fn bitand(&self, other: &JsSeries) -> napi::Result<JsSeries> {
        let out = self
            .series
            .bitand(&other.series)
            .map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }
    #[napi]
    pub fn bitor(&self, other: &JsSeries) -> napi::Result<JsSeries> {
        let out = self
            .series
            .bitor(&other.series)
            .map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }
    #[napi]
    pub fn bitxor(&self, other: &JsSeries) -> napi::Result<JsSeries> {
        let out = self
            .series
            .bitxor(&other.series)
            .map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }
    #[napi]
    pub fn cumsum(&self, reverse: Option<bool>) -> JsSeries {
        let reverse = reverse.unwrap_or(false);
        self.series.cumsum(reverse).into()
    }
    #[napi]
    pub fn cummax(&self, reverse: Option<bool>) -> JsSeries {
        let reverse = reverse.unwrap_or(false);
        self.series.cummax(reverse).into()
    }
    #[napi]
    pub fn cummin(&self, reverse: Option<bool>) -> JsSeries {
        let reverse = reverse.unwrap_or(false);
        self.series.cummin(reverse).into()
    }
    #[napi]
    pub fn cumprod(&self, reverse: Option<bool>) -> JsSeries {
        let reverse = reverse.unwrap_or(false);
        self.series.cumprod(reverse).into()
    }
    #[napi]
    pub fn chunk_lengths(&self) -> Vec<u32> {
        self.series.chunk_lengths().map(|i| i as u32).collect()
    }

    #[napi]
    pub fn rename(&mut self, name: String) {
        self.series.rename(&name);
    }

    #[napi]
    pub fn mean(&self) -> Option<f64> {
        match self.series.dtype() {
            DataType::Boolean => {
                let s = self.series.cast(&DataType::UInt8).unwrap();
                s.mean()
            }
            _ => self.series.mean(),
        }
    }

    #[napi]
    pub fn max(&self) -> Option<Either3<f64, bool, i64>> {
        match self.series.dtype() {
            DataType::Float32 | DataType::Float64 => self.series.max::<f64>().map(Either3::A),
            DataType::Boolean => self.series.max::<u32>().map(|v| v == 1).map(Either3::B),
            _ => self.series.max::<i64>().map(Either3::C),
        }
    }
    #[napi]
    pub fn min(&self) -> Option<Either3<f64, bool, i64>> {
        match self.series.dtype() {
            DataType::Float32 | DataType::Float64 => self.series.min::<f64>().map(Either3::A),
            DataType::Boolean => self.series.min::<u32>().map(|v| v == 1).map(Either3::B),
            _ => self.series.min::<i64>().map(Either3::C),
        }
    }
    #[napi]
    pub fn sum(&self) -> Option<Either3<f64, bool, i64>> {
        match self.series.dtype() {
            DataType::Float32 | DataType::Float64 => self.series.sum::<f64>().map(Either3::A),
            DataType::Boolean => self.series.sum::<u32>().map(|v| v == 1).map(Either3::B),
            _ => self.series.sum::<i64>().map(Either3::C),
        }
    }
    #[napi]
    pub fn n_chunks(&self) -> u32 {
        self.series.n_chunks() as u32
    }
    #[napi]
    pub fn limit(&self, num_elements: f64) -> JsSeries {
        self.series.limit(num_elements as usize).into()
    }
    #[napi]
    pub fn slice(&self, offset: i64, length: f64) -> JsSeries {
        self.series.slice(offset, length as usize).into()
    }
    #[napi]
    pub fn append(&mut self, other: &JsSeries) -> napi::Result<()> {
        self.series
            .append(&other.series)
            .map_err(JsPolarsErr::from)?;
        Ok(())
    }
    #[napi]
    pub fn extend(&mut self, other: &JsSeries) -> napi::Result<()> {
        self.series
            .extend(&other.series)
            .map_err(JsPolarsErr::from)?;
        Ok(())
    }
    #[napi]
    pub fn filter(&self, filter: &JsSeries) -> napi::Result<JsSeries> {
        let filter_series = &filter.series;
        if let Ok(ca) = filter_series.bool() {
            let series = self.series.filter(ca).map_err(JsPolarsErr::from)?;
            Ok(JsSeries { series })
        } else {
            let err = napi::Error::from_reason("Expected a boolean mask".to_owned());
            Err(err)
        }
    }
    #[napi]
    pub fn add(&self, other: &JsSeries) -> JsSeries {
        (&self.series + &other.series).into()
    }
    #[napi]
    pub fn sub(&self, other: &JsSeries) -> JsSeries {
        (&self.series - &other.series).into()
    }
    #[napi]
    pub fn mul(&self, other: &JsSeries) -> JsSeries {
        (&self.series * &other.series).into()
    }
    #[napi]
    pub fn div(&self, other: &JsSeries) -> JsSeries {
        (&self.series / &other.series).into()
    }
    #[napi]
    pub fn rem(&self, other: &JsSeries) -> JsSeries {
        (&self.series % &other.series).into()
    }
    #[napi]
    pub fn head(&self, length: Option<i64>) -> JsSeries {
        (self.series.head(length.map(|l| l as usize))).into()
    }
    #[napi]
    pub fn tail(&self, length: Option<i64>) -> JsSeries {
        (self.series.tail(length.map(|l| l as usize))).into()
    }
    #[napi]
    pub fn sort(&self, reverse: Option<bool>) -> JsSeries {
        let reverse = reverse.unwrap_or(false);
        self.series.sort(reverse).into()
    }
    #[napi]
    pub fn argsort(&self, reverse: bool, nulls_last: bool) -> JsSeries {
        self.series
            .argsort(SortOptions {
                descending: reverse,
                nulls_last,
            })
            .into_series()
            .into()
    }
    #[napi]
    pub fn unique(&self) -> napi::Result<JsSeries> {
        let unique = self.series.unique().map_err(JsPolarsErr::from)?;
        Ok(unique.into())
    }
    #[napi]
    pub fn unique_stable(&self) -> napi::Result<JsSeries> {
        let unique = self.series.unique_stable().map_err(JsPolarsErr::from)?;
        Ok(unique.into())
    }
    #[napi]
    pub fn value_counts(&self) -> napi::Result<Object> {
        todo!()
    }
    #[napi]
    pub fn arg_unique(&self) -> napi::Result<JsSeries> {
        let arg_unique = self.series.arg_unique().map_err(JsPolarsErr::from)?;
        Ok(arg_unique.into_series().into())
    }
    #[napi]
    pub fn arg_min(&self) -> Option<i64> {
        self.series.arg_min().map(|v| v as i64)
    }
    #[napi]
    pub fn arg_max(&self) -> Option<i64> {
        self.series.arg_max().map(|v| v as i64)
    }
    #[napi]
    pub fn take(&self, indices: Vec<u32>) -> napi::Result<JsSeries> {
        let indices = UInt32Chunked::from_vec("", indices);

        let take = self.series.take(&indices).map_err(JsPolarsErr::from)?;
        Ok(JsSeries::new(take))
    }
    #[napi]
    pub fn take_with_series(&self, indices: &JsSeries) -> napi::Result<JsSeries> {
        let idx = indices.series.u32().map_err(JsPolarsErr::from)?;
        let take = self.series.take(idx).map_err(JsPolarsErr::from)?;
        Ok(JsSeries::new(take))
    }

    #[napi]
    pub fn null_count(&self) -> napi::Result<i64> {
        Ok(self.series.null_count() as i64)
    }

    #[napi]
    pub fn has_validity(&self) -> bool {
        self.series.has_validity()
    }

    #[napi]
    pub fn is_null(&self) -> JsSeries {
        Self::new(self.series.is_null().into_series())
    }

    #[napi]
    pub fn is_not_null(&self) -> JsSeries {
        Self::new(self.series.is_not_null().into_series())
    }

    #[napi]
    pub fn is_not_nan(&self) -> napi::Result<JsSeries> {
        let ca = self.series.is_not_nan().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }

    #[napi]
    pub fn is_nan(&self) -> napi::Result<JsSeries> {
        let ca = self.series.is_nan().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }

    #[napi]
    pub fn is_finite(&self) -> napi::Result<JsSeries> {
        let ca = self.series.is_finite().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }

    #[napi]
    pub fn is_infinite(&self) -> napi::Result<JsSeries> {
        let ca = self.series.is_infinite().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }

    #[napi]
    pub fn is_unique(&self) -> napi::Result<JsSeries> {
        let ca = self.series.is_unique().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }

    #[napi]
    pub fn arg_true(&self) -> napi::Result<JsSeries> {
        let ca = self.series.arg_true().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }
    #[napi]
    pub fn sample_n(
        &self,
        n: u32,
        with_replacement: bool,
        seed: Option<Wrap<u64>>,
    ) -> napi::Result<Self> {
        // Safety:
        // Wrap is transparent.
        todo!()
        // let seed: Option<u64> = unsafe { std::mem::transmute(seed) };
        // let s = self
        //     .series
        //     .sample_n(n as usize, with_replacement, seed)
        //     .map_err(JsPolarsErr::from)?;
        // Ok(s.into())
    }

    #[napi]
    pub fn sample_frac(
        &self,
        frac: f64,
        with_replacement: bool,
        seed: Option<Wrap<u64>>,
    ) -> napi::Result<Self> {
        todo!()
        // Safety:
        // Wrap is transparent.
        // let seed: Option<u64> = unsafe { std::mem::transmute(seed) };
        // let s = self
        //     .series
        //     .sample_frac(frac, with_replacement, seed)
        //     .map_err(JsPolarsErr::from)?;
        // Ok(s.into())
    }
    #[napi]
    pub fn is_duplicated(&self) -> napi::Result<JsSeries> {
        let ca = self.series.is_duplicated().map_err(JsPolarsErr::from)?;
        Ok(ca.into_series().into())
    }
    #[napi]
    pub fn explode(&self) -> napi::Result<JsSeries> {
        let s = self.series.explode().map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }
    #[napi]
    pub fn take_every(&self, n: i64) -> JsSeries {
        let s = self.series.take_every(n as usize);
        s.into()
    }
    #[napi]
    pub fn series_equal(&self, other: &JsSeries, null_equal: bool, strict: bool) -> bool {
        if strict {
            self.series.eq(&other.series)
        } else if null_equal {
            self.series.series_equal_missing(&other.series)
        } else {
            self.series.series_equal(&other.series)
        }
    }
    #[napi]
    pub fn eq(&self, rhs: &JsSeries) -> napi::Result<JsSeries> {
        Ok(Self::new(
            self.series
                .equal(&rhs.series)
                .map_err(JsPolarsErr::from)?
                .into_series(),
        ))
    }

    #[napi]
    pub fn neq(&self, rhs: &JsSeries) -> napi::Result<JsSeries> {
        Ok(Self::new(
            self.series
                .not_equal(&rhs.series)
                .map_err(JsPolarsErr::from)?
                .into_series(),
        ))
    }

    #[napi]
    pub fn gt(&self, rhs: &JsSeries) -> napi::Result<JsSeries> {
        Ok(Self::new(
            self.series
                .gt(&rhs.series)
                .map_err(JsPolarsErr::from)?
                .into_series(),
        ))
    }

    #[napi]
    pub fn gt_eq(&self, rhs: &JsSeries) -> napi::Result<JsSeries> {
        Ok(Self::new(
            self.series
                .gt_eq(&rhs.series)
                .map_err(JsPolarsErr::from)?
                .into_series(),
        ))
    }

    #[napi]
    pub fn lt(&self, rhs: &JsSeries) -> napi::Result<JsSeries> {
        Ok(Self::new(
            self.series
                .lt(&rhs.series)
                .map_err(JsPolarsErr::from)?
                .into_series(),
        ))
    }

    #[napi]
    pub fn lt_eq(&self, rhs: &JsSeries) -> napi::Result<JsSeries> {
        Ok(Self::new(
            self.series
                .lt_eq(&rhs.series)
                .map_err(JsPolarsErr::from)?
                .into_series(),
        ))
    }

    #[napi]
    pub fn _not(&self) -> napi::Result<JsSeries> {
        let bool = self.series.bool().map_err(JsPolarsErr::from)?;
        Ok((!bool).into_series().into())
    }
    #[napi]
    pub fn as_str(&self) -> napi::Result<String> {
        Ok(format!("{:?}", self.series))
    }
    #[napi]
    pub fn len(&self) -> i64 {
        self.series.len() as i64
    }
    #[napi]
    pub fn to_physical(&self) -> JsSeries {
        let s = self.series.to_physical_repr().into_owned();
        s.into()
    }

    #[napi]
    pub fn to_typed_array(&self) -> TypedArrayBuffer {
        let series = &self.series;
        series.into()
    }
    #[napi]
    pub fn to_array(&self) -> Wrap<&Series> {
        Wrap(&self.series)
    }

    #[napi]
    pub fn median(&self) -> Option<f64> {
        match self.series.dtype() {
            DataType::Boolean => {
                let s = self.series.cast(&DataType::UInt8).unwrap();
                s.median()
            }
            _ => self.series.median(),
        }
    }
    #[napi]
    pub fn quantile(
        &self,
        quantile: f64,
        interpolation: Wrap<QuantileInterpolOptions>,
    ) -> JsAnyValue {
        let v = self
            .series
            .quantile_as_series(quantile, interpolation.0)
            .expect("invalid quantile");
        let v = v.get(0);
        v.into()
    }
    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    pub fn as_single_ptr(&mut self) -> napi::Result<usize> {
        let ptr = self.series.as_single_ptr().map_err(JsPolarsErr::from)?;
        Ok(ptr)
    }

    #[napi]
    pub fn drop_nulls(&self) -> JsSeries {
        self.series.drop_nulls().into()
    }

    #[napi]
    pub fn fill_null(&self, strategy: Wrap<FillNullStrategy>) -> napi::Result<JsSeries> {
        let series = self
            .series
            .fill_null(strategy.0)
            .map_err(JsPolarsErr::from)?;
        Ok(JsSeries::new(series))
    }

    #[napi]
    pub fn is_in(&self, other: &JsSeries) -> napi::Result<JsSeries> {
        let out = self
            .series
            .is_in(&other.series)
            .map_err(JsPolarsErr::from)?;
        Ok(out.into_series().into())
    }

    #[napi]
    pub fn clone(&self) -> JsSeries {
        JsSeries::new(self.series.clone())
    }

    #[napi]
    pub fn shift(&self, periods: i64) -> JsSeries {
        let s = self.series.shift(periods);
        JsSeries::new(s)
    }
    #[napi]
    pub fn zip_with(&self, mask: &JsSeries, other: &JsSeries) -> napi::Result<JsSeries> {
        let mask = mask.series.bool().map_err(JsPolarsErr::from)?;
        let s = self
            .series
            .zip_with(mask, &other.series)
            .map_err(JsPolarsErr::from)?;
        Ok(JsSeries::new(s))
    }

    // Struct namespace
    #[napi]
    pub fn struct_to_frame(&self) -> napi::Result<crate::dataframe::JsDataFrame> {
        let ca = self.series.struct_().map_err(JsPolarsErr::from)?;
        let df: DataFrame = ca.clone().into();
        Ok(df.into())
    }

    #[napi]
    pub fn struct_fields(&self) -> napi::Result<Vec<&str>> {
        let ca = self.series.struct_().map_err(JsPolarsErr::from)?;
        Ok(ca.fields().iter().map(|s| s.name()).collect())
    }
    // String Namespace

    #[napi]
    pub fn str_lengths(&self) -> napi::Result<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca.str_lengths().into_series();
        Ok(JsSeries::new(s))
    }

    #[napi]
    pub fn str_contains(&self, pat: String) -> napi::Result<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca.contains(&pat).map_err(JsPolarsErr::from)?.into_series();
        Ok(s.into())
    }

    #[napi]
    pub fn str_json_path_match(&self, pat: String) -> napi::Result<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca
            .json_path_match(&pat)
            .map_err(JsPolarsErr::from)?
            .into_series();
        Ok(s.into())
    }

    #[napi]
    pub fn str_extract(&self, pat: String, group_index: i64) -> napi::Result<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca
            .extract(&pat, group_index as usize)
            .map_err(JsPolarsErr::from)?
            .into_series();
        Ok(s.into())
    }

    #[napi]
    pub fn str_replace(&self, pat: String, val: String) -> napi::Result<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca
            .replace(&pat, &val)
            .map_err(JsPolarsErr::from)?
            .into_series();
        Ok(s.into())
    }

    #[napi]
    pub fn str_replace_all(&self, pat: String, val: String) -> napi::Result<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca
            .replace_all(&pat, &val)
            .map_err(JsPolarsErr::from)?
            .into_series();
        Ok(s.into())
    }

    #[napi]
    pub fn str_to_uppercase(&self) -> napi::Result<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca.to_uppercase().into_series();
        Ok(s.into())
    }

    #[napi]
    pub fn str_to_lowercase(&self) -> napi::Result<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca.to_lowercase().into_series();
        Ok(s.into())
    }

    #[napi]
    pub fn str_slice(&self, start: i64, length: Option<i64>) -> napi::Result<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca
            .str_slice(start, length.map(|l| l as u64))
            .map_err(JsPolarsErr::from)?
            .into_series();
        Ok(s.into())
    }

    #[napi]
    pub fn str_hex_encode(&self) -> napi::Result<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca.hex_encode().into_series();
        Ok(s.into())
    }

    #[napi]
    pub fn str_hex_decode(&self, strict: Option<bool>) -> napi::Result<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca
            .hex_decode(strict)
            .map_err(JsPolarsErr::from)?
            .into_series();

        Ok(s.into())
    }

    #[napi]
    pub fn str_base64_encode(&self) -> napi::Result<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca.base64_encode().into_series();
        Ok(s.into())
    }

    #[napi]
    pub fn str_base64_decode(&self, strict: Option<bool>) -> napi::Result<JsSeries> {
        let ca = self.series.utf8().map_err(JsPolarsErr::from)?;
        let s = ca
            .base64_decode(strict)
            .map_err(JsPolarsErr::from)?
            .into_series();
        Ok(s.into())
    }

    #[napi]
    pub fn strftime(&self, fmt: String) -> napi::Result<JsSeries> {
        let s = self.series.strftime(&fmt).map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }
    #[napi]
    pub fn arr_lengths(&self) -> napi::Result<JsSeries> {
        let ca = self.series.list().map_err(JsPolarsErr::from)?;
        let s = ca.lst_lengths().into_series();
        Ok(JsSeries::new(s))
    }
    // #[napi]
    // pub fn timestamp(&self, tu: Wrap<TimeUnit>) -> napi::Result<JsSeries> {
    //   let ca = self.series.timestamp(tu.0).map_err(JsPolarsErr::from)?;
    //   Ok(ca.into_series().into())
    // }
    #[napi]
    pub fn to_dummies(&self) -> napi::Result<JsSeries> {
        todo!()
        // let df = self.series.to_dummies().map_err(JsPolarsErr::from)?;
        // Ok(df.into())
    }

    #[napi]
    pub fn get_list(&self, index: i64) -> Option<JsSeries> {
        if let Ok(ca) = &self.series.list() {
            let s = ca.get(index as usize);
            s.map(|s| s.into())
        } else {
            None
        }
    }

    #[napi]
    pub fn rolling_sum(&self, options: JsRollingOptions) -> napi::Result<JsSeries> {
        let s = self
            .series
            .rolling_sum(options.into())
            .map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }
    #[napi]
    pub fn rolling_mean(&self, options: JsRollingOptions) -> napi::Result<JsSeries> {
        let s = self
            .series
            .rolling_mean(options.into())
            .map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }
    #[napi]
    pub fn rolling_median(&self, options: JsRollingOptions) -> napi::Result<JsSeries> {
        let s = self
            .series
            .rolling_median(options.into())
            .map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }
    #[napi]
    pub fn rolling_max(&self, options: JsRollingOptions) -> napi::Result<JsSeries> {
        let s = self
            .series
            .rolling_max(options.into())
            .map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }
    #[napi]
    pub fn rolling_min(&self, options: JsRollingOptions) -> napi::Result<JsSeries> {
        let s = self
            .series
            .rolling_min(options.into())
            .map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }
    #[napi]
    pub fn rolling_var(&self, options: JsRollingOptions) -> napi::Result<JsSeries> {
        let s = self
            .series
            .rolling_var(options.into())
            .map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }
    #[napi]
    pub fn rolling_std(&self, options: JsRollingOptions) -> napi::Result<JsSeries> {
        let s = self
            .series
            .rolling_std(options.into())
            .map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }
    #[napi]
    pub fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: Wrap<QuantileInterpolOptions>,
        options: JsRollingOptions,
    ) -> napi::Result<JsSeries> {
        let interpol = interpolation.0;
        let s = self
            .series
            .rolling_quantile(quantile, interpol, options.into())
            .map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }
    #[napi]
    pub fn year(&self) -> napi::Result<JsSeries> {
        let s = self.series.year().map_err(JsPolarsErr::from)?;
        Ok(s.into_series().into())
    }
    #[napi]
    pub fn month(&self) -> napi::Result<JsSeries> {
        let s = self.series.month().map_err(JsPolarsErr::from)?;
        Ok(s.into_series().into())
    }
    #[napi]
    pub fn weekday(&self) -> napi::Result<JsSeries> {
        let s = self.series.weekday().map_err(JsPolarsErr::from)?;
        Ok(s.into_series().into())
    }
    #[napi]
    pub fn week(&self) -> napi::Result<JsSeries> {
        let s = self.series.week().map_err(JsPolarsErr::from)?;
        Ok(s.into_series().into())
    }
    #[napi]
    pub fn day(&self) -> napi::Result<JsSeries> {
        let s = self.series.day().map_err(JsPolarsErr::from)?;
        Ok(s.into_series().into())
    }
    #[napi]
    pub fn ordinal_day(&self) -> napi::Result<JsSeries> {
        let s = self.series.ordinal_day().map_err(JsPolarsErr::from)?;
        Ok(s.into_series().into())
    }
    #[napi]
    pub fn hour(&self) -> napi::Result<JsSeries> {
        let s = self.series.hour().map_err(JsPolarsErr::from)?;
        Ok(s.into_series().into())
    }
    #[napi]
    pub fn minute(&self) -> napi::Result<JsSeries> {
        let s = self.series.minute().map_err(JsPolarsErr::from)?;
        Ok(s.into_series().into())
    }
    #[napi]
    pub fn second(&self) -> napi::Result<JsSeries> {
        let s = self.series.second().map_err(JsPolarsErr::from)?;
        Ok(s.into_series().into())
    }
    #[napi]
    pub fn nanosecond(&self) -> napi::Result<JsSeries> {
        let s = self.series.nanosecond().map_err(JsPolarsErr::from)?;
        Ok(s.into_series().into())
    }
    #[napi]
    pub fn dt_epoch_seconds(&self) -> napi::Result<JsSeries> {
        let ms = self
            .series
            .timestamp(TimeUnit::Milliseconds)
            .map_err(JsPolarsErr::from)?;
        Ok((ms / 1000).into_series().into())
    }
    #[napi]
    pub fn peak_max(&self) -> JsSeries {
        self.series.peak_max().into_series().into()
    }
    #[napi]
    pub fn peak_min(&self) -> JsSeries {
        self.series.peak_min().into_series().into()
    }

    #[napi]
    pub fn n_unique(&self) -> napi::Result<i64> {
        let n = self.series.n_unique().map_err(JsPolarsErr::from)?;
        Ok(n as i64)
    }

    #[napi]
    pub fn is_first(&self) -> napi::Result<JsSeries> {
        let out = self
            .series
            .is_first()
            .map_err(JsPolarsErr::from)?
            .into_series();
        Ok(out.into())
    }

    #[napi]
    pub fn round(&self, decimals: u32) -> napi::Result<JsSeries> {
        let s = self.series.round(decimals).map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }

    #[napi]
    pub fn floor(&self) -> napi::Result<JsSeries> {
        let s = self.series.floor().map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }

    #[napi]
    pub fn ceil(&self) -> napi::Result<JsSeries> {
        let s = self.series.ceil().map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }
    #[napi]
    pub fn shrink_to_fit(&mut self) {
        self.series.shrink_to_fit();
    }

    #[napi]
    pub fn dot(&self, other: &JsSeries) -> Option<f64> {
        self.series.dot(&other.series)
    }

    #[napi]
    pub fn hash(&self, k0: Wrap<u64>, k1: Wrap<u64>, k2: Wrap<u64>, k3: Wrap<u64>) -> JsSeries {
        let hb = ahash::RandomState::with_seeds(k0.0, k1.0, k2.0, k3.0);
        self.series.hash(hb).into_series().into()
    }
    #[napi]
    pub fn reinterpret(&self, signed: bool) -> napi::Result<JsSeries> {
        let s = reinterpret(&self.series, signed).map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }

    #[napi]
    pub fn mode(&self) -> napi::Result<JsSeries> {
        let s = self.series.mode().map_err(JsPolarsErr::from)?;
        Ok(s.into())
    }

    #[napi]
    pub fn interpolate(&self) -> JsSeries {
        let s = self.series.interpolate();
        s.into()
    }

    #[napi]
    pub fn rank(&self, method: Wrap<RankMethod>, reverse: Option<bool>) -> napi::Result<JsSeries> {
        let reverse = reverse.unwrap_or(false);

        let options = RankOptions {
            method: method.0,
            descending: reverse,
        };
        Ok(self.series.rank(options).into())
    }
    #[napi]
    pub fn diff(&self, n: i64, null_behavior: Wrap<NullBehavior>) -> napi::Result<JsSeries> {
        Ok(self.series.diff(n as usize, null_behavior.0).into())
    }

    #[napi]
    pub fn skew(&self, bias: bool) -> napi::Result<Option<f64>> {
        let out = self.series.skew(bias).map_err(JsPolarsErr::from)?;
        Ok(out)
    }

    #[napi]
    pub fn kurtosis(&self, fisher: bool, bias: bool) -> napi::Result<Option<f64>> {
        let out = self
            .series
            .kurtosis(fisher, bias)
            .map_err(JsPolarsErr::from)?;
        Ok(out)
    }

    #[napi]
    pub fn cast(&self, dtype: Wrap<DataType>, strict: Option<bool>) -> napi::Result<JsSeries> {
        let strict = strict.unwrap_or(false);
        let dtype = dtype.0;
        let out = if strict {
            self.series.strict_cast(&dtype)
        } else {
            self.series.cast(&dtype)
        };
        let out = out.map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }

    #[napi]
    pub fn abs(&self) -> napi::Result<JsSeries> {
        let out = self.series.abs().map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }

    #[napi]
    pub fn reshape(&self, dims: Vec<i64>) -> napi::Result<JsSeries> {
        let out = self.series.reshape(&dims).map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }

    #[napi]
    pub fn shuffle(&self, seed: Wrap<u64>) -> JsSeries {
        self.series.shuffle(seed.0).into()
    }

    #[napi]
    pub fn extend_constant(&self, value: Wrap<AnyValue>, n: i64) -> napi::Result<JsSeries> {
        let out = self
            .series
            .extend_constant(value.0, n as usize)
            .map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }

    #[napi]
    pub fn time_unit(&self) -> Option<String> {
        if let DataType::Datetime(tu, _) | DataType::Duration(tu) = self.series.dtype() {
            Some(
                match tu {
                    TimeUnit::Nanoseconds => "ns",
                    TimeUnit::Microseconds => "us",
                    TimeUnit::Milliseconds => "ms",
                }
                .to_owned(),
            )
        } else {
            None
        }
    }
}

macro_rules! impl_set_at_idx_wrap {
    ($name:ident, $native:ty, $cast:ident) => {
        #[napi]
        pub fn $name(
            series: &JsSeries,
            indices: Vec<u32>,
            value: Option<Wrap<$native>>,
        ) -> napi::Result<JsSeries> {
            let value = value.map(|v| v.0);
            let s = series
                .series
                .$cast()
                .map_err(JsPolarsErr::from)?
                .set_at_idx(indices.iter().map(|idx| *idx as usize), value)
                .map_err(JsPolarsErr::from)?
                .into_series();
            Ok(s.into())
        }
    };
}
macro_rules! impl_set_at_idx {
    ($name:ident, $native:ty, $cast:ident) => {
        #[napi]
        pub fn $name(
            series: &JsSeries,
            indices: Vec<u32>,
            value: Option<$native>,
        ) -> napi::Result<JsSeries> {
            let s = series
                .series
                .$cast()
                .map_err(JsPolarsErr::from)?
                .set_at_idx(indices.iter().map(|idx| *idx as usize), value)
                .map_err(JsPolarsErr::from)?
                .into_series();
            Ok(s.into())
        }
    };
}

impl_set_at_idx_wrap!(series_set_at_idx_str, &str, utf8);
impl_set_at_idx!(series_set_at_idx_f64, f64, f64);
impl_set_at_idx_wrap!(series_set_at_idx_f32, f32, f32);
impl_set_at_idx_wrap!(series_set_at_idx_u8, u8, u8);
impl_set_at_idx_wrap!(series_set_at_idx_u16, u16, u16);
impl_set_at_idx!(series_set_at_idx_u32, u32, u32);
impl_set_at_idx_wrap!(series_set_at_idx_u64, u64, u64);
impl_set_at_idx_wrap!(series_set_at_idx_i8, i8, i8);
impl_set_at_idx_wrap!(series_set_at_idx_i16, i16, i16);
impl_set_at_idx!(series_set_at_idx_i32, i32, i32);
impl_set_at_idx!(series_set_at_idx_i64, i64, i64);

macro_rules! impl_set_with_mask_wrap {
    ($name:ident, $native:ty, $cast:ident) => {
        #[napi]
        pub fn $name(
            series: &JsSeries,
            mask: &JsSeries,
            value: Option<Wrap<$native>>,
        ) -> napi::Result<JsSeries> {
            let value = value.map(|v| v.0);
            let mask = mask.series.bool().map_err(JsPolarsErr::from)?;
            let ca = series.series.$cast().map_err(JsPolarsErr::from)?;
            let new = ca
                .set(mask, value)
                .map_err(JsPolarsErr::from)?
                .into_series();
            Ok(new.into())
        }
    };
}

macro_rules! impl_set_with_mask {
    ($name:ident, $native:ty, $cast:ident) => {
        #[napi]
        pub fn $name(
            series: &JsSeries,
            mask: &JsSeries,
            value: Option<$native>,
        ) -> napi::Result<JsSeries> {
            let mask = mask.series.bool().map_err(JsPolarsErr::from)?;
            let ca = series.series.$cast().map_err(JsPolarsErr::from)?;
            let new = ca
                .set(mask, value)
                .map_err(JsPolarsErr::from)?
                .into_series();
            Ok(new.into())
        }
    };
}

impl_set_with_mask_wrap!(series_set_with_mask_str, &str, utf8);
impl_set_with_mask!(series_set_with_mask_f64, f64, f64);
impl_set_with_mask_wrap!(series_set_with_mask_f32, f32, f32);
impl_set_with_mask_wrap!(series_set_with_mask_u8, u8, u8);
impl_set_with_mask_wrap!(series_set_with_mask_u16, u16, u16);
impl_set_with_mask!(series_set_with_mask_u32, u32, u32);
impl_set_with_mask_wrap!(series_set_with_mask_u64, u64, u64);
impl_set_with_mask_wrap!(series_set_with_mask_i8, i8, i8);
impl_set_with_mask_wrap!(series_set_with_mask_i16, i16, i16);
impl_set_with_mask!(series_set_with_mask_i32, i32, i32);
impl_set_with_mask!(series_set_with_mask_i64, i64, i64);

macro_rules! impl_get {
    ($name:ident, $series_variant:ident, $type:ty, $cast:ty) => {
        #[napi]
        pub fn $name(s: &JsSeries, index: i64) -> Option<$cast> {
            if let Ok(ca) = s.series.$series_variant() {
                let index = if index < 0 {
                    (ca.len() as i64 + index) as usize
                } else {
                    index as usize
                };
                ca.get(index).map(|s| s as $cast)
            } else {
                None
            }
        }
    };
}

impl_get!(series_get_f32, f32, f32, f64);
impl_get!(series_get_f64, f64, f64, f64);
impl_get!(series_get_u8, u8, u8, i32);
impl_get!(series_get_u16, u16, u16, u32);
impl_get!(series_get_u32, u32, u32, u32);
impl_get!(series_get_u64, u64, u64, i64);
impl_get!(series_get_i8, i8, i8, i32);
impl_get!(series_get_i16, i16, i16, i32);
impl_get!(series_get_i32, i32, i32, i32);
impl_get!(series_get_i64, i64, i64, i32);
// impl_get!(series_get_str, utf8, &str);
impl_get!(series_get_date, date, i32, i32);
impl_get!(series_get_datetime, datetime, i64, i64);
impl_get!(series_get_duration, duration, i64, i64);

#[napi]
pub fn series_get_str(s: &JsSeries, index: i64) -> Option<String> {
    if let Ok(ca) = s.series.utf8() {
        let index = if index < 0 {
            (ca.len() as i64 + index) as usize
        } else {
            index as usize
        };
        ca.get(index).map(|s| s.to_owned())
    } else {
        None
    }
}
macro_rules! impl_arithmetic {
  ($name:ident, $type:ty, $operand:tt) => {
      #[napi]
      pub fn $name(s: &JsSeries, other: JsAnyValue) -> napi::Result<JsSeries> {
          let other: $type = other.try_into()?;
          Ok(JsSeries::new(&s.series $operand other))
      }
  };
}
impl_arithmetic!(series_add_u8, u8, +);
impl_arithmetic!(series_add_u16, u16, +);
impl_arithmetic!(series_add_u32, u32, +);
impl_arithmetic!(series_add_u64, u64, +);
impl_arithmetic!(series_add_i8, i8, +);
impl_arithmetic!(series_add_i16, i16, +);
impl_arithmetic!(series_add_i32, i32, +);
impl_arithmetic!(series_add_i64, i64, +);
impl_arithmetic!(series_add_datetime, i64, +);
impl_arithmetic!(series_add_duration, i64, +);
impl_arithmetic!(series_add_f32, f32, +);
impl_arithmetic!(series_add_f64, f64, +);
impl_arithmetic!(series_sub_u8, u8, -);
impl_arithmetic!(series_sub_u16, u16, -);
impl_arithmetic!(series_sub_u32, u32, -);
impl_arithmetic!(series_sub_u64, u64, -);
impl_arithmetic!(series_sub_i8, i8, -);
impl_arithmetic!(series_sub_i16, i16, -);
impl_arithmetic!(series_sub_i32, i32, -);
impl_arithmetic!(series_sub_i64, i64, -);
impl_arithmetic!(series_sub_datetime, i64, -);
impl_arithmetic!(series_sub_duration, i64, -);
impl_arithmetic!(series_sub_f32, f32, -);
impl_arithmetic!(series_sub_f64, f64, -);
impl_arithmetic!(series_div_u8, u8, /);
impl_arithmetic!(series_div_u16, u16, /);
impl_arithmetic!(series_div_u32, u32, /);
impl_arithmetic!(series_div_u64, u64, /);
impl_arithmetic!(series_div_i8, i8, /);
impl_arithmetic!(series_div_i16, i16, /);
impl_arithmetic!(series_div_i32, i32, /);
impl_arithmetic!(series_div_i64, i64, /);
impl_arithmetic!(series_div_f32, f32, /);
impl_arithmetic!(series_div_f64, f64, /);
impl_arithmetic!(series_mul_u8, u8, *);
impl_arithmetic!(series_mul_u16, u16, *);
impl_arithmetic!(series_mul_u32, u32, *);
impl_arithmetic!(series_mul_u64, u64, *);
impl_arithmetic!(series_mul_i8, i8, *);
impl_arithmetic!(series_mul_i16, i16, *);
impl_arithmetic!(series_mul_i32, i32, *);
impl_arithmetic!(series_mul_i64, i64, *);
impl_arithmetic!(series_mul_f32, f32, *);
impl_arithmetic!(series_mul_f64, f64, *);
impl_arithmetic!(series_rem_u8, u8, %);
impl_arithmetic!(series_rem_u16, u16, %);
impl_arithmetic!(series_rem_u32, u32, %);
impl_arithmetic!(series_rem_u64, u64, %);
impl_arithmetic!(series_rem_i8, i8, %);
impl_arithmetic!(series_rem_i16, i16, %);
impl_arithmetic!(series_rem_i32, i32, %);
impl_arithmetic!(series_rem_i64, i64, %);
impl_arithmetic!(series_rem_f32, f32, %);
impl_arithmetic!(series_rem_f64, f64, %);

macro_rules! impl_rhs_arithmetic {
    ($name:ident, $type:ty, $operand:ident) => {
        #[napi]

        pub fn $name(s: &JsSeries, other: JsAnyValue) -> napi::Result<JsSeries> {
            let other: $type = other.try_into()?;
            Ok(JsSeries::new(other.$operand(&s.series)))
        }
    };
}

impl_rhs_arithmetic!(series_add_u8_rhs, u8, add);
impl_rhs_arithmetic!(series_add_u16_rhs, u16, add);
impl_rhs_arithmetic!(series_add_u32_rhs, u32, add);
impl_rhs_arithmetic!(series_add_u64_rhs, u64, add);
impl_rhs_arithmetic!(series_add_i8_rhs, i8, add);
impl_rhs_arithmetic!(series_add_i16_rhs, i16, add);
impl_rhs_arithmetic!(series_add_i32_rhs, i32, add);
impl_rhs_arithmetic!(series_add_i64_rhs, i64, add);
impl_rhs_arithmetic!(series_add_f32_rhs, f32, add);
impl_rhs_arithmetic!(series_add_f64_rhs, f64, add);
impl_rhs_arithmetic!(series_sub_u8_rhs, u8, sub);
impl_rhs_arithmetic!(series_sub_u16_rhs, u16, sub);
impl_rhs_arithmetic!(series_sub_u32_rhs, u32, sub);
impl_rhs_arithmetic!(series_sub_u64_rhs, u64, sub);
impl_rhs_arithmetic!(series_sub_i8_rhs, i8, sub);
impl_rhs_arithmetic!(series_sub_i16_rhs, i16, sub);
impl_rhs_arithmetic!(series_sub_i32_rhs, i32, sub);
impl_rhs_arithmetic!(series_sub_i64_rhs, i64, sub);
impl_rhs_arithmetic!(series_sub_f32_rhs, f32, sub);
impl_rhs_arithmetic!(series_sub_f64_rhs, f64, sub);
impl_rhs_arithmetic!(series_div_u8_rhs, u8, div);
impl_rhs_arithmetic!(series_div_u16_rhs, u16, div);
impl_rhs_arithmetic!(series_div_u32_rhs, u32, div);
impl_rhs_arithmetic!(series_div_u64_rhs, u64, div);
impl_rhs_arithmetic!(series_div_i8_rhs, i8, div);
impl_rhs_arithmetic!(series_div_i16_rhs, i16, div);
impl_rhs_arithmetic!(series_div_i32_rhs, i32, div);
impl_rhs_arithmetic!(series_div_i64_rhs, i64, div);
impl_rhs_arithmetic!(series_div_f32_rhs, f32, div);
impl_rhs_arithmetic!(series_div_f64_rhs, f64, div);
impl_rhs_arithmetic!(series_mul_u8_rhs, u8, mul);
impl_rhs_arithmetic!(series_mul_u16_rhs, u16, mul);
impl_rhs_arithmetic!(series_mul_u32_rhs, u32, mul);
impl_rhs_arithmetic!(series_mul_u64_rhs, u64, mul);
impl_rhs_arithmetic!(series_mul_i8_rhs, i8, mul);
impl_rhs_arithmetic!(series_mul_i16_rhs, i16, mul);
impl_rhs_arithmetic!(series_mul_i32_rhs, i32, mul);
impl_rhs_arithmetic!(series_mul_i64_rhs, i64, mul);
impl_rhs_arithmetic!(series_mul_f32_rhs, f32, mul);
impl_rhs_arithmetic!(series_mul_f64_rhs, f64, mul);
impl_rhs_arithmetic!(series_rem_u8_rhs, u8, rem);
impl_rhs_arithmetic!(series_rem_u16_rhs, u16, rem);
impl_rhs_arithmetic!(series_rem_u32_rhs, u32, rem);
impl_rhs_arithmetic!(series_rem_u64_rhs, u64, rem);
impl_rhs_arithmetic!(series_rem_i8_rhs, i8, rem);
impl_rhs_arithmetic!(series_rem_i16_rhs, i16, rem);
impl_rhs_arithmetic!(series_rem_i32_rhs, i32, rem);
impl_rhs_arithmetic!(series_rem_i64_rhs, i64, rem);
impl_rhs_arithmetic!(series_rem_f32_rhs, f32, rem);
impl_rhs_arithmetic!(series_rem_f64_rhs, f64, rem);

macro_rules! impl_eq_num {
    ($name:ident, $type:ty) => {
        #[napi]
        pub fn $name(s: &JsSeries, rhs: JsAnyValue) -> napi::Result<JsSeries> {
            let rhs: $type = rhs.try_into()?;
            Ok(JsSeries::new(
                s.series
                    .equal(rhs)
                    .map_err(JsPolarsErr::from)?
                    .into_series(),
            ))
        }
    };
}

impl_eq_num!(series_eq_u8, u8);
impl_eq_num!(series_eq_u16, u16);
impl_eq_num!(series_eq_u32, u32);
impl_eq_num!(series_eq_u64, u64);
impl_eq_num!(series_eq_i8, i8);
impl_eq_num!(series_eq_i16, i16);
impl_eq_num!(series_eq_i32, i32);
impl_eq_num!(series_eq_i64, i64);
impl_eq_num!(series_eq_f32, f32);
impl_eq_num!(series_eq_f64, f64);
impl_eq_num!(series_eq_str, &str);

macro_rules! impl_neq_num {
    ($name:ident, $type:ty) => {
        #[napi]
        pub fn $name(s: &JsSeries, rhs: JsAnyValue) -> napi::Result<JsSeries> {
            let rhs: $type = rhs.try_into()?;
            Ok(JsSeries::new(
                s.series
                    .not_equal(rhs)
                    .map_err(JsPolarsErr::from)?
                    .into_series(),
            ))
        }
    };
}
impl_neq_num!(series_neq_u8, u8);
impl_neq_num!(series_neq_u16, u16);
impl_neq_num!(series_neq_u32, u32);
impl_neq_num!(series_neq_u64, u64);
impl_neq_num!(series_neq_i8, i8);
impl_neq_num!(series_neq_i16, i16);
impl_neq_num!(series_neq_i32, i32);
impl_neq_num!(series_neq_i64, i64);
impl_neq_num!(series_neq_f32, f32);
impl_neq_num!(series_neq_f64, f64);
impl_neq_num!(series_neq_str, &str);

macro_rules! impl_gt_num {
    ($name:ident, $type:ty) => {
        #[napi]
        pub fn $name(s: &JsSeries, rhs: JsAnyValue) -> napi::Result<JsSeries> {
            let rhs: $type = rhs.try_into()?;
            Ok(JsSeries::new(
                s.series.gt(rhs).map_err(JsPolarsErr::from)?.into_series(),
            ))
        }
    };
}
impl_gt_num!(series_gt_u8, u8);
impl_gt_num!(series_gt_u16, u16);
impl_gt_num!(series_gt_u32, u32);
impl_gt_num!(series_gt_u64, u64);
impl_gt_num!(series_gt_i8, i8);
impl_gt_num!(series_gt_i16, i16);
impl_gt_num!(series_gt_i32, i32);
impl_gt_num!(series_gt_i64, i64);
impl_gt_num!(series_gt_f32, f32);
impl_gt_num!(series_gt_f64, f64);
impl_gt_num!(series_gt_str, &str);

macro_rules! impl_gt_eq_num {
    ($name:ident, $type:ty) => {
        #[napi]
        pub fn $name(s: &JsSeries, rhs: JsAnyValue) -> napi::Result<JsSeries> {
            let rhs: $type = rhs.try_into()?;
            Ok(JsSeries::new(
                s.series
                    .gt_eq(rhs)
                    .map_err(JsPolarsErr::from)?
                    .into_series(),
            ))
        }
    };
}
impl_gt_eq_num!(series_gt_eq_u8, u8);
impl_gt_eq_num!(series_gt_eq_u16, u16);
impl_gt_eq_num!(series_gt_eq_u32, u32);
impl_gt_eq_num!(series_gt_eq_u64, u64);
impl_gt_eq_num!(series_gt_eq_i8, i8);
impl_gt_eq_num!(series_gt_eq_i16, i16);
impl_gt_eq_num!(series_gt_eq_i32, i32);
impl_gt_eq_num!(series_gt_eq_i64, i64);
impl_gt_eq_num!(series_gt_eq_f32, f32);
impl_gt_eq_num!(series_gt_eq_f64, f64);
impl_gt_eq_num!(series_gt_eq_str, &str);

macro_rules! impl_lt_num {
    ($name:ident, $type:ty) => {
        #[napi]
        pub fn $name(s: &JsSeries, rhs: JsAnyValue) -> napi::Result<JsSeries> {
            let rhs: $type = rhs.try_into()?;
            Ok(JsSeries::new(
                s.series.lt(rhs).map_err(JsPolarsErr::from)?.into_series(),
            ))
        }
    };
}
impl_lt_num!(series_lt_u8, u8);
impl_lt_num!(series_lt_u16, u16);
impl_lt_num!(series_lt_u32, u32);
impl_lt_num!(series_lt_u64, u64);
impl_lt_num!(series_lt_i8, i8);
impl_lt_num!(series_lt_i16, i16);
impl_lt_num!(series_lt_i32, i32);
impl_lt_num!(series_lt_i64, i64);
impl_lt_num!(series_lt_f32, f32);
impl_lt_num!(series_lt_f64, f64);
impl_lt_num!(series_lt_str, &str);

macro_rules! impl_lt_eq_num {
    ($name:ident, $type:ty) => {
        #[napi]
        pub fn $name(s: &JsSeries, rhs: JsAnyValue) -> napi::Result<JsSeries> {
            let rhs: $type = rhs.try_into()?;
            Ok(JsSeries::new(
                s.series
                    .lt_eq(rhs)
                    .map_err(JsPolarsErr::from)?
                    .into_series(),
            ))
        }
    };
}
impl_lt_eq_num!(series_lt_eq_u8, u8);
impl_lt_eq_num!(series_lt_eq_u16, u16);
impl_lt_eq_num!(series_lt_eq_u32, u32);
impl_lt_eq_num!(series_lt_eq_u64, u64);
impl_lt_eq_num!(series_lt_eq_i8, i8);
impl_lt_eq_num!(series_lt_eq_i16, i16);
impl_lt_eq_num!(series_lt_eq_i32, i32);
impl_lt_eq_num!(series_lt_eq_i64, i64);
impl_lt_eq_num!(series_lt_eq_f32, f32);
impl_lt_eq_num!(series_lt_eq_f64, f64);
impl_lt_eq_num!(series_lt_eq_str, &str);
