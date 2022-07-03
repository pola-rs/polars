use crate::datatypes::JsDataType;
use crate::lazy::dsl::JsExpr;
use crate::prelude::*;
use napi::bindgen_prelude::*;
use napi::{
    JsBigInt, JsBoolean, JsDate, JsNumber, JsObject, JsString, JsUnknown, Result, ValueType,
};
use polars::frame::NullStrategy;
use polars::io::RowCount;
use polars::lazy::dsl::Expr;
use polars::prelude::*;
use polars_core::prelude::FillNullStrategy;
use polars_core::prelude::{Field, Schema};
use polars_core::series::ops::NullBehavior;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Wrap<T>(pub T);

impl<T> Clone for Wrap<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Wrap(self.0.clone())
    }
}
impl<T> From<T> for Wrap<T> {
    fn from(t: T) -> Self {
        Wrap(t)
    }
}
impl TypeName for Wrap<Utf8Chunked> {
    fn type_name() -> &'static str {
        "Utf8Chunked"
    }

    fn value_type() -> ValueType {
        ValueType::Object
    }
}
/// Safety.
/// it is up to the consumer to make sure the item is valid/
pub(crate) trait ToSeries {
    unsafe fn to_series(&self) -> Series;
}

impl ToSeries for Array {
    unsafe fn to_series(&self) -> Series {
        let len = self.len();
        let mut v: Vec<AnyValue> = Vec::with_capacity(len as usize);
        for i in 0..len {
            let av: Wrap<AnyValue> = self.get(i).unwrap().unwrap_or(Wrap(AnyValue::Null));
            v.push(av.0);
        }
        Series::new("", v)
    }
}

impl ToSeries for JsUnknown {
    unsafe fn to_series(&self) -> Series {
        let obj = self.cast::<JsObject>();
        let len = obj.get_array_length_unchecked().unwrap();
        let mut v: Vec<AnyValue> = Vec::with_capacity(len as usize);
        for i in 0..len {
            let unknown: JsUnknown = obj.get_element_unchecked(i).unwrap();
            let av = AnyValue::from_js(unknown).unwrap();
            v.push(av);
        }
        Series::new("", v)
    }
}
impl ToNapiValue for Wrap<&Series> {
    unsafe fn to_napi_value(napi_env: sys::napi_env, val: Self) -> napi::Result<sys::napi_value> {
        let s = val.0;
        let len = s.len();
        let dtype = s.dtype();
        let env = Env::from_raw(napi_env);

        match dtype {
            DataType::Struct(_) => {
                let ca = s.struct_().map_err(JsPolarsErr::from)?;
                let df: DataFrame = ca.clone().into();
                let (height, _) = df.shape();
                let mut rows = env.create_array(height as u32)?;
                for idx in 0..height {
                    let mut row = env.create_object()?;
                    for col in df.get_columns() {
                        let key = col.name();
                        let val = col.get(idx);
                        row.set(key, Wrap(val))?;
                    }
                    rows.set(idx as u32, row)?;
                }
                Array::to_napi_value(napi_env, rows)
            }
            _ => {
                let mut arr = env.create_array(len as u32)?;
                for (idx, val) in s.iter().enumerate() {
                    arr.set(idx as u32, Wrap(val))?;
                }
                Array::to_napi_value(napi_env, arr)
            }
        }
    }
}
impl<'a> ToNapiValue for Wrap<AnyValue<'a>> {
    unsafe fn to_napi_value(env: sys::napi_env, val: Self) -> Result<sys::napi_value> {
        match val.0 {
            AnyValue::Null => {
                napi::bindgen_prelude::Null::to_napi_value(env, napi::bindgen_prelude::Null)
            }
            AnyValue::Boolean(b) => bool::to_napi_value(env, b),
            AnyValue::Int8(n) => i32::to_napi_value(env, n as i32),
            AnyValue::Int16(n) => i32::to_napi_value(env, n as i32),
            AnyValue::Int32(n) => i32::to_napi_value(env, n),
            AnyValue::Int64(n) => i64::to_napi_value(env, n),
            AnyValue::UInt8(n) => u32::to_napi_value(env, n as u32),
            AnyValue::UInt16(n) => u32::to_napi_value(env, n as u32),
            AnyValue::UInt32(n) => u32::to_napi_value(env, n),
            AnyValue::UInt64(n) => u64::to_napi_value(env, n),
            AnyValue::Float32(n) => f64::to_napi_value(env, n as f64),
            AnyValue::Float64(n) => f64::to_napi_value(env, n),
            AnyValue::Utf8(s) => String::to_napi_value(env, s.to_owned()),
            AnyValue::Date(v) => {
                let mut ptr = std::ptr::null_mut();

                check_status!(
                    napi::sys::napi_create_date(env, v as f64, &mut ptr),
                    "Failed to convert rust type `AnyValue::Date` into napi value",
                )?;

                Ok(ptr)
            }
            AnyValue::Datetime(v, _, _) => {
                let mut ptr = std::ptr::null_mut();

                check_status!(
                    napi::sys::napi_create_date(env, v as f64, &mut ptr),
                    "Failed to convert rust type `AnyValue::Date` into napi value",
                )?;

                Ok(ptr)
            }
            AnyValue::Duration(v, _) => i64::to_napi_value(env, v),
            AnyValue::Time(v) => i64::to_napi_value(env, v),
            AnyValue::List(ser) => Wrap::<&Series>::to_napi_value(env, Wrap(&ser)),
            AnyValue::Struct(vals, flds) => {
                let env_obj = Env::from_raw(env);
                let mut obj = env_obj.create_object()?;

                for (val, fld) in vals.iter().zip(flds) {
                    let key = fld.name();

                    obj.set(key, Wrap(val.clone()))?;
                }
                Object::to_napi_value(env, obj)
            }
            _ => todo!(), // Value::Object(obj) => unsafe { Map::to_napi_value(env, obj) },
        }
    }
}
impl FromNapiValue for Wrap<Utf8Chunked> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let arr = Array::from_napi_value(env, napi_val)?;
        let len = arr.len() as usize;
        let mut builder = Utf8ChunkedBuilder::new("", len, len * 25);
        for i in 0..len {
            match arr.get::<String>(i as u32) {
                Ok(val) => match val {
                    Some(str_val) => builder.append_value(str_val),
                    None => builder.append_null(),
                },
                Err(_) => builder.append_null(),
            }
        }

        Ok(Wrap(builder.finish()))
    }
}
impl FromNapiValue for Wrap<BooleanChunked> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let arr = Array::from_napi_value(env, napi_val)?;
        let len = arr.len() as usize;
        let mut builder = BooleanChunkedBuilder::new("", len);
        for i in 0..len {
            match arr.get::<bool>(i as u32) {
                Ok(val) => match val {
                    Some(str_val) => builder.append_value(str_val),
                    None => builder.append_null(),
                },
                Err(_) => builder.append_null(),
            }
        }

        Ok(Wrap(builder.finish()))
    }
}

impl FromNapiValue for Wrap<Float32Chunked> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let arr = Array::from_napi_value(env, napi_val)?;
        let len = arr.len() as usize;
        let mut builder = PrimitiveChunkedBuilder::<Float32Type>::new("", len);
        for i in 0..len {
            match arr.get::<f64>(i as u32) {
                Ok(val) => match val {
                    Some(v) => builder.append_value(v as f32),
                    None => builder.append_null(),
                },
                Err(_) => builder.append_null(),
            }
        }

        Ok(Wrap(builder.finish()))
    }
}
macro_rules! impl_chunked {
    ($type:ty, $native:ty) => {
        impl FromNapiValue for Wrap<ChunkedArray<$type>> {
            unsafe fn from_napi_value(
                env: sys::napi_env,
                napi_val: sys::napi_value,
            ) -> JsResult<Self> {
                let arr = Array::from_napi_value(env, napi_val)?;
                let len = arr.len() as usize;
                let mut builder = PrimitiveChunkedBuilder::<$type>::new("", len);
                for i in 0..len {
                    match arr.get::<$native>(i as u32) {
                        Ok(val) => match val {
                            Some(v) => builder.append_value(v),
                            None => builder.append_null(),
                        },
                        Err(_) => builder.append_null(),
                    }
                }
                Ok(Wrap(builder.finish()))
            }
        }
    };
}
impl_chunked!(Float64Type, f64);
impl_chunked!(Int32Type, i32);
impl_chunked!(UInt32Type, u32);
impl_chunked!(Int64Type, i64);

impl FromNapiValue for Wrap<ChunkedArray<UInt64Type>> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let arr = Array::from_napi_value(env, napi_val)?;
        let len = arr.len() as usize;
        let mut builder = PrimitiveChunkedBuilder::<UInt64Type>::new("", len);
        for i in 0..len {
            match arr.get::<BigInt>(i as u32) {
                Ok(val) => match val {
                    Some(v) => {
                        let (_, v, _) = v.get_u64();
                        builder.append_value(v)
                    }
                    None => builder.append_null(),
                },
                Err(_) => builder.append_null(),
            }
        }
        Ok(Wrap(builder.finish()))
    }
}
impl FromNapiValue for Wrap<Expr> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let obj = Object::from_napi_value(env, napi_val)?;
        let expr: &JsExpr = obj.get("_expr")?.unwrap();
        let expr = expr.inner.clone();
        Ok(Wrap(expr))
    }
}

impl TypeName for Wrap<QuantileInterpolOptions> {
    fn type_name() -> &'static str {
        "QuantileInterpolOptions"
    }

    fn value_type() -> ValueType {
        ValueType::Object
    }
}

impl FromNapiValue for Wrap<QuantileInterpolOptions> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let interpolation = String::from_napi_value(env, napi_val)?;
        let interpol = match interpolation.as_ref() {
            "nearest" => QuantileInterpolOptions::Nearest,
            "lower" => QuantileInterpolOptions::Lower,
            "higher" => QuantileInterpolOptions::Higher,
            "midpoint" => QuantileInterpolOptions::Midpoint,
            "linear" => QuantileInterpolOptions::Linear,
            _ => return Err(napi::Error::from_reason("not supported".to_owned())),
        };
        Ok(Wrap(interpol))
    }
}
impl FromNapiValue for Wrap<ClosedWindow> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let s = String::from_napi_value(env, napi_val)?;
        let cw = match s.as_ref() {
            "none" => ClosedWindow::None,
            "both" => ClosedWindow::Both,
            "left" => ClosedWindow::Left,
            "right" => ClosedWindow::Right,
            _ => {
                return Err(napi::Error::from_reason(
                    "closed should be any of {'none', 'left', 'right'}".to_owned(),
                ))
            }
        };
        Ok(Wrap(cw))
    }
}

impl FromNapiValue for Wrap<RankMethod> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let method = String::from_napi_value(env, napi_val)?;
        let method = match method.as_ref() {
            "min" => RankMethod::Min,
            "max" => RankMethod::Max,
            "average" => RankMethod::Average,
            "dense" => RankMethod::Dense,
            "ordinal" => RankMethod::Ordinal,
            "random" => RankMethod::Random,
            _ => {
                return Err(napi::Error::from_reason(
                    "use one of 'avg, min, max, dense, ordinal'".to_owned(),
                ))
            }
        };
        Ok(Wrap(method))
    }
}

impl FromNapiValue for Wrap<ParquetCompression> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let compression = String::from_napi_value(env, napi_val)?;
        let compression = match compression.as_ref() {
            "snappy" => ParquetCompression::Snappy,
            "gzip" => ParquetCompression::Gzip(None),
            "lzo" => ParquetCompression::Lzo,
            "brotli" => ParquetCompression::Brotli(None),
            "lz4" => ParquetCompression::Lz4Raw,
            "zstd" => ParquetCompression::Zstd(None),
            _ => ParquetCompression::Uncompressed,
        };
        Ok(Wrap(compression))
    }
}

impl FromNapiValue for Wrap<Option<IpcCompression>> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let compression = String::from_napi_value(env, napi_val)?;
        let compression = match compression.as_ref() {
            "lz4" => Some(IpcCompression::LZ4),
            "zstd" => Some(IpcCompression::ZSTD),
            _ => None,
        };
        Ok(Wrap(compression))
    }
}

impl FromNapiValue for Wrap<UniqueKeepStrategy> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let method = String::from_napi_value(env, napi_val)?;
        let method = match method.as_ref() {
            "first" => UniqueKeepStrategy::First,
            "last" => UniqueKeepStrategy::Last,
            _ => {
                return Err(napi::Error::from_reason(
                    "use one of 'first, last'".to_owned(),
                ))
            }
        };
        Ok(Wrap(method))
    }
}

impl FromNapiValue for Wrap<NullStrategy> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let method = String::from_napi_value(env, napi_val)?;
        let method = match method.as_ref() {
            "ignore" => NullStrategy::Ignore,
            "propagate" => NullStrategy::Propagate,
            _ => {
                return Err(napi::Error::from_reason(
                    "use one of 'ignore', 'propagate'".to_owned(),
                ))
            }
        };
        Ok(Wrap(method))
    }
}

impl FromNapiValue for Wrap<NullBehavior> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let method = String::from_napi_value(env, napi_val)?;
        let method = match method.as_ref() {
            "drop" => NullBehavior::Drop,
            "ignore" => NullBehavior::Ignore,
            _ => {
                return Err(napi::Error::from_reason(
                    "use one of 'drop', 'ignore'".to_owned(),
                ))
            }
        };
        Ok(Wrap(method))
    }
}

impl FromNapiValue for Wrap<FillNullStrategy> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let method = String::from_napi_value(env, napi_val)?;
        let method = match method.as_ref() {
            "backward" => FillNullStrategy::Backward(None),
            "forward" => FillNullStrategy::Forward(None),
            "min" => FillNullStrategy::Min,
            "max" => FillNullStrategy::Max,
            "mean" => FillNullStrategy::Mean,
            "zero" => FillNullStrategy::Zero,
            "one" => FillNullStrategy::One,
            _ => {
                return Err(napi::Error::from_reason(
                    "Strategy not supported".to_owned(),
                ))
            }
        };
        Ok(Wrap(method))
    }
}
impl FromNapiValue for Wrap<PivotAgg> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let method = String::from_napi_value(env, napi_val)?;
        match method.as_ref() {
            "sum" => Ok(Wrap(PivotAgg::Sum)),
            "min" => Ok(Wrap(PivotAgg::Min)),
            "max" => Ok(Wrap(PivotAgg::Max)),
            "first" => Ok(Wrap(PivotAgg::First)),
            "mean" => Ok(Wrap(PivotAgg::Mean)),
            "median" => Ok(Wrap(PivotAgg::Median)),
            "count" => Ok(Wrap(PivotAgg::Count)),
            "last" => Ok(Wrap(PivotAgg::Last)),
            s => {
                return Err(napi::Error::from_reason(
                    format!("aggregation {} not supported", s).to_owned(),
                ))
            }
        }
    }
}

impl FromNapiValue for Wrap<u8> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let n = u32::from_napi_value(env, napi_val)?;
        Ok(Wrap(n as u8))
    }
}
impl FromNapiValue for Wrap<u16> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let n = u32::from_napi_value(env, napi_val)?;
        Ok(Wrap(n as u16))
    }
}
impl FromNapiValue for Wrap<i8> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let n = u32::from_napi_value(env, napi_val)?;
        Ok(Wrap(n as i8))
    }
}
impl FromNapiValue for Wrap<i16> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let n = u32::from_napi_value(env, napi_val)?;
        Ok(Wrap(n as i16))
    }
}
impl FromNapiValue for Wrap<f32> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let n = f64::from_napi_value(env, napi_val)?;
        Ok(Wrap(n as f32))
    }
}

impl FromNapiValue for Wrap<u64> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let big = BigInt::from_napi_value(env, napi_val)?;
        let (_, value, _) = big.get_u64();
        Ok(Wrap(value))
    }
}

impl<'a> FromNapiValue for Wrap<&'a str> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let s = String::from_napi_value(env, napi_val)?;
        Ok(Wrap(Box::leak::<'a>(s.into_boxed_str())))
    }
}

#[napi(object)]
pub struct JsRollingOptions {
    pub window_size: String,
    pub weights: Option<Vec<f64>>,
    pub min_periods: i64,
    pub center: bool,
}

impl From<JsRollingOptions> for RollingOptionsImpl<'static> {
    fn from(o: JsRollingOptions) -> Self {
        RollingOptionsImpl {
            window_size: Duration::parse(&o.window_size),
            weights: o.weights,
            min_periods: o.min_periods as usize,
            center: o.center,
            by: None,
            tu: None,
            closed_window: None,
        }
    }
}

impl From<JsRollingOptions> for RollingOptions {
    fn from(o: JsRollingOptions) -> Self {
        RollingOptions {
            window_size: Duration::parse(&o.window_size),
            weights: o.weights,
            min_periods: o.min_periods as usize,
            center: o.center,
            by: None,
            closed_window: None,
        }
    }
}

#[napi(object)]
pub struct JsRowCount {
    pub name: String,
    pub offset: u32,
}

impl From<JsRowCount> for RowCount {
    fn from(o: JsRowCount) -> Self {
        RowCount {
            name: o.name,
            offset: o.offset,
        }
    }
}

#[napi(object)]
pub struct WriteCsvOptions {
    pub has_header: Option<bool>,
    pub sep: Option<String>,
    pub quote: Option<String>,
}

#[napi(object)]
pub struct Shape {
    pub height: i64,
    pub width: i64,
}

impl From<(usize, usize)> for Shape {
    fn from(s: (usize, usize)) -> Self {
        let (height, width) = s;
        Shape {
            height: height as i64,
            width: width as i64,
        }
    }
}

pub(crate) fn str_to_polarstype(dtype: &str) -> DataType {
    match dtype {
        "Int8" => DataType::Int8,
        "Int16" => DataType::Int16,
        "Int32" => DataType::Int32,
        "Int64" => DataType::Int64,
        "UInt8" => DataType::UInt8,
        "UInt16" => DataType::UInt16,
        "UInt32" => DataType::UInt32,
        "UInt64" => DataType::UInt64,
        "Float32" => DataType::Float32,
        "Float64" => DataType::Float64,
        "Bool" => DataType::Boolean,
        "Utf8" => DataType::Utf8,
        "List" => DataType::List(DataType::Null.into()),
        "Date" => DataType::Date,
        "Datetime" => DataType::Datetime(TimeUnit::Milliseconds, None),
        "Time" => DataType::Time,
        "Object" => DataType::Object("object"),
        "Categorical" => DataType::Categorical(None),
        "Struct" => DataType::Struct(vec![]),
        tp => panic!("Type {} not implemented in str_to_polarstype", tp),
    }
}

// Don't change the order of these!
pub(crate) fn num_to_polarstype(n: u32) -> DataType {
    match n {
        0 => DataType::Int8,
        1 => DataType::Int16,
        2 => DataType::Int32,
        3 => DataType::Int64,
        4 => DataType::UInt8,
        5 => DataType::UInt16,
        6 => DataType::UInt32,
        7 => DataType::UInt64,
        8 => DataType::Float32,
        9 => DataType::Float64,
        10 => DataType::Boolean,
        11 => DataType::Utf8,
        12 => DataType::List(DataType::Null.into()),
        13 => DataType::Date,
        14 => DataType::Datetime(TimeUnit::Milliseconds, None),
        15 => DataType::Time,
        16 => DataType::Object("object"),
        17 => DataType::Categorical(None),
        18 => DataType::Struct(vec![]),
        tp => panic!("Type {} not implemented in num_to_polarstype", tp),
    }
}

impl FromNapiValue for Wrap<DataType> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> napi::Result<Self> {
        let ty = type_of!(env, napi_val)?;
        let dtype = match ty {
            ValueType::String => {
                let s = String::from_napi_value(env, napi_val)?;
                str_to_polarstype(&s)
            }
            ValueType::Number => {
                let n = u32::from_napi_value(env, napi_val)?;
                num_to_polarstype(n)
            }
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    "not a valid conversion to 'DataType'".to_owned(),
                ))
            }
        };
        Ok(Wrap(dtype))
    }
}

impl FromNapiValue for Wrap<Schema> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> napi::Result<Self> {
        let ty = type_of!(env, napi_val)?;
        match ty {
            ValueType::Object => {
                let obj = Object::from_napi_value(env, napi_val)?;

                let keys = Object::keys(&obj)?;
                let fields: Vec<Field> = keys
                    .iter()
                    .map(|key| {
                        let value = obj.get::<_, Either<String, u32>>(&key).unwrap().unwrap();
                        let dtype = match value {
                            Either::A(v) => str_to_polarstype(&v),
                            Either::B(v) => num_to_polarstype(v),
                        };
                        let fld = Field::new(key, dtype.into());
                        fld
                    })
                    .collect();
                Ok(Wrap(Schema::from(fields)))
            }
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    "not a valid conversion to 'Schema'".to_owned(),
                ))
            }
        }
    }
}

pub enum TypedArrayBuffer {
    Int8(Int8Array),
    Int16(Int16Array),
    Int32(Int32Array),
    Int64(BigInt64Array),
    UInt8(Uint8Array),
    UInt16(Uint16Array),
    UInt32(Uint32Array),
    UInt64(BigUint64Array),
    Float32(Float32Array),
    Float64(Float64Array),
}

impl From<&Series> for TypedArrayBuffer {
    fn from(series: &Series) -> Self {
        let dt = series.dtype();
        match dt {
            DataType::Int8 => TypedArrayBuffer::Int8(Int8Array::with_data_copied(
                series.i8().unwrap().rechunk().cont_slice().unwrap(),
            )),
            DataType::Int16 => TypedArrayBuffer::Int16(Int16Array::with_data_copied(
                series.i16().unwrap().rechunk().cont_slice().unwrap(),
            )),
            DataType::Int32 => TypedArrayBuffer::Int32(Int32Array::with_data_copied(
                series.i32().unwrap().rechunk().cont_slice().unwrap(),
            )),
            DataType::Int64 => TypedArrayBuffer::Int64(BigInt64Array::with_data_copied(
                series.i64().unwrap().rechunk().cont_slice().unwrap(),
            )),
            DataType::UInt8 => TypedArrayBuffer::UInt8(Uint8Array::with_data_copied(
                series.u8().unwrap().rechunk().cont_slice().unwrap(),
            )),
            DataType::UInt16 => TypedArrayBuffer::UInt16(Uint16Array::with_data_copied(
                series.u16().unwrap().rechunk().cont_slice().unwrap(),
            )),
            DataType::UInt32 => TypedArrayBuffer::UInt32(Uint32Array::with_data_copied(
                series.u32().unwrap().rechunk().cont_slice().unwrap(),
            )),
            DataType::UInt64 => TypedArrayBuffer::UInt64(BigUint64Array::with_data_copied(
                series.u64().unwrap().rechunk().cont_slice().unwrap(),
            )),
            DataType::Float32 => TypedArrayBuffer::Float32(Float32Array::with_data_copied(
                series.f32().unwrap().rechunk().cont_slice().unwrap(),
            )),
            DataType::Float64 => TypedArrayBuffer::Float64(Float64Array::with_data_copied(
                series.f64().unwrap().rechunk().cont_slice().unwrap(),
            )),
            dt => panic!("to_list() not implemented for {:?}", dt),
        }
    }
}

impl ToNapiValue for TypedArrayBuffer {
    unsafe fn to_napi_value(env: sys::napi_env, val: Self) -> napi::Result<sys::napi_value> {
        match val {
            TypedArrayBuffer::Int8(v) => Int8Array::to_napi_value(env, v),
            TypedArrayBuffer::Int16(v) => Int16Array::to_napi_value(env, v),
            TypedArrayBuffer::Int32(v) => Int32Array::to_napi_value(env, v),
            TypedArrayBuffer::Int64(v) => BigInt64Array::to_napi_value(env, v),
            TypedArrayBuffer::UInt8(v) => Uint8Array::to_napi_value(env, v),
            TypedArrayBuffer::UInt16(v) => Uint16Array::to_napi_value(env, v),
            TypedArrayBuffer::UInt32(v) => Uint32Array::to_napi_value(env, v),
            TypedArrayBuffer::UInt64(v) => BigUint64Array::to_napi_value(env, v),
            TypedArrayBuffer::Float32(v) => Float32Array::to_napi_value(env, v),
            TypedArrayBuffer::Float64(v) => Float64Array::to_napi_value(env, v),
        }
    }
}

impl FromNapiValue for Wrap<NullValues> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        if let Ok(s) = String::from_napi_value(env, napi_val) {
            Ok(Wrap(NullValues::AllColumns(s)))
        } else if let Ok(s) = Vec::<String>::from_napi_value(env, napi_val) {
            Ok(Wrap(NullValues::Columns(s)))
        } else if let Ok(s) = HashMap::<String, String>::from_napi_value(env, napi_val) {
            let null_values: Vec<(String, String)> = s.into_iter().collect();
            Ok(Wrap(NullValues::Named(null_values)))
        } else {
            Err(
                JsPolarsErr::Other("could not extract value from null_values argument".into())
                    .into(),
            )
        }
    }
}

impl ToNapiValue for Wrap<NullValues> {
    unsafe fn to_napi_value(env: sys::napi_env, val: Self) -> napi::Result<sys::napi_value> {
        match val.0 {
            NullValues::AllColumns(s) => String::to_napi_value(env, s),
            NullValues::Columns(arr) => Vec::<String>::to_napi_value(env, arr),
            NullValues::Named(obj) => {
                let o: HashMap<String, String> = obj.into_iter().collect();
                HashMap::<String, String>::to_napi_value(env, o)
            }
        }
    }
}

pub trait FromJsUnknown: Sized + Send {
    fn from_js(obj: JsUnknown) -> Result<Self>;
}

impl FromJsUnknown for String {
    fn from_js(val: JsUnknown) -> Result<Self> {
        let s: JsString = val.try_into()?;
        s.into_utf8()?.into_owned()
    }
}

impl FromJsUnknown for AnyValue<'_> {
    fn from_js(val: JsUnknown) -> Result<Self> {
        match val.get_type()? {
            ValueType::Undefined | ValueType::Null => Ok(AnyValue::Null),
            ValueType::Boolean => bool::from_js(val).map(AnyValue::Boolean),
            ValueType::Number => f64::from_js(val).map(AnyValue::Float64),
            ValueType::String => {
                String::from_js(val).map(|s| AnyValue::Utf8(Box::leak::<'_>(s.into_boxed_str())))
            }
            ValueType::BigInt => u64::from_js(val).map(AnyValue::UInt64),
            ValueType::Object => {
                if val.is_date()? {
                    let d: JsDate = unsafe { val.cast() };
                    let d = d.value_of()?;
                    let d = d as i64;
                    Ok(AnyValue::Datetime(d, TimeUnit::Milliseconds, &None))
                } else {
                    Err(JsPolarsErr::Other("Unsupported Data type".to_owned()).into())
                }
            }
            _ => panic!("not supported"),
        }
    }
}

impl FromJsUnknown for DataType {
    fn from_js(val: JsUnknown) -> Result<Self> {
        match val.get_type()? {
            ValueType::Undefined | ValueType::Null => Ok(DataType::Null),
            ValueType::Boolean => Ok(DataType::Boolean),
            ValueType::Number => Ok(DataType::Float64),
            ValueType::String => Ok(DataType::Utf8),
            ValueType::BigInt => Ok(DataType::UInt64),
            ValueType::Object => {
                if val.is_date()? {
                    Ok(DataType::Datetime(TimeUnit::Milliseconds, None))
                } else {
                    Ok(DataType::Utf8)
                }
            }
            _ => panic!("not supported"),
        }
    }
}

impl<'a> FromJsUnknown for &'a str {
    fn from_js(val: JsUnknown) -> Result<Self> {
        let s: JsString = val.try_into()?;
        let s = s.into_utf8()?.into_owned()?;
        Ok(Box::leak::<'a>(s.into_boxed_str()))
    }
}

impl FromJsUnknown for bool {
    fn from_js(val: JsUnknown) -> Result<Self> {
        let s: JsBoolean = val.try_into()?;
        s.try_into()
    }
}

impl FromJsUnknown for f64 {
    fn from_js(val: JsUnknown) -> Result<Self> {
        let s: JsNumber = val.try_into()?;
        s.try_into()
    }
}

impl FromJsUnknown for i64 {
    fn from_js(val: JsUnknown) -> Result<Self> {
        match val.get_type()? {
            ValueType::BigInt => {
                let big: JsBigInt = unsafe { val.cast() };
                big.try_into()
            }
            ValueType::Number => {
                let s: JsNumber = val.try_into()?;
                s.try_into()
            }
            dt => Err(JsPolarsErr::Other(format!("cannot cast {} to u64", dt)).into()),
        }
    }
}

impl FromJsUnknown for u64 {
    fn from_js(val: JsUnknown) -> Result<Self> {
        match val.get_type()? {
            ValueType::BigInt => {
                let big: JsBigInt = unsafe { val.cast() };
                big.try_into()
            }
            ValueType::Number => {
                let s: JsNumber = val.try_into()?;
                Ok(s.get_int64()? as u64)
            }
            dt => Err(JsPolarsErr::Other(format!("cannot cast {} to u64", dt)).into()),
        }
    }
}
impl FromJsUnknown for u32 {
    fn from_js(val: JsUnknown) -> Result<Self> {
        let s: JsNumber = val.try_into()?;
        s.get_uint32()
    }
}
impl FromJsUnknown for f32 {
    fn from_js(val: JsUnknown) -> Result<Self> {
        let s: JsNumber = val.try_into()?;
        s.get_double().map(|s| s as f32)
    }
}

impl FromJsUnknown for usize {
    fn from_js(val: JsUnknown) -> Result<Self> {
        let s: JsNumber = val.try_into()?;
        Ok(s.get_uint32()? as usize)
    }
}
impl FromJsUnknown for u8 {
    fn from_js(val: JsUnknown) -> Result<Self> {
        let s: JsNumber = val.try_into()?;
        Ok(s.get_uint32()? as u8)
    }
}
impl FromJsUnknown for u16 {
    fn from_js(val: JsUnknown) -> Result<Self> {
        let s: JsNumber = val.try_into()?;
        Ok(s.get_uint32()? as u16)
    }
}
impl FromJsUnknown for i8 {
    fn from_js(val: JsUnknown) -> Result<Self> {
        let s: JsNumber = val.try_into()?;
        Ok(s.get_int32()? as i8)
    }
}
impl FromJsUnknown for i16 {
    fn from_js(val: JsUnknown) -> Result<Self> {
        let s: JsNumber = val.try_into()?;
        Ok(s.get_int32()? as i16)
    }
}

impl FromJsUnknown for i32 {
    fn from_js(val: JsUnknown) -> Result<Self> {
        let s: JsNumber = val.try_into()?;
        s.try_into()
    }
}

impl<V> FromJsUnknown for Option<V>
where
    V: FromJsUnknown,
{
    fn from_js(val: JsUnknown) -> Result<Self> {
        let v = V::from_js(val);
        match v {
            Ok(v) => Ok(Some(v)),
            Err(_) => Ok(None),
        }
    }
}
