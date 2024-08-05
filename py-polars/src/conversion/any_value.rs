use std::borrow::Cow;

#[cfg(feature = "object")]
use polars::chunked_array::object::PolarsObjectSafe;
#[cfg(feature = "object")]
use polars::datatypes::OwnedObject;
use polars::datatypes::{DataType, Field, PlHashMap, TimeUnit};
use polars::prelude::{AnyValue, Series};
use polars_core::export::chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, Timelike};
use polars_core::utils::any_values_to_supertype_and_n_dtypes;
use polars_core::utils::arrow::temporal_conversions::date32_to_date;
use pyo3::exceptions::{PyOverflowError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PySequence, PyString, PyTuple};

use super::datetime::{
    elapsed_offset_to_timedelta, nanos_since_midnight_to_naivetime, timestamp_to_naive_datetime,
};
use super::{decimal_to_digits, struct_dict, ObjectValue, Wrap};
use crate::error::PyPolarsErr;
use crate::py_modules::{SERIES, UTILS};
use crate::series::PySeries;

impl IntoPy<PyObject> for Wrap<AnyValue<'_>> {
    fn into_py(self, py: Python) -> PyObject {
        any_value_into_py_object(self.0, py)
    }
}

impl ToPyObject for Wrap<AnyValue<'_>> {
    fn to_object(&self, py: Python) -> PyObject {
        self.clone().into_py(py)
    }
}

impl<'py> FromPyObject<'py> for Wrap<AnyValue<'py>> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        py_object_to_any_value(ob, true).map(Wrap)
    }
}

pub(crate) fn any_value_into_py_object(av: AnyValue, py: Python) -> PyObject {
    let utils = UTILS.bind(py);
    match av {
        AnyValue::UInt8(v) => v.into_py(py),
        AnyValue::UInt16(v) => v.into_py(py),
        AnyValue::UInt32(v) => v.into_py(py),
        AnyValue::UInt64(v) => v.into_py(py),
        AnyValue::Int8(v) => v.into_py(py),
        AnyValue::Int16(v) => v.into_py(py),
        AnyValue::Int32(v) => v.into_py(py),
        AnyValue::Int64(v) => v.into_py(py),
        AnyValue::Float32(v) => v.into_py(py),
        AnyValue::Float64(v) => v.into_py(py),
        AnyValue::Null => py.None(),
        AnyValue::Boolean(v) => v.into_py(py),
        AnyValue::String(v) => v.into_py(py),
        AnyValue::StringOwned(v) => v.into_py(py),
        AnyValue::Categorical(idx, rev, arr) | AnyValue::Enum(idx, rev, arr) => {
            let s = if arr.is_null() {
                rev.get(idx)
            } else {
                unsafe { arr.deref_unchecked().value(idx as usize) }
            };
            s.into_py(py)
        },
        AnyValue::Date(v) => {
            let date = date32_to_date(v);
            date.into_py(py)
        },
        AnyValue::Datetime(v, time_unit, time_zone) => {
            if let Some(time_zone) = time_zone {
                // When https://github.com/pola-rs/polars/issues/16199 is
                // implemented, we'll switch to something like:
                //
                // let tz: chrono_tz::Tz = time_zone.parse().unwrap();
                // let datetime = tz.from_local_datetime(&naive_datetime).earliest().unwrap();
                // datetime.into_py(py)
                let convert = utils.getattr(intern!(py, "to_py_datetime")).unwrap();
                let time_unit = time_unit.to_ascii();
                convert
                    .call1((v, time_unit, time_zone.as_str()))
                    .unwrap()
                    .into_py(py)
            } else {
                timestamp_to_naive_datetime(v, time_unit).into_py(py)
            }
        },
        AnyValue::Duration(v, time_unit) => {
            let time_delta = elapsed_offset_to_timedelta(v, time_unit);
            time_delta.into_py(py)
        },
        AnyValue::Time(v) => nanos_since_midnight_to_naivetime(v).into_py(py),
        AnyValue::Array(v, _) | AnyValue::List(v) => PySeries::new(v).to_list(),
        ref av @ AnyValue::Struct(_, _, flds) => struct_dict(py, av._iter_struct_av(), flds),
        AnyValue::StructOwned(payload) => struct_dict(py, payload.0.into_iter(), &payload.1),
        #[cfg(feature = "object")]
        AnyValue::Object(v) => {
            let object = v.as_any().downcast_ref::<ObjectValue>().unwrap();
            object.inner.clone()
        },
        #[cfg(feature = "object")]
        AnyValue::ObjectOwned(v) => {
            let object = v.0.as_any().downcast_ref::<ObjectValue>().unwrap();
            object.inner.clone()
        },
        AnyValue::Binary(v) => v.into_py(py),
        AnyValue::BinaryOwned(v) => v.into_py(py),
        AnyValue::Decimal(v, scale) => {
            let convert = utils.getattr(intern!(py, "to_py_decimal")).unwrap();
            const N: usize = 3;
            let mut buf = [0_u128; N];
            let n_digits = decimal_to_digits(v.abs(), &mut buf);
            let buf = unsafe {
                std::slice::from_raw_parts(
                    buf.as_slice().as_ptr() as *const u8,
                    N * std::mem::size_of::<u128>(),
                )
            };
            let digits = PyTuple::new_bound(py, buf.iter().take(n_digits));
            convert
                .call1((v.is_negative() as u8, digits, n_digits, -(scale as i32)))
                .unwrap()
                .into_py(py)
        },
    }
}

type TypeObjectPtr = usize;
type InitFn = for<'py> fn(&Bound<'py, PyAny>, bool) -> PyResult<AnyValue<'py>>;
pub(crate) static LUT: crate::gil_once_cell::GILOnceCell<PlHashMap<TypeObjectPtr, InitFn>> =
    crate::gil_once_cell::GILOnceCell::new();

/// Convert a Python object to an [`AnyValue`].
pub(crate) fn py_object_to_any_value<'py>(
    ob: &Bound<'py, PyAny>,
    strict: bool,
) -> PyResult<AnyValue<'py>> {
    // Conversion functions.
    fn get_null(_ob: &Bound<'_, PyAny>, _strict: bool) -> PyResult<AnyValue<'static>> {
        Ok(AnyValue::Null)
    }

    fn get_bool(ob: &Bound<'_, PyAny>, _strict: bool) -> PyResult<AnyValue<'static>> {
        let b = ob.extract::<bool>()?;
        Ok(AnyValue::Boolean(b))
    }

    fn get_int(ob: &Bound<'_, PyAny>, strict: bool) -> PyResult<AnyValue<'static>> {
        if let Ok(v) = ob.extract::<i64>() {
            Ok(AnyValue::Int64(v))
        } else if let Ok(v) = ob.extract::<u64>() {
            Ok(AnyValue::UInt64(v))
        } else if !strict {
            let f = ob.extract::<f64>()?;
            Ok(AnyValue::Float64(f))
        } else {
            Err(PyOverflowError::new_err(format!(
                "int value too large for Polars integer types: {ob}"
            )))
        }
    }

    fn get_float(ob: &Bound<'_, PyAny>, _strict: bool) -> PyResult<AnyValue<'static>> {
        Ok(AnyValue::Float64(ob.extract::<f64>()?))
    }

    fn get_str(ob: &Bound<'_, PyAny>, _strict: bool) -> PyResult<AnyValue<'static>> {
        // Ideally we'd be returning an AnyValue::String(&str) instead, as was
        // the case in previous versions of this function. However, if compiling
        // with abi3 for versions older than Python 3.10, the APIs that purport
        // to return &str actually just encode to UTF-8 as a newly allocated
        // PyBytes object, and then return reference to that. So what we're
        // doing here isn't any different fundamentally, and the APIs to for
        // converting to &str are deprecated in PyO3 0.21.
        //
        // Once Python 3.10 is the minimum supported version, converting to &str
        // will be cheaper, and we should do that. Python 3.9 security updates
        // end-of-life is Oct 31, 2025.
        Ok(AnyValue::StringOwned(ob.extract::<String>()?.into()))
    }

    fn get_bytes<'py>(ob: &Bound<'py, PyAny>, _strict: bool) -> PyResult<AnyValue<'py>> {
        let value = ob.extract::<Vec<u8>>()?;
        Ok(AnyValue::BinaryOwned(value))
    }

    fn get_date(ob: &Bound<'_, PyAny>, _strict: bool) -> PyResult<AnyValue<'static>> {
        const UNIX_EPOCH: NaiveDate = NaiveDateTime::UNIX_EPOCH.date();
        let date = ob.extract::<NaiveDate>()?;
        let elapsed = date.signed_duration_since(UNIX_EPOCH);
        Ok(AnyValue::Date(elapsed.num_days() as i32))
    }

    fn get_datetime(ob: &Bound<'_, PyAny>, _strict: bool) -> PyResult<AnyValue<'static>> {
        // Probably needs to wait for
        // https://github.com/pola-rs/polars/issues/16199 to do it a faster way.
        Python::with_gil(|py| {
            let date = UTILS
                .bind(py)
                .getattr(intern!(py, "datetime_to_int"))
                .unwrap()
                .call1((ob, intern!(py, "us")))
                .unwrap();
            let v = date.extract::<i64>()?;
            Ok(AnyValue::Datetime(v, TimeUnit::Microseconds, &None))
        })
    }

    fn get_timedelta(ob: &Bound<'_, PyAny>, _strict: bool) -> PyResult<AnyValue<'static>> {
        let timedelta = ob.extract::<TimeDelta>()?;
        if let Some(micros) = timedelta.num_microseconds() {
            Ok(AnyValue::Duration(micros, TimeUnit::Microseconds))
        } else {
            Ok(AnyValue::Duration(
                timedelta.num_milliseconds(),
                TimeUnit::Milliseconds,
            ))
        }
    }

    fn get_time(ob: &Bound<'_, PyAny>, _strict: bool) -> PyResult<AnyValue<'static>> {
        let time = ob.extract::<NaiveTime>()?;

        Ok(AnyValue::Time(
            (time.num_seconds_from_midnight() as i64) * 1_000_000_000 + time.nanosecond() as i64,
        ))
    }

    fn get_decimal(ob: &Bound<'_, PyAny>, _strict: bool) -> PyResult<AnyValue<'static>> {
        fn abs_decimal_from_digits(
            digits: impl IntoIterator<Item = u8>,
            exp: i32,
        ) -> Option<(i128, usize)> {
            const MAX_ABS_DEC: i128 = 10_i128.pow(38) - 1;
            let mut v = 0_i128;
            for (i, d) in digits.into_iter().map(i128::from).enumerate() {
                if i < 38 {
                    v = v * 10 + d;
                } else {
                    v = v.checked_mul(10).and_then(|v| v.checked_add(d))?;
                }
            }
            // We only support non-negative scale (=> non-positive exponent).
            let scale = if exp > 0 {
                // The decimal may be in a non-canonical representation, try to fix it first.
                v = 10_i128
                    .checked_pow(exp as u32)
                    .and_then(|factor| v.checked_mul(factor))?;
                0
            } else {
                (-exp) as usize
            };
            // TODO: Do we care for checking if it fits in MAX_ABS_DEC? (if we set precision to None anyway?)
            (v <= MAX_ABS_DEC).then_some((v, scale))
        }

        // Note: Using Vec<u8> is not the most efficient thing here (input is a tuple)
        let (sign, digits, exp): (i8, Vec<u8>, i32) = ob
            .call_method0(intern!(ob.py(), "as_tuple"))
            .unwrap()
            .extract()
            .unwrap();
        let (mut v, scale) = abs_decimal_from_digits(digits, exp).ok_or_else(|| {
            PyErr::from(PyPolarsErr::Other(
                "Decimal is too large to fit in Decimal128".into(),
            ))
        })?;
        if sign > 0 {
            v = -v; // Won't overflow since -i128::MAX > i128::MIN
        }
        Ok(AnyValue::Decimal(v, scale))
    }

    fn get_list(ob: &Bound<'_, PyAny>, strict: bool) -> PyResult<AnyValue<'static>> {
        fn get_list_with_constructor(
            ob: &Bound<'_, PyAny>,
            strict: bool,
        ) -> PyResult<AnyValue<'static>> {
            // Use the dedicated constructor.
            // This constructor is able to go via dedicated type constructors
            // so it can be much faster.
            let py = ob.py();
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("strict", strict)?;
            let s = SERIES.call_bound(py, (ob,), Some(&kwargs))?;
            get_list_from_series(s.bind(py), strict)
        }

        if ob.is_empty()? {
            Ok(AnyValue::List(Series::new_empty("", &DataType::Null)))
        } else if ob.is_instance_of::<PyList>() | ob.is_instance_of::<PyTuple>() {
            const INFER_SCHEMA_LENGTH: usize = 25;

            let list = ob.downcast::<PySequence>().unwrap();

            let mut avs = Vec::with_capacity(INFER_SCHEMA_LENGTH);
            let mut iter = list.iter()?;
            let mut items = Vec::with_capacity(INFER_SCHEMA_LENGTH);
            for item in (&mut iter).take(INFER_SCHEMA_LENGTH) {
                items.push(item?);
                let av = py_object_to_any_value(items.last().unwrap(), strict)?;
                avs.push(av)
            }
            let (dtype, n_dtypes) = any_values_to_supertype_and_n_dtypes(&avs)
                .map_err(|e| PyTypeError::new_err(e.to_string()))?;

            // This path is only taken if there is no question about the data type.
            if dtype.is_primitive() && n_dtypes == 1 {
                get_list_with_constructor(ob, strict)
            } else {
                // Push the rest.
                let length = list.len()?;
                avs.reserve(length);
                let mut rest = Vec::with_capacity(length);
                for item in iter {
                    rest.push(item?);
                    let av = py_object_to_any_value(rest.last().unwrap(), strict)?;
                    avs.push(av)
                }

                let s = Series::from_any_values_and_dtype("", &avs, &dtype, strict)
                    .map_err(|e| {
                        PyTypeError::new_err(format!(
                            "{e}\n\nHint: Try setting `strict=False` to allow passing data with mixed types."
                        ))
                    })?;
                Ok(AnyValue::List(s))
            }
        } else {
            // range will take this branch
            get_list_with_constructor(ob, strict)
        }
    }

    fn get_list_from_series(ob: &Bound<'_, PyAny>, _strict: bool) -> PyResult<AnyValue<'static>> {
        let s = super::get_series(ob)?;
        Ok(AnyValue::List(s))
    }

    fn get_struct<'py>(ob: &Bound<'py, PyAny>, strict: bool) -> PyResult<AnyValue<'py>> {
        let dict = ob.downcast::<PyDict>().unwrap();
        let len = dict.len();
        let mut keys = Vec::with_capacity(len);
        let mut vals = Vec::with_capacity(len);
        for (k, v) in dict.into_iter() {
            let key = k.extract::<Cow<str>>()?;
            let val = py_object_to_any_value(&v, strict)?;
            let dtype = val.dtype();
            keys.push(Field::new(&key, dtype));
            vals.push(val)
        }
        Ok(AnyValue::StructOwned(Box::new((vals, keys))))
    }

    fn get_object(ob: &Bound<'_, PyAny>, _strict: bool) -> PyResult<AnyValue<'static>> {
        #[cfg(feature = "object")]
        {
            // This is slow, but hey don't use objects.
            let v = &ObjectValue {
                inner: ob.clone().unbind(),
            };
            Ok(AnyValue::ObjectOwned(OwnedObject(v.to_boxed())))
        }
        #[cfg(not(feature = "object"))]
        panic!("activate object")
    }

    /// Determine which conversion function to use for the given object.
    ///
    /// Note: This function is only ran if the object's type is not already in the
    /// lookup table.
    fn get_conversion_function(ob: &Bound<'_, PyAny>, py: Python<'_>) -> InitFn {
        if ob.is_none() {
            get_null
        }
        // bool must be checked before int because Python bool is an instance of int.
        else if ob.is_instance_of::<PyBool>() {
            get_bool
        } else if ob.is_instance_of::<PyInt>() {
            get_int
        } else if ob.is_instance_of::<PyFloat>() {
            get_float
        } else if ob.is_instance_of::<PyString>() {
            get_str
        } else if ob.is_instance_of::<PyBytes>() {
            get_bytes
        } else if ob.is_instance_of::<PyList>() || ob.is_instance_of::<PyTuple>() {
            get_list
        } else if ob.is_instance_of::<PyDict>() {
            get_struct
        } else if ob.hasattr(intern!(py, "_s")).unwrap() {
            get_list_from_series
        } else {
            let type_name = ob.get_type().qualname().unwrap();
            match &*type_name {
                // Can't use pyo3::types::PyDateTime with abi3-py37 feature,
                // so need this workaround instead of `isinstance(ob, datetime)`.
                "date" => get_date as InitFn,
                "time" => get_time as InitFn,
                "datetime" => get_datetime as InitFn,
                "timedelta" => get_timedelta as InitFn,
                "Decimal" => get_decimal as InitFn,
                "range" => get_list as InitFn,
                _ => {
                    // Support NumPy scalars.
                    if ob.extract::<i64>().is_ok() || ob.extract::<u64>().is_ok() {
                        return get_int as InitFn;
                    } else if ob.extract::<f64>().is_ok() {
                        return get_float as InitFn;
                    }

                    // Support custom subclasses of datetime/date.
                    let ancestors = ob.get_type().getattr(intern!(py, "__mro__")).unwrap();
                    let ancestors_str_iter = ancestors
                        .iter()
                        .unwrap()
                        .map(|b| b.unwrap().str().unwrap().to_string());
                    for c in ancestors_str_iter {
                        match &*c {
                            // datetime must be checked before date because
                            // Python datetime is an instance of date.
                            "<class 'datetime.datetime'>" => return get_datetime as InitFn,
                            "<class 'datetime.date'>" => return get_date as InitFn,
                            _ => (),
                        }
                    }

                    get_object as InitFn
                },
            }
        }
    }

    let type_object_ptr = ob.get_type().as_type_ptr() as usize;

    Python::with_gil(|py| {
        LUT.with_gil(py, |lut| {
            let convert_fn = lut
                .entry(type_object_ptr)
                .or_insert_with(|| get_conversion_function(ob, py));
            convert_fn(ob, strict)
        })
    })
}
