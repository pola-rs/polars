use std::borrow::{Borrow, Cow};
use std::sync::Arc;

use chrono::{
    DateTime, Datelike, FixedOffset, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, Timelike,
};
use chrono_tz::Tz;
#[cfg(feature = "object")]
use polars::chunked_array::object::PolarsObjectSafe;
#[cfg(feature = "object")]
use polars::datatypes::OwnedObject;
use polars::datatypes::{DataType, Field, PlHashMap, TimeUnit};
use polars::prelude::{AnyValue, PlSmallStr, Series};
use polars_core::utils::any_values_to_supertype_and_n_dtypes;
use polars_core::utils::arrow::temporal_conversions::date32_to_date;
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PySequence, PyString, PyTuple, PyType,
};
use pyo3::{intern, IntoPyObjectExt};

use super::datetime::{
    datetime_to_py_object, elapsed_offset_to_timedelta, nanos_since_midnight_to_naivetime,
};
use super::{decimal_to_digits, struct_dict, ObjectValue, Wrap};
use crate::error::PyPolarsErr;
use crate::py_modules::{pl_series, pl_utils};
use crate::series::PySeries;

impl<'py> IntoPyObject<'py> for Wrap<AnyValue<'_>> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        any_value_into_py_object(self.0, py)
    }
}

impl<'py> IntoPyObject<'py> for &Wrap<AnyValue<'_>> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        self.clone().into_pyobject(py)
    }
}

impl<'py> FromPyObject<'py> for Wrap<AnyValue<'py>> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        py_object_to_any_value(ob, true, true).map(Wrap)
    }
}

pub(crate) fn any_value_into_py_object<'py>(
    av: AnyValue,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let utils = pl_utils(py).bind(py);
    match av {
        AnyValue::UInt8(v) => v.into_bound_py_any(py),
        AnyValue::UInt16(v) => v.into_bound_py_any(py),
        AnyValue::UInt32(v) => v.into_bound_py_any(py),
        AnyValue::UInt64(v) => v.into_bound_py_any(py),
        AnyValue::Int8(v) => v.into_bound_py_any(py),
        AnyValue::Int16(v) => v.into_bound_py_any(py),
        AnyValue::Int32(v) => v.into_bound_py_any(py),
        AnyValue::Int64(v) => v.into_bound_py_any(py),
        AnyValue::Int128(v) => v.into_bound_py_any(py),
        AnyValue::Float32(v) => v.into_bound_py_any(py),
        AnyValue::Float64(v) => v.into_bound_py_any(py),
        AnyValue::Null => py.None().into_bound_py_any(py),
        AnyValue::Boolean(v) => v.into_bound_py_any(py),
        AnyValue::String(v) => v.into_bound_py_any(py),
        AnyValue::StringOwned(v) => v.into_bound_py_any(py),
        AnyValue::Categorical(idx, rev, arr) | AnyValue::Enum(idx, rev, arr) => {
            let s = if arr.is_null() {
                rev.get(idx)
            } else {
                unsafe { arr.deref_unchecked().value(idx as usize) }
            };
            s.into_bound_py_any(py)
        },
        AnyValue::CategoricalOwned(idx, rev, arr) | AnyValue::EnumOwned(idx, rev, arr) => {
            let s = if arr.is_null() {
                rev.get(idx)
            } else {
                unsafe { arr.deref_unchecked().value(idx as usize) }
            };
            s.into_bound_py_any(py)
        },
        AnyValue::Date(v) => {
            let date = date32_to_date(v);
            date.into_bound_py_any(py)
        },
        AnyValue::Datetime(v, time_unit, time_zone) => {
            datetime_to_py_object(py, v, time_unit, time_zone)
        },
        AnyValue::DatetimeOwned(v, time_unit, time_zone) => {
            datetime_to_py_object(py, v, time_unit, time_zone.as_ref().map(AsRef::as_ref))
        },
        AnyValue::Duration(v, time_unit) => {
            let time_delta = elapsed_offset_to_timedelta(v, time_unit);
            time_delta.into_bound_py_any(py)
        },
        AnyValue::Time(v) => nanos_since_midnight_to_naivetime(v).into_bound_py_any(py),
        AnyValue::Array(v, _) | AnyValue::List(v) => PySeries::new(v).to_list(py),
        ref av @ AnyValue::Struct(_, _, flds) => {
            Ok(struct_dict(py, av._iter_struct_av(), flds)?.into_any())
        },
        AnyValue::StructOwned(payload) => {
            Ok(struct_dict(py, payload.0.into_iter(), &payload.1)?.into_any())
        },
        #[cfg(feature = "object")]
        AnyValue::Object(v) => {
            let object = v.as_any().downcast_ref::<ObjectValue>().unwrap();
            Ok(object.inner.clone_ref(py).into_bound(py))
        },
        #[cfg(feature = "object")]
        AnyValue::ObjectOwned(v) => {
            let object = v.0.as_any().downcast_ref::<ObjectValue>().unwrap();
            Ok(object.inner.clone_ref(py).into_bound(py))
        },
        AnyValue::Binary(v) => PyBytes::new(py, v).into_bound_py_any(py),
        AnyValue::BinaryOwned(v) => PyBytes::new(py, &v).into_bound_py_any(py),
        AnyValue::Decimal(v, scale) => {
            let convert = utils.getattr(intern!(py, "to_py_decimal"))?;
            const N: usize = 3;
            let mut buf = [0_u128; N];
            let n_digits = decimal_to_digits(v.abs(), &mut buf);
            let buf = unsafe {
                std::slice::from_raw_parts(
                    buf.as_slice().as_ptr() as *const u8,
                    N * size_of::<u128>(),
                )
            };
            let digits = PyTuple::new(py, buf.iter().take(n_digits))?;
            convert.call1((v.is_negative() as u8, digits, n_digits, -(scale as i32)))
        },
    }
}

/// Holds a Python type object and implements hashing / equality based on the pointer address of the
/// type object. This is used as a hashtable key instead of only the `usize` pointer value, as we
/// need to hold a ref to the Python type object to keep it alive.
#[derive(Debug)]
pub struct TypeObjectKey {
    #[allow(unused)]
    type_object: Py<PyType>,
    /// We need to store this in a field for `Borrow<usize>`
    address: usize,
}

impl TypeObjectKey {
    fn new(type_object: Py<PyType>) -> Self {
        let address = type_object.as_ptr() as usize;
        Self {
            type_object,
            address,
        }
    }
}

impl PartialEq for TypeObjectKey {
    fn eq(&self, other: &Self) -> bool {
        self.address == other.address
    }
}

impl Eq for TypeObjectKey {}

impl std::borrow::Borrow<usize> for TypeObjectKey {
    fn borrow(&self) -> &usize {
        &self.address
    }
}

impl std::hash::Hash for TypeObjectKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let v: &usize = self.borrow();
        v.hash(state)
    }
}

type InitFn = for<'py> fn(&Bound<'py, PyAny>, bool) -> PyResult<AnyValue<'py>>;
pub(crate) static LUT: crate::gil_once_cell::GILOnceCell<PlHashMap<TypeObjectKey, InitFn>> =
    crate::gil_once_cell::GILOnceCell::new();

/// Convert a Python object to an [`AnyValue`].
pub(crate) fn py_object_to_any_value<'py>(
    ob: &Bound<'py, PyAny>,
    strict: bool,
    allow_object: bool,
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
        } else if let Ok(v) = ob.extract::<i128>() {
            Ok(AnyValue::Int128(v))
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
        let py = ob.py();
        let tzinfo = ob.getattr(intern!(py, "tzinfo"))?;

        if tzinfo.is_none() {
            let datetime = ob.extract::<NaiveDateTime>()?;
            let delta = datetime - NaiveDateTime::UNIX_EPOCH;
            let timestamp = delta.num_microseconds().unwrap();
            return Ok(AnyValue::Datetime(timestamp, TimeUnit::Microseconds, None));
        }

        let (timestamp, tz) = if tzinfo.hasattr(intern!(py, "key"))? {
            let datetime = ob.extract::<DateTime<Tz>>()?;
            let tz = datetime.timezone().name().into();
            if datetime.year() >= 2100 {
                // chrono-tz does not support dates after 2100
                // https://github.com/chronotope/chrono-tz/issues/135
                (
                    pl_utils(py)
                        .bind(py)
                        .getattr(intern!(py, "datetime_to_int"))?
                        .call1((ob, intern!(py, "us")))?
                        .extract::<i64>()?,
                    tz,
                )
            } else {
                let delta = datetime.to_utc() - DateTime::UNIX_EPOCH;
                (delta.num_microseconds().unwrap(), tz)
            }
        } else {
            let datetime = ob.extract::<DateTime<FixedOffset>>()?;
            let delta = datetime.to_utc() - DateTime::UNIX_EPOCH;
            (delta.num_microseconds().unwrap(), "UTC".into())
        };

        Ok(AnyValue::DatetimeOwned(
            timestamp,
            TimeUnit::Microseconds,
            Some(Arc::new(tz)),
        ))
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
            let kwargs = PyDict::new(py);
            kwargs.set_item("strict", strict)?;
            let s = pl_series(py).call(py, (ob,), Some(&kwargs))?;
            get_list_from_series(s.bind(py), strict)
        }

        if ob.is_empty()? {
            Ok(AnyValue::List(Series::new_empty(
                PlSmallStr::EMPTY,
                &DataType::Null,
            )))
        } else if ob.is_instance_of::<PyList>() | ob.is_instance_of::<PyTuple>() {
            const INFER_SCHEMA_LENGTH: usize = 25;

            let list = ob.downcast::<PySequence>()?;

            let mut avs = Vec::with_capacity(INFER_SCHEMA_LENGTH);
            let mut iter = list.try_iter()?;
            let mut items = Vec::with_capacity(INFER_SCHEMA_LENGTH);
            for item in (&mut iter).take(INFER_SCHEMA_LENGTH) {
                items.push(item?);
                let av = py_object_to_any_value(items.last().unwrap(), strict, true)?;
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
                    let av = py_object_to_any_value(rest.last().unwrap(), strict, true)?;
                    avs.push(av)
                }

                let s = Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &avs, &dtype, strict)
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
            let val = py_object_to_any_value(&v, strict, true)?;
            let dtype = val.dtype();
            keys.push(Field::new(key.as_ref().into(), dtype));
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
    fn get_conversion_function(
        ob: &Bound<'_, PyAny>,
        py: Python<'_>,
        allow_object: bool,
    ) -> PyResult<InitFn> {
        if ob.is_none() {
            Ok(get_null)
        }
        // bool must be checked before int because Python bool is an instance of int.
        else if ob.is_instance_of::<PyBool>() {
            Ok(get_bool)
        } else if ob.is_instance_of::<PyInt>() {
            Ok(get_int)
        } else if ob.is_instance_of::<PyFloat>() {
            Ok(get_float)
        } else if ob.is_instance_of::<PyString>() {
            Ok(get_str)
        } else if ob.is_instance_of::<PyBytes>() {
            Ok(get_bytes)
        } else if ob.is_instance_of::<PyList>() || ob.is_instance_of::<PyTuple>() {
            Ok(get_list)
        } else if ob.is_instance_of::<PyDict>() {
            Ok(get_struct)
        } else {
            let ob_type = ob.get_type();
            let type_name = ob_type.fully_qualified_name()?.to_string();
            match type_name.as_str() {
                // Can't use pyo3::types::PyDateTime with abi3-py37 feature,
                // so need this workaround instead of `isinstance(ob, datetime)`.
                "datetime.date" => Ok(get_date as InitFn),
                "datetime.time" => Ok(get_time as InitFn),
                "datetime.datetime" => Ok(get_datetime as InitFn),
                "datetime.timedelta" => Ok(get_timedelta as InitFn),
                "decimal.Decimal" => Ok(get_decimal as InitFn),
                "range" => Ok(get_list as InitFn),
                _ => {
                    // Support NumPy scalars.
                    if ob.extract::<i64>().is_ok() || ob.extract::<u64>().is_ok() {
                        return Ok(get_int as InitFn);
                    } else if ob.extract::<f64>().is_ok() {
                        return Ok(get_float as InitFn);
                    }

                    // Support custom subclasses of datetime/date.
                    let ancestors = ob_type.getattr(intern!(py, "__mro__"))?;
                    let ancestors_str_iter = ancestors
                        .try_iter()?
                        .map(|b| b.unwrap().str().unwrap().to_string());
                    for c in ancestors_str_iter {
                        match &*c {
                            // datetime must be checked before date because
                            // Python datetime is an instance of date.
                            "<class 'datetime.datetime'>" => {
                                return Ok(get_datetime as InitFn);
                            },
                            "<class 'datetime.date'>" => return Ok(get_date as InitFn),
                            "<class 'datetime.timedelta'>" => return Ok(get_timedelta as InitFn),
                            "<class 'datetime.time'>" => return Ok(get_time as InitFn),
                            _ => (),
                        }
                    }

                    if allow_object {
                        Ok(get_object as InitFn)
                    } else {
                        Err(PyValueError::new_err(format!("Cannot convert {ob}")))
                    }
                },
            }
        }
    }

    let py_type = ob.get_type();
    let py_type_address = py_type.as_ptr() as usize;

    Python::with_gil(move |py| {
        LUT.with_gil(py, move |lut| {
            if !lut.contains_key(&py_type_address) {
                let k = TypeObjectKey::new(py_type.clone().unbind());

                assert_eq!(k.address, py_type_address);

                unsafe {
                    lut.insert_unique_unchecked(k, get_conversion_function(ob, py, allow_object)?);
                }
            }

            let conversion_func = lut.get(&py_type_address).unwrap();
            conversion_func(ob, strict)
        })
    })
}
