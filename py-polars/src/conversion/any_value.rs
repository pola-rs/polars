#[cfg(feature = "object")]
use polars::chunked_array::object::PolarsObjectSafe;
use polars::datatypes::{DataType, Field, OwnedObject, PlHashMap, TimeUnit};
use polars::prelude::{AnyValue, Series};
use polars_core::frame::row::any_values_to_dtype;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList, PySequence, PyString, PyTuple, PyType};

use super::{decimal_to_digits, struct_dict, ObjectValue, Wrap};
use crate::error::PyPolarsErr;
use crate::py_modules::{SERIES, UTILS};
use crate::series::PySeries;

impl IntoPy<PyObject> for Wrap<AnyValue<'_>> {
    fn into_py(self, py: Python) -> PyObject {
        let utils = UTILS.as_ref(py);
        match self.0 {
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
                let convert = utils.getattr(intern!(py, "_to_python_date")).unwrap();
                convert.call1((v,)).unwrap().into_py(py)
            },
            AnyValue::Datetime(v, time_unit, time_zone) => {
                let convert = utils.getattr(intern!(py, "_to_python_datetime")).unwrap();
                let time_unit = time_unit.to_ascii();
                convert
                    .call1((v, time_unit, time_zone.as_ref().map(|s| s.as_str())))
                    .unwrap()
                    .into_py(py)
            },
            AnyValue::Duration(v, time_unit) => {
                let convert = utils.getattr(intern!(py, "_to_python_timedelta")).unwrap();
                let time_unit = time_unit.to_ascii();
                convert.call1((v, time_unit)).unwrap().into_py(py)
            },
            AnyValue::Time(v) => {
                let convert = utils.getattr(intern!(py, "_to_python_time")).unwrap();
                convert.call1((v,)).unwrap().into_py(py)
            },
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
                let convert = utils.getattr(intern!(py, "_to_python_decimal")).unwrap();
                const N: usize = 3;
                let mut buf = [0_u128; N];
                let n_digits = decimal_to_digits(v.abs(), &mut buf);
                let buf = unsafe {
                    std::slice::from_raw_parts(
                        buf.as_slice().as_ptr() as *const u8,
                        N * std::mem::size_of::<u128>(),
                    )
                };
                let digits = PyTuple::new(py, buf.iter().take(n_digits));
                convert
                    .call1((v.is_negative() as u8, digits, n_digits, -(scale as i32)))
                    .unwrap()
                    .into_py(py)
            },
        }
    }
}

impl ToPyObject for Wrap<AnyValue<'_>> {
    fn to_object(&self, py: Python) -> PyObject {
        self.clone().into_py(py)
    }
}

type TypeObjectPtr = usize;
type InitFn = fn(&PyAny) -> PyResult<Wrap<AnyValue<'_>>>;
pub(crate) static LUT: crate::gil_once_cell::GILOnceCell<PlHashMap<TypeObjectPtr, InitFn>> =
    crate::gil_once_cell::GILOnceCell::new();

impl<'s> FromPyObject<'s> for Wrap<AnyValue<'s>> {
    fn extract(ob: &'s PyAny) -> PyResult<Self> {
        // conversion functions
        fn get_bool(ob: &PyAny) -> PyResult<Wrap<AnyValue<'_>>> {
            Ok(AnyValue::Boolean(ob.extract::<bool>().unwrap()).into())
        }

        fn get_int(ob: &PyAny) -> PyResult<Wrap<AnyValue<'_>>> {
            // can overflow
            match ob.extract::<i64>() {
                Ok(v) => Ok(AnyValue::Int64(v).into()),
                Err(_) => Ok(AnyValue::UInt64(ob.extract::<u64>()?).into()),
            }
        }

        fn get_float(ob: &PyAny) -> PyResult<Wrap<AnyValue<'_>>> {
            Ok(AnyValue::Float64(ob.extract::<f64>().unwrap()).into())
        }

        fn get_str(ob: &PyAny) -> PyResult<Wrap<AnyValue<'_>>> {
            let value = ob.extract::<&str>().unwrap();
            Ok(AnyValue::String(value).into())
        }

        fn get_struct(ob: &PyAny) -> PyResult<Wrap<AnyValue<'_>>> {
            let dict = ob.downcast::<PyDict>().unwrap();
            let len = dict.len();
            let mut keys = Vec::with_capacity(len);
            let mut vals = Vec::with_capacity(len);
            for (k, v) in dict.into_iter() {
                let key = k.extract::<&str>()?;
                let val = v.extract::<Wrap<AnyValue>>()?.0;
                let dtype = DataType::from(&val);
                keys.push(Field::new(key, dtype));
                vals.push(val)
            }
            Ok(Wrap(AnyValue::StructOwned(Box::new((vals, keys)))))
        }

        fn get_list(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            fn get_list_with_constructor(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
                // Use the dedicated constructor
                // this constructor is able to go via dedicated type constructors
                // so it can be much faster
                Python::with_gil(|py| {
                    let s = SERIES.call1(py, (ob,))?;
                    get_series_el(s.as_ref(py))
                })
            }

            if ob.is_empty()? {
                Ok(Wrap(AnyValue::List(Series::new_empty("", &DataType::Null))))
            } else if ob.is_instance_of::<PyList>() | ob.is_instance_of::<PyTuple>() {
                let list = ob.downcast::<PySequence>().unwrap();

                let mut avs = Vec::with_capacity(25);
                let mut iter = list.iter()?;

                for item in (&mut iter).take(25) {
                    avs.push(item?.extract::<Wrap<AnyValue>>()?.0)
                }

                let (dtype, n_types) = any_values_to_dtype(&avs).map_err(PyPolarsErr::from)?;

                // we only take this path if there is no question of the data-type
                if dtype.is_primitive() && n_types == 1 {
                    get_list_with_constructor(ob)
                } else {
                    // push the rest
                    avs.reserve(list.len()?);
                    for item in iter {
                        avs.push(item?.extract::<Wrap<AnyValue>>()?.0)
                    }

                    let s = Series::from_any_values_and_dtype("", &avs, &dtype, true)
                        .map_err(PyPolarsErr::from)?;
                    Ok(Wrap(AnyValue::List(s)))
                }
            } else {
                // range will take this branch
                get_list_with_constructor(ob)
            }
        }

        fn get_series_el(ob: &PyAny) -> PyResult<Wrap<AnyValue<'static>>> {
            let py_pyseries = ob.getattr(intern!(ob.py(), "_s")).unwrap();
            let series = py_pyseries.extract::<PySeries>().unwrap().series;
            Ok(Wrap(AnyValue::List(series)))
        }

        fn get_bin(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            let value = ob.extract::<&[u8]>().unwrap();
            Ok(AnyValue::Binary(value).into())
        }

        fn get_null(_ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            Ok(AnyValue::Null.into())
        }

        fn get_timedelta(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            Python::with_gil(|py| {
                let td = UTILS
                    .as_ref(py)
                    .getattr(intern!(py, "_timedelta_to_pl_timedelta"))
                    .unwrap()
                    .call1((ob, intern!(py, "us")))
                    .unwrap();
                let v = td.extract::<i64>().unwrap();
                Ok(Wrap(AnyValue::Duration(v, TimeUnit::Microseconds)))
            })
        }

        fn get_time(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            Python::with_gil(|py| {
                let time = UTILS
                    .as_ref(py)
                    .getattr(intern!(py, "_time_to_pl_time"))
                    .unwrap()
                    .call1((ob,))
                    .unwrap();
                let v = time.extract::<i64>().unwrap();
                Ok(Wrap(AnyValue::Time(v)))
            })
        }

        fn get_decimal(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            let (sign, digits, exp): (i8, Vec<u8>, i32) = ob
                .call_method0(intern!(ob.py(), "as_tuple"))
                .unwrap()
                .extract()
                .unwrap();
            // note: using Vec<u8> is not the most efficient thing here (input is a tuple)
            let (mut v, scale) = abs_decimal_from_digits(digits, exp).ok_or_else(|| {
                PyErr::from(PyPolarsErr::Other(
                    "Decimal is too large to fit in Decimal128".into(),
                ))
            })?;
            if sign > 0 {
                v = -v; // won't overflow since -i128::MAX > i128::MIN
            }
            Ok(Wrap(AnyValue::Decimal(v, scale)))
        }

        fn get_object(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
            #[cfg(feature = "object")]
            {
                // this is slow, but hey don't use objects
                let v = &ObjectValue { inner: ob.into() };
                Ok(Wrap(AnyValue::ObjectOwned(OwnedObject(v.to_boxed()))))
            }
            #[cfg(not(feature = "object"))]
            {
                panic!("activate object")
            }
        }

        // TYPE key
        let type_object_ptr = PyType::as_type_ptr(ob.get_type()) as usize;

        Python::with_gil(|py| {
            LUT.with_gil(py, |lut| {
                // get the conversion function
                let convert_fn = lut.entry(type_object_ptr).or_insert_with(
                    // This only runs if type is not in LUT
                    || {
                        if ob.is_instance_of::<PyBool>() {
                            get_bool
                            // TODO: this heap allocs on failure
                        } else if ob.extract::<i64>().is_ok() || ob.extract::<u64>().is_ok() {
                            get_int
                        } else if ob.is_instance_of::<PyFloat>() {
                            get_float
                        } else if ob.is_instance_of::<PyString>() {
                            get_str
                        } else if ob.is_instance_of::<PyDict>() {
                            get_struct
                        } else if ob.is_instance_of::<PyList>() || ob.is_instance_of::<PyTuple>() {
                            get_list
                        } else if ob.hasattr(intern!(py, "_s")).unwrap() {
                            get_series_el
                        }
                        // TODO: this heap allocs on failure
                        else if ob.extract::<&'s [u8]>().is_ok() {
                            get_bin
                        } else if ob.is_none() {
                            get_null
                        } else {
                            let type_name = ob.get_type().name().unwrap();
                            match type_name {
                                "datetime" => convert_datetime,
                                "date" => convert_date,
                                "timedelta" => get_timedelta,
                                "time" => get_time,
                                "Decimal" => get_decimal,
                                "range" => get_list,
                                _ => {
                                    // special branch for np.float as this fails isinstance float
                                    if ob.extract::<f64>().is_ok() {
                                        return get_float;
                                    }

                                    // Can't use pyo3::types::PyDateTime with abi3-py37 feature,
                                    // so need this workaround instead of `isinstance(ob, datetime)`.
                                    let bases = ob
                                        .get_type()
                                        .getattr(intern!(py, "__bases__"))
                                        .unwrap()
                                        .iter()
                                        .unwrap();
                                    for base in bases {
                                        let parent_type =
                                            base.unwrap().str().unwrap().to_str().unwrap();
                                        match parent_type {
                                            "<class 'datetime.datetime'>" => {
                                                // `datetime.datetime` is a subclass of `datetime.date`,
                                                // so need to check `datetime.datetime` first
                                                return convert_datetime;
                                            },
                                            "<class 'datetime.date'>" => {
                                                return convert_date;
                                            },
                                            _ => (),
                                        }
                                    }

                                    get_object
                                },
                            }
                        }
                    },
                );

                convert_fn(ob)
            })
        })
    }
}

fn convert_date(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
    Python::with_gil(|py| {
        let date = UTILS
            .as_ref(py)
            .getattr(intern!(py, "_date_to_pl_date"))
            .unwrap()
            .call1((ob,))
            .unwrap();
        let v = date.extract::<i32>().unwrap();
        Ok(Wrap(AnyValue::Date(v)))
    })
}
fn convert_datetime(ob: &PyAny) -> PyResult<Wrap<AnyValue>> {
    Python::with_gil(|py| {
        // windows
        #[cfg(target_arch = "windows")]
        let (seconds, microseconds) = {
            let convert = UTILS
                .getattr(py, intern!(py, "_datetime_for_any_value_windows"))
                .unwrap();
            let out = convert.call1(py, (ob,)).unwrap();
            let out: (i64, i64) = out.extract(py).unwrap();
            out
        };
        // unix
        #[cfg(not(target_arch = "windows"))]
        let (seconds, microseconds) = {
            let convert = UTILS
                .getattr(py, intern!(py, "_datetime_for_any_value"))
                .unwrap();
            let out = convert.call1(py, (ob,)).unwrap();
            let out: (i64, i64) = out.extract(py).unwrap();
            out
        };

        // s to us
        let mut v = seconds * 1_000_000;
        v += microseconds;

        // choose "us" as that is python's default unit
        Ok(AnyValue::Datetime(v, TimeUnit::Microseconds, &None).into())
    })
}

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
    // we only support non-negative scale (=> non-positive exponent)
    let scale = if exp > 0 {
        // the decimal may be in a non-canonical representation, try to fix it first
        v = 10_i128
            .checked_pow(exp as u32)
            .and_then(|factor| v.checked_mul(factor))?;
        0
    } else {
        (-exp) as usize
    };
    // TODO: do we care for checking if it fits in MAX_ABS_DEC? (if we set precision to None anyway?)
    (v <= MAX_ABS_DEC).then_some((v, scale))
}
