use std::fmt::Write;

use crate::prelude::*;

fn any_values_to_primitive<T: PolarsNumericType>(avs: &[AnyValue]) -> ChunkedArray<T> {
    avs.iter()
        .map(|av| av.extract::<T::Native>())
        .collect_trusted()
}

fn any_values_to_utf8(avs: &[AnyValue]) -> Utf8Chunked {
    let mut builder = Utf8ChunkedBuilder::new("", avs.len(), avs.len() * 10);

    // amortize allocations
    let mut owned = String::new();

    for av in avs {
        match av {
            AnyValue::Utf8(s) => builder.append_value(s),
            AnyValue::Utf8Owned(s) => builder.append_value(s),
            AnyValue::Null => builder.append_null(),
            #[cfg(feature = "dtype-binary")]
            AnyValue::Binary(_) | AnyValue::BinaryOwned(_) => builder.append_null(),
            av => {
                owned.clear();
                write!(owned, "{av}").unwrap();
                builder.append_value(&owned);
            }
        }
    }
    builder.finish()
}

#[cfg(feature = "dtype-binary")]
fn any_values_to_binary(avs: &[AnyValue]) -> BinaryChunked {
    avs.iter()
        .map(|av| match av {
            AnyValue::Binary(s) => Some(*s),
            AnyValue::BinaryOwned(s) => Some(&**s),
            _ => None,
        })
        .collect_trusted()
}

fn any_values_to_bool(avs: &[AnyValue]) -> BooleanChunked {
    avs.iter()
        .map(|av| match av {
            AnyValue::Boolean(b) => Some(*b),
            _ => None,
        })
        .collect_trusted()
}

fn any_values_to_list(avs: &[AnyValue], inner_type: &DataType) -> ListChunked {
    // this is handled downstream. The builder will choose the first non null type
    if inner_type == &DataType::Null {
        avs.iter()
            .map(|av| match av {
                AnyValue::List(b) => Some(b.clone()),
                _ => None,
            })
            .collect_trusted()
    }
    // make sure that wrongly inferred anyvalues don't deviate from the datatype
    else {
        avs.iter()
            .map(|av| match av {
                AnyValue::List(b) => {
                    if b.dtype() == inner_type {
                        Some(b.clone())
                    } else {
                        match b.cast(inner_type) {
                            Ok(out) => Some(out),
                            Err(_) => Some(Series::full_null(b.name(), b.len(), inner_type)),
                        }
                    }
                }
                _ => None,
            })
            .collect_trusted()
    }
}

impl<'a, T: AsRef<[AnyValue<'a>]>> NamedFrom<T, [AnyValue<'a>]> for Series {
    fn new(name: &str, v: T) -> Self {
        let av = v.as_ref();
        Series::from_any_values(name, av).unwrap()
    }
}

impl Series {
    pub fn from_any_values_and_dtype(
        name: &str,
        av: &[AnyValue],
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        let mut s = match dtype {
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => any_values_to_primitive::<Int8Type>(av).into_series(),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => any_values_to_primitive::<Int16Type>(av).into_series(),
            DataType::Int32 => any_values_to_primitive::<Int32Type>(av).into_series(),
            DataType::Int64 => any_values_to_primitive::<Int64Type>(av).into_series(),
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => any_values_to_primitive::<UInt8Type>(av).into_series(),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => any_values_to_primitive::<UInt16Type>(av).into_series(),
            DataType::UInt32 => any_values_to_primitive::<UInt32Type>(av).into_series(),
            DataType::UInt64 => any_values_to_primitive::<UInt64Type>(av).into_series(),
            DataType::Float32 => any_values_to_primitive::<Float32Type>(av).into_series(),
            DataType::Float64 => any_values_to_primitive::<Float64Type>(av).into_series(),
            DataType::Utf8 => any_values_to_utf8(av).into_series(),
            #[cfg(feature = "dtype-binary")]
            DataType::Binary => any_values_to_binary(av).into_series(),
            DataType::Boolean => any_values_to_bool(av).into_series(),
            #[cfg(feature = "dtype-date")]
            DataType::Date => any_values_to_primitive::<Int32Type>(av)
                .into_date()
                .into_series(),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(tu, tz) => any_values_to_primitive::<Int64Type>(av)
                .into_datetime(*tu, (*tz).clone())
                .into_series(),
            #[cfg(feature = "dtype-time")]
            DataType::Time => any_values_to_primitive::<Int64Type>(av)
                .into_time()
                .into_series(),
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(tu) => any_values_to_primitive::<Int64Type>(av)
                .into_duration(*tu)
                .into_series(),
            DataType::List(inner) => any_values_to_list(av, inner).into_series(),
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(dtype_fields) => {
                // fast path for empty structs
                if dtype_fields.is_empty() {
                    return Ok(StructChunked::full_null(name, av.len()).into_series());
                }
                // the physical series fields of the struct
                let mut series_fields = Vec::with_capacity(dtype_fields.len());
                for (i, field) in dtype_fields.iter().enumerate() {
                    let mut field_avs = Vec::with_capacity(av.len());

                    for av in av.iter() {
                        match av {
                            AnyValue::StructOwned(payload) => {
                                // TODO: optimize
                                let av_fields = &payload.1;
                                let av_values = &payload.0;

                                let mut append_by_search = || {
                                    // search for the name
                                    let mut pushed = false;
                                    for (av_fld, av_val) in av_fields.iter().zip(av_values) {
                                        if av_fld.name == field.name {
                                            field_avs.push(av_val.clone());
                                            pushed = true;
                                            break;
                                        }
                                    }
                                    if !pushed {
                                        field_avs.push(AnyValue::Null)
                                    }
                                };

                                // all fields are available in this single value
                                // we can use the index to get value
                                if dtype_fields.len() == av_fields.len() {
                                    let mut search = false;
                                    for (l, r) in dtype_fields.iter().zip(av_fields.iter()) {
                                        if l.name() != r.name() {
                                            search = true;
                                        }
                                    }
                                    if search {
                                        append_by_search()
                                    } else {
                                        let av_val =
                                            av_values.get(i).cloned().unwrap_or(AnyValue::Null);
                                        field_avs.push(av_val)
                                    }
                                }
                                // not all fields are available, we search the proper field
                                else {
                                    // search for the name
                                    append_by_search()
                                }
                            }
                            _ => field_avs.push(AnyValue::Null),
                        }
                    }
                    // if the inferred dtype is null, we let auto inference work
                    let s = if matches!(field.dtype, DataType::Null) {
                        Series::new(field.name(), &field_avs)
                    } else {
                        Series::from_any_values_and_dtype(field.name(), &field_avs, &field.dtype)?
                    };
                    series_fields.push(s)
                }
                return Ok(StructChunked::new(name, &series_fields)
                    .unwrap()
                    .into_series());
            }
            #[cfg(feature = "object")]
            DataType::Object(_) => {
                use crate::chunked_array::object::registry;
                let converter = registry::get_object_converter();
                let mut builder = registry::get_object_builder(name, av.len());
                for av in av {
                    if let AnyValue::Object(val) = av {
                        builder.append_value(val.as_any())
                    } else {
                        let any = converter(av.as_borrowed());
                        builder.append_value(&*any)
                    }
                }
                return Ok(builder.to_series());
            }
            DataType::Null => Series::full_null(name, av.len(), &DataType::Null),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                let ca = if let Some(single_av) = av.first() {
                    match single_av {
                        AnyValue::Utf8(_) | AnyValue::Utf8Owned(_) => any_values_to_utf8(av),
                        _ => {
                            return Err(PolarsError::ComputeError(
                                format!(
                                    "categorical dtype with AnyValues of dtype: {} not supported",
                                    single_av.dtype()
                                )
                                .into(),
                            ))
                        }
                    }
                } else {
                    Utf8Chunked::full("", "", 0)
                };

                ca.cast(&DataType::Categorical(None)).unwrap()
            }
            dt => panic!("{dt:?} not supported"),
        };
        s.rename(name);
        Ok(s)
    }

    pub fn from_any_values(name: &str, av: &[AnyValue]) -> PolarsResult<Series> {
        match av.iter().find(|av| !matches!(av, AnyValue::Null)) {
            None => Ok(Series::full_null(name, av.len(), &DataType::Int32)),
            Some(av_) => {
                let dtype: DataType = av_.into();
                Series::from_any_values_and_dtype(name, av, &dtype)
            }
        }
    }
}

impl<'a> From<&AnyValue<'a>> for DataType {
    fn from(val: &AnyValue<'a>) -> Self {
        use AnyValue::*;
        match val {
            Null => DataType::Null,
            Boolean(_) => DataType::Boolean,
            Utf8(_) | Utf8Owned(_) => DataType::Utf8,
            #[cfg(feature = "dtype-binary")]
            Binary(_) | BinaryOwned(_) => DataType::Binary,
            UInt32(_) => DataType::UInt32,
            UInt64(_) => DataType::UInt64,
            Int32(_) => DataType::Int32,
            Int64(_) => DataType::Int64,
            Float32(_) => DataType::Float32,
            Float64(_) => DataType::Float64,
            #[cfg(feature = "dtype-date")]
            Date(_) => DataType::Date,
            #[cfg(feature = "dtype-datetime")]
            Datetime(_, tu, tz) => DataType::Datetime(*tu, (*tz).clone()),
            #[cfg(feature = "dtype-time")]
            Time(_) => DataType::Time,
            List(s) => DataType::List(Box::new(s.dtype().clone())),
            #[cfg(feature = "dtype-struct")]
            StructOwned(payload) => DataType::Struct(payload.1.to_vec()),
            #[cfg(feature = "dtype-struct")]
            Struct(_, _, flds) => DataType::Struct(flds.to_vec()),
            #[cfg(feature = "dtype-duration")]
            Duration(_, tu) => DataType::Duration(*tu),
            UInt8(_) => DataType::UInt8,
            UInt16(_) => DataType::UInt16,
            Int8(_) => DataType::Int8,
            Int16(_) => DataType::Int16,
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, rev_map, arr) => {
                if arr.is_null() {
                    DataType::Categorical(Some(Arc::new((*rev_map).clone())))
                } else {
                    let array = unsafe { arr.deref_unchecked().clone() };
                    let rev_map = RevMapping::Local(array);
                    DataType::Categorical(Some(Arc::new(rev_map)))
                }
            }
            #[cfg(feature = "object")]
            Object(o) => DataType::Object(o.type_name()),
            #[cfg(feature = "object")]
            ObjectOwned(o) => DataType::Object(o.0.type_name()),
        }
    }
}
