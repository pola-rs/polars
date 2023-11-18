use std::fmt::Write;

use crate::prelude::*;

fn any_values_to_primitive<T: PolarsNumericType>(avs: &[AnyValue]) -> ChunkedArray<T> {
    avs.iter()
        .map(|av| av.extract::<T::Native>())
        .collect_trusted()
}

fn any_values_to_utf8(avs: &[AnyValue], strict: bool) -> PolarsResult<Utf8Chunked> {
    let mut builder = Utf8ChunkedBuilder::new("", avs.len(), avs.len() * 10);

    // amortize allocations
    let mut owned = String::new();

    for av in avs {
        match av {
            AnyValue::Utf8(s) => builder.append_value(s),
            AnyValue::Utf8Owned(s) => builder.append_value(s),
            AnyValue::Null => builder.append_null(),
            AnyValue::Binary(_) | AnyValue::BinaryOwned(_) => {
                if strict {
                    polars_bail!(ComputeError: "mixed dtypes found when building Utf8 Series")
                }
                builder.append_null()
            },
            av => {
                if strict {
                    polars_bail!(ComputeError: "mixed dtypes found when building Utf8 Series")
                }
                owned.clear();
                write!(owned, "{av}").unwrap();
                builder.append_value(&owned);
            },
        }
    }
    Ok(builder.finish())
}

#[cfg(feature = "dtype-decimal")]
fn any_values_to_decimal(
    avs: &[AnyValue],
    precision: Option<usize>,
    scale: Option<usize>, // if None, we're inferring the scale
) -> PolarsResult<DecimalChunked> {
    // two-pass approach, first we scan and record the scales, then convert (or not)
    let mut scale_range: Option<(usize, usize)> = None;
    for av in avs {
        let s_av = if av.is_signed_integer() || av.is_unsigned_integer() {
            0 // integers are treated as decimals with scale of zero
        } else if let AnyValue::Decimal(_, scale) = av {
            *scale
        } else if matches!(av, AnyValue::Null) {
            continue;
        } else {
            polars_bail!(
                ComputeError: "unable to convert any-value of dtype {} to decimal", av.dtype(),
            );
        };
        scale_range = match scale_range {
            None => Some((s_av, s_av)),
            Some((s_min, s_max)) => Some((s_min.min(s_av), s_max.max(s_av))),
        };
    }
    let Some((s_min, s_max)) = scale_range else {
        // empty array or all nulls, return a decimal array with given scale (or 0 if inferring)
        return Ok(Int128Chunked::full_null("", avs.len())
            .into_decimal_unchecked(precision, scale.unwrap_or(0)));
    };
    let scale = scale.unwrap_or(s_max);
    if s_max > scale {
        // scale is provided but is lower than actual
        // TODO: do we want lossy conversions here or not?
        polars_bail!(
            ComputeError:
            "unable to losslessly convert any-value of scale {s_max} to scale {}", scale,
        );
    }
    let mut builder = PrimitiveChunkedBuilder::<Int128Type>::new("", avs.len());
    let is_equally_scaled = s_min == s_max && s_max == scale;
    for av in avs {
        let (v, s_av) = if av.is_signed_integer() || av.is_unsigned_integer() {
            (
                av.try_extract::<i128>().unwrap_or_else(|_| unreachable!()),
                0,
            )
        } else if let AnyValue::Decimal(v, scale) = av {
            (*v, *scale)
        } else {
            // it has to be a null because we've already checked it
            builder.append_null();
            continue;
        };
        if is_equally_scaled {
            builder.append_value(v);
        } else {
            let factor = 10_i128.pow((scale - s_av) as _); // this cast is safe
            builder.append_value(v.checked_mul(factor).ok_or_else(|| {
                polars_err!(ComputeError: "overflow while converting to decimal scale {}", scale)
            })?);
        }
    }
    // build the array and do a precision check if needed
    builder.finish().into_decimal(precision, scale)
}

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

fn any_values_to_list(
    avs: &[AnyValue],
    inner_type: &DataType,
    strict: bool,
) -> PolarsResult<ListChunked> {
    // this is handled downstream. The builder will choose the first non null type
    let mut valid = true;
    #[allow(unused_mut)]
    let mut out: ListChunked = if inner_type == &DataType::Null {
        avs.iter()
            .map(|av| match av {
                AnyValue::List(b) => Some(b.clone()),
                AnyValue::Null => None,
                _ => {
                    valid = false;
                    None
                },
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
                },
                AnyValue::Null => None,
                _ => {
                    valid = false;
                    None
                },
            })
            .collect_trusted()
    };
    #[cfg(feature = "dtype-struct")]
    if !matches!(inner_type, DataType::Null)
        && matches!(out.inner_dtype(), DataType::Struct(_) | DataType::List(_))
    {
        // ensure the logical type is correct
        unsafe {
            out.set_dtype(DataType::List(Box::new(inner_type.clone())));
        };
    }
    if valid || !strict {
        Ok(out)
    } else {
        polars_bail!(ComputeError: "got mixed dtypes while constructing List Series")
    }
}

impl<'a, T: AsRef<[AnyValue<'a>]>> NamedFrom<T, [AnyValue<'a>]> for Series {
    fn new(name: &str, v: T) -> Self {
        let av = v.as_ref();
        Series::from_any_values(name, av, true).unwrap()
    }
}

impl Series {
    pub fn from_any_values_and_dtype(
        name: &str,
        av: &[AnyValue],
        dtype: &DataType,
        strict: bool,
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
            DataType::Utf8 => any_values_to_utf8(av, strict)?.into_series(),
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
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(precision, scale) => {
                any_values_to_decimal(av, *precision, *scale)?.into_series()
            },
            DataType::List(inner) => any_values_to_list(av, inner, strict)?.into_series(),
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
                            },
                            _ => field_avs.push(AnyValue::Null),
                        }
                    }
                    // if the inferred dtype is null, we let auto inference work
                    let s = if matches!(field.dtype, DataType::Null) {
                        Series::new(field.name(), &field_avs)
                    } else {
                        Series::from_any_values_and_dtype(
                            field.name(),
                            &field_avs,
                            &field.dtype,
                            strict,
                        )?
                    };
                    series_fields.push(s)
                }
                return StructChunked::new(name, &series_fields).map(|ca| ca.into_series());
            },
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
            },
            DataType::Null => Series::full_null(name, av.len(), &DataType::Null),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                let ca = if let Some(single_av) = av.first() {
                    match single_av {
                        AnyValue::Utf8(_) | AnyValue::Utf8Owned(_) | AnyValue::Null => {
                            any_values_to_utf8(av, strict)?
                        },
                        _ => polars_bail!(
                             ComputeError:
                             "categorical dtype with any-values of dtype {} not supported",
                             single_av.dtype()
                        ),
                    }
                } else {
                    Utf8Chunked::full("", "", 0)
                };

                ca.cast(&DataType::Categorical(None)).unwrap()
            },
            dt => panic!("{dt:?} not supported"),
        };
        s.rename(name);
        Ok(s)
    }

    pub fn from_any_values(name: &str, avs: &[AnyValue], strict: bool) -> PolarsResult<Series> {
        let mut all_flat_null = true;
        match avs.iter().find(|av| {
            if !matches!(av, AnyValue::Null) {
                all_flat_null = false;
            }
            !av.is_nested_null()
        }) {
            None => {
                if all_flat_null {
                    Ok(Series::full_null(name, avs.len(), &DataType::Null))
                } else {
                    // second pass and check for the nested null value that toggled `all_flat_null` to false
                    // e.g. a list<null>
                    if let Some(av) = avs.iter().find(|av| !matches!(av, AnyValue::Null)) {
                        let dtype: DataType = av.into();
                        Series::from_any_values_and_dtype(name, avs, &dtype, strict)
                    } else {
                        unreachable!()
                    }
                }
            },
            Some(av) => {
                #[cfg(feature = "dtype-decimal")]
                {
                    if let AnyValue::Decimal(_, _) = av {
                        let mut s = any_values_to_decimal(avs, None, None)?.into_series();
                        s.rename(name);
                        return Ok(s);
                    }
                }
                let dtype: DataType = av.into();
                Series::from_any_values_and_dtype(name, avs, &dtype, strict)
            },
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
            #[cfg(feature = "dtype-array")]
            Array(s, size) => DataType::Array(Box::new(s.dtype().clone()), *size),
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
                    let rev_map = RevMapping::build_local(array);
                    DataType::Categorical(Some(Arc::new(rev_map)))
                }
            },
            #[cfg(feature = "object")]
            Object(o) => DataType::Object(o.type_name()),
            #[cfg(feature = "object")]
            ObjectOwned(o) => DataType::Object(o.0.type_name()),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, scale) => DataType::Decimal(None, Some(*scale)),
        }
    }
}
