use std::fmt::Write;

use arrow::bitmap::MutableBitmap;

#[cfg(feature = "dtype-categorical")]
use crate::chunked_array::cast::CastOptions;
#[cfg(feature = "object")]
use crate::chunked_array::object::registry::ObjectRegistry;
use crate::prelude::*;
use crate::utils::any_values_to_supertype;

impl<'a, T: AsRef<[AnyValue<'a>]>> NamedFrom<T, [AnyValue<'a>]> for Series {
    /// Construct a new [`Series`] from a collection of [`AnyValue`].
    ///
    /// # Panics
    ///
    /// Panics if the values do not all share the same data type (with the exception
    /// of [`DataType::Null`], which is always allowed).
    ///
    /// [`AnyValue`]: crate::datatypes::AnyValue
    fn new(name: &str, values: T) -> Self {
        let values = values.as_ref();
        Series::from_any_values(name, values, true).expect("data types of values should match")
    }
}

impl Series {
    /// Construct a new [`Series`] from a slice of AnyValues.
    ///
    /// The data type of the resulting Series is determined by the `values`
    /// and the `strict` parameter:
    /// - If `strict` is `true`, the data type is equal to the data type of the
    ///   first non-null value. If any other non-null values do not match this
    ///   data type, an error is raised.
    /// - If `strict` is `false`, the data type is the supertype of the `values`.
    ///   An error is returned if no supertype can be determined.
    ///   **WARNING**: A full pass over the values is required to determine the supertype.
    /// - If no values were passed, the resulting data type is `Null`.
    pub fn from_any_values(name: &str, values: &[AnyValue], strict: bool) -> PolarsResult<Self> {
        fn get_first_non_null_dtype(values: &[AnyValue]) -> DataType {
            let mut all_flat_null = true;
            let first_non_null = values.iter().find(|av| {
                if !av.is_null() {
                    all_flat_null = false
                };
                !av.is_nested_null()
            });
            match first_non_null {
                Some(av) => av.dtype(),
                None => {
                    if all_flat_null {
                        DataType::Null
                    } else {
                        // Second pass to check for the nested null value that
                        // toggled `all_flat_null` to false, e.g. a List(Null).
                        let first_nested_null = values.iter().find(|av| !av.is_null()).unwrap();
                        first_nested_null.dtype()
                    }
                },
            }
        }
        let dtype = if strict {
            get_first_non_null_dtype(values)
        } else {
            // Currently does not work correctly for Decimal because equality is not implemented.
            any_values_to_supertype(values)?
        };

        // TODO: Remove this when Decimal data type equality is implemented.
        #[cfg(feature = "dtype-decimal")]
        if dtype.is_decimal() {
            let dtype = DataType::Decimal(None, None);
            return Self::from_any_values_and_dtype(name, values, &dtype, strict);
        }

        Self::from_any_values_and_dtype(name, values, &dtype, strict)
    }

    /// Construct a new [`Series`] with the given `dtype` from a slice of AnyValues.
    ///
    /// If `strict` is `true`, an error is returned if the values do not match the given
    /// data type. If `strict` is `false`, values that do not match the given data type
    /// are cast. If casting is not possible, the values are set to null instead.
    pub fn from_any_values_and_dtype(
        name: &str,
        values: &[AnyValue],
        dtype: &DataType,
        strict: bool,
    ) -> PolarsResult<Self> {
        use crate::chunked_array::metadata::MetadataCollectable;

        if values.is_empty() {
            return Ok(Self::new_empty(name, dtype));
        }

        let mut s = match dtype {
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => any_values_to_integer::<Int8Type>(values, strict)?
                .with_cheap_metadata()
                .into_series(),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => any_values_to_integer::<Int16Type>(values, strict)?
                .with_cheap_metadata()
                .into_series(),
            DataType::Int32 => any_values_to_integer::<Int32Type>(values, strict)?
                .with_cheap_metadata()
                .into_series(),
            DataType::Int64 => any_values_to_integer::<Int64Type>(values, strict)?
                .with_cheap_metadata()
                .into_series(),
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => any_values_to_integer::<UInt8Type>(values, strict)?
                .with_cheap_metadata()
                .into_series(),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => any_values_to_integer::<UInt16Type>(values, strict)?
                .with_cheap_metadata()
                .into_series(),
            DataType::UInt32 => any_values_to_integer::<UInt32Type>(values, strict)?
                .with_cheap_metadata()
                .into_series(),
            DataType::UInt64 => any_values_to_integer::<UInt64Type>(values, strict)?
                .with_cheap_metadata()
                .into_series(),
            DataType::Float32 => any_values_to_f32(values, strict)?
                .with_cheap_metadata()
                .into_series(),
            DataType::Float64 => any_values_to_f64(values, strict)?
                .with_cheap_metadata()
                .into_series(),
            DataType::Boolean => any_values_to_bool(values, strict)?.into_series(),
            DataType::String => any_values_to_string(values, strict)?.into_series(),
            DataType::Binary => any_values_to_binary(values, strict)?.into_series(),
            #[cfg(feature = "dtype-date")]
            DataType::Date => any_values_to_date(values, strict)?.into_series(),
            #[cfg(feature = "dtype-time")]
            DataType::Time => any_values_to_time(values, strict)?.into_series(),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(tu, tz) => {
                any_values_to_datetime(values, *tu, (*tz).clone(), strict)?.into_series()
            },
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(tu) => any_values_to_duration(values, *tu, strict)?.into_series(),
            #[cfg(feature = "dtype-categorical")]
            dt @ (DataType::Categorical(_, _) | DataType::Enum(_, _)) => {
                any_values_to_categorical(values, dt, strict)?
            },
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(precision, scale) => {
                any_values_to_decimal(values, *precision, *scale, strict)?.into_series()
            },
            DataType::List(inner) => any_values_to_list(values, inner, strict)?.into_series(),
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner, size) => any_values_to_array(values, inner, strict, *size)?
                .into_series()
                .cast(&DataType::Array(inner.clone(), *size))?,
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => any_values_to_struct(values, fields, strict)?,
            #[cfg(feature = "object")]
            DataType::Object(_, registry) => any_values_to_object(values, registry)?,
            DataType::Null => Series::new_null(name, values.len()),
            dt => {
                polars_bail!(
                    InvalidOperation:
                    "constructing a Series with data type {dt:?} from AnyValues is not supported"
                )
            },
        };
        s.rename(name);
        Ok(s)
    }
}

fn any_values_to_primitive_nonstrict<T: PolarsNumericType>(values: &[AnyValue]) -> ChunkedArray<T> {
    values
        .iter()
        .map(|av| av.extract::<T::Native>())
        .collect_trusted()
}

fn any_values_to_integer<T: PolarsIntegerType>(
    values: &[AnyValue],
    strict: bool,
) -> PolarsResult<ChunkedArray<T>> {
    fn any_values_to_integer_strict<T: PolarsIntegerType>(
        values: &[AnyValue],
    ) -> PolarsResult<ChunkedArray<T>> {
        let mut builder = PrimitiveChunkedBuilder::<T>::new("", values.len());
        for av in values {
            match av {
                av if av.is_integer() => {
                    let opt_val = av.extract::<T::Native>();
                    let val = match opt_val {
                        Some(v) => v,
                        None => return Err(invalid_value_error(&T::get_dtype(), av)),
                    };
                    builder.append_value(val)
                },
                AnyValue::Null => builder.append_null(),
                av => return Err(invalid_value_error(&T::get_dtype(), av)),
            }
        }
        Ok(builder.finish())
    }

    if strict {
        any_values_to_integer_strict::<T>(values)
    } else {
        Ok(any_values_to_primitive_nonstrict::<T>(values))
    }
}

fn any_values_to_f32(values: &[AnyValue], strict: bool) -> PolarsResult<Float32Chunked> {
    fn any_values_to_f32_strict(values: &[AnyValue]) -> PolarsResult<Float32Chunked> {
        let mut builder = PrimitiveChunkedBuilder::<Float32Type>::new("", values.len());
        for av in values {
            match av {
                AnyValue::Float32(i) => builder.append_value(*i),
                AnyValue::Null => builder.append_null(),
                av => return Err(invalid_value_error(&DataType::Float32, av)),
            }
        }
        Ok(builder.finish())
    }
    if strict {
        any_values_to_f32_strict(values)
    } else {
        Ok(any_values_to_primitive_nonstrict::<Float32Type>(values))
    }
}
fn any_values_to_f64(values: &[AnyValue], strict: bool) -> PolarsResult<Float64Chunked> {
    fn any_values_to_f64_strict(values: &[AnyValue]) -> PolarsResult<Float64Chunked> {
        let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("", values.len());
        for av in values {
            match av {
                AnyValue::Float64(i) => builder.append_value(*i),
                AnyValue::Float32(i) => builder.append_value(*i as f64),
                AnyValue::Null => builder.append_null(),
                av => return Err(invalid_value_error(&DataType::Float64, av)),
            }
        }
        Ok(builder.finish())
    }
    if strict {
        any_values_to_f64_strict(values)
    } else {
        Ok(any_values_to_primitive_nonstrict::<Float64Type>(values))
    }
}

fn any_values_to_bool(values: &[AnyValue], strict: bool) -> PolarsResult<BooleanChunked> {
    let mut builder = BooleanChunkedBuilder::new("", values.len());
    for av in values {
        match av {
            AnyValue::Boolean(b) => builder.append_value(*b),
            AnyValue::Null => builder.append_null(),
            av => {
                if strict {
                    return Err(invalid_value_error(&DataType::Boolean, av));
                }
                match av.cast(&DataType::Boolean) {
                    AnyValue::Boolean(b) => builder.append_value(b),
                    _ => builder.append_null(),
                }
            },
        }
    }
    Ok(builder.finish())
}

fn any_values_to_string(values: &[AnyValue], strict: bool) -> PolarsResult<StringChunked> {
    fn any_values_to_string_strict(values: &[AnyValue]) -> PolarsResult<StringChunked> {
        let mut builder = StringChunkedBuilder::new("", values.len());
        for av in values {
            match av {
                AnyValue::String(s) => builder.append_value(s),
                AnyValue::StringOwned(s) => builder.append_value(s),
                AnyValue::Null => builder.append_null(),
                av => return Err(invalid_value_error(&DataType::String, av)),
            }
        }
        Ok(builder.finish())
    }
    fn any_values_to_string_nonstrict(values: &[AnyValue]) -> StringChunked {
        let mut builder = StringChunkedBuilder::new("", values.len());
        let mut owned = String::new(); // Amortize allocations.
        for av in values {
            match av {
                AnyValue::String(s) => builder.append_value(s),
                AnyValue::StringOwned(s) => builder.append_value(s),
                AnyValue::Null => builder.append_null(),
                AnyValue::Binary(_) | AnyValue::BinaryOwned(_) => builder.append_null(),
                av => {
                    owned.clear();
                    write!(owned, "{av}").unwrap();
                    builder.append_value(&owned);
                },
            }
        }
        builder.finish()
    }
    if strict {
        any_values_to_string_strict(values)
    } else {
        Ok(any_values_to_string_nonstrict(values))
    }
}

fn any_values_to_binary(values: &[AnyValue], strict: bool) -> PolarsResult<BinaryChunked> {
    fn any_values_to_binary_strict(values: &[AnyValue]) -> PolarsResult<BinaryChunked> {
        let mut builder = BinaryChunkedBuilder::new("", values.len());
        for av in values {
            match av {
                AnyValue::Binary(s) => builder.append_value(*s),
                AnyValue::BinaryOwned(s) => builder.append_value(&**s),
                AnyValue::Null => builder.append_null(),
                av => return Err(invalid_value_error(&DataType::Binary, av)),
            }
        }
        Ok(builder.finish())
    }
    fn any_values_to_binary_nonstrict(values: &[AnyValue]) -> BinaryChunked {
        values
            .iter()
            .map(|av| match av {
                AnyValue::Binary(b) => Some(*b),
                AnyValue::BinaryOwned(b) => Some(&**b),
                AnyValue::String(s) => Some(s.as_bytes()),
                AnyValue::StringOwned(s) => Some(s.as_bytes()),
                _ => None,
            })
            .collect_trusted()
    }
    if strict {
        any_values_to_binary_strict(values)
    } else {
        Ok(any_values_to_binary_nonstrict(values))
    }
}

#[cfg(feature = "dtype-date")]
fn any_values_to_date(values: &[AnyValue], strict: bool) -> PolarsResult<DateChunked> {
    let mut builder = PrimitiveChunkedBuilder::<Int32Type>::new("", values.len());
    for av in values {
        match av {
            AnyValue::Date(i) => builder.append_value(*i),
            AnyValue::Null => builder.append_null(),
            av => {
                if strict {
                    return Err(invalid_value_error(&DataType::Date, av));
                }
                match av.cast(&DataType::Date) {
                    AnyValue::Date(i) => builder.append_value(i),
                    _ => builder.append_null(),
                }
            },
        }
    }
    Ok(builder.finish().into())
}

#[cfg(feature = "dtype-time")]
fn any_values_to_time(values: &[AnyValue], strict: bool) -> PolarsResult<TimeChunked> {
    let mut builder = PrimitiveChunkedBuilder::<Int64Type>::new("", values.len());
    for av in values {
        match av {
            AnyValue::Time(i) => builder.append_value(*i),
            AnyValue::Null => builder.append_null(),
            av => {
                if strict {
                    return Err(invalid_value_error(&DataType::Time, av));
                }
                match av.cast(&DataType::Time) {
                    AnyValue::Time(i) => builder.append_value(i),
                    _ => builder.append_null(),
                }
            },
        }
    }
    Ok(builder.finish().into())
}

#[cfg(feature = "dtype-datetime")]
fn any_values_to_datetime(
    values: &[AnyValue],
    time_unit: TimeUnit,
    time_zone: Option<TimeZone>,
    strict: bool,
) -> PolarsResult<DatetimeChunked> {
    let mut builder = PrimitiveChunkedBuilder::<Int64Type>::new("", values.len());
    let target_dtype = DataType::Datetime(time_unit, time_zone.clone());
    for av in values {
        match av {
            AnyValue::Datetime(i, tu, _) if *tu == time_unit => builder.append_value(*i),
            AnyValue::Null => builder.append_null(),
            av => {
                if strict {
                    return Err(invalid_value_error(&target_dtype, av));
                }
                match av.cast(&target_dtype) {
                    AnyValue::Datetime(i, _, _) => builder.append_value(i),
                    _ => builder.append_null(),
                }
            },
        }
    }
    Ok(builder.finish().into_datetime(time_unit, time_zone))
}

#[cfg(feature = "dtype-duration")]
fn any_values_to_duration(
    values: &[AnyValue],
    time_unit: TimeUnit,
    strict: bool,
) -> PolarsResult<DurationChunked> {
    let mut builder = PrimitiveChunkedBuilder::<Int64Type>::new("", values.len());
    let target_dtype = DataType::Duration(time_unit);
    for av in values {
        match av {
            AnyValue::Duration(i, tu) if *tu == time_unit => builder.append_value(*i),
            AnyValue::Null => builder.append_null(),
            av => {
                if strict {
                    return Err(invalid_value_error(&target_dtype, av));
                }
                match av.cast(&target_dtype) {
                    AnyValue::Duration(i, _) => builder.append_value(i),
                    _ => builder.append_null(),
                }
            },
        }
    }
    Ok(builder.finish().into_duration(time_unit))
}

#[cfg(feature = "dtype-categorical")]
fn any_values_to_categorical(
    values: &[AnyValue],
    dtype: &DataType,
    strict: bool,
) -> PolarsResult<Series> {
    // TODO: Handle AnyValues of type Categorical/Enum.
    // TODO: Avoid materializing to String before casting to Categorical/Enum.
    let ca = any_values_to_string(values, strict)?;
    if strict {
        ca.into_series().strict_cast(dtype)
    } else {
        ca.cast_with_options(dtype, CastOptions::NonStrict)
    }
}

#[cfg(feature = "dtype-decimal")]
fn any_values_to_decimal(
    values: &[AnyValue],
    precision: Option<usize>,
    scale: Option<usize>, // If None, we're inferring the scale.
    strict: bool,
) -> PolarsResult<DecimalChunked> {
    /// Get the maximum scale among AnyValues
    fn infer_scale(
        values: &[AnyValue],
        precision: Option<usize>,
        strict: bool,
    ) -> PolarsResult<usize> {
        let mut max_scale = 0;
        for av in values {
            let av_scale = match av {
                AnyValue::Decimal(_, scale) => *scale,
                AnyValue::Null => continue,
                av => {
                    if strict {
                        let target_dtype = DataType::Decimal(precision, None);
                        return Err(invalid_value_error(&target_dtype, av));
                    }
                    continue;
                },
            };
            max_scale = max_scale.max(av_scale);
        }
        Ok(max_scale)
    }
    let scale = match scale {
        Some(s) => s,
        None => infer_scale(values, precision, strict)?,
    };
    let target_dtype = DataType::Decimal(precision, Some(scale));

    let mut builder = PrimitiveChunkedBuilder::<Int128Type>::new("", values.len());
    for av in values {
        match av {
            // Allow equal or less scale. We do want to support different scales even in 'strict' mode.
            AnyValue::Decimal(v, s) if *s <= scale => {
                if *s == scale {
                    builder.append_value(*v)
                } else {
                    match av.strict_cast(&target_dtype) {
                        Some(AnyValue::Decimal(i, _)) => builder.append_value(i),
                        _ => builder.append_null(),
                    }
                }
            },
            AnyValue::Null => builder.append_null(),
            av => {
                if strict {
                    return Err(invalid_value_error(&target_dtype, av));
                }
                // TODO: Precision check, else set to null
                match av.strict_cast(&target_dtype) {
                    Some(AnyValue::Decimal(i, _)) => builder.append_value(i),
                    _ => builder.append_null(),
                }
            },
        };
    }

    // Build the array and do a precision check if needed.
    builder.finish().into_decimal(precision, scale)
}

fn any_values_to_list(
    avs: &[AnyValue],
    inner_type: &DataType,
    strict: bool,
) -> PolarsResult<ListChunked> {
    let it = match inner_type {
        // Structs don't support empty fields yet.
        // We must ensure the data-types match what we do physical
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(fields) if fields.is_empty() => {
            DataType::Struct(vec![Field::new("", DataType::Null)])
        },
        _ => inner_type.clone(),
    };
    let target_dtype = DataType::List(Box::new(it));

    // This is handled downstream. The builder will choose the first non-null type.
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
    // Make sure that wrongly inferred AnyValues don't deviate from the datatype.
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

    if strict && !valid {
        polars_bail!(SchemaMismatch: "unexpected value while building Series of type {:?}", target_dtype);
    }

    // Ensure the logical type is correct for nested types.
    #[cfg(feature = "dtype-struct")]
    if !matches!(inner_type, DataType::Null) && out.inner_dtype().is_nested() {
        unsafe {
            out.set_dtype(target_dtype.clone());
        };
    }

    Ok(out)
}

#[cfg(feature = "dtype-array")]
fn any_values_to_array(
    avs: &[AnyValue],
    inner_type: &DataType,
    strict: bool,
    width: usize,
) -> PolarsResult<ArrayChunked> {
    fn to_arr(s: &Series) -> Option<ArrayRef> {
        if s.chunks().len() > 1 {
            let s = s.rechunk();
            Some(s.chunks()[0].clone())
        } else {
            Some(s.chunks()[0].clone())
        }
    }

    let target_dtype = DataType::Array(Box::new(inner_type.clone()), width);

    // This is handled downstream. The builder will choose the first non null type.
    let mut valid = true;
    #[allow(unused_mut)]
    let mut out: ArrayChunked = if inner_type == &DataType::Null {
        avs.iter()
            .map(|av| match av {
                AnyValue::List(b) | AnyValue::Array(b, _) => to_arr(b),
                AnyValue::Null => None,
                _ => {
                    valid = false;
                    None
                },
            })
            .collect_ca_with_dtype("", target_dtype.clone())
    }
    // Make sure that wrongly inferred AnyValues don't deviate from the datatype.
    else {
        avs.iter()
            .map(|av| match av {
                AnyValue::List(b) | AnyValue::Array(b, _) => {
                    if b.dtype() == inner_type {
                        to_arr(b)
                    } else {
                        let s = match b.cast(inner_type) {
                            Ok(out) => out,
                            Err(_) => Series::full_null(b.name(), b.len(), inner_type),
                        };
                        to_arr(&s)
                    }
                },
                AnyValue::Null => None,
                _ => {
                    valid = false;
                    None
                },
            })
            .collect_ca_with_dtype("", target_dtype.clone())
    };

    if strict && !valid {
        polars_bail!(SchemaMismatch: "unexpected value while building Series of type {:?}", target_dtype);
    }
    polars_ensure!(
        out.width() == width,
        SchemaMismatch: "got mixed size array widths where width {} was expected", width
    );

    // Ensure the logical type is correct for nested types.
    #[cfg(feature = "dtype-struct")]
    if !matches!(inner_type, DataType::Null) && out.inner_dtype().is_nested() {
        unsafe {
            out.set_dtype(target_dtype.clone());
        };
    }

    Ok(out)
}

#[cfg(feature = "dtype-struct")]
fn any_values_to_struct(
    values: &[AnyValue],
    fields: &[Field],
    strict: bool,
) -> PolarsResult<Series> {
    // Fast path for structs with no fields.
    if fields.is_empty() {
        return Ok(StructChunked::full_null("", values.len()).into_series());
    }

    // The physical series fields of the struct.
    let mut series_fields = Vec::with_capacity(fields.len());
    let mut has_outer_validity = false;
    let mut field_avs = Vec::with_capacity(values.len());
    for (i, field) in fields.iter().enumerate() {
        field_avs.clear();

        for av in values.iter() {
            match av {
                AnyValue::StructOwned(payload) => {
                    // TODO: Optimize.
                    let av_fields = &payload.1;
                    let av_values = &payload.0;

                    let mut append_by_search = || {
                        // Search for the name.
                        if let Some(i) = av_fields
                            .iter()
                            .position(|av_fld| av_fld.name == field.name)
                        {
                            field_avs.push(av_values[i].clone());
                            return;
                        }
                        field_avs.push(AnyValue::Null)
                    };

                    // All fields are available in this single value.
                    // We can use the index to get value.
                    if fields.len() == av_fields.len() {
                        if fields.iter().zip(av_fields.iter()).any(|(l, r)| l != r) {
                            append_by_search()
                        } else {
                            let av_val = av_values.get(i).cloned().unwrap_or(AnyValue::Null);
                            field_avs.push(av_val)
                        }
                    }
                    // Not all fields are available, we search the proper field.
                    else {
                        // Search for the name.
                        append_by_search()
                    }
                },
                _ => {
                    has_outer_validity = true;
                    field_avs.push(AnyValue::Null)
                },
            }
        }
        // If the inferred dtype is null, we let auto inference work.
        let s = if matches!(field.dtype, DataType::Null) {
            Series::from_any_values(field.name(), &field_avs, strict)?
        } else {
            Series::from_any_values_and_dtype(field.name(), &field_avs, &field.dtype, strict)?
        };
        series_fields.push(s)
    }

    let mut out = StructChunked::from_series("", &series_fields)?;
    if has_outer_validity {
        let mut validity = MutableBitmap::new();
        validity.extend_constant(values.len(), true);
        for (i, v) in values.iter().enumerate() {
            if matches!(v, AnyValue::Null) {
                unsafe { validity.set_unchecked(i, false) }
            }
        }
        out.set_outer_validity(Some(validity.freeze()))
    }
    Ok(out.into_series())
}

#[cfg(feature = "object")]
fn any_values_to_object(
    values: &[AnyValue],
    registry: &Option<Arc<ObjectRegistry>>,
) -> PolarsResult<Series> {
    let mut builder = match registry {
        None => {
            use crate::chunked_array::object::registry;
            let converter = registry::get_object_converter();
            let mut builder = registry::get_object_builder("", values.len());
            for av in values {
                match av {
                    AnyValue::Object(val) => builder.append_value(val.as_any()),
                    AnyValue::Null => builder.append_null(),
                    _ => {
                        // This is needed because in Python users can send mixed types.
                        // This only works if you set a global converter.
                        let any = converter(av.as_borrowed());
                        builder.append_value(&*any)
                    },
                }
            }
            builder
        },
        Some(registry) => {
            let mut builder = (*registry.builder_constructor)("", values.len());
            for av in values {
                match av {
                    AnyValue::Object(val) => builder.append_value(val.as_any()),
                    AnyValue::Null => builder.append_null(),
                    _ => {
                        polars_bail!(SchemaMismatch: "expected object");
                    },
                }
            }
            builder
        },
    };

    Ok(builder.to_series())
}

fn invalid_value_error(dtype: &DataType, value: &AnyValue) -> PolarsError {
    polars_err!(
        SchemaMismatch:
        "unexpected value while building Series of type {:?}; found value of type {:?}: {}",
        dtype,
        value.dtype(),
        value
    )
}
