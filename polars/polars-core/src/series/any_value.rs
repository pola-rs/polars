use crate::prelude::*;

fn any_values_to_primitive<T: PolarsNumericType>(avs: &[AnyValue]) -> ChunkedArray<T> {
    avs.iter()
        .map(|av| av.extract::<T::Native>())
        .collect_trusted()
}

fn any_values_to_utf8(avs: &[AnyValue]) -> Utf8Chunked {
    avs.iter()
        .map(|av| match av {
            AnyValue::Utf8(s) => Some(*s),
            AnyValue::Utf8Owned(s) => Some(&**s),
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
                        Some(Series::full_null("", b.len(), inner_type))
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
    pub fn from_any_values_and_dtype<'a>(
        name: &str,
        av: &[AnyValue<'a>],
        dtype: &DataType,
    ) -> Result<Series> {
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
            DataType::Struct(fields) => {
                // the fields of the struct
                let mut series_fields = Vec::with_capacity(fields.len());
                for (i, field) in fields.iter().enumerate() {
                    let mut field_avs = Vec::with_capacity(av.len());

                    for av in av.iter() {
                        match av {
                            AnyValue::StructOwned(payload) => {
                                for (l, r) in fields.iter().zip(payload.1.iter()) {
                                    if l.name() != r.name() {
                                        return Err(PolarsError::ComputeError(
                                            "struct orders must remain the same".into(),
                                        ));
                                    }
                                }

                                let av_val = payload.0[i].clone();
                                field_avs.push(av_val)
                            }
                            _ => field_avs.push(AnyValue::Null),
                        }
                    }
                    series_fields.push(Series::new(field.name(), &field_avs))
                }
                return Ok(StructChunked::new(name, &series_fields)
                    .unwrap()
                    .into_series());
            }
            dtype => panic!("dtype {:?} not implemented", dtype),
        };
        s.rename(name);
        Ok(s)
    }

    pub fn from_any_values<'a>(name: &str, av: &[AnyValue<'a>]) -> Result<Series> {
        match av.iter().find(|av| !matches!(av, AnyValue::Null)) {
            None => Ok(Series::full_null(name, av.len(), &DataType::Int32)),
            Some(av_) => {
                let dtype: DataType = av_.into();
                Series::from_any_values_and_dtype(name, av, &dtype)
            }
        }
    }
}
