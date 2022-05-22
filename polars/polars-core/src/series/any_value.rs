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

fn any_values_to_list(avs: &[AnyValue]) -> ListChunked {
    avs.iter()
        .map(|av| match av {
            AnyValue::List(b) => Some(b.clone()),
            _ => None,
        })
        .collect_trusted()
}

impl<'a, T: AsRef<[AnyValue<'a>]>> NamedFrom<T, [AnyValue<'a>]> for Series {
    fn new(name: &str, v: T) -> Self {
        let av = v.as_ref();
        Series::from_any_values(name, av).unwrap()
    }
}

impl Series {
    pub fn from_any_values<'a>(name: &str, av: &[AnyValue<'a>]) -> Result<Series> {
        match av.iter().find(|av| !matches!(av, AnyValue::Null)) {
            None => Ok(Series::full_null(name, av.len(), &DataType::Int32)),
            Some(av_) => {
                let mut s = match av_ {
                    AnyValue::Int32(_) => any_values_to_primitive::<Int32Type>(av).into_series(),
                    AnyValue::Int64(_) => any_values_to_primitive::<Int64Type>(av).into_series(),
                    AnyValue::UInt32(_) => any_values_to_primitive::<UInt32Type>(av).into_series(),
                    AnyValue::UInt64(_) => any_values_to_primitive::<UInt64Type>(av).into_series(),
                    AnyValue::Float32(_) => {
                        any_values_to_primitive::<Float32Type>(av).into_series()
                    }
                    AnyValue::Float64(_) => {
                        any_values_to_primitive::<Float64Type>(av).into_series()
                    }
                    AnyValue::Utf8(_) | AnyValue::Utf8Owned(_) => {
                        any_values_to_utf8(av).into_series()
                    }
                    AnyValue::Boolean(_) => any_values_to_bool(av).into_series(),
                    AnyValue::List(_) => any_values_to_list(av).into_series(),
                    #[cfg(feature = "dtype-date")]
                    AnyValue::Date(_) => any_values_to_primitive::<Int32Type>(av)
                        .into_date()
                        .into_series(),
                    #[cfg(feature = "dtype-datetime")]
                    AnyValue::Datetime(_, tu, tz) => any_values_to_primitive::<Int64Type>(av)
                        .into_datetime(*tu, (*tz).clone())
                        .into_series(),
                    #[cfg(feature = "dtype-time")]
                    AnyValue::Time(_) => any_values_to_primitive::<Int64Type>(av)
                        .into_time()
                        .into_series(),
                    #[cfg(feature = "dtype-duration")]
                    AnyValue::Duration(_, tu) => any_values_to_primitive::<Int64Type>(av)
                        .into_duration(*tu)
                        .into_series(),
                    #[cfg(feature = "dtype-struct")]
                    AnyValue::StructOwned(payload) => {
                        let vals = &payload.0;
                        let fields = &payload.1;

                        // the fields of the struct
                        let mut series_fields = Vec::with_capacity(vals.len());
                        for (i, field) in fields.iter().enumerate() {
                            let mut field_avs = Vec::with_capacity(av.len());

                            for av in av.iter() {
                                match av {
                                    AnyValue::StructOwned(pl) => {
                                        for (l, r) in fields.iter().zip(pl.1.iter()) {
                                            if l.name() != r.name() {
                                                return Err(PolarsError::ComputeError(
                                                    "struct orders must remain the same".into(),
                                                ));
                                            }
                                        }

                                        let av_val = pl.0[i].clone();
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
                    av => panic!("av {:?} not implemented", av),
                };
                s.rename(name);
                Ok(s)
            }
        }
    }
}
