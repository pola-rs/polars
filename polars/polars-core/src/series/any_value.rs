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
            AnyValue::List(b) => Some(b),
            _ => None,
        })
        .collect_trusted()
}

impl<'a, T: AsRef<[AnyValue<'a>]>> NamedFrom<T, [AnyValue<'a>]> for Series {
    fn new(name: &str, v: T) -> Self {
        let av = v.as_ref();
        Series::from_any_values(name, av)
    }
}

impl Series {
    fn from_any_values<'a>(name: &str, av: &[AnyValue<'a>]) -> Series {
        match av.iter().find(|av| !matches!(av, AnyValue::Null)) {
            None => Series::full_null(name, av.len(), &DataType::Int32),
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
                    _ => todo!(),
                };
                s.rename(name);
                s
            }
        }
    }
}
