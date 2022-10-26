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

fn coerce_recursively(a: &Series, dtype: &DataType) -> Series {
    match (a.dtype(), dtype) {
        (lhs, rhs) if lhs == rhs => a.clone(),
        #[cfg(feature = "dtype-struct")]
        (DataType::Struct(_), DataType::Struct(dtype_fields)) => {
            let a = a.struct_().unwrap();
            let mut new_fields = Vec::with_capacity(a.fields().len());
            for (s_field, fld) in a.fields().iter().zip(dtype_fields) {
                let mut new_s = coerce_recursively(s_field, fld.data_type());
                if new_s.name() != fld.name {
                    new_s.rename(&fld.name);
                }
                new_fields.push(new_s);
            }
            StructChunked::new(a.name(), &new_fields)
                .unwrap()
                .into_series()
        }
        (DataType::List(_), DataType::List(inner_type)) => {
            let a = a.list().unwrap();
            let a = a.rechunk();
            let arr = a.downcast_iter().next().unwrap();
            let s = Series::try_from(("", arr.values().clone())).unwrap();
            let new_inner = coerce_recursively(&s, inner_type);
            let new_values = new_inner.array_ref(0).clone();

            let data_type = ListArray::<i64>::default_datatype(new_values.data_type().clone());
            let new_arr = ListArray::<i64>::new(
                data_type,
                arr.offsets().clone(),
                new_values,
                arr.validity().cloned(),
            );
            Series::try_from((s.name(), Box::new(new_arr) as ArrayRef)).unwrap()
        }
        _ => match a.cast(dtype) {
            Ok(s) => s,
            _ => Series::full_null("", a.len(), dtype),
        },
    }
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
                        Some(coerce_recursively(b, inner_type))
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
            dt => panic!("{:?} not supported", dt),
        };
        s.rename(name);
        Ok(s)
    }

    pub fn from_any_values<'a>(name: &str, av: &[AnyValue<'a>]) -> PolarsResult<Series> {
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
            Struct(_, fields) => DataType::Struct(fields.to_vec()),
            #[cfg(feature = "dtype-duration")]
            Duration(_, tu) => DataType::Duration(*tu),
            UInt8(_) => DataType::UInt8,
            UInt16(_) => DataType::UInt16,
            Int8(_) => DataType::Int8,
            Int16(_) => DataType::Int16,
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, rev_map) => DataType::Categorical(Some(Arc::new((*rev_map).clone()))),
            #[cfg(feature = "object")]
            Object(o) => DataType::Object(o.type_name()),
        }
    }
}
