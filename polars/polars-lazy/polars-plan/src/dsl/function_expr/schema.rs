use super::*;

/// set a dtype
pub(crate) fn with_dtype(fields: &[Field], dtype: DataType) -> PolarsResult<Field> {
    Ok(Field::new(fields[0].name(), dtype))
}
/// map a single dtype
pub(crate) fn map_dtype(
    fields: &[Field],
    func: &impl Fn(&DataType) -> DataType,
) -> PolarsResult<Field> {
    let dtype = func(fields[0].data_type());
    Ok(Field::new(fields[0].name(), dtype))
}
/// map a single dtype
#[cfg(feature = "timezones")]
pub(crate) fn try_map_dtype(
    fields: &[Field],
    func: &impl Fn(&DataType) -> PolarsResult<DataType>,
) -> PolarsResult<Field> {
    let dtype = func(fields[0].data_type())?;
    let out: PolarsResult<_> = Ok(Field::new(fields[0].name(), dtype));
    out
}
/// map all dtypes
pub(crate) fn try_map_dtypes(
    fields: &[Field],
    func: &impl Fn(&[&DataType]) -> PolarsResult<DataType>,
) -> PolarsResult<Field> {
    let mut fld = fields[0].clone();
    let dtypes = fields.iter().map(|fld| fld.data_type()).collect::<Vec<_>>();
    let new_type = func(&dtypes)?;
    fld.coerce(new_type);
    Ok(fld)
}

/// map to same type
pub(crate) fn same_type(fields: &[Field]) -> PolarsResult<Field> {
    map_dtype(fields, &|dtype| dtype.clone())
}

/// get supertype of all types
pub(crate) fn super_type(fields: &[Field]) -> PolarsResult<Field> {
    let mut first = fields[0].clone();
    let mut st = first.data_type().clone();
    for field in &fields[1..] {
        st = try_get_supertype(&st, field.data_type())?
    }
    first.coerce(st);
    Ok(first)
}

impl FunctionExpr {
    pub(crate) fn get_field(
        &self,
        input_schema: &Schema,
        cntxt: Context,
        fields: &[Field],
    ) -> PolarsResult<Field> {
        #[cfg(any(feature = "rolling_window", feature = "trigonometry"))]
        // set float supertype
        let float_dtype = || {
            map_dtype(fields, &|dtype| match dtype {
                DataType::Float32 => DataType::Float32,
                _ => DataType::Float64,
            })
        };

        let inner_type_list = || {
            let mut first = fields[0].clone();
            let dt = first
                .data_type()
                .inner_dtype()
                .cloned()
                .unwrap_or(DataType::Unknown);
            first.coerce(dt);
            Ok(first)
        };

        // inner super type of lists
        let inner_super_type_list = || {
            try_map_dtypes(fields, &|dts| {
                let mut super_type_inner = None;

                for dt in dts {
                    match dt {
                        DataType::List(inner) => match super_type_inner {
                            None => super_type_inner = Some(*inner.clone()),
                            Some(st_inner) => {
                                super_type_inner = Some(try_get_supertype(&st_inner, inner)?)
                            }
                        },
                        dt => match super_type_inner {
                            None => super_type_inner = Some((*dt).clone()),
                            Some(st_inner) => {
                                super_type_inner = Some(try_get_supertype(&st_inner, dt)?)
                            }
                        },
                    }
                }
                Ok(DataType::List(Box::new(super_type_inner.unwrap())))
            })
        };

        #[cfg(feature = "timezones")]
        let cast_tz = |tz: &TimeZone| {
            try_map_dtype(fields, &|dt| {
                if let DataType::Datetime(tu, _) = dt {
                    Ok(DataType::Datetime(*tu, Some(tz.clone())))
                } else {
                    Err(PolarsError::SchemaMisMatch(
                        format!("expected Datetime got {dt:?}").into(),
                    ))
                }
            })
        };

        use FunctionExpr::*;
        match self {
            NullCount => with_dtype(fields, IDX_DTYPE),
            Pow => super_type(fields),
            Coalesce => super_type(fields),
            #[cfg(feature = "row_hash")]
            Hash(..) => with_dtype(fields, DataType::UInt64),
            #[cfg(feature = "is_in")]
            IsIn => with_dtype(fields, DataType::Boolean),
            #[cfg(feature = "arg_where")]
            ArgWhere => with_dtype(fields, IDX_DTYPE),
            #[cfg(feature = "search_sorted")]
            SearchSorted(_) => with_dtype(fields, IDX_DTYPE),
            #[cfg(feature = "strings")]
            StringExpr(s) => {
                use StringFunction::*;
                match s {
                    Contains { .. } | EndsWith(_) | StartsWith(_) => {
                        with_dtype(fields, DataType::Boolean)
                    }
                    Extract { .. } => same_type(fields),
                    ExtractAll => with_dtype(fields, DataType::List(Box::new(DataType::Utf8))),
                    CountMatch(_) => with_dtype(fields, DataType::UInt32),
                    #[cfg(feature = "string_justify")]
                    Zfill { .. } | LJust { .. } | RJust { .. } => same_type(fields),
                    #[cfg(feature = "temporal")]
                    Strptime(options) => with_dtype(fields, options.date_dtype.clone()),
                    #[cfg(feature = "concat_str")]
                    ConcatVertical(_) | ConcatHorizontal(_) => with_dtype(fields, DataType::Utf8),
                    #[cfg(feature = "regex")]
                    Replace { .. } => with_dtype(fields, DataType::Utf8),
                    Uppercase | Lowercase | Strip(_) | LStrip(_) | RStrip(_) => {
                        with_dtype(fields, DataType::Utf8)
                    }
                    Split { .. } => with_dtype(fields, DataType::List(Box::new(DataType::Utf8))),
                    SplitExact { n, .. } => with_dtype(
                        fields,
                        DataType::Struct(
                            (0..n + 1)
                                .map(|i| Field::from_owned(format!("field_{i}"), DataType::Utf8))
                                .collect(),
                        ),
                    ),
                    #[cfg(feature = "dtype-struct")]
                    SplitN { n, .. } => with_dtype(
                        fields,
                        DataType::Struct(
                            (0..*n)
                                .map(|i| Field::from_owned(format!("field_{i}"), DataType::Utf8))
                                .collect(),
                        ),
                    ),
                }
            }
            #[cfg(feature = "dtype-binary")]
            BinaryExpr(s) => {
                use BinaryFunction::*;
                match s {
                    Contains { .. } | EndsWith(_) | StartsWith(_) => {
                        with_dtype(fields, DataType::Boolean)
                    }
                }
            }
            #[cfg(feature = "temporal")]
            TemporalExpr(fun) => {
                use TemporalFunction::*;
                let dtype = match fun {
                    Year | IsoYear => DataType::Int32,
                    Month | Quarter | Week | WeekDay | Day | OrdinalDay | Hour | Minute
                    | Millisecond | Microsecond | Nanosecond | Second => DataType::UInt32,
                    TimeStamp(_) => DataType::Int64,
                    Truncate(..) => same_type(fields).unwrap().dtype,
                    Round(..) => same_type(fields).unwrap().dtype,
                    #[cfg(feature = "timezones")]
                    CastTimezone(tz) | TzLocalize(tz) => return cast_tz(tz),
                    DateRange { .. } => return super_type(fields),
                    ToDatetime => DataType::Datetime(TimeUnit::Microseconds, None),
                    ToDuration => DataType::Duration(TimeUnit::Nanoseconds),
                };
                with_dtype(fields, dtype)
            }

            #[cfg(feature = "date_offset")]
            DateOffset(_) => same_type(fields),
            #[cfg(feature = "trigonometry")]
            Trigonometry(_) => float_dtype(),
            #[cfg(feature = "sign")]
            Sign => with_dtype(fields, DataType::Int64),
            FillNull { super_type, .. } => with_dtype(fields, super_type.clone()),
            #[cfg(all(feature = "rolling_window", feature = "moment"))]
            RollingSkew { .. } => float_dtype(),
            ShiftAndFill { .. } => same_type(fields),
            Nan(n) => n.get_field(fields),
            #[cfg(feature = "round_series")]
            Clip { .. } => same_type(fields),
            ListExpr(l) => {
                use ListFunction::*;
                match l {
                    Concat => inner_super_type_list(),
                    #[cfg(feature = "is_in")]
                    Contains => with_dtype(fields, DataType::Boolean),
                    Slice => same_type(fields),
                    Get => inner_type_list(),
                    #[cfg(feature = "list_take")]
                    Take(_) => same_type(fields),
                }
            }
            #[cfg(feature = "dtype-struct")]
            StructExpr(s) => {
                use polars_core::utils::slice_offsets;
                use StructFunction::*;
                match s {
                    FieldByIndex(index) => {
                        let (index, _) = slice_offsets(*index, 0, fields.len());
                        fields.get(index).cloned().ok_or_else(|| {
                            PolarsError::ComputeError(
                                "index out of bounds in 'struct.field'".into(),
                            )
                        })
                    }
                    FieldByName(name) => {
                        if let DataType::Struct(flds) = &fields[0].dtype {
                            let fld = flds
                                .iter()
                                .find(|fld| fld.name() == name.as_ref())
                                .ok_or_else(|| {
                                    PolarsError::NotFound(name.as_ref().to_string().into())
                                })?;
                            Ok(fld.clone())
                        } else {
                            Err(PolarsError::NotFound(name.as_ref().to_string().into()))
                        }
                    }
                    RenameFields(names) => {
                        map_dtype(fields, &|dt| match dt {
                            DataType::Struct(fields) => {
                                let fields = fields
                                    .iter()
                                    .zip(names.iter())
                                    .map(|(fld, name)| Field::new(name, fld.data_type().clone()))
                                    .collect();
                                DataType::Struct(fields)
                            }
                            // The types will be incorrect, but its better than nothing
                            // we can get an incorrect type with python lambdas, because we only know return type when running
                            // the query
                            dt => DataType::Struct(
                                names
                                    .iter()
                                    .map(|name| Field::new(name, dt.clone()))
                                    .collect(),
                            ),
                        })
                    }
                }
            }
            #[cfg(feature = "top_k")]
            TopK { .. } => same_type(fields),
            Shift(..) | Reverse => same_type(fields),
            IsNotNull | IsNull | Not | IsUnique | IsDuplicated => {
                with_dtype(fields, DataType::Boolean)
            }
            #[cfg(feature = "diff")]
            Diff(_, _) => map_dtype(fields, &|dt| match dt {
                #[cfg(feature = "dtype-datetime")]
                DataType::Datetime(tu, _) => DataType::Duration(*tu),
                #[cfg(feature = "dtype-date")]
                DataType::Date => DataType::Duration(TimeUnit::Milliseconds),
                #[cfg(feature = "dtype-time")]
                DataType::Time => DataType::Duration(TimeUnit::Nanoseconds),
                DataType::UInt64 | DataType::UInt32 => DataType::Int64,
                DataType::UInt16 => DataType::Int32,
                DataType::UInt8 => DataType::Int8,
                dt => dt.clone(),
            }),
            #[cfg(feature = "interpolate")]
            Interpolate(_) => same_type(fields),
            ShrinkType => {
                // we return the smallest type this can return
                // this might not be correct once the actual data
                // comes in, but if we set the smallest datatype
                // we have the least chance that the smaller dtypes
                // get cast to larger types in type-coercion
                // this will lead to an incorrect schema in polars
                // but we because only the numeric types deviate in
                // bit size this will likely not lead to issues
                map_dtype(fields, &|dt| {
                    if dt.is_numeric() {
                        if dt.is_float() {
                            DataType::Float32
                        } else if dt.is_unsigned() {
                            DataType::Int8
                        } else {
                            DataType::UInt8
                        }
                    } else {
                        dt.clone()
                    }
                })
            }
            NumFunction(fun) => fun.get_field(input_schema, cntxt, fields),
            ArgSortBy { .. } => with_dtype(fields, IDX_DTYPE),
            Unique | UniqueStable => same_type(fields),
        }
    }
}
