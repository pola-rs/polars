use super::*;

impl FunctionExpr {
    pub(crate) fn get_field(
        &self,
        _input_schema: &Schema,
        _cntxt: Context,
        fields: &[Field],
    ) -> PolarsResult<Field> {
        // set a dtype
        let with_dtype = |dtype: DataType| Ok(Field::new(fields[0].name(), dtype));

        // map a single dtype
        let map_dtype = |func: &dyn Fn(&DataType) -> DataType| {
            let dtype = func(fields[0].data_type());
            Ok(Field::new(fields[0].name(), dtype))
        };

        // map a single dtype
        #[cfg(feature = "timezones")]
        let try_map_dtype = |func: &dyn Fn(&DataType) -> PolarsResult<DataType>| {
            let dtype = func(fields[0].data_type())?;
            let out: PolarsResult<_> = Ok(Field::new(fields[0].name(), dtype));
            out
        };

        // map all dtypes
        let map_dtypes = |func: &dyn Fn(&[&DataType]) -> DataType| {
            let mut fld = fields[0].clone();
            let dtypes = fields.iter().map(|fld| fld.data_type()).collect::<Vec<_>>();
            let new_type = func(&dtypes);
            fld.coerce(new_type);
            Ok(fld)
        };

        #[cfg(any(feature = "rolling_window", feature = "trigonometry"))]
        // set float supertype
        let float_dtype = || {
            map_dtype(&|dtype| match dtype {
                DataType::Float32 => DataType::Float32,
                _ => DataType::Float64,
            })
        };

        // map to same type
        let same_type = || map_dtype(&|dtype| dtype.clone());

        // get supertype of all types
        let super_type = || {
            let mut first = fields[0].clone();
            let mut st = first.data_type().clone();
            for field in &fields[1..] {
                st = try_get_supertype(&st, field.data_type())?
            }
            first.coerce(st);
            Ok(first)
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
            map_dtypes(&|dts| {
                let mut super_type_inner = None;

                for dt in dts {
                    match dt {
                        DataType::List(inner) => match super_type_inner {
                            None => super_type_inner = Some(*inner.clone()),
                            Some(st_inner) => {
                                super_type_inner = try_get_supertype(&st_inner, inner).ok()
                            }
                        },
                        dt => match super_type_inner {
                            None => super_type_inner = Some((*dt).clone()),
                            Some(st_inner) => {
                                super_type_inner = try_get_supertype(&st_inner, dt).ok()
                            }
                        },
                    }
                }
                DataType::List(Box::new(super_type_inner.unwrap()))
            })
        };

        #[cfg(feature = "timezones")]
        let cast_tz = |tz: &TimeZone| {
            try_map_dtype(&|dt| {
                if let DataType::Datetime(tu, _) = dt {
                    Ok(DataType::Datetime(*tu, Some(tz.clone())))
                } else {
                    Err(PolarsError::SchemaMisMatch(
                        format!("expected Datetime got {:?}", dt).into(),
                    ))
                }
            })
        };

        use FunctionExpr::*;
        match self {
            NullCount => with_dtype(IDX_DTYPE),
            Pow => super_type(),
            Coalesce => super_type(),
            #[cfg(feature = "row_hash")]
            Hash(..) => with_dtype(DataType::UInt64),
            #[cfg(feature = "is_in")]
            IsIn => with_dtype(DataType::Boolean),
            #[cfg(feature = "arg_where")]
            ArgWhere => with_dtype(IDX_DTYPE),
            #[cfg(feature = "search_sorted")]
            SearchSorted => with_dtype(IDX_DTYPE),
            #[cfg(feature = "strings")]
            StringExpr(s) => {
                use StringFunction::*;
                match s {
                    Contains { .. } | EndsWith(_) | StartsWith(_) => with_dtype(DataType::Boolean),
                    Extract { .. } => same_type(),
                    ExtractAll(_) => with_dtype(DataType::List(Box::new(DataType::Utf8))),
                    CountMatch(_) => with_dtype(DataType::UInt32),
                    #[cfg(feature = "string_justify")]
                    Zfill { .. } | LJust { .. } | RJust { .. } => same_type(),
                    #[cfg(feature = "temporal")]
                    Strptime(options) => with_dtype(options.date_dtype.clone()),
                    #[cfg(feature = "concat_str")]
                    ConcatVertical(_) | ConcatHorizontal(_) => with_dtype(DataType::Utf8),
                    #[cfg(feature = "regex")]
                    Replace { .. } => with_dtype(DataType::Utf8),
                    Uppercase | Lowercase | Strip(_) | LStrip(_) | RStrip(_) => {
                        with_dtype(DataType::Utf8)
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
                    Truncate(..) => same_type().unwrap().dtype,
                    Round(..) => same_type().unwrap().dtype,
                    #[cfg(feature = "timezones")]
                    CastTimezone(tz) | TzLocalize(tz) => return cast_tz(tz),
                    DateRange { .. } => return super_type(),
                };
                with_dtype(dtype)
            }

            #[cfg(feature = "date_offset")]
            DateOffset(_) => same_type(),
            #[cfg(feature = "trigonometry")]
            Trigonometry(_) => float_dtype(),
            #[cfg(feature = "sign")]
            Sign => with_dtype(DataType::Int64),
            FillNull { super_type, .. } => with_dtype(super_type.clone()),
            #[cfg(all(feature = "rolling_window", feature = "moment"))]
            RollingSkew { .. } => float_dtype(),
            ShiftAndFill { .. } => same_type(),
            Nan(n) => n.get_field(fields),
            #[cfg(feature = "round_series")]
            Clip { .. } => same_type(),
            ListExpr(l) => {
                use ListFunction::*;
                match l {
                    Concat => inner_super_type_list(),
                    #[cfg(feature = "is_in")]
                    Contains => with_dtype(DataType::Boolean),
                    Slice => same_type(),
                    Get => inner_type_list(),
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
                }
            }
            #[cfg(feature = "top_k")]
            TopK { .. } => same_type(),
            Shift(..) | Reverse => same_type(),
            IsNotNull | IsNull | Not | IsUnique | IsDuplicated => with_dtype(DataType::Boolean),
            #[cfg(feature = "diff")]
            Diff(_, _) => map_dtype(&|dt| match dt {
                #[cfg(feature = "dtype-datetime")]
                DataType::Datetime(tu, _) => DataType::Duration(*tu),
                #[cfg(feature = "dtype-date")]
                DataType::Date => DataType::Duration(TimeUnit::Milliseconds),
                #[cfg(feature = "dtype-time")]
                DataType::Time => DataType::Duration(TimeUnit::Nanoseconds),
                dt => dt.clone(),
            }),
            ShrinkType => {
                // we return the smallest type this can return
                // this might not be correct once the actual data
                // comes in, but if we set the smallest datatype
                // we have the least chance that the smaller dtypes
                // get cast to larger types in type-coercion
                // this will lead to an incorrect schema in polars
                // but we because only the numeric types deviate in
                // bit size this will likely not lead to issues
                map_dtype(&|dt| {
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
        }
    }
}
