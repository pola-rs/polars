use polars_core::utils::materialize_dyn_int;

use super::*;

impl IRFunctionExpr {
    pub(crate) fn get_field(
        &self,
        _input_schema: &Schema,
        _cntxt: Context,
        fields: &[Field],
    ) -> PolarsResult<Field> {
        use IRFunctionExpr::*;

        let mapper = FieldsMapper { fields };
        match self {
            // Namespaces
            #[cfg(feature = "dtype-array")]
            ArrayExpr(func) => func.get_field(mapper),
            BinaryExpr(s) => s.get_field(mapper),
            #[cfg(feature = "dtype-categorical")]
            Categorical(func) => func.get_field(mapper),
            ListExpr(func) => func.get_field(mapper),
            #[cfg(feature = "strings")]
            StringExpr(s) => s.get_field(mapper),
            #[cfg(feature = "dtype-struct")]
            StructExpr(s) => s.get_field(mapper),
            #[cfg(feature = "temporal")]
            TemporalExpr(fun) => fun.get_field(mapper),
            #[cfg(feature = "bitwise")]
            Bitwise(fun) => fun.get_field(mapper),

            // Other expressions
            Boolean(func) => func.get_field(mapper),
            #[cfg(feature = "business")]
            Business(func) => func.get_field(mapper),
            #[cfg(feature = "abs")]
            Abs => mapper.with_same_dtype(),
            Negate => mapper.with_same_dtype(),
            NullCount => mapper.with_dtype(IDX_DTYPE),
            Pow(pow_function) => match pow_function {
                IRPowFunction::Generic => mapper.pow_dtype(),
                _ => mapper.map_numeric_to_float_dtype(true),
            },
            Coalesce => mapper.map_to_supertype(),
            #[cfg(feature = "row_hash")]
            Hash(..) => mapper.with_dtype(DataType::UInt64),
            #[cfg(feature = "arg_where")]
            ArgWhere => mapper.with_dtype(IDX_DTYPE),
            #[cfg(feature = "index_of")]
            IndexOf => mapper.with_dtype(IDX_DTYPE),
            #[cfg(feature = "search_sorted")]
            SearchSorted { .. } => mapper.with_dtype(IDX_DTYPE),
            #[cfg(feature = "range")]
            Range(func) => func.get_field(mapper),
            #[cfg(feature = "trigonometry")]
            Trigonometry(_) => mapper.map_to_float_dtype(),
            #[cfg(feature = "trigonometry")]
            Atan2 => mapper.map_to_float_dtype(),
            #[cfg(feature = "sign")]
            Sign => mapper.ensure_satisfies(|_, dtype| dtype.is_primitive_numeric(), "sign")?.with_same_dtype(),
            FillNull  => mapper.map_to_supertype(),
            #[cfg(feature = "rolling_window")]
            RollingExpr { function, options } => {
                use IRRollingFunction::*;
                match function {
                    Min | Max => mapper.with_same_dtype(),
                    Mean | Quantile | Std => mapper.moment_dtype(),
                    Var => mapper.var_dtype(),
                    Sum => mapper.sum_dtype(),
                    #[cfg(feature = "cov")]
                    CorrCov {..} => mapper.map_to_float_dtype(),
                    #[cfg(feature = "moment")]
                    Skew | Kurtosis => mapper.map_to_float_dtype(),
                    Map(_) => mapper.try_map_field(|field| {
                        if options.weights.is_some() {
                            let dtype = match field.dtype() {
                                DataType::Float32 => DataType::Float32,
                                _ => DataType::Float64,
                            };
                            Ok(Field::new(field.name().clone(), dtype))
                        } else {
                            Ok(field.clone())
                        }
                    }),
                }
            },
            #[cfg(feature = "rolling_window_by")]
            RollingExprBy{function_by, ..} => {
                use IRRollingFunctionBy::*;
                match function_by {
                    MinBy | MaxBy => mapper.with_same_dtype(),
                    MeanBy | QuantileBy | StdBy=> mapper.moment_dtype(),
                    VarBy => mapper.var_dtype(),
                    SumBy => mapper.sum_dtype(),
                }
            },
            Append { upcast } => if *upcast {
                mapper.map_to_supertype()
            } else {
                mapper.with_same_dtype()
            },
            ShiftAndFill => mapper.with_same_dtype(),
            DropNans => mapper.with_same_dtype(),
            DropNulls => mapper.with_same_dtype(),
            #[cfg(feature = "round_series")]
            Clip { .. } => mapper.with_same_dtype(),
            #[cfg(feature = "mode")]
            Mode => mapper.with_same_dtype(),
            #[cfg(feature = "moment")]
            Skew(_) => mapper.with_dtype(DataType::Float64),
            #[cfg(feature = "moment")]
            Kurtosis(..) => mapper.with_dtype(DataType::Float64),
            ArgUnique | ArgMin | ArgMax | ArgSort { .. } => mapper.with_dtype(IDX_DTYPE),
            Product => mapper.map_dtype(|dtype| {
                use DataType as T;
                match dtype {
                    T::Float32 => T::Float32,
                    T::Float64 => T::Float64,
                    T::UInt64 => T::UInt64,
                    #[cfg(feature = "dtype-i128")]
                    T::Int128 => T::Int128,
                    _ => T::Int64,
                }
            }),
            Repeat => mapper.with_same_dtype(),
            #[cfg(feature = "rank")]
            Rank { options, .. } => mapper.with_dtype(match options.method {
                RankMethod::Average => DataType::Float64,
                _ => IDX_DTYPE,
            }),
            #[cfg(feature = "dtype-struct")]
            AsStruct => {
                let mut field_names = PlHashSet::with_capacity(fields.len() - 1);
                let struct_fields = fields.iter().map(|f| {
                    polars_ensure!(field_names.insert(f.name.as_str()), duplicate_field = f.name());
                    Ok(f.clone())
                }).collect::<PolarsResult<Vec<_>>>()?;
                Ok(Field::new(
                    fields[0].name().clone(),
                    DataType::Struct(struct_fields),
                ))
            },
            #[cfg(feature = "top_k")]
            TopK { .. } => mapper.with_same_dtype(),
            #[cfg(feature = "top_k")]
            TopKBy { .. } => mapper.with_same_dtype(),
            #[cfg(feature = "dtype-struct")]
            ValueCounts {
                sort: _,
                parallel: _,
                name,
                normalize,
            } => mapper.map_dtype(|dt| {
                let count_dt = if *normalize {
                    DataType::Float64
                } else {
                    IDX_DTYPE
                };
                DataType::Struct(vec![
                    Field::new(fields[0].name().clone(), dt.clone()),
                    Field::new(name.clone(), count_dt),
                ])
            }),
            #[cfg(feature = "unique_counts")]
            UniqueCounts => mapper.with_dtype(IDX_DTYPE),
            Shift | Reverse => mapper.with_same_dtype(),
            #[cfg(feature = "cum_agg")]
            CumCount { .. } => mapper.with_dtype(IDX_DTYPE),
            #[cfg(feature = "cum_agg")]
            CumSum { .. } => mapper.map_dtype(cum::dtypes::cum_sum),
            #[cfg(feature = "cum_agg")]
            CumProd { .. } => mapper.map_dtype(cum::dtypes::cum_prod),
            #[cfg(feature = "cum_agg")]
            CumMin { .. } => mapper.with_same_dtype(),
            #[cfg(feature = "cum_agg")]
            CumMax { .. } => mapper.with_same_dtype(),
            #[cfg(feature = "approx_unique")]
            ApproxNUnique => mapper.with_dtype(IDX_DTYPE),
            #[cfg(feature = "hist")]
            Hist {
                include_category,
                include_breakpoint,
                ..
            } => {
                if *include_breakpoint || *include_category {
                    let mut fields = Vec::with_capacity(3);
                    if *include_breakpoint {
                        fields.push(Field::new(
                            PlSmallStr::from_static("breakpoint"),
                            DataType::Float64,
                        ));
                    }
                    if *include_category {
                        fields.push(Field::new(
                            PlSmallStr::from_static("category"),
                            DataType::from_categories(Categories::global()),
                        ));
                    }
                    fields.push(Field::new(PlSmallStr::from_static("count"), IDX_DTYPE));
                    mapper.with_dtype(DataType::Struct(fields))
                } else {
                    mapper.with_dtype(IDX_DTYPE)
                }
            },
            #[cfg(feature = "diff")]
            Diff(_) => mapper.map_dtype(|dt| match dt {
                #[cfg(feature = "dtype-datetime")]
                DataType::Datetime(tu, _) => DataType::Duration(*tu),
                #[cfg(feature = "dtype-date")]
                DataType::Date => DataType::Duration(TimeUnit::Milliseconds),
                #[cfg(feature = "dtype-time")]
                DataType::Time => DataType::Duration(TimeUnit::Nanoseconds),
                DataType::UInt64 | DataType::UInt32 => DataType::Int64,
                DataType::UInt16 => DataType::Int32,
                DataType::UInt8 => DataType::Int16,
                dt => dt.clone(),
            }),
            #[cfg(feature = "pct_change")]
            PctChange => mapper.map_dtype(|dt| match dt {
                DataType::Float64 | DataType::Float32 => dt.clone(),
                _ => DataType::Float64,
            }),
            #[cfg(feature = "interpolate")]
            Interpolate(method) => match method {
                InterpolationMethod::Linear => mapper.map_numeric_to_float_dtype(false),
                InterpolationMethod::Nearest => mapper.with_same_dtype(),
            },
            #[cfg(feature = "interpolate_by")]
            InterpolateBy => mapper.map_numeric_to_float_dtype(true),
            ShrinkType => {
                // we return the smallest type this can return
                // this might not be correct once the actual data
                // comes in, but if we set the smallest datatype
                // we have the least chance that the smaller dtypes
                // get cast to larger types in type-coercion
                // this will lead to an incorrect schema in polars
                // but we because only the numeric types deviate in
                // bit size this will likely not lead to issues
                mapper.map_dtype(|dt| {
                    if dt.is_primitive_numeric() {
                        if dt.is_float() {
                            DataType::Float32
                        } else if dt.is_unsigned_integer() {
                            DataType::Int8
                        } else {
                            DataType::UInt8
                        }
                    } else {
                        dt.clone()
                    }
                })
            },
            #[cfg(feature = "log")]
            Entropy { .. } | Log { .. } | Log1p | Exp => mapper.map_to_float_dtype(),
            Unique(_) => mapper.with_same_dtype(),
            #[cfg(feature = "round_series")]
            Round { .. } | RoundSF { .. } | Floor | Ceil => mapper.with_same_dtype(),
            UpperBound | LowerBound => mapper.with_same_dtype(),
            #[cfg(feature = "fused")]
            Fused(_) => mapper.map_to_supertype(),
            ConcatExpr(_) => mapper.map_to_supertype(),
            #[cfg(feature = "cov")]
            Correlation { .. } => mapper.map_to_float_dtype(),
            #[cfg(feature = "peaks")]
            PeakMin | PeakMax => mapper.with_dtype(DataType::Boolean),
            #[cfg(feature = "cutqcut")]
            Cut {
                include_breaks: false,
                ..
            } => mapper.with_dtype(DataType::from_categories(Categories::global())),
            #[cfg(feature = "cutqcut")]
            Cut {
                include_breaks: true,
                ..
            } => {
                let struct_dt = DataType::Struct(vec![
                    Field::new(PlSmallStr::from_static("breakpoint"), DataType::Float64),
                    Field::new(
                        PlSmallStr::from_static("category"),
                        DataType::from_categories(Categories::global()),
                    ),
                ]);
                mapper.with_dtype(struct_dt)
            },
            #[cfg(feature = "repeat_by")]
            RepeatBy => mapper.map_dtype(|dt| DataType::List(dt.clone().into())),
            #[cfg(feature = "dtype-array")]
            Reshape(dims) => mapper.try_map_dtype(|dt: &DataType| {
                let dtype = dt.inner_dtype().unwrap_or(dt).clone();

                if dims.len() == 1 {
                    return Ok(dtype);
                }

                let num_infers = dims.iter().filter(|d| matches!(d, ReshapeDimension::Infer)).count();

                polars_ensure!(num_infers <= 1, InvalidOperation: "can only specify one inferred dimension");

                let mut inferred_size = 0;
                if num_infers == 1 {
                    let mut total_size = 1u64;
                    let mut current = dt;
                    while let DataType::Array(dt, width) = current {
                        if *width == 0 {
                            total_size = 0;
                            break;
                        }

                        current = dt.as_ref();
                        total_size *= *width as u64;
                    }

                    let current_size = dims.iter().map(|d| d.get_or_infer(1)).product::<u64>();
                    inferred_size = total_size / current_size;
                }

                let mut prev_dtype = dtype.leaf_dtype().clone();

                // We pop the outer dimension as that is the height of the series.
                for dim in &dims[1..] {
                    prev_dtype = DataType::Array(Box::new(prev_dtype), dim.get_or_infer(inferred_size) as usize);
                }
                Ok(prev_dtype)
            }),
            #[cfg(feature = "cutqcut")]
            QCut {
                include_breaks: false,
                ..
            } => mapper.with_dtype(DataType::from_categories(Categories::global())),
            #[cfg(feature = "cutqcut")]
            QCut {
                include_breaks: true,
                ..
            } => {
                let struct_dt = DataType::Struct(vec![
                    Field::new(PlSmallStr::from_static("breakpoint"), DataType::Float64),
                    Field::new(
                        PlSmallStr::from_static("category"),
                        DataType::from_categories(Categories::global()),
                    ),
                ]);
                mapper.with_dtype(struct_dt)
            },
            #[cfg(feature = "rle")]
            RLE => mapper.map_dtype(|dt| {
                DataType::Struct(vec![
                    Field::new(PlSmallStr::from_static("len"), IDX_DTYPE),
                    Field::new(PlSmallStr::from_static("value"), dt.clone()),
                ])
            }),
            #[cfg(feature = "rle")]
            RLEID => mapper.with_dtype(IDX_DTYPE),
            ToPhysical => mapper.to_physical_type(),
            #[cfg(feature = "random")]
            Random { .. } => mapper.with_same_dtype(),
            SetSortedFlag(_) => mapper.with_same_dtype(),
            #[cfg(feature = "ffi_plugin")]
            FfiPlugin {
                flags: _,
                lib,
                symbol,
                kwargs,
            } => unsafe { plugin::plugin_field(fields, lib, symbol.as_ref(), kwargs) },

            FoldHorizontal { return_dtype, .. } => match return_dtype {
                None => mapper.with_same_dtype(),
                Some(dtype) => mapper.with_dtype(dtype.clone()),
            },
            ReduceHorizontal { return_dtype, .. } => match return_dtype {
                // @2.0: This should probably map to `Unknown`.
                None => mapper.map_to_supertype(),
                Some(dtype) => mapper.with_dtype(dtype.clone()),
            },
            #[cfg(feature = "dtype-struct")]
            CumReduceHorizontal {
                return_dtype, ..
            }=> match return_dtype {
                // @2.0: This should probably map to `Unknown`.
                None => mapper.with_dtype(DataType::Struct(fields.to_vec())),
                Some(dtype) => mapper.with_dtype(DataType::Struct(fields.iter().map(|f| Field::new(f.name().clone(), dtype.clone())).collect())),
            },
            #[cfg(feature = "dtype-struct")]
            CumFoldHorizontal { return_dtype, include_init, .. } => match return_dtype {
                // @2.0: This should probably map to `Unknown`.
                None => mapper.with_dtype(DataType::Struct(fields.iter().skip(usize::from(!include_init)).map(|f| Field::new(f.name().clone(), fields[0].dtype().clone())).collect())),
                Some(dtype) => mapper.with_dtype(DataType::Struct(fields.iter().skip(usize::from(!include_init)).map(|f| Field::new(f.name().clone(), dtype.clone())).collect())),
            },

            MaxHorizontal => mapper.map_to_supertype(),
            MinHorizontal => mapper.map_to_supertype(),
            SumHorizontal { .. } => {
                mapper.map_to_supertype().map(|mut f| {
                    if f.dtype == DataType::Boolean {
                        f.dtype = IDX_DTYPE;
                    }
                    f
                })
            },
            MeanHorizontal { .. } => {
                mapper.map_to_supertype().map(|mut f| {
                    match f.dtype {
                        dt @ DataType::Float32 => { f.dtype = dt; },
                        _ => { f.dtype = DataType::Float64; },
                    };
                    f
                })
            }
            #[cfg(feature = "ewma")]
            EwmMean { .. } => mapper.map_numeric_to_float_dtype(true),
            #[cfg(feature = "ewma_by")]
            EwmMeanBy { .. } => mapper.map_numeric_to_float_dtype(true),
            #[cfg(feature = "ewma")]
            EwmStd { .. } => mapper.map_numeric_to_float_dtype(true),
            #[cfg(feature = "ewma")]
            EwmVar { .. } => mapper.var_dtype(),
            #[cfg(feature = "replace")]
            Replace => mapper.with_same_dtype(),
            #[cfg(feature = "replace")]
            ReplaceStrict { return_dtype } => mapper.replace_dtype(return_dtype.clone()),
            FillNullWithStrategy(_) => mapper.with_same_dtype(),
            GatherEvery { .. } => mapper.with_same_dtype(),
            #[cfg(feature = "reinterpret")]
            Reinterpret(signed) => {
                let dt = if *signed {
                    DataType::Int64
                } else {
                    DataType::UInt64
                };
                mapper.with_dtype(dt)
            },
            ExtendConstant => mapper.with_same_dtype(),
        }
    }

    pub(crate) fn output_name(&self) -> Option<OutputName> {
        match self {
            #[cfg(feature = "dtype-struct")]
            IRFunctionExpr::StructExpr(IRStructFunction::FieldByName(name)) => {
                Some(OutputName::Field(name.clone()))
            },
            _ => None,
        }
    }
}

pub struct FieldsMapper<'a> {
    fields: &'a [Field],
}

impl<'a> FieldsMapper<'a> {
    pub fn new(fields: &'a [Field]) -> Self {
        Self { fields }
    }

    pub fn args(&self) -> &[Field] {
        self.fields
    }

    /// Field with the same dtype.
    pub fn with_same_dtype(&self) -> PolarsResult<Field> {
        self.map_dtype(|dtype| dtype.clone())
    }

    /// Set a dtype.
    pub fn with_dtype(&self, dtype: DataType) -> PolarsResult<Field> {
        Ok(Field::new(self.fields[0].name().clone(), dtype))
    }

    /// Map a single dtype.
    pub fn map_dtype(&self, func: impl FnOnce(&DataType) -> DataType) -> PolarsResult<Field> {
        let dtype = func(self.fields[0].dtype());
        Ok(Field::new(self.fields[0].name().clone(), dtype))
    }

    pub fn get_fields_lens(&self) -> usize {
        self.fields.len()
    }

    /// Map a single field with a potentially failing mapper function.
    pub fn try_map_field(
        &self,
        func: impl FnOnce(&Field) -> PolarsResult<Field>,
    ) -> PolarsResult<Field> {
        func(&self.fields[0])
    }

    pub fn var_dtype(&self) -> PolarsResult<Field> {
        if self.fields[0].dtype().leaf_dtype().is_duration() {
            let map_inner = |dt: &DataType| match dt {
                dt if dt.is_temporal() => {
                    polars_bail!(InvalidOperation: "variance of type {dt} is not supported")
                },
                dt => Ok(dt.clone()),
            };

            self.try_map_dtype(|dt| match dt {
                #[cfg(feature = "dtype-array")]
                DataType::Array(inner, _) => map_inner(inner),
                DataType::List(inner) => map_inner(inner),
                _ => map_inner(dt),
            })
        } else {
            self.moment_dtype()
        }
    }

    pub fn moment_dtype(&self) -> PolarsResult<Field> {
        let map_inner = |dt: &DataType| match dt {
            #[cfg(feature = "dtype-datetime")]
            dt @ DataType::Datetime(_, _) => dt.clone(),
            #[cfg(feature = "dtype-duration")]
            dt @ DataType::Duration(_) => dt.clone(),
            #[cfg(feature = "dtype-time")]
            dt @ DataType::Time => dt.clone(),
            DataType::Float32 => DataType::Float32,
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(..) => DataType::Float64,
            DataType::Boolean => DataType::Float64,
            _ => DataType::Float64,
        };

        self.map_dtype(|dt| match dt {
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner, _) => map_inner(inner),
            DataType::List(inner) => map_inner(inner),
            _ => map_inner(dt),
        })
    }

    /// Map to a float supertype.
    pub fn map_to_float_dtype(&self) -> PolarsResult<Field> {
        self.map_dtype(|dtype| match dtype {
            DataType::Float32 => DataType::Float32,
            _ => DataType::Float64,
        })
    }

    /// Map to a float supertype if numeric, else preserve
    pub fn map_numeric_to_float_dtype(&self, coerce_decimal: bool) -> PolarsResult<Field> {
        self.map_dtype(|dt| {
            let should_coerce = match dt {
                DataType::Float32 => false,
                #[cfg(feature = "dtype-decimal")]
                DataType::Decimal(..) => coerce_decimal,
                DataType::Boolean => true,
                dt => dt.is_primitive_numeric(),
            };

            if should_coerce {
                DataType::Float64
            } else {
                dt.clone()
            }
        })
    }

    /// Map to a physical type.
    pub fn to_physical_type(&self) -> PolarsResult<Field> {
        self.map_dtype(|dtype| dtype.to_physical())
    }

    /// Map a single dtype with a potentially failing mapper function.
    pub fn try_map_dtype(
        &self,
        func: impl FnOnce(&DataType) -> PolarsResult<DataType>,
    ) -> PolarsResult<Field> {
        let dtype = func(self.fields[0].dtype())?;
        Ok(Field::new(self.fields[0].name().clone(), dtype))
    }

    /// Map all dtypes with a potentially failing mapper function.
    pub fn try_map_dtypes(
        &self,
        func: impl FnOnce(&[&DataType]) -> PolarsResult<DataType>,
    ) -> PolarsResult<Field> {
        let mut fld = self.fields[0].clone();
        let dtypes = self
            .fields
            .iter()
            .map(|fld| fld.dtype())
            .collect::<Vec<_>>();
        let new_type = func(&dtypes)?;
        fld.coerce(new_type);
        Ok(fld)
    }

    /// Map the dtype to the "supertype" of all fields.
    pub fn map_to_supertype(&self) -> PolarsResult<Field> {
        let st = args_to_supertype(self.fields)?;
        let mut first = self.fields[0].clone();
        first.coerce(st);
        Ok(first)
    }

    /// Map the dtype to the dtype of the list/array elements.
    pub fn map_to_list_and_array_inner_dtype(&self) -> PolarsResult<Field> {
        let mut first = self.fields[0].clone();
        let dt = first
            .dtype()
            .inner_dtype()
            .cloned()
            .unwrap_or_else(|| DataType::Unknown(Default::default()));
        first.coerce(dt);
        Ok(first)
    }

    #[cfg(feature = "dtype-array")]
    /// Map the dtype to the dtype of the array elements, with typo validation.
    pub fn try_map_to_array_inner_dtype(&self) -> PolarsResult<Field> {
        let dt = self.fields[0].dtype();
        match dt {
            DataType::Array(_, _) => self.map_to_list_and_array_inner_dtype(),
            _ => polars_bail!(InvalidOperation: "expected Array type, got: {}", dt),
        }
    }

    /// Map the dtypes to the "supertype" of a list of lists.
    pub fn map_to_list_supertype(&self) -> PolarsResult<Field> {
        self.try_map_dtypes(|dts| {
            let mut super_type_inner = None;

            for dt in dts {
                match dt {
                    DataType::List(inner) => match super_type_inner {
                        None => super_type_inner = Some(*inner.clone()),
                        Some(st_inner) => {
                            super_type_inner = Some(try_get_supertype(&st_inner, inner)?)
                        },
                    },
                    dt => match super_type_inner {
                        None => super_type_inner = Some((*dt).clone()),
                        Some(st_inner) => {
                            super_type_inner = Some(try_get_supertype(&st_inner, dt)?)
                        },
                    },
                }
            }
            Ok(DataType::List(Box::new(super_type_inner.unwrap())))
        })
    }

    /// Set the timezone of a datetime dtype.
    #[cfg(feature = "timezones")]
    pub fn map_datetime_dtype_timezone(&self, tz: Option<&TimeZone>) -> PolarsResult<Field> {
        self.try_map_dtype(|dt| {
            if let DataType::Datetime(tu, _) = dt {
                Ok(DataType::Datetime(*tu, tz.cloned()))
            } else {
                polars_bail!(op = "replace-time-zone", got = dt, expected = "Datetime");
            }
        })
    }

    pub fn sum_dtype(&self) -> PolarsResult<Field> {
        use DataType::*;
        self.map_dtype(|dtype| match dtype {
            Int8 | UInt8 | Int16 | UInt16 => Int64,
            dt => dt.clone(),
        })
    }

    pub fn nested_sum_type(&self) -> PolarsResult<Field> {
        let mut first = self.fields[0].clone();
        use DataType::*;
        let dt = first
            .dtype()
            .inner_dtype()
            .cloned()
            .unwrap_or_else(|| Unknown(Default::default()));

        match dt {
            Boolean => first.coerce(IDX_DTYPE),
            UInt8 | Int8 | Int16 | UInt16 => first.coerce(Int64),
            _ => first.coerce(dt),
        }
        Ok(first)
    }

    pub fn nested_mean_median_type(&self) -> PolarsResult<Field> {
        let mut first = self.fields[0].clone();
        use DataType::*;
        let dt = first
            .dtype()
            .inner_dtype()
            .cloned()
            .unwrap_or_else(|| Unknown(Default::default()));

        let new_dt = match dt {
            #[cfg(feature = "dtype-datetime")]
            Date => Datetime(TimeUnit::Milliseconds, None),
            dt if dt.is_temporal() => dt,
            Float32 => Float32,
            _ => Float64,
        };
        first.coerce(new_dt);
        Ok(first)
    }

    pub(super) fn pow_dtype(&self) -> PolarsResult<Field> {
        let base_dtype = self.fields[0].dtype();
        let exponent_dtype = self.fields[1].dtype();
        if base_dtype.is_integer() {
            if exponent_dtype.is_float() {
                Ok(Field::new(
                    self.fields[0].name().clone(),
                    exponent_dtype.clone(),
                ))
            } else {
                Ok(Field::new(
                    self.fields[0].name().clone(),
                    base_dtype.clone(),
                ))
            }
        } else {
            Ok(Field::new(
                self.fields[0].name().clone(),
                base_dtype.clone(),
            ))
        }
    }

    #[cfg(feature = "extract_jsonpath")]
    pub fn with_opt_dtype(&self, dtype: Option<DataType>) -> PolarsResult<Field> {
        let dtype = dtype.unwrap_or_else(|| DataType::Unknown(Default::default()));
        self.with_dtype(dtype)
    }

    #[cfg(feature = "replace")]
    pub fn replace_dtype(&self, return_dtype: Option<DataType>) -> PolarsResult<Field> {
        let dtype = match return_dtype {
            Some(dtype) => dtype,
            None => {
                let new = &self.fields[2];
                let default = self.fields.get(3);

                // @HACK: Related to implicit implode see #22149.
                let inner_dtype = new.dtype().inner_dtype().unwrap_or(new.dtype());

                match default {
                    Some(default) => try_get_supertype(default.dtype(), inner_dtype)?,
                    None => inner_dtype.clone(),
                }
            },
        };
        self.with_dtype(dtype)
    }

    fn ensure_satisfies(
        self,
        mut f: impl FnMut(usize, &DataType) -> bool,
        op: &'static str,
    ) -> PolarsResult<Self> {
        for (i, field) in self.fields.iter().enumerate() {
            polars_ensure!(
                f(i, field.dtype()),
                opidx = op,
                idx = i,
                self.fields[i].dtype()
            );
        }

        Ok(self)
    }
}

pub(crate) fn args_to_supertype<D: AsRef<DataType>>(dtypes: &[D]) -> PolarsResult<DataType> {
    let mut st = dtypes[0].as_ref().clone();
    for dt in &dtypes[1..] {
        st = try_get_supertype(&st, dt.as_ref())?
    }

    match (dtypes[0].as_ref(), &st) {
        #[cfg(feature = "dtype-categorical")]
        (cat @ DataType::Categorical(_, _), DataType::String) => st = cat.clone(),
        _ => {
            if let DataType::Unknown(kind) = st {
                match kind {
                    UnknownKind::Float => st = DataType::Float64,
                    UnknownKind::Int(v) => {
                        st = materialize_dyn_int(v).dtype();
                    },
                    UnknownKind::Str => st = DataType::String,
                    _ => {},
                }
            }
        },
    }

    Ok(st)
}
