use super::*;

impl FunctionExpr {
    pub(crate) fn get_field(
        &self,
        _input_schema: &Schema,
        _cntxt: Context,
        fields: &[Field],
    ) -> PolarsResult<Field> {
        use FunctionExpr::*;

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

            // Other expressions
            Boolean(func) => func.get_field(mapper),
            #[cfg(feature = "abs")]
            Abs => mapper.with_same_dtype(),
            Negate => mapper.with_same_dtype(),
            NullCount => mapper.with_dtype(IDX_DTYPE),
            Pow(pow_function) => match pow_function {
                PowFunction::Generic => mapper.pow_dtype(),
                _ => mapper.map_to_float_dtype(),
            },
            Coalesce => mapper.map_to_supertype(),
            #[cfg(feature = "row_hash")]
            Hash(..) => mapper.with_dtype(DataType::UInt64),
            #[cfg(feature = "arg_where")]
            ArgWhere => mapper.with_dtype(IDX_DTYPE),
            #[cfg(feature = "search_sorted")]
            SearchSorted(_) => mapper.with_dtype(IDX_DTYPE),
            #[cfg(feature = "range")]
            Range(func) => func.get_field(mapper),
            #[cfg(feature = "date_offset")]
            DateOffset { .. } => mapper.with_same_dtype(),
            #[cfg(feature = "trigonometry")]
            Trigonometry(_) => mapper.map_to_float_dtype(),
            #[cfg(feature = "trigonometry")]
            Atan2 => mapper.map_to_float_dtype(),
            #[cfg(feature = "sign")]
            Sign => mapper.with_dtype(DataType::Int64),
            FillNull { super_type, .. } => mapper.with_dtype(super_type.clone()),
            #[cfg(feature = "rolling_window")]
            RollingExpr(rolling_func, ..) => {
                use RollingFunction::*;
                match rolling_func {
                    Min(_) | MinBy(_) | Max(_) | MaxBy(_) | Sum(_) | SumBy(_) => {
                        mapper.with_same_dtype()
                    },
                    Mean(_) | MeanBy(_) | Quantile(_) | QuantileBy(_) | Var(_) | VarBy(_)
                    | Std(_) | StdBy(_) => mapper.map_to_float_dtype(),
                    #[cfg(feature = "moment")]
                    Skew(..) => mapper.map_to_float_dtype(),
                }
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
            ArgUnique => mapper.with_dtype(IDX_DTYPE),
            #[cfg(feature = "rank")]
            Rank { options, .. } => mapper.with_dtype(match options.method {
                RankMethod::Average => DataType::Float64,
                _ => IDX_DTYPE,
            }),
            #[cfg(feature = "dtype-struct")]
            AsStruct => Ok(Field::new(
                fields[0].name(),
                DataType::Struct(fields.to_vec()),
            )),
            #[cfg(feature = "top_k")]
            TopK(_) => mapper.with_same_dtype(),
            #[cfg(feature = "dtype-struct")]
            ValueCounts { .. } => mapper.map_dtype(|dt| {
                DataType::Struct(vec![
                    Field::new(fields[0].name().as_str(), dt.clone()),
                    Field::new("count", IDX_DTYPE),
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
                        fields.push(Field::new("break_point", DataType::Float64));
                    }
                    if *include_category {
                        fields.push(Field::new(
                            "category",
                            DataType::Categorical(None, Default::default()),
                        ));
                    }
                    fields.push(Field::new("count", IDX_DTYPE));
                    mapper.with_dtype(DataType::Struct(fields))
                } else {
                    mapper.with_dtype(IDX_DTYPE)
                }
            },
            #[cfg(feature = "diff")]
            Diff(_, _) => mapper.map_dtype(|dt| match dt {
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
                InterpolationMethod::Linear => mapper.map_numeric_to_float_dtype(),
                InterpolationMethod::Nearest => mapper.with_same_dtype(),
            },
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
                    if dt.is_numeric() {
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
            PeakMin => mapper.with_same_dtype(),
            #[cfg(feature = "peaks")]
            PeakMax => mapper.with_same_dtype(),
            #[cfg(feature = "cutqcut")]
            Cut {
                include_breaks: false,
                ..
            } => mapper.with_dtype(DataType::Categorical(None, Default::default())),
            #[cfg(feature = "cutqcut")]
            Cut {
                include_breaks: true,
                ..
            } => {
                let name = fields[0].name();
                let name_bin = format!("{}_bin", name);
                let struct_dt = DataType::Struct(vec![
                    Field::new("brk", DataType::Float64),
                    Field::new(
                        name_bin.as_str(),
                        DataType::Categorical(None, Default::default()),
                    ),
                ]);
                mapper.with_dtype(struct_dt)
            },
            #[cfg(feature = "repeat_by")]
            RepeatBy => mapper.map_dtype(|dt| DataType::List(dt.clone().into())),
            Reshape(dims) => mapper.map_dtype(|dt| {
                let dtype = dt.inner_dtype().unwrap_or(dt).clone();
                if dims.len() == 1 {
                    dtype
                } else {
                    DataType::List(Box::new(dtype))
                }
            }),
            #[cfg(feature = "cutqcut")]
            QCut {
                include_breaks: false,
                ..
            } => mapper.with_dtype(DataType::Categorical(None, Default::default())),
            #[cfg(feature = "cutqcut")]
            QCut {
                include_breaks: true,
                ..
            } => {
                let name = fields[0].name();
                let name_bin = format!("{}_bin", name);
                let struct_dt = DataType::Struct(vec![
                    Field::new("brk", DataType::Float64),
                    Field::new(
                        name_bin.as_str(),
                        DataType::Categorical(None, Default::default()),
                    ),
                ]);
                mapper.with_dtype(struct_dt)
            },
            #[cfg(feature = "rle")]
            RLE => mapper.map_dtype(|dt| {
                DataType::Struct(vec![
                    Field::new("lengths", DataType::UInt64),
                    Field::new("values", dt.clone()),
                ])
            }),
            #[cfg(feature = "rle")]
            RLEID => mapper.with_dtype(DataType::UInt32),
            ToPhysical => mapper.to_physical_type(),
            #[cfg(feature = "random")]
            Random { .. } => mapper.with_same_dtype(),
            SetSortedFlag(_) => mapper.with_same_dtype(),
            #[cfg(feature = "ffi_plugin")]
            FfiPlugin {
                lib,
                symbol,
                kwargs,
            } => unsafe { plugin::plugin_field(fields, lib, symbol.as_ref(), kwargs) },
            BackwardFill { .. } => mapper.with_same_dtype(),
            ForwardFill { .. } => mapper.with_same_dtype(),
            SumHorizontal => mapper.map_to_supertype(),
            MaxHorizontal => mapper.map_to_supertype(),
            MinHorizontal => mapper.map_to_supertype(),
            #[cfg(feature = "ewma")]
            EwmMean { .. } => mapper.map_to_float_dtype(),
            #[cfg(feature = "ewma")]
            EwmStd { .. } => mapper.map_to_float_dtype(),
            #[cfg(feature = "ewma")]
            EwmVar { .. } => mapper.map_to_float_dtype(),
            #[cfg(feature = "replace")]
            Replace { return_dtype } => mapper.replace_dtype(return_dtype.clone()),
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

    /// Field with the same dtype.
    pub fn with_same_dtype(&self) -> PolarsResult<Field> {
        self.map_dtype(|dtype| dtype.clone())
    }

    /// Set a dtype.
    pub fn with_dtype(&self, dtype: DataType) -> PolarsResult<Field> {
        Ok(Field::new(self.fields[0].name(), dtype))
    }

    /// Map a single dtype.
    pub fn map_dtype(&self, func: impl FnOnce(&DataType) -> DataType) -> PolarsResult<Field> {
        let dtype = func(self.fields[0].data_type());
        Ok(Field::new(self.fields[0].name(), dtype))
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

    /// Map to a float supertype.
    pub fn map_to_float_dtype(&self) -> PolarsResult<Field> {
        self.map_dtype(|dtype| match dtype {
            DataType::Float32 => DataType::Float32,
            _ => DataType::Float64,
        })
    }

    /// Map to a float supertype if numeric, else preserve
    pub fn map_numeric_to_float_dtype(&self) -> PolarsResult<Field> {
        self.map_dtype(|dtype| {
            if dtype.is_numeric() {
                match dtype {
                    DataType::Float32 => DataType::Float32,
                    _ => DataType::Float64,
                }
            } else {
                dtype.clone()
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
        let dtype = func(self.fields[0].data_type())?;
        Ok(Field::new(self.fields[0].name(), dtype))
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
            .map(|fld| fld.data_type())
            .collect::<Vec<_>>();
        let new_type = func(&dtypes)?;
        fld.coerce(new_type);
        Ok(fld)
    }

    /// Map the dtype to the "supertype" of all fields.
    pub fn map_to_supertype(&self) -> PolarsResult<Field> {
        let mut first = self.fields[0].clone();
        let mut st = first.data_type().clone();
        for field in &self.fields[1..] {
            st = try_get_supertype(&st, field.data_type())?
        }
        first.coerce(st);
        Ok(first)
    }

    /// Map the dtype to the dtype of the list/array elements.
    pub fn map_to_list_and_array_inner_dtype(&self) -> PolarsResult<Field> {
        let mut first = self.fields[0].clone();
        let dt = first
            .data_type()
            .inner_dtype()
            .cloned()
            .unwrap_or(DataType::Unknown);
        first.coerce(dt);
        Ok(first)
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

    pub fn nested_sum_type(&self) -> PolarsResult<Field> {
        let mut first = self.fields[0].clone();
        use DataType::*;
        let dt = first.data_type().inner_dtype().cloned().unwrap_or(Unknown);

        if matches!(dt, UInt8 | Int8 | Int16 | UInt16) {
            first.coerce(Int64);
        } else {
            first.coerce(dt);
        }
        Ok(first)
    }

    pub(super) fn pow_dtype(&self) -> PolarsResult<Field> {
        // base, exponent
        match (self.fields[0].data_type(), self.fields[1].data_type()) {
            (
                base_dtype,
                DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64,
            ) => Ok(Field::new(self.fields[0].name(), base_dtype.clone())),
            (DataType::Float32, _) => Ok(Field::new(self.fields[0].name(), DataType::Float32)),
            (_, _) => Ok(Field::new(self.fields[0].name(), DataType::Float64)),
        }
    }

    #[cfg(feature = "extract_jsonpath")]
    pub fn with_opt_dtype(&self, dtype: Option<DataType>) -> PolarsResult<Field> {
        let dtype = dtype.unwrap_or(DataType::Unknown);
        self.with_dtype(dtype)
    }

    #[cfg(feature = "replace")]
    pub fn replace_dtype(&self, return_dtype: Option<DataType>) -> PolarsResult<Field> {
        let dtype = match return_dtype {
            Some(dtype) => dtype,
            // Supertype of `new` and `default`
            None => {
                let default = if let Some(default) = self.fields.get(3) {
                    default
                } else {
                    &self.fields[0]
                };
                try_get_supertype(self.fields[2].data_type(), default.data_type())?
            },
        };
        self.with_dtype(dtype)
    }
}
