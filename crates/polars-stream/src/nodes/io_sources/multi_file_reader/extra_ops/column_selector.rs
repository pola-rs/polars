use arrow::array::{Array, LIST_VALUES_NAME};
use arrow::datatypes::{ArrowDataType, Field as ArrowField};
use polars_core::chunked_array::cast::CastOptions;
use polars_core::chunked_array::flags::StatisticsFlags;
use polars_core::prelude::{Column, DataType, InitHashMaps, IntoColumn, PlHashMap};
use polars_core::scalar::Scalar;
use polars_core::schema::Schema;
use polars_core::schema::iceberg::{IcebergColumn, IcebergColumnType};
use polars_core::series::{IntoSeries, Series};
use polars_core::utils::get_numeric_upcast_supertype_lossless;
use polars_error::{PolarsResult, feature_gated, polars_bail};
use polars_plan::dsl::{CastColumnsPolicy, ExtraColumnsPolicy, MissingColumnsPolicy};
use polars_utils::pl_str::PlSmallStr;
use recursive::recursive;

use crate::nodes::io_sources::multi_file_reader::extra_ops::missing_column_err;

/// This is a physical expression that is specialized for performing positional column selections,
/// column renaming, as well as type-casting.
///
/// Handles:
/// * Type-casting
///   * Only allowing a restricted set of non-nested casts for now
/// * Column re-ordering
/// * Struct field re-ordering
/// * Inserting missing struct fields
/// * Dropping extra struct fields (we just don't select them)
#[derive(Debug, Clone)]
pub enum ColumnSelector {
    // Note that we Box enum variants to keep `ColumnSelector` small (16 bytes).
    // This is an optimization that benefits cases where there are many `Position` selectors.

    // Leaf selectors
    /// Take the column at this position.
    Position(usize),
    /// Materialize a constant column.
    /// `(column_name, value)`
    Constant(Box<(PlSmallStr, Scalar)>),

    /// `(input_selector, _)`
    Transformed(Box<(ColumnSelector, ColumnTransform)>),
}

impl ColumnSelector {
    #[recursive]
    pub fn select_from_columns(
        &self,
        columns: &[Column],
        output_height: usize,
    ) -> PolarsResult<Column> {
        use ColumnSelector as S;

        Ok(match self {
            S::Position(i) => columns[*i].clone(),

            S::Constant(parts) => {
                let (name, scalar) = parts.as_ref();
                Column::new_scalar(name.clone(), scalar.clone(), output_height)
            },

            S::Transformed(transform) => {
                let input: Column = transform.0.select_from_columns(columns, output_height)?;
                transform.1.apply_transform(input)?
            },
        })
    }

    /// Replaces the leaf selector with the given `input` selector.
    pub fn replace_input(&mut self, input: ColumnSelector) {
        let mut current = self;

        while let Self::Transformed(v) = current {
            current = &mut v.as_mut().0;
        }

        *current = input;
    }
}

#[derive(Debug, Clone)]
pub enum ColumnTransform {
    /// Cast the column to a dtype.
    Cast {
        dtype: DataType,
        options: CastOptions,
    },
    /// Set the name of the column.
    Rename { name: PlSmallStr },
    /// Construct a struct column by applying column selectors onto the field arrays.
    StructFieldsMapping {
        field_selectors: Box<[ColumnSelector]>,
    },
    /// Construct a list column by applying column selectors onto the values array.
    ListValuesMapping { values_selector: ColumnSelector },
    #[cfg(feature = "dtype-array")]
    FixedSizeListValuesMapping { values_selector: ColumnSelector },
}

impl ColumnTransform {
    pub fn into_selector(self, input_selector: ColumnSelector) -> ColumnSelector {
        ColumnSelector::Transformed(Box::new((input_selector, self)))
    }

    pub fn apply_transform(&self, input: Column) -> PolarsResult<Column> {
        use ColumnTransform as TF;

        let out = match self {
            TF::Cast { dtype, options } => {
                // Recursion currently does not propagate NULLs across nesting levels.
                debug_assert!(!matches!(options, CastOptions::Strict));

                input.cast_with_options(dtype, *options)?
            },

            TF::Rename { name } => input.with_name(name.clone()),

            TF::StructFieldsMapping { field_selectors } => {
                use polars_core::prelude::StructChunked;

                let input_s = input._get_backing_series();
                let struct_ca = input_s.struct_().unwrap();
                let field_columns: Vec<Column> = struct_ca.fields_as_columns();

                let field_columns: Vec<Column> = field_selectors
                    .iter()
                    .map(|x| x.select_from_columns(&field_columns, struct_ca.len()))
                    .collect::<PolarsResult<_>>()?;

                input._to_new_from_backing(
                    StructChunked::from_columns(
                        struct_ca.name().clone(),
                        struct_ca.len(),
                        &field_columns,
                    )?
                    .with_outer_validity(struct_ca.rechunk_validity())
                    .into_series(),
                )
            },

            TF::ListValuesMapping { values_selector } => {
                use polars_core::prelude::{LargeListArray, ListChunked};

                let input_list_ca = input._get_backing_series().list().unwrap().clone();

                let values_dtype = {
                    let DataType::List(inner) = input_list_ca.dtype() else {
                        unreachable!()
                    };
                    inner.as_ref()
                };

                let mut values_output_dtype = None;

                let mut out_chunks: Vec<Box<dyn Array>> =
                    Vec::with_capacity(input_list_ca.chunks().len());

                for list_arr in input_list_ca.downcast_iter() {
                    let values: Box<dyn Array> = list_arr.values().clone();
                    let values: Column = unsafe {
                        Series::from_chunks_and_dtype_unchecked(
                            LIST_VALUES_NAME,
                            vec![values],
                            values_dtype,
                        )
                    }
                    .into_column();
                    let len = values.len();

                    let values: Column = values_selector.select_from_columns(&[values], len)?;

                    if values_output_dtype.is_none() {
                        values_output_dtype = Some(values.dtype().clone());
                    }

                    let values: Box<dyn Array> = values
                        .as_materialized_series()
                        .rechunk()
                        .into_chunks()
                        .pop()
                        .unwrap();

                    let list_arr = LargeListArray::new(
                        ArrowDataType::LargeList(Box::new(ArrowField::new(
                            LIST_VALUES_NAME,
                            values.dtype().clone(),
                            true,
                        ))),
                        list_arr.offsets().clone(),
                        values,
                        list_arr.validity().cloned(),
                    );

                    out_chunks.push(list_arr.boxed())
                }

                let mut out =
                    unsafe { ListChunked::from_chunks(input_list_ca.name().clone(), out_chunks) };

                // Ensure logical types are restored.
                out.set_inner_dtype(values_output_dtype.unwrap());

                // Casts on the values should not affect outer NULLs.
                out.retain_flags_from(&input_list_ca, StatisticsFlags::CAN_FAST_EXPLODE_LIST);

                input._to_new_from_backing(out.into_series())
            },

            #[cfg(feature = "dtype-array")]
            TF::FixedSizeListValuesMapping { values_selector } => {
                use arrow::array::FixedSizeListArray;
                use polars_core::prelude::ArrayChunked;

                let input_array_ca = input._get_backing_series().array().unwrap().clone();

                let values_dtype = {
                    let DataType::Array(inner, _) = input_array_ca.dtype() else {
                        unreachable!()
                    };
                    inner.as_ref()
                };

                let mut values_output_dtype = None;

                let mut out_chunks: Vec<Box<dyn Array>> =
                    Vec::with_capacity(input_array_ca.chunks().len());

                for fixed_size_list_arr in input_array_ca.downcast_iter() {
                    let values: Box<dyn Array> = fixed_size_list_arr.values().clone();
                    let values: Column = unsafe {
                        Series::from_chunks_and_dtype_unchecked(
                            LIST_VALUES_NAME,
                            vec![values],
                            values_dtype,
                        )
                    }
                    .into_column();
                    let len = values.len();

                    let values: Column = values_selector.select_from_columns(&[values], len)?;

                    if values_output_dtype.is_none() {
                        values_output_dtype = Some(values.dtype().clone());
                    }

                    let values: Box<dyn Array> = values
                        .as_materialized_series()
                        .rechunk()
                        .into_chunks()
                        .pop()
                        .unwrap();

                    let fixed_size_list_arr = FixedSizeListArray::new(
                        ArrowDataType::FixedSizeList(
                            Box::new(ArrowField::new(
                                LIST_VALUES_NAME,
                                values.dtype().clone(),
                                true,
                            )),
                            fixed_size_list_arr.size(),
                        ),
                        fixed_size_list_arr.len(),
                        values,
                        fixed_size_list_arr.validity().cloned(),
                    );

                    out_chunks.push(fixed_size_list_arr.boxed())
                }

                let mut out =
                    unsafe { ArrayChunked::from_chunks(input_array_ca.name().clone(), out_chunks) };

                // Ensure logical types are restored.
                out.set_inner_dtype(values_output_dtype.unwrap());

                input._to_new_from_backing(out.into_series())
            },
        };

        Ok(out)
    }
}

#[derive(Clone)]
pub struct ColumnSelectorBuilder {
    pub cast_columns_policy: CastColumnsPolicy,
    pub missing_columns_policy: MissingColumnsPolicy,
    // This doesn't take an `ExtraColumnsPolicy`, as it only gets called with the projected output columns.
}

impl ColumnSelectorBuilder {
    /// Build the selector for an output column.
    pub fn build_column_selector(
        &self,
        incoming_schema: &Schema,
        target_name: &PlSmallStr,
        target_dtype: &DataType,
    ) -> PolarsResult<ColumnSelector> {
        let out = if let Some((index, _, incoming_dtype)) = incoming_schema.get_full(target_name) {
            let input = ColumnSelector::Position(index);
            self.attach_transforms(input, incoming_dtype, target_dtype, target_name)?
        } else {
            match &self.missing_columns_policy {
                MissingColumnsPolicy::Insert => ColumnSelector::Constant(Box::new((
                    target_name.clone(),
                    Scalar::null(target_dtype.clone()),
                ))),
                MissingColumnsPolicy::Raise => return Err(missing_column_err(target_name)),
            }
        };

        Ok(out)
    }

    /// Adds transforms on top of the `input_selector` if necessary.
    pub fn attach_transforms(
        &self,
        input_selector: ColumnSelector,
        incoming_dtype: &DataType,
        target_dtype: &DataType,
        // Note: This is used for logging purposes only
        target_name: &str,
    ) -> PolarsResult<ColumnSelector> {
        let mismatch_err = |hint: &str| {
            let hint_spacing = if hint.is_empty() { "" } else { ", " };

            polars_bail!(
                SchemaMismatch:
                "data type mismatch for column {}: incoming: {:?} != target: {:?}{}{}",
                target_name,
                incoming_dtype,
                target_dtype,
                hint_spacing,
                hint,
            )
        };

        // We intercept the nested types first to prevent an expensive recursive eq - recursion
        // is instead done manually through this function.

        if let DataType::Struct(target_fields) = target_dtype {
            let DataType::Struct(incoming_fields) = incoming_dtype else {
                return mismatch_err("");
            };

            // Construct index lookups. We don't construct a full schema here to avoid recursive
            // cloning of nested DataTypes.
            let mut target_fields_lookup: PlHashMap<&str, usize> =
                PlHashMap::with_capacity(target_fields.len());
            let mut incoming_fields_lookup: PlHashMap<&str, usize> =
                PlHashMap::with_capacity(target_fields.len());

            for (i, field) in target_fields.iter().enumerate() {
                if target_fields_lookup.contains_key(field.name.as_str()) {
                    polars_bail!(
                        ComputeError:
                        "duplicate struct field '{}'",
                        &field.name,
                    )
                }

                target_fields_lookup.insert(field.name.as_str(), i);
            }

            for (i, field) in incoming_fields.iter().enumerate() {
                if incoming_fields_lookup.contains_key(field.name.as_str()) {
                    polars_bail!(
                        ComputeError:
                        "duplicate struct field '{}'",
                        &field.name,
                    )
                }

                incoming_fields_lookup.insert(field.name.as_str(), i);
            }

            if matches!(
                &self.cast_columns_policy.extra_struct_fields,
                ExtraColumnsPolicy::Raise
            ) && let Some(extra_field) = incoming_fields
                .iter()
                .find(|x| !target_fields_lookup.contains_key(x.name().as_str()))
            {
                return mismatch_err(&format!(
                    "encountered extra struct field: {}, \
                    hint: specify this field in the schema, or pass \
                    cast_options=pl.ScanCastOptions(extra_struct_fields='ignore')",
                    extra_field.name(),
                ));
            }

            let mut field_selectors: Vec<ColumnSelector> = Vec::with_capacity(target_fields.len());
            let mut is_input_passthrough = incoming_fields.len() == target_fields.len();

            for (output_index, output_field) in target_fields.iter().enumerate() {
                let selector = if let Some(incoming_index) = incoming_fields_lookup
                    .get(output_field.name().as_str())
                    .copied()
                {
                    self.attach_transforms(
                        ColumnSelector::Position(incoming_index),
                        incoming_fields[incoming_index].dtype(),
                        output_field.dtype(),
                        output_field.name().as_str(),
                    )?
                } else {
                    match &self.cast_columns_policy.missing_struct_fields {
                        MissingColumnsPolicy::Insert => ColumnSelector::Constant(Box::new((
                            output_field.name().clone(),
                            Scalar::null(output_field.dtype().clone()),
                        ))),
                        MissingColumnsPolicy::Raise => {
                            return mismatch_err(&format!(
                                "encountered missing struct field: {}, \
                                hint: pass cast_options=pl.ScanCastOptions(missing_struct_fields='insert')",
                                output_field.name(),
                            ));
                        },
                    }
                };

                is_input_passthrough &= match &selector {
                    ColumnSelector::Position(input_index) => *input_index == output_index,
                    _ => false,
                };

                field_selectors.push(selector);
            }

            return Ok(if is_input_passthrough {
                input_selector
            } else {
                ColumnTransform::StructFieldsMapping {
                    field_selectors: field_selectors.into_boxed_slice(),
                }
                .into_selector(input_selector)
            });
        }

        if let DataType::List(target_inner) = target_dtype {
            let DataType::List(incoming_inner) = incoming_dtype else {
                return mismatch_err("");
            };

            return Ok(
                match self.attach_transforms(
                    ColumnSelector::Position(0),
                    incoming_inner,
                    target_inner,
                    target_name,
                )? {
                    ColumnSelector::Position(0) => input_selector,
                    values_selector => ColumnTransform::ListValuesMapping { values_selector }
                        .into_selector(input_selector),
                },
            );
        }

        #[cfg(feature = "dtype-array")]
        if let DataType::Array(target_inner, target_width) = target_dtype {
            let DataType::Array(incoming_inner, incoming_width) = incoming_dtype else {
                return mismatch_err("");
            };

            if incoming_width != target_width {
                return mismatch_err("");
            }

            return Ok(
                match self.attach_transforms(
                    ColumnSelector::Position(0),
                    incoming_inner,
                    target_inner,
                    target_name,
                )? {
                    ColumnSelector::Position(0) => input_selector,
                    values_selector => {
                        ColumnTransform::FixedSizeListValuesMapping { values_selector }
                            .into_selector(input_selector)
                    },
                },
            );
        }

        // Eq here should be cheap as we have intercepted all nested types above.

        debug_assert!(!target_dtype.is_nested());

        if target_dtype == incoming_dtype {
            return Ok(input_selector);
        }

        //
        // After this point the dtypes are mismatching.
        //

        // Note: Only call this with non-nested types for performance
        let materialize_unknown = |dtype: &DataType| -> std::borrow::Cow<DataType> {
            dtype
                .clone()
                .materialize_unknown(true)
                .map(std::borrow::Cow::Owned)
                .unwrap_or(std::borrow::Cow::Borrowed(incoming_dtype))
        };

        let incoming_dtype = materialize_unknown(incoming_dtype);
        let target_dtype = materialize_unknown(target_dtype);

        // Attaches a cast to the target dtype.
        let attach_cast = |options: CastOptions| -> PolarsResult<ColumnSelector> {
            Ok(ColumnTransform::Cast {
                dtype: target_dtype.clone().into_owned(),
                options,
            }
            .into_selector(input_selector))
        };

        let incoming_dtype = incoming_dtype.as_ref();
        let target_dtype = target_dtype.as_ref();

        if target_dtype.is_integer() && incoming_dtype.is_integer() {
            if !self.cast_columns_policy.integer_upcast {
                return mismatch_err(
                    "hint: pass cast_options=pl.ScanCastOptions(integer_cast='upcast')",
                );
            }

            return match get_numeric_upcast_supertype_lossless(incoming_dtype, target_dtype) {
                Some(ref v) if v == target_dtype => {
                    // Use overflowing on lossless cast to elide validation.
                    attach_cast(CastOptions::Overflowing)
                },
                _ => mismatch_err("incoming dtype cannot safely cast to target dtype"),
            };
        }

        if target_dtype.is_float() && incoming_dtype.is_float() {
            match (target_dtype, incoming_dtype) {
                (DataType::Float64, DataType::Float32) => {
                    if !self.cast_columns_policy.float_upcast {
                        return mismatch_err(
                            "hint: pass cast_options=pl.ScanCastOptions(float_cast='upcast')",
                        );
                    }
                },

                (DataType::Float32, DataType::Float64) => {
                    if !self.cast_columns_policy.float_downcast {
                        return mismatch_err(
                            "hint: pass cast_options=pl.ScanCastOptions(float_cast='downcast')",
                        );
                    }
                },

                _ => unreachable!(),
            };

            return attach_cast(CastOptions::NonStrict);
        }

        if let (
            DataType::Datetime(target_unit, target_zone),
            DataType::Datetime(incoming_unit, incoming_zone),
        ) = (target_dtype, incoming_dtype)
        {
            use polars_core::prelude::{TimeUnit, TimeZone};

            // Check timezone
            if !self.cast_columns_policy.datetime_convert_timezone
                && !TimeZone::eq_none_as_utc(incoming_zone.as_ref(), target_zone.as_ref())
            {
                return mismatch_err(
                    "hint: pass cast_options=pl.ScanCastOptions(datetime_cast='convert-timezone')",
                );
            }

            // Check unit
            if target_unit != incoming_unit {
                match (incoming_unit, target_unit) {
                    (TimeUnit::Nanoseconds, _) => {
                        if !self.cast_columns_policy.datetime_nanoseconds_downcast {
                            return mismatch_err(
                                "hint: pass cast_options=pl.ScanCastOptions(datetime_cast='nanosecond-downcast')",
                            );
                        }
                    },

                    (TimeUnit::Microseconds, TimeUnit::Milliseconds) => {
                        if !self.cast_columns_policy.datetime_microseconds_downcast {
                            // TODO
                            return mismatch_err(
                                "unimplemented: 'microsecond-downcast' in scan cast options",
                            );
                        }
                    },

                    _ => return mismatch_err(""),
                };
            }

            // Dtype differs and we are allowed to coerce
            return attach_cast(CastOptions::NonStrict);
        }

        mismatch_err("")
    }

    /// Adds transforms on top of the `input_selector` if necessary.
    pub fn attach_iceberg_transforms(
        &self,
        input_selector: ColumnSelector,
        incoming_column: &IcebergColumn,
        target_column: &IcebergColumn,
    ) -> PolarsResult<ColumnSelector> {
        let selector = (|| {
            let target_dtype = &target_column.type_;
            let incoming_dtype = &incoming_column.type_;

            let mismatch_err = |hint: &str| {
                let hint_spacing = if hint.is_empty() { "" } else { ", " };
                let target_name = &target_column.name;

                polars_bail!(
                    SchemaMismatch:
                    "data type mismatch for column {}: incoming: {:?} != target: {:?}{}{}",
                    target_name,
                    incoming_column,
                    target_column,
                    hint_spacing,
                    hint,
                )
            };

            use IcebergColumnType as ICT;

            let out = match target_dtype {
                ICT::Struct(target_fields) => {
                    let ICT::Struct(incoming_fields) = incoming_dtype else {
                        return mismatch_err("");
                    };

                    if matches!(
                        &self.cast_columns_policy.extra_struct_fields,
                        ExtraColumnsPolicy::Raise
                    ) && let Some(extra_col) =
                        incoming_fields
                            .iter()
                            .find_map(|(physical_id, iceberg_column)| {
                                (!target_fields.contains_key(physical_id)).then_some(iceberg_column)
                            })
                    {
                        return mismatch_err(&format!(
                            "encountered extra struct field: {}, \
                            hint: specify this field in the schema, or pass \
                            cast_options=pl.ScanCastOptions(extra_struct_fields='ignore')",
                            &extra_col.name,
                        ));
                    }

                    let mut field_selectors: Vec<ColumnSelector> =
                        Vec::with_capacity(target_fields.len());
                    let mut is_input_passthrough = incoming_fields.len() == target_fields.len();

                    for (output_index, (physical_id, output_column)) in
                        target_fields.iter().enumerate()
                    {
                        let selector = if let Some((incoming_index, _, incoming_column)) =
                            incoming_fields.get_full(physical_id)
                        {
                            self.attach_iceberg_transforms(
                                ColumnSelector::Position(incoming_index),
                                incoming_column,
                                output_column,
                            )?
                        } else {
                            match &self.cast_columns_policy.missing_struct_fields {
                                MissingColumnsPolicy::Insert => {
                                    ColumnSelector::Constant(Box::new((
                                        output_column.name.clone(),
                                        Scalar::null(output_column.type_.to_polars_dtype()),
                                    )))
                                },
                                MissingColumnsPolicy::Raise => {
                                    return mismatch_err(&format!(
                                        "encountered missing struct field: {}, \
                                        hint: pass cast_options=pl.ScanCastOptions(missing_struct_fields='insert')",
                                        &output_column.name,
                                    ));
                                },
                            }
                        };

                        is_input_passthrough &= match &selector {
                            ColumnSelector::Position(input_index) => *input_index == output_index,
                            _ => false,
                        };

                        field_selectors.push(selector);
                    }

                    if is_input_passthrough {
                        input_selector
                    } else {
                        ColumnTransform::StructFieldsMapping {
                            field_selectors: field_selectors.into_boxed_slice(),
                        }
                        .into_selector(input_selector)
                    }
                },

                ICT::List(target_inner) => {
                    let ICT::List(incoming_inner) = incoming_dtype else {
                        return mismatch_err("");
                    };

                    if incoming_inner.physical_id != target_inner.physical_id {
                        return mismatch_err("physical ID mismatch for list values column");
                    }

                    match self.attach_iceberg_transforms(
                        ColumnSelector::Position(0),
                        incoming_inner,
                        target_inner,
                    )? {
                        ColumnSelector::Position(0) => input_selector,
                        values_selector => ColumnTransform::ListValuesMapping { values_selector }
                            .into_selector(input_selector),
                    }
                },

                ICT::FixedSizeList(target_inner, target_width) => {
                    feature_gated!("dtype-array", {
                        let ICT::FixedSizeList(incoming_inner, incoming_width) = incoming_dtype
                        else {
                            return mismatch_err("");
                        };

                        if incoming_width != target_width {
                            return mismatch_err("");
                        }

                        if incoming_inner.physical_id != target_inner.physical_id {
                            return mismatch_err(
                                "physical ID mismatch for fixed size list values column",
                            );
                        }

                        match self.attach_iceberg_transforms(
                            ColumnSelector::Position(0),
                            incoming_inner,
                            target_inner,
                        )? {
                            ColumnSelector::Position(0) => input_selector,
                            values_selector => {
                                ColumnTransform::FixedSizeListValuesMapping { values_selector }
                                    .into_selector(input_selector)
                            },
                        }
                    })
                },

                ICT::Primitive {
                    dtype: target_dtype,
                } => {
                    let ICT::Primitive {
                        dtype: incoming_dtype,
                    } = incoming_dtype
                    else {
                        return mismatch_err("");
                    };

                    // Primitive type defers to the native `attach_transforms()` function.
                    self.attach_transforms(
                        input_selector,
                        incoming_dtype,
                        target_dtype,
                        &target_column.name,
                    )?
                },
            };

            Ok(out)
        })()?;

        let selector = if incoming_column.name != target_column.name {
            ColumnSelector::Transformed(Box::new((
                selector,
                ColumnTransform::Rename {
                    name: target_column.name.clone(),
                },
            )))
        } else {
            selector
        };

        Ok(selector)
    }
}
