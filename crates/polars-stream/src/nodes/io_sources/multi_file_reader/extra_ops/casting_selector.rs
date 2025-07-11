use arrow::array::{Array, FixedSizeListArray};
use arrow::datatypes::{ArrowDataType, Field as ArrowField};
use polars_core::chunked_array::cast::CastOptions;
use polars_core::chunked_array::flags::StatisticsFlags;
use polars_core::frame::column::ScalarColumn;
use polars_core::prelude::{
    ArrayChunked, Column, DataType, InitHashMaps, IntoColumn, LargeListArray, ListChunked,
    PlHashMap, StructChunked, TimeUnit, TimeZone,
};
use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_core::utils::get_numeric_upcast_supertype_lossless;
use polars_error::{PolarsResult, polars_bail};
use polars_plan::dsl::{CastColumnsPolicy, MissingColumnsPolicy};
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::io_sources::multi_file_reader::extra_ops::apply_extra_columns_policy_impl;

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
#[derive(Debug)]
pub enum CastingSelector {
    // Note that we Box enum variants to keep `CastingSelector` small (16 bytes).
    // This is an optimization that benefits the case where there are many `Position` selectors.

    // Leaf selectors
    /// Take the column at this position.
    Position(usize),
    /// Materialize a constant column.
    Constant(Box<ScalarColumn>),

    /// `(input_selector, _)`
    Nested(Box<(CastingSelector, NestedCastingSelector)>),
}

#[derive(Debug)]
pub enum NestedCastingSelector {
    /// Cast the column to a dtype.
    Cast {
        dtype: DataType,
        options: CastOptions,
    },
    /// Set the name of the column.
    #[expect(unused)]
    Rename {
        name: PlSmallStr,
    },
    /// Construct a struct column by applying column selectors onto the field arrays.
    StructFieldsMapping {
        field_selectors: Box<[CastingSelector]>,
    },
    /// Construct a list column by applying column selectors onto the values array.
    ListValuesMapping {
        values_selector: CastingSelector,
    },
    FixedSizeListValuesMapping {
        values_selector: CastingSelector,
    },
}

impl NestedCastingSelector {
    pub fn into_selector(self, input_selector: CastingSelector) -> CastingSelector {
        CastingSelector::Nested(Box::new((input_selector, self)))
    }
}

impl CastingSelector {
    pub fn select_from_columns(
        &self,
        columns: &[Column],
        output_height: usize,
    ) -> PolarsResult<Column> {
        use CastingSelector as S;

        Ok(match self {
            S::Position(i) => columns[*i].clone(),

            S::Constant(scalar_column) => scalar_column
                .clone()
                .into_column()
                .new_from_index(0, output_height),

            S::Nested(nested) => {
                let input: Column = nested.0.select_from_columns(columns, output_height)?;

                use NestedCastingSelector as NS;

                match &nested.1 {
                    NS::Cast { dtype, options } => input.cast_with_options(dtype, *options)?,

                    NS::Rename { .. } => {
                        // TODO
                        // Will be used for Iceberg column mapping.
                        unreachable!()
                    },

                    NS::StructFieldsMapping { field_selectors } => {
                        let struct_ca = input.struct_().unwrap();
                        let field_columns: Vec<Column> = struct_ca.fields_as_columns();

                        let field_columns: Vec<Column> = field_selectors
                            .iter()
                            .map(|x| x.select_from_columns(&field_columns, struct_ca.len()))
                            .collect::<PolarsResult<_>>()?;

                        StructChunked::from_columns(
                            struct_ca.name().clone(),
                            struct_ca.len(),
                            &field_columns,
                        )?
                        .into_column()
                    },

                    NS::ListValuesMapping { values_selector } => {
                        let list_ca = input.list().unwrap().clone();

                        let values_dtype = {
                            let DataType::List(inner) = list_ca.dtype() else {
                                unreachable!()
                            };
                            inner.as_ref()
                        };

                        let mut values_output_dtype = None;

                        let mut out_chunks: Vec<Box<dyn Array>> =
                            Vec::with_capacity(list_ca.chunks().len());

                        for list_arr in list_ca.downcast_iter() {
                            let values: Box<dyn Array> = list_arr.values().clone();
                            let values: Column = unsafe {
                                Series::from_chunks_and_dtype_unchecked(
                                    PlSmallStr::from_static("item"),
                                    vec![values],
                                    values_dtype,
                                )
                            }
                            .into_column();
                            let len = values.len();

                            let values: Column =
                                values_selector.select_from_columns(&[values], len)?;

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
                                    PlSmallStr::from_static("item"),
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
                            unsafe { ListChunked::from_chunks(list_ca.name().clone(), out_chunks) };

                        // Ensure logical types are restored.
                        out.set_inner_dtype(values_output_dtype.unwrap());

                        // Casts on the values should not affect outer NULLs.
                        out.retain_flags_from(
                            input.list().unwrap(),
                            StatisticsFlags::CAN_FAST_EXPLODE_LIST,
                        );

                        out.into_column()
                    },

                    NS::FixedSizeListValuesMapping { values_selector } => {
                        let array_ca = input.array().unwrap().clone();

                        let values_dtype = {
                            let DataType::Array(inner, _) = array_ca.dtype() else {
                                unreachable!()
                            };
                            inner.as_ref()
                        };

                        let mut values_output_dtype = None;

                        let mut out_chunks: Vec<Box<dyn Array>> =
                            Vec::with_capacity(array_ca.chunks().len());

                        for fixed_size_list_arr in array_ca.downcast_iter() {
                            let values: Box<dyn Array> = fixed_size_list_arr.values().clone();
                            let values: Column = unsafe {
                                Series::from_chunks_and_dtype_unchecked(
                                    PlSmallStr::from_static("item"),
                                    vec![values],
                                    values_dtype,
                                )
                            }
                            .into_column();
                            let len = values.len();

                            let values: Column =
                                values_selector.select_from_columns(&[values], len)?;

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
                                        PlSmallStr::from_static("item"),
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

                        let mut out = unsafe {
                            ArrayChunked::from_chunks(array_ca.name().clone(), out_chunks)
                        };

                        // Ensure logical types are restored.
                        out.set_inner_dtype(values_output_dtype.unwrap());

                        out.into_column()
                    },
                }
            },
        })
    }
}

#[derive(Clone)]
pub struct CastingSelectorBuilder {
    pub cast_columns_policy: CastColumnsPolicy,
    pub missing_columns_policy: MissingColumnsPolicy,
    // This doesn't take an `ExtraColumnsPolicy`, as it only gets called with the projected output columns.
}

impl CastingSelectorBuilder {
    pub fn build_selector_for_column(
        &self,
        incoming_schema: &Schema,
        target_name: &PlSmallStr,
        target_dtype: &DataType,
    ) -> PolarsResult<CastingSelector> {
        let out = if let Some((index, _, incoming_dtype)) = incoming_schema.get_full(target_name) {
            let input = CastingSelector::Position(index);
            self.build_casting_selector(input, incoming_dtype, target_dtype, target_name)?
        } else {
            match &self.missing_columns_policy {
                MissingColumnsPolicy::Insert => CastingSelector::Constant(Box::new(
                    ScalarColumn::full_null(target_name.clone(), 1, target_dtype),
                )),
                MissingColumnsPolicy::Raise => polars_bail!(
                    ColumnNotFound:
                    "did not find column {}, consider passing `missing_columns='insert'`",
                    target_name,
                ),
            }
        };

        Ok(out)
    }

    /// Adds casting on top of a selector where necessary.
    pub fn build_casting_selector(
        &self,
        input_selector: CastingSelector,
        incoming_dtype: &DataType,
        target_dtype: &DataType,
        // Note: This is used for logging purposes only
        target_name: &str,
    ) -> PolarsResult<CastingSelector> {
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

            // Construct lookups
            let mut target_fields_lookup = PlHashMap::with_capacity(target_fields.len());

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

            let mut incoming_fields_lookup = PlHashMap::with_capacity(target_fields.len());

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

            apply_extra_columns_policy_impl(
                &self.cast_columns_policy.extra_struct_fields,
                &|x| target_fields_lookup.contains_key(x),
                &mut incoming_fields.iter().map(|x| x.name.as_str()),
            )?;

            let mut field_selectors: Vec<CastingSelector> = Vec::with_capacity(target_fields.len());
            let mut is_input_passthrough = incoming_fields.len() == target_fields.len();

            for (target_index, target_field) in target_fields.iter().enumerate() {
                let selector = if let Some(incoming_index) = incoming_fields_lookup
                    .get(target_field.name().as_str())
                    .copied()
                {
                    self.build_casting_selector(
                        CastingSelector::Position(incoming_index),
                        incoming_fields[incoming_index].dtype(),
                        &target_field.dtype(),
                        target_field.name().as_str(),
                    )?
                } else {
                    match &self.cast_columns_policy.missing_struct_fields {
                        MissingColumnsPolicy::Insert => {
                            CastingSelector::Constant(Box::new(ScalarColumn::full_null(
                                target_field.name().clone(),
                                1,
                                target_field.dtype(),
                            )))
                        },
                        MissingColumnsPolicy::Raise => polars_bail!(
                            ColumnNotFound:
                            "did not find column {}, consider passing `missing_columns='insert'`",
                            target_field.name(),
                        ),
                    }
                };

                is_input_passthrough &= match &selector {
                    CastingSelector::Position(i) => *i == target_index,
                    _ => false,
                };

                field_selectors.push(selector);
            }

            return Ok(if is_input_passthrough {
                input_selector
            } else {
                NestedCastingSelector::StructFieldsMapping {
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
                match self.build_casting_selector(
                    CastingSelector::Position(0),
                    incoming_inner,
                    target_inner,
                    target_name,
                )? {
                    CastingSelector::Position(0) => input_selector,
                    values_selector => NestedCastingSelector::ListValuesMapping { values_selector }
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
                match self.build_casting_selector(
                    CastingSelector::Position(0),
                    incoming_inner,
                    target_inner,
                    target_name,
                )? {
                    CastingSelector::Position(0) => input_selector,
                    values_selector => {
                        NestedCastingSelector::FixedSizeListValuesMapping { values_selector }
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

        let build_casting_selector = |options: CastOptions| -> PolarsResult<CastingSelector> {
            Ok(NestedCastingSelector::Cast {
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
                    build_casting_selector(CastOptions::Overflowing)
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

            return build_casting_selector(CastOptions::NonStrict);
        }

        if let (
            DataType::Datetime(target_unit, target_zone),
            DataType::Datetime(incoming_unit, incoming_zone),
        ) = (target_dtype, incoming_dtype)
        {
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
            return build_casting_selector(CastOptions::Strict);
        }

        mismatch_err("")
    }
}
