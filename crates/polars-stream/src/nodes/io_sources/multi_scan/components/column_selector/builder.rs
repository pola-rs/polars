use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::{Column, DataType, InitHashMaps, IntoColumn, PlHashMap};
use polars_core::scalar::Scalar;
use polars_core::schema::Schema;
use polars_core::schema::iceberg::{IcebergColumn, IcebergColumnType, LIST_ELEMENT_DEFAULT_ID};
use polars_core::series::{IntoSeries, Series};
use polars_core::utils::get_numeric_upcast_supertype_lossless;
use polars_error::{PolarsResult, feature_gated, polars_bail};
use polars_plan::dsl::{CastColumnsPolicy, ExtraColumnsPolicy, MissingColumnsPolicy};
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::io_sources::multi_scan::components::column_selector::{
    ColumnSelector, ColumnTransform,
};
use crate::nodes::io_sources::multi_scan::components::default_field_values::IcebergDefaultValueProviderRef;
use crate::nodes::io_sources::multi_scan::components::errors::missing_column_err;

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

        if incoming_dtype.is_null() && !target_dtype.is_null() {
            return if self.cast_columns_policy.null_upcast {
                Ok(ColumnTransform::Cast {
                    dtype: target_dtype.clone(),
                    options: CastOptions::NonStrict,
                }
                .into_selector(input_selector))
            } else {
                mismatch_err("unimplemented: 'null-upcast' in scan cast options")
            };
        }

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
        debug_assert!(!target_dtype.is_nested() || target_dtype.is_extension());

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
            return if self.cast_columns_policy.integer_upcast {
                match get_numeric_upcast_supertype_lossless(incoming_dtype, target_dtype) {
                    Some(ref v) if v == target_dtype => {
                        // Use overflowing on lossless cast to elide validation.
                        attach_cast(CastOptions::Overflowing)
                    },
                    _ => mismatch_err("incoming dtype cannot safely cast to target dtype"),
                }
            } else {
                mismatch_err("hint: pass cast_options=pl.ScanCastOptions(integer_cast='upcast')")
            };
        }

        if target_dtype.is_float() && incoming_dtype.is_float() {
            match (target_dtype, incoming_dtype) {
                (DataType::Float64, DataType::Float32)
                | (DataType::Float64, DataType::Float16)
                | (DataType::Float32, DataType::Float16) => {
                    if !self.cast_columns_policy.float_upcast {
                        return mismatch_err(
                            "hint: pass cast_options=pl.ScanCastOptions(float_cast='upcast')",
                        );
                    }
                },

                (DataType::Float16, DataType::Float32)
                | (DataType::Float16, DataType::Float64)
                | (DataType::Float32, DataType::Float64) => {
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

        if target_dtype.is_float() && incoming_dtype.is_integer() {
            return if self.cast_columns_policy.integer_to_float_cast {
                attach_cast(CastOptions::Overflowing)
            } else {
                mismatch_err(
                    "hint: pass cast_options=pl.ScanCastOptions(integer_cast='allow-float')",
                )
            };
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

        if target_dtype.is_string() && incoming_dtype.is_categorical() {
            return if self.cast_columns_policy.categorical_to_string {
                attach_cast(CastOptions::NonStrict)
            } else {
                mismatch_err(
                    "hint: pass cast_options=pl.ScanCastOptions(categorical_to_string='allow')",
                )
            };
        }

        mismatch_err("")
    }

    /// Adds transforms on top of the `input_selector` if necessary.
    pub fn attach_iceberg_transforms(
        &self,
        input_selector: ColumnSelector,
        incoming_column: &IcebergColumn,
        target_column: &IcebergColumn,
        mut iceberg_default_value_provider: Option<IcebergDefaultValueProviderRef>,
    ) -> PolarsResult<ColumnSelector> {
        use IcebergColumnType as ICT;

        match &target_column.type_ {
            ICT::FixedSizeList(..) | ICT::List(_) => iceberg_default_value_provider = None,
            ICT::Struct(_) | ICT::Primitive { .. } => {},
        }

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
                                iceberg_default_value_provider,
                            )?
                        } else {
                            match &self.cast_columns_policy.missing_struct_fields {
                                MissingColumnsPolicy::Insert => {
                                    ColumnSelector::Constant(Box::new((
                                        output_column.name.clone(),
                                        iceberg_default_value_provider
                                            .map(|x| build_iceberg_default_value(x, target_column))
                                            .transpose()?
                                            .flatten()
                                            .unwrap_or_else(|| {
                                                Scalar::null(output_column.type_.to_polars_dtype())
                                            }),
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

                    if incoming_inner.physical_id != target_inner.physical_id
                        && incoming_inner.physical_id != LIST_ELEMENT_DEFAULT_ID
                        && target_inner.physical_id != LIST_ELEMENT_DEFAULT_ID
                    {
                        return mismatch_err("physical ID mismatch for list values column");
                    }

                    match self.attach_iceberg_transforms(
                        ColumnSelector::Position(0),
                        incoming_inner,
                        target_inner,
                        iceberg_default_value_provider,
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

                        if incoming_inner.physical_id != target_inner.physical_id
                            && incoming_inner.physical_id != LIST_ELEMENT_DEFAULT_ID
                            && target_inner.physical_id != LIST_ELEMENT_DEFAULT_ID
                        {
                            return mismatch_err(
                                "physical ID mismatch for fixed size list values column",
                            );
                        }

                        match self.attach_iceberg_transforms(
                            ColumnSelector::Position(0),
                            incoming_inner,
                            target_inner,
                            iceberg_default_value_provider,
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

pub fn build_iceberg_default_value(
    iceberg_default_value_provider: IcebergDefaultValueProviderRef,
    target_column: &IcebergColumn,
) -> PolarsResult<Option<Scalar>> {
    let Some(c) = build_iceberg_default_value_impl(iceberg_default_value_provider, target_column)?
    else {
        return Ok(None);
    };

    assert_eq!(c.len(), 1);

    Ok(Some(Scalar::new(
        c.dtype().clone(),
        c.get(0).unwrap().into_static(),
    )))
}

pub fn build_iceberg_default_value_impl(
    iceberg_default_value_provider: IcebergDefaultValueProviderRef,
    target_column: &IcebergColumn,
) -> PolarsResult<Option<Column>> {
    use IcebergColumnType as ICT;

    match &target_column.type_ {
        ICT::FixedSizeList(..) | ICT::List(_) => Ok(None),

        ICT::Struct(fields) => {
            use polars_core::prelude::StructChunked;

            let mut is_input_passthrough = true;

            let mut field_columns = Vec::with_capacity(fields.len());

            for field in fields.values() {
                let opt_default =
                    build_iceberg_default_value_impl(iceberg_default_value_provider, field)?;

                is_input_passthrough &= opt_default.is_none();

                field_columns.push(if let Some(default) = opt_default {
                    assert_eq!(default.len(), 1);
                    default.with_name(field.name.clone())
                } else {
                    Column::full_null(field.name.clone(), 1, &field.type_.to_polars_dtype())
                });
            }

            if is_input_passthrough {
                Ok(None)
            } else {
                Ok(Some(
                    StructChunked::from_columns(target_column.name.clone(), 1, &field_columns)?
                        .into_series()
                        .into_column(),
                ))
            }
        },

        ICT::Primitive { dtype } => Ok(iceberg_default_value_provider
            .get_default_value(target_column.physical_id)?
            .map(|any_value| {
                debug_assert_eq!(&any_value.dtype(), dtype);

                PolarsResult::Ok(
                    Series::from_any_values(PlSmallStr::EMPTY, &[any_value], true)?.into_column(),
                )
            })
            .transpose()?),
    }
}
