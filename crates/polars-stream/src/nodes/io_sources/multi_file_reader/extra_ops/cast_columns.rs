use polars_core::chunked_array::cast::CastOptions;
use polars_core::frame::DataFrame;
use polars_core::prelude::{DataType, PlHashMap, TimeUnit, TimeZone};
use polars_core::schema::SchemaRef;
use polars_core::utils::get_numeric_upcast_supertype_lossless;
use polars_error::{PolarsResult, polars_bail};
use polars_plan::dsl::{CastColumnsPolicy, ExtraColumnsPolicy, MissingColumnsPolicy};

#[derive(Debug)]
pub struct CastColumns {
    casting_list: Vec<ColumnCast>,
}

/// We rely on the default cast dispatch after performing validation on the dtypes, as the
/// default cast dispatch does everything that we need (for now).
#[derive(Debug)]
struct ColumnCast {
    index: usize,
    dtype: DataType,
}

impl CastColumns {
    pub fn try_init_from_policy(
        policy: &CastColumnsPolicy,
        target_schema: &SchemaRef,
        incoming_schema: &SchemaRef,
    ) -> PolarsResult<Option<Self>> {
        Self::try_init_from_policy_from_iter(
            policy,
            target_schema,
            &mut incoming_schema
                .iter()
                .map(|(name, dtype)| (name.as_ref(), dtype)),
        )
    }

    pub fn try_init_from_policy_from_iter(
        policy: &CastColumnsPolicy,
        target_schema: &SchemaRef,
        incoming_schema_iter: &mut dyn Iterator<Item = (&str, &DataType)>,
    ) -> PolarsResult<Option<Self>> {
        let get_target_dtype = |name: &str| {
            target_schema.get(name).unwrap_or_else(|| {
                panic!("impl error: column '{}' should exist in casting map", name)
            })
        };

        let mut casting_list = vec![];

        for (i, (name, incoming_dtype)) in incoming_schema_iter.enumerate() {
            let target_dtype = get_target_dtype(name);

            if PolicyWrap(policy).should_cast_column(name, target_dtype, incoming_dtype)? {
                casting_list.push(ColumnCast {
                    index: i,
                    dtype: target_dtype.clone(),
                })
            }
        }

        if casting_list.is_empty() {
            Ok(None)
        } else {
            Ok(Some(CastColumns { casting_list }))
        }
    }

    pub fn apply_cast(&self, df: &mut DataFrame) -> PolarsResult<()> {
        // Should only be called if there's something to cast.
        debug_assert!(!self.casting_list.is_empty());

        df.clear_schema();

        let columns = unsafe { df.get_columns_mut() };

        for ColumnCast { index, dtype } in &self.casting_list {
            *columns.get_mut(*index).unwrap() =
                columns[*index].cast_with_options(dtype, CastOptions::Strict)?;
        }

        Ok(())
    }
}

struct PolicyWrap<'a>(&'a CastColumnsPolicy);

impl std::ops::Deref for PolicyWrap<'_> {
    type Target = CastColumnsPolicy;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl PolicyWrap<'_> {
    /// # Returns
    /// * Ok(true): Cast needed to target dtype
    /// * Ok(false): No casting needed
    /// * Err(_): Forbidden by configuration, or incompatible types.
    fn should_cast_column(
        &self,
        column_name: &str,
        target_dtype: &DataType,
        incoming_dtype: &DataType,
    ) -> PolarsResult<bool> {
        let mismatch_err = |hint: &str| {
            let hint_spacing = if hint.is_empty() { "" } else { ", " };

            polars_bail!(
                SchemaMismatch:
                "data type mismatch for column {}: incoming: {:?} != target: {:?}{}{}",
                column_name,
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

            let incoming_fields_schema = PlHashMap::from_iter(
                incoming_fields
                    .iter()
                    .enumerate()
                    .map(|(i, fld)| (fld.name.as_str(), (i, &fld.dtype))),
            );

            let mut should_cast = incoming_fields.len() != target_fields.len();

            for (target_idx, target_field) in target_fields.iter().enumerate() {
                let Some((incoming_idx, incoming_field_dtype)) =
                    incoming_fields_schema.get(target_field.name().as_str())
                else {
                    match self.missing_struct_fields {
                        MissingColumnsPolicy::Raise => {
                            return mismatch_err(&format!(
                                "encountered missing struct field: {}, \
                                hint: pass cast_options=pl.ScanCastOptions(missing_struct_fields='insert')",
                                target_field.name(),
                            ));
                        },
                        MissingColumnsPolicy::Insert => {
                            should_cast = true;
                            // Must keep checking the rest of the fields.
                            continue;
                        },
                    };
                };

                // # Note
                // We also need to cast if the struct fields are out of order. Currently there is
                // no API parameter to control this - we always do this by default.
                should_cast |= *incoming_idx != target_idx;

                should_cast |= self.should_cast_column(
                    column_name,
                    &target_field.dtype,
                    incoming_field_dtype,
                )?;
            }

            // Casting is also needed if there are extra fields, check them here.

            // Take and re-use hashmap
            let mut target_fields_schema = incoming_fields_schema;
            target_fields_schema.clear();

            target_fields_schema.extend(
                target_fields
                    .iter()
                    .enumerate()
                    .map(|(i, fld)| (fld.name.as_str(), (i, &fld.dtype))),
            );

            for fld in incoming_fields {
                if !target_fields_schema.contains_key(fld.name.as_str()) {
                    match self.extra_struct_fields {
                        ExtraColumnsPolicy::Ignore => {
                            should_cast = true;
                            break;
                        },
                        ExtraColumnsPolicy::Raise => {
                            return mismatch_err(&format!(
                                "encountered extra struct field: {}, \
                                hint: pass cast_options=pl.ScanCastOptions(extra_struct_fields='ignore')",
                                &fld.name,
                            ));
                        },
                    }
                }
            }

            return Ok(should_cast);
        }

        if let DataType::List(target_inner) = target_dtype {
            let DataType::List(incoming_inner) = incoming_dtype else {
                return mismatch_err("");
            };

            return self.should_cast_column(column_name, target_inner, incoming_inner);
        }

        #[cfg(feature = "dtype-array")]
        if let DataType::Array(target_inner, target_width) = target_dtype {
            let DataType::Array(incoming_inner, incoming_width) = incoming_dtype else {
                return mismatch_err("");
            };

            if incoming_width != target_width {
                return mismatch_err("");
            }

            return self.should_cast_column(column_name, target_inner, incoming_inner);
        }

        // Eq here should be cheap as we have intercepted all nested types above.

        debug_assert!(!target_dtype.is_nested());

        if target_dtype == incoming_dtype {
            return Ok(false);
        }

        //
        // After this point the dtypes are mismatching.
        //

        if target_dtype.is_integer() && incoming_dtype.is_integer() {
            if !self.integer_upcast {
                return mismatch_err(
                    "hint: pass cast_options=pl.ScanCastOptions(integer_cast='upcast')",
                );
            }

            return match get_numeric_upcast_supertype_lossless(incoming_dtype, target_dtype) {
                Some(ref v) if v == target_dtype => Ok(true),
                _ => mismatch_err("incoming dtype cannot safely cast to target dtype"),
            };
        }

        if target_dtype.is_float() && incoming_dtype.is_float() {
            return match (target_dtype, incoming_dtype) {
                (DataType::Float64, DataType::Float32) => {
                    if self.float_upcast {
                        Ok(true)
                    } else {
                        mismatch_err(
                            "hint: pass cast_options=pl.ScanCastOptions(float_cast='upcast')",
                        )
                    }
                },

                (DataType::Float32, DataType::Float64) => {
                    if self.float_downcast {
                        Ok(true)
                    } else {
                        mismatch_err(
                            "hint: pass cast_options=pl.ScanCastOptions(float_cast='downcast')",
                        )
                    }
                },

                _ => unreachable!(),
            };
        }

        if let (
            DataType::Datetime(target_unit, target_zone),
            DataType::Datetime(incoming_unit, incoming_zone),
        ) = (target_dtype, incoming_dtype)
        {
            // Check timezone
            if !self.datetime_convert_timezone
                && !TimeZone::eq_none_as_utc(incoming_zone.as_ref(), target_zone.as_ref())
            {
                return mismatch_err(
                    "hint: pass cast_options=pl.ScanCastOptions(datetime_cast='convert-timezone')",
                );
            }

            // Check unit
            if target_unit != incoming_unit {
                return if let TimeUnit::Nanoseconds = incoming_unit {
                    if self.datetime_nanoseconds_downcast {
                        Ok(true)
                    } else {
                        mismatch_err(
                            "hint: pass cast_options=pl.ScanCastOptions(datetime_cast='nanosecond-downcast')",
                        )
                    }
                } else {
                    // Currently don't have parameter for controlling arbitrary time unit casting.
                    mismatch_err("")
                };
            }

            // Dtype differs and we are allowed to coerce
            return Ok(true);
        }

        mismatch_err("")
    }
}
