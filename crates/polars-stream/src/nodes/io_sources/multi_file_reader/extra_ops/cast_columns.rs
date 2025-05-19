use polars_core::chunked_array::cast::CastOptions;
use polars_core::frame::DataFrame;
use polars_core::prelude::DataType;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_plan::dsl::CastColumnsPolicy;

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
    pub fn should_cast_column(
        &self,
        column_name: &str,
        target_dtype: &DataType,
        incoming_dtype: &DataType,
    ) -> PolarsResult<bool> {
        self.0
            .should_cast_column(column_name, target_dtype, incoming_dtype)
    }
}
