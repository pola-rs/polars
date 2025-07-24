use polars_core::chunked_array::cast::CastOptions;
use polars_core::frame::DataFrame;
use polars_core::prelude::DataType;
use polars_core::schema::{Schema, SchemaRef};
use polars_error::PolarsResult;
use polars_plan::dsl::CastColumnsPolicy;
use polars_utils::format_pl_smallstr;

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
                panic!("impl error: column '{name}' should exist in casting map")
            })
        };

        let mut casting_list = vec![];

        for (i, (name, incoming_dtype)) in incoming_schema_iter.enumerate() {
            let target_dtype = get_target_dtype(name);

            if policy.should_cast_column(name, target_dtype, incoming_dtype)? {
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

    /// `DataFrame` containing `{column_name}_min`, `{column_name}_max`.
    ///
    /// # Panics
    /// Panics if there is a cast for a column whose statistics are not present in `statistics_df`.
    /// This can happen e.g. if `incoming_schema` at initialization contained non-live predicate columns.
    pub fn apply_cast_to_statistics(
        &self,
        statistics_df: &mut DataFrame,
        // Schema of the file that the statistics are built from.
        incoming_schema: &Schema,
    ) -> PolarsResult<()> {
        let statistics_schema = statistics_df.schema().clone();
        statistics_df.clear_schema();

        let columns = unsafe { statistics_df.get_columns_mut() };

        for ColumnCast { index, dtype } in &self.casting_list {
            let column_name = incoming_schema.get_at_index(*index).unwrap().0;

            let i = statistics_schema
                .index_of(&format_pl_smallstr!("{column_name}_min"))
                .unwrap();

            // Note: Currently casting in scans do not allow any casts that
            // would raise errors, so we use `strict_cast` here.
            *columns.get_mut(i).unwrap() =
                columns[i].cast_with_options(dtype, CastOptions::Strict)?;

            let i = statistics_schema
                .index_of(&format_pl_smallstr!("{column_name}_max"))
                .unwrap();

            *columns.get_mut(i).unwrap() =
                columns[i].cast_with_options(dtype, CastOptions::Strict)?;
        }

        Ok(())
    }
}
