use polars_core::frame::column::ScalarColumn;
use polars_core::prelude::AnyValue;
use polars_core::scalar::Scalar;
use polars_core::schema::Schema;
use polars_error::{PolarsResult, polars_bail};

/// TODO: Eventually move this enum to polars-plan
#[derive(Debug, Clone)]
pub enum MissingColumnsPolicy {
    Raise,
    Insert,
}

impl MissingColumnsPolicy {
    // Either error or extend `extra_columns` with the missing ones.
    pub fn initialize_policy(
        &self,
        target_schema: &Schema,
        incoming_schema: &Schema,
        extra_cols: &mut Vec<ScalarColumn>,
    ) -> PolarsResult<()> {
        use MissingColumnsPolicy::*;
        match self {
            Raise => {
                if let Some(col) = target_schema
                    .iter_names()
                    .find(|name| !incoming_schema.contains(name))
                {
                    polars_bail!(
                        ColumnNotFound:
                        "did not find column {}, consider enabling `allow_missing_columns`",
                        col,
                    )
                }

                Ok(())
            },

            Insert => {
                extra_cols.extend(
                    target_schema
                        .iter()
                        .filter(|(name, _)| !incoming_schema.contains(name))
                        .map(|(name, dtype)| {
                            ScalarColumn::new(
                                name.clone(),
                                Scalar::new(dtype.clone(), AnyValue::Null),
                                1,
                            )
                        }),
                );
                Ok(())
            },
        }
    }
}
