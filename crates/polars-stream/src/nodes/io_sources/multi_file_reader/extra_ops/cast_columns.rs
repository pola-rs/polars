use polars_core::frame::DataFrame;
use polars_core::prelude::DataType;
use polars_core::schema::SchemaRef;
use polars_error::{PolarsResult, polars_bail};
use polars_plan::dsl::CastColumnsPolicy;

#[derive(Debug)]
pub struct CastColumns {}

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
        match policy {
            CastColumnsPolicy::ErrorOnMismatch => {
                for (name, dtype) in incoming_schema_iter {
                    let Some(target_dtype) = target_schema.get(name) else {
                        panic!("impl error: column '{}' should exist in casting map", name)
                    };

                    if dtype != target_dtype {
                        polars_bail!(
                            SchemaMismatch:
                            "data type mismatch for column {}: expected: {}, found: {}",
                            name, target_dtype, dtype
                        )
                    }
                }

                Ok(None)
            },
        }
    }

    pub fn apply_cast(&self, _df: &mut DataFrame) -> PolarsResult<()> {
        unimplemented!()
        // df.clear_schema(); // remember to do this if cast was applied
    }
}
