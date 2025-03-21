use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_error::{PolarsResult, polars_bail};

/// TODO: Eventually move this enum to polars-plan
#[derive(Debug, Clone, Default)]
pub enum CastColumnsPolicy {
    /// Raise an error if the datatypes do not match
    #[default]
    ErrorOnMismatch,
}

#[derive(Debug)]
pub struct CastColumns {}

impl CastColumns {
    pub fn try_init_from_policy(
        policy: CastColumnsPolicy,
        target_schema: &SchemaRef,
        incoming_schema: &SchemaRef,
    ) -> PolarsResult<Option<Self>> {
        match policy {
            CastColumnsPolicy::ErrorOnMismatch => {
                for (name, dtype) in incoming_schema.iter() {
                    let target_dtype = target_schema
                        .get(name)
                        .expect("impl error: column should exist in casting map");

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
