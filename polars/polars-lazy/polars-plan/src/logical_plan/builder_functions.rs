use super::*;

// Has functions that create schema's for both the `LogicalPlan` and the `AlogicalPlan` builders.

pub fn explode_schema(schema: &mut Schema, columns: &[Arc<str>]) -> PolarsResult<()> {
    // columns to string
    columns.iter().try_for_each(|name| {
        if let DataType::List(inner) = schema.try_get(name)? {
            let inner = *inner.clone();
            schema.with_column(name.as_ref().into(), inner);
        };
        Ok(())
    })
}
