use super::*;

pub trait SchemaExtPl {
    // Answers if this schema matches the given schema.
    //
    // Allows (nested) Null types in this schema to match any type in the schema,
    // but not vice versa. In such a case Ok(true) is returned, because a cast
    // is necessary. If no cast is necessary Ok(false) is returned, and an
    // error is returned if the types are incompatible.
    fn matches_schema(&self, other: &Schema) -> PolarsResult<bool>;

    fn ensure_is_exact_match(&self, other: &Schema) -> PolarsResult<()>;
}

impl SchemaExtPl for Schema {
    fn matches_schema(&self, other: &Schema) -> PolarsResult<bool> {
        polars_ensure!(self.len() == other.len(), SchemaMismatch: "found different number of fields in schema's\n\nLeft schema: {} fields, right schema: {} fields.", self.len(), other.len());
        let mut cast = false;
        for (a, b) in self.iter_values().zip(other.iter_values()) {
            cast |= a.matches_schema_type(b)?;
        }
        Ok(cast)
    }

    fn ensure_is_exact_match(&self, other: &Schema) -> PolarsResult<()> {
        polars_ensure!(self.len() == other.len(), SchemaMismatch: "found different number of fields in schema's\n\nLeft schema: {} fields, right schema: {} fields.", self.len(), other.len());
        let mut different_columns = Vec::new();
        for (a, b) in self.iter_fields().zip(other.iter_fields()) {
            if a != b {
                different_columns.push((a, b));
            }
        }
        if !different_columns.is_empty() {
            use std::fmt::Write;
            let mut context = String::new();
            for (a, b) in different_columns {
                writeln!(
                    &mut context,
                    "('{}': {}) != ('{}': {})",
                    a.name(),
                    a.dtype(),
                    b.name(),
                    b.dtype()
                )
                .unwrap();
            }

            return Err(
                polars_err!(SchemaMismatch: "mismatching fields in schema's.\n\n{context}"),
            );
        }

        Ok(())
    }
}
