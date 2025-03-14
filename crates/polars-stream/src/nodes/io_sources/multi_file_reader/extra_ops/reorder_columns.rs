use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;

#[derive(Debug)]
pub enum ReorderColumns {
    /// Column indices to take.
    Gather {
        indices: Vec<usize>,
        /// This is only stored for a debug assertion.
        incoming_schema: SchemaRef,
    },
    /// No reordering needed
    Passthrough,
}

impl ReorderColumns {
    /// # Panics
    /// Panics if schemas do not have the same the same set of column names.
    pub fn initialize(target_schema: &SchemaRef, incoming_schema: &SchemaRef) -> Self {
        assert_eq!(target_schema.len(), incoming_schema.len());

        if let Some(first_mismatch_idx) = target_schema
            .iter_names()
            .zip(incoming_schema.iter_names())
            .position(|(l, r)| l != r)
        {
            let mut indices = Vec::with_capacity(incoming_schema.len());

            indices.extend(0..first_mismatch_idx);
            indices.extend(
                target_schema
                    .iter_names()
                    .skip(first_mismatch_idx)
                    .map(|name| incoming_schema.index_of(name).unwrap()),
            );

            Self::Gather {
                indices,
                incoming_schema: incoming_schema.clone(),
            }
        } else {
            Self::Passthrough
        }
    }

    /// Note: The column order of the incoming DataFrame must exactly match that of the schema used
    /// to initialize this `ReorderColumns`.
    ///
    /// # Panics
    /// Panics if self is uninitialized.
    pub fn reorder_columns(&self, df: &mut DataFrame) {
        use ReorderColumns::*;

        match self {
            Passthrough => {},

            Gather {
                indices,
                incoming_schema,
            } => {
                debug_assert_eq!(df.schema(), incoming_schema);

                let incoming_cols = df.get_columns();
                let out = indices.iter().map(|i| incoming_cols[*i].clone()).collect();
                *df = out;
            },
        }
    }
}
