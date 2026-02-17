use std::sync::Arc;

use polars_core::prelude::Column;
use polars_core::schema::Schema;
use polars_utils::marked_usize::MarkedUsize;

/// Applies a `with_columns()` operation with pre-computed indices.
#[derive(Clone)]
pub struct HStackColumns {
    gather_indices: Arc<[MarkedUsize]>,
}

impl HStackColumns {
    /// Note:
    /// * Dtypes of the schemas are unused.
    pub fn new(output_schema: &Schema, schema_left: &Schema, schema_right: &Schema) -> Self {
        assert!(schema_left.len() <= MarkedUsize::UNMARKED_MAX);
        assert!(schema_right.len() <= MarkedUsize::UNMARKED_MAX);

        let gather_indices: Arc<[MarkedUsize]> = output_schema
            .iter_names()
            .map(|name| {
                if let Some((idx, ..)) = schema_right.get_full(name) {
                    MarkedUsize::new(idx, true)
                } else {
                    MarkedUsize::new(schema_left.get_full(name).unwrap().0, false)
                }
            })
            .collect();

        Self { gather_indices }
    }

    #[expect(unused)]
    pub fn output_width(&self) -> usize {
        self.gather_indices.len()
    }

    /// Broadcasts unit-length columns from the RHS.
    pub fn hstack_columns_broadcast(
        &self,
        height: usize,
        cols_left: &[Column],
        cols_right: &[Column],
    ) -> Vec<Column> {
        self.gather_indices
            .iter()
            .copied()
            .map(|mi| {
                let i = mi.to_usize();

                if mi.marked() {
                    let c = &cols_right[i];

                    if c.len() != height {
                        assert_eq!(c.len(), 1);
                        c.new_from_index(0, height)
                    } else {
                        c.clone()
                    }
                } else {
                    cols_left[i].clone()
                }
            })
            .collect()
    }
}
