use std::sync::Arc;

use arrow::array::builder::ShareStrategy;
use polars_utils::IdxSize;

use crate::frame::DataFrame;
use crate::prelude::*;
use crate::schema::Schema;
use crate::series::builder::SeriesBuilder;

pub struct DataFrameBuilder {
    schema: Arc<Schema>,
    builders: Vec<SeriesBuilder>,
    height: usize,
}

impl DataFrameBuilder {
    pub fn new(schema: Arc<Schema>) -> Self {
        let builders = schema
            .iter_values()
            .map(|dt| SeriesBuilder::new(dt.clone()))
            .collect();
        Self {
            schema,
            builders,
            height: 0,
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        for builder in &mut self.builders {
            builder.reserve(additional);
        }
    }

    pub fn freeze(self) -> DataFrame {
        let columns = self
            .schema
            .iter_names()
            .zip(self.builders)
            .map(|(n, b)| {
                let s = b.freeze(n.clone());
                assert!(s.len() == self.height);
                Column::from(s)
            })
            .collect();

        // SAFETY: we checked the lengths and the names are unique because they
        // come from Schema.
        unsafe { DataFrame::new_no_checks(self.height, columns) }
    }

    pub fn freeze_reset(&mut self) -> DataFrame {
        let columns = self
            .schema
            .iter_names()
            .zip(&mut self.builders)
            .map(|(n, b)| {
                let s = b.freeze_reset(n.clone());
                assert!(s.len() == self.height);
                Column::from(s)
            })
            .collect();

        // SAFETY: we checked the lengths and the names are unique because they
        // come from Schema.
        let out = unsafe { DataFrame::new_no_checks(self.height, columns) };
        self.height = 0;
        out
    }

    pub fn len(&self) -> usize {
        self.height
    }

    pub fn is_empty(&self) -> bool {
        self.height == 0
    }

    /// Extends this builder with the contents of the given dataframe. May panic
    /// if other does not match the schema of this builder.
    pub fn extend(&mut self, other: &DataFrame, share: ShareStrategy) {
        self.subslice_extend(other, 0, other.height(), share);
        self.height += other.height();
    }

    /// Extends this builder with the contents of the given dataframe subslice.
    /// May panic if other does not match the schema of this builder.
    pub fn subslice_extend(
        &mut self,
        other: &DataFrame,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        let columns = other.get_columns();
        assert!(self.builders.len() == columns.len());
        for (builder, column) in self.builders.iter_mut().zip(columns) {
            match column {
                Column::Series(s) => {
                    builder.subslice_extend(s, start, length, share);
                },
                Column::Partitioned(p) => {
                    // @scalar-opt
                    builder.subslice_extend(p.as_materialized_series(), start, length, share);
                },
                Column::Scalar(sc) => {
                    let len = sc.len().saturating_sub(start).min(length);
                    let scalar_as_series = sc.scalar().clone().into_series(PlSmallStr::default());
                    builder.subslice_extend_repeated(&scalar_as_series, 0, 1, len, share);
                },
            }
        }

        self.height += length.min(other.height().saturating_sub(start));
    }

    /// Extends this builder with the contents of the given dataframe subslice, repeating it `repeats` times.
    /// May panic if other does not match the schema of this builder.
    pub fn subslice_extend_repeated(
        &mut self,
        other: &DataFrame,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        let columns = other.get_columns();
        assert!(self.builders.len() == columns.len());
        for (builder, column) in self.builders.iter_mut().zip(columns) {
            match column {
                Column::Series(s) => {
                    builder.subslice_extend_repeated(s, start, length, repeats, share);
                },
                Column::Partitioned(p) => {
                    // @scalar-opt
                    builder.subslice_extend_repeated(
                        p.as_materialized_series(),
                        start,
                        length,
                        repeats,
                        share,
                    );
                },
                Column::Scalar(sc) => {
                    let len = sc.len().saturating_sub(start).min(length);
                    let scalar_as_series = sc.scalar().clone().into_series(PlSmallStr::default());
                    builder.subslice_extend_repeated(&scalar_as_series, 0, 1, len * repeats, share);
                },
            }
        }

        self.height += length.min(other.height().saturating_sub(start)) * repeats;
    }

    /// Extends this builder with the contents of the given dataframe subslice.
    /// Each element is repeated repeats times. May panic if other does not
    /// match the schema of this builder.
    pub fn subslice_extend_each_repeated(
        &mut self,
        other: &DataFrame,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        let columns = other.get_columns();
        assert!(self.builders.len() == columns.len());
        for (builder, column) in self.builders.iter_mut().zip(columns) {
            match column {
                Column::Series(s) => {
                    builder.subslice_extend_each_repeated(s, start, length, repeats, share);
                },
                Column::Partitioned(p) => {
                    // @scalar-opt
                    builder.subslice_extend_each_repeated(
                        p.as_materialized_series(),
                        start,
                        length,
                        repeats,
                        share,
                    );
                },
                Column::Scalar(sc) => {
                    let len = sc.len().saturating_sub(start).min(length);
                    let scalar_as_series = sc.scalar().clone().into_series(PlSmallStr::default());
                    builder.subslice_extend_repeated(&scalar_as_series, 0, 1, len * repeats, share);
                },
            }
        }

        self.height += length.min(other.height().saturating_sub(start)) * repeats;
    }

    /// Extends this builder with the contents of the given dataframe at the given
    /// indices. That is, `other[idxs[i]]` is appended to this builder in order,
    /// for each i=0..idxs.len(). May panic if other does not match the schema
    /// of this builder, or if the other dataframe is not rechunked.
    ///
    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn gather_extend(
        &mut self,
        other: &DataFrame,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        let columns = other.get_columns();
        assert!(self.builders.len() == columns.len());
        for (builder, column) in self.builders.iter_mut().zip(columns) {
            match column {
                Column::Series(s) => {
                    builder.gather_extend(s, idxs, share);
                },
                Column::Partitioned(p) => {
                    // @scalar-opt
                    builder.gather_extend(p.as_materialized_series(), idxs, share);
                },
                Column::Scalar(sc) => {
                    let scalar_as_series = sc.scalar().clone().into_series(PlSmallStr::default());
                    builder.subslice_extend_repeated(&scalar_as_series, 0, 1, idxs.len(), share);
                },
            }
        }

        self.height += idxs.len();
    }

    /// Extends this builder with the contents of the given dataframe at the given
    /// indices. That is, `other[idxs[i]]` is appended to this builder in order,
    /// for each i=0..idxs.len(). Out-of-bounds indices extend with nulls.
    /// May panic if other does not match the schema of this builder, or if the
    /// other dataframe is not rechunked.
    pub fn opt_gather_extend(&mut self, other: &DataFrame, idxs: &[IdxSize], share: ShareStrategy) {
        let mut trans_idxs = Vec::new();
        let columns = other.get_columns();
        assert!(self.builders.len() == columns.len());
        for (builder, column) in self.builders.iter_mut().zip(columns) {
            match column {
                Column::Series(s) => {
                    builder.opt_gather_extend(s, idxs, share);
                },
                Column::Partitioned(p) => {
                    // @scalar-opt
                    builder.opt_gather_extend(p.as_materialized_series(), idxs, share);
                },
                Column::Scalar(sc) => {
                    let scalar_as_series = sc.scalar().clone().into_series(PlSmallStr::default());
                    // Reduce call overhead by transforming indices to 0/1 and dispatching to
                    // opt_gather_extend on the scalar as series.
                    for idx_chunk in idxs.chunks(4096) {
                        trans_idxs.clear();
                        trans_idxs.extend(
                            idx_chunk
                                .iter()
                                .map(|idx| ((*idx as usize) >= sc.len()) as IdxSize),
                        );
                        builder.opt_gather_extend(&scalar_as_series, &trans_idxs, share);
                    }
                },
            }
        }

        self.height += idxs.len();
    }
}
