use polars_core::prelude::{Column, DataType, IDX_DTYPE, IdxSize};
use polars_core::scalar::Scalar;
use polars_error::PolarsResult;
use polars_plan::plans::predicates::null_count_dtype;
use polars_utils::pl_str::PlSmallStr;
use recursive::recursive;

use crate::nodes::io_sources::multi_scan::components::column_selector::transform::ColumnTransform;

pub mod builder;
pub mod transform;

/// This is a physical expression that is specialized for performing positional column selections,
/// column renaming, as well as type-casting.
///
/// Handles:
/// * Type-casting
///   * Only allowing a restricted set of non-nested casts for now
/// * Column re-ordering
/// * Struct field re-ordering
/// * Inserting missing struct fields
/// * Dropping extra struct fields (we just don't select them)
#[derive(Debug, Clone)]
pub enum ColumnSelector {
    // Note that we Box enum variants to keep `ColumnSelector` small (16 bytes).
    // This is an optimization that benefits cases where there are many `Position` selectors.

    // Leaf selectors
    /// Take the column at this position.
    Position(usize),
    /// Materialize a constant column.
    /// `(column_name, value)`
    Constant(Box<(PlSmallStr, Scalar)>),

    /// `(input_selector, _)`
    Transformed(Box<(ColumnSelector, ColumnTransform)>),
}

impl ColumnSelector {
    #[recursive]
    pub fn select_from_columns(
        &self,
        columns: &[Column],
        output_height: usize,
    ) -> PolarsResult<Column> {
        use ColumnSelector as S;

        Ok(match self {
            S::Position(i) => columns[*i].clone(),

            S::Constant(parts) => {
                let (name, scalar) = parts.as_ref();
                Column::new_scalar(name.clone(), scalar.clone(), output_height)
            },

            S::Transformed(transform) => {
                let input: Column = transform.0.select_from_columns(columns, output_height)?;
                transform.1.apply_transform(input)?
            },
        })
    }

    /// Replaces the leaf selector with the given `input` selector.
    pub fn replace_input(&mut self, input: ColumnSelector) {
        let mut current = self;

        while let Self::Transformed(v) = current {
            current = &mut v.as_mut().0;
        }

        *current = input;
    }

    /// Derives the parallel `null_count` selector. `None` means "fall back to unknown".
    #[recursive]
    pub fn null_count_transform(&self) -> Option<ColumnSelector> {
        use ColumnSelector as S;
        use ColumnTransform as TF;

        Some(match self {
            S::Position(i) => S::Position(*i),

            S::Constant(parts) => {
                let (name, scalar) = parts.as_ref();

                // A struct constant's leaves may mix defaulted and always-null, so unknown.
                let scalar = if scalar.is_null() || matches!(scalar.dtype(), DataType::Struct(_)) {
                    Scalar::null(null_count_dtype(scalar.dtype()))
                } else {
                    Scalar::new(IDX_DTYPE, (0 as IdxSize).into())
                };

                S::Constant(Box::new((name.clone(), scalar)))
            },

            S::Transformed(parts) => {
                let (input, transform) = parts.as_ref();
                let input = input.null_count_transform()?;

                match transform {
                    TF::Rename { name } => TF::Rename { name: name.clone() }.into_selector(input),

                    TF::StructFieldsMapping { field_selectors } => TF::StructFieldsMapping {
                        field_selectors: field_selectors
                            .iter()
                            .map(ColumnSelector::null_count_transform)
                            .collect::<Option<_>>()?,
                    }
                    .into_selector(input),

                    // Unsafe only for a null-upcast's struct target - no source shape to drop.
                    TF::Cast { dtype, .. } => {
                        if matches!(null_count_dtype(dtype), DataType::Struct(_)) {
                            return None;
                        }
                        input
                    },

                    // List/map/array elements have no per-element null count.
                    TF::ListValuesMapping { .. } => input,

                    #[cfg(feature = "dtype-array")]
                    TF::FixedSizeListValuesMapping { .. } => input,

                    #[cfg(feature = "dtype-map")]
                    TF::MapValuesMapping { .. } => input,
                }
            },
        })
    }
}
