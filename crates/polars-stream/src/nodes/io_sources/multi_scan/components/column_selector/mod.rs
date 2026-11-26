use polars_core::prelude::Column;
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
#[derive(Debug, Clone, PartialEq)]
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

    /// Selector for the `null_count` statistics. Underivable fields get an unknown null count.
    #[recursive]
    pub fn null_count_transform(&self) -> ColumnSelector {
        use ColumnSelector as S;
        use ColumnTransform as TF;

        match self {
            S::Position(i) => S::Position(*i),

            // Unknown: file leaf null counts include rows with a null ancestor.
            S::Constant(parts) => {
                let (name, scalar) = parts.as_ref();
                S::Constant(Box::new((
                    name.clone(),
                    Scalar::null(null_count_dtype(scalar.dtype())),
                )))
            },

            S::Transformed(parts) => {
                let (input, transform) = parts.as_ref();
                let input = input.null_count_transform();

                match transform {
                    TF::Rename { name } => TF::Rename { name: name.clone() }.into_selector(input),

                    TF::StructFieldsMapping { field_selectors } => TF::StructFieldsMapping {
                        field_selectors: field_selectors
                            .iter()
                            .map(ColumnSelector::null_count_transform)
                            .collect(),
                    }
                    .into_selector(input),

                    // `FullNull` keeps the input name, which `StructFieldsMapping` needs.
                    TF::Cast {
                        dtype,
                        preserves_null_count,
                        ..
                    } => {
                        if *preserves_null_count && !dtype.is_struct() {
                            input
                        } else {
                            TF::FullNull {
                                dtype: null_count_dtype(dtype),
                            }
                            .into_selector(input)
                        }
                    },

                    // List/map/array elements have no per-element null count.
                    TF::ListValuesMapping { .. } => input,

                    #[cfg(feature = "dtype-array")]
                    TF::FixedSizeListValuesMapping { .. } => input,

                    #[cfg(feature = "dtype-map")]
                    TF::MapValuesMapping { .. } => input,

                    #[cfg(feature = "dtype-map")]
                    TF::MapKeysMapping { .. } => input,

                    TF::FullNull { dtype } => TF::FullNull {
                        dtype: null_count_dtype(dtype),
                    }
                    .into_selector(input),
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use polars_core::chunked_array::cast::CastOptions;
    use polars_core::datatypes::{AnyValue, DataType, Field};

    use super::*;

    fn constant(name: &str, scalar: Scalar) -> ColumnSelector {
        ColumnSelector::Constant(Box::new((name.into(), scalar)))
    }

    fn unknown(name: &str, dtype: &DataType) -> ColumnSelector {
        constant(name, Scalar::null(null_count_dtype(dtype)))
    }

    fn full_null(input: ColumnSelector, dtype: &DataType) -> ColumnSelector {
        ColumnTransform::FullNull {
            dtype: null_count_dtype(dtype),
        }
        .into_selector(input)
    }

    fn cast(dtype: DataType, preserves_null_count: bool) -> ColumnTransform {
        ColumnTransform::Cast {
            dtype,
            options: CastOptions::NonStrict,
            preserves_null_count,
        }
    }

    fn struct_fields(fields: impl IntoIterator<Item = ColumnSelector>) -> ColumnSelector {
        ColumnTransform::StructFieldsMapping {
            field_selectors: fields.into_iter().collect(),
        }
        .into_selector(ColumnSelector::Position(0))
    }

    fn struct_dtype() -> DataType {
        DataType::Struct(vec![Field::new("y".into(), DataType::Int64)])
    }

    #[test]
    fn test_null_count_transform_passthrough() {
        let s = ColumnSelector::Position(3);
        assert_eq!(s.null_count_transform(), s);

        let s =
            ColumnTransform::Rename { name: "b".into() }.into_selector(ColumnSelector::Position(0));
        assert_eq!(s.null_count_transform(), s);

        let s = ColumnTransform::ListValuesMapping {
            values_selector: ColumnSelector::Position(1),
        }
        .into_selector(ColumnSelector::Position(0));
        assert_eq!(s.null_count_transform(), ColumnSelector::Position(0));
    }

    #[test]
    fn test_null_count_transform_constant_is_unknown() {
        for scalar in [
            Scalar::new(DataType::Int64, AnyValue::Int64(5)),
            Scalar::null(DataType::Int64),
        ] {
            let s = struct_fields([constant("b", scalar)]);
            let expect = struct_fields([unknown("b", &DataType::Int64)]);
            assert_eq!(s.null_count_transform(), expect);
        }
    }

    #[test]
    fn test_null_count_transform_cast() {
        let s = cast(DataType::Int64, true).into_selector(ColumnSelector::Position(0));
        assert_eq!(s.null_count_transform(), ColumnSelector::Position(0));

        let s = cast(DataType::Int8, false).into_selector(ColumnSelector::Position(0));
        assert_eq!(
            s.null_count_transform(),
            full_null(ColumnSelector::Position(0), &DataType::Int8)
        );

        // Null-upcast to a struct.
        let s = cast(struct_dtype(), true).into_selector(ColumnSelector::Position(0));
        assert_eq!(
            s.null_count_transform(),
            full_null(ColumnSelector::Position(0), &struct_dtype())
        );
    }

    #[test]
    fn test_null_count_transform_unknown_field_keeps_siblings() {
        let s = struct_fields([
            ColumnSelector::Position(0),
            cast(struct_dtype(), true).into_selector(ColumnSelector::Position(1)),
        ]);
        let expect = struct_fields([
            ColumnSelector::Position(0),
            full_null(ColumnSelector::Position(1), &struct_dtype()),
        ]);
        assert_eq!(s.null_count_transform(), expect);
    }
}
