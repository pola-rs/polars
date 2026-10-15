use polars_core::prelude::{Column, DataType};
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

    /// Derives the parallel `null_count` selector: never bails out to "fall back to unknown"
    /// for the whole column any more. A branch that can't derive a real null_count for one
    /// field (e.g. a null-upcast to a struct, or an unsafe cast - see the `Cast` arm below)
    /// returns a `ForceUnknown` placeholder *for just that field*, so sibling fields in the
    /// same struct keep their own real, narrowable null_count.
    #[recursive]
    pub fn null_count_transform(&self) -> ColumnSelector {
        use ColumnSelector as S;
        use ColumnTransform as TF;

        match self {
            S::Position(i) => S::Position(*i),

            // A `Constant` stands in for a field missing from the file, backfilled with one
            // scalar for every row - it has no visibility into whether some *ancestor* struct
            // is null for a given row, unlike a real file leaf (whose own null_count statistic
            // already counts those rows too, since Parquet's definition levels don't distinguish
            // "this leaf is null" from "an ancestor group is null"). A `Constant` only ever
            // appears nested inside a `StructFieldsMapping`'s fields in practice (a genuinely
            // missing *top-level* column is resolved before this selector tree is ever built,
            // not through `ArrowFieldProjection::Mapped`), so treating it as always-unknown here
            // costs nothing: there's no case where "known zero" would actually apply.
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

                    // A null-upcast's struct target has no source shape to drop from, no
                    // matter how safe the cast itself is; mark just this field unknown so a
                    // sibling field in the same struct (if any) keeps its own real null_count.
                    // `ForceUnknown` (not `Constant`) keeps `input`'s name resolvable at
                    // execution time, which `StructFieldsMapping` needs: it names each
                    // assembled field from its own `Column`, not from any outside schema.
                    TF::Cast { dtype, .. } if matches!(dtype, DataType::Struct(_)) => {
                        TF::ForceUnknown {
                            dtype: null_count_dtype(dtype),
                        }
                        .into_selector(input)
                    },

                    // `preserves_null_count` is set by whoever built this transform (see its
                    // doc-comment in `transform.rs`) - not re-derived from `dtype`/`options`
                    // here, since a target dtype alone can't tell a safe `NonStrict` cast from
                    // an unsafe one without knowing what it's cast *from*.
                    TF::Cast {
                        dtype,
                        preserves_null_count,
                        ..
                    } => {
                        if *preserves_null_count {
                            input
                        } else {
                            TF::ForceUnknown {
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

                    // Synthesized only by this function itself (see the two `Cast` arms
                    // above); never present in a selector tree this function is called on,
                    // but idempotent rather than panicking if that ever stops being true.
                    TF::ForceUnknown { dtype } => TF::ForceUnknown {
                        dtype: dtype.clone(),
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
    use polars_core::datatypes::{AnyValue, Field};

    use super::*;

    fn int_scalar(v: i64) -> Scalar {
        Scalar::new(DataType::Int64, AnyValue::Int64(v))
    }

    fn constant(name: &str, scalar: Scalar) -> ColumnSelector {
        ColumnSelector::Constant(Box::new((name.into(), scalar)))
    }

    fn unknown(name: &str, dtype: &DataType) -> ColumnSelector {
        constant(name, Scalar::null(null_count_dtype(dtype)))
    }

    fn force_unknown(input: ColumnSelector, dtype: &DataType) -> ColumnSelector {
        ColumnTransform::ForceUnknown {
            dtype: null_count_dtype(dtype),
        }
        .into_selector(input)
    }

    fn cast(dtype: DataType, options: CastOptions, preserves_null_count: bool) -> ColumnTransform {
        ColumnTransform::Cast {
            dtype,
            options,
            preserves_null_count,
        }
    }

    #[test]
    fn non_null_constant_is_always_unknown() {
        // A Constant only ever appears nested inside a StructFieldsMapping's fields in
        // practice (see the doc-comment), so there's no case where "known zero" would apply:
        // its own value is never null, but the *real* leaf it stands in for can still be null
        // whenever some ancestor struct is null for that row - information a bare scalar
        // can't see.
        let s = constant("x", int_scalar(5));
        assert_eq!(s.null_count_transform(), unknown("x", &DataType::Int64));
    }

    #[test]
    fn null_constant_is_unknown() {
        let s = constant("x", Scalar::null(DataType::Int64));
        assert_eq!(s.null_count_transform(), unknown("x", &DataType::Int64));
    }

    #[test]
    fn position_passes_through_unchanged() {
        let s = ColumnSelector::Position(3);
        assert_eq!(s.null_count_transform(), ColumnSelector::Position(3));
    }

    #[test]
    fn rename_carries_through() {
        let s = ColumnTransform::Rename {
            name: "renamed".into(),
        }
        .into_selector(ColumnSelector::Position(0));
        assert_eq!(s.null_count_transform(), s);
    }

    #[test]
    fn list_values_mapping_passes_through() {
        let s = ColumnTransform::ListValuesMapping {
            values_selector: ColumnSelector::Position(1),
        }
        .into_selector(ColumnSelector::Position(0));
        assert_eq!(s.null_count_transform(), ColumnSelector::Position(0));
    }

    #[test]
    fn cast_preserves_null_count_true_passes_through() {
        // `preserves_null_count` is read directly, regardless of which `CastOptions` is paired
        // with it - the builder is the one that knows why a given cast is safe.
        for options in [
            CastOptions::Strict,
            CastOptions::Overflowing,
            CastOptions::NonStrict,
        ] {
            let s = cast(DataType::Int64, options, true).into_selector(ColumnSelector::Position(0));
            assert_eq!(s.null_count_transform(), ColumnSelector::Position(0));
        }
    }

    #[test]
    fn cast_preserves_null_count_false_is_unknown_not_passed_through() {
        // Conservative default: whether a cast marked `preserves_null_count: false` can
        // actually null a value depends on the actual data, which isn't knowable here.
        let s = cast(DataType::Int8, CastOptions::NonStrict, false)
            .into_selector(ColumnSelector::Position(0));
        assert_eq!(
            s.null_count_transform(),
            force_unknown(ColumnSelector::Position(0), &DataType::Int8)
        );
    }

    #[test]
    fn struct_field_non_null_constant_is_unknown_not_zero() {
        // A Constant used as one field of a StructFieldsMapping: it never claims known-zero,
        // matching `non_null_constant_is_always_unknown` above (there's no "top level" case
        // for a field nested this way to be confused with).
        let field = constant("b", int_scalar(5));
        let s = ColumnTransform::StructFieldsMapping {
            field_selectors: Box::new([field]),
        }
        .into_selector(ColumnSelector::Position(0));

        let expect = ColumnTransform::StructFieldsMapping {
            field_selectors: Box::new([unknown("b", &DataType::Int64)]),
        }
        .into_selector(ColumnSelector::Position(0));

        assert_eq!(s.null_count_transform(), expect);
    }

    #[test]
    fn cast_to_struct_null_upcast_is_isolated_unknown_even_if_preserves_null_count() {
        // A null-upcast to a struct target is always unknown - no source shape to narrow
        // from - *regardless* of `preserves_null_count` (true here, matching what the real
        // null-upcast builder arm actually sets, since the source is already 100% null).
        let struct_dtype = DataType::Struct(vec![Field::new("y".into(), DataType::Int64)]);
        let s = cast(struct_dtype.clone(), CastOptions::NonStrict, true)
            .into_selector(ColumnSelector::Position(0));

        assert_eq!(
            s.null_count_transform(),
            force_unknown(ColumnSelector::Position(0), &struct_dtype)
        );
    }

    #[test]
    fn struct_insert_unsafe_cast_does_not_poison_sibling_field() {
        // Two fields: one real (Position), one unsafe (a null-upcast to a struct target).
        // The unsafe field's unknown-ness must stay local to it - the sibling keeps its own
        // real, narrowable null_count rather than the whole struct falling back to unknown.
        let struct_dtype = DataType::Struct(vec![Field::new("y".into(), DataType::Int64)]);
        let safe_field = ColumnSelector::Position(0);
        let unsafe_field = cast(struct_dtype.clone(), CastOptions::NonStrict, true)
            .into_selector(ColumnSelector::Position(1));

        let s = ColumnTransform::StructFieldsMapping {
            field_selectors: Box::new([safe_field, unsafe_field]),
        }
        .into_selector(ColumnSelector::Position(2));

        let expect = ColumnTransform::StructFieldsMapping {
            field_selectors: Box::new([
                ColumnSelector::Position(0),
                force_unknown(ColumnSelector::Position(1), &struct_dtype),
            ]),
        }
        .into_selector(ColumnSelector::Position(2));

        assert_eq!(s.null_count_transform(), expect);
    }

    #[test]
    fn force_unknown_fed_back_in_is_idempotent_not_a_panic() {
        // `ForceUnknown` is only ever synthesized as this function's own output, never
        // present in a selector tree it's called on - but a defensive re-entry must stay
        // inert rather than panic.
        let s = ColumnTransform::ForceUnknown {
            dtype: DataType::Int64,
        }
        .into_selector(ColumnSelector::Position(0));
        assert_eq!(s.null_count_transform(), s);
    }
}
