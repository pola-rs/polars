use polars_array::bitmap::combine_validities_and3;
use polars_core::prelude::*;
use polars_core::utils::align_chunks_ternary;
use polars_core::with_match_physical_numeric_polars_type;

/// Defines a fused elementwise kernel over three chunks, one that reads each of them in whatever
/// representation it is in rather than having it written out first.
///
/// Three repeated values fuse once into a repeated answer; three values buffers that already hold
/// one slot per element are read as the slices they are, which is what vectorizes. Only a mix of
/// the two reads an element at a time, and even that allocates nothing beyond the answer.
macro_rules! fused_kernel {
    ($(#[$meta:meta])* $name:ident, |$a:ident, $b:ident, $c:ident| $fuse:expr) => {
        $(#[$meta])*
        fn $name<T: NumericNative>(
            a: &PlPrimitiveArray<T>,
            b: &PlPrimitiveArray<T>,
            c: &PlPrimitiveArray<T>,
        ) -> PlPrimitiveArray<T> {
            assert_eq!(a.len(), b.len());
            assert_eq!(b.len(), c.len());
            let length = a.len();
            let validity = combine_validities_and3(a.validity(), b.validity(), c.validity());

            if let (Some($a), Some($b), Some($c)) =
                (a.scalar_values(), b.scalar_values(), c.scalar_values())
            {
                return PlPrimitiveArray::new_scalar($fuse, length).with_validity(validity);
            }

            let out: Vec<T> = match (a.flat_values(), b.flat_values(), c.flat_values()) {
                (Some(a), Some(b), Some(c)) => a
                    .iter()
                    .zip(b.iter())
                    .zip(c.iter())
                    .map(|((&$a, &$b), &$c)| $fuse)
                    .collect(),
                _ => a
                    .broadcast_values_iter(length)
                    .zip(b.broadcast_values_iter(length))
                    .zip(c.broadcast_values_iter(length))
                    .map(|(($a, $b), $c)| $fuse)
                    .collect(),
            };

            PlPrimitiveArray::from_vec(out).with_validity(validity)
        }
    };
}

fused_kernel!(
    /// `(a * b) + c`, element for element.
    fma_arr,
    |a, b, c| a * b + c
);

fused_kernel!(
    /// `a - (b * c)`, element for element.
    fsm_arr,
    |a, b, c| a - (b * c)
);

fused_kernel!(
    /// `(a * b) - c`, element for element.
    fms_arr,
    |a, b, c| (a * b) - c
);

fn fma_ca<T: PolarsNumericType>(
    a: &ChunkedArray<T>,
    b: &ChunkedArray<T>,
    c: &ChunkedArray<T>,
) -> ChunkedArray<T> {
    let (a, b, c) = align_chunks_ternary(a, b, c);
    let chunks = a
        .downcast_iter()
        .zip(b.downcast_iter())
        .zip(c.downcast_iter())
        .map(|((a, b), c)| fma_arr(a, b, c));
    ChunkedArray::from_chunk_iter(a.name().clone(), chunks)
}

pub fn fma_columns(a: &Column, b: &Column, c: &Column) -> Column {
    if a.len() == b.len() && a.len() == c.len() {
        with_match_physical_numeric_polars_type!(a.dtype(), |$T| {
            let a: &ChunkedArray<$T> = a.as_materialized_series().as_ref().as_ref().as_ref();
            let b: &ChunkedArray<$T> = b.as_materialized_series().as_ref().as_ref().as_ref();
            let c: &ChunkedArray<$T> = c.as_materialized_series().as_ref().as_ref().as_ref();

            fma_ca(a, b, c).into_column()
        })
    } else {
        (&(a * b).unwrap() + c).unwrap()
    }
}

fn fsm_ca<T: PolarsNumericType>(
    a: &ChunkedArray<T>,
    b: &ChunkedArray<T>,
    c: &ChunkedArray<T>,
) -> ChunkedArray<T> {
    let (a, b, c) = align_chunks_ternary(a, b, c);
    let chunks = a
        .downcast_iter()
        .zip(b.downcast_iter())
        .zip(c.downcast_iter())
        .map(|((a, b), c)| fsm_arr(a, b, c));
    ChunkedArray::from_chunk_iter(a.name().clone(), chunks)
}

pub fn fsm_columns(a: &Column, b: &Column, c: &Column) -> Column {
    if a.len() == b.len() && a.len() == c.len() {
        with_match_physical_numeric_polars_type!(a.dtype(), |$T| {
            let a: &ChunkedArray<$T> = a.as_materialized_series().as_ref().as_ref().as_ref();
            let b: &ChunkedArray<$T> = b.as_materialized_series().as_ref().as_ref().as_ref();
            let c: &ChunkedArray<$T> = c.as_materialized_series().as_ref().as_ref().as_ref();

            fsm_ca(a, b, c).into_column()
        })
    } else {
        (a - &(b * c).unwrap()).unwrap()
    }
}

fn fms_ca<T: PolarsNumericType>(
    a: &ChunkedArray<T>,
    b: &ChunkedArray<T>,
    c: &ChunkedArray<T>,
) -> ChunkedArray<T> {
    let (a, b, c) = align_chunks_ternary(a, b, c);
    let chunks = a
        .downcast_iter()
        .zip(b.downcast_iter())
        .zip(c.downcast_iter())
        .map(|((a, b), c)| fms_arr(a, b, c));
    ChunkedArray::from_chunk_iter(a.name().clone(), chunks)
}

pub fn fms_columns(a: &Column, b: &Column, c: &Column) -> Column {
    if a.len() == b.len() && a.len() == c.len() {
        with_match_physical_numeric_polars_type!(a.dtype(), |$T| {
            let a: &ChunkedArray<$T> = a.as_materialized_series().as_ref().as_ref().as_ref();
            let b: &ChunkedArray<$T> = b.as_materialized_series().as_ref().as_ref().as_ref();
            let c: &ChunkedArray<$T> = c.as_materialized_series().as_ref().as_ref().as_ref();

            fms_ca(a, b, c).into_column()
        })
    } else {
        (&(a * b).unwrap() - c).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use polars_array::PlBitmap;

    use super::*;

    /// The three sides read the same whichever representation each is stored in, and a chunk that
    /// repeats a value stays repeated through the kernel rather than being written out.
    #[test]
    fn every_mix_of_representations_fuses_alike() {
        const LENGTH: usize = 40;

        let scalar = |value: i32| PlPrimitiveArray::new_scalar(value, LENGTH);
        let flat = |value: i32| PlPrimitiveArray::from_vec(vec![value; LENGTH]);

        for a_scalar in [false, true] {
            for b_scalar in [false, true] {
                for c_scalar in [false, true] {
                    let a = if a_scalar { scalar(3) } else { flat(3) };
                    let b = if b_scalar { scalar(5) } else { flat(5) };
                    let c = if c_scalar { scalar(7) } else { flat(7) };

                    let fma = fma_arr(&a, &b, &c);
                    assert_eq!(fma.len(), LENGTH);
                    assert!(fma.iter().all(|value| value == Some(3 * 5 + 7)));
                    assert_eq!(
                        fma.is_scalar(),
                        a_scalar && b_scalar && c_scalar,
                        "three repeated values fuse to a repeated answer",
                    );

                    assert!(
                        fsm_arr(&a, &b, &c)
                            .iter()
                            .all(|value| value == Some(3 - 5 * 7))
                    );
                    assert!(
                        fms_arr(&a, &b, &c)
                            .iter()
                            .all(|value| value == Some(3 * 5 - 7))
                    );
                }
            }
        }
    }

    /// A null on any side leaves the fused element null, whichever representation the masks are in.
    #[test]
    fn a_null_on_any_side_nulls_the_element() {
        let a = PlPrimitiveArray::from_iter([Some(1i32), None, Some(3), Some(4)]);
        let b = PlPrimitiveArray::new_scalar(2i32, 4);
        let c = PlPrimitiveArray::from_iter([Some(10i32), Some(20), None, Some(40)]);

        let fused = fma_arr(&a, &b, &c);
        assert_eq!(
            fused.iter().collect::<Vec<_>>(),
            [Some(12), None, None, Some(48)]
        );

        // A repeated unset bit nulls every element without the mask being written out.
        let all_null = PlPrimitiveArray::new_scalar(2i32, 4)
            .with_validity(Some(PlBitmap::new_scalar(false, 4)));
        let fused = fma_arr(&a, &all_null, &c);
        assert_eq!(fused.null_count(), 4);
        assert!(
            fused
                .validity()
                .is_some_and(|validity| validity.is_scalar())
        );
    }
}
