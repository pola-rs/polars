use polars_array::bitmap::combine_validities_and3;
use polars_core::prelude::*;
use polars_core::utils::align_chunks_ternary;
use polars_core::with_match_physical_numeric_polars_type;

/// Defines a fused elementwise kernel over three chunks, in whatever representation each is in.
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
