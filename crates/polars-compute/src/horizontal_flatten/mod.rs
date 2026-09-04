//! The kernel behind `concat_arr`, which lays a row of arrays out end to end.
//!
//! Each of the `arrays` contributes a fixed number of values — its *width* — to every output row,
//! and the output holds those runs one after another, `output_height` rows of them. An array that
//! holds a single row's worth of values stands for that row repeated: it is broadcast over the
//! output rather than being written out one row per output row on the way in.

use polars_array::builder::{ShareStrategy, builder_like};
use polars_array::{PlArray, PlArrayBuilder};

/// Lays the `arrays` out end to end, `output_height` rows of `widths[i]` values from `arrays[i]`.
///
/// Every array either holds `widths[i] * output_height` values — one row's worth per output row —
/// or `widths[i]` of them, the one row it stands for at every output row.
///
/// # Panics
/// Panics if `arrays` is empty, if `arrays` and `widths` are of different lengths, if the arrays
/// are not all of the same type, or if any array holds neither of the two admissible number of
/// values.
pub fn horizontal_flatten(
    arrays: &[Box<dyn PlArray>],
    widths: &[usize],
    output_height: usize,
) -> Box<dyn PlArray> {
    assert!(!arrays.is_empty(), "there is no array to take a type from");
    assert_eq!(
        arrays.len(),
        widths.len(),
        "every array contributes a width to the output row",
    );

    // Whether each array stands for one row repeated over the output, rather than holding a row of
    // its own per output row. The two coincide for an `output_height` of one, where either
    // reading is the same one.
    let repeats: Vec<bool> = arrays
        .iter()
        .zip(widths)
        .map(|(array, &width)| is_broadcast(&**array, width, output_height))
        .collect();

    // One array is the output, save for how many times its row is repeated: there is no second
    // array to interleave it with, so nothing has to be copied out of it a row at a time.
    if let ([array], [width], [repeats]) = (arrays, widths, repeats.as_slice()) {
        if !repeats {
            return array.to_boxed();
        }

        let mut builder = builder_like(&**array);
        builder.subslice_extend_repeated(&**array, 0, *width, output_height, ShareStrategy::Always);
        return builder.freeze();
    }

    let row_width: usize = widths.iter().sum();
    let mut builder = builder_like(&*arrays[0]);
    builder.reserve(row_width.saturating_mul(output_height));

    for row in 0..output_height {
        for ((array, &width), &repeats) in arrays.iter().zip(widths).zip(&repeats) {
            let start = if repeats { 0 } else { row * width };
            builder.subslice_extend(&**array, start, width, ShareStrategy::Always);
        }
    }

    builder.freeze()
}

/// Whether `array` holds the one row of `width` values it stands for at every output row, rather
/// than a row of its own per output row.
///
/// # Panics
/// Panics unless it holds one or the other.
fn is_broadcast(array: &dyn PlArray, width: usize, output_height: usize) -> bool {
    let flat = width.checked_mul(output_height);
    if flat == Some(array.len()) {
        // A single output row is the same array either way; reading it as the flat one saves the
        // repetition below.
        return false;
    }

    assert_eq!(
        array.len(),
        width,
        "an array of {} values is neither {width} values wide nor {flat:?} values long",
        array.len(),
    );
    true
}

#[cfg(test)]
mod tests {
    use polars_array::{PlPrimitiveArray, PlStructArray, PlUtf8ViewArray};

    use super::*;

    /// The elements of a primitive chunk the kernel handed back.
    fn elements_of<T: arrow::types::NativeType>(array: &dyn PlArray) -> Vec<Option<T>> {
        array
            .as_any()
            .downcast_ref::<PlPrimitiveArray<T>>()
            .expect("the kernel hands back a chunk of the type it was given")
            .iter()
            .collect()
    }

    /// The rows are laid out one after another, each of them a run per array.
    #[test]
    fn rows_are_laid_out_end_to_end() {
        let arrays: Vec<Box<dyn PlArray>> = vec![
            // Two values wide, three rows of them.
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5, 6])),
            // One value wide, three rows of it.
            Box::new(PlPrimitiveArray::from_iter([Some(7i32), None, Some(9)])),
        ];

        let out = horizontal_flatten(&arrays, &[2, 1], 3);

        assert_eq!(
            elements_of::<i32>(&*out),
            [
                Some(1),
                Some(2),
                Some(7),
                Some(3),
                Some(4),
                None,
                Some(5),
                Some(6),
                Some(9)
            ],
        );
    }

    /// An array of a single row's worth of values stands for that row at every output row.
    #[test]
    fn a_single_row_is_repeated() {
        let arrays: Vec<Box<dyn PlArray>> = vec![
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4])),
            // Two values wide, and only the one row of them.
            Box::new(PlPrimitiveArray::from_vec(vec![-1i32, -2])),
        ];

        let out = horizontal_flatten(&arrays, &[2, 2], 2);

        assert_eq!(
            elements_of::<i32>(&*out),
            [
                Some(1),
                Some(2),
                Some(-1),
                Some(-2),
                Some(3),
                Some(4),
                Some(-1),
                Some(-2)
            ],
        );
    }

    /// A chunk that repeats one value is one such row, and is read as one: the kernel neither
    /// writes it out on the way in nor loses the representation where the whole output is that
    /// chunk.
    #[test]
    fn a_scalar_chunk_is_read_as_the_row_it_repeats() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 2);
        let flat = PlPrimitiveArray::from_vec(vec![7i32; 2]);

        for array in [scalar.clone(), flat] {
            let arrays: Vec<Box<dyn PlArray>> = vec![
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4])),
                Box::new(array),
            ];

            let out = horizontal_flatten(&arrays, &[2, 2], 2);
            assert_eq!(
                elements_of::<i32>(&*out),
                [
                    Some(1),
                    Some(2),
                    Some(7),
                    Some(7),
                    Some(3),
                    Some(4),
                    Some(7),
                    Some(7)
                ],
            );
        }

        // The one array there is comes back as it was, still repeating its one value.
        let arrays: Vec<Box<dyn PlArray>> = vec![Box::new(scalar)];
        let out = horizontal_flatten(&arrays, &[2], 1);
        assert!(out.is_scalar(), "{out:?} was written out");
    }

    /// The kernel is generic over the array: a nested one is laid out through its own builder,
    /// which carries the fields along.
    #[test]
    fn nested_arrays_keep_their_fields() {
        let row = |values: [i32; 3], names: [&str; 3]| -> Box<dyn PlArray> {
            Box::new(PlStructArray::new(
                vec![
                    Box::new(PlPrimitiveArray::from_vec(values.to_vec())),
                    Box::new(PlUtf8ViewArray::from_iter(names.map(Some))),
                ],
                3,
                None,
            ))
        };

        let arrays = vec![
            row([1, 2, 3], ["a", "b", "c"]),
            row([4, 5, 6], ["d", "e", "f"]),
        ];
        let out = horizontal_flatten(&arrays, &[1, 1], 3);

        let out = out
            .as_any()
            .downcast_ref::<PlStructArray>()
            .expect("a struct array is flattened into one");
        assert_eq!(out.len(), 6);
        assert_eq!(
            elements_of::<i32>(&*out.fields()[0]),
            [1, 4, 2, 5, 3, 6].map(Some)
        );
    }

    /// No output row is no output at all, however wide the arrays are.
    #[test]
    fn no_rows_yield_no_values() {
        let arrays: Vec<Box<dyn PlArray>> = vec![Box::new(PlPrimitiveArray::<i32>::new_empty())];
        assert_eq!(horizontal_flatten(&arrays, &[2], 0).len(), 0);
    }
}
