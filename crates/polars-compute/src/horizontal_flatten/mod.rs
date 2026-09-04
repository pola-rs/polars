//! The kernel behind `concat_arr`, which lays a row of arrays out end to end.
//!
//! Each of the `arrays` contributes a fixed number of values — its *width* — to every output row,
//! and the output holds those runs one after another, `output_height` rows of them. An array that
//! holds a single row's worth of values stands for that row repeated: it is broadcast over the
//! output rather than being written out one row per output row on the way in.
//!
//! A run at a time is what the arrays are copied in, through the builder of whichever array they
//! are — which is what lets a list array hand over a stretch of its values and offsets at once
//! rather than one element at a time. A struct array is the one that is taken apart first: see
//! [`flatten_structs`].

use polars_array::builder::{ShareStrategy, builder_like};
use polars_array::{PlArray, PlArrayBuilder, PlArrayType, PlBooleanArray, PlStructArray};

use crate::nesting::downcast;

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
    let out_len = row_width.saturating_mul(output_height);

    // A struct array is laid out one field at a time, which is what its every field being the
    // same layout of the arrays' matching fields means.
    if arrays[0].array_type() == PlArrayType::Struct {
        return Box::new(flatten_structs(arrays, widths, output_height, out_len));
    }

    let mut builder = builder_like(&*arrays[0]);
    builder.reserve(out_len);

    for row in 0..output_height {
        for ((array, &width), &repeats) in arrays.iter().zip(widths).zip(&repeats) {
            let start = if repeats { 0 } else { row * width };
            builder.subslice_extend(&**array, start, width, ShareStrategy::Always);
        }
    }

    builder.freeze()
}

/// Lays struct arrays out end to end, a field at a time.
///
/// Every field of the output is the same layout of the arrays' matching fields — a struct array
/// holds one element of each field per element of its own — and the outer validity is one more
/// such field, of booleans. So each of them is laid out on its own here, which asks the field's
/// own builder for a run of values at a time; going through the struct builder instead would take
/// every one of those runs apart into a call per field again, and pay for finding that field's
/// builder each time.
///
/// A struct array with no validity mask of its own contributes a run of elements that are all
/// there. What it contributes on the way in is a single bit standing for its every element, so no
/// mask is written out to be laid out; and where no array has a mask at all, the output has none
/// either.
fn flatten_structs(
    arrays: &[Box<dyn PlArray>],
    widths: &[usize],
    output_height: usize,
    out_len: usize,
) -> PlStructArray {
    let structs: Vec<&PlStructArray> = arrays
        .iter()
        .map(|array| downcast::<PlStructArray>(&**array))
        .collect();

    // A field array holds one element per element of the struct it belongs to, so it is as wide
    // and as long as that struct: which of the two the flatten reads it as is the same either way.
    let mut field = Vec::with_capacity(structs.len());
    let fields: Vec<Box<dyn PlArray>> = (0..structs[0].num_fields())
        .map(|i| {
            field.clear();
            field.extend(structs.iter().map(|array| array.field(i).to_boxed()));
            horizontal_flatten(&field, widths, output_height)
        })
        .collect();

    let validity = structs
        .iter()
        .any(|array| array.validity().is_some())
        .then(|| {
            let masks: Vec<Box<dyn PlArray>> = structs
                .iter()
                .map(|array| {
                    let mask = match array.validity() {
                        // Every element of this array is there, however many that is.
                        None => PlBooleanArray::new_scalar(true, array.len()),
                        Some(validity) => PlBooleanArray::from_pl_bitmap(validity.into()),
                    };
                    Box::new(mask) as Box<dyn PlArray>
                })
                .collect();

            let flattened = horizontal_flatten(&masks, widths, output_height);
            downcast::<PlBooleanArray>(&*flattened)
                .values()
                .to_flat_or_scalar()
        });

    // A struct of no fields carries nothing but its length, which the widths still say.
    PlStructArray::new(fields, out_len, None).with_validity_broadcast(validity)
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
    use arrow::bitmap::Bitmap;
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

    /// A struct array is laid out a field at a time, and its outer validity is one more such
    /// field: an array carrying no mask of its own contributes a run of elements that are there.
    #[test]
    fn struct_validity_is_laid_out_like_a_field() {
        let boxed = |values: [i32; 4], validity: Option<[bool; 4]>| -> Box<dyn PlArray> {
            Box::new(PlStructArray::new(
                vec![Box::new(PlPrimitiveArray::from_vec(values.to_vec()))],
                4,
                validity.map(|v| v.into_iter().collect()),
            ))
        };

        let arrays = vec![
            boxed([1, 2, 3, 4], Some([true, false, true, true])),
            boxed([5, 6, 7, 8], None),
        ];
        let out = horizontal_flatten(&arrays, &[2, 2], 2);
        let out = downcast::<PlStructArray>(&*out);

        assert_eq!(out.len(), 8);
        assert_eq!(
            elements_of::<i32>(out.field(0)),
            [1, 2, 5, 6, 3, 4, 7, 8].map(Some),
        );
        assert_eq!(
            (0..out.len()).map(|i| out.is_valid(i)).collect::<Vec<_>>(),
            [true, false, true, true, true, true, true, true],
        );

        // No array has a mask, so neither has the output.
        let arrays = vec![boxed([1, 2, 3, 4], None), boxed([5, 6, 7, 8], None)];
        let out = horizontal_flatten(&arrays, &[2, 2], 2);
        assert!(downcast::<PlStructArray>(&*out).validity().is_none());
    }

    /// A struct array whose mask repeats one bit says the same of its every element, and is read
    /// as the one bit it holds. The masks of two arrays interleave, so the one they are laid out
    /// into holds a bit per element — but neither of theirs was written out to get there.
    #[test]
    fn a_struct_mask_that_repeats_one_bit_is_read_as_one() {
        let boxed = |value: i32, validity: Option<bool>| -> Box<dyn PlArray> {
            Box::new(PlStructArray::new_broadcast(
                vec![Box::new(PlPrimitiveArray::new_scalar(value, 4))],
                4,
                validity.map(|bit| Bitmap::new_with_value(bit, 1)),
            ))
        };

        let arrays = vec![boxed(1, Some(false)), boxed(2, None)];
        let out = horizontal_flatten(&arrays, &[2, 2], 2);
        let out = downcast::<PlStructArray>(&*out);

        assert_eq!(
            elements_of::<i32>(out.field(0)),
            [1, 1, 2, 2, 1, 1, 2, 2].map(Some),
        );
        assert_eq!(
            (0..out.len()).map(|i| out.is_valid(i)).collect::<Vec<_>>(),
            [false, false, true, true, false, false, true, true],
        );

        // Where every array says the same of its every element, so does the output.
        let arrays = vec![boxed(1, Some(false)), boxed(2, Some(false))];
        let out = horizontal_flatten(&arrays, &[2, 2], 2);
        assert_eq!(downcast::<PlStructArray>(&*out).null_count(), 8);

        // And where none of them carries a mask at all, neither does the output.
        let arrays = vec![boxed(1, None), boxed(2, None)];
        let out = horizontal_flatten(&arrays, &[2, 2], 2);
        assert!(downcast::<PlStructArray>(&*out).validity().is_none());
    }

    /// A struct of no fields carries nothing but its length, which the widths still say.
    #[test]
    fn a_struct_of_no_fields_is_a_length() {
        let arrays: Vec<Box<dyn PlArray>> = vec![
            Box::new(PlStructArray::new(vec![], 4, None)),
            Box::new(PlStructArray::new(
                vec![],
                4,
                Some([true, false, true, true].into_iter().collect()),
            )),
        ];

        let out = horizontal_flatten(&arrays, &[2, 2], 2);
        let out = downcast::<PlStructArray>(&*out);
        assert_eq!(out.num_fields(), 0);
        assert_eq!(out.len(), 8);
        assert_eq!(
            (0..out.len()).map(|i| out.is_valid(i)).collect::<Vec<_>>(),
            [true, true, true, false, true, true, true, true],
        );
    }

    /// A struct inside a struct is taken apart in turn, mask and all.
    #[test]
    fn a_nested_struct_is_taken_apart_in_turn() {
        let boxed = |values: [i32; 2], validity: Option<[bool; 2]>| -> Box<dyn PlArray> {
            let inner = PlStructArray::new(
                vec![Box::new(PlPrimitiveArray::from_vec(values.to_vec()))],
                2,
                validity.map(|v| v.into_iter().collect()),
            );
            Box::new(PlStructArray::new(vec![Box::new(inner)], 2, None))
        };

        let arrays = vec![
            boxed([1, 2], Some([true, false])),
            boxed([3, 4], Some([false, true])),
        ];
        let out = horizontal_flatten(&arrays, &[1, 1], 2);
        let out = downcast::<PlStructArray>(&*out);

        let inner = downcast::<PlStructArray>(out.field(0));
        assert_eq!(elements_of::<i32>(inner.field(0)), [1, 3, 2, 4].map(Some));
        assert_eq!(
            (0..inner.len())
                .map(|i| inner.is_valid(i))
                .collect::<Vec<_>>(),
            [true, false, false, true],
        );
    }

    /// No output row is no output at all, however wide the arrays are.
    #[test]
    fn no_rows_yield_no_values() {
        let arrays: Vec<Box<dyn PlArray>> = vec![Box::new(PlPrimitiveArray::<i32>::new_empty())];
        assert_eq!(horizontal_flatten(&arrays, &[2], 0).len(), 0);
    }
}
