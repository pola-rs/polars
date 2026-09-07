//! The kernel behind `concat_arr`, which lays a row of arrays out end to end.

use polars_array::builder::{ShareStrategy, builder_like};
use polars_array::{PlArray, PlArrayBuilder, PlArrayType};

mod struct_;

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
        return Box::new(struct_::flatten_structs(
            arrays,
            widths,
            output_height,
            out_len,
        ));
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

/// Whether `array` holds the one row of `width` values it stands for at every output row.
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
