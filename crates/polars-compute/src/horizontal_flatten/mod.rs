//! The kernel behind `concat_arr`, which lays a row of arrays out end to end.

use polars_array::builder::{ShareStrategy, StaticArrayBuilder, builder_like};
use polars_array::collect::{ArrayCollectIterExt, ArrayFromIter};
use polars_array::{
    PlArray, PlArrayBuilder, PlArrayType, PlBinaryArrayBuilder, PlBinaryViewArrayBuilder,
    PlBooleanArray, PlPrimitiveArray, PlUtf8ViewArrayBuilder, StaticArray,
    with_match_pl_primitive_array_type,
};

mod struct_;

/// Lays the `arrays` out end to end, `output_height` rows of `widths[i]` values from `arrays[i]`.
///
/// # Panics
/// Panics unless the arrays are of one type and each holds one of the two admissible lengths.
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

    let repeats: Vec<bool> = arrays
        .iter()
        .zip(widths)
        .map(|(array, &width)| is_broadcast(&**array, width, output_height))
        .collect();

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

    if arrays[0].array_type() == PlArrayType::Struct {
        return Box::new(struct_::flatten_structs(
            arrays,
            widths,
            output_height,
            out_len,
        ));
    }

    // A row is a handful of values, so entering the builder is the whole cost: through
    // `dyn PlArrayBuilder` each of the `output_height * arrays.len()` calls resolves the
    // builder's array type and downcasts the array to it again.  Resolve it once where the
    // builder needs nothing from the array to be constructed.
    macro_rules! typed {
        ($builder:expr) => {
            return flatten_typed($builder, arrays, widths, &repeats, output_height, out_len)
        };
    }

    match arrays[0].array_type() {
        PlArrayType::Primitive(_) => {
            return with_match_pl_primitive_array_type!(&*arrays[0], |$T| {
                Box::new(flatten_elementwise::<PlPrimitiveArray<$T>>(
                    arrays,
                    widths,
                    &repeats,
                    out_len,
                )) as Box<dyn PlArray>
            })
            .expect("a primitive array has a primitive element type");
        },
        PlArrayType::Boolean => {
            return Box::new(flatten_elementwise::<PlBooleanArray>(
                arrays, widths, &repeats, out_len,
            ));
        },
        PlArrayType::BinaryView => typed!(PlBinaryViewArrayBuilder::new()),
        PlArrayType::Utf8View => typed!(PlUtf8ViewArrayBuilder::new()),
        PlArrayType::Binary => typed!(PlBinaryArrayBuilder::new()),
        _ => {},
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

/// Lays the rows out one value at a time, collecting them into the array they belong to.
///
/// A row takes `widths[i]` values from each array, which is a handful — so entering a builder for
/// each run costs more than the run copies, wherever a value is a plain copy rather than a slice
/// of a shared buffer.  Walking the output instead keeps the whole lay-out to one trusted collect.
fn flatten_elementwise<A>(
    arrays: &[Box<dyn PlArray>],
    widths: &[usize],
    repeats: &[bool],
    out_len: usize,
) -> A
where
    A: StaticArray + for<'a> ArrayFromIter<Option<<A as StaticArray>::ValueT<'a>>>,
{
    let typed: Vec<&A> = arrays
        .iter()
        .map(|array| {
            array
                .as_any()
                .downcast_ref::<A>()
                .expect("the arrays all hold the array type dispatched on")
        })
        .collect();

    // Where in the output the walk is: which array the next value comes from, how many of its
    // values this row has taken already, and which row that is.
    let mut array = 0;
    let mut taken = 0;
    let mut row = 0;

    (0..out_len)
        .map(|_| {
            // An array may be zero values wide, in which case the row takes nothing from it and
            // the walk moves straight on to the next.
            while taken == widths[array] {
                taken = 0;
                array += 1;
                if array == widths.len() {
                    array = 0;
                    row += 1;
                }
            }

            // A broadcast array holds the one row it stands for, and is read at that row's offsets.
            let start = if repeats[array] { 0 } else { row * widths[array] };
            taken += 1;

            // SAFETY: the array holds `widths[array]` values per row it covers, and the walk is
            // inside the run this row takes from it — the assertions in `is_broadcast` above are
            // what say the array covers every row.
            unsafe { typed[array].get_unchecked(start + taken - 1) }
        })
        .collect_arr_trusted()
}

/// Lays the rows out through a builder whose array type is already known.
fn flatten_typed<B: StaticArrayBuilder>(
    mut builder: B,
    arrays: &[Box<dyn PlArray>],
    widths: &[usize],
    repeats: &[bool],
    output_height: usize,
    out_len: usize,
) -> Box<dyn PlArray> {
    let typed: Vec<&B::Array> = arrays
        .iter()
        .map(|array| {
            array
                .as_any()
                .downcast_ref::<B::Array>()
                .expect("the arrays all hold the array type dispatched on")
        })
        .collect();

    builder.reserve(out_len);
    for row in 0..output_height {
        for ((array, &width), &repeats) in typed.iter().zip(widths).zip(repeats) {
            let start = if repeats { 0 } else { row * width };
            builder.subslice_extend(array, start, width, ShareStrategy::Always);
        }
    }

    Box::new(builder.freeze())
}

/// Whether `array` holds the one row of `width` values it stands for at every output row.
fn is_broadcast(array: &dyn PlArray, width: usize, output_height: usize) -> bool {
    let flat = width.checked_mul(output_height);
    if flat == Some(array.len()) {
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
