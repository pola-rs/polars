pub mod cov;
pub mod mean;
pub mod options;
pub mod sum;
use arrow::types::NativeType;
pub use cov::{EwmCovState, EwmStdState, EwmVarState, ewm_std, ewm_var};
pub use mean::{EwmMeanState, ewm_mean};
pub use options::EWMOptions;
use polars_array::{PlArray, PlPrimitiveArray};
pub use sum::{EwmSumState, ewm_sum};

/// A stateful exponentially weighted kernel, folded over the chunks of a column in order.
///
/// Every element advances a recurrence over the elements before it, so — unlike the reducing
/// kernels — a chunk that repeats one value has no answer to be read in `O(1)`: the state has to
/// walk it either way. What the scalar representation does still save is the memory. The elements
/// are read through the chunk's own iterator, which repeats the single value where it lies, so
/// such a chunk is never written out one value per element on the way in.
pub trait EwmStateUpdate {
    fn ewm_state_update(&mut self, values: &dyn PlArray) -> Box<dyn PlArray>;
}

/// The elements of `values`, as the primitive chunk the state was built to read.
///
/// # Panics
/// Panics unless `values` is a [`PlPrimitiveArray<T>`], which is what the physical type of the
/// column the state was built for makes it.
fn chunk_of<T: NativeType>(values: &dyn PlArray) -> &PlPrimitiveArray<T> {
    values
        .as_any()
        .downcast_ref()
        .expect("EWM state reads a different primitive type than the chunk it was given")
}

#[cfg(test)]
macro_rules! assert_allclose {
    ($xs:expr, $ys:expr, $tol:expr) => {{
        // Bound once: the operands are call expressions that consume what they are given.
        let (xs, ys) = (&$xs, &$ys);
        assert_eq!(xs.len(), ys.len(), "compared chunks of different lengths");
        assert!(xs.iter().zip(ys.iter()).all(|(x, z)| {
            match (x, z) {
                (Some(a), Some(b)) => (a - b).abs() < $tol,
                (None, None) => true,
                _ => false,
            }
        }));
    }};
}

#[cfg(test)]
use assert_allclose;

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;
    use polars_array::{PlArray, PlPrimitiveArray};

    use super::{EwmCovState, EwmMeanState, EwmStateUpdate, EwmStdState, EwmSumState, EwmVarState};

    const ALPHA: f64 = 0.3;

    /// One freshly built state per kernel, over `f64`, to run the same chunks through. The
    /// options are varied across them so that neither branch of `adjust`, `bias` or
    /// `ignore_nulls` goes unread.
    fn states() -> Vec<(&'static str, Box<dyn EwmStateUpdate>)> {
        vec![
            (
                "mean",
                Box::new(EwmMeanState::<f64>::new(ALPHA, true, 1, false)),
            ),
            (
                "mean, unadjusted",
                Box::new(EwmMeanState::<f64>::new(ALPHA, false, 2, true)),
            ),
            ("sum", Box::new(EwmSumState::<f64>::new(ALPHA, 1, false))),
            (
                "var",
                Box::new(EwmVarState::new(EwmCovState::<f64>::new(
                    ALPHA, true, false, 1, false,
                ))),
            ),
            (
                "std",
                Box::new(EwmStdState::new(EwmCovState::<f64>::new(
                    ALPHA, false, true, 1, true,
                ))),
            ),
        ]
    }

    /// The elements of a chunk a kernel handed back.
    fn elements_of(chunk: &dyn PlArray) -> Vec<Option<f64>> {
        chunk
            .as_any()
            .downcast_ref::<PlPrimitiveArray<f64>>()
            .expect("the kernels hand back a chunk of the type they were given")
            .iter()
            .collect()
    }

    /// `length` copies of `value`, marked by `validity`, in both representations.
    fn repeated(
        value: f64,
        validity: Option<&Bitmap>,
        length: usize,
    ) -> [PlPrimitiveArray<f64>; 2] {
        let scalar =
            PlPrimitiveArray::new_scalar(value, length).with_validity_broadcast(validity.cloned());
        let flat = PlPrimitiveArray::from_vec(vec![value; length])
            .with_validity_broadcast(validity.cloned());
        assert_eq!(scalar, flat);
        [scalar, flat]
    }

    /// A chunk that repeats one value advances the state exactly as the same elements laid out one
    /// per slot do, down to the last bit: the kernel reads either one through the chunk's own
    /// iterator, which repeats a single value where it lies instead of writing it out.
    #[test]
    fn a_scalar_chunk_reads_like_a_flat_one() {
        for length in [0, 1, 2, 3, 65] {
            for valid in 0..=length {
                let mask: Bitmap = (0..length).map(|i| i < valid).collect();
                for validity in [None, Some(&mask)] {
                    let [scalar, flat] = repeated(2.5, validity, length);

                    for ((name, mut over_scalar), (_, mut over_flat)) in
                        states().into_iter().zip(states())
                    {
                        assert_eq!(
                            elements_of(&*over_scalar.ewm_state_update(&scalar)),
                            elements_of(&*over_flat.ewm_state_update(&flat)),
                            "{name} over {length} copies of 2.5, {valid} of them valid",
                        );
                    }
                }
            }
        }
    }

    /// The state is a recurrence over every element it has seen, so folding it over two chunks in
    /// turn has to read the same as folding it over the one chunk that holds them all.
    #[test]
    fn the_state_carries_across_chunks() {
        let elements = [
            Some(1.0f64),
            None,
            Some(4.0),
            Some(2.0),
            None,
            Some(3.5),
            Some(-1.0),
        ];
        let whole = PlPrimitiveArray::from_iter(elements);

        for split in 0..=elements.len() {
            let (head, tail) = elements.split_at(split);
            let head = PlPrimitiveArray::from_iter(head.iter().copied());
            let tail = PlPrimitiveArray::from_iter(tail.iter().copied());

            for ((name, mut in_one), (_, mut in_two)) in states().into_iter().zip(states()) {
                let one = elements_of(&*in_one.ewm_state_update(&whole));

                let mut two = elements_of(&*in_two.ewm_state_update(&head));
                two.extend(elements_of(&*in_two.ewm_state_update(&tail)));

                assert_eq!(one, two, "{name}, split after {split} of {elements:?}");
            }
        }
    }

    /// A chunk stored as a repeated null has no value under it for the state to read, and leaves
    /// nothing but nulls behind however long it is.
    #[test]
    fn a_null_chunk_yields_nothing() {
        let nulls = PlPrimitiveArray::<f64>::new_full_null(5);

        for (name, mut state) in states() {
            assert_eq!(
                elements_of(&*state.ewm_state_update(&nulls)),
                vec![None; 5],
                "{name} over five nulls",
            );
        }
    }
}
