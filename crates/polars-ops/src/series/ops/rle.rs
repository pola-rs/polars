use std::hash::Hash;

use polars_arrow::bitmap::utils::SlicesIterator;
use polars_core::prelude::*;
use polars_core::series::{BitRepr, IsSorted};
use polars_core::with_match_physical_float_polars_type;
use polars_utils::select::select_unpredictable;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

pub static RLE_VALUE_COLUMN_NAME: &str = "value";
pub static RLE_LENGTH_COLUMN_NAME: &str = "len";

/// The average run a validity mask has to hold for reading its runs off the bits to be worth it
/// over walking the elements, in [`rle_lengths`].
const SHORTEST_WORTHWHILE_MASK_RUN: usize = 2;

/// Get the run-lengths of values.
pub fn rle_lengths(s: &Column, lengths: &mut Vec<IdxSize>) -> PolarsResult<()> {
    lengths.clear();
    if s.is_empty() {
        return Ok(());
    }
    if s.len() == 1 {
        lengths.push(1);
        return Ok(());
    }

    if let Some(sc) = s.as_scalar_column() {
        lengths.push(sc.len() as IdxSize);
        return Ok(());
    }

    let s = s.as_materialized_series();

    // A column that repeats one element is one run of that element, whatever the element is: the
    // typed helpers below would read the repeat out one element at a time to say so. Several
    // chunks that all repeat the same element are still the one run, which is the shape the
    // streaming engine hands this op.
    if s.repeats_one_element() {
        lengths.push(s.len() as IdxSize);
        return Ok(());
    }

    // The same chunk under a mask of one bit per element: every element still holds the one value
    // the values repeat, so two of them differ only where the mask does -- a null equals a null
    // and differs from a value. The runs are the mask's own runs, counted off its bits rather
    // than found by reading a million copies of one value against each other.
    if let [chunk] = s.chunks().as_slice()
        && let Some(validity) = chunk.validity()
        && chunk.without_validity().is_scalar()
    {
        // The mask is not a single bit: a chunk whose mask repeats one bit as well is scalar
        // throughout, and was answered above.
        let (bits, length) = validity.into_inner();

        // Counting the mask's runs off its words is far cheaper than walking them, but where the
        // runs are as short as single elements there is nothing left to win: one length is
        // written per run either way, and finding each run in the bits then costs more than the
        // comparison walk below costs per element. Over 200k elements, a mask that opens a run
        // every other element reads 0.46 ms this way against 0.50 walked, and one that opens a
        // run at every element reads 0.73 against 0.56.
        if (bits.num_edges() + 1) * SHORTEST_WORTHWHILE_MASK_RUN <= length {
            let mut opened = 0;
            for (start, run) in SlicesIterator::new(bits) {
                if start > opened {
                    lengths.push((start - opened) as IdxSize);
                }
                lengths.push(run as IdxSize);
                opened = start + run;
            }
            if opened < length {
                lengths.push((length - opened) as IdxSize);
            }
            return Ok(());
        }
    }

    let s = s.to_physical_repr();
    match s.dtype() {
        DataType::Boolean => {
            let ca: &BooleanChunked = s.as_ref().as_ref().as_ref();
            rle_lengths_helper_ca(ca, lengths);
            return Ok(());
        },
        dt if dt.is_float() => {
            with_match_physical_float_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                rle_lengths_helper_ca(ca, lengths);
                return Ok(());
            })
        },
        dt if dt.is_numeric() => {
            use BitRepr as B;
            match s.bit_repr().unwrap() {
                B::U8(ca) => rle_lengths_helper_ca(&ca, lengths),
                B::U16(ca) => rle_lengths_helper_ca(&ca, lengths),
                B::U32(ca) => rle_lengths_helper_ca(&ca, lengths),
                B::U64(ca) => rle_lengths_helper_ca(&ca, lengths),
                #[cfg(feature = "dtype-u128")]
                B::U128(ca) => rle_lengths_helper_ca(&ca, lengths),
            }
            return Ok(());
        },
        DataType::String => {
            let ca: &StringChunked = s.as_ref().as_ref().as_ref();
            rle_lengths_helper_ca(&ca.as_binary(), lengths);
            return Ok(());
        },
        DataType::Binary => {
            let ca: &BinaryChunked = s.as_ref().as_ref().as_ref();
            rle_lengths_helper_ca(ca, lengths);
            return Ok(());
        },
        DataType::BinaryOffset => {
            let ca: &BinaryOffsetChunked = s.as_ref().as_ref().as_ref();
            rle_lengths_helper_ca(ca, lengths);
            return Ok(());
        },
        _ => {},
    }

    let (s1, s2) = (s.slice(0, s.len() - 1), s.slice(1, s.len()));
    let s_neq = s1.not_equal_missing(&s2)?;
    let n_runs = s_neq.sum().unwrap() + 1;

    lengths.reserve(n_runs as usize);
    lengths.push(1);

    assert!(!s_neq.has_nulls());
    for arr in s_neq.downcast_iter() {
        // A scalar chunk stands for one bit at every element: either nothing in it differs from
        // the element before, and the run carries on through the whole chunk, or everything does
        // and every element opens a run of its own. Neither needs the bits written out.
        if let Some(differs) = arr.values().scalar_value() {
            if differs {
                lengths.resize(lengths.len() + arr.len(), 1);
            } else {
                *lengths.last_mut().unwrap() += arr.len() as IdxSize;
            }
            continue;
        }

        // What is left holds one bit per element already, so this borrows rather than writes out.
        let mut values = arr.values().to_flat().into_owned();
        while !values.is_empty() {
            // @NOTE: This `as IdxSize` is safe because it is less than or equal to the a ChunkedArray
            // length.
            *lengths.last_mut().unwrap() += values.take_leading_zeros() as IdxSize;

            if !values.is_empty() {
                lengths.push(1);
                values.slice(1, values.len() - 1);
            }
        }
    }
    Ok(())
}

pub fn rle_lengths_helper_ca<'a, T>(ca: &'a ChunkedArray<T>, lengths: &mut Vec<IdxSize>)
where
    T: PolarsDataType,
    T::Physical<'a>: TotalHash + TotalEq + ToTotalOrd + Copy,
    <T::Physical<'a> as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    lengths.clear();
    if ca.is_empty() {
        return;
    }

    unsafe {
        lengths.reserve(ca.len());
        // One run is written per element at most, and the buffer holds that many: the pointer is
        // settled here rather than read back out of the `Vec` once an element.
        let out = lengths.as_mut_ptr();

        // Each element is compared against the one before it and, where the two differ, opens a
        // run of its own. The walk is a fold so that a chunk's representation is resolved before
        // it rather than at every element of it -- which over a column whose elements never
        // differ, and so never stall the walk, is the whole of the work.
        let out_idx = if ca.has_nulls() {
            let prev = ca.get_unchecked(0).map(|v| v.to_total_ord());
            ca.downcast_iter()
                .fold((prev, 0usize, 0), |acc, arr| {
                    arr.iter().fold(acc, |(prev, out_idx, run_len), val| {
                        let val = val.map(|v| v.to_total_ord());
                        let diff = val != prev;
                        let run_len = 1 + select_unpredictable(diff, 0, run_len);
                        let out_idx = out_idx + diff as usize;
                        out.add(out_idx).write(run_len);
                        (val, out_idx, run_len)
                    })
                })
                .1
        } else {
            let prev = ca.value_unchecked(0).to_total_ord();
            ca.downcast_iter()
                .fold((prev, 0usize, 0), |acc, arr| {
                    arr.values_iter()
                        .fold(acc, |(prev, out_idx, run_len), val| {
                            let val = val.to_total_ord();
                            let diff = val != prev;
                            let run_len = 1 + select_unpredictable(diff, 0, run_len);
                            let out_idx = out_idx + diff as usize;
                            out.add(out_idx).write(run_len);
                            (val, out_idx, run_len)
                        })
                })
                .1
        };

        lengths.set_len(out_idx + 1);
    }
}

/// Get the lengths of runs of identical values.
pub fn rle(s: &Column) -> PolarsResult<Column> {
    let mut lengths = Vec::new();
    rle_lengths(s, &mut lengths)?;

    let mut idxs = Vec::with_capacity(lengths.len());
    if !lengths.is_empty() {
        idxs.push(0);
        for length in &lengths[..lengths.len() - 1] {
            idxs.push(*idxs.last().unwrap() + length);
        }
    }

    let vals = s
        .take_slice(&idxs)
        .unwrap()
        .with_name(PlSmallStr::from_static(RLE_VALUE_COLUMN_NAME));
    let outvals = vec![
        Series::from_vec(PlSmallStr::from_static(RLE_LENGTH_COLUMN_NAME), lengths).into(),
        vals,
    ];
    Ok(StructChunked::from_columns(s.name().clone(), idxs.len(), &outvals)?.into_column())
}

/// Similar to `rle`, but maps values to run IDs.
pub fn rle_id(s: &Column) -> PolarsResult<Column> {
    if s.is_empty() {
        return Ok(Column::new_empty(s.name().clone(), &IDX_DTYPE));
    }

    let (s1, s2) = (s.slice(0, s.len() - 1), s.slice(1, s.len()));
    let s_neq = s1
        .as_materialized_series()
        .not_equal_missing(s2.as_materialized_series())?;

    // A column whose neighbours never differ is a single run, and every element carries its id.
    if let Some(Some(false)) = s_neq.scalar_value() {
        return Ok(IdxCa::full(s.name().clone(), 0, s.len()).into_column());
    }

    let mut out = Vec::<IdxSize>::with_capacity(s.len());
    let mut last = 0;
    out.push(last); // Run numbers start at zero
    assert_eq!(s_neq.null_count(), 0);
    for a in s_neq.downcast_iter() {
        for aa in a.values_iter() {
            last += aa as IdxSize;
            out.push(last);
        }
    }
    Ok(IdxCa::from_vec(s.name().clone(), out)
        .with_sorted_flag(IsSorted::Ascending)
        .into_column())
}

#[cfg(test)]
mod test {
    use polars_arrow::bitmap::Bitmap;

    use super::*;

    /// The runs of a chunk whose values repeat under a mask are the mask's own runs, and which
    /// side of the guard in [`rle_lengths`] the mask falls on must not change them.
    #[test]
    fn a_masked_repeat_has_the_masks_runs_however_they_are_found() {
        let n = 301;
        for period in [2usize, 3, 4, 5, 64, 300] {
            let mask: Vec<bool> = (0..n).map(|i| i % period != 0).collect();

            // One value repeated under the mask, which is the chunk the guard is about.
            let repeated = Int32Chunked::full(PlSmallStr::from_static("x"), 7, n)
                .with_validity(Some(PlBitmap::from_bitmap(Bitmap::from_trusted_len_iter(
                    mask.iter().copied(),
                ))))
                .into_column();

            // The same elements with one slot per element, which takes the walk either way.
            let written_out = Int32Chunked::from_slice_options(
                PlSmallStr::from_static("x"),
                &mask.iter().map(|m| m.then_some(7)).collect::<Vec<_>>(),
            )
            .into_column();

            let mut from_mask = Vec::new();
            let mut from_walk = Vec::new();
            rle_lengths(&repeated, &mut from_mask).unwrap();
            rle_lengths(&written_out, &mut from_walk).unwrap();

            assert_eq!(from_mask, from_walk, "period {period}");
            assert_eq!(
                from_mask.iter().sum::<IdxSize>(),
                n as IdxSize,
                "period {period}"
            );
        }
    }
}
