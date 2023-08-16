use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum RangeFunction {
    IntRange { step: i64 },
    IntRanges { step: i64 },
}

impl Display for RangeFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use RangeFunction::*;
        match self {
            IntRange { .. } => write!(f, "int_range"),
            IntRanges { .. } => write!(f, "int_ranges"),
        }
    }
}

fn int_range_impl<T>(start: T::Native, end: T::Native, step: i64) -> PolarsResult<Series>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
    std::ops::Range<T::Native>: Iterator<Item = T::Native>,
    std::ops::RangeInclusive<T::Native>: DoubleEndedIterator<Item = T::Native>,
{
    let name = "int";

    let mut ca = match step {
        0 => polars_bail!(InvalidOperation: "step must not be zero"),
        1 => ChunkedArray::<T>::from_iter_values(name, start..end),
        2.. => ChunkedArray::<T>::from_iter_values(name, (start..end).step_by(step as usize)),
        _ => {
            polars_ensure!(start > end, InvalidOperation: "range must be decreasing if 'step' is negative");
            ChunkedArray::<T>::from_iter_values(
                name,
                (end..=start).rev().step_by(step.unsigned_abs() as usize),
            )
        },
    };

    let is_sorted = if end < start {
        IsSorted::Descending
    } else {
        IsSorted::Ascending
    };
    ca.set_sorted_flag(is_sorted);

    Ok(ca.into_series())
}

pub(super) fn int_range(s: &[Series], step: i64) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];

    match start.dtype() {
        dt if dt == &IDX_DTYPE => {
            let start = start
                .idx()?
                .get(0)
                .ok_or_else(|| polars_err!(NoData: "no data in `start` evaluation"))?;
            let end = end.cast(&IDX_DTYPE)?;
            let end = end
                .idx()?
                .get(0)
                .ok_or_else(|| polars_err!(NoData: "no data in `end` evaluation"))?;

            int_range_impl::<IdxType>(start, end, step)
        },
        _ => {
            let start = start.cast(&DataType::Int64)?;
            let end = end.cast(&DataType::Int64)?;
            let start = start
                .i64()?
                .get(0)
                .ok_or_else(|| polars_err!(NoData: "no data in `start` evaluation"))?;
            let end = end
                .i64()?
                .get(0)
                .ok_or_else(|| polars_err!(NoData: "no data in `end` evaluation"))?;
            int_range_impl::<Int64Type>(start, end, step)
        },
    }
}

pub(super) fn int_ranges(s: &[Series], step: i64) -> PolarsResult<Series> {
    let start = &s[0].rechunk();
    let end = &s[1].rechunk();

    let output_name = "int_range";

    let mut start = start.cast(&DataType::Int64)?;
    let mut end = end.cast(&DataType::Int64)?;

    if start.len() != end.len() {
        if start.len() == 1 {
            start = start.new_from_index(0, end.len())
        } else if end.len() == 1 {
            end = end.new_from_index(0, start.len())
        } else {
            polars_bail!(
                ComputeError:
                "lengths of `start`: {} and `end`: {} arguments `\
                cannot be matched in the `int_ranges` expression",
                start.len(), end.len()
            );
        }
    }

    let start = start.i64()?;
    let end = end.i64()?;

    let start = start.downcast_iter().next().unwrap();
    let end = end.downcast_iter().next().unwrap();

    // First do a pass to determine the required value capacity.
    let mut values_capacity = 0;
    for (opt_start, opt_end) in start.into_iter().zip(end) {
        if let (Some(start_v), Some(end_v)) = (opt_start, opt_end) {
            match step {
                1 => {
                    values_capacity += (end_v - start_v) as usize;
                },
                2.. => {
                    values_capacity += ((end_v - start_v) as usize / step as usize) + 1;
                },
                _ => {
                    polars_ensure!(start_v > end_v, InvalidOperation: "range must be decreasing if 'step' is negative");
                    values_capacity +=
                        ((end_v - start_v) as usize / step.unsigned_abs() as usize) + 1;
                },
            }
        }
    }

    let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
        output_name,
        start.len(),
        values_capacity,
        DataType::Int64,
    );

    for (opt_start, opt_end) in start.into_iter().zip(end) {
        match (opt_start, opt_end) {
            (Some(&start_v), Some(&end_v)) => match step {
                1 => {
                    builder.append_iter_values(start_v..end_v);
                },
                2.. => {
                    builder.append_iter_values((start_v..end_v).step_by(step as usize));
                },
                _ => builder.append_iter_values(
                    (end_v..=start_v)
                        .rev()
                        .step_by(step.unsigned_abs() as usize),
                ),
            },
            _ => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}
