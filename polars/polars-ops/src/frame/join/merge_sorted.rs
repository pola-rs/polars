use polars_arrow::utils::{CustomIterTools, FromTrustedLenIterator};
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

pub fn _merge_sorted_dfs(
    left: &DataFrame,
    right: &DataFrame,
    left_s: &Series,
    right_s: &Series,
    check_schema: bool,
) -> PolarsResult<DataFrame> {
    if check_schema {
        left.frame_equal_schema(right)?;
    }
    let dtype_lhs = left_s.dtype();
    let dtype_rhs = right_s.dtype();

    if dtype_lhs != dtype_rhs {
        return Err(PolarsError::ComputeError(
            "DataTypes in merge sort should be equal".into(),
        ));
    }

    let merge_indicator = series_to_merge_indicator(left_s, right_s);
    let new_columns = left
        .get_columns()
        .iter()
        .zip(right.get_columns())
        .map(|(lhs, rhs)| {
            let lhs_phys = lhs.to_physical_repr();
            let rhs_phys = rhs.to_physical_repr();

            let out = merge_series(&lhs_phys, &rhs_phys, &merge_indicator);
            let mut out = out.cast(lhs.dtype()).unwrap();
            out.rename(lhs.name());
            out
        })
        .collect();

    Ok(DataFrame::new_no_checks(new_columns))
}

fn merge_series(lhs: &Series, rhs: &Series, merge_indicator: &[bool]) -> Series {
    use DataType::*;
    match lhs.dtype() {
        Boolean => {
            let lhs = lhs.bool().unwrap();
            let rhs = rhs.bool().unwrap();

            merge_ca(lhs, rhs, merge_indicator).into_series()
        }
        Utf8 => {
            let lhs = lhs.utf8().unwrap();
            let rhs = rhs.utf8().unwrap();
            merge_ca(lhs, rhs, merge_indicator).into_series()
        }
        #[cfg(feature = "dtype-binary")]
        Binary => {
            let lhs = lhs.binary().unwrap();
            let rhs = rhs.binary().unwrap();
            merge_ca(lhs, rhs, merge_indicator).into_series()
        }
        #[cfg(feature = "dtype-binary")]
        Struct(_) => {
            let lhs = lhs.struct_().unwrap();
            let rhs = rhs.struct_().unwrap();

            let new_fields = lhs
                .fields()
                .iter()
                .zip(rhs.fields())
                .map(|(lhs, rhs)| merge_series(lhs, rhs, merge_indicator))
                .collect::<Vec<_>>();
            StructChunked::new("", &new_fields).unwrap().into_series()
        }
        List(_) => {
            let lhs = lhs.list().unwrap();
            let rhs = rhs.list().unwrap();
            merge_ca(lhs, rhs, merge_indicator).into_series()
        }
        dt => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                    let lhs: &ChunkedArray<$T> = lhs.as_ref().as_ref().as_ref();
                    let rhs: &ChunkedArray<$T> = rhs.as_ref().as_ref().as_ref();
                    merge_ca(lhs, rhs, merge_indicator).into_series()
            })
        }
    }
}

fn merge_ca<'a, T>(
    a: &'a ChunkedArray<T>,
    b: &'a ChunkedArray<T>,
    merge_indicator: &[bool],
) -> ChunkedArray<T>
where
    T: PolarsDataType + 'static,
    &'a ChunkedArray<T>: IntoIterator,
    ChunkedArray<T>:
        FromTrustedLenIterator<<<&'a ChunkedArray<T> as IntoIterator>::IntoIter as Iterator>::Item>,
{
    let total_len = a.len() + b.len();
    let mut a = a.into_iter();
    let mut b = b.into_iter();

    let iter = merge_indicator.iter().map(|a_indicator| {
        if *a_indicator {
            a.next().unwrap()
        } else {
            b.next().unwrap()
        }
    });

    // Safety: length is correct
    unsafe { iter.trust_my_length(total_len).collect_trusted() }
}

fn series_to_merge_indicator(lhs: &Series, rhs: &Series) -> Vec<bool> {
    let lhs_s = lhs.to_physical_repr().into_owned();
    let rhs_s = rhs.to_physical_repr().into_owned();

    match lhs_s.dtype() {
        DataType::Boolean => {
            let lhs = lhs_s.bool().unwrap();
            let rhs = rhs_s.bool().unwrap();
            get_merge_indicator(lhs.into_iter(), rhs.into_iter())
        }
        DataType::Utf8 => {
            let lhs = lhs_s.utf8().unwrap();
            let rhs = rhs_s.utf8().unwrap();

            get_merge_indicator(lhs.into_iter(), rhs.into_iter())
        }
        _ => {
            with_match_physical_numeric_polars_type!(lhs_s.dtype(), |$T| {
                    let lhs: &ChunkedArray<$T> = lhs_s.as_ref().as_ref().as_ref();
                    let rhs: &ChunkedArray<$T> = rhs_s.as_ref().as_ref().as_ref();

                    get_merge_indicator(lhs.into_iter(), rhs.into_iter())

            })
        }
    }
}

// get a boolean values, left: true, right: false
// that indicate from which side we should take a value
fn get_merge_indicator<T>(
    mut a_iter: impl ExactSizeIterator<Item = T>,
    mut b_iter: impl ExactSizeIterator<Item = T>,
) -> Vec<bool>
where
    T: PartialOrd + Default + Copy,
{
    const A_INDICATOR: bool = true;
    const B_INDICATOR: bool = false;

    let a_len = a_iter.size_hint().0;
    let b_len = b_iter.size_hint().0;
    if a_len == 0 {
        return vec![true; b_len];
    };
    if b_len == 0 {
        return vec![false; a_len];
    }

    let mut current_a = T::default();
    let cap = a_len + b_len;
    let mut out = Vec::with_capacity(cap);

    let mut current_b = b_iter.next().unwrap();

    for a in &mut a_iter {
        current_a = a;
        if a <= current_b {
            // safety
            // we pre-allocated enough
            out.push(A_INDICATOR);
            continue;
        }
        out.push(B_INDICATOR);

        loop {
            if let Some(b) = b_iter.next() {
                current_b = b;
                if b >= a {
                    out.push(A_INDICATOR);
                    break;
                }
                out.push(B_INDICATOR);
                continue;
            }
            // b is depleted fill with a indicator
            let remaining = cap - out.len();
            out.extend(std::iter::repeat(A_INDICATOR).take(remaining));
            return out;
        }
    }
    if current_a < current_b {
        out.push(B_INDICATOR);
    }
    // check if current value already is added
    if *out.last().unwrap() == A_INDICATOR {
        out.push(B_INDICATOR);
    }
    // take remaining
    out.extend(b_iter.map(|_| B_INDICATOR));
    assert_eq!(out.len(), b_len + a_len);

    out
}

#[test]
fn test_merge_sorted() {
    fn get_merge_indicator_sliced<T: PartialOrd + Default + Copy>(a: &[T], b: &[T]) -> Vec<bool> {
        get_merge_indicator(a.iter().copied(), b.iter().copied())
    }

    let a = [1, 2, 4, 6, 9];
    let b = [2, 3, 4, 5, 10];

    let out = get_merge_indicator_sliced(&a, &b);
    let expected = [
        true, true, false, false, true, false, false, true, true, false,
    ];
    //                       1     2     2      3      4     4      5      6     9     10
    assert_eq!(out, expected);

    // swap
    // it is not the inverse because left is preferred when both are equal
    let out = get_merge_indicator_sliced(&b, &a);
    let expected = [
        false, true, false, true, true, false, true, false, false, true,
    ];
    assert_eq!(out, expected);

    let a = [5, 6, 7, 10];
    let b = [1, 2, 5];
    let out = get_merge_indicator_sliced(&a, &b);
    let expected = [false, false, true, false, true, true, true];
    assert_eq!(out, expected);

    // swap
    // it is not the inverse because left is preferred when both are equal
    let out = get_merge_indicator_sliced(&b, &a);
    let expected = [true, true, true, false, false, false, false];
    assert_eq!(out, expected);
}
