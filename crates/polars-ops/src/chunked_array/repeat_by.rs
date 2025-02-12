use arrow::array::ListArray;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

type LargeListArray = ListArray<i64>;

fn check_lengths(length_srs: usize, length_by: usize) -> PolarsResult<()> {
    polars_ensure!(
       (length_srs == length_by) | (length_by == 1) | (length_srs == 1),
       ComputeError: "repeat_by argument and the Series should have equal length, or at least one of them should have length 1. Series length {}, by length {}",
       length_srs, length_by
    );
    Ok(())
}

fn new_by(by: &IdxCa, len: usize) -> IdxCa {
    if let Some(x) = by.get(0) {
        let values = std::iter::repeat(x).take(len).collect::<Vec<IdxSize>>();
        IdxCa::new(PlSmallStr::EMPTY, values)
    } else {
        IdxCa::full_null(PlSmallStr::EMPTY, len)
    }
}

fn repeat_by_primitive<T>(ca: &ChunkedArray<T>, by: &IdxCa) -> PolarsResult<ListChunked>
where
    T: PolarsNumericType,
{
    check_lengths(ca.len(), by.len())?;

    match (ca.len(), by.len()) {
        (left_len, right_len) if left_len == right_len => {
            Ok(arity::binary(ca, by, |arr, by| {
                let iter = arr.into_iter().zip(by).map(|(opt_v, opt_by)| {
                    opt_by.map(|by| std::iter::repeat(opt_v.copied()).take(*by as usize))
                });

                // SAFETY: length of iter is trusted.
                unsafe {
                    LargeListArray::from_iter_primitive_trusted_len(
                        iter,
                        T::get_dtype().to_arrow(CompatLevel::newest()),
                    )
                }
            }))
        },
        (_, 1) => {
            let by = new_by(by, ca.len());
            repeat_by_primitive(ca, &by)
        },
        (1, _) => {
            let new_array = ca.new_from_index(0, by.len());
            repeat_by_primitive(&new_array, by)
        },
        // we have already checked the length
        _ => unreachable!(),
    }
}

fn repeat_by_bool(ca: &BooleanChunked, by: &IdxCa) -> PolarsResult<ListChunked> {
    check_lengths(ca.len(), by.len())?;

    match (ca.len(), by.len()) {
        (left_len, right_len) if left_len == right_len => {
            Ok(arity::binary(ca, by, |arr, by| {
                let iter = arr.into_iter().zip(by).map(|(opt_v, opt_by)| {
                    opt_by.map(|by| std::iter::repeat(opt_v).take(*by as usize))
                });

                // SAFETY: length of iter is trusted.
                unsafe { LargeListArray::from_iter_bool_trusted_len(iter) }
            }))
        },
        (_, 1) => {
            let by = new_by(by, ca.len());
            repeat_by_bool(ca, &by)
        },
        (1, _) => {
            let new_array = ca.new_from_index(0, by.len());
            repeat_by_bool(&new_array, by)
        },
        // we have already checked the length
        _ => unreachable!(),
    }
}

fn repeat_by_binary(ca: &BinaryChunked, by: &IdxCa) -> PolarsResult<ListChunked> {
    check_lengths(ca.len(), by.len())?;

    match (ca.len(), by.len()) {
        (left_len, right_len) if left_len == right_len => {
            Ok(arity::binary(ca, by, |arr, by| {
                let iter = arr.into_iter().zip(by).map(|(opt_v, opt_by)| {
                    opt_by.map(|by| std::iter::repeat(opt_v).take(*by as usize))
                });

                // SAFETY: length of iter is trusted.
                unsafe { LargeListArray::from_iter_binary_trusted_len(iter, ca.len()) }
            }))
        },
        (_, 1) => {
            let by = new_by(by, ca.len());
            repeat_by_binary(ca, &by)
        },
        (1, _) => {
            let new_array = ca.new_from_index(0, by.len());
            repeat_by_binary(&new_array, by)
        },
        // we have already checked the length
        _ => unreachable!(),
    }
}

fn repeat_by_list(ca: &ListChunked, by: &IdxCa) -> PolarsResult<ListChunked> {
    check_lengths(ca.len(), by.len())?;

    match (ca.len(), by.len()) {
        (left_len, right_len) if left_len == right_len => Ok(arity::binary(ca, by, |arr, by| {
            let iter = arr.into_iter().zip(by).map(|(opt_v, opt_by)| {
                opt_by.map(|by| std::iter::repeat(opt_v).take(*by as usize))
            });
            unsafe {
                LargeListArray::from_iter_nested_trusted_len(
                    iter,
                    ca.dtype().to_physical().to_arrow(CompatLevel::newest()),
                )
                .unwrap()
            }
        })),
        (_, 1) => {
            let by = new_by(by, ca.len());
            repeat_by_list(ca, &by)
        },
        (1, _) => {
            let new_array = ca.new_from_index(0, by.len());
            repeat_by_list(&new_array, by)
        },
        // we have already checked the length
        _ => unreachable!(),
    }
}

#[cfg(feature = "dtype-struct")]
fn repeat_by_struct(ca: &StructChunked, by: &IdxCa) -> PolarsResult<ListChunked> {
    check_lengths(ca.len(), by.len())?;

    match (ca.len(), by.len()) {
        (left_len, right_len) if left_len == right_len => Ok(arity::binary(ca, by, |arr, by| {
            let iter = arr.into_iter().zip(by).map(|(opt_v, opt_by)| {
                opt_by.map(|by| std::iter::repeat(opt_v).take(*by as usize))
            });
            unsafe {
                LargeListArray::from_iter_nested_trusted_len(
                    iter,
                    ca.dtype().to_physical().to_arrow(CompatLevel::newest()),
                )
                .unwrap()
            }
        })),
        (_, 1) => {
            let by = new_by(by, ca.len());
            repeat_by_struct(ca, &by)
        },
        (1, _) => {
            let new_array = ca.new_from_index(0, by.len());
            repeat_by_struct(&new_array, by)
        },
        // we have already checked the length
        _ => unreachable!(),
    }
}

pub fn repeat_by(s: &Series, by: &IdxCa) -> PolarsResult<ListChunked> {
    let s_phys = s.to_physical_repr();
    use DataType::*;
    let out = match s_phys.dtype() {
        Boolean => repeat_by_bool(s_phys.bool().unwrap(), by),
        String => {
            let ca = s_phys.str().unwrap();
            repeat_by_binary(&ca.as_binary(), by)
                .and_then(|ca| ca.apply_to_inner(&|s| unsafe { s.cast_unchecked(&String) }))
        },
        Binary => repeat_by_binary(s_phys.binary().unwrap(), by),
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s_phys.as_ref().as_ref().as_ref();
                repeat_by_primitive(ca, by)
            })
        },
        List(_) => repeat_by_list(s_phys.list().unwrap(), by),
        #[cfg(feature = "dtype-struct")]
        Struct(_) => repeat_by_struct(s_phys.struct_().unwrap(), by),
        _ => polars_bail!(opq = repeat_by, s.dtype()),
    };
    out.and_then(|ca| {
        let logical_type = s.dtype();
        ca.apply_to_inner(&|s| unsafe { s.from_physical_unchecked(logical_type) })
    })
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::*;

    use super::*;

    #[test]
    fn test_str_raw() {
        let data = vec![
            Some("foo".to_owned()),
            Some("bar".to_owned()),
            None,
            Some("xxx".to_owned()),
        ];
        let s = Series::new("s".into(), data);
        println!("{:?}", s);
        let r = UInt32Chunked::new_vec("x".into(), vec![2, 0, 3, 4]);
        let out = repeat_by(&s, &r).unwrap();
        println!("{:?}", out);
    }

    #[test]
    fn test_i32() {
        let mut x = Vec::<Option<i32>>::new();
        x.push(None);
        let data = vec![
            Some(Series::new("".into(), vec![Some(4), None, Some(6)])),
            Some(Series::new("".into(), Vec::<i32>::new())),
            Some(Series::new("".into(), vec![3, 4, 5])),
            Some(Series::new("".into(), x)),
            None,
            Some(Series::new("".into(), vec![7])),
        ];
        let s = Series::new("s".into(), data);
        println!("{:?}", s);
        let r = UInt32Chunked::new_vec("x".into(), vec![2, 1, 0, 3, 3, 4]);
        let out = repeat_by(&s, &r).unwrap();
        println!("{:?}", out);
    }

    #[test]
    fn test_bool() {
        let mut x = Vec::<Option<bool>>::new();
        x.push(None);
        let data = vec![
            Some(Series::new("".into(), vec![Some(true), None, Some(true)])),
            Some(Series::new("".into(), Vec::<bool>::new())),
            Some(Series::new("".into(), vec![false, true, false])),
            Some(Series::new("".into(), x)),
            None,
            Some(Series::new("".into(), vec![false])),
        ];
        let s = Series::new("s".into(), data);
        println!("{:?}", s);
        let r = UInt32Chunked::new_vec("x".into(), vec![2, 1, 0, 3, 3, 4]);
        let out = repeat_by(&s, &r).unwrap();
        println!("{:?}", out);
    }

    #[test]
    fn test_str() {
        let mut x = Vec::<Option<String>>::new();
        x.push(None);
        let data = vec![
            Some(Series::new(
                "".into(),
                vec![Some("foo".to_owned()), None, Some("bar".to_owned())],
            )),
            Some(Series::new("".into(), Vec::<String>::new())),
            Some(Series::new(
                "".into(),
                vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
            )),
            Some(Series::new("".into(), x)),
            None,
            Some(Series::new("".into(), vec!["xxx".to_owned()])),
        ];
        let s = Series::new("s".into(), data);
        println!("{:?}", s);
        let r = UInt32Chunked::new_vec("x".into(), vec![2, 1, 0, 3, 3, 4]);
        let out = repeat_by(&s, &r).unwrap();
        println!("{:?}", out);
    }

    #[test]
    fn test_nested_i32() {
        let mut x = Vec::<Option<i32>>::new();
        x.push(None);
        let data = vec![
            Some(Series::new("".into(), vec![Some(4), None, Some(6)])),
            Some(Series::new("".into(), Vec::<i32>::new())),
            Some(Series::new("".into(), vec![3, 4, 5])),
            Some(Series::new("".into(), x)),
            None,
            Some(Series::new("".into(), vec![7])),
        ];
        let s = Series::new("s".into(), data);
        let ss = Series::new("ss".into(), vec![s.slice(0, 2).clone(), s.slice(2, 4)]);
        println!("{:?}", ss);
        let r = UInt32Chunked::new_vec("x".into(), vec![2, 1]);
        let out = repeat_by(&ss, &r).unwrap();
        println!("{:?}", out);
    }
}
