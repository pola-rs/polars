use super::*;

pub(super) fn flatten_df(df: &DataFrame) -> impl Iterator<Item = DataFrame> + '_ {
    df.iter_chunks_physical().flat_map(|chunk| {
        let df = DataFrame::new_no_checks(
            df.iter()
                .zip(chunk.into_arrays())
                .map(|(s, arr)| {
                    // Safety:
                    // datatypes are correct
                    let mut out = unsafe {
                        Series::from_chunks_and_dtype_unchecked(s.name(), vec![arr], s.dtype())
                    };
                    out.set_sorted_flag(s.is_sorted_flag());
                    out
                })
                .collect(),
        );
        if df.height() == 0 {
            None
        } else {
            Some(df)
        }
    })
}

pub fn flatten_series(s: &Series) -> Vec<Series> {
    let name = s.name();
    let dtype = s.dtype();
    unsafe {
        s.chunks()
            .iter()
            .map(|arr| Series::from_chunks_and_dtype_unchecked(name, vec![arr.clone()], dtype))
            .collect()
    }
}

pub(crate) fn cap_and_offsets<I>(v: &[Vec<I>]) -> (usize, Vec<usize>) {
    let cap = v.iter().map(|v| v.len()).sum::<usize>();
    let offsets = v
        .iter()
        .scan(0_usize, |acc, v| {
            let out = *acc;
            *acc += v.len();
            Some(out)
        })
        .collect::<Vec<_>>();
    (cap, offsets)
}
