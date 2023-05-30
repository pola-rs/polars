use polars_utils::sync::SyncPtr;

use super::*;

pub fn flatten_df_iter(df: &DataFrame) -> impl Iterator<Item = DataFrame> + '_ {
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

pub fn flatten_par<T: Send + Sync + Copy, S: AsRef<[T]>>(bufs: &[S]) -> Vec<T> {
    let mut len = 0;
    let mut offsets = Vec::with_capacity(bufs.len());
    let bufs = bufs
        .iter()
        .map(|s| {
            offsets.push(len);
            let slice = s.as_ref();
            len += slice.len();
            slice
        })
        .collect::<Vec<_>>();
    flatten_par_impl(&bufs, len, offsets)
}

fn flatten_par_impl<T: Send + Sync + Copy>(
    bufs: &[&[T]],
    len: usize,
    offsets: Vec<usize>,
) -> Vec<T> {
    let mut out = Vec::with_capacity(len);
    let out_ptr = unsafe { SyncPtr::new(out.as_mut_ptr()) };

    POOL.install(|| {
        offsets.into_par_iter().enumerate().for_each(|(i, offset)| {
            let buf = bufs[i];
            let ptr: *mut T = out_ptr.get();
            unsafe {
                let dst = ptr.add(offset);
                let src = buf.as_ptr();
                std::ptr::copy_nonoverlapping(src, dst, buf.len())
            }
        })
    });
    unsafe {
        out.set_len(len);
    }
    out
}
