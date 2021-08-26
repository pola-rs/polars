use crate::prelude::*;

#[derive(Copy, Clone)]
pub enum RankMethod {
    Dense,
    Ordinal,
    Min,
    Max,
    Average,
}

pub(crate) fn rank(s: &Series, method: RankMethod) -> Series {
    // See: https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/stats.py#L8631-L8737

    let len = s.len();
    let sort_idx_ca = s.argsort(false);
    let sort_idx = sort_idx_ca.downcast_iter().next().unwrap().values();

    let mut inv: AlignedVec<u32> = AlignedVec::with_capacity(len);
    // Safety:
    // Values will be filled next and there is only primitive data
    unsafe { inv.set_len(len) }
    let inv_values = inv.as_mut_slice();

    let mut count = if let RankMethod::Ordinal = method {
        1u32
    } else {
        0
    };

    // Safety:
    // we are in bounds
    unsafe {
        sort_idx.iter().for_each(|&i| {
            *inv_values.get_unchecked_mut(i as usize) = count;
            count += 1;
        });
    }
    let inv_ca = UInt32Chunked::new_from_aligned_vec(s.name(), inv);

    use RankMethod::*;
    match method {
        Ordinal => inv_ca.into_series(),
        _ => {
            // Safety:
            // in bounds
            let arr = unsafe { s.take_unchecked(&sort_idx_ca).unwrap() };
            let is_consecutive_same = (&arr.slice(1, len - 1))
                .neq(&arr.slice(0, len - 1))
                .rechunk();
            // this obs is shorter than that of scipy stats, because we can just start the cumsum by 1
            // instead of 0
            let obs = is_consecutive_same.downcast_iter().next().unwrap();
            let mut dense = AlignedVec::with_capacity(len);

            // this offset save an offset on the whole column, what scipy does in:
            //
            // ```python
            //     if method == 'min':
            //         return count[dense - 1] + 1
            // ```
            let mut cumsum = if let RankMethod::Min = method { 0 } else { 1 };

            dense.push(cumsum);
            obs.iter().for_each(|opt_value| {
                if opt_value.unwrap() {
                    cumsum += 1;
                }
                dense.push(cumsum)
            });
            let dense = UInt32Chunked::new_from_aligned_vec(s.name(), dense);
            // Safety:
            // in bounds
            let dense = unsafe { dense.take_unchecked((&inv_ca).into()) };

            if let RankMethod::Dense = method {
                return dense.into_series();
            }

            let bitmap = obs.values();
            let cap = bitmap.len() - bitmap.null_count();
            let mut count = AlignedVec::with_capacity(cap + 1);
            let mut cnt = 0u32;
            count.push(cnt);

            obs.iter().for_each(|opt_value| {
                cnt += 1;
                if opt_value.unwrap() {
                    count.push(cnt)
                }
            });
            count.push(len as u32);
            let count = UInt32Chunked::new_from_aligned_vec(s.name(), count);

            match method {
                Max => {
                    // Safety:
                    // within bounds
                    unsafe { count.take_unchecked((&dense).into()).into_series() }
                }
                Min => {
                    // Safety:
                    // within bounds
                    unsafe { (count.take_unchecked((&dense).into()) + 1).into_series() }
                }
                Average => {
                    // Safety:
                    // in bounds
                    let a = unsafe { count.take_unchecked((&dense).into()) }
                        .cast::<Float32Type>()
                        .unwrap();
                    let iter = dense.cont_slice().unwrap().iter().map(|i| *i as usize - 1);
                    let b = unsafe { count.take_unchecked(iter.into()) }
                        .cast::<Float32Type>()
                        .unwrap()
                        + 1.0;
                    ((a + b) * 0.5).into_series()
                }
                Dense | Ordinal => unimplemented!(),
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rank() -> Result<()> {
        let s = Series::new("a", &[1, 2, 3, 2, 2, 3, 0]);

        let out = rank(&s, RankMethod::Ordinal)
            .u32()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[2, 3, 6, 4, 5, 7, 1]);

        let out = rank(&s, RankMethod::Dense)
            .u32()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[2, 3, 4, 3, 3, 4, 1]);

        let out = rank(&s, RankMethod::Max)
            .u32()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[2, 5, 7, 5, 5, 7, 1]);

        let out = rank(&s, RankMethod::Min)
            .u32()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[2, 3, 6, 3, 3, 6, 1]);

        let out = rank(&s, RankMethod::Average)
            .f32()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[2.0f32, 4.0, 6.5, 4.0, 4.0, 6.5, 1.0]);

        Ok(())
    }
}
