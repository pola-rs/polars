use crate::prelude::*;

#[cfg(feature = "random")]
use rand::prelude::SliceRandom;
#[cfg(feature = "random")]
use rand::thread_rng;

#[derive(Copy, Clone)]
pub enum RankMethod {
    Average,
    Min,
    Max,
    Dense,
    Ordinal,
    #[cfg(feature = "random")]
    Random,
}

pub(crate) fn rank(s: &Series, method: RankMethod) -> Series {
    if s.len() == 1 {
        return match method {
            Average => Series::new(s.name(), &[1.0]),
            _ => Series::new(s.name(), &[1u32]),
        };
    }
    // don't fully understand how to deal with nulls yet
    // impute with the maximum value possible.
    // todo! maybe add + 1 at the end on the null values.
    if s.has_validity() {
        // replace null values with the maximum value of that dtype
        let s = s.fill_null(FillNullStrategy::MaxBound).unwrap();
        return rank(&s, method);
    }

    // See: https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/stats.py#L8631-L8737

    let len = s.len();
    let null_count = s.null_count();
    let sort_idx_ca = s.argsort(false);
    let sort_idx = sort_idx_ca.downcast_iter().next().unwrap().values();

    let mut inv: AlignedVec<u32> = AlignedVec::with_capacity(len);
    // Safety:
    // Values will be filled next and there is only primitive data
    unsafe { inv.set_len(len) }
    let inv_values = inv.as_mut_slice();

    #[cfg(feature = "random")]
    let mut count = if let RankMethod::Ordinal | RankMethod::Random = method {
        1u32
    } else {
        0
    };

    #[cfg(not(feature = "random"))]
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

    use RankMethod::*;
    match method {
        Ordinal => {
            let inv_ca = UInt32Chunked::new_from_aligned_vec(s.name(), inv);
            inv_ca.into_series()
        }
        #[cfg(feature = "random")]
        Random => {
            // Safety:
            // in bounds
            let arr = unsafe { s.take_unchecked(&sort_idx_ca).unwrap() };
            let not_consecutive_same = (&arr.slice(1, len - 1))
                .not_equal(&arr.slice(0, len - 1))
                .rechunk();
            let obs = not_consecutive_same.downcast_iter().next().unwrap();

            // Collect slice indices for sort_idx which point to ties in the original series.
            let mut ties_indices = Vec::with_capacity(len + 1);
            let mut ties_index: usize = 0;

            ties_indices.push(ties_index);
            obs.iter().for_each(|b| {
                if let Some(b) = b {
                    ties_index += 1;
                    if b {
                        ties_indices.push(ties_index)
                    }
                }
            });
            // Close last slice (if there where nulls in the original series, they will always be in the last slice).
            ties_indices.push(len);

            let mut sort_idx = sort_idx.to_vec();

            let rng = &mut thread_rng();

            // Shuffle sort_idx positions which point to ties in the original series.
            for i in 0..(ties_indices.len() - 1) as usize {
                let ties_index_start = ties_indices[i];
                let ties_index_end = ties_indices[i + 1];
                if ties_index_end - ties_index_start > 1 {
                    sort_idx[ties_index_start..ties_index_end].shuffle(rng);
                }
            }

            // Recreate inv_ca (where ties are randomly shuffled compared with Ordinal).
            let mut count = 1u32;
            unsafe {
                sort_idx.iter().for_each(|&i| {
                    *inv_values.get_unchecked_mut(i as usize) = count;
                    count += 1;
                });
            }

            let inv_ca = UInt32Chunked::new_from_aligned_vec(s.name(), inv);
            inv_ca.into_series()
        }
        _ => {
            let inv_ca = UInt32Chunked::new_from_aligned_vec(s.name(), inv);
            // Safety:
            // in bounds
            let arr = unsafe { s.take_unchecked(&sort_idx_ca).unwrap() };
            let validity = arr.chunks()[0].validity().cloned();
            let not_consecutive_same = (&arr.slice(1, len - 1))
                .not_equal(&arr.slice(0, len - 1))
                .rechunk();
            // this obs is shorter than that of scipy stats, because we can just start the cumsum by 1
            // instead of 0
            let obs = not_consecutive_same.downcast_iter().next().unwrap();
            let mut dense = AlignedVec::with_capacity(len);

            // this offset save an offset on the whole column, what scipy does in:
            //
            // ```python
            //     if method == 'min':
            //         return count[dense - 1] + 1
            // ```
            let mut cumsum: u32 = if let RankMethod::Min = method { 0 } else { 1 };

            dense.push(cumsum);
            obs.values_iter().for_each(|b| {
                if b {
                    cumsum += 1;
                }
                dense.push(cumsum)
            });
            let arr =
                PrimitiveArray::from_data(DataType::UInt32.to_arrow(), dense.into(), validity);
            let dense: UInt32Chunked = (s.name(), arr).into();
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

            if null_count > 0 {
                obs.iter().for_each(|b| {
                    if let Some(b) = b {
                        cnt += 1;
                        if b {
                            count.push(cnt)
                        }
                    }
                });
            } else {
                obs.values_iter().for_each(|b| {
                    cnt += 1;
                    if b {
                        count.push(cnt)
                    }
                });
            }

            count.push((len - null_count) as u32);
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
                        .cast(&DataType::Float32)
                        .unwrap();
                    let b = unsafe { count.take_unchecked((&(dense - 1)).into()) }
                        .cast(&DataType::Float32)
                        .unwrap()
                        + 1.0;
                    (&a + &b) * 0.5
                }
                #[cfg(feature = "random")]
                Dense | Ordinal | Random => unimplemented!(),
                #[cfg(not(feature = "random"))]
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

        #[cfg(feature = "random")]
        {
            let out = rank(&s, RankMethod::Random)
                .u32()?
                .into_no_null_iter()
                .collect::<Vec<_>>();
            assert_eq!(out[0], 2);
            assert_eq!(out[6], 1);
            assert_eq!(out[1] + out[3] + out[4], 12);
            assert_eq!(out[2] + out[5], 13);
            assert_ne!(out[1], out[3]);
            assert_ne!(out[1], out[4]);
            assert_ne!(out[3], out[4]);
        }

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

        let s = Series::new(
            "a",
            &[Some(1), Some(2), Some(3), Some(2), None, None, Some(0)],
        );

        let out = rank(&s, RankMethod::Average)
            .f32()?
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(
            out,
            &[
                Some(2.0f32),
                Some(3.5),
                Some(5.0),
                Some(3.5),
                Some(6.5),
                Some(6.5),
                Some(1.0)
            ]
        );
        let s = Series::new(
            "a",
            &[
                Some(5),
                Some(6),
                Some(4),
                None,
                Some(78),
                Some(4),
                Some(2),
                Some(8),
            ],
        );
        let out = rank(&s, RankMethod::Max)
            .u32()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[4, 5, 3, 8, 7, 3, 1, 6]);

        Ok(())
    }
}
