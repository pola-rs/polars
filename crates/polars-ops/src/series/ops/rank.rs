use arrow::array::BooleanArray;
use arrow::compute::concatenate::concatenate_validities;
use polars_core::prelude::*;
use rand::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::SeriesSealed;

#[derive(Copy, Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RankMethod {
    Average,
    Min,
    Max,
    Dense,
    Ordinal,
    #[cfg(feature = "random")]
    Random,
}

// We might want to add a `nulls_last` or `null_behavior` field.
#[derive(Copy, Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RankOptions {
    pub method: RankMethod,
    pub descending: bool,
}

impl Default for RankOptions {
    fn default() -> Self {
        Self {
            method: RankMethod::Dense,
            descending: false,
        }
    }
}

#[cfg(feature = "random")]
fn get_random_seed() -> u64 {
    let mut rng = SmallRng::from_entropy();

    rng.next_u64()
}

unsafe fn rank_impl<F: FnMut(&mut [IdxSize])>(idxs: &IdxCa, neq: &BooleanArray, mut flush_ties: F) {
    let mut ties_indices = Vec::with_capacity(128);
    let mut idx_it = idxs.downcast_iter().flat_map(|arr| arr.values_iter());
    let Some(first_idx) = idx_it.next() else {
        return;
    };
    ties_indices.push(*first_idx);

    for (eq_idx, idx) in idx_it.enumerate() {
        if neq.value_unchecked(eq_idx) {
            flush_ties(&mut ties_indices);
            ties_indices.clear()
        }

        ties_indices.push(*idx);
    }
    flush_ties(&mut ties_indices);
}

fn rank(s: &Series, method: RankMethod, descending: bool, seed: Option<u64>) -> Series {
    let len = s.len();
    let null_count = s.null_count();

    if null_count == len {
        let dt = match method {
            Average => DataType::Float64,
            _ => IDX_DTYPE,
        };
        return Series::full_null(s.name(), s.len(), &dt);
    }

    match len {
        1 => {
            return match method {
                Average => Series::new(s.name(), &[1.0f64]),
                _ => Series::new(s.name(), &[1 as IdxSize]),
            };
        },
        0 => {
            return match method {
                Average => Float64Chunked::from_slice(s.name(), &[]).into_series(),
                _ => IdxCa::from_slice(s.name(), &[]).into_series(),
            };
        },
        _ => {},
    }

    if null_count == len {
        return match method {
            Average => Float64Chunked::full_null(s.name(), len).into_series(),
            _ => IdxCa::full_null(s.name(), len).into_series(),
        };
    }

    let sort_idx_ca = s
        .arg_sort(SortOptions {
            descending,
            nulls_last: true,
            ..Default::default()
        })
        .slice(0, len - null_count);

    let chunk_refs: Vec<_> = s.chunks().iter().map(|c| &**c).collect();
    let validity = concatenate_validities(&chunk_refs);

    use RankMethod::*;
    if let Ordinal = method {
        let mut out = vec![0 as IdxSize; s.len()];
        let mut rank = 0;
        for arr in sort_idx_ca.downcast_iter() {
            for i in arr.values_iter() {
                out[*i as usize] = rank + 1;
                rank += 1;
            }
        }
        IdxCa::from_vec_validity(s.name(), out, validity).into_series()
    } else {
        let sorted_values = unsafe { s.take_unchecked(&sort_idx_ca) };
        let not_consecutive_same = sorted_values
            .slice(1, sorted_values.len() - 1)
            .not_equal(&sorted_values.slice(0, sorted_values.len() - 1))
            .unwrap()
            .rechunk();
        let neq = not_consecutive_same.downcast_iter().next().unwrap();

        let mut rank = 1;
        match method {
            #[cfg(feature = "random")]
            Random => unsafe {
                let mut rng = SmallRng::seed_from_u64(seed.unwrap_or_else(get_random_seed));
                let mut out = vec![0 as IdxSize; s.len()];
                rank_impl(&sort_idx_ca, neq, |ties| {
                    ties.shuffle(&mut rng);
                    for i in ties {
                        *out.get_unchecked_mut(*i as usize) = rank;
                        rank += 1;
                    }
                });
                IdxCa::from_vec_validity(s.name(), out, validity).into_series()
            },
            Average => unsafe {
                let mut out = vec![0.0; s.len()];
                rank_impl(&sort_idx_ca, neq, |ties| {
                    let first = rank;
                    rank += ties.len() as IdxSize;
                    let last = rank - 1;
                    let avg = 0.5 * (first as f64 + last as f64);
                    for i in ties {
                        *out.get_unchecked_mut(*i as usize) = avg;
                    }
                });
                Float64Chunked::from_vec_validity(s.name(), out, validity).into_series()
            },
            Min => unsafe {
                let mut out = vec![0 as IdxSize; s.len()];
                rank_impl(&sort_idx_ca, neq, |ties| {
                    for i in ties.iter() {
                        *out.get_unchecked_mut(*i as usize) = rank;
                    }
                    rank += ties.len() as IdxSize;
                });
                IdxCa::from_vec_validity(s.name(), out, validity).into_series()
            },
            Max => unsafe {
                let mut out = vec![0 as IdxSize; s.len()];
                rank_impl(&sort_idx_ca, neq, |ties| {
                    rank += ties.len() as IdxSize;
                    for i in ties {
                        *out.get_unchecked_mut(*i as usize) = rank - 1;
                    }
                });
                IdxCa::from_vec_validity(s.name(), out, validity).into_series()
            },
            Dense => unsafe {
                let mut out = vec![0 as IdxSize; s.len()];
                rank_impl(&sort_idx_ca, neq, |ties| {
                    for i in ties {
                        *out.get_unchecked_mut(*i as usize) = rank;
                    }
                    rank += 1;
                });
                IdxCa::from_vec_validity(s.name(), out, validity).into_series()
            },
            Ordinal => unreachable!(),
        }
    }
}

pub trait SeriesRank: SeriesSealed {
    fn rank(&self, options: RankOptions, seed: Option<u64>) -> Series {
        rank(self.as_series(), options.method, options.descending, seed)
    }
}

impl SeriesRank for Series {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rank() -> PolarsResult<()> {
        let s = Series::new("a", &[1, 2, 3, 2, 2, 3, 0]);

        let out = rank(&s, RankMethod::Ordinal, false, None)
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[2 as IdxSize, 3, 6, 4, 5, 7, 1]);

        #[cfg(feature = "random")]
        {
            let out = rank(&s, RankMethod::Random, false, None)
                .idx()?
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

        let out = rank(&s, RankMethod::Dense, false, None)
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[2, 3, 4, 3, 3, 4, 1]);

        let out = rank(&s, RankMethod::Max, false, None)
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[2, 5, 7, 5, 5, 7, 1]);

        let out = rank(&s, RankMethod::Min, false, None)
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[2, 3, 6, 3, 3, 6, 1]);

        let out = rank(&s, RankMethod::Average, false, None)
            .f64()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[2.0f64, 4.0, 6.5, 4.0, 4.0, 6.5, 1.0]);

        let s = Series::new(
            "a",
            &[Some(1), Some(2), Some(3), Some(2), None, None, Some(0)],
        );

        let out = rank(&s, RankMethod::Average, false, None)
            .f64()?
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(
            out,
            &[
                Some(2.0f64),
                Some(3.5),
                Some(5.0),
                Some(3.5),
                None,
                None,
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
        let out = rank(&s, RankMethod::Max, false, None)
            .idx()?
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            out,
            &[
                Some(4),
                Some(5),
                Some(3),
                None,
                Some(7),
                Some(3),
                Some(1),
                Some(6)
            ]
        );

        Ok(())
    }

    #[test]
    fn test_rank_all_null() -> PolarsResult<()> {
        let s = UInt32Chunked::new("", &[None, None, None]).into_series();
        let out = rank(&s, RankMethod::Average, false, None)
            .f64()?
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None]);
        let out = rank(&s, RankMethod::Dense, false, None)
            .idx()?
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None]);
        Ok(())
    }

    #[test]
    fn test_rank_empty() {
        let s = UInt32Chunked::from_slice("", &[]).into_series();
        let out = rank(&s, RankMethod::Average, false, None);
        assert_eq!(out.dtype(), &DataType::Float64);
        let out = rank(&s, RankMethod::Max, false, None);
        assert_eq!(out.dtype(), &IDX_DTYPE);
    }

    #[test]
    fn test_rank_reverse() -> PolarsResult<()> {
        let s = Series::new("", &[None, Some(1), Some(1), Some(5), None]);
        let out = rank(&s, RankMethod::Dense, true, None)
            .idx()?
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(2 as IdxSize), Some(2), Some(1), None]);

        Ok(())
    }
}
