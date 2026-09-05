use polars_core::chunked_array::ops::row_encode::{
    encode_rows_unordered, encode_rows_vertical_par_unordered_broadcast_nulls,
};
use polars_core::series::BitRepr;
use polars_core::utils::split;
use polars_core::with_match_physical_float_polars_type;
use polars_utils::aliases::PlRandomState;
use polars_utils::hashing::DirtyHash;
use polars_utils::nulls::IsNull;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

use super::*;
use crate::series::SeriesSealed;

pub trait SeriesJoin: SeriesSealed + Sized {
    #[doc(hidden)]
    fn hash_join_left(
        &self,
        other: &Series,
        validate: JoinValidation,
        nulls_equal: bool,
    ) -> PolarsResult<LeftJoinIds> {
        let s_self = self.as_series();
        let (lhs, rhs) = (s_self.to_physical_repr(), other.to_physical_repr());
        validate.validate_probe(&lhs, &rhs, false, nulls_equal)?;

        let lhs_dtype = lhs.dtype();
        let rhs_dtype = rhs.dtype();

        use DataType as T;
        match lhs_dtype {
            T::String | T::Binary => {
                let lhs = lhs.cast(&T::Binary).unwrap();
                let rhs = rhs.cast(&T::Binary).unwrap();
                let lhs = lhs.binary().unwrap();
                let rhs = rhs.binary().unwrap();
                let (lhs, rhs, _, _) = prepare_binary::<BinaryType>(lhs, rhs, false);
                let lhs = lhs.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
                let rhs = rhs.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
                let build_null_count = other.null_count();
                hash_join_tuples_left(
                    lhs,
                    rhs,
                    None,
                    None,
                    validate,
                    nulls_equal,
                    build_null_count,
                )
            },
            T::BinaryOffset => {
                let lhs = lhs.binary_offset().unwrap();
                let rhs = rhs.binary_offset().unwrap();
                let (lhs, rhs, _, _) = prepare_binary::<BinaryOffsetType>(lhs, rhs, false);
                // Take slices so that vecs are not copied
                let lhs = lhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                let rhs = rhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                let build_null_count = other.null_count();
                hash_join_tuples_left(
                    lhs,
                    rhs,
                    None,
                    None,
                    validate,
                    nulls_equal,
                    build_null_count,
                )
            },
            T::List(_) => {
                let lhs = &encode_join_nested_key(lhs.into_owned(), nulls_equal)?;
                let rhs = &encode_join_nested_key(rhs.into_owned(), nulls_equal)?;
                lhs.hash_join_left(rhs, validate, nulls_equal)
            },
            #[cfg(feature = "dtype-array")]
            T::Array(_, _) => {
                let lhs = &encode_join_nested_key(lhs.into_owned(), nulls_equal)?;
                let rhs = &encode_join_nested_key(rhs.into_owned(), nulls_equal)?;
                lhs.hash_join_left(rhs, validate, nulls_equal)
            },
            #[cfg(feature = "dtype-struct")]
            T::Struct(_) => {
                let lhs = &encode_join_nested_key(lhs.into_owned(), nulls_equal)?;
                let rhs = &encode_join_nested_key(rhs.into_owned(), nulls_equal)?;
                lhs.hash_join_left(rhs, validate, nulls_equal)
            },
            x if x.is_float() => {
                with_match_physical_float_polars_type!(lhs.dtype(), |$T| {
                    let lhs: &ChunkedArray<$T> = lhs.as_ref().as_ref().as_ref();
                    let rhs: &ChunkedArray<$T> = rhs.as_ref().as_ref().as_ref();
                    num_group_join_left(lhs, rhs, validate, nulls_equal)
                })
            },
            _ => {
                let lhs = s_self.bit_repr();
                let rhs = other.bit_repr();

                let (Some(lhs), Some(rhs)) = (lhs, rhs) else {
                    polars_bail!(nyi = "Hash Left Join between {lhs_dtype} and {rhs_dtype}");
                };

                use BitRepr as B;
                match (lhs, rhs) {
                    (B::U8(lhs), B::U8(rhs)) => {
                        num_group_join_left(&lhs, &rhs, validate, nulls_equal)
                    },
                    (B::U16(lhs), B::U16(rhs)) => {
                        num_group_join_left(&lhs, &rhs, validate, nulls_equal)
                    },
                    (B::U32(lhs), B::U32(rhs)) => {
                        num_group_join_left(&lhs, &rhs, validate, nulls_equal)
                    },
                    (B::U64(lhs), B::U64(rhs)) => {
                        num_group_join_left(&lhs, &rhs, validate, nulls_equal)
                    },
                    #[cfg(feature = "dtype-u128")]
                    (B::U128(lhs), B::U128(rhs)) => {
                        num_group_join_left(&lhs, &rhs, validate, nulls_equal)
                    },
                    _ => {
                        polars_bail!(
                            nyi = "Mismatch bit repr Hash Left Join between {lhs_dtype} and {rhs_dtype}",
                        );
                    },
                }
            },
        }
    }

    #[cfg(feature = "semi_anti_join")]
    fn hash_join_semi_anti(
        &self,
        other: &Series,
        anti: bool,
        nulls_equal: bool,
    ) -> PolarsResult<Vec<IdxSize>> {
        let s_self = self.as_series();
        let (lhs, rhs) = (s_self.to_physical_repr(), other.to_physical_repr());

        let lhs_dtype = lhs.dtype();
        let rhs_dtype = rhs.dtype();

        use DataType as T;
        Ok(match lhs_dtype {
            T::String | T::Binary => {
                let lhs = lhs.cast(&T::Binary).unwrap();
                let rhs = rhs.cast(&T::Binary).unwrap();
                let lhs = lhs.binary().unwrap();
                let rhs = rhs.binary().unwrap();
                let (lhs, rhs, _, _) = prepare_binary::<BinaryType>(lhs, rhs, false);
                // Take slices so that vecs are not copied
                let lhs = lhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                let rhs = rhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                if anti {
                    hash_join_tuples_left_anti(lhs, rhs, nulls_equal)
                } else {
                    hash_join_tuples_left_semi(lhs, rhs, nulls_equal)
                }
            },
            T::BinaryOffset => {
                let lhs = lhs.binary_offset().unwrap();
                let rhs = rhs.binary_offset().unwrap();
                let (lhs, rhs, _, _) = prepare_binary::<BinaryOffsetType>(lhs, rhs, false);
                // Take slices so that vecs are not copied
                let lhs = lhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                let rhs = rhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                if anti {
                    hash_join_tuples_left_anti(lhs, rhs, nulls_equal)
                } else {
                    hash_join_tuples_left_semi(lhs, rhs, nulls_equal)
                }
            },
            T::List(_) => {
                let lhs = &encode_join_nested_key(lhs.into_owned(), nulls_equal)?;
                let rhs = &encode_join_nested_key(rhs.into_owned(), nulls_equal)?;
                lhs.hash_join_semi_anti(rhs, anti, nulls_equal)?
            },
            #[cfg(feature = "dtype-array")]
            T::Array(_, _) => {
                let lhs = &encode_join_nested_key(lhs.into_owned(), nulls_equal)?;
                let rhs = &encode_join_nested_key(rhs.into_owned(), nulls_equal)?;
                lhs.hash_join_semi_anti(rhs, anti, nulls_equal)?
            },
            #[cfg(feature = "dtype-struct")]
            T::Struct(_) => {
                let lhs = &encode_join_nested_key(lhs.into_owned(), nulls_equal)?;
                let rhs = &encode_join_nested_key(rhs.into_owned(), nulls_equal)?;
                lhs.hash_join_semi_anti(rhs, anti, nulls_equal)?
            },
            x if x.is_float() => {
                with_match_physical_float_polars_type!(lhs.dtype(), |$T| {
                    let lhs: &ChunkedArray<$T> = lhs.as_ref().as_ref().as_ref();
                    let rhs: &ChunkedArray<$T> = rhs.as_ref().as_ref().as_ref();
                    num_group_join_anti_semi(lhs, rhs, anti, nulls_equal)
                })
            },
            _ => {
                let lhs = s_self.bit_repr();
                let rhs = other.bit_repr();

                let (Some(lhs), Some(rhs)) = (lhs, rhs) else {
                    polars_bail!(nyi = "Hash Semi-Anti Join between {lhs_dtype} and {rhs_dtype}");
                };

                use BitRepr as B;
                match (lhs, rhs) {
                    (B::U8(lhs), B::U8(rhs)) => {
                        num_group_join_anti_semi(&lhs, &rhs, anti, nulls_equal)
                    },
                    (B::U16(lhs), B::U16(rhs)) => {
                        num_group_join_anti_semi(&lhs, &rhs, anti, nulls_equal)
                    },
                    (B::U32(lhs), B::U32(rhs)) => {
                        num_group_join_anti_semi(&lhs, &rhs, anti, nulls_equal)
                    },
                    (B::U64(lhs), B::U64(rhs)) => {
                        num_group_join_anti_semi(&lhs, &rhs, anti, nulls_equal)
                    },
                    #[cfg(feature = "dtype-u128")]
                    (B::U128(lhs), B::U128(rhs)) => {
                        num_group_join_anti_semi(&lhs, &rhs, anti, nulls_equal)
                    },
                    _ => {
                        polars_bail!(
                            nyi = "Mismatch bit repr Hash Semi-Anti Join between {lhs_dtype} and {rhs_dtype}",
                        );
                    },
                }
            },
        })
    }

    // returns the join tuples and whether or not the lhs tuples are sorted
    fn hash_join_inner(
        &self,
        other: &Series,
        validate: JoinValidation,
        nulls_equal: bool,
    ) -> PolarsResult<(InnerJoinIds, bool)> {
        let s_self = self.as_series();
        let (lhs, rhs) = (s_self.to_physical_repr(), other.to_physical_repr());
        validate.validate_probe(&lhs, &rhs, true, nulls_equal)?;

        let lhs_dtype = lhs.dtype();
        let rhs_dtype = rhs.dtype();

        use DataType as T;
        match lhs_dtype {
            T::String | T::Binary => {
                let lhs = lhs.cast(&T::Binary).unwrap();
                let rhs = rhs.cast(&T::Binary).unwrap();
                let lhs = lhs.binary().unwrap();
                let rhs = rhs.binary().unwrap();
                let (lhs, rhs, swapped, _) = prepare_binary::<BinaryType>(lhs, rhs, true);
                // Take slices so that vecs are not copied
                let lhs = lhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                let rhs = rhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                let build_null_count = if swapped {
                    s_self.null_count()
                } else {
                    other.null_count()
                };
                Ok((
                    hash_join_tuples_inner(
                        lhs,
                        rhs,
                        swapped,
                        validate,
                        nulls_equal,
                        build_null_count,
                    )?,
                    !swapped,
                ))
            },
            T::BinaryOffset => {
                let lhs = lhs.binary_offset().unwrap();
                let rhs = rhs.binary_offset()?;
                let (lhs, rhs, swapped, _) = prepare_binary::<BinaryOffsetType>(lhs, rhs, true);
                // Take slices so that vecs are not copied
                let lhs = lhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                let rhs = rhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                let build_null_count = if swapped {
                    s_self.null_count()
                } else {
                    other.null_count()
                };
                Ok((
                    hash_join_tuples_inner(
                        lhs,
                        rhs,
                        swapped,
                        validate,
                        nulls_equal,
                        build_null_count,
                    )?,
                    !swapped,
                ))
            },
            T::List(_) => {
                let lhs = &encode_join_nested_key(lhs.into_owned(), nulls_equal)?;
                let rhs = &encode_join_nested_key(rhs.into_owned(), nulls_equal)?;
                lhs.hash_join_inner(rhs, validate, nulls_equal)
            },
            #[cfg(feature = "dtype-array")]
            T::Array(_, _) => {
                let lhs = &encode_join_nested_key(lhs.into_owned(), nulls_equal)?;
                let rhs = &encode_join_nested_key(rhs.into_owned(), nulls_equal)?;
                lhs.hash_join_inner(rhs, validate, nulls_equal)
            },
            #[cfg(feature = "dtype-struct")]
            T::Struct(_) => {
                let lhs = &encode_join_nested_key(lhs.into_owned(), nulls_equal)?;
                let rhs = &encode_join_nested_key(rhs.into_owned(), nulls_equal)?;
                lhs.hash_join_inner(rhs, validate, nulls_equal)
            },
            x if x.is_float() => {
                with_match_physical_float_polars_type!(lhs.dtype(), |$T| {
                    let lhs: &ChunkedArray<$T> = lhs.as_ref().as_ref().as_ref();
                    let rhs: &ChunkedArray<$T> = rhs.as_ref().as_ref().as_ref();
                    group_join_inner::<$T>(lhs, rhs, validate, nulls_equal)
                })
            },
            _ => {
                let lhs = s_self.bit_repr();
                let rhs = other.bit_repr();

                let (Some(lhs), Some(rhs)) = (lhs, rhs) else {
                    polars_bail!(nyi = "Hash Inner Join between {lhs_dtype} and {rhs_dtype}");
                };

                use BitRepr as B;
                match (lhs, rhs) {
                    (B::U8(lhs), B::U8(rhs)) => {
                        group_join_inner::<UInt8Type>(&lhs, &rhs, validate, nulls_equal)
                    },
                    (B::U16(lhs), B::U16(rhs)) => {
                        group_join_inner::<UInt16Type>(&lhs, &rhs, validate, nulls_equal)
                    },
                    (B::U32(lhs), B::U32(rhs)) => {
                        group_join_inner::<UInt32Type>(&lhs, &rhs, validate, nulls_equal)
                    },
                    (B::U64(lhs), BitRepr::U64(rhs)) => {
                        group_join_inner::<UInt64Type>(&lhs, &rhs, validate, nulls_equal)
                    },
                    #[cfg(feature = "dtype-u128")]
                    (B::U128(lhs), BitRepr::U128(rhs)) => {
                        group_join_inner::<UInt128Type>(&lhs, &rhs, validate, nulls_equal)
                    },
                    _ => {
                        polars_bail!(
                            nyi = "Mismatch bit repr Hash Inner Join between {lhs_dtype} and {rhs_dtype}"
                        );
                    },
                }
            },
        }
    }

    fn hash_join_outer(
        &self,
        other: &Series,
        validate: JoinValidation,
        nulls_equal: bool,
    ) -> PolarsResult<(PlPrimitiveArray<IdxSize>, PlPrimitiveArray<IdxSize>)> {
        let s_self = self.as_series();
        let (lhs, rhs) = (s_self.to_physical_repr(), other.to_physical_repr());
        validate.validate_probe(&lhs, &rhs, true, nulls_equal)?;

        let lhs_dtype = lhs.dtype();
        let rhs_dtype = rhs.dtype();

        use DataType as T;
        match lhs_dtype {
            T::String | T::Binary => {
                let lhs = lhs.cast(&T::Binary).unwrap();
                let rhs = rhs.cast(&T::Binary).unwrap();
                let lhs = lhs.binary().unwrap();
                let rhs = rhs.binary().unwrap();
                let (lhs, rhs, swapped, _) = prepare_binary::<BinaryType>(lhs, rhs, true);
                // Take slices so that vecs are not copied
                let lhs = lhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                let rhs = rhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                hash_join_tuples_outer(lhs, rhs, swapped, validate, nulls_equal)
            },
            T::BinaryOffset => {
                let lhs = lhs.binary_offset().unwrap();
                let rhs = rhs.binary_offset()?;
                let (lhs, rhs, swapped, _) = prepare_binary::<BinaryOffsetType>(lhs, rhs, true);
                // Take slices so that vecs are not copied
                let lhs = lhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                let rhs = rhs.iter().map(|k| k.as_slice()).collect::<Vec<_>>();
                hash_join_tuples_outer(lhs, rhs, swapped, validate, nulls_equal)
            },
            T::List(_) => {
                let lhs = &encode_join_nested_key(lhs.into_owned(), nulls_equal)?;
                let rhs = &encode_join_nested_key(rhs.into_owned(), nulls_equal)?;
                lhs.hash_join_outer(rhs, validate, nulls_equal)
            },
            #[cfg(feature = "dtype-array")]
            T::Array(_, _) => {
                let lhs = &encode_join_nested_key(lhs.into_owned(), nulls_equal)?;
                let rhs = &encode_join_nested_key(rhs.into_owned(), nulls_equal)?;
                lhs.hash_join_outer(rhs, validate, nulls_equal)
            },
            #[cfg(feature = "dtype-struct")]
            T::Struct(_) => {
                let lhs = &encode_join_nested_key(lhs.into_owned(), nulls_equal)?;
                let rhs = &encode_join_nested_key(rhs.into_owned(), nulls_equal)?;
                lhs.hash_join_outer(rhs, validate, nulls_equal)
            },
            x if x.is_float() => {
                with_match_physical_float_polars_type!(lhs.dtype(), |$T| {
                    let lhs: &ChunkedArray<$T> = lhs.as_ref().as_ref().as_ref();
                    let rhs: &ChunkedArray<$T> = rhs.as_ref().as_ref().as_ref();
                    hash_join_outer(lhs, rhs, validate, nulls_equal)
                })
            },
            _ => {
                let (Some(lhs), Some(rhs)) = (s_self.bit_repr(), other.bit_repr()) else {
                    polars_bail!(nyi = "Hash Join Outer between {lhs_dtype} and {rhs_dtype}");
                };

                use BitRepr as B;
                match (lhs, rhs) {
                    (B::U8(lhs), B::U8(rhs)) => hash_join_outer(&lhs, &rhs, validate, nulls_equal),
                    (B::U16(lhs), B::U16(rhs)) => {
                        hash_join_outer(&lhs, &rhs, validate, nulls_equal)
                    },
                    (B::U32(lhs), B::U32(rhs)) => {
                        hash_join_outer(&lhs, &rhs, validate, nulls_equal)
                    },
                    (B::U64(lhs), B::U64(rhs)) => {
                        hash_join_outer(&lhs, &rhs, validate, nulls_equal)
                    },
                    #[cfg(feature = "dtype-u128")]
                    (B::U128(lhs), B::U128(rhs)) => {
                        hash_join_outer(&lhs, &rhs, validate, nulls_equal)
                    },
                    _ => {
                        polars_bail!(
                            nyi = "Mismatch bit repr Hash Join Outer between {lhs_dtype} and {rhs_dtype}"
                        );
                    },
                }
            },
        }
    }
}

impl SeriesJoin for Series {}

/// The values of every chunk of every array, as the slices they are. Every array must have been
/// written out flat with [`ChunkedArray::flatten_mut`] first.
fn chunks_as_slices<T>(splitted: &[ChunkedArray<T>]) -> Vec<&[T::Native]>
where
    T: PolarsNumericType,
{
    splitted
        .iter()
        .flat_map(|ca| {
            ca.as_flat()
                .expect("the chunks were written out flat first")
                .flat_chunks()
                .map(|arr| arr.as_slice())
        })
        .collect()
}

/// Splits `ca` for `n_threads`, writing out every chunk that is scalar. TODO(polars-array-scalar):
/// the hash join reads the keys as slices, so a repeated key is written out rather than hashed once.
fn split_flat<T: PolarsDataType>(ca: &ChunkedArray<T>, n_threads: usize) -> Vec<ChunkedArray<T>> {
    let mut splitted = split(ca, n_threads);
    for ca in &mut splitted {
        ca.flatten_mut();
    }
    splitted
}

fn encode_join_nested_key(s: Series, nulls_equal: bool) -> PolarsResult<Series> {
    let by = [s.into_column()];
    let encoded = if nulls_equal {
        encode_rows_unordered(&by)?
    } else {
        encode_rows_vertical_par_unordered_broadcast_nulls(&by)?
    };
    Ok(encoded.into_series())
}

fn get_arrays<T: PolarsDataType>(cas: &[ChunkedArray<T>]) -> Vec<&T::Array> {
    cas.iter().flat_map(|arr| arr.downcast_iter()).collect()
}

fn group_join_inner<T>(
    left: &ChunkedArray<T>,
    right: &ChunkedArray<T>,
    validate: JoinValidation,
    nulls_equal: bool,
) -> PolarsResult<(InnerJoinIds, bool)>
where
    T: PolarsNumericType,
    for<'a> &'a T::Array: IntoIterator<Item = Option<T::Physical<'a>>>,
    T::Native: Send + Sync + Copy + TotalHash + TotalEq + DirtyHash + IsNull + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Send + Sync + Copy + Hash + Eq + DirtyHash + IsNull,
{
    let n_threads = RAYON.current_num_threads();
    let (a, b, swapped) = det_hash_prone_order!(left, right);

    match (left.null_count(), right.null_count()) {
        (0, 0) => {
            // Neither side has a null, so the keys are the values themselves.
            let splitted_a = split_flat(a, n_threads);
            let splitted_b = split_flat(b, n_threads);
            Ok((
                hash_join_tuples_inner(
                    chunks_as_slices(&splitted_a),
                    chunks_as_slices(&splitted_b),
                    swapped,
                    validate,
                    nulls_equal,
                    0,
                )?,
                !swapped,
            ))
        },
        _ => {
            let splitted_a = split(a, n_threads);
            let splitted_b = split(b, n_threads);
            let build_null_count = if swapped {
                left.null_count()
            } else {
                right.null_count()
            };
            Ok((
                hash_join_tuples_inner(
                    get_arrays(&splitted_a),
                    get_arrays(&splitted_b),
                    swapped,
                    validate,
                    nulls_equal,
                    build_null_count,
                )?,
                !swapped,
            ))
        },
    }
}

#[cfg(feature = "chunked_ids")]
fn create_mappings(
    chunks_left: &[PlArrayRef],
    chunks_right: &[PlArrayRef],
    left_len: usize,
    right_len: usize,
) -> (Option<Vec<ChunkId>>, Option<Vec<ChunkId>>) {
    let mapping_left = || {
        if chunks_left.len() > 1 {
            Some(create_chunked_index_mapping(chunks_left, left_len))
        } else {
            None
        }
    };

    let mapping_right = || {
        if chunks_right.len() > 1 {
            Some(create_chunked_index_mapping(chunks_right, right_len))
        } else {
            None
        }
    };

    RAYON.join(mapping_left, mapping_right)
}

#[cfg(not(feature = "chunked_ids"))]
fn create_mappings(
    _chunks_left: &[PlArrayRef],
    _chunks_right: &[PlArrayRef],
    _left_len: usize,
    _right_len: usize,
) -> (Option<Vec<ChunkId>>, Option<Vec<ChunkId>>) {
    (None, None)
}

fn num_group_join_left<T>(
    left: &ChunkedArray<T>,
    right: &ChunkedArray<T>,
    validate: JoinValidation,
    nulls_equal: bool,
) -> PolarsResult<LeftJoinIds>
where
    T: PolarsNumericType,
    T::Native: TotalHash + TotalEq + DirtyHash + IsNull + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Send + Sync + Copy + Hash + Eq + DirtyHash + IsNull,
    T::Native: DirtyHash + Copy + ToTotalOrd,
    <Option<T::Native> as ToTotalOrd>::TotalOrdItem: Send + Sync + DirtyHash,
{
    let n_threads = RAYON.current_num_threads();
    match (
        left.null_count(),
        right.null_count(),
        left.chunks().len(),
        right.chunks().len(),
    ) {
        (0, 0, 1, 1) => {
            let splitted_a = split_flat(left, n_threads);
            let splitted_b = split_flat(right, n_threads);
            let keys_a = chunks_as_slices(&splitted_a);
            let keys_b = chunks_as_slices(&splitted_b);
            hash_join_tuples_left(keys_a, keys_b, None, None, validate, nulls_equal, 0)
        },
        (0, 0, _, _) => {
            let splitted_a = split_flat(left, n_threads);
            let splitted_b = split_flat(right, n_threads);
            let keys_a = chunks_as_slices(&splitted_a);
            let keys_b = chunks_as_slices(&splitted_b);

            let (mapping_left, mapping_right) =
                create_mappings(left.chunks(), right.chunks(), left.len(), right.len());
            hash_join_tuples_left(
                keys_a,
                keys_b,
                mapping_left.as_deref(),
                mapping_right.as_deref(),
                validate,
                nulls_equal,
                0,
            )
        },
        _ => {
            let splitted_a = split(left, n_threads);
            let splitted_b = split(right, n_threads);
            let keys_a = get_arrays(&splitted_a);
            let keys_b = get_arrays(&splitted_b);
            let (mapping_left, mapping_right) =
                create_mappings(left.chunks(), right.chunks(), left.len(), right.len());
            let build_null_count = right.null_count();
            hash_join_tuples_left(
                keys_a,
                keys_b,
                mapping_left.as_deref(),
                mapping_right.as_deref(),
                validate,
                nulls_equal,
                build_null_count,
            )
        },
    }
}

fn hash_join_outer<T>(
    ca_in: &ChunkedArray<T>,
    other: &ChunkedArray<T>,
    validate: JoinValidation,
    nulls_equal: bool,
) -> PolarsResult<(PlPrimitiveArray<IdxSize>, PlPrimitiveArray<IdxSize>)>
where
    T: PolarsNumericType,
    T::Native: TotalHash + TotalEq + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Send + Sync + Copy + Hash + Eq + IsNull,
{
    let (a, b, swapped) = det_hash_prone_order!(ca_in, other);

    let n_partitions = _set_partition_size();

    match (a.null_count(), b.null_count()) {
        (0, 0) => {
            let splitted_a = split_flat(a, n_partitions);
            let splitted_b = split_flat(b, n_partitions);
            let iters_a = chunks_as_slices(&splitted_a);
            let iters_b = chunks_as_slices(&splitted_b);
            hash_join_tuples_outer(iters_a, iters_b, swapped, validate, nulls_equal)
        },
        _ => {
            let splitted_a = split(a, n_partitions);
            let splitted_b = split(b, n_partitions);
            let iters_a = splitted_a
                .iter()
                .flat_map(|ca| ca.downcast_iter().map(|arr| arr.iter()))
                .collect::<Vec<_>>();
            let iters_b = splitted_b
                .iter()
                .flat_map(|ca| ca.downcast_iter().map(|arr| arr.iter()))
                .collect::<Vec<_>>();
            hash_join_tuples_outer(iters_a, iters_b, swapped, validate, nulls_equal)
        },
    }
}

pub(crate) fn prepare_binary<'a, T>(
    ca: &'a ChunkedArray<T>,
    other: &'a ChunkedArray<T>,
    // In inner join and outer join, the shortest relation will be used to create a hash table.
    // In left join, always use the right side to create.
    build_shortest_table: bool,
) -> (
    Vec<Vec<BytesHash<'a>>>,
    Vec<Vec<BytesHash<'a>>>,
    bool,
    PlRandomState,
)
where
    T: PolarsDataType,
    for<'b> <T::Array as StaticArray>::ValueT<'b>: AsRef<[u8]>,
{
    let (a, b, swapped) = if build_shortest_table {
        det_hash_prone_order!(ca, other)
    } else {
        (ca, other, false)
    };
    let hb = PlRandomState::default();
    let bh_a = a.to_bytes_hashes(true, hb.clone());
    let bh_b = b.to_bytes_hashes(true, hb.clone());

    (bh_a, bh_b, swapped, hb)
}

#[cfg(feature = "semi_anti_join")]
fn num_group_join_anti_semi<T>(
    left: &ChunkedArray<T>,
    right: &ChunkedArray<T>,
    anti: bool,
    nulls_equal: bool,
) -> Vec<IdxSize>
where
    T: PolarsNumericType,
    T::Native: TotalHash + TotalEq + DirtyHash + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Send + Sync + Copy + Hash + Eq + DirtyHash + IsNull,
    <Option<T::Native> as ToTotalOrd>::TotalOrdItem: Send + Sync + DirtyHash + IsNull,
{
    let n_threads = RAYON.current_num_threads();
    let splitted_a = split(left, n_threads);
    let splitted_b = split(right, n_threads);
    match (
        left.null_count(),
        right.null_count(),
        left.chunks().len(),
        right.chunks().len(),
    ) {
        (0, 0, 1, 1) => {
            let keys_a = chunks_as_slices(&splitted_a);
            let keys_b = chunks_as_slices(&splitted_b);
            if anti {
                hash_join_tuples_left_anti(keys_a, keys_b, nulls_equal)
            } else {
                hash_join_tuples_left_semi(keys_a, keys_b, nulls_equal)
            }
        },
        (0, 0, _, _) => {
            let keys_a = chunks_as_slices(&splitted_a);
            let keys_b = chunks_as_slices(&splitted_b);
            if anti {
                hash_join_tuples_left_anti(keys_a, keys_b, nulls_equal)
            } else {
                hash_join_tuples_left_semi(keys_a, keys_b, nulls_equal)
            }
        },
        _ => {
            let keys_a = get_arrays(&splitted_a);
            let keys_b = get_arrays(&splitted_b);
            if anti {
                hash_join_tuples_left_anti(keys_a, keys_b, nulls_equal)
            } else {
                hash_join_tuples_left_semi(keys_a, keys_b, nulls_equal)
            }
        },
    }
}
