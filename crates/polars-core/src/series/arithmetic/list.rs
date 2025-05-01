//! Allow arithmetic operations for ListChunked.
//! use polars_error::{feature_gated, PolarsResult};

use polars_error::{PolarsResult, feature_gated};

use super::list_utils::NumericOp;
use super::{IntoSeries, ListChunked, ListType, NumOpsDispatchInner, Series};

impl NumOpsDispatchInner for ListType {
    fn add_to(lhs: &ListChunked, rhs: &Series) -> PolarsResult<Series> {
        NumericListOp::add().execute(&lhs.clone().into_series(), rhs)
    }

    fn subtract(lhs: &ListChunked, rhs: &Series) -> PolarsResult<Series> {
        NumericListOp::sub().execute(&lhs.clone().into_series(), rhs)
    }

    fn multiply(lhs: &ListChunked, rhs: &Series) -> PolarsResult<Series> {
        NumericListOp::mul().execute(&lhs.clone().into_series(), rhs)
    }

    fn divide(lhs: &ListChunked, rhs: &Series) -> PolarsResult<Series> {
        NumericListOp::div().execute(&lhs.clone().into_series(), rhs)
    }

    fn remainder(lhs: &ListChunked, rhs: &Series) -> PolarsResult<Series> {
        NumericListOp::rem().execute(&lhs.clone().into_series(), rhs)
    }
}

#[derive(Clone)]
pub struct NumericListOp(NumericOp);

impl NumericListOp {
    pub fn add() -> Self {
        Self(NumericOp::Add)
    }

    pub fn sub() -> Self {
        Self(NumericOp::Sub)
    }

    pub fn mul() -> Self {
        Self(NumericOp::Mul)
    }

    pub fn div() -> Self {
        Self(NumericOp::Div)
    }

    pub fn rem() -> Self {
        Self(NumericOp::Rem)
    }

    pub fn floor_div() -> Self {
        Self(NumericOp::FloorDiv)
    }
}

impl NumericListOp {
    #[cfg_attr(not(feature = "list_arithmetic"), allow(unused))]
    pub fn execute(&self, lhs: &Series, rhs: &Series) -> PolarsResult<Series> {
        feature_gated!("list_arithmetic", {
            use std::borrow::Cow;

            use either::Either;

            // `trim_to_normalized_offsets` ensures we don't perform excessive
            // memory allocation / compute on memory regions that have been
            // sliced out.
            let lhs = lhs
                .trim_lists_to_normalized_offsets()
                .map_or(Cow::Borrowed(lhs), Cow::Owned);
            let rhs = rhs
                .trim_lists_to_normalized_offsets()
                .map_or(Cow::Borrowed(rhs), Cow::Owned);

            let lhs = lhs.rechunk();
            let rhs = rhs.rechunk();

            let binary_op_exec = match ListNumericOpHelper::try_new(
                self.clone(),
                lhs.name().clone(),
                lhs.dtype(),
                rhs.dtype(),
                lhs.len(),
                rhs.len(),
                {
                    let (a, b) = lhs.list_offsets_and_validities_recursive();
                    debug_assert!(a.iter().all(|x| *x.first() as usize == 0));
                    (a, b, lhs.clone())
                },
                {
                    let (a, b) = rhs.list_offsets_and_validities_recursive();
                    debug_assert!(a.iter().all(|x| *x.first() as usize == 0));
                    (a, b, rhs.clone())
                },
                lhs.rechunk_validity(),
                rhs.rechunk_validity(),
            )? {
                Either::Left(v) => v,
                Either::Right(ca) => return Ok(ca.into_series()),
            };

            Ok(binary_op_exec.finish()?.into_series())
        })
    }
}

#[cfg(feature = "list_arithmetic")]
use inner::ListNumericOpHelper;

#[cfg(feature = "list_arithmetic")]
mod inner {
    use arrow::bitmap::Bitmap;
    use arrow::compute::utils::combine_validities_and;
    use arrow::offset::OffsetsBuffer;
    use either::Either;
    use list_utils::with_match_pl_num_arith;
    use num_traits::Zero;
    use polars_compute::arithmetic::pl_num::PlNumArithmetic;
    use polars_utils::float::IsFloat;

    use super::super::list_utils::{BinaryOpApplyType, Broadcast, NumericOp};
    use super::super::*;

    /// Utility to perform a binary operation between the primitive values of
    /// 2 columns, where at least one of the columns is a `ListChunked` type.
    pub(super) struct ListNumericOpHelper {
        op: NumericListOp,
        output_name: PlSmallStr,
        op_apply_type: BinaryOpApplyType,
        broadcast: Broadcast,
        output_dtype: DataType,
        output_primitive_dtype: DataType,
        output_len: usize,
        /// Outer validity of the result, we always materialize this to reduce the
        /// amount of code paths we need.
        outer_validity: Bitmap,
        // The series are stored as they are used for list broadcasting.
        data_lhs: (Vec<OffsetsBuffer<i64>>, Vec<Option<Bitmap>>, Series),
        data_rhs: (Vec<OffsetsBuffer<i64>>, Vec<Option<Bitmap>>, Series),
        list_to_prim_lhs: Option<(Box<dyn Array>, usize)>,
        swapped: bool,
    }

    /// This lets us separate some logic into `new()` to reduce the amount of
    /// monomorphized code.
    impl ListNumericOpHelper {
        /// Checks that:
        /// * Dtypes are compatible:
        ///   * list<->primitive | primitive<->list
        ///   * list<->list both contain primitives (e.g. List<Int8>)
        /// * Primitive dtypes match
        /// * Lengths are compatible:
        ///   * 1<->n | n<->1
        ///   * n<->n
        /// * Both sides have at least 1 non-NULL outer row.
        ///
        /// Does not check:
        /// * Whether the offsets are aligned for list<->list, this will be checked during execution.
        ///
        /// This returns an `Either` which may contain the final result to simplify
        /// the implementation.
        #[allow(clippy::too_many_arguments)]
        pub(super) fn try_new(
            op: NumericListOp,
            output_name: PlSmallStr,
            dtype_lhs: &DataType,
            dtype_rhs: &DataType,
            len_lhs: usize,
            len_rhs: usize,
            data_lhs: (Vec<OffsetsBuffer<i64>>, Vec<Option<Bitmap>>, Series),
            data_rhs: (Vec<OffsetsBuffer<i64>>, Vec<Option<Bitmap>>, Series),
            validity_lhs: Option<Bitmap>,
            validity_rhs: Option<Bitmap>,
        ) -> PolarsResult<Either<Self, ListChunked>> {
            let prim_dtype_lhs = dtype_lhs.leaf_dtype();
            let prim_dtype_rhs = dtype_rhs.leaf_dtype();

            let output_primitive_dtype =
                op.0.try_get_leaf_supertype(prim_dtype_lhs, prim_dtype_rhs)?;

            fn is_list_type_at_all_levels(dtype: &DataType) -> bool {
                match dtype {
                    DataType::List(inner) => is_list_type_at_all_levels(inner),
                    dt if dt.is_supported_list_arithmetic_input() => true,
                    _ => false,
                }
            }

            let op_err_msg = |err_reason: &str| {
                polars_err!(
                    InvalidOperation:
                    "cannot {} columns: {}: (left: {}, right: {})",
                    op.0.name(), err_reason, dtype_lhs, dtype_rhs,
                )
            };

            let ensure_list_type_at_all_levels = |dtype: &DataType| {
                if !is_list_type_at_all_levels(dtype) {
                    Err(op_err_msg("dtype was not list on all nesting levels"))
                } else {
                    Ok(())
                }
            };

            let (op_apply_type, output_dtype) = match (dtype_lhs, dtype_rhs) {
                (l @ DataType::List(a), r @ DataType::List(b)) => {
                    // `get_arithmetic_field()` in the DSL checks this, but we also have to check here because if a user
                    // directly adds 2 series together it bypasses the DSL.
                    // This is currently duplicated code and should be replaced one day with an assert after Series ops get
                    // checked properly.
                    if ![a, b]
                        .into_iter()
                        .all(|x| x.is_supported_list_arithmetic_input())
                    {
                        polars_bail!(
                            InvalidOperation:
                            "cannot {} two list columns with non-numeric inner types: (left: {}, right: {})",
                            op.0.name(), l, r,
                        );
                    }
                    (BinaryOpApplyType::ListToList, l)
                },
                (list_dtype @ DataType::List(_), x) if x.is_supported_list_arithmetic_input() => {
                    ensure_list_type_at_all_levels(list_dtype)?;
                    (BinaryOpApplyType::ListToPrimitive, list_dtype)
                },
                (x, list_dtype @ DataType::List(_)) if x.is_supported_list_arithmetic_input() => {
                    ensure_list_type_at_all_levels(list_dtype)?;
                    (BinaryOpApplyType::PrimitiveToList, list_dtype)
                },
                (l, r) => polars_bail!(
                    InvalidOperation:
                    "{} operation not supported for dtypes: {} != {}",
                    op.0.name(), l, r,
                ),
            };

            let output_dtype = output_dtype.cast_leaf(output_primitive_dtype.clone());

            let (broadcast, output_len) = match (len_lhs, len_rhs) {
                (l, r) if l == r => (Broadcast::NoBroadcast, l),
                (1, v) => (Broadcast::Left, v),
                (v, 1) => (Broadcast::Right, v),
                (l, r) => polars_bail!(
                    ShapeMismatch:
                    "cannot {} two columns of differing lengths: {} != {}",
                    op.0.name(), l, r
                ),
            };

            let DataType::List(output_inner_dtype) = &output_dtype else {
                unreachable!()
            };

            // # NULL semantics
            // * [[1, 2]] (List[List[Int64]]) + NULL (Int64) => [[NULL, NULL]]
            //   * Essentially as if the NULL primitive was added to every primitive in the row of the list column.
            // * NULL (List[Int64]) + 1   (Int64)       => NULL
            // * NULL (List[Int64]) + [1] (List[Int64]) => NULL

            if output_len == 0
                || (matches!(
                    &op_apply_type,
                    BinaryOpApplyType::ListToList | BinaryOpApplyType::ListToPrimitive
                ) && validity_lhs.as_ref().is_some_and(|x| x.set_bits() == 0))
                || (matches!(
                    &op_apply_type,
                    BinaryOpApplyType::ListToList | BinaryOpApplyType::PrimitiveToList
                ) && validity_rhs.as_ref().is_some_and(|x| x.set_bits() == 0))
            {
                return Ok(Either::Right(ListChunked::full_null_with_dtype(
                    output_name.clone(),
                    output_len,
                    output_inner_dtype.as_ref(),
                )));
            }

            // At this point:
            // * All unit length list columns have a valid outer value.

            // The outer validity is just the validity of any non-broadcasting lists.
            let outer_validity = match (&op_apply_type, &broadcast, validity_lhs, validity_rhs) {
                // Both lists with same length, we combine the validity.
                (BinaryOpApplyType::ListToList, Broadcast::NoBroadcast, l, r) => {
                    combine_validities_and(l.as_ref(), r.as_ref())
                },
                // Match all other combinations that have non-broadcasting lists.
                (
                    BinaryOpApplyType::ListToList | BinaryOpApplyType::ListToPrimitive,
                    Broadcast::NoBroadcast | Broadcast::Right,
                    v,
                    _,
                )
                | (
                    BinaryOpApplyType::ListToList | BinaryOpApplyType::PrimitiveToList,
                    Broadcast::NoBroadcast | Broadcast::Left,
                    _,
                    v,
                ) => v,
                _ => None,
            }
            .unwrap_or_else(|| Bitmap::new_with_value(true, output_len));

            Ok(Either::Left(Self {
                op,
                output_name,
                op_apply_type,
                broadcast,
                output_dtype: output_dtype.clone(),
                output_primitive_dtype,
                output_len,
                outer_validity,
                data_lhs,
                data_rhs,
                list_to_prim_lhs: None,
                swapped: false,
            }))
        }

        pub(super) fn finish(mut self) -> PolarsResult<ListChunked> {
            // We have physical codepaths for a subset of the possible combinations of broadcasting and
            // column types. The remaining combinations are handled by dispatching to the physical
            // codepaths after operand swapping and/or materialized broadcasting.
            //
            // # Physical impl table
            // Legend
            // * |  N  | // impl "N"
            // * | [N] | // dispatches to impl "N"
            //
            //                  |  L  |  N  |  R  | // Broadcast (L)eft, (N)oBroadcast, (R)ight
            // ListToList       | [1] |  0  |  1  |
            // ListToPrimitive  | [2] |  2  |  3  | // list broadcasting just materializes and dispatches to NoBroadcast
            // PrimitiveToList  | [3] | [2] | [2] |

            self.swapped = true;

            match (&self.op_apply_type, &self.broadcast) {
                (BinaryOpApplyType::ListToList, Broadcast::NoBroadcast)
                | (BinaryOpApplyType::ListToList, Broadcast::Right)
                | (BinaryOpApplyType::ListToPrimitive, Broadcast::NoBroadcast)
                | (BinaryOpApplyType::ListToPrimitive, Broadcast::Right) => {
                    self.swapped = false;
                    self._finish_impl_dispatch()
                },
                (BinaryOpApplyType::ListToList, Broadcast::Left) => {
                    self.broadcast = Broadcast::Right;

                    std::mem::swap(&mut self.data_lhs, &mut self.data_rhs);
                    self._finish_impl_dispatch()
                },
                (BinaryOpApplyType::ListToPrimitive, Broadcast::Left) => {
                    self.list_to_prim_lhs
                        .replace(Self::materialize_broadcasted_list(
                            &mut self.data_lhs,
                            self.output_len,
                            &self.output_primitive_dtype,
                        ));

                    self.broadcast = Broadcast::NoBroadcast;

                    // This does not swap! We are just dispatching to `NoBroadcast`
                    // after materializing the broadcasted list array.
                    self.swapped = false;
                    self._finish_impl_dispatch()
                },
                (BinaryOpApplyType::PrimitiveToList, Broadcast::NoBroadcast) => {
                    self.op_apply_type = BinaryOpApplyType::ListToPrimitive;

                    std::mem::swap(&mut self.data_lhs, &mut self.data_rhs);
                    self._finish_impl_dispatch()
                },
                (BinaryOpApplyType::PrimitiveToList, Broadcast::Right) => {
                    // We materialize the list columns with `new_from_index`, as otherwise we'd have to
                    // implement logic that broadcasts the offsets and validities across multiple levels
                    // of nesting. But we will re-use the materialized memory to store the result.

                    self.list_to_prim_lhs
                        .replace(Self::materialize_broadcasted_list(
                            &mut self.data_rhs,
                            self.output_len,
                            &self.output_primitive_dtype,
                        ));

                    self.op_apply_type = BinaryOpApplyType::ListToPrimitive;
                    self.broadcast = Broadcast::NoBroadcast;

                    std::mem::swap(&mut self.data_lhs, &mut self.data_rhs);
                    self._finish_impl_dispatch()
                },
                (BinaryOpApplyType::PrimitiveToList, Broadcast::Left) => {
                    self.op_apply_type = BinaryOpApplyType::ListToPrimitive;
                    self.broadcast = Broadcast::Right;

                    std::mem::swap(&mut self.data_lhs, &mut self.data_rhs);
                    self._finish_impl_dispatch()
                },
            }
        }

        fn _finish_impl_dispatch(&mut self) -> PolarsResult<ListChunked> {
            let output_dtype = self.output_dtype.clone();
            let output_len = self.output_len;

            let prim_lhs = self
                .data_lhs
                .2
                .get_leaf_array()
                .cast(&self.output_primitive_dtype)?
                .rechunk();
            let prim_rhs = self
                .data_rhs
                .2
                .get_leaf_array()
                .cast(&self.output_primitive_dtype)?
                .rechunk();

            debug_assert_eq!(prim_lhs.dtype(), prim_rhs.dtype());
            let prim_dtype = prim_lhs.dtype();
            debug_assert_eq!(prim_dtype, &self.output_primitive_dtype);

            // Safety: Leaf dtypes have been checked to be numeric by `try_new()`
            let out = with_match_physical_numeric_polars_type!(&prim_dtype, |$T| {
                self._finish_impl::<$T>(prim_lhs, prim_rhs)
            })?;

            debug_assert_eq!(out.dtype(), &output_dtype);
            assert_eq!(out.len(), output_len);

            Ok(out)
        }

        /// Internal use only - contains physical impls.
        fn _finish_impl<T: PolarsNumericType>(
            &mut self,
            prim_s_lhs: Series,
            prim_s_rhs: Series,
        ) -> PolarsResult<ListChunked>
        where
            T::Native: PlNumArithmetic,
            PrimitiveArray<T::Native>:
                polars_compute::comparisons::TotalEqKernel<Scalar = T::Native>,
            T::Native: Zero + IsFloat,
        {
            #[inline(never)]
            fn check_mismatch_pos(
                mismatch_pos: usize,
                offsets_lhs: &OffsetsBuffer<i64>,
                offsets_rhs: &OffsetsBuffer<i64>,
            ) -> PolarsResult<()> {
                if mismatch_pos < offsets_lhs.len_proxy() {
                    // RHS could be broadcasted
                    let len_r = offsets_rhs.length_at(if offsets_rhs.len_proxy() == 1 {
                        0
                    } else {
                        mismatch_pos
                    });
                    polars_bail!(
                        ShapeMismatch:
                        "list lengths differed at index {}: {} != {}",
                        mismatch_pos,
                        offsets_lhs.length_at(mismatch_pos), len_r
                    )
                }
                Ok(())
            }

            let mut arr_lhs = {
                let ca: &ChunkedArray<T> = prim_s_lhs.as_ref().as_ref();
                assert_eq!(ca.chunks().len(), 1);
                ca.downcast_get(0).unwrap().clone()
            };

            let mut arr_rhs = {
                let ca: &ChunkedArray<T> = prim_s_rhs.as_ref().as_ref();
                assert_eq!(ca.chunks().len(), 1);
                ca.downcast_get(0).unwrap().clone()
            };

            match (&self.op_apply_type, &self.broadcast) {
                // We skip for this because it dispatches to `ArithmeticKernel`, which handles the
                // validities for us.
                (BinaryOpApplyType::ListToPrimitive, Broadcast::Right) => {},
                _ if self.list_to_prim_lhs.is_none() => {
                    self.op.0.prepare_numeric_op_side_validities::<T>(
                        &mut arr_lhs,
                        &mut arr_rhs,
                        self.swapped,
                    )
                },
                (BinaryOpApplyType::ListToPrimitive, Broadcast::NoBroadcast) => {
                    // `self.list_to_prim_lhs` is `Some(_)`, this is handled later.
                },
                _ => unreachable!(),
            }

            //
            // General notes
            // * Lists can be:
            //   * Sliced, in which case the primitive/leaf array needs to be indexed starting from an
            //     offset instead of 0.
            //   * Masked, in which case the masked rows are permitted to have non-matching widths.
            //

            let out = match (&self.op_apply_type, &self.broadcast) {
                (BinaryOpApplyType::ListToList, Broadcast::NoBroadcast) => {
                    let offsets_lhs = &self.data_lhs.0[0];
                    let offsets_rhs = &self.data_rhs.0[0];

                    assert_eq!(offsets_lhs.len_proxy(), offsets_rhs.len_proxy());

                    // Output primitive (and optional validity) are aligned to the LHS input.
                    let n_values = arr_lhs.len();
                    let mut out_vec: Vec<T::Native> = Vec::with_capacity(n_values);
                    let out_ptr: *mut T::Native = out_vec.as_mut_ptr();

                    // Counter that stops being incremented at the first row position with mismatching
                    // list lengths.
                    let mut mismatch_pos = 0;

                    with_match_pl_num_arith!(&self.op.0, self.swapped, |$OP| {
                        for (i, ((lhs_start, lhs_len), (rhs_start, rhs_len))) in offsets_lhs
                            .offset_and_length_iter()
                            .zip(offsets_rhs.offset_and_length_iter())
                            .enumerate()
                        {
                            if
                                (mismatch_pos == i)
                                & (
                                    (lhs_len == rhs_len)
                                    | unsafe { !self.outer_validity.get_bit_unchecked(i) }
                                )
                            {
                                mismatch_pos += 1;
                            }

                            // Both sides are lists, we restrict the index to the min length to avoid
                            // OOB memory access.
                            let len: usize = lhs_len.min(rhs_len);

                            for i in 0..len {
                                let l_idx = i + lhs_start;
                                let r_idx = i + rhs_start;

                                let l = unsafe { arr_lhs.value_unchecked(l_idx) };
                                let r = unsafe { arr_rhs.value_unchecked(r_idx) };
                                let v = $OP(l, r);

                                unsafe { out_ptr.add(l_idx).write(v) };
                            }
                        }
                    });

                    check_mismatch_pos(mismatch_pos, offsets_lhs, offsets_rhs)?;

                    unsafe { out_vec.set_len(n_values) };

                    /// Reduce monomorphization
                    #[inline(never)]
                    fn combine_validities_list_to_list_no_broadcast(
                        offsets_lhs: &OffsetsBuffer<i64>,
                        offsets_rhs: &OffsetsBuffer<i64>,
                        validity_lhs: Option<&Bitmap>,
                        validity_rhs: Option<&Bitmap>,
                        len_lhs: usize,
                    ) -> Option<Bitmap> {
                        match (validity_lhs, validity_rhs) {
                            (Some(l), Some(r)) => Some((l.clone().make_mut(), r)),
                            (Some(v), None) => return Some(v.clone()),
                            (None, Some(v)) => {
                                Some((Bitmap::new_with_value(true, len_lhs).make_mut(), v))
                            },
                            (None, None) => None,
                        }
                        .map(|(mut validity_out, validity_rhs)| {
                            for ((lhs_start, lhs_len), (rhs_start, rhs_len)) in offsets_lhs
                                .offset_and_length_iter()
                                .zip(offsets_rhs.offset_and_length_iter())
                            {
                                let len: usize = lhs_len.min(rhs_len);

                                for i in 0..len {
                                    let l_idx = i + lhs_start;
                                    let r_idx = i + rhs_start;

                                    let l_valid = unsafe { validity_out.get_unchecked(l_idx) };
                                    let r_valid = unsafe { validity_rhs.get_bit_unchecked(r_idx) };
                                    let is_valid = l_valid & r_valid;

                                    // Size and alignment of validity vec are based on LHS.
                                    unsafe { validity_out.set_unchecked(l_idx, is_valid) };
                                }
                            }

                            validity_out.freeze()
                        })
                    }

                    let leaf_validity = combine_validities_list_to_list_no_broadcast(
                        offsets_lhs,
                        offsets_rhs,
                        arr_lhs.validity(),
                        arr_rhs.validity(),
                        arr_lhs.len(),
                    );

                    let arr =
                        PrimitiveArray::<T::Native>::from_vec(out_vec).with_validity(leaf_validity);

                    let (offsets, validities, _) = std::mem::take(&mut self.data_lhs);
                    assert_eq!(offsets.len(), 1);

                    self.finish_offsets_and_validities(Box::new(arr), offsets, validities)
                },
                (BinaryOpApplyType::ListToList, Broadcast::Right) => {
                    let offsets_lhs = &self.data_lhs.0[0];
                    let offsets_rhs = &self.data_rhs.0[0];

                    // Output primitive (and optional validity) are aligned to the LHS input.
                    let n_values = arr_lhs.len();
                    let mut out_vec: Vec<T::Native> = Vec::with_capacity(n_values);
                    let out_ptr: *mut T::Native = out_vec.as_mut_ptr();

                    assert_eq!(offsets_rhs.len_proxy(), 1);
                    let rhs_start = *offsets_rhs.first() as usize;
                    let width = offsets_rhs.range() as usize;

                    let mut mismatch_pos = 0;

                    with_match_pl_num_arith!(&self.op.0, self.swapped, |$OP| {
                        for (i, (lhs_start, lhs_len)) in offsets_lhs.offset_and_length_iter().enumerate() {
                            if ((lhs_len == width) & (mismatch_pos == i))
                                | unsafe { !self.outer_validity.get_bit_unchecked(i) }
                            {
                                mismatch_pos += 1;
                            }

                            let len: usize = lhs_len.min(width);

                            for i in 0..len {
                                let l_idx = i + lhs_start;
                                let r_idx = i + rhs_start;

                                let l = unsafe { arr_lhs.value_unchecked(l_idx) };
                                let r = unsafe { arr_rhs.value_unchecked(r_idx) };
                                let v = $OP(l, r);

                                unsafe {
                                    out_ptr.add(l_idx).write(v);
                                }
                            }
                        }
                    });

                    check_mismatch_pos(mismatch_pos, offsets_lhs, offsets_rhs)?;

                    unsafe { out_vec.set_len(n_values) };

                    #[inline(never)]
                    fn combine_validities_list_to_list_broadcast_right(
                        offsets_lhs: &OffsetsBuffer<i64>,
                        validity_lhs: Option<&Bitmap>,
                        validity_rhs: Option<&Bitmap>,
                        len_lhs: usize,
                        width: usize,
                        rhs_start: usize,
                    ) -> Option<Bitmap> {
                        match (validity_lhs, validity_rhs) {
                            (Some(l), Some(r)) => Some((l.clone().make_mut(), r)),
                            (Some(v), None) => return Some(v.clone()),
                            (None, Some(v)) => {
                                Some((Bitmap::new_with_value(true, len_lhs).make_mut(), v))
                            },
                            (None, None) => None,
                        }
                        .map(|(mut validity_out, validity_rhs)| {
                            for (lhs_start, lhs_len) in offsets_lhs.offset_and_length_iter() {
                                let len: usize = lhs_len.min(width);

                                for i in 0..len {
                                    let l_idx = i + lhs_start;
                                    let r_idx = i + rhs_start;

                                    let l_valid = unsafe { validity_out.get_unchecked(l_idx) };
                                    let r_valid = unsafe { validity_rhs.get_bit_unchecked(r_idx) };
                                    let is_valid = l_valid & r_valid;

                                    // Size and alignment of validity vec are based on LHS.
                                    unsafe { validity_out.set_unchecked(l_idx, is_valid) };
                                }
                            }

                            validity_out.freeze()
                        })
                    }

                    let leaf_validity = combine_validities_list_to_list_broadcast_right(
                        offsets_lhs,
                        arr_lhs.validity(),
                        arr_rhs.validity(),
                        arr_lhs.len(),
                        width,
                        rhs_start,
                    );

                    let arr =
                        PrimitiveArray::<T::Native>::from_vec(out_vec).with_validity(leaf_validity);

                    let (offsets, validities, _) = std::mem::take(&mut self.data_lhs);
                    assert_eq!(offsets.len(), 1);

                    self.finish_offsets_and_validities(Box::new(arr), offsets, validities)
                },
                (BinaryOpApplyType::ListToPrimitive, Broadcast::NoBroadcast)
                    if self.list_to_prim_lhs.is_none() =>
                {
                    let offsets_lhs = self.data_lhs.0.as_slice();

                    // Notes
                    // * Primitive indexing starts from 0
                    // * Output is aligned to LHS array

                    let n_values = arr_lhs.len();
                    let mut out_vec = Vec::<T::Native>::with_capacity(n_values);
                    let out_ptr = out_vec.as_mut_ptr();

                    with_match_pl_num_arith!(&self.op.0, self.swapped, |$OP| {
                        for (i, l_range) in OffsetsBuffer::<i64>::leaf_ranges_iter(offsets_lhs).enumerate()
                        {
                            let r = unsafe { arr_rhs.value_unchecked(i) };
                            for l_idx in l_range {
                                unsafe {
                                    let l = arr_lhs.value_unchecked(l_idx);
                                    let v = $OP(l, r);
                                    out_ptr.add(l_idx).write(v);
                                }
                            }
                        }
                    });

                    unsafe { out_vec.set_len(n_values) }

                    let leaf_validity = combine_validities_list_to_primitive_no_broadcast(
                        offsets_lhs,
                        arr_lhs.validity(),
                        arr_rhs.validity(),
                        arr_lhs.len(),
                    );

                    let arr =
                        PrimitiveArray::<T::Native>::from_vec(out_vec).with_validity(leaf_validity);

                    let (offsets, validities, _) = std::mem::take(&mut self.data_lhs);
                    self.finish_offsets_and_validities(Box::new(arr), offsets, validities)
                },
                // If we are dispatched here, it means that the LHS array is a unique allocation created
                // after a unit-length list column was broadcasted, so this codepath mutably stores the
                // results back into the LHS array to save memory.
                (BinaryOpApplyType::ListToPrimitive, Broadcast::NoBroadcast) => {
                    let offsets_lhs = self.data_lhs.0.as_slice();

                    let (mut arr, n_values) = Option::take(&mut self.list_to_prim_lhs).unwrap();
                    let arr = arr
                        .as_any_mut()
                        .downcast_mut::<PrimitiveArray<T::Native>>()
                        .unwrap();
                    let mut arr_lhs = std::mem::take(arr);

                    self.op.0.prepare_numeric_op_side_validities::<T>(
                        &mut arr_lhs,
                        &mut arr_rhs,
                        self.swapped,
                    );

                    let arr_lhs_mut_slice = arr_lhs.get_mut_values().unwrap();
                    assert_eq!(arr_lhs_mut_slice.len(), n_values);

                    with_match_pl_num_arith!(&self.op.0, self.swapped, |$OP| {
                        for (i, l_range) in OffsetsBuffer::<i64>::leaf_ranges_iter(offsets_lhs).enumerate()
                        {
                            let r = unsafe { arr_rhs.value_unchecked(i) };
                            for l_idx in l_range {
                                unsafe {
                                    let l = arr_lhs_mut_slice.get_unchecked_mut(l_idx);
                                    *l = $OP(*l, r);
                                }
                            }
                        }
                    });

                    let leaf_validity = combine_validities_list_to_primitive_no_broadcast(
                        offsets_lhs,
                        arr_lhs.validity(),
                        arr_rhs.validity(),
                        arr_lhs.len(),
                    );

                    let arr = arr_lhs.with_validity(leaf_validity);

                    let (offsets, validities, _) = std::mem::take(&mut self.data_lhs);
                    self.finish_offsets_and_validities(Box::new(arr), offsets, validities)
                },
                (BinaryOpApplyType::ListToPrimitive, Broadcast::Right) => {
                    assert_eq!(arr_rhs.len(), 1);

                    let Some(r) = (unsafe { arr_rhs.get_unchecked(0) }) else {
                        // RHS is single primitive NULL, create the result by setting the leaf validity to all-NULL.
                        let (offsets, validities, _) = std::mem::take(&mut self.data_lhs);
                        return Ok(self.finish_offsets_and_validities(
                            Box::new(
                                arr_lhs.clone().with_validity(Some(Bitmap::new_with_value(
                                    false,
                                    arr_lhs.len(),
                                ))),
                            ),
                            offsets,
                            validities,
                        ));
                    };

                    let arr = self
                        .op
                        .0
                        .apply_array_to_scalar::<T>(arr_lhs, r, self.swapped);
                    let (offsets, validities, _) = std::mem::take(&mut self.data_lhs);

                    self.finish_offsets_and_validities(Box::new(arr), offsets, validities)
                },
                v @ (BinaryOpApplyType::PrimitiveToList, Broadcast::Right)
                | v @ (BinaryOpApplyType::ListToList, Broadcast::Left)
                | v @ (BinaryOpApplyType::ListToPrimitive, Broadcast::Left)
                | v @ (BinaryOpApplyType::PrimitiveToList, Broadcast::Left)
                | v @ (BinaryOpApplyType::PrimitiveToList, Broadcast::NoBroadcast) => {
                    if cfg!(debug_assertions) {
                        panic!("operation was not re-written: {:?}", v)
                    } else {
                        unreachable!()
                    }
                },
            };

            Ok(out)
        }

        /// Construct the result `ListChunked` from the leaf array and the offsets/validities of every
        /// level.
        fn finish_offsets_and_validities(
            &mut self,
            leaf_array: Box<dyn Array>,
            offsets: Vec<OffsetsBuffer<i64>>,
            validities: Vec<Option<Bitmap>>,
        ) -> ListChunked {
            assert!(!offsets.is_empty());
            assert_eq!(offsets.len(), validities.len());
            let mut results = leaf_array;

            let mut iter = offsets.into_iter().zip(validities).rev();

            while iter.len() > 1 {
                let (offsets, validity) = iter.next().unwrap();
                let dtype = LargeListArray::default_datatype(results.dtype().clone());
                results = Box::new(LargeListArray::new(dtype, offsets, results, validity));
            }

            // The combined outer validity is pre-computed during `try_new()`
            let (offsets, _) = iter.next().unwrap();
            let validity = std::mem::take(&mut self.outer_validity);
            let dtype = LargeListArray::default_datatype(results.dtype().clone());
            let results = LargeListArray::new(dtype, offsets, results, Some(validity));

            ListChunked::with_chunk(std::mem::take(&mut self.output_name), results)
        }

        fn materialize_broadcasted_list(
            side_data: &mut (Vec<OffsetsBuffer<i64>>, Vec<Option<Bitmap>>, Series),
            output_len: usize,
            output_primitive_dtype: &DataType,
        ) -> (Box<dyn Array>, usize) {
            let s = &side_data.2;
            assert_eq!(s.len(), 1);

            let expected_n_values = {
                let offsets = s.list_offsets_and_validities_recursive().0;
                output_len * OffsetsBuffer::<i64>::leaf_full_start_end(&offsets).len()
            };

            let ca = s.list().unwrap();
            // Remember to cast the leaf primitives to the supertype.
            let ca = ca
                .cast(&ca.dtype().cast_leaf(output_primitive_dtype.clone()))
                .unwrap();
            assert!(output_len > 1); // In case there is a fast-path that doesn't give us owned data.
            let ca = ca.new_from_index(0, output_len).rechunk();

            let s = ca.into_series();

            *side_data = {
                let (a, b) = s.list_offsets_and_validities_recursive();
                // `Series::default()`: This field in the tuple is no longer used.
                (a, b, Series::default())
            };

            let n_values = OffsetsBuffer::<i64>::leaf_full_start_end(&side_data.0).len();
            assert_eq!(n_values, expected_n_values);

            let mut s = s.get_leaf_array();
            let v = unsafe { s.chunks_mut() };

            assert_eq!(v.len(), 1);
            (v.swap_remove(0), n_values)
        }
    }

    /// Used in 2 places, so it's outside here.
    #[inline(never)]
    fn combine_validities_list_to_primitive_no_broadcast(
        offsets_lhs: &[OffsetsBuffer<i64>],
        validity_lhs: Option<&Bitmap>,
        validity_rhs: Option<&Bitmap>,
        len_lhs: usize,
    ) -> Option<Bitmap> {
        match (validity_lhs, validity_rhs) {
            (Some(l), Some(r)) => Some((l.clone().make_mut(), r)),
            (Some(v), None) => return Some(v.clone()),
            // Materialize a full-true validity to re-use the codepath, as we still
            // need to spread the bits from the RHS to the correct positions.
            (None, Some(v)) => Some((Bitmap::new_with_value(true, len_lhs).make_mut(), v)),
            (None, None) => None,
        }
        .map(|(mut validity_out, validity_rhs)| {
            for (i, l_range) in OffsetsBuffer::<i64>::leaf_ranges_iter(offsets_lhs).enumerate() {
                let r_valid = unsafe { validity_rhs.get_bit_unchecked(i) };
                for l_idx in l_range {
                    let l_valid = unsafe { validity_out.get_unchecked(l_idx) };
                    let is_valid = l_valid & r_valid;

                    // Size and alignment of validity vec are based on LHS.
                    unsafe { validity_out.set_unchecked(l_idx, is_valid) };
                }
            }

            validity_out.freeze()
        })
    }
}
