use polars_error::{PolarsResult, feature_gated};

use super::list_utils::NumericOp;
use super::{ArrayChunked, FixedSizeListType, IntoSeries, NumOpsDispatchInner, Series};

impl NumOpsDispatchInner for FixedSizeListType {
    fn add_to(lhs: &ArrayChunked, rhs: &Series) -> PolarsResult<Series> {
        NumericFixedSizeListOp::add().execute(&lhs.clone().into_series(), rhs)
    }

    fn subtract(lhs: &ArrayChunked, rhs: &Series) -> PolarsResult<Series> {
        NumericFixedSizeListOp::sub().execute(&lhs.clone().into_series(), rhs)
    }

    fn multiply(lhs: &ArrayChunked, rhs: &Series) -> PolarsResult<Series> {
        NumericFixedSizeListOp::mul().execute(&lhs.clone().into_series(), rhs)
    }

    fn divide(lhs: &ArrayChunked, rhs: &Series) -> PolarsResult<Series> {
        NumericFixedSizeListOp::div().execute(&lhs.clone().into_series(), rhs)
    }

    fn remainder(lhs: &ArrayChunked, rhs: &Series) -> PolarsResult<Series> {
        NumericFixedSizeListOp::rem().execute(&lhs.clone().into_series(), rhs)
    }
}

#[derive(Clone)]
pub struct NumericFixedSizeListOp(NumericOp);

impl NumericFixedSizeListOp {
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

impl NumericFixedSizeListOp {
    #[cfg_attr(not(feature = "array_arithmetic"), allow(unused))]
    pub fn execute(&self, lhs: &Series, rhs: &Series) -> PolarsResult<Series> {
        feature_gated!("array_arithmetic", {
            NumericFixedSizeListOpHelper::execute_op(self.clone(), lhs.rechunk(), rhs.rechunk())
                .map(|x| x.into_series())
        })
    }
}

#[cfg(feature = "array_arithmetic")]
use inner::NumericFixedSizeListOpHelper;

#[cfg(feature = "array_arithmetic")]
mod inner {
    use arrow::bitmap::{Bitmap, BitmapBuilder};
    use arrow::compute::utils::combine_validities_and;
    use fixed_size_list::NumericFixedSizeListOp;
    use list_utils::with_match_pl_num_arith;
    use num_traits::Zero;
    use polars_compute::arithmetic::pl_num::PlNumArithmetic;
    use polars_utils::float::IsFloat;

    use super::super::list_utils::{BinaryOpApplyType, Broadcast, NumericOp};
    use super::super::*;

    /// Utility to perform a binary operation between the primitive values of
    /// 2 columns, where at least one of the columns is a `ArrayChunked` type.
    pub(super) struct NumericFixedSizeListOpHelper {
        op: NumericFixedSizeListOp,
        output_name: PlSmallStr,
        /// We are just re-using the enum used for list arithmetic.
        op_apply_type: BinaryOpApplyType,
        broadcast: Broadcast,
        /// Stride of the leaf array
        stride: usize,
        /// Widths at every level
        output_widths: Vec<usize>,
        output_dtype: DataType,
        output_primitive_dtype: DataType,
        /// Length of the outermost level
        output_len: usize,
        data_lhs: (Series, Vec<Option<Bitmap>>),
        data_rhs: (Series, Vec<Option<Bitmap>>),
        swapped: bool,
    }

    /// This lets us separate some logic into `new()` to reduce the amount of
    /// monomorphized code.
    impl NumericFixedSizeListOpHelper {
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
        /// This returns an `Either` which may contain the final result to simplify
        /// the implementation.
        pub(super) fn execute_op(
            op: NumericFixedSizeListOp,
            lhs: Series,
            rhs: Series,
        ) -> PolarsResult<ArrayChunked> {
            assert_eq!(lhs.chunks().len(), 1);
            assert_eq!(rhs.chunks().len(), 1);

            let dtype_lhs = lhs.dtype();
            let dtype_rhs = rhs.dtype();

            let prim_dtype_lhs = dtype_lhs.leaf_dtype();
            let prim_dtype_rhs = dtype_rhs.leaf_dtype();

            //
            // Check leaf dtypes
            //

            if !(prim_dtype_lhs.is_supported_list_arithmetic_input()
                && prim_dtype_rhs.is_supported_list_arithmetic_input())
            {
                polars_bail!(
                    ComputeError: "cannot {} non-numeric inner dtypes: (left: {}, right: {})",
                    op.0.name(), prim_dtype_lhs, prim_dtype_rhs
                )
            }

            let output_primitive_dtype =
                op.0.try_get_leaf_supertype(prim_dtype_lhs, prim_dtype_rhs)?;

            fn is_array_type_at_all_levels(dtype: &DataType) -> bool {
                match dtype {
                    DataType::Array(inner, ..) => is_array_type_at_all_levels(inner),
                    dt if dt.is_supported_list_arithmetic_input() => true,
                    _ => false,
                }
            }

            fn array_stride_and_widths(dtype: &DataType, widths: &mut Vec<usize>) -> usize {
                if let DataType::Array(inner, size_inner) = dtype {
                    widths.push(*size_inner);
                    *size_inner * array_stride_and_widths(inner.as_ref(), widths)
                } else {
                    1
                }
            }

            //
            // Get broadcasting information and output length
            //

            let len_lhs = lhs.len();
            let len_rhs = rhs.len();

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

            //
            // Get validities for array levels
            //

            fn push_array_validities_recursive(s: &Series, out: &mut Vec<Option<Bitmap>>) {
                let mut opt_arr = s.array().ok().map(|x| {
                    assert_eq!(x.chunks().len(), 1);
                    x.downcast_get(0).unwrap()
                });

                while let Some(arr) = opt_arr {
                    // Push none if all-valid, this can potentially save some `repeat_bitmap()`
                    // materializations on broadcasting paths.
                    out.push(arr.validity().filter(|x| x.unset_bits() > 0).cloned());
                    opt_arr = arr.values().as_any().downcast_ref::<FixedSizeListArray>();
                }
            }

            let mut array_validities_lhs = vec![];
            let mut array_validities_rhs = vec![];

            push_array_validities_recursive(&lhs, &mut array_validities_lhs);
            push_array_validities_recursive(&rhs, &mut array_validities_rhs);

            let op_err_msg = |err_reason: &str| {
                polars_err!(
                    InvalidOperation:
                    "cannot {} columns: {}: (left: {}, right: {})",
                    op.0.name(), err_reason, dtype_lhs, dtype_rhs,
                )
            };

            let ensure_array_type_at_all_levels = |dtype: &DataType| {
                if !is_array_type_at_all_levels(dtype) {
                    Err(op_err_msg("dtype was not array on all nesting levels"))
                } else {
                    Ok(())
                }
            };

            //
            // Check full dtypes and get output widths
            //

            let mut output_widths = vec![];

            let (op_apply_type, stride, output_dtype) = match (dtype_lhs, dtype_rhs) {
                (dtype_lhs @ DataType::Array(..), dtype_rhs @ DataType::Array(..)) => {
                    // `get_arithmetic_field()` in the DSL checks this, but we also have to check here because if a user
                    // directly adds 2 series together it bypasses the DSL.
                    // This is currently duplicated code and should be replaced one day with an assert after Series ops get
                    // checked properly.

                    if dtype_lhs.cast_leaf(output_primitive_dtype.clone())
                        != dtype_rhs.cast_leaf(output_primitive_dtype.clone())
                    {
                        return Err(op_err_msg("differing dtypes"));
                    };

                    // We only check dtype_lhs since we already checked dtype_lhs == dtype_rhs
                    ensure_array_type_at_all_levels(dtype_lhs)?;

                    let stride = array_stride_and_widths(dtype_lhs, &mut output_widths);

                    // For array<->array without broadcasting we return early here to avoid the rest
                    // of the setup code and dispatch layers.
                    if let Broadcast::NoBroadcast = broadcast {
                        let out = op.0.apply_series(
                            &lhs.get_leaf_array().cast(&output_primitive_dtype)?,
                            &rhs.get_leaf_array().cast(&output_primitive_dtype)?,
                        );

                        return Ok(finish_array_to_array_no_broadcast(
                            lhs.name().clone(),
                            &output_widths,
                            output_len,
                            &array_validities_lhs,
                            &array_validities_rhs,
                            out,
                        ));
                    }

                    (BinaryOpApplyType::ListToList, stride, dtype_lhs)
                },
                (array_dtype @ DataType::Array(..), x)
                    if x.is_supported_list_arithmetic_input() =>
                {
                    ensure_array_type_at_all_levels(array_dtype)?;

                    let stride = array_stride_and_widths(array_dtype, &mut output_widths);
                    (BinaryOpApplyType::ListToPrimitive, stride, array_dtype)
                },
                (x, array_dtype @ DataType::Array(..))
                    if x.is_supported_list_arithmetic_input() =>
                {
                    ensure_array_type_at_all_levels(array_dtype)?;

                    let stride = array_stride_and_widths(array_dtype, &mut output_widths);
                    (BinaryOpApplyType::PrimitiveToList, stride, array_dtype)
                },
                (l, r) => polars_bail!(
                    InvalidOperation:
                    "cannot {} dtypes: {} != {}",
                    op.0.name(), l, r,
                ),
            };

            let output_dtype = output_dtype.cast_leaf(output_primitive_dtype.clone());

            assert!(!output_widths.is_empty());

            if cfg!(debug_assertions) {
                match (array_validities_lhs.len(), array_validities_rhs.len()) {
                    (l, r) if l == output_widths.len() && l == r && l > 0 => {},
                    (v, 0) | (0, v) if v == output_widths.len() => {},
                    _ => panic!(), // One side should have been an array.
                }
            }

            if output_len == 0
                || (matches!(
                    &op_apply_type,
                    BinaryOpApplyType::ListToList | BinaryOpApplyType::ListToPrimitive
                ) && lhs.rechunk_validity().is_some_and(|x| x.set_bits() == 0))
                || (matches!(
                    &op_apply_type,
                    BinaryOpApplyType::ListToList | BinaryOpApplyType::PrimitiveToList
                ) && rhs.rechunk_validity().is_some_and(|x| x.set_bits() == 0))
            {
                let DataType::Array(inner_dtype, width) = output_dtype else {
                    unreachable!()
                };

                Ok(ArrayChunked::full_null_with_dtype(
                    lhs.name().clone(),
                    output_len,
                    inner_dtype.as_ref(),
                    width,
                ))
            } else {
                Self {
                    op,
                    output_name: lhs.name().clone(),
                    op_apply_type,
                    broadcast,
                    stride,
                    output_widths,
                    output_dtype,
                    output_primitive_dtype,
                    output_len,
                    data_lhs: (lhs, array_validities_lhs),
                    data_rhs: (rhs, array_validities_rhs),
                    swapped: false,
                }
                .finish()
            }
        }

        pub(super) fn finish(mut self) -> PolarsResult<ArrayChunked> {
            // We have physical codepaths for a subset of the possible combinations of broadcasting and
            // column types. The remaining combinations are handled by dispatching to the physical
            // codepaths after operand swapping.
            //
            // # Physical impl table
            // Legend
            // * |  N  | // impl "N"
            // * | [N] | // dispatches to impl "N"
            //
            //                  |  L  |  N  |  R  | // Broadcast (L)eft, (N)oBroadcast, (R)ight
            // ListToList       | [1] |  0  |  1  |
            // ListToPrimitive  |  2  |  3  |  4  |
            // PrimitiveToList  | [4] | [3] | [2] |

            self.swapped = true;

            match (&self.op_apply_type, &self.broadcast) {
                // Mostly the same as ListNumericOp, however with fixed size list we also have
                // (BinaryOpApplyType::ListToPrimitive, Broadcast::Left) as a physical impl.
                (BinaryOpApplyType::ListToList, Broadcast::NoBroadcast) => unreachable!(), // We return earlier for this
                (BinaryOpApplyType::ListToList, Broadcast::Right)
                | (BinaryOpApplyType::ListToPrimitive, Broadcast::Left)
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
                (BinaryOpApplyType::PrimitiveToList, Broadcast::Right) => {
                    self.op_apply_type = BinaryOpApplyType::ListToPrimitive;
                    self.broadcast = Broadcast::Left;

                    std::mem::swap(&mut self.data_lhs, &mut self.data_rhs);
                    self._finish_impl_dispatch()
                },

                (BinaryOpApplyType::PrimitiveToList, Broadcast::NoBroadcast) => {
                    self.op_apply_type = BinaryOpApplyType::ListToPrimitive;

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

        fn _finish_impl_dispatch(&mut self) -> PolarsResult<ArrayChunked> {
            let output_dtype = self.output_dtype.clone();
            let output_len = self.output_len;

            let prim_lhs = self
                .data_lhs
                .0
                .get_leaf_array()
                .cast(&self.output_primitive_dtype)?
                .rechunk();
            let prim_rhs = self
                .data_rhs
                .0
                .get_leaf_array()
                .cast(&self.output_primitive_dtype)?
                .rechunk();

            debug_assert_eq!(prim_lhs.dtype(), prim_rhs.dtype());
            let prim_dtype = prim_lhs.dtype();
            debug_assert_eq!(prim_dtype, &self.output_primitive_dtype);

            // Safety: Leaf dtypes have been checked to be numeric by `try_new()`
            let out = with_match_physical_numeric_polars_type!(&prim_dtype, |$T| {
                self._finish_impl::<$T>(prim_lhs, prim_rhs)
            });

            debug_assert_eq!(out.dtype(), &output_dtype);
            assert_eq!(out.len(), output_len);

            Ok(out)
        }

        /// Internal use only - contains physical impls.
        fn _finish_impl<T: PolarsNumericType>(
            &mut self,
            prim_s_lhs: Series,
            prim_s_rhs: Series,
        ) -> ArrayChunked
        where
            T::Native: PlNumArithmetic,
            PrimitiveArray<T::Native>:
                polars_compute::comparisons::TotalEqKernel<Scalar = T::Native>,
            T::Native: Zero + IsFloat,
        {
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

            self.op.0.prepare_numeric_op_side_validities::<T>(
                &mut arr_lhs,
                &mut arr_rhs,
                self.swapped,
            );

            match (&self.op_apply_type, &self.broadcast) {
                (BinaryOpApplyType::ListToList, Broadcast::Right) => {
                    let mut out_vec: Vec<T::Native> =
                        Vec::with_capacity(self.output_len * self.stride);
                    let out_ptr: *mut T::Native = out_vec.as_mut_ptr();
                    let stride = self.stride;

                    with_match_pl_num_arith!(&self.op.0, self.swapped, |$OP| {
                        unsafe {
                            for outer_idx in 0..self.output_len {
                                for inner_idx in 0..stride {
                                    let l = arr_lhs.value_unchecked(stride * outer_idx + inner_idx);
                                    let r = arr_rhs.value_unchecked(inner_idx);

                                    *out_ptr.add(stride * outer_idx + inner_idx) = $OP(l, r);
                                }
                            }
                        }
                    });

                    unsafe { out_vec.set_len(self.output_len * self.stride) };

                    let leaf_validity = combine_validities_and(
                        arr_lhs.validity(),
                        arr_rhs
                            .validity()
                            .map(|x| repeat_bitmap(x, self.output_len))
                            .as_ref(),
                    );

                    let arr =
                        PrimitiveArray::<T::Native>::from_vec(out_vec).with_validity(leaf_validity);

                    let (_, validities_lhs) = std::mem::take(&mut self.data_lhs);
                    let (_, mut validities_rhs) = std::mem::take(&mut self.data_rhs);

                    for v in validities_rhs.iter_mut() {
                        if let Some(v) = v.as_mut() {
                            *v = repeat_bitmap(v, self.output_len);
                        }
                    }

                    finish_array_to_array_no_broadcast(
                        std::mem::take(&mut self.output_name),
                        &self.output_widths,
                        self.output_len,
                        &validities_lhs,
                        &validities_rhs,
                        Box::new(arr),
                    )
                },
                (BinaryOpApplyType::ListToPrimitive, Broadcast::Left) => {
                    let mut out_vec: Vec<T::Native> =
                        Vec::with_capacity(self.output_len * self.stride);
                    let out_ptr: *mut T::Native = out_vec.as_mut_ptr();
                    let stride = self.stride;

                    with_match_pl_num_arith!(&self.op.0, self.swapped, |$OP| {
                        unsafe {
                            for outer_idx in 0..self.output_len {
                                let r = arr_rhs.value_unchecked(outer_idx);

                                for inner_idx in 0..stride {
                                    let l = arr_lhs.value_unchecked(inner_idx);

                                    *out_ptr.add(stride * outer_idx + inner_idx) = $OP(l, r);
                                }
                            }
                        }
                    });

                    unsafe { out_vec.set_len(self.output_len * self.stride) };

                    let leaf_validity = combine_validities_array_to_primitive_no_broadcast(
                        arr_lhs
                            .validity()
                            .map(|x| repeat_bitmap(x, self.output_len))
                            .as_ref(),
                        arr_rhs.validity(),
                        self.stride,
                    );

                    let arr =
                        PrimitiveArray::<T::Native>::from_vec(out_vec).with_validity(leaf_validity);

                    let (_, mut validities) = std::mem::take(&mut self.data_lhs);

                    for v in validities.iter_mut() {
                        if let Some(v) = v.as_mut() {
                            *v = repeat_bitmap(v, self.output_len);
                        }
                    }

                    finish_with_level_validities(
                        std::mem::take(&mut self.output_name),
                        &self.output_widths,
                        self.output_len,
                        &validities,
                        Box::new(arr),
                    )
                },
                (BinaryOpApplyType::ListToPrimitive, Broadcast::NoBroadcast) => {
                    let mut out_vec: Vec<T::Native> =
                        Vec::with_capacity(self.output_len * self.stride);
                    let out_ptr: *mut T::Native = out_vec.as_mut_ptr();
                    let stride = self.stride;

                    with_match_pl_num_arith!(&self.op.0, self.swapped, |$OP| {
                        unsafe {
                            for outer_idx in 0..self.output_len {
                                let r = arr_rhs.value_unchecked(outer_idx);

                                for inner_idx in 0..stride {
                                    let idx = stride * outer_idx + inner_idx;
                                    let l = arr_lhs.value_unchecked(idx);

                                    *out_ptr.add(idx) = $OP(l, r);
                                }
                            }
                        }
                    });

                    unsafe { out_vec.set_len(self.output_len * self.stride) };

                    let leaf_validity = combine_validities_array_to_primitive_no_broadcast(
                        arr_lhs.validity(),
                        arr_rhs.validity(),
                        self.stride,
                    );

                    let arr =
                        PrimitiveArray::<T::Native>::from_vec(out_vec).with_validity(leaf_validity);

                    let (_, validities) = std::mem::take(&mut self.data_lhs);

                    finish_with_level_validities(
                        std::mem::take(&mut self.output_name),
                        &self.output_widths,
                        self.output_len,
                        &validities,
                        Box::new(arr),
                    )
                },
                (BinaryOpApplyType::ListToPrimitive, Broadcast::Right) => {
                    assert_eq!(arr_rhs.len(), 1);

                    let Some(r) = (unsafe { arr_rhs.get_unchecked(0) }) else {
                        // RHS is single primitive NULL, create the result by setting the leaf validity to all-NULL.
                        let (_, validities) = std::mem::take(&mut self.data_lhs);
                        return finish_with_level_validities(
                            std::mem::take(&mut self.output_name),
                            &self.output_widths,
                            self.output_len,
                            &validities,
                            Box::new(
                                arr_lhs.clone().with_validity(Some(Bitmap::new_with_value(
                                    false,
                                    arr_lhs.len(),
                                ))),
                            ),
                        );
                    };

                    let arr = self
                        .op
                        .0
                        .apply_array_to_scalar::<T>(arr_lhs, r, self.swapped);

                    let (_, validities) = std::mem::take(&mut self.data_lhs);

                    finish_with_level_validities(
                        std::mem::take(&mut self.output_name),
                        &self.output_widths,
                        self.output_len,
                        &validities,
                        Box::new(arr),
                    )
                },
                v @ (BinaryOpApplyType::ListToList, Broadcast::NoBroadcast)
                | v @ (BinaryOpApplyType::PrimitiveToList, Broadcast::Right)
                | v @ (BinaryOpApplyType::ListToList, Broadcast::Left)
                | v @ (BinaryOpApplyType::PrimitiveToList, Broadcast::Left)
                | v @ (BinaryOpApplyType::PrimitiveToList, Broadcast::NoBroadcast) => {
                    if cfg!(debug_assertions) {
                        panic!("operation was not re-written: {:?}", v)
                    } else {
                        unreachable!()
                    }
                },
            }
        }
    }

    /// Build the result of an array<->array operation.
    #[inline(never)]
    fn finish_array_to_array_no_broadcast(
        output_name: PlSmallStr,
        widths: &[usize],
        outer_len: usize,
        validities_lhs: &[Option<Bitmap>],
        validities_rhs: &[Option<Bitmap>],
        output_leaf_array: Box<dyn Array>,
    ) -> ArrayChunked {
        assert_eq!(
            [widths.len(), validities_lhs.len(), validities_rhs.len()],
            [widths.len(); 3]
        );

        let mut builder = FixedSizeListLevelBuilder::new(outer_len, widths);

        let validities_iter = validities_lhs
            .iter()
            .zip(validities_rhs)
            .map(|(l, r)| combine_validities_and(l.as_ref(), r.as_ref()));
        // `.rev()` - we build this from the inner level.
        let mut iter = widths.iter().zip(validities_iter).rev();

        let mut out = {
            let (width, opt_validity) = iter.next().unwrap();
            builder.build_level(*width, opt_validity, output_leaf_array)
        };

        for (width, opt_validity) in iter {
            out = builder.build_level(*width, opt_validity, Box::new(out))
        }

        ArrayChunked::with_chunk(output_name, out)
    }

    /// Used when we are operating between array<->primitive, as in that case we only need the
    /// validities from the array side.
    #[inline(never)]
    fn finish_with_level_validities(
        output_name: PlSmallStr,
        widths: &[usize],
        outer_len: usize,
        validities: &[Option<Bitmap>],
        output_leaf_array: Box<dyn Array>,
    ) -> ArrayChunked {
        assert_eq!(widths.len(), validities.len());

        let mut builder = FixedSizeListLevelBuilder::new(outer_len, widths);

        let validities_iter = validities.iter().cloned();
        // `.rev()` - we build this from the inner level.
        let mut iter = widths.iter().zip(validities_iter).rev();

        let mut out = {
            let (width, opt_validity) = iter.next().unwrap();
            builder.build_level(*width, opt_validity, output_leaf_array)
        };

        for (width, opt_validity) in iter {
            out = builder.build_level(*width, opt_validity, Box::new(out))
        }

        ArrayChunked::with_chunk(output_name, out)
    }

    /// ```text
    /// array      [x, x, x, x, ..] (stride 2)
    ///             | /   | /
    ///             |/    |/
    /// primitive  [x,    x,    ..]
    /// ```
    #[inline(never)]
    fn combine_validities_array_to_primitive_no_broadcast(
        array_leaf_validity: Option<&Bitmap>,
        primitive_validity: Option<&Bitmap>,
        stride: usize,
    ) -> Option<Bitmap> {
        match (array_leaf_validity, primitive_validity) {
            (Some(l), Some(r)) => Some((l.clone().make_mut(), r)),
            (Some(v), None) => return Some(v.clone()),
            // Materialize a full-true validity to re-use the codepath, as we still
            // need to spread the bits from the RHS to the correct positions.
            (None, Some(v)) => Some((Bitmap::new_with_value(true, stride * v.len()).make_mut(), v)),
            (None, None) => None,
        }
        .map(|(mut validity_out, primitive_validity)| {
            assert_eq!(validity_out.len(), stride * primitive_validity.len());

            unsafe {
                for outer_idx in 0..primitive_validity.len() {
                    let r = primitive_validity.get_bit_unchecked(outer_idx);

                    for inner_idx in 0..stride {
                        let idx = stride * outer_idx + inner_idx;
                        let l = validity_out.get_unchecked(idx);

                        validity_out.set_unchecked(idx, l & r);
                    }
                }
            }

            validity_out.freeze()
        })
    }

    /// Returns `n_repeats` concatenated copies of the bitmap.
    #[inline(never)]
    fn repeat_bitmap(bitmap: &Bitmap, n_repeats: usize) -> Bitmap {
        let mut out = BitmapBuilder::with_capacity(bitmap.len() * n_repeats);

        for _ in 0..n_repeats {
            for bit in bitmap.iter() {
                unsafe { out.push_unchecked(bit) }
            }
        }

        out.freeze()
    }

    struct FixedSizeListLevelBuilder {
        heights: <Vec<usize> as IntoIterator>::IntoIter,
    }

    impl FixedSizeListLevelBuilder {
        fn new(outer_len: usize, widths: &[usize]) -> Self {
            let mut current_height = outer_len;
            // We need to calculate heights here like this rather than dividing the stride because
            // there can be 0-width arrays.
            let mut heights = Vec::with_capacity(widths.len());

            heights.push(current_height);
            heights.extend(widths.iter().take(widths.len() - 1).map(|width| {
                current_height *= *width;
                current_height
            }));

            Self {
                heights: heights.into_iter(),
            }
        }
    }

    impl FixedSizeListLevelBuilder {
        fn build_level(
            &mut self,
            width: usize,
            opt_validity: Option<Bitmap>,
            inner_array: Box<dyn Array>,
        ) -> FixedSizeListArray {
            let level_height = self.heights.next_back().unwrap();
            assert_eq!(inner_array.len(), level_height * width);

            FixedSizeListArray::new(
                ArrowDataType::FixedSizeList(
                    Box::new(ArrowField::new(
                        PlSmallStr::from_static("item"),
                        inner_array.dtype().clone(),
                        // is_nullable, we always set true otherwise the Eq kernels would panic
                        // when they assert == on the arrow `Field`
                        true,
                    )),
                    width,
                ),
                level_height,
                inner_array,
                opt_validity,
            )
        }
    }
}
