//! The arithmetic kernels of `polars-compute`, over the arrays of `polars-array`. Every method
//! takes a [`Flat`] array, which shares its buffers with the Arrow array the kernel wants.

use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use polars_array::{Flat, PlPrimitiveArray};
use polars_compute::arithmetic::ArithmeticKernel;

/// Hands a flat array's backing buffers to the Arrow array that holds the same elements.
#[inline]
fn to_arrow<T: NativeType>(array: Flat<PlPrimitiveArray<T>>) -> PrimitiveArray<T> {
    let (values, validity) = array.into_inner();
    PrimitiveArray::new(T::PRIMITIVE.into(), values, validity)
}

/// Takes the backing buffers of an Arrow array back, which is flat by construction.
#[inline]
fn from_arrow<T: NativeType>(array: PrimitiveArray<T>) -> PlPrimitiveArray<T> {
    let (_, values, validity) = array.into_inner();
    let length = values.len();
    PlPrimitiveArray::new(values, length, validity)
}

/// The counterpart of [`ArithmeticKernel`] for the arrays of `polars-array`. Every method
/// dispatches to the Arrow kernel of the same name; the result is always laid out flat.
pub trait PlArithmeticKernel: Sized {
    /// The element type of the array this operates on.
    type Native: NativeType;

    /// The element type a true division produces: the float an integer divides into.
    type TrueDivT: NativeType;

    fn wrapping_abs(self) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_neg(self) -> PlPrimitiveArray<Self::Native>;

    fn wrapping_add(self, rhs: Self) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_sub(self, rhs: Self) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_mul(self, rhs: Self) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_floor_div(self, rhs: Self) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_trunc_div(self, rhs: Self) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_mod(self, rhs: Self) -> PlPrimitiveArray<Self::Native>;

    fn wrapping_add_scalar(self, rhs: Self::Native) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_sub_scalar(self, rhs: Self::Native) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_sub_scalar_lhs(lhs: Self::Native, rhs: Self) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_mul_scalar(self, rhs: Self::Native) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_floor_div_scalar(self, rhs: Self::Native) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_floor_div_scalar_lhs(
        lhs: Self::Native,
        rhs: Self,
    ) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_trunc_div_scalar(self, rhs: Self::Native) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_trunc_div_scalar_lhs(
        lhs: Self::Native,
        rhs: Self,
    ) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_mod_scalar(self, rhs: Self::Native) -> PlPrimitiveArray<Self::Native>;
    fn wrapping_mod_scalar_lhs(lhs: Self::Native, rhs: Self) -> PlPrimitiveArray<Self::Native>;

    fn checked_mul_scalar(self, rhs: Self::Native) -> PlPrimitiveArray<Self::Native>;

    fn true_div(self, rhs: Self) -> PlPrimitiveArray<Self::TrueDivT>;
    fn true_div_scalar(self, rhs: Self::Native) -> PlPrimitiveArray<Self::TrueDivT>;
    fn true_div_scalar_lhs(lhs: Self::Native, rhs: Self) -> PlPrimitiveArray<Self::TrueDivT>;

    // These are flooring division for integer types, true division for floating point types.
    fn legacy_div(self, rhs: Self) -> PlPrimitiveArray<Self::Native>;
    fn legacy_div_scalar(self, rhs: Self::Native) -> PlPrimitiveArray<Self::Native>;
    fn legacy_div_scalar_lhs(lhs: Self::Native, rhs: Self) -> PlPrimitiveArray<Self::Native>;
}

macro_rules! delegate {
    // A kernel over one array.
    (unary $name:ident) => {
        #[inline]
        fn $name(self) -> PlPrimitiveArray<T> {
            from_arrow(ArithmeticKernel::$name(to_arrow(self)))
        }
    };
    // A kernel over two arrays of the same length.
    (binary $name:ident) => {
        #[inline]
        fn $name(self, rhs: Self) -> PlPrimitiveArray<T> {
            from_arrow(ArithmeticKernel::$name(to_arrow(self), to_arrow(rhs)))
        }
    };
    // A kernel over an array and a value broadcast against it.
    (scalar_rhs $name:ident) => {
        #[inline]
        fn $name(self, rhs: T) -> PlPrimitiveArray<T> {
            from_arrow(ArithmeticKernel::$name(to_arrow(self), rhs))
        }
    };
    (scalar_lhs $name:ident) => {
        #[inline]
        fn $name(lhs: T, rhs: Self) -> PlPrimitiveArray<T> {
            from_arrow(ArithmeticKernel::$name(lhs, to_arrow(rhs)))
        }
    };
}

impl<T> PlArithmeticKernel for Flat<PlPrimitiveArray<T>>
where
    T: NativeType,
    PrimitiveArray<T>: ArithmeticKernel<Scalar = T>,
{
    type Native = T;
    type TrueDivT = <PrimitiveArray<T> as ArithmeticKernel>::TrueDivT;

    delegate!(unary wrapping_abs);
    delegate!(unary wrapping_neg);

    delegate!(binary wrapping_add);
    delegate!(binary wrapping_sub);
    delegate!(binary wrapping_mul);
    delegate!(binary wrapping_floor_div);
    delegate!(binary wrapping_trunc_div);
    delegate!(binary wrapping_mod);
    delegate!(binary legacy_div);

    delegate!(scalar_rhs wrapping_add_scalar);
    delegate!(scalar_rhs wrapping_sub_scalar);
    delegate!(scalar_rhs wrapping_mul_scalar);
    delegate!(scalar_rhs wrapping_floor_div_scalar);
    delegate!(scalar_rhs wrapping_trunc_div_scalar);
    delegate!(scalar_rhs wrapping_mod_scalar);
    delegate!(scalar_rhs checked_mul_scalar);
    delegate!(scalar_rhs legacy_div_scalar);

    delegate!(scalar_lhs wrapping_sub_scalar_lhs);
    delegate!(scalar_lhs wrapping_floor_div_scalar_lhs);
    delegate!(scalar_lhs wrapping_trunc_div_scalar_lhs);
    delegate!(scalar_lhs wrapping_mod_scalar_lhs);
    delegate!(scalar_lhs legacy_div_scalar_lhs);

    #[inline]
    fn true_div(self, rhs: Self) -> PlPrimitiveArray<Self::TrueDivT> {
        from_arrow(ArithmeticKernel::true_div(to_arrow(self), to_arrow(rhs)))
    }

    #[inline]
    fn true_div_scalar(self, rhs: T) -> PlPrimitiveArray<Self::TrueDivT> {
        from_arrow(ArithmeticKernel::true_div_scalar(to_arrow(self), rhs))
    }

    #[inline]
    fn true_div_scalar_lhs(lhs: T, rhs: Self) -> PlPrimitiveArray<Self::TrueDivT> {
        from_arrow(ArithmeticKernel::true_div_scalar_lhs(lhs, to_arrow(rhs)))
    }
}
