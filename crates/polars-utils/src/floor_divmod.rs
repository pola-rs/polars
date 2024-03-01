pub trait FloorDivMod: Sized {
    // Returns the flooring division and associated modulo of lhs / rhs.
    // This is the same division / modulo combination as Python.
    //
    // Returns (0, 0) if other == 0.
    fn wrapping_floor_div_mod(self, other: Self) -> (Self, Self);
}

macro_rules! impl_float_div_mod {
    ($T:ty) => {
        impl FloorDivMod for $T {
            #[inline]
            fn wrapping_floor_div_mod(self, other: Self) -> (Self, Self) {
                let div = (self / other).floor();
                let mod_ = self - other * div;
                (div, mod_)
            }
        }
    };
}

macro_rules! impl_unsigned_div_mod {
    ($T:ty) => {
        impl FloorDivMod for $T {
            #[inline]
            fn wrapping_floor_div_mod(self, other: Self) -> (Self, Self) {
                (self / other, self % other)
            }
        }
    };
}

macro_rules! impl_signed_div_mod {
    ($T:ty) => {
        impl FloorDivMod for $T {
            #[inline]
            fn wrapping_floor_div_mod(self, other: Self) -> (Self, Self) {
                if other == 0 {
                    return (0, 0);
                }

                // Rust/C-style remainder is in the correct congruence
                // class, but may not have the right sign. We want a
                // remainder with the same sign as the RHS, which we
                // can get by adding RHS to the remainder if the sign of
                // the non-zero remainder differs from our RHS.
                //
                // Similarly, Rust/C-style division truncates instead of floors.
                // If the remainder was non-zero and the signs were different
                // (we'd have a negative result before truncating), we need to
                // subtract 1 from the result.
                let mut div = self.wrapping_div(other);
                let mut mod_ = self.wrapping_rem(other);
                if mod_ != 0 && (self < 0) != (other < 0) {
                    div -= 1;
                    mod_ += other;
                }
                (div, mod_)
            }
        }
    };
}

impl_unsigned_div_mod!(u8);
impl_unsigned_div_mod!(u16);
impl_unsigned_div_mod!(u32);
impl_unsigned_div_mod!(u64);
impl_unsigned_div_mod!(u128);
impl_unsigned_div_mod!(usize);
impl_signed_div_mod!(i8);
impl_signed_div_mod!(i16);
impl_signed_div_mod!(i32);
impl_signed_div_mod!(i64);
impl_signed_div_mod!(i128);
impl_signed_div_mod!(isize);
impl_float_div_mod!(f32);
impl_float_div_mod!(f64);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_signed_wrapping_div_mod() {
        // Test for all i8, should transfer to other values.
        for lhs in i8::MIN..=i8::MAX {
            for rhs in i8::MIN..=i8::MAX {
                let ans = if rhs != 0 {
                    let fdiv = (lhs as f64 / rhs as f64).floor();
                    let fmod = lhs as f64 - rhs as f64 * fdiv;

                    // float -> int conversion saturates, we want wrapping, double convert.
                    ((fdiv as i32) as i8, (fmod as i32) as i8)
                } else {
                    (0, 0)
                };

                assert_eq!(lhs.wrapping_floor_div_mod(rhs), ans);
            }
        }
    }
}
