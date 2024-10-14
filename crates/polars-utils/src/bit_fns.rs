/// Get the index of the `n`th set bit in `x`.
///
/// This is starting from `0`.
pub fn nth_setbit(x: u64, n: u32) -> u32 {
    #[cfg(target_feature = "bmi2")]
    {
        use std::arch::x86_64::*;

        return unsafe { _tzcnt_u64(_pdep_u64(1u64 << n, x)) as u32 };
    }


    #[cfg(not(target_feature = "bmi2"))]
    {
        let mut x = x;
        let mut n = n;

        assert!(x.count_ones() > n);

        let mut idx = 0;

        while x & 1 == 0 || n > 0 {
            n -= (x & 1) as u32;
            x >>= 1;
            idx += 1;
        }

        return idx;
    }
}
