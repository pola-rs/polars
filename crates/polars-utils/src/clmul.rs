#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
fn intel_clmul64(x: u64, y: u64) -> u64 {
    use core::arch::x86_64::*;
    unsafe {
        // SAFETY: we have the target feature.
        _mm_cvtsi128_si64(_mm_clmulepi64_si128(
            _mm_cvtsi64_si128(x as i64),
            _mm_cvtsi64_si128(y as i64),
            0,
        )) as u64
    }
}

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "aes"
))]
fn arm_clmul64(x: u64, y: u64) -> u64 {
    unsafe {
        // SAFETY: we have the target feature.
        use core::arch::aarch64::*;
        vmull_p64(x, y) as u64
    }
}

#[inline]
pub fn portable_clmul64(x: u64, mut y: u64) -> u64 {
    let mut out = 0;
    while y > 0 {
        let lsb = y & y.wrapping_neg();
        out ^= x.wrapping_mul(lsb);
        y ^= lsb;
    }
    out
}

// Computes the carryless multiplication of x and y.
#[inline]
pub fn clmul64(x: u64, y: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    return intel_clmul64(x, y);

    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    return arm_clmul64(x, y);

    #[allow(unreachable_code)]
    portable_clmul64(x, y)
}

#[inline]
pub fn portable_prefix_xorsum(mut x: u64) -> u64 {
    x <<= 1;
    for i in 0..6 {
        x ^= x << (1 << i);
    }
    x
}

// Computes for each bit i the XOR of all less significant bits.
#[inline]
pub fn prefix_xorsum(x: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    return intel_clmul64(x, u64::MAX ^ 1);

    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    return arm_clmul64(x, u64::MAX ^ 1);

    #[allow(unreachable_code)]
    portable_prefix_xorsum(x)
}

#[cfg(test)]
mod test {
    use rand::prelude::*;

    use super::*;

    #[test]
    fn test_clmul() {
        // Verify platform-specific clmul to portable.
        let mut rng = StdRng::seed_from_u64(0xdeadbeef);
        for _ in 0..100 {
            let x = rng.gen();
            let y = rng.gen();
            assert_eq!(portable_clmul64(x, y), clmul64(x, y));
        }

        // Verify portable clmul for known test vectors.
        assert_eq!(
            portable_clmul64(0x8b44729195dde0ef, 0xb976c5ae2726fab0),
            0x4ae14eae84899290
        );
        assert_eq!(
            portable_clmul64(0x399b6ed00c44b301, 0x693341db5acb2ff0),
            0x48dfa88344823ff0
        );
        assert_eq!(
            portable_clmul64(0xdf4c9f6e60deb640, 0x6d4bcdb217ac4880),
            0x7300ffe474792000
        );
        assert_eq!(
            portable_clmul64(0xa7adf3c53a200a51, 0x818cb40fe11b431e),
            0x6a280181d521797e
        );
        assert_eq!(
            portable_clmul64(0x5e78e12b744f228c, 0x4225ff19e9273266),
            0xa48b73cafb9665a8
        );
    }

    #[test]
    fn test_prefix_xorsum() {
        // Verify platform-specific prefix_xorsum to portable.
        let mut rng = StdRng::seed_from_u64(0xdeadbeef);
        for _ in 0..100 {
            let x = rng.gen();
            assert_eq!(portable_prefix_xorsum(x), prefix_xorsum(x));
        }

        // Verify portable prefix_xorsum for known test vectors.
        assert_eq!(
            portable_prefix_xorsum(0x8b44729195dde0ef),
            0x0d87a31ee696bf4a
        );
        assert_eq!(
            portable_prefix_xorsum(0xb976c5ae2726fab0),
            0x2e5b79343a3b5320
        );
        assert_eq!(
            portable_prefix_xorsum(0x399b6ed00c44b301),
            0xd1124b600878ddfe
        );
        assert_eq!(
            portable_prefix_xorsum(0x693341db5acb2ff0),
            0x4e227e926c8dcaa0
        );
        assert_eq!(
            portable_prefix_xorsum(0xdf4c9f6e60deb640),
            0x6a7715b44094db80
        );
    }
}
