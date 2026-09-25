/*
    Decimal implementation.

    Throughout this module it's assumed that p and s fit in the maximum precision,
    giving panics otherwise.

    Constants for division have been generated with the following Python code:

        # Finds integer (c, s) such that floor(n / d) == (n * c) >> s for all n in [0, N].
        # From Integer division by constants: optimal bounds by Lemire et. al, Theorem 1.
        # This constant allows for fast division, as well as a divisibility check,
        # namely we find that n % d == 0 for all n in [0, N] iff (n * c) % 2**s < c.
        def inv_mult_shift(d, N):
            s = 0
            m = 1
            c = 1
            K = N - (N + 1) % d
            while True:
                if c * d * K < (1 + K)*m:
                    break
                s += 1
                m *= 2
                c = (m + d - 1) // d  # ceil(m / d)
            return (c, s)

    Also from that paper is the algorithm we use for round-to-nearest division.
    We compute z = n + floor(d/2), and then return floor(z / d) unless z % d == 0
    and the result is odd in which case we subtract 1.
*/

use std::cmp::Ordering;

use polars_error::{PolarsResult, polars_ensure};

/// The maximum precision of a Decimal128.
pub const DEC128_MAX_PREC: usize = 38;

pub fn dec128_verify_prec_scale(p: usize, s: usize) -> PolarsResult<()> {
    polars_ensure!((1..=DEC128_MAX_PREC).contains(&p), InvalidOperation: "precision must be between 1 and 38");
    polars_ensure!(s <= p, InvalidOperation: "scale must be less than or equal to precision");
    Ok(())
}

pub const POW10_I128: &[i128; 39] = &{
    let mut out = [0; 39];
    let mut i = 0;
    while i < 39 {
        out[i] = 10_i128.pow(i as u32);
        i += 1;
    }
    out
};

pub const POW10_F64: &[f64; 39] = &{
    let mut out = [0.0; 39];
    let mut i = 0;
    while i < 39 {
        out[i] = POW10_I128[i] as f64;
        i += 1;
    }
    out
};

// for e in range(39):
//     c, s = inv_mult_shift(10**e, 2**127-1)
#[rustfmt::skip]
const POW10_127_INV_MUL: &[u128; 39] = &[
    0x00000000000000000000000000000001, 0x66666666666666666666666666666667,
    0x28f5c28f5c28f5c28f5c28f5c28f5c29, 0x4189374bc6a7ef9db22d0e5604189375,
    0x68db8bac710cb295e9e1b089a0275255, 0x29f16b11c6d1e108c3f3e0370cdc8755,
    0x08637bd05af6c69b5a63f9a49c2c1b11, 0xd6bf94d5e57a42bc3d32907604691b4d,
    0x55e63b88c230e77e7ee106959b5d3e1f, 0x89705f4136b4a59731680a88f8953031,
    0x36f9bfb3af7b756fad5cd10396a21347, 0x57f5ff85e592557f7bc7b4d28a9ceba5,
    0x232f33025bd42232fe4fe1edd10b9175, 0x709709a125da07099432d2f9035837dd,
    0xb424dc35095cd80f538484c19ef38c95, 0x901d7cf73ab0acd90f9d37014bf60a11,
    0x39a5652fb1137856d30baf9a1e626a6d, 0x2e1dea8c8da92d12426fbfae7eb521f1,
    0x09392ee8e921d5d073aff322e62439fd, 0x760f253edb4ab0d29598f4f1e8361973,
    0xbce5086492111aea88f4bb1ca6bcf585, 0x25c768141d369efbb4fdbf05baf29781,
    0xf1c90080baf72cb15324c68b12dd6339, 0x305b66802564a289dd6dc14f03c5e0a5,
    0x9abe14cd44753b52c4926a9672793543, 0x7bcb43d769f762a89d41eedec1fa9103,
    0x63090312bb2c4eed4a9b257f019540cf, 0x4f3a68dbc8f03f243baf513267aa9a3f,
    0xfd87b5f28300ca0d8bca9d6e188853fd, 0xcad2f7f5359a3b3e096ee45813a04331,
    0x51212ffbaf0a7e18d092c1bcd4a68147, 0x1039d66589687f9e901d59f290ee19db,
    0xcfb11ead453994ba67de18eda5814af3, 0xa6274bbdd0fadd61ecb1ad8aeacdd58f,
    0x84ec3c97da624ab4bd5af13bef0b113f, 0xd4ad2dbfc3d07787955e4ec64b44e865,
    0x5512124cb4b9c9696ef285e8eae85cf5, 0x881cea14545c75757e50d64177da2e55,
    0x6ce3ee76a9e3912acb73de9ac6482511,
];

const POW10_127_SHIFT: &[u8; 39] = &[
    0, 2, 4, 8, 12, 14, 15, 23, 25, 29, 31, 35, 37, 42, 46, 49, 51, 54, 55, 62, 66, 67, 73, 74, 79,
    82, 85, 88, 93, 96, 98, 99, 106, 109, 112, 116, 118, 122, 125,
];

// for e in range(39):
//     c, s = inv_mult_shift(10**e, 2**255-1)
#[rustfmt::skip]
const POW10_255_INV_MUL: &[U256; 39] = &[
    U256([0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000]),
    U256([0x6666666666666667, 0x6666666666666666, 0x6666666666666666, 0x6666666666666666]),
    U256([0xae147ae147ae147b, 0x7ae147ae147ae147, 0x47ae147ae147ae14, 0x147ae147ae147ae1]),
    U256([0x5604189374bc6a7f, 0x4bc6a7ef9db22d0e, 0xdb22d0e560418937, 0x04189374bc6a7ef9]),
    U256([0x19ce075f6fd21ff3, 0x305532617c1bda51, 0xf4f0d844d013a92a, 0x346dc5d63886594a]),
    U256([0x85c67dfe32a0663d, 0xcddd6e04c0592103, 0x0fcf80dc33721d53, 0xa7c5ac471b478423]),
    U256([0x37d1fe64f54d1e97, 0xd7e45803cd141a69, 0xa63f9a49c2c1b10f, 0x8637bd05af6c69b5]),
    U256([0xc6419850c43db213, 0x4650466970dce1ed, 0x1e99483b02348da6, 0x6b5fca6af2bd215e]),
    U256([0x7068f3b46d2f8351, 0x3d4d3d758161697c, 0xfdc20d2b36ba7c3d, 0xabcc77118461cefc]),
    U256([0xf387295d242602a7, 0xfdd7645e011abac9, 0x31680a88f8953030, 0x89705f4136b4a597]),
    U256([0x2e36108ba80f3443, 0xcbefc1bf33a44ab7, 0xad5cd10396a21346, 0x36f9bfb3af7b756f]),
    U256([0x49f01a790ce5206b, 0x797f9c651f6d4458, 0x7bc7b4d28a9ceba4, 0x57f5ff85e592557f]),
    U256([0x4319c3f4e16e9a45, 0xf598fa3b657ba08d, 0xf93f87b7442e45d3, 0x8cbccc096f5088cb]),
    U256([0x409ec0ca937c8541, 0x311e9872477f201c, 0x650cb4be40d60df7, 0x1c25c268497681c2]),
    U256([0x01fc02883e5b4403, 0x36c84e3a7e6399f4, 0xa9c24260cf79c64a, 0x5a126e1a84ae6c07]),
    U256([0x3660040d3092066b, 0x57a6e390ca38f653, 0x0f9d37014bf60a10, 0x901d7cf73ab0acd9]),
    U256([0x48f334d2136d9c2b, 0xefdc5b06b749fc21, 0xd30baf9a1e626a6c, 0x39a5652fb1137856]),
    U256([0x4fd70f6d0af85a23, 0xff8df0157db98d37, 0x09befeb9fad487c2, 0xb877aa3236a4b449]),
    U256([0x8656062b9dfcf0db, 0x996bf9a2324a387c, 0x9d7f99173121cfe7, 0x49c97747490eae83]),
    U256([0xe11346f1f98fcf89, 0x1e2652070753e7f4, 0x2b31e9e3d06c32e5, 0xec1e4a7db69561a5]),
    U256([0x26d482c7309fec9d, 0x0c0f5402cfbb2995, 0x447a5d8e535e7ac2, 0x5e72843249088d75]),
    U256([0xa48737a51a997a95, 0x467eecd14c5ea8ee, 0xd3f6fc16ebca5e03, 0x971da05074da7bee]),
    U256([0x1d38f950e2146211, 0x38658a4109e553f2, 0xa9926345896eb19c, 0x78e480405d7b9658]),
    U256([0x05d831dcfa04139d, 0x71ade873686110ca, 0xeeb6e0a781e2f052, 0x182db34012b25144]),
    U256([0xf23472530ce6e3ed, 0xd78c3615cf3a050c, 0xc4926a9672793542, 0x9abe14cd44753b52]),
    U256([0xe9ed83b814a49fe1, 0x8c1389bc7ec33b47, 0x3a83ddbd83f52204, 0xf79687aed3eec551]),
    U256([0x87f1362cdd507fe7, 0x3cdc6e306568fc39, 0x95364afe032a819d, 0xc612062576589dda]),
    U256([0xcffa15ab8bb9ccc3, 0xe524f8e0289064e3, 0x3baf513267aa9a3e, 0x4f3a68dbc8f03f24]),
    U256([0xe65cef78df8fae05, 0x3b6e5b0040e707d2, 0xc5e54eb70c4429fe, 0x7ec3daf941806506]),
    U256([0x28f1f9638c9fdf35, 0x17c5be0019f60321, 0x825bb91604e810cc, 0x32b4bdfd4d668ecf]),
    U256([0x3b6398471c1ff971, 0xd18df2ccd1fe00a0, 0x1a1258379a94d028, 0x0a2425ff75e14fc3]),
    U256([0x7c1701c71a663c6d, 0xa38c78520cc00401, 0x407567ca43b8676b, 0x40e7599625a1fe7a]),
    U256([0x59e338e387ad8e29, 0x0b5b1aa028ccd99e, 0x67de18eda5814af2, 0xcfb11ead453994ba]),
    U256([0x11fa3e93e7ef82d5, 0x9bdf05533b5c2b86, 0x7b2c6b62bab37563, 0x2989d2ef743eb758]),
    U256([0xe990641fd97f37bb, 0x5fcb3bb85ef9df3c, 0x5ead789df785889f, 0x42761e4bed31255a]),
    U256([0x90a0280cbd66164b, 0x8cb7b17cf2ca594b, 0xf2abc9d8c9689d0c, 0x1a95a5b7f87a0ef0]),
    U256([0xb4337347957023ab, 0x7abf82618476f545, 0xb77942f475742e7a, 0x2a8909265a5ce4b4]),
    U256([0x40a4a418449a0bbd, 0xbbfe6e04db164412, 0x7e50d64177da2e54, 0x881cea14545c7575]),
    U256([0x735420d1a7520259, 0x259949342bd140d0, 0xb2dcf7a6b1920944, 0x1b38fb9daa78e44a]),
];

const POW10_255_SHIFT: &[u8; 39] = &[
    0, 2, 3, 4, 11, 16, 19, 22, 26, 29, 31, 35, 39, 40, 45, 49, 51, 56, 58, 63, 65, 69, 72, 73, 79,
    83, 86, 88, 92, 94, 95, 101, 106, 107, 111, 113, 117, 122, 123,
];

// Limbs in little-endian order (limb 0 is least significant).
#[derive(Copy, Clone, PartialEq, Eq)]
struct U256([u64; 4]);

impl U256 {
    #[inline(always)]
    fn from_lo_hi(lo: u128, hi: u128) -> Self {
        Self([lo as u64, (lo >> 64) as u64, hi as u64, (hi >> 64) as u64])
    }
}

impl PartialOrd for U256 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for U256 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0[3]
            .cmp(&other.0[3])
            .then(self.0[2].cmp(&other.0[2]))
            .then(self.0[1].cmp(&other.0[1]))
            .then(self.0[0].cmp(&other.0[0]))
    }
}

#[inline]
fn u128_from_lo_hi(lo: u64, hi: u64) -> u128 {
    (lo as u128) | ((hi as u128) << 64)
}

#[inline(always)]
fn widening_mul_64(a: u64, b: u64) -> (u64, u64) {
    let t = (a as u128) * (b as u128);
    (t as u64, (t >> 64) as u64)
}

#[inline(always)]
fn carrying_add_64(a: u64, b: u64, carry: bool) -> (u64, bool) {
    let (t0, c1) = a.overflowing_add(b);
    let (t1, c2) = t0.overflowing_add(carry as u64);
    (t1, c1 | c2)
}

#[inline]
fn widening_mul_128(a: u128, b: u128) -> (u128, u128) {
    let a_lo = a as u64;
    let a_hi = (a >> 64) as u64;
    let b_lo = b as u64;
    let b_hi = (b >> 64) as u64;
    let (x0, x1) = widening_mul_64(a_lo, b_lo);
    let (y1, y2) = widening_mul_64(a_hi, b_lo);
    let (z1, z2) = widening_mul_64(a_lo, b_hi);
    let (w2, w3) = widening_mul_64(a_hi, b_hi);

    let mut out = [0; 4];
    let (c1, c2, c3, c4);
    out[0] = x0;
    (out[1], c1) = carrying_add_64(x1, y1, false);
    (out[1], c2) = carrying_add_64(out[1], z1, false);
    (out[2], c3) = carrying_add_64(y2, z2, c1);
    (out[2], c4) = carrying_add_64(out[2], w2, c2);
    out[3] = w3.wrapping_add(c3 as u64 + c4 as u64);

    (
        out[0] as u128 | ((out[1] as u128) << 64),
        out[2] as u128 | ((out[3] as u128) << 64),
    )
}

fn widening_mul_256(a: U256, b: U256) -> (U256, U256) {
    let mut out = [0; 8];

    // Algorithm M from TAOCP: Seminumerical algorithms, ch 4.3.1.
    // We represent the carry as carry_word + c1 + c2, which fits in a u64.
    for i in 0..4 {
        let mut carry_word = 0;
        let mut c1 = false;
        let mut c2 = false;
        for j in 0..4 {
            let (mut lo, hi) = widening_mul_64(a.0[i], b.0[j]);
            (lo, c1) = carrying_add_64(lo, out[i + j], c1);
            (lo, c2) = carrying_add_64(lo, carry_word, c2);
            out[i + j] = lo;
            carry_word = hi;
        }
        out[i + 4] = carry_word + c1 as u64 + c2 as u64;
    }

    let (lo, hi) = out.split_at(4);
    (U256(lo.try_into().unwrap()), U256(hi.try_into().unwrap()))
}

/// Returns x * 10^e, with e <= DEC128_MAX_PREC.
///
/// Returns None if the multiplication overflows.
#[inline]
fn mul_128_pow10(x: i128, e: usize) -> Option<i128> {
    x.checked_mul(POW10_I128[e])
}

/// Returns round(x / 10^e), with e <= DEC128_MAX_PREC, rounding to nearest even.
#[inline]
fn div_128_pow10(x: i128, e: usize) -> i128 {
    if e == 0 {
        return x;
    }

    let n = x.unsigned_abs();
    let z = n + ((POW10_I128[e] as u128) / 2); // Can't overflow.

    // Most values fit in 64 bits, where a single division replaces the
    // 128x128 widening multiply.
    if z <= u64::MAX as u128 && POW10_I128[e] <= u64::MAX as i128 {
        let d = POW10_I128[e] as u64;
        let zu = z as u64;
        let mut ret = (zu / d) as i128;
        // z = n + d/2, so d divides z iff n is exactly halfway; round to even.
        if zu.is_multiple_of(d) && ret % 2 == 1 {
            ret -= 1;
        }
        return if x < 0 { -ret } else { ret };
    }

    let c = POW10_127_INV_MUL[e];
    let s = POW10_127_SHIFT[e];
    let (lo, hi) = widening_mul_128(z, c);
    let mut ret = (hi >> s) as i128;
    if lo < c && ret % 2 == 1 && (hi << (128 - s)) == 0 {
        ret -= 1;
    }
    if x < 0 { -ret } else { ret }
}

/// Returns round(x / 10^e), with e <= DEC128_MAX_PREC, rounding to nearest even.
/// x is assumed to be < 2^255. Returns None if the result doesn't fit in a u128.
#[inline]
fn div_255_pow10(x: U256, e: usize) -> Option<u128> {
    if e == 0 {
        if x.0[2] == 0 && x.0[3] == 0 {
            return Some(u128_from_lo_hi(x.0[0], x.0[1]));
        } else {
            return None;
        }
    }

    let half = (POW10_I128[e] as u128) / 2;
    let mut carry;
    let mut z = x;
    (z.0[0], carry) = z.0[0].overflowing_add(half as u64);
    (z.0[1], carry) = carrying_add_64(z.0[1], (half >> 64) as u64, carry);
    (z.0[2], carry) = z.0[2].overflowing_add(carry as u64);
    z.0[3] += carry as u64;
    let c = POW10_255_INV_MUL[e];
    let s = POW10_255_SHIFT[e];
    let (lo, hi) = widening_mul_256(z, c);
    let shifted_out_is_zero;
    let mut ret = if s < 64 {
        if (hi.0[2] >> s) != 0 || hi.0[3] != 0 {
            return None;
        }
        shifted_out_is_zero = (hi.0[0] << (64 - s)) == 0;
        (u128_from_lo_hi(hi.0[0], hi.0[1]) >> s) | u128_from_lo_hi(0, hi.0[2] << (64 - s))
    } else {
        debug_assert!(s < 128);
        let s = s - 64;
        if (hi.0[3] >> s) != 0 {
            return None;
        }
        shifted_out_is_zero = hi.0[0] == 0 && (hi.0[1] << (64 - s)) == 0;
        (u128_from_lo_hi(hi.0[1], hi.0[2]) >> s) | u128_from_lo_hi(0, hi.0[3] << (64 - s))
    };

    if lo < c && ret % 2 == 1 && shifted_out_is_zero {
        ret -= 1;
    }

    Some(ret)
}

/// Calculates n / d, returning quotient and remainder.
///
/// # Safety
/// Assumes quotient fits in u64, and d != 0.
unsafe fn divrem_128_64(n: u128, d: u64) -> (u64, u64) {
    let quo: u64;
    let rem: u64;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let nlo = n as u64;
        let nhi = (n >> 64) as u64;
        std::arch::asm!(
            "div {d}",
            d = in(reg) d,
            inlateout("rax") nlo => quo,
            inlateout("rdx") nhi => rem,
            options(pure, nomem, nostack)
        );
    }

    #[cfg(not(target_arch = "x86_64"))]
    unsafe {
        // TODO: more optimized implementation.
        if n < (1 << 64) {
            quo = (n as u64).checked_div(d).unwrap_unchecked();
            rem = (n as u64).checked_rem(d).unwrap_unchecked();
        } else {
            quo = n.checked_div(d as u128).unwrap_unchecked() as u64;
            rem = n.checked_rem(d as u128).unwrap_unchecked() as u64;
        }
    }

    (quo, rem)
}

/// Calculates the quotient and remainder of ((hi << 128) | lo) / d.
/// Returns None if the quotient overflows a 128-bit integer.
fn divrem_256_128(lo: u128, hi: u128, d: u128) -> Option<(u128, u128)> {
    if d == 0 || hi >= d {
        return None;
    }

    if hi == 0 {
        return Some(((lo / d), (lo % d)));
    }

    if d < (1 << 64) {
        // Short division (exercise 16, TAOCP, 4.3.1).
        let d = d as u64;
        let (q_hi, r_hi) =
            unsafe { divrem_128_64(u128_from_lo_hi((lo >> 64) as u64, hi as u64), d) };
        let (q_lo, r_lo) = unsafe { divrem_128_64(u128_from_lo_hi(lo as u64, r_hi), d) };
        return Some((u128_from_lo_hi(q_lo, q_hi), u128_from_lo_hi(r_lo, 0)));
    }

    // Long division (algorithm D, TAOCP, 4.3.1).
    // Normalize d, n so that d has the top bit set.
    let shift = ((d >> 64) as u64).leading_zeros();
    let d1 = ((d << shift) >> 64) as u64;
    let d0 = (d as u64) << shift;
    let mut n3 = (hi >> 64) as u64;
    let mut n2 = hi as u64;
    let mut n1 = (lo >> 64) as u64;
    let mut n0 = lo as u64;
    n3 = ((u128_from_lo_hi(n2, n3) << shift) >> 64) as u64;
    n2 = ((u128_from_lo_hi(n1, n2) << shift) >> 64) as u64;
    n1 = ((u128_from_lo_hi(n0, n1) << shift) >> 64) as u64;
    n0 <<= shift;

    // We want to calculate
    //    (qhat, rhat) = divmod(n3n2, d1)
    // and then do the test qhat * d0 > (rhat << 64) + n1, possibly twice, to
    // adjust qhat downwards. But we have to be very careful around overflow,
    // as both the division and intermediate steps can overflow.
    let (mut qhat, mut rhat, mut qhd0_lo, mut qhd0_hi, mut borrow);
    if n3 < d1 {
        (qhat, rhat) = unsafe { divrem_128_64(u128_from_lo_hi(n2, n3), d1) };
        (qhd0_lo, qhd0_hi) = widening_mul_64(qhat, d0);
    } else {
        qhat = 0; // Represents 1 << 64, will be corrected below.
        rhat = n2;
        qhd0_lo = 0;
        qhd0_hi = d0;
    };

    if qhd0_hi > rhat || qhd0_hi == rhat && qhd0_lo > n1 {
        qhat = qhat.wrapping_sub(1);
        let rhat_overflow;
        (rhat, rhat_overflow) = rhat.overflowing_add(d1);
        (qhd0_lo, borrow) = qhd0_lo.overflowing_sub(d0);
        qhd0_hi -= borrow as u64;
        if !rhat_overflow && (qhd0_hi > rhat || qhd0_hi == rhat && qhd0_lo > n1) {
            qhat -= 1;
            (qhd0_lo, borrow) = qhd0_lo.overflowing_sub(d0);
            qhd0_hi -= borrow as u64;
        }
    }

    // Subtract qhat*d from n3n2n1, this zeroes out n3. We don't need to worry
    // about our number going negative like in the original Algorithm D because
    // we only have two limbs worth of divisor (making qhat exact).
    let q_hi = qhat;
    n2 = n2.wrapping_sub(qhat.wrapping_mul(d1));
    (n1, borrow) = n1.overflowing_sub(qhd0_lo);
    n2 = n2.wrapping_sub(qhd0_hi + borrow as u64);

    // Repeat the whole process again with n2n1n0.
    if n2 < d1 {
        (qhat, rhat) = unsafe { divrem_128_64(u128_from_lo_hi(n1, n2), d1) };
        (qhd0_lo, qhd0_hi) = widening_mul_64(qhat, d0);
    } else {
        qhat = 0; // Represents 1 << 64, will be corrected below.
        rhat = n1;
        qhd0_lo = 0;
        qhd0_hi = d0;
    };

    if qhd0_hi > rhat || qhd0_hi == rhat && qhd0_lo > n0 {
        qhat = qhat.wrapping_sub(1);
        let rhat_overflow;
        (rhat, rhat_overflow) = rhat.overflowing_add(d1);
        (qhd0_lo, borrow) = qhd0_lo.overflowing_sub(d0);
        qhd0_hi -= borrow as u64;
        if !rhat_overflow && (qhd0_hi > rhat || qhd0_hi == rhat && qhd0_lo > n0) {
            qhat -= 1;
            (qhd0_lo, borrow) = qhd0_lo.overflowing_sub(d0);
            qhd0_hi -= borrow as u64;
        }
    }

    let q_lo = qhat;
    n1 = n1.wrapping_sub(qhat.wrapping_mul(d1));
    (n0, borrow) = n0.overflowing_sub(qhd0_lo);
    n1 = n1.wrapping_sub(qhd0_hi + borrow as u64);

    // n1n0 is now our remainder, once we account for the shift.
    let r_lo = (u128_from_lo_hi(n0, n1) >> shift) as u64;
    let r_hi = n1 >> shift;

    Some((u128_from_lo_hi(q_lo, q_hi), u128_from_lo_hi(r_lo, r_hi)))
}

/// Returns whether the given Decimal128 fits in the given precision.
#[inline]
pub fn dec128_fits(x: i128, p: usize) -> bool {
    (-POW10_I128[p] < x) & (x < POW10_I128[p])
}

#[inline]
pub fn dec128_to_i128(x: i128, s: usize) -> i128 {
    if s == 0 { x } else { div_128_pow10(x, s) }
}

/// Returns `x * 10^e`, or None on overflow; `e` is not bounded by `DEC128_MAX_PREC`.
#[inline]
pub fn i128_mul_pow10(x: i128, mut e: usize) -> Option<i128> {
    let mut r = x;
    while e > DEC128_MAX_PREC {
        r = r.checked_mul(POW10_I128[DEC128_MAX_PREC])?;
        e -= DEC128_MAX_PREC;
    }
    r.checked_mul(POW10_I128[e])
}

/// Exact scalar arithmetic on `(mantissa, scale)` fixed-point values, bounded only by
/// `i128`: the result scale is the larger input scale for `add`/`sub` and the sum of the
/// input scales for `mul`, so no rounding takes place. Returns None on overflow.
pub mod exact {
    use super::i128_mul_pow10;

    fn align(l: (i128, usize), r: (i128, usize)) -> Option<(i128, i128, usize)> {
        let s = l.1.max(r.1);
        Some((
            i128_mul_pow10(l.0, s - l.1)?,
            i128_mul_pow10(r.0, s - r.1)?,
            s,
        ))
    }

    #[inline]
    pub fn add(l: (i128, usize), r: (i128, usize)) -> Option<(i128, usize)> {
        let (l, r, s) = align(l, r)?;
        Some((l.checked_add(r)?, s))
    }

    #[inline]
    pub fn sub(l: (i128, usize), r: (i128, usize)) -> Option<(i128, usize)> {
        let (l, r, s) = align(l, r)?;
        Some((l.checked_sub(r)?, s))
    }

    #[inline]
    pub fn mul(l: (i128, usize), r: (i128, usize)) -> Option<(i128, usize)> {
        Some((l.0.checked_mul(r.0)?, l.1 + r.1))
    }

    #[inline]
    pub fn neg(x: (i128, usize)) -> Option<(i128, usize)> {
        Some((x.0.checked_neg()?, x.1))
    }
}

/// Converts an i128 to a Decimal128 with the given precision and scale,
/// returning None if the value doesn't fit.
#[inline]
pub fn i128_to_dec128(x: i128, p: usize, s: usize) -> Option<i128> {
    let r = x.checked_mul(POW10_I128[s])?;
    dec128_fits(r, p).then_some(r)
}

/// Converts a Decimal128 with the given scale to the nearest f64.
#[inline]
pub fn dec128_to_f64(x: i128, s: usize) -> f64 {
    // With an exact numerator and denominator, the one division rounds correctly.
    if s == 0 || (x.unsigned_abs() <= 1 << 53 && s <= 22) {
        return x as f64 / POW10_F64[s];
    }
    parse_dec128(x, s)
}

/// Converts a Decimal128 with the given scale to the nearest f32.
#[inline]
pub fn dec128_to_f32(x: i128, s: usize) -> f32 {
    if s == 0 {
        return x as f32;
    }
    // Rounding the nearest f64 again is correct unless it is exactly halfway
    // between two normal f32s: the true value then lies on the same side.
    let q = dec128_to_f64(x, s);
    if q.to_bits() & 0x1FFF_FFFF != 0x1000_0000 && q.abs() >= f32::MIN_POSITIVE as f64 {
        return q as f32;
    }
    parse_dec128(x, s)
}

/// Rounds `x * 10^-s` once, through the correctly rounded float parser.
#[cold]
fn parse_dec128<F: std::str::FromStr>(x: i128, s: usize) -> F
where
    F::Err: std::fmt::Debug,
{
    use std::io::Write;

    let mut buf = [0u8; 48];
    let mut w = &mut buf[..];
    write!(w, "{x}e-{s}").unwrap();
    let len = 48 - w.len();
    std::str::from_utf8(&buf[..len]).unwrap().parse().unwrap()
}

/// Converts a f64 to a Decimal128 with the given precision and scale, returning
/// None if the value doesn't fit.
#[inline]
pub fn f64_to_dec128(x: f64, p: usize, s: usize) -> Option<i128> {
    // TODO: correctly rounded result. This rounds multiple times.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(x.abs() < POW10_F64[p]) {
        // Comparison will fail for NaN, making us return None.
        return None;
    }
    unsafe { Some((x * POW10_F64[s]).round_ties_even().to_int_unchecked()) }
}

/// Converts between two Decimal128s, with a new precision and scale, returning
/// None if the value doesn't fit.
#[inline]
pub fn dec128_rescale(x: i128, old_s: usize, new_p: usize, new_s: usize) -> Option<i128> {
    let r = if new_s < old_s {
        div_128_pow10(x, old_s - new_s)
    } else if new_s > old_s {
        mul_128_pow10(x, new_s - old_s)?
    } else {
        return Some(x);
    };

    dec128_fits(r, new_p).then_some(r)
}

/// A value in `(-2^255, 2^255)` in sign-magnitude form.
#[derive(Copy, Clone)]
struct I256 {
    neg: bool,
    mag: U256,
}

impl U256 {
    #[inline(always)]
    fn from_u128(x: u128) -> Self {
        Self::from_lo_hi(x, 0)
    }

    #[inline(always)]
    fn lo_hi(self) -> (u128, u128) {
        (
            u128_from_lo_hi(self.0[0], self.0[1]),
            u128_from_lo_hi(self.0[2], self.0[3]),
        )
    }

    #[inline(always)]
    fn to_u128(self) -> Option<u128> {
        let (lo, hi) = self.lo_hi();
        (hi == 0).then_some(lo)
    }

    /// Assumes the sum doesn't overflow.
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        let (a_lo, a_hi) = self.lo_hi();
        let (b_lo, b_hi) = other.lo_hi();
        let (lo, carry) = a_lo.overflowing_add(b_lo);
        Self::from_lo_hi(lo, a_hi + b_hi + carry as u128)
    }

    /// Assumes self >= other.
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        let (a_lo, a_hi) = self.lo_hi();
        let (b_lo, b_hi) = other.lo_hi();
        let (lo, borrow) = a_lo.overflowing_sub(b_lo);
        Self::from_lo_hi(lo, a_hi - b_hi - borrow as u128)
    }
}

/// Returns x * 10^e, with e <= DEC128_MAX_PREC.
#[inline]
fn u256_mul_pow10(x: u128, e: usize) -> U256 {
    let (lo, hi) = widening_mul_128(x, POW10_I128[e] as u128);
    U256::from_lo_hi(lo, hi)
}

/// Returns x / d and x % d for d != 0.
fn divrem_256_by_128(x: U256, d: u128) -> (U256, u128) {
    let (lo, hi) = x.lo_hi();
    let (q_lo, r) = divrem_256_128(lo, hi % d, d).unwrap();
    (U256::from_lo_hi(q_lo, hi / d), r)
}

/// Returns round(x / 10^e), with e <= 2 * DEC128_MAX_PREC, rounding to nearest
/// even. x is assumed to be < 2^255. Returns None if the result doesn't fit in
/// a u128.
fn div_255_pow10_wide(x: U256, e: usize) -> Option<u128> {
    if e <= DEC128_MAX_PREC {
        return div_255_pow10(x, e);
    }

    let (q, r) = divrem_256_by_128(x, POW10_I128[DEC128_MAX_PREC] as u128);
    let d = POW10_I128[e - DEC128_MAX_PREC] as u128;
    let (q, rq) = divrem_256_by_128(q, d);
    let mut q = q.to_u128()?;
    let half = d / 2;
    if rq > half || (rq == half && (r != 0 || q % 2 == 1)) {
        q += 1;
    }
    Some(q)
}

#[inline]
fn signed_dec128(neg: bool, mag: u128) -> Option<i128> {
    if mag >= POW10_I128[DEC128_MAX_PREC] as u128 {
        return None;
    }
    Some(if neg { -(mag as i128) } else { mag as i128 })
}

/// Rescales x from scale s (<= 2 * DEC128_MAX_PREC) to s_out, rounding to
/// nearest even, returning None if the result doesn't fit a Decimal128.
fn i256_to_dec128(x: I256, s: usize, s_out: usize) -> Option<i128> {
    let mag = if s_out <= s {
        div_255_pow10_wide(x.mag, s - s_out)?
    } else {
        x.mag
            .to_u128()?
            .checked_mul(POW10_I128[s_out - s] as u128)?
    };
    signed_dec128(x.neg, mag)
}

/// Rescales x from scale s to s_out, rounding to nearest even, returning None
/// if the result doesn't fit a Decimal128.
#[inline]
fn i128_to_dec128_scaled(x: i128, s: usize, s_out: usize) -> Option<i128> {
    let r = if s_out < s {
        div_128_pow10(x, s - s_out)
    } else {
        x.checked_mul(POW10_I128[s_out - s])?
    };
    dec128_fits(r, DEC128_MAX_PREC).then_some(r)
}

fn dec128_add_sub_scaled(
    l: i128,
    sl: usize,
    r: i128,
    sr: usize,
    s_out: usize,
    sub: bool,
) -> Option<i128> {
    let s = sl.max(sr);
    if let (Some(la), Some(ra)) = (
        l.checked_mul(POW10_I128[s - sl]),
        r.checked_mul(POW10_I128[s - sr]),
    ) {
        let res = if sub {
            la.checked_sub(ra)
        } else {
            la.checked_add(ra)
        };
        if let Some(res) = res {
            return i128_to_dec128_scaled(res, s, s_out);
        }
    }

    // An aligned operand overflows i128, the result may still fit.
    let a = I256 {
        neg: l < 0,
        mag: u256_mul_pow10(l.unsigned_abs(), s - sl),
    };
    let b = I256 {
        neg: (r < 0) ^ sub,
        mag: u256_mul_pow10(r.unsigned_abs(), s - sr),
    };
    let res = if a.neg == b.neg {
        I256 {
            neg: a.neg,
            mag: a.mag.add(b.mag),
        }
    } else if a.mag >= b.mag {
        I256 {
            neg: a.neg,
            mag: a.mag.sub(b.mag),
        }
    } else {
        I256 {
            neg: b.neg,
            mag: b.mag.sub(a.mag),
        }
    };
    i256_to_dec128(res, s, s_out)
}

/// Adds Decimal128s l (scale sl) and r (scale sr), returning the sum at scale
/// s_out rounded to nearest even, or None if it doesn't fit a Decimal128.
#[inline]
pub fn dec128_add_scaled(l: i128, sl: usize, r: i128, sr: usize, s_out: usize) -> Option<i128> {
    if sl == sr && sr == s_out {
        return l
            .checked_add(r)
            .filter(|x| dec128_fits(*x, DEC128_MAX_PREC));
    }
    dec128_add_sub_scaled(l, sl, r, sr, s_out, false)
}

/// Subtracts Decimal128s l (scale sl) and r (scale sr), returning the
/// difference at scale s_out rounded to nearest even, or None if it doesn't fit
/// a Decimal128.
#[inline]
pub fn dec128_sub_scaled(l: i128, sl: usize, r: i128, sr: usize, s_out: usize) -> Option<i128> {
    if sl == sr && sr == s_out {
        return l
            .checked_sub(r)
            .filter(|x| dec128_fits(*x, DEC128_MAX_PREC));
    }
    dec128_add_sub_scaled(l, sl, r, sr, s_out, true)
}

/// Multiplies Decimal128s l (scale sl) and r (scale sr), returning the product
/// at scale s_out rounded to nearest even, or None if it doesn't fit a
/// Decimal128.
#[inline]
pub fn dec128_mul_scaled(l: i128, sl: usize, r: i128, sr: usize, s_out: usize) -> Option<i128> {
    let s = sl + sr;
    if let (Ok(a), Ok(b)) = (i64::try_from(l), i64::try_from(r))
        && s <= s_out + DEC128_MAX_PREC
    {
        return i128_to_dec128_scaled(a as i128 * b as i128, s, s_out);
    }

    let (lo, hi) = widening_mul_128(l.unsigned_abs(), r.unsigned_abs());
    let prod = I256 {
        neg: (l < 0) ^ (r < 0),
        mag: U256::from_lo_hi(lo, hi),
    };
    i256_to_dec128(prod, s, s_out)
}

/// Divides Decimal128s l (scale sl) and r (scale sr), returning the quotient at
/// scale s_out rounded to nearest even, or None if r is zero or the quotient
/// doesn't fit a Decimal128.
#[inline]
pub fn dec128_div_scaled(l: i128, sl: usize, r: i128, sr: usize, s_out: usize) -> Option<i128> {
    if r == 0 {
        return None;
    }

    // Computes round(l * 10^k / r) with k = s_out + sr - sl.
    let lu = l.unsigned_abs();
    let ru = r.unsigned_abs();
    if s_out + sr >= sl && s_out + sr - sl <= 18 && lu < 1 << 63 {
        // Fast path: l * 10^k < 2^63 * 10^18 < 2^123, so adding r / 2 < 2^127 can't
        // overflow a u128.
        let z = lu * POW10_I128[s_out + sr - sl] as u128 + ru / 2;
        let (mut q, rem) = match (u64::try_from(z), u64::try_from(ru)) {
            // A native 64-bit division is much cheaper than a 128-bit one.
            (Ok(z), Ok(d)) => ((z / d) as u128, (z % d) as u128),
            _ => (z / ru, z % ru),
        };
        if ru.is_multiple_of(2) && rem == 0 && !q.is_multiple_of(2) {
            q -= 1;
        }
        return signed_dec128((l < 0) ^ (r < 0), q);
    }
    if s_out + sr >= sl && s_out + sr - sl <= DEC128_MAX_PREC {
        // l * 10^k < 2^255 fits the (lo, hi) pair, and so does adding r / 2.
        let (lo, hi) = widening_mul_128(lu, POW10_I128[s_out + sr - sl] as u128);
        let (lo, carry) = lo.overflowing_add(ru / 2);
        let hi = hi + carry as u128;
        let (mut q, rem) = if hi == 0 {
            (lo / ru, lo % ru)
        } else {
            // hi >= r means the quotient is at least 2^128, which doesn't fit.
            divrem_256_128(lo, hi, ru)?
        };
        if ru.is_multiple_of(2) && rem == 0 && !q.is_multiple_of(2) {
            q -= 1;
        }
        return signed_dec128((l < 0) ^ (r < 0), q);
    }
    dec128_div_scaled_wide(l, sl, r, sr, s_out)
}

/// [`dec128_div_scaled`] for a dividend shift above 38 or below 0.
#[cold]
fn dec128_div_scaled_wide(l: i128, sl: usize, r: i128, sr: usize, s_out: usize) -> Option<i128> {
    let lu = l.unsigned_abs();
    let ru = r.unsigned_abs();
    let (n, d) = if s_out + sr >= sl {
        let k = s_out + sr - sl;
        let n = if k <= DEC128_MAX_PREC {
            u256_mul_pow10(lu, k)
        } else {
            let t = u256_mul_pow10(lu, DEC128_MAX_PREC);
            let m = U256::from_u128(POW10_I128[k - DEC128_MAX_PREC] as u128);
            let (lo, hi) = widening_mul_256(t, m);
            // If n >= 2^255 then n / r > 2^128, which doesn't fit.
            if hi != U256([0; 4]) || lo.0[3] >> 63 != 0 {
                return None;
            }
            lo
        };
        (n, ru)
    } else {
        let (d, d_hi) = widening_mul_128(ru, POW10_I128[sl - sr - s_out] as u128);
        if d_hi != 0 {
            // |l| < 2^127 <= d / 2, so the quotient rounds to zero.
            return Some(0);
        }
        (U256::from_u128(lu), d)
    };

    let z = n.add(U256::from_u128(d / 2));
    let (lo, hi) = z.lo_hi();
    // hi >= d means the quotient is at least 2^128, which doesn't fit.
    let (mut q, rem) = divrem_256_128(lo, hi, d)?;
    if d % 2 == 0 && rem == 0 && q % 2 == 1 {
        q -= 1;
    }
    signed_dec128((l < 0) ^ (r < 0), q)
}

/// The integer quotient q and remainder of l (scale sl) by r (scale sr), with
/// l = q * r + rem and the remainder at scale max(sl, sr). Truncating gives the
/// remainder the sign of l, flooring the sign of r. Returns None if r is zero.
fn dec128_divrem(l: i128, sl: usize, r: i128, sr: usize, floor: bool) -> Option<(I256, I256)> {
    if r == 0 {
        return None;
    }
    let s = sl.max(sr);
    let lm = u256_mul_pow10(l.unsigned_abs(), s - sl);
    let rm = u256_mul_pow10(r.unsigned_abs(), s - sr);
    let zero = U256::from_u128(0);
    let (q, rem) = match rm.to_u128() {
        Some(d) => {
            let (q, rem) = divrem_256_by_128(lm, d);
            (q, U256::from_u128(rem))
        },
        // Only an operand below scale s can exceed a u128 when aligned, so here
        // |l| < 10^38 < |r|.
        None => (zero, lm),
    };
    let neg = (l < 0) != (r < 0);
    if floor && neg && rem != zero {
        let q = I256 {
            neg: true,
            mag: q.add(U256::from_u128(1)),
        };
        let rem = I256 {
            neg: r < 0,
            mag: rm.sub(rem),
        };
        return Some((q, rem));
    }
    Some((
        I256 { neg, mag: q },
        I256 {
            neg: l < 0,
            mag: rem,
        },
    ))
}

/// Returns the remainder of Decimal128s l (scale sl) and r (scale sr) at scale
/// s_out, rounded to nearest even: with the sign of r if `floor`, else with the
/// sign of l. Returns None if r is zero or the result doesn't fit a Decimal128.
pub fn dec128_rem_scaled(
    l: i128,
    sl: usize,
    r: i128,
    sr: usize,
    s_out: usize,
    floor: bool,
) -> Option<i128> {
    let (_, rem) = dec128_divrem(l, sl, r, sr, floor)?;
    i256_to_dec128(rem, sl.max(sr), s_out)
}

/// Returns the integer quotient of Decimal128s l (scale sl) and r (scale sr),
/// rounded down if `floor`, else toward zero, as a Decimal128 at scale s_out.
/// Returns None if r is zero or the result doesn't fit a Decimal128.
pub fn dec128_int_div_scaled(
    l: i128,
    sl: usize,
    r: i128,
    sr: usize,
    s_out: usize,
    floor: bool,
) -> Option<i128> {
    let (q, _) = dec128_divrem(l, sl, r, sr, floor)?;
    i256_to_dec128(q, 0, s_out)
}

/// Returns x * 10^e, saturating at the i128 bounds, with e <= DEC128_MAX_PREC.
///
/// Rescaling one side of a Decimal128 comparison with this preserves the
/// result: a saturated value compares like the true value against any
/// Decimal128.
#[inline]
pub fn dec128_upscale_saturating(x: i128, e: usize) -> i128 {
    x.saturating_mul(POW10_I128[e])
}

/// Checks if two Decimal128s are equal in value.
#[inline]
pub fn dec128_eq(mut lv: i128, ls: usize, mut rv: i128, rs: usize) -> bool {
    // Rescale to largest scale. If this overflows the numbers can't be equal anyway.
    if ls < rs {
        let Some(scaled_lv) = mul_128_pow10(lv, rs - ls) else {
            return false;
        };
        lv = scaled_lv;
    } else if ls > rs {
        let Some(scaled_rv) = mul_128_pow10(rv, ls - rs) else {
            return false;
        };
        rv = scaled_rv;
    }

    lv == rv
}

/// Checks how two Decimal128s compare.
#[inline]
pub fn dec128_cmp(mut lv: i128, ls: usize, mut rv: i128, rs: usize) -> Ordering {
    // Rescale to largest scale. If this overflows we know the magnitude of the
    // (attempted) rescaled number is larger and we can resolve the answer just
    // using its sign.
    if ls < rs {
        let Some(scaled_lv) = mul_128_pow10(lv, rs - ls) else {
            return if lv < 0 {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        };
        lv = scaled_lv;
    } else if ls > rs {
        let Some(scaled_rv) = mul_128_pow10(rv, ls - rs) else {
            return if 0 < rv {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        };
        rv = scaled_rv;
    }

    lv.cmp(&rv)
}

/// Get the sign (either -1, 0, 1) of a Decimal128.
#[inline]
pub fn dec128_sign(x: i128, s: usize) -> i128 {
    if x < 0 {
        -POW10_I128[s]
    } else if x > 0 {
        POW10_I128[s]
    } else {
        0
    }
}

/// Deserialize bytes to a single i128 representing a decimal, at a specified
/// precision and scale. The number is checked to ensure it fits within the
/// specified precision and scale.  Consistent with float parsing, no decimal
/// separator is required (eg "500", "500.", and "500.0" are all accepted);
/// this allows mixed integer/decimal sequences to be parsed as decimals.
/// Returns None if the number is not well-formed, or does not fit. The decimal
/// separator defaults to b'.', but can be changed to b',' by setting `decimal_comma`
/// to true.
pub fn str_to_dec128(bytes: &[u8], p: usize, s: usize, decimal_comma: bool) -> Option<i128> {
    assert!(dec128_verify_prec_scale(p, s).is_ok());
    let exp_pos = bytes
        .iter()
        .position(|b| *b == b'e' || *b == b'E')
        .unwrap_or(bytes.len());
    let (num_bytes, exp_bytes) = bytes.split_at(exp_pos);
    let decimal_sep = if decimal_comma { b',' } else { b'.' };
    let sep_pos = num_bytes
        .iter()
        .position(|b| *b == decimal_sep)
        .unwrap_or(num_bytes.len());
    let (mut int_bytes, mut frac_bytes) = num_bytes.split_at(sep_pos);

    if frac_bytes.is_empty() && exp_bytes.is_empty() {
        // Integer-only fast path.
        let n: i128 = atoi_simd::parse::<_, true, true>(int_bytes).ok()?;
        return i128_to_dec128(n, p, s);
    }

    // Skip sign and separator to get clean integers.
    let negative = match int_bytes.first() {
        Some(sign @ (b'+' | b'-')) => {
            int_bytes = &int_bytes[1..];
            *sign == b'-'
        },
        _ => false,
    };

    if !frac_bytes.is_empty() {
        frac_bytes = &frac_bytes[1..];
    }

    if frac_bytes.is_empty() && int_bytes.is_empty() {
        // No digits at all.
        return None;
    }

    let exp_scale: i32 = if !exp_bytes.is_empty() {
        atoi_simd::parse::<_, true, true>(&exp_bytes[1..]).ok()?
    } else {
        0
    };

    // Round if digits extend beyond the scale.
    let comb_scale = exp_scale + s as i32;
    let (next_digit, all_zero_after);
    let (int_scale, frac_scale) = if comb_scale < 0 {
        let int_part_len = int_bytes.len().saturating_sub((-comb_scale) as usize);
        if !int_bytes[int_part_len..]
            .iter()
            .chain(frac_bytes)
            .all(|b| b.is_ascii_digit())
        {
            return None;
        }
        if (-comb_scale) as usize > int_bytes.len() {
            // All digits are valid (so no error), but also irrelevant.
            return Some(0);
        }

        next_digit = int_bytes[int_part_len];
        all_zero_after = int_bytes[int_part_len + 1..]
            .iter()
            .chain(frac_bytes)
            .all(|b| *b == b'0');
        int_bytes = &int_bytes[..int_part_len];
        frac_bytes = &[];
        (0, 0)
    } else if frac_bytes.len() > comb_scale as usize {
        let cs = comb_scale as usize;
        if !frac_bytes[cs..].iter().all(|b| b.is_ascii_digit()) {
            return None;
        }
        next_digit = frac_bytes[cs];
        all_zero_after = frac_bytes[cs + 1..].iter().all(|b| *b == b'0');
        frac_bytes = &frac_bytes[..cs];
        (cs, 0)
    } else {
        let cs = comb_scale as usize;
        next_digit = b'0';
        all_zero_after = true;
        (cs, cs - frac_bytes.len())
    };

    // Parse parts.
    let mut pint = if int_bytes.is_empty() {
        0
    } else {
        atoi_simd::parse_pos::<i128, true>(int_bytes).ok()?
    };

    let mut pfrac = if frac_bytes.is_empty() {
        0
    } else {
        atoi_simd::parse_pos::<i128, true>(frac_bytes).ok()?
    };

    // Round-to-even.
    if next_digit > b'5' || next_digit == b'5' && !all_zero_after {
        pfrac += 1;
    } else if next_digit == b'5' {
        if comb_scale <= 0 {
            pint += pint % 2;
        } else {
            pfrac += pfrac % 2;
        }
    }

    // Apply scales.
    if pint > 0 {
        if int_scale > DEC128_MAX_PREC {
            return None;
        }
        pint = mul_128_pow10(pint, int_scale)?;
    }

    if pfrac > 0 {
        if frac_scale > DEC128_MAX_PREC {
            return None;
        }
        pfrac = mul_128_pow10(pfrac, frac_scale)?;
    }

    let ret = pint + pfrac;
    if !dec128_fits(ret, p) {
        return None;
    }
    if negative { Some(-ret) } else { Some(ret) }
}

const DEC128_MAX_LEN: usize = 39 + 2;

#[derive(Clone, Copy)]
pub struct DecimalFmtBuffer {
    data: [u8; DEC128_MAX_LEN],
    len: usize,
}

impl Default for DecimalFmtBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl DecimalFmtBuffer {
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: [0; DEC128_MAX_LEN],
            len: 0,
        }
    }

    pub fn format_dec128(
        &mut self,
        x: i128,
        scale: usize,
        trim_zeros: bool,
        decimal_comma: bool,
    ) -> &str {
        let decimal_sep = if decimal_comma { b',' } else { b'.' };

        let mut itoa_buf = itoa::Buffer::new();
        let xs = itoa_buf.format(x.unsigned_abs()).as_bytes();

        if x >= 0 {
            self.len = 0;
        } else {
            self.data[0] = b'-';
            self.len = 1;
        }

        if scale == 0 {
            self.data[self.len..self.len + xs.len()].copy_from_slice(xs);
            self.len += xs.len();
        } else {
            let whole_len = xs.len().saturating_sub(scale);
            let frac_len = xs.len() - whole_len;
            if whole_len == 0 {
                self.data[self.len] = b'0';
                self.data[self.len + 1] = decimal_sep;
                self.data[self.len + 2..self.len + 2 + scale - frac_len].fill(b'0');
                self.len += 2 + scale - frac_len;
            } else {
                self.data[self.len..self.len + whole_len].copy_from_slice(&xs[..whole_len]);
                self.data[self.len + whole_len] = decimal_sep;
                self.len += whole_len + 1;
            }

            self.data[self.len..self.len + frac_len].copy_from_slice(&xs[whole_len..]);
            self.len += frac_len;

            if trim_zeros {
                while self.data.get(self.len - 1) == Some(&b'0') {
                    self.len -= 1;
                }
                if self.data.get(self.len - 1) == Some(&decimal_sep) {
                    self.len -= 1;
                }
            }
        }

        unsafe { std::str::from_utf8_unchecked(&self.data[..self.len]) }
    }
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use bigdecimal::{BigDecimal, RoundingMode};
    use num_bigint::BigInt;
    use num_traits::Signed;
    use polars_utils::aliases::PlHashSet;
    use rand::prelude::*;

    use super::*;

    #[test]
    fn test_exact_scalar_arithmetic() {
        // 0.06 + 0.01 = 0.07, exact at scale 2
        assert_eq!(exact::add((6, 2), (1, 2)), Some((7, 2)));
        // 1.5 - 3 aligns to the larger scale
        assert_eq!(exact::sub((15, 1), (3, 0)), Some((-15, 1)));
        // 1.10 * 1.10 = 1.2100 at the summed scale
        assert_eq!(exact::mul((110, 2), (110, 2)), Some((12100, 4)));
        assert_eq!(exact::neg((5, 1)), Some((-5, 1)));

        // scales beyond DEC128_MAX_PREC stay exact
        assert_eq!(exact::mul((1, 38), (1, 38)), Some((1, 76)));
        assert_eq!(exact::add((0, 0), (1, 76)), Some((1, 76)));
        assert_eq!(i128_mul_pow10(0, 200), Some(0));

        // overflow is reported rather than wrapped
        assert_eq!(exact::add((i128::MAX, 0), (1, 0)), None);
        assert_eq!(exact::add((1, 0), (1, 39)), None);
        assert_eq!(exact::mul((i128::MAX, 0), (2, 0)), None);
        assert_eq!(exact::neg((i128::MIN, 0)), None);
        assert_eq!(i128_mul_pow10(1, 39), None);
    }

    fn bigdecimal_to_dec128(x: &BigDecimal, p: usize, s: usize) -> Option<i128> {
        let n = x
            .with_scale_round(s as i64, RoundingMode::HalfEven)
            .into_bigint_and_scale()
            .0;
        if n.abs() < POW10_I128[p].into() {
            Some(n.try_into().unwrap())
        } else {
            None
        }
    }

    fn dec128_to_bigdecimal(x: i128, s: usize) -> BigDecimal {
        BigDecimal::from_bigint(BigInt::from(x), s as i64)
    }

    fn round_half_even_div(n: &BigInt, d: &BigInt) -> BigInt {
        let neg = n.is_negative() ^ d.is_negative();
        let (n, d) = (n.abs(), d.abs());
        let mut q = &n / &d;
        let twice_rem = (&n % &d) * 2;
        if twice_rem > d || (twice_rem == d && (&q % 2u8) == BigInt::from(1u8)) {
            q += 1u8;
        }
        if neg { -q } else { q }
    }

    fn pow10_big(e: usize) -> BigInt {
        BigInt::from(10u8).pow(e as u32)
    }

    fn big_to_dec128(x: BigInt) -> Option<i128> {
        (x.abs() < BigInt::from(POW10_I128[DEC128_MAX_PREC])).then(|| x.try_into().unwrap())
    }

    fn add_oracle(l: i128, sl: usize, r: i128, sr: usize, s_out: usize, sub: bool) -> Option<i128> {
        let r = if sub {
            -BigInt::from(r)
        } else {
            BigInt::from(r)
        };
        let sum = dec128_to_bigdecimal(l, sl) + BigDecimal::from_bigint(r, sr as i64);
        bigdecimal_to_dec128(&sum, DEC128_MAX_PREC, s_out)
    }

    fn mul_oracle(l: i128, sl: usize, r: i128, sr: usize, s_out: usize) -> Option<i128> {
        let prod = dec128_to_bigdecimal(l, sl) * dec128_to_bigdecimal(r, sr);
        bigdecimal_to_dec128(&prod, DEC128_MAX_PREC, s_out)
    }

    fn div_oracle(l: i128, sl: usize, r: i128, sr: usize, s_out: usize) -> Option<i128> {
        if r == 0 {
            return None;
        }
        let n = BigInt::from(l) * pow10_big(s_out + sr);
        let d = BigInt::from(r) * pow10_big(sl);
        big_to_dec128(round_half_even_div(&n, &d))
    }

    /// The integer quotient and remainder of l / r at the common scale, truncating or
    /// flooring the quotient.
    fn divrem_oracle(
        l: i128,
        sl: usize,
        r: i128,
        sr: usize,
        floor: bool,
    ) -> Option<(BigInt, BigInt, usize)> {
        if r == 0 {
            return None;
        }
        let s = sl.max(sr);
        let a = BigInt::from(l) * pow10_big(s - sl);
        let b = BigInt::from(r) * pow10_big(s - sr);
        // BigInt division truncates toward zero.
        let mut q = &a / &b;
        let mut m = &a - &q * &b;
        if floor && m != BigInt::from(0) && m.is_negative() != b.is_negative() {
            q -= 1u8;
            m += &b;
        }
        Some((q, m, s))
    }

    fn rem_oracle(
        l: i128,
        sl: usize,
        r: i128,
        sr: usize,
        s_out: usize,
        floor: bool,
    ) -> Option<i128> {
        let (_, m, s) = divrem_oracle(l, sl, r, sr, floor)?;
        bigdecimal_to_dec128(
            &BigDecimal::from_bigint(m, s as i64),
            DEC128_MAX_PREC,
            s_out,
        )
    }

    fn int_div_oracle(
        l: i128,
        sl: usize,
        r: i128,
        sr: usize,
        s_out: usize,
        floor: bool,
    ) -> Option<i128> {
        let (q, _, _) = divrem_oracle(l, sl, r, sr, floor)?;
        big_to_dec128(q * pow10_big(s_out))
    }

    fn check_all_scaled(l: i128, sl: usize, r: i128, sr: usize, s_out: usize) {
        let ctx = format!("l={l}, sl={sl}, r={r}, sr={sr}, s_out={s_out}");
        for floor in [false, true] {
            assert_eq!(
                dec128_rem_scaled(l, sl, r, sr, s_out, floor),
                rem_oracle(l, sl, r, sr, s_out, floor),
                "rem floor={floor} {ctx}"
            );
            assert_eq!(
                dec128_int_div_scaled(l, sl, r, sr, s_out, floor),
                int_div_oracle(l, sl, r, sr, s_out, floor),
                "int_div floor={floor} {ctx}"
            );
        }
        assert_eq!(
            dec128_add_scaled(l, sl, r, sr, s_out),
            add_oracle(l, sl, r, sr, s_out, false),
            "add {ctx}"
        );
        assert_eq!(
            dec128_sub_scaled(l, sl, r, sr, s_out),
            add_oracle(l, sl, r, sr, s_out, true),
            "sub {ctx}"
        );
        assert_eq!(
            dec128_mul_scaled(l, sl, r, sr, s_out),
            mul_oracle(l, sl, r, sr, s_out),
            "mul {ctx}"
        );
        assert_eq!(
            dec128_div_scaled(l, sl, r, sr, s_out),
            div_oracle(l, sl, r, sr, s_out),
            "div {ctx}"
        );
        let expected_cmp = dec128_to_bigdecimal(l, sl).cmp(&dec128_to_bigdecimal(r, sr));
        assert_eq!(dec128_cmp(l, sl, r, sr), expected_cmp, "cmp {ctx}");
        assert_eq!(
            dec128_eq(l, sl, r, sr),
            expected_cmp == Ordering::Equal,
            "eq {ctx}"
        );
        let (ul, ur) = if sl < sr {
            (dec128_upscale_saturating(l, sr - sl), r)
        } else {
            (l, dec128_upscale_saturating(r, sl - sr))
        };
        assert_eq!(ul.cmp(&ur), expected_cmp, "upscale cmp {ctx}");
    }

    static SCALED_GRID: [usize; 6] = [0, 1, 6, 18, 37, 38];

    #[test]
    fn test_scaled_arith_against_bigdecimal() {
        let max = POW10_I128[DEC128_MAX_PREC] - 1;
        let mut r = SmallRng::seed_from_u64(42);
        let mut base: Vec<i128> = vec![0, 1, 2, 5, 9, max, max - 1];
        base.extend((0..39).map(|e| POW10_I128[e]));
        base.extend((0..39).map(|e| POW10_I128[e] / 2));
        base.extend((0..127).map(|e| 1i128 << e).filter(|x| *x <= max));
        base.extend((0..16).map(|_| r.random::<u32>() as i128));
        base.extend((0..16).map(|_| r.random::<u64>() as i128));
        base.extend((0..16).map(|_| (r.random::<u128>() % (max as u128 + 1)) as i128));
        base.extend(base.clone().into_iter().map(|x| -x));

        for &sl in &SCALED_GRID {
            for &sr in &SCALED_GRID {
                for &s_out in &SCALED_GRID {
                    for _ in 0..200 {
                        let l = *base.choose(&mut r).unwrap();
                        let rv = *base.choose(&mut r).unwrap();
                        check_all_scaled(l, sl, rv, sr, s_out);
                    }
                }
            }
        }
    }

    #[test]
    fn test_dec128_to_float_is_correctly_rounded() {
        let mut r = SmallRng::seed_from_u64(42);
        let mut base: Vec<i128> = vec![0, 1, 9_728_340_843_400_927, (1 << 53) + 1, (1 << 24) + 1];
        base.extend((0..39).map(|e| POW10_I128[e] - 1));
        base.extend((0..64).map(|_| r.random::<u32>() as i128));
        base.extend((0..64).map(|_| r.random::<u64>() as i128));
        base.extend(
            (0..64).map(|_| (r.random::<u128>() % POW10_I128[DEC128_MAX_PREC] as u128) as i128),
        );
        base.extend(base.clone().into_iter().map(|x| -x));
        for &x in &base {
            for s in 0..=DEC128_MAX_PREC {
                let spelled = format!("{x}e-{s}");
                assert_eq!(
                    dec128_to_f64(x, s),
                    spelled.parse::<f64>().unwrap(),
                    "{spelled}"
                );
                assert_eq!(
                    dec128_to_f32(x, s),
                    spelled.parse::<f32>().unwrap(),
                    "{spelled}"
                );
            }
        }
        // Dividing the rounded mantissa by 10^16 is one ULP off here.
        assert_eq!(dec128_to_f64(9_728_340_843_400_927, 16), 0.9728340843400927);
        // 1 + 2^-24 is halfway between two f32s and rounds to even. Just above it,
        // the nearest f64 is that halfway point, so rounding it again is wrong.
        assert_eq!(dec128_to_f32(1_000_000_059_604_644_775_390_625, 24), 1.0);
        assert_eq!(
            dec128_to_f32(1_000_000_059_604_644_775_390_625_000_001, 30),
            1.0 + f32::EPSILON
        );
    }

    #[test]
    fn test_scaled_arith_targeted() {
        let max = POW10_I128[DEC128_MAX_PREC] - 1;
        let p38 = POW10_I128[DEC128_MAX_PREC];

        // Signed half-even ties at scale 6: ±0.0000025 -> ±0.000002, ±0.0000035 -> ±0.000004.
        for sign in [1, -1] {
            assert_eq!(dec128_mul_scaled(sign * 25, 6, 1, 1, 6), Some(sign * 2));
            assert_eq!(dec128_mul_scaled(sign * 35, 6, 1, 1, 6), Some(sign * 4));
            assert_eq!(dec128_div_scaled(sign * 5, 6, 2, 0, 6), Some(sign * 2));
            assert_eq!(dec128_div_scaled(sign * 7, 6, 2, 0, 6), Some(sign * 4));
            // Ties that only exist in the 256-bit product, e.g. 0.5 * 10^37 * 10^-37.
            assert_eq!(
                dec128_mul_scaled(sign * 5 * POW10_I128[37], 38, POW10_I128[37], 37, 0),
                Some(0)
            );
            assert_eq!(
                dec128_mul_scaled(sign * 15 * POW10_I128[36], 38, POW10_I128[37], 36, 0),
                Some(sign * 2)
            );
            assert_eq!(
                dec128_mul_scaled(sign * 25 * POW10_I128[36], 38, POW10_I128[37], 36, 0),
                Some(sign * 2)
            );
        }

        // Extreme scales, including a negative and a > 38 division exponent k.
        assert_eq!(dec128_mul_scaled(max, 0, 1, 38, 38), Some(max));
        assert_eq!(dec128_mul_scaled(max, 0, 1, 38, 0), Some(1));
        assert_eq!(dec128_mul_scaled(max, 0, 1, 37, 38), None);
        assert_eq!(dec128_mul_scaled(7, 0, 3, 38, 38), Some(21));
        assert_eq!(dec128_mul_scaled(7, 1, 3, 1, 38), Some(21 * POW10_I128[36]));
        assert_eq!(dec128_div_scaled(1, 38, 3, 0, 38), Some(0));
        assert_eq!(dec128_div_scaled(max, 38, 1, 38, 0), Some(max));
        assert_eq!(dec128_div_scaled(5 * POW10_I128[37], 38, 1, 0, 0), Some(0));
        assert_eq!(dec128_div_scaled(15 * POW10_I128[36], 37, 1, 0, 0), Some(2));
        assert_eq!(dec128_div_scaled(1, 38, max, 0, 0), Some(0));
        assert_eq!(dec128_div_scaled(3, 0, 1, 38, 0), None);
        assert_eq!(
            dec128_div_scaled(1, 0, 3, 38, 0),
            Some(POW10_I128[37] * 10 / 3)
        );
        assert_eq!(dec128_div_scaled(1, 0, max, 38, 38), None);

        // Remainder and integer quotient: truncating follows the dividend's sign,
        // flooring the divisor's.
        let (rem, idiv) = (dec128_rem_scaled, dec128_int_div_scaled);
        assert_eq!(rem(-75, 1, 2, 0, 1, false), Some(-15));
        assert_eq!(rem(-75, 1, 2, 0, 1, true), Some(5));
        assert_eq!(rem(75, 1, -2, 0, 1, false), Some(15));
        assert_eq!(rem(75, 1, -2, 0, 1, true), Some(-5));
        assert_eq!(rem(-80, 1, 2, 0, 1, true), Some(0));
        assert_eq!(idiv(-75, 1, 2, 0, 0, false), Some(-3));
        assert_eq!(idiv(-75, 1, 2, 0, 0, true), Some(-4));
        assert_eq!(idiv(75, 1, -2, 0, 0, true), Some(-4));
        assert_eq!(idiv(7, 0, 2, 0, 2, true), Some(300));
        // Aligning 10^37 to scale 1 exceeds 38 digits: 10^37 % 0.3 = 0.1.
        assert_eq!(rem(POW10_I128[37], 0, 3, 1, 1, false), Some(1));
        assert_eq!(idiv(POW10_I128[37], 0, 3, 1, 0, false), Some(p38 / 3));
        // 10^37 aligned to scale 38 exceeds a u128: 10^-38 % 10^37 = 10^-38.
        assert_eq!(rem(1, 38, POW10_I128[37], 0, 38, false), Some(1));
        assert_eq!(rem(-1, 38, POW10_I128[37], 0, 38, false), Some(-1));
        assert_eq!(idiv(-1, 38, POW10_I128[37], 0, 0, false), Some(0));
        assert_eq!(idiv(-1, 38, POW10_I128[37], 0, 0, true), Some(-1));
        // Flooring it would need 75 digits.
        assert_eq!(rem(-1, 38, POW10_I128[37], 0, 38, true), None);
        assert_eq!(idiv(max, 0, 1, 38, 0, true), None);
        assert_eq!(rem(max, 0, -max, 0, 0, true), Some(0));
        // |i128::MIN + 1| = 2^127 - 1, which is 1 mod 3.
        assert_eq!(rem(i128::MIN + 1, 0, 3, 0, 0, false), Some(-1));
        assert_eq!(rem(1, 0, 0, 0, 0, false), None);
        assert_eq!(idiv(1, 0, 0, 3, 0, true), None);

        // Results at the precision boundary.
        assert_eq!(dec128_add_scaled(max - 1, 0, 1, 0, 0), Some(max));
        assert_eq!(dec128_add_scaled(max, 0, 1, 0, 0), None);
        assert_eq!(dec128_sub_scaled(-max + 1, 0, 1, 0, 0), Some(-max));
        assert_eq!(dec128_sub_scaled(-max, 0, 1, 0, 0), None);
        assert_eq!(dec128_mul_scaled(max, 0, -1, 0, 0), Some(-max));
        assert_eq!(dec128_mul_scaled(p38 / 2, 0, 2, 0, 0), None);
        assert_eq!(dec128_div_scaled(max, 0, 1, 0, 0), Some(max));
        assert_eq!(dec128_div_scaled(-max, 0, 1, 0, 0), Some(-max));

        // i128::MIN-adjacent operands.
        for x in [i128::MIN, i128::MIN + 1, i128::MAX] {
            assert_eq!(dec128_add_scaled(x, 0, 0, 0, 0), None);
            assert_eq!(
                dec128_mul_scaled(x, 0, 1, 38, 0),
                mul_oracle(x, 0, 1, 38, 0)
            );
            assert_eq!(
                dec128_div_scaled(x, 38, 2, 0, 0),
                div_oracle(x, 38, 2, 0, 0)
            );
            assert_eq!(dec128_div_scaled(1, 0, x, 0, 6), div_oracle(1, 0, x, 0, 6));
            assert_eq!(dec128_sub_scaled(x, 0, x, 0, 0), Some(0));
        }

        // 10^35 / 0.001 = 10^38 doesn't fit at scale 3, 0.001 / 10^35 rounds to 0.
        assert_eq!(dec128_div_scaled(POW10_I128[37], 2, 1, 3, 3), None);
        assert_eq!(dec128_div_scaled(1, 3, POW10_I128[37], 2, 3), Some(0));
        // l * 10^k beyond 2^255.
        assert_eq!(dec128_div_scaled(max, 0, 1, 38, 38), None);
        assert_eq!(dec128_div_scaled(-max, 0, max, 38, 38), None);
        // Division by zero.
        assert_eq!(dec128_div_scaled(1, 2, 0, 3, 3), None);
        assert_eq!(dec128_div_scaled(0, 0, 0, 0, 0), None);

        // Cancellation: aligning 10^37 to scale 1 exceeds 38 digits, the result fits.
        let a = POW10_I128[37];
        let b = POW10_I128[38] - 1; // 9999...9.9 at scale 1
        assert_eq!(dec128_sub_scaled(a, 0, b, 1, 1), Some(1));
        assert_eq!(dec128_add_scaled(-a, 0, b, 1, 1), Some(-1));
        // Aligning 10^19 to scale 20 overflows i128, the result fits at scale 0.
        let y = 5 * POW10_I128[37];
        assert_eq!(
            dec128_sub_scaled(POW10_I128[19], 0, y, 20, 0),
            Some(95 * POW10_I128[17])
        );
        assert_eq!(
            dec128_add_scaled(y, 20, -POW10_I128[19], 0, 0),
            Some(-95 * POW10_I128[17])
        );
        assert_eq!(dec128_sub_scaled(POW10_I128[19], 0, y, 20, 20), None);
        let x = POW10_I128[20];
        assert_eq!(
            dec128_sub_scaled(x, 0, x * POW10_I128[18] - 1, 18, 18),
            Some(1)
        );
        assert_eq!(
            dec128_sub_scaled(x, 0, -(x * POW10_I128[18] - 1), 18, 18),
            None
        );
        assert_eq!(dec128_add_scaled(max, 0, -max, 0, 38), Some(0));

        // Comparisons at the 10^38 boundary and across scales.
        assert!(dec128_eq(10, 1, 100, 2));
        assert_eq!(dec128_cmp(max, 0, 1, 38), Ordering::Greater);
        assert_eq!(dec128_cmp(-max, 0, 1, 38), Ordering::Less);
        assert_eq!(dec128_cmp(max, 38, max, 0), Ordering::Less);
        assert_eq!(dec128_upscale_saturating(max, 38), i128::MAX);
        assert_eq!(dec128_upscale_saturating(-max, 38), i128::MIN);
        assert!(dec128_upscale_saturating(max, 1) > max);
        assert_eq!(dec128_upscale_saturating(10, 1), 100);
    }

    static INTERESTING_SCALE_PREC: [usize; 13] = [0, 1, 2, 3, 5, 8, 11, 16, 21, 27, 32, 37, 38];

    static INTERESTING_VALUES: LazyLock<Vec<BigDecimal>> = LazyLock::new(|| {
        let mut r = SmallRng::seed_from_u64(42);
        let mut base = Vec::new();
        base.extend((0..128).map(|e| BigDecimal::from(1i128 << e)));
        base.extend((0..39).map(|e| BigDecimal::from(POW10_I128[e])));
        base.extend((0..32).map(BigDecimal::from));
        base.extend((0..32).map(|_| BigDecimal::from(r.random::<u32>())));
        base.extend((0..32).map(|_| BigDecimal::from(r.random::<u64>())));
        base.extend((0..32).map(|_| BigDecimal::from(r.random::<u128>())));
        base.extend(base.clone().into_iter().map(|x| -x));

        let mut out = PlHashSet::default();
        out.extend(base.iter().cloned());

        let zero = BigDecimal::from(0u8);
        for l in &base {
            for r in &base {
                out.insert(l + r);
                out.insert(l * r);
                if *r != zero {
                    out.insert(l / r);
                }
            }
        }

        let mut out: Vec<_> = out.into_iter().collect();
        out.sort_by_key(|d| d.abs());
        out
    });

    #[test]
    #[rustfmt::skip]
    fn test_str_to_dec() {
        fn str_to_dec128_dot(bytes: &[u8], p: usize, s: usize) -> Option<i128> {
            str_to_dec128(bytes, p, s, false)
        }

        assert_eq!(str_to_dec128_dot(b"12.09", 8, 2), Some(1209));
        assert_eq!(str_to_dec128_dot(b"1200.90", 8, 2), Some(120090));
        assert_eq!(str_to_dec128_dot(b"143.9", 8, 2), Some(14390));

        assert_eq!(str_to_dec128_dot(b"+000000.5", 8, 2), Some(50));
        assert_eq!(str_to_dec128_dot(b"-0.5", 8, 2), Some(-50));
        assert_eq!(str_to_dec128_dot(b"-1.5", 8, 2), Some(-150));

        assert_eq!(str_to_dec128_dot(b"12ABC.34", 8, 5), None);
        assert_eq!(str_to_dec128_dot(b"1ABC2.34", 8, 5), None);
        assert_eq!(str_to_dec128_dot(b"12.3ABC4", 8, 5), None);
        assert_eq!(str_to_dec128_dot(b"12.3.ABC4", 8, 5), None);

        assert_eq!(str_to_dec128_dot(b"12.-3", 8, 5), None);
        assert_eq!(str_to_dec128_dot(b"", 8, 5), None);
        assert_eq!(str_to_dec128_dot(b"5.", 8, 5), Some(500000i128));
        assert_eq!(str_to_dec128_dot(b"5", 8, 5), Some(500000i128));
        assert_eq!(str_to_dec128_dot(b".5", 8, 5), Some(50000i128));

        // Precision and scale fitting.
        let val = b"1200";
        assert_eq!(str_to_dec128_dot(val, 4, 0), Some(1200));
        assert_eq!(str_to_dec128_dot(val, 3, 0), None);
        assert_eq!(str_to_dec128_dot(val, 4, 1), None);

        let val = b"1200.010";
        assert_eq!(str_to_dec128_dot(val, 7, 0), Some(1200));
        assert_eq!(str_to_dec128_dot(val, 7, 3), Some(1200010));
        assert_eq!(str_to_dec128_dot(val, 10, 6), Some(1200010000));
        assert_eq!(str_to_dec128_dot(val, 5, 3), None);
        assert_eq!(str_to_dec128_dot(val, 12, 5), Some(120001000));
        assert_eq!(str_to_dec128_dot(val, 38, 35), None);

        // Rounding.
        assert_eq!(str_to_dec128_dot(b"2.10", 5, 1), Some(21));
        assert_eq!(str_to_dec128_dot(b"2.14", 5, 1), Some(21));
        assert_eq!(str_to_dec128_dot(b"2.15", 5, 1), Some(22));
        assert_eq!(str_to_dec128_dot(b"2.24", 5, 1), Some(22));
        assert_eq!(str_to_dec128_dot(b"2.25", 5, 1), Some(22));
        assert_eq!(str_to_dec128_dot(b"2.26", 5, 1), Some(23));

        assert_eq!(str_to_dec128_dot(b"-.6", 5, 0), Some(-1));
        assert_eq!(str_to_dec128_dot(b"-.5", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b"-.4", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b"-.0", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b"-0", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b".4", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b".5", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b".6", 5, 0), Some(1));

        // Rounding + scientific.
        assert_eq!(str_to_dec128_dot(b"-6e-1", 5, 0), Some(-1));
        assert_eq!(str_to_dec128_dot(b"-5E-0001", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b"-4e-0000001", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b"0E-1", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b"4.e-1", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b"5.E-1", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b"6.e-1", 5, 0), Some(1));
        assert_eq!(str_to_dec128_dot(b"-.6e0", 5, 0), Some(-1));
        assert_eq!(str_to_dec128_dot(b"-.5e+0", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b"-.4e-0", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b"-.0e000", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b"-0e000000000", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b".4e0", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b".5e000", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b".6e0", 5, 0), Some(1));
        assert_eq!(str_to_dec128_dot(b"-.06e+1", 5, 0), Some(-1));
        assert_eq!(str_to_dec128_dot(b"-.05E+0001", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b"-.04e1", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b".0E+1", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b".004e02", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b".005E000002", 5, 0), Some(0));
        assert_eq!(str_to_dec128_dot(b".006e+2", 5, 0), Some(1));

        // Other scientific.
        assert_eq!(str_to_dec128_dot(b"600e-2", 5, 0), Some(6));
        assert_eq!(str_to_dec128_dot(b"600e-2", 5, 1), Some(60));
        assert_eq!(str_to_dec128_dot(b"600e-2", 5, 2), Some(600));
        assert_eq!(str_to_dec128_dot(b"600e-2", 5, 3), Some(6000));
        assert_eq!(str_to_dec128_dot(b"600e-2", 5, 4), Some(60000));
        assert_eq!(str_to_dec128_dot(b"1600e-3", 5, 4), Some(16000));
        assert_eq!(str_to_dec128_dot(b"1600e-2", 5, 4), None);
        assert_eq!(str_to_dec128_dot(b"123456000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-125", 10, 6), Some(1234560));

        assert_eq!(str_to_dec128_dot(b"99999999999999999999999999999999999999", 38, 0), Some(99999999999999999999999999999999999999));
        assert_eq!(str_to_dec128_dot(b"99999999999999999999999999999999999999e0", 38, 0), Some(99999999999999999999999999999999999999));
        assert_eq!(str_to_dec128_dot(b"999999999999999999999999999999999999990e-1", 38, 0), Some(99999999999999999999999999999999999999));
        assert_eq!(str_to_dec128_dot(b"999999999999999999999999999999999999994e-1", 38, 0), Some(99999999999999999999999999999999999999));
        assert_eq!(str_to_dec128_dot(b"999999999999999999999999999999999999995e-1", 38, 0), None);
        assert_eq!(str_to_dec128_dot(b"99999999999999999999999999999999999999.12345", 38, 0), Some(99999999999999999999999999999999999999));
        assert_eq!(str_to_dec128_dot(b"99999999999999999999999999999999999999.12345e0", 38, 0), Some(99999999999999999999999999999999999999));
        assert_eq!(str_to_dec128_dot(b"999999999999999999999999999999999999990.e-1", 38, 0), Some(99999999999999999999999999999999999999));
        assert_eq!(str_to_dec128_dot(b"999999999999999999999999999999999999994.99999e-1", 38, 0), Some(99999999999999999999999999999999999999));
        assert_eq!(str_to_dec128_dot(b"999999999999999999999999999999999999995.00000e-1", 38, 0), None);
        assert_eq!(str_to_dec128_dot( b"00.0000000000000000000000000000000000000000000000000000e+50", 5, 0), Some(0));

        // Decimal comma.
        assert_eq!(str_to_dec128(b"12,09", 8, 2, true), Some(1209));
        assert_eq!(str_to_dec128(b"1200,90", 8, 2, true), Some(120090));
        assert_eq!(str_to_dec128(b"143,9", 8, 2, true), Some(14390));

        // Edge-cases around missing components.
        assert_eq!(str_to_dec128_dot(b"0", 8, 2), Some(0));
        assert_eq!(str_to_dec128_dot(b"-0", 8, 2), Some(0));
        assert_eq!(str_to_dec128_dot(b"0.", 8, 2), Some(0));
        assert_eq!(str_to_dec128_dot(b".0", 8, 2), Some(0));
        assert_eq!(str_to_dec128_dot(b"-.0", 8, 2), Some(0));
        assert_eq!(str_to_dec128_dot(b"-0.", 8, 2), Some(0));
        assert_eq!(str_to_dec128_dot(b"-.", 8, 2), None);
        assert_eq!(str_to_dec128_dot(b"-", 8, 2), None);
        assert_eq!(str_to_dec128_dot(b".", 8, 2), None);
    }

    #[test]
    #[ignore = "very slow, meant for development"]
    fn test_str_to_dec_against_ref() {
        let exps = [0, 1, 3, 20, 50, 150];
        let digits = [0, 4, 5, 6, 9];
        let mut pos_parts = Vec::new();
        for leading_zeroes in exps {
            let leading = "0".repeat(leading_zeroes);
            for trailing_zeroes in exps {
                let trailing = "0".repeat(trailing_zeroes);
                for a in &digits {
                    for b in &digits {
                        pos_parts.push(format!("{leading}{a}{trailing}{b}"));
                    }
                }
            }
        }

        for scale in &INTERESTING_SCALE_PREC {
            for int in &pos_parts {
                for frac in &pos_parts {
                    for exp in &exps {
                        for exp_sign in &["+", "-"] {
                            let s = format!("{int}.{frac}e{exp_sign}{exp}");
                            let ours = str_to_dec128(s.as_bytes(), 38, *scale, false);
                            let theirs = BigDecimal::parse_bytes(s.as_bytes(), 10)
                                .and_then(|d| bigdecimal_to_dec128(&d, 38, *scale));
                            assert_eq!(ours, theirs, "input: {s}, scale: {scale}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "very slow, meant for development"]
    fn test_str_dec_roundtrip() {
        let mut buf = DecimalFmtBuffer::new();
        for &p in &INTERESTING_SCALE_PREC {
            for &s in &INTERESTING_SCALE_PREC {
                if s > p || p == 0 {
                    continue;
                }
                for x in INTERESTING_VALUES.iter() {
                    for d_comma in [true, false] {
                        if let Some(d) = bigdecimal_to_dec128(x, p, s) {
                            let fmt = buf.format_dec128(d, s, false, d_comma);
                            let d2 = str_to_dec128(fmt.as_bytes(), p, s, d_comma);
                            assert_eq!(d, d2.unwrap());
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }
}
