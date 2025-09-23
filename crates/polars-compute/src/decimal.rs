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
    x.abs() < POW10_I128[p]
}

#[inline]
pub fn dec128_to_i128(x: i128, s: usize) -> i128 {
    if s == 0 { x } else { div_128_pow10(x, s) }
}

/// Converts an i128 to a Decimal128 with the given precision and scale,
/// returning None if the value doesn't fit.
#[inline]
pub fn i128_to_dec128(x: i128, p: usize, s: usize) -> Option<i128> {
    let r = x.checked_mul(POW10_I128[s])?;
    dec128_fits(r, p).then_some(r)
}

/// Converts a Decimal128 with the given scale to a f64.
#[inline]
pub fn dec128_to_f64(x: i128, s: usize) -> f64 {
    // TODO: correctly rounded result. This rounds multiple times.
    x as f64 / POW10_F64[s]
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

/// Adds two Decimal128s, assuming they have the same scale.
#[inline]
pub fn dec128_add(l: i128, r: i128, p: usize) -> Option<i128> {
    l.checked_add(r).filter(|x| dec128_fits(*x, p))
}

/// Subs two Decimal128s, assuming they have the same scale.
#[inline]
pub fn dec128_sub(l: i128, r: i128, p: usize) -> Option<i128> {
    l.checked_sub(r).filter(|x| dec128_fits(*x, p))
}

/// Multiplies two Decimal128s, assuming they have the same scale s.
#[inline]
pub fn dec128_mul(l: i128, r: i128, p: usize, s: usize) -> Option<i128> {
    // Computes round(l * r / 10^s), rounding to nearest even.
    if let (Ok(ls), Ok(rs)) = (i64::try_from(l), i64::try_from(r)) {
        // Fast path, both small.
        let ret = div_128_pow10(ls as i128 * rs as i128, s);
        dec128_fits(ret, p).then_some(ret)
    } else {
        let negative = (l < 0) ^ (r < 0);
        let lu = l.unsigned_abs();
        let ru = r.unsigned_abs();

        let (lo, hi) = widening_mul_128(lu, ru);
        let retu = if hi == 0 && lo <= i128::MAX as u128 {
            div_128_pow10(lo as i128, s) as u128
        } else {
            div_255_pow10(U256::from_lo_hi(lo, hi), s)?
        };
        if retu >= POW10_I128[p] as u128 {
            return None;
        }
        if negative {
            Some(-(retu as i128))
        } else {
            Some(retu as i128)
        }
    }
}

/// Divides two Decimal128s, assuming they have the same scale s.
#[inline]
pub fn dec128_div(l: i128, r: i128, p: usize, s: usize) -> Option<i128> {
    if r == 0 {
        return None;
    }

    let negative = (l < 0) ^ (r < 0);
    let lu = l.unsigned_abs();
    let ru = r.unsigned_abs();

    // Computes round((l / r) * 10^s), rounding to nearest even.
    let (mut retu, rem) = if s == 0 {
        // Fast path, integer division.
        let z = lu + ru / 2; // Can't overflow, 10^38 + 10^38 / 2 < 2^128.
        (z / ru, z % ru)
    } else {
        let m = POW10_I128[s];

        if let (Ok(ls), Ok(ms)) = (i64::try_from(l), u64::try_from(m)) {
            // Fast path, intermediate product representable as u128.
            let lsu = ls.unsigned_abs();
            let mut tmp = lsu as u128 * ms as u128;
            tmp += ru / 2; // Checked that adding this can't overflow, assuming l < 2^63 and m, r < POW10_I128[DEC128_MAX_PREC].
            (tmp / ru, tmp % ru)
        } else {
            let (mut lo, mut hi) = widening_mul_128(lu, m as u128);
            let carry;
            (lo, carry) = lo.overflowing_add(ru / 2);
            hi += carry as u128;
            divrem_256_128(lo, hi, ru)?
        }
    };

    // Round to nearest even.
    if r % 2 == 0 && retu % 2 == 1 && rem == 0 {
        retu -= 1;
    }

    if retu >= POW10_I128[p] as u128 {
        return None;
    }
    if negative {
        Some(-(retu as i128))
    } else {
        Some(retu as i128)
    }
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

/// Deserialize bytes to a single i128 representing a decimal, at a specified
/// precision and scale. The number is checked to ensure it fits within the
/// specified precision and scale.  Consistent with float parsing, no decimal
/// separator is required (eg "500", "500.", and "500.0" are all accepted);
/// this allows mixed integer/decimal sequences to be parsed as decimals.
/// Returns None if the number is not well-formed, or does not fit.
/// Only b'.' is allowed as a decimal separator (issue #6698).
#[inline]
pub fn str_to_dec128(bytes: &[u8], p: usize, s: usize) -> Option<i128> {
    assert!(dec128_verify_prec_scale(p, s).is_ok());

    let separator = bytes.iter().position(|b| *b == b'.').unwrap_or(bytes.len());
    let (mut int, mut frac) = bytes.split_at(separator);

    // Trim trailing zeroes.
    while let Some((b'0', rest)) = frac.split_last() {
        frac = rest;
    }

    if frac.len() <= 1 {
        // Only integer fast path.
        let n: i128 = atoi_simd::parse(int).ok()?;
        return i128_to_dec128(n, p, s);
    }

    // Skip period.
    frac = &frac[1..];

    // Skip sign.
    let negative = match int.first() {
        Some(s @ (b'+' | b'-')) => {
            int = &int[1..];
            *s == b'-'
        },
        _ => false,
    };

    // Round if digits extend beyond the scale.
    let (next_digit, all_zero_after);
    let frac_scale = if frac.len() > s {
        if !frac[s..].iter().all(|b| b.is_ascii_digit()) {
            return None;
        }
        next_digit = frac[s];
        all_zero_after = frac[s + 1..].iter().all(|b| *b == b'0');
        frac = &frac[..s];
        0
    } else {
        next_digit = b'0';
        all_zero_after = true;
        s - frac.len()
    };

    // Parse and combine parts.
    let mut pint: i128 = if int.is_empty() {
        0
    } else {
        atoi_simd::parse_pos(int).ok()?
    };

    let mut pfrac: i128 = if frac.is_empty() {
        0
    } else {
        atoi_simd::parse_pos(frac).ok()?
    };

    // Round-to-even.
    if next_digit > b'5' || next_digit == b'5' && !all_zero_after {
        pfrac += 1;
    } else if next_digit == b'5' {
        if s == 0 {
            pint += pint % 2;
        } else {
            pfrac += pfrac % 2;
        }
    }

    let ret = mul_128_pow10(pint, s)? + mul_128_pow10(pfrac, frac_scale)?;
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

    pub fn format_dec128(&mut self, x: i128, scale: usize, trim_zeros: bool) -> &str {
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
                self.data[self.len + 1] = b'.';
                self.data[self.len + 2..self.len + 2 + scale - frac_len].fill(b'0');
                self.len += 2 + scale - frac_len;
            } else {
                self.data[self.len..self.len + whole_len].copy_from_slice(&xs[..whole_len]);
                self.data[self.len + whole_len] = b'.';
                self.len += whole_len + 1;
            }

            self.data[self.len..self.len + frac_len].copy_from_slice(&xs[whole_len..]);
            self.len += frac_len;

            if trim_zeros {
                while self.data.get(self.len - 1) == Some(&b'0') {
                    self.len -= 1;
                }
                if self.data.get(self.len - 1) == Some(&b'.') {
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
    fn test_str_to_dec() {
        assert_eq!(str_to_dec128(b"12.09", 8, 2), Some(1209));
        assert_eq!(str_to_dec128(b"1200.90", 8, 2), Some(120090));
        assert_eq!(str_to_dec128(b"143.9", 8, 2), Some(14390));

        assert_eq!(str_to_dec128(b"+000000.5", 8, 2), Some(50));
        assert_eq!(str_to_dec128(b"-0.5", 8, 2), Some(-50));
        assert_eq!(str_to_dec128(b"-1.5", 8, 2), Some(-150));

        assert_eq!(str_to_dec128(b"12ABC.34", 8, 5), None);
        assert_eq!(str_to_dec128(b"1ABC2.34", 8, 5), None);
        assert_eq!(str_to_dec128(b"12.3ABC4", 8, 5), None);
        assert_eq!(str_to_dec128(b"12.3.ABC4", 8, 5), None);

        assert_eq!(str_to_dec128(b"12.-3", 8, 5), None);
        assert_eq!(str_to_dec128(b"", 8, 5), None);
        assert_eq!(str_to_dec128(b"5.", 8, 5), Some(500000i128));
        assert_eq!(str_to_dec128(b"5", 8, 5), Some(500000i128));
        assert_eq!(str_to_dec128(b".5", 8, 5), Some(50000i128));

        // Precision and scale fitting.
        let val = b"1200";
        assert_eq!(str_to_dec128(val, 4, 0), Some(1200));
        assert_eq!(str_to_dec128(val, 3, 0), None);
        assert_eq!(str_to_dec128(val, 4, 1), None);

        let val = b"1200.010";
        assert_eq!(str_to_dec128(val, 7, 0), Some(1200));
        assert_eq!(str_to_dec128(val, 7, 3), Some(1200010));
        assert_eq!(str_to_dec128(val, 10, 6), Some(1200010000));
        assert_eq!(str_to_dec128(val, 5, 3), None);
        assert_eq!(str_to_dec128(val, 12, 5), Some(120001000));
        assert_eq!(str_to_dec128(val, 38, 35), None);

        // Rounding.
        assert_eq!(str_to_dec128(b"2.10", 5, 1), Some(21));
        assert_eq!(str_to_dec128(b"2.14", 5, 1), Some(21));
        assert_eq!(str_to_dec128(b"2.15", 5, 1), Some(22));
        assert_eq!(str_to_dec128(b"2.24", 5, 1), Some(22));
        assert_eq!(str_to_dec128(b"2.25", 5, 1), Some(22));
        assert_eq!(str_to_dec128(b"2.26", 5, 1), Some(23));
    }

    #[test]
    fn str_dec_roundtrip() {
        let mut buf = DecimalFmtBuffer::new();
        for &p in &INTERESTING_SCALE_PREC {
            for &s in &INTERESTING_SCALE_PREC {
                if s > p || p == 0 {
                    continue;
                }
                for x in INTERESTING_VALUES.iter() {
                    if let Some(d) = bigdecimal_to_dec128(x, p, s) {
                        let fmt = buf.format_dec128(d, s, false);
                        let d2 = str_to_dec128(fmt.as_bytes(), p, s);
                        assert_eq!(d, d2.unwrap());
                    } else {
                        break;
                    }
                }
            }
        }
    }

    #[test]
    fn test_mul() {
        for &p in &INTERESTING_SCALE_PREC {
            for &s in &INTERESTING_SCALE_PREC {
                if s > p || p == 0 {
                    continue;
                }
                let values: Vec<_> = INTERESTING_VALUES
                    .iter()
                    .map_while(|x| bigdecimal_to_dec128(x, p, s))
                    .map(|d| (d, dec128_to_bigdecimal(d, s)))
                    .collect();
                let mut r = SmallRng::seed_from_u64(42);
                for _ in 0..1_000 {
                    // Kept small for CI, ran with 10 million during development.
                    let (x, xb) = values.choose(&mut r).unwrap();
                    let (y, yb) = values.choose(&mut r).unwrap();
                    let prod = dec128_mul(*x, *y, p, s);
                    let prodb = bigdecimal_to_dec128(&(xb * yb), p, s);
                    assert_eq!(prod, prodb);
                }
            }
        }
    }

    #[test]
    fn test_div() {
        for &p in &INTERESTING_SCALE_PREC {
            for &s in &INTERESTING_SCALE_PREC {
                if s > p || p == 0 {
                    continue;
                }
                let values: Vec<_> = INTERESTING_VALUES
                    .iter()
                    .map_while(|x| bigdecimal_to_dec128(x, p, s))
                    .map(|d| (d, dec128_to_bigdecimal(d, s)))
                    .collect();
                let mut r = SmallRng::seed_from_u64(42);
                for _ in 0..1_000 {
                    // Kept small for CI, ran with 10 million during development.
                    let (x, xb) = values.choose(&mut r).unwrap();
                    let (y, yb) = values.choose(&mut r).unwrap();
                    if *y == 0 {
                        assert!(dec128_div(*x, *y, p, s).is_none());
                        continue;
                    }
                    let prod = dec128_mul(*x, *y, p, s);
                    let prodb = bigdecimal_to_dec128(&(xb * yb), p, s);
                    assert_eq!(prod, prodb);
                }
            }
        }
    }
}
