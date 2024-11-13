use polars_core::config;
use polars_core::prelude::*;

pub fn dec_round(ca: &DecimalChunked, decimals: u32) -> DecimalChunked {
    dec_round_with_rm(ca, decimals, config::get_decimal_rounding_mode())
}

pub fn dec_round_with_rm(ca: &DecimalChunked, decimals: u32, rm: RoundingMode) -> DecimalChunked {
    match rm {
        RoundingMode::Ceiling => dec_round_ceiling(ca, decimals),
        RoundingMode::Down => dec_round_down(ca, decimals),
        RoundingMode::Floor => dec_round_floor(ca, decimals),
        RoundingMode::HalfDown => dec_round_half_down(ca, decimals),
        RoundingMode::HalfEven => dec_round_half_even(ca, decimals),
        RoundingMode::HalfUp => dec_round_half_up(ca, decimals),
        RoundingMode::Up => dec_round_up(ca, decimals),
        RoundingMode::Up05 => dec_round_up05(ca, decimals),
    }
}

fn dec_round_generic(
    ca: &DecimalChunked,
    decimals: u32,
    f: impl Fn(i128, i128, i128) -> i128,
) -> DecimalChunked {
    let precision = ca.precision();
    let scale = ca.scale() as u32;
    if scale <= decimals {
        return ca.clone();
    }

    let decimal_delta = scale - decimals;
    let multiplier = 10i128.pow(decimal_delta);
    let threshold = multiplier / 2;

    ca.apply_values(|v| f(v, multiplier, threshold))
        .into_decimal_unchecked(precision, scale as usize)
}

pub fn dec_round_ceiling(ca: &DecimalChunked, decimals: u32) -> DecimalChunked {
    dec_round_generic(ca, decimals, |v, multiplier, _| {
        // @TODO: Optimize
        let rem = v % multiplier;
        if v < 0 {
            v + rem.abs()
        } else {
            if rem == 0 {
                v
            } else {
                v + (multiplier - rem)
            }
        }
    })
}

pub fn dec_round_down(ca: &DecimalChunked, decimals: u32) -> DecimalChunked {
    dec_round_generic(ca, decimals, |v, multiplier, _| v - (v % multiplier))
}

pub fn dec_round_floor(ca: &DecimalChunked, decimals: u32) -> DecimalChunked {
    dec_round_generic(ca, decimals, |v, multiplier, _| {
        // @TODO: Optimize
        let rem = v % multiplier;
        if v < 0 {
            if rem == 0 {
                v
            } else {
                v - (multiplier - rem.abs())
            }
        } else {
            v - rem
        }
    })
}

pub fn dec_round_half_down(ca: &DecimalChunked, decimals: u32) -> DecimalChunked {
    dec_round_generic(ca, decimals, |v, multiplier, threshold| {
        let rem = v % multiplier;
        let round_offset = if rem.abs() > threshold { multiplier } else { 0 };
        let round_offset = if v < 0 { -round_offset } else { round_offset };
        v - rem + round_offset
    })
}

pub fn dec_round_half_even(ca: &DecimalChunked, decimals: u32) -> DecimalChunked {
    dec_round_generic(ca, decimals, |v, multiplier, threshold| {
        let rem = v % multiplier;
        let is_v_floor_even = ((v - rem) / multiplier) % 2 == 0;
        let threshold = threshold + i128::from(is_v_floor_even);
        let round_offset = if rem.abs() >= threshold {
            multiplier
        } else {
            0
        };
        let round_offset = if v < 0 { -round_offset } else { round_offset };
        v - rem + round_offset
    })
}

pub fn dec_round_half_up(ca: &DecimalChunked, decimals: u32) -> DecimalChunked {
    dec_round_generic(ca, decimals, |v, multiplier, threshold| {
        let rem = v % multiplier;
        let round_offset = if rem.abs() >= threshold {
            multiplier
        } else {
            0
        };
        let round_offset = if v < 0 { -round_offset } else { round_offset };
        v - rem + round_offset
    })
}

pub fn dec_round_up(ca: &DecimalChunked, decimals: u32) -> DecimalChunked {
    dec_round_generic(ca, decimals, |v, multiplier, _| v + (multiplier - (v % multiplier)))
}

pub fn dec_round_up05(_ca: &DecimalChunked, _decimals: u32) -> DecimalChunked {
    // assert_eq!(v.len(), target.len());
    //
    // if scale <= decimals {
    //     target.copy_from_slice(v);
    //     return;
    // }
    //
    // let decimal_delta = scale - decimals;
    // let multiplier = 10i128.pow(decimal_delta);
    // let threshold = multiplier / 2;

    todo!()
}
