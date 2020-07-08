use std::mem;

/// Used to split the mantissa and exponent of floating point numbers
/// https://stackoverflow.com/questions/39638363/how-can-i-use-a-hashmap-with-f64-as-key-in-rust
pub(crate) fn integer_decode(val: f64) -> (u64, i16, i8) {
    let bits: u64 = unsafe { mem::transmute(val) };
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}

pub(crate) fn floating_encode_f64(mantissa: u64, exponent: i16, sign: i8) -> f64 {
    sign as f64 * mantissa as f64 * (2.0f64).powf(exponent as f64)
}

#[macro_export]
macro_rules! exec_concurrent {
    ($block_a:block, $block_b:block) => {{
        thread::scope(|s| {
            let handle_left = s.spawn(|_| $block_a);
            let handle_right = s.spawn(|_| $block_b);
            let return_left = handle_left.join().expect("thread panicked");
            let return_right = handle_right.join().expect("thread panicked");
            (return_left, return_right)
        })
        .expect("could not join threads or thread panicked")
    }};
}
