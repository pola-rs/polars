use super::*;

pub(super) fn shift_and_fill(args: &mut [Series], periods: i64) -> Result<Series> {
    let s = &args[0];
    let fill_value = &args[1];

    let mask: BooleanChunked = if periods > 0 {
        let len = s.len();
        let mut bits = MutableBitmap::with_capacity(s.len());
        bits.extend_constant(periods as usize, false);
        bits.extend_constant(len.saturating_sub(periods as usize), true);
        let mask = BooleanArray::from_data_default(bits.into(), None);
        mask.into()
    } else {
        let length = s.len() as i64;
        // periods is negative, so subtraction.
        let tipping_point = std::cmp::max(length + periods, 0);
        let mut bits = MutableBitmap::with_capacity(s.len());
        bits.extend_constant(tipping_point as usize, true);
        bits.extend_constant(-periods as usize, false);
        let mask = BooleanArray::from_data_default(bits.into(), None);
        mask.into()
    };

    s.shift(periods).zip_with_same_type(&mask, fill_value)
}
