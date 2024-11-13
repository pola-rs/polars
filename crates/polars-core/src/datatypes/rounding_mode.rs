#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    /// Round towards Infinity.
    #[default]
    Ceiling,
    /// Round towards zero.
    Down,
    /// Round towards -Infinity.
    Floor,
    /// Round to nearest with ties going towards zero.
    HalfDown,
    /// Round to nearest with ties going to nearest even integer.
    HalfEven,
    /// Round to nearest with ties going away from zero.
    HalfUp,
    /// Round away from zero.
    Up,
    /// Round away from zero if last digit after rounding towards zero would have been 0 or 5; otherwise round towards zero.
    Up05,
}
