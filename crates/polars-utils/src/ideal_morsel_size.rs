use std::num::NonZeroUsize;
use std::sync::OnceLock;

pub fn get_ideal_morsel_size() -> NonZeroUsize {
    return *IDEAL_MORSEL_SIZE.get_or_init(|| {
        std::env::var("POLARS_IDEAL_MORSEL_SIZE")
            .map(|x| {
                x.parse::<NonZeroUsize>()
                    .ok()
                    .unwrap_or_else(|| panic!("invalid value for POLARS_IDEAL_MORSEL_SIZE: {x}"))
            })
            .unwrap_or(const { NonZeroUsize::new(100_000).unwrap() })
    });

    static IDEAL_MORSEL_SIZE: OnceLock<NonZeroUsize> = OnceLock::new();
}
