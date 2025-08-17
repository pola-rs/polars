macro_rules! with_arms {
    ($num_bytes:ident, $block:block, ($($values:literal),+)) => {
        match $num_bytes {
            $(
            $values => {
                #[allow(non_upper_case_globals)]
                const $num_bytes: usize = $values;
                $block
            },
            )+
            _ => unreachable!(),
        }
    };
}

pub mod boolean;
pub mod decimal;
pub mod numeric;
