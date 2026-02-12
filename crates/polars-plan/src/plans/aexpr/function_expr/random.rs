use strum_macros::IntoStaticStr;

use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, PartialEq, Debug, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum IRRandomMethod {
    Shuffle,
    Sample {
        is_fraction: bool,
        with_replacement: bool,
        shuffle: bool,
    },
}

impl Hash for IRRandomMethod {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state)
    }
}
