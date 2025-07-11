use polars_core::random::get_global_random_u64;

/// A random seed that gets used to perform operations that need non-determinism.
///
/// This seed itself is randomly generated when converting to the IR and is thereafter fixed, this
/// allows proper reasoning about which expressions should behave the same operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Seed {
    /// The actual seed value.
    ///
    /// This might have been explicitely given by the user or might have been randomly generated
    /// upon conversion to IR.
    value: u64,

    /// Should the operation vary its seed between query runs even if the IR remains the exact
    /// same.
    is_deterministic: bool,
}

impl Seed {
    /// Create a new seed from an optionally given explicit seed.
    pub fn from_optional(value: Option<u64>) -> Self {
        match value {
            None => Seed { value: get_global_random_u64(), is_deterministic: false },
            Some(value) => Seed { value, is_deterministic: true },
        }
    }

    /// Turn the seed back into an optionally explicit seed.
    pub fn into_optional(self) -> Option<u64> {
        self.is_deterministic.then_some(self.value)
    }

    pub fn is_deterministic(self) -> bool {
        self.is_deterministic
    }

    /// Get the seed to use given the seed specific to this query execution.
    pub fn get(self, execution_seed: u64) -> u64 {
        if self.is_deterministic {
            self.value
        } else {
            self.value ^ execution_seed
        }
    }
}