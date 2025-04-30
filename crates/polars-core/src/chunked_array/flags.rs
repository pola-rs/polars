use std::sync::atomic::{AtomicU32, Ordering};

use crate::series::IsSorted;

/// An interior mutable version of [`StatisticsFlags`]
pub struct StatisticsFlagsIM {
    inner: AtomicU32,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct StatisticsFlags: u32 {
        const IS_SORTED_ANY = 0x03;

        const IS_SORTED_ASC = 0x01;
        const IS_SORTED_DSC = 0x02;
        const CAN_FAST_EXPLODE_LIST = 0x04;

        /// Recursive version of `CAN_FAST_EXPLODE_LIST`.
        ///
        /// This can also apply to other nested chunked arrays and signals that there all lists
        /// have been compacted recursively.
        const HAS_TRIMMED_LISTS_TO_NORMALIZED_OFFSETS = 0x08;
        /// All masked out values have their nulls propagated.
        const HAS_PROPAGATED_NULLS = 0x10;
    }
}

impl std::fmt::Debug for StatisticsFlagsIM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ChunkedArrayFlagsIM")
            .field(&self.get())
            .finish()
    }
}

impl Clone for StatisticsFlagsIM {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

impl PartialEq for StatisticsFlagsIM {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}
impl Eq for StatisticsFlagsIM {}

impl From<StatisticsFlags> for StatisticsFlagsIM {
    fn from(value: StatisticsFlags) -> Self {
        Self {
            inner: AtomicU32::new(value.bits()),
        }
    }
}

impl StatisticsFlagsIM {
    pub fn new(value: StatisticsFlags) -> Self {
        Self {
            inner: AtomicU32::new(value.bits()),
        }
    }

    pub fn empty() -> Self {
        Self::new(StatisticsFlags::empty())
    }

    pub fn get_mut(&mut self) -> StatisticsFlags {
        StatisticsFlags::from_bits(*self.inner.get_mut()).unwrap()
    }
    pub fn set_mut(&mut self, value: StatisticsFlags) {
        *self.inner.get_mut() = value.bits();
    }

    pub fn get(&self) -> StatisticsFlags {
        StatisticsFlags::from_bits(self.inner.load(Ordering::Relaxed)).unwrap()
    }
    pub fn set(&self, value: StatisticsFlags) {
        self.inner.store(value.bits(), Ordering::Relaxed);
    }
}

impl StatisticsFlags {
    pub fn is_sorted(&self) -> IsSorted {
        let is_sorted_asc = self.contains(Self::IS_SORTED_ASC);
        let is_sorted_dsc = self.contains(Self::IS_SORTED_DSC);

        assert!(!is_sorted_asc || !is_sorted_dsc);

        if is_sorted_asc {
            IsSorted::Ascending
        } else if is_sorted_dsc {
            IsSorted::Descending
        } else {
            IsSorted::Not
        }
    }

    pub fn set_sorted(&mut self, is_sorted: IsSorted) {
        let is_sorted = match is_sorted {
            IsSorted::Not => Self::empty(),
            IsSorted::Ascending => Self::IS_SORTED_ASC,
            IsSorted::Descending => Self::IS_SORTED_DSC,
        };
        self.remove(Self::IS_SORTED_ASC | Self::IS_SORTED_DSC);
        self.insert(is_sorted);
    }

    pub fn is_sorted_any(&self) -> bool {
        self.contains(Self::IS_SORTED_ASC) | self.contains(Self::IS_SORTED_DSC)
    }
    pub fn is_sorted_ascending(&self) -> bool {
        self.contains(Self::IS_SORTED_ASC)
    }
    pub fn is_sorted_descending(&self) -> bool {
        self.contains(Self::IS_SORTED_DSC)
    }

    pub fn can_fast_explode_list(&self) -> bool {
        self.contains(Self::CAN_FAST_EXPLODE_LIST)
    }

    pub fn has_propagated_nulls(&self) -> bool {
        self.contains(Self::HAS_PROPAGATED_NULLS)
    }

    pub fn has_trimmed_lists_to_normalized_offsets(&self) -> bool {
        self.contains(Self::HAS_TRIMMED_LISTS_TO_NORMALIZED_OFFSETS)
    }
}
