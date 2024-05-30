use std::fmt;

use bitflags::bitflags;
use polars_utils::IdxSize;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::PolarsDataType;
use crate::series::IsSorted;

bitflags! {
    #[derive(Default, Debug, Clone, Copy, PartialEq)]
    pub struct MetadataProperties: u32 {
        const SORTED = 0x01;
        const FAST_EXPLODE_LIST = 0x02;
        const MIN_VALUE = 0x04;
        const MAX_VALUE = 0x08;
        const DISTINCT_COUNT = 0x10;
    }
}

pub struct Metadata<T: PolarsDataType> {
    flags: MetadataFlags,

    min_value: Option<T::OwnedPhysical>,
    max_value: Option<T::OwnedPhysical>,

    distinct_count: Option<IdxSize>,
}

bitflags! {
    #[derive(Default, Debug, Clone, Copy, PartialEq)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(transparent))]
    pub struct MetadataFlags: u8 {
        const SORTED_ASC = 0x01;
        const SORTED_DSC = 0x02;
        const FAST_EXPLODE_LIST = 0x04;
    }
}

impl MetadataFlags {
    pub fn set_sorted_flag(&mut self, sorted: IsSorted) {
        match sorted {
            IsSorted::Not => {
                self.remove(MetadataFlags::SORTED_ASC | MetadataFlags::SORTED_DSC);
            },
            IsSorted::Ascending => {
                self.remove(MetadataFlags::SORTED_DSC);
                self.insert(MetadataFlags::SORTED_ASC)
            },
            IsSorted::Descending => {
                self.remove(MetadataFlags::SORTED_ASC);
                self.insert(MetadataFlags::SORTED_DSC)
            },
        }
    }

    pub fn get_sorted_flag(&self) -> IsSorted {
        if self.contains(MetadataFlags::SORTED_ASC) {
            IsSorted::Ascending
        } else if self.contains(MetadataFlags::SORTED_DSC) {
            IsSorted::Descending
        } else {
            IsSorted::Not
        }
    }

    pub fn get_fast_explode_list(&self) -> bool {
        self.contains(MetadataFlags::FAST_EXPLODE_LIST)
    }
}

impl<T: PolarsDataType> Default for Metadata<T> {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl<T: PolarsDataType> Clone for Metadata<T> {
    fn clone(&self) -> Self {
        Self {
            flags: self.flags,
            min_value: self.min_value.clone(),
            max_value: self.max_value.clone(),
            distinct_count: self.distinct_count,
        }
    }
}

impl<T: PolarsDataType> fmt::Debug for Metadata<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Metadata")
            .field("flags", &self.flags)
            .field("min_value", &self.min_value)
            .field("max_value", &self.max_value)
            .field("distinct_count", &self.distinct_count)
            .finish()
    }
}

impl<T: PolarsDataType> Metadata<T> {
    pub const DEFAULT: Metadata<T> = Self {
        flags: MetadataFlags::empty(),

        min_value: None,
        max_value: None,

        distinct_count: None,
    };

    pub fn is_sorted_ascending(&self) -> bool {
        self.flags.contains(MetadataFlags::SORTED_ASC)
    }

    pub fn set_sorted_ascending(&mut self, value: bool) {
        self.flags.set(MetadataFlags::SORTED_ASC, value);
    }

    pub fn is_sorted_descending(&self) -> bool {
        self.flags.contains(MetadataFlags::SORTED_DSC)
    }

    pub fn set_sorted_descending(&mut self, value: bool) {
        self.flags.set(MetadataFlags::SORTED_DSC, value);
    }

    pub fn get_fast_explode_list(&self) -> bool {
        self.flags.contains(MetadataFlags::FAST_EXPLODE_LIST)
    }

    pub fn set_fast_explode_list(&mut self, value: bool) {
        self.flags.set(MetadataFlags::FAST_EXPLODE_LIST, value);
    }

    pub fn is_sorted(&self) -> IsSorted {
        let ascending = self.is_sorted_ascending();
        let descending = self.is_sorted_descending();

        match (ascending, descending) {
            (true, false) => IsSorted::Ascending,
            (false, true) => IsSorted::Descending,
            (false, false) => IsSorted::Not,
            (true, true) => unreachable!(),
        }
    }

    pub fn set_sorted_flag(&mut self, is_sorted: IsSorted) {
        let (ascending, descending) = match is_sorted {
            IsSorted::Ascending => (true, false),
            IsSorted::Descending => (false, true),
            IsSorted::Not => (false, false),
        };

        self.set_sorted_ascending(ascending);
        self.set_sorted_descending(descending);
    }

    pub fn set_distinct_count(&mut self, distinct_count: Option<IdxSize>) {
        self.distinct_count = distinct_count;
    }
    pub fn set_min_value(&mut self, min_value: Option<T::OwnedPhysical>) {
        self.min_value = min_value;
    }
    pub fn set_max_value(&mut self, max_value: Option<T::OwnedPhysical>) {
        self.max_value = max_value;
    }

    pub fn set_flags(&mut self, flags: MetadataFlags) {
        self.flags = flags;
    }

    pub fn get_distinct_count(&self) -> Option<IdxSize> {
        self.distinct_count
    }

    pub fn get_min_value(&self) -> Option<&T::OwnedPhysical> {
        self.min_value.as_ref()
    }

    pub fn get_max_value(&self) -> Option<&T::OwnedPhysical> {
        self.max_value.as_ref()
    }

    pub fn get_flags(&self) -> MetadataFlags {
        self.flags
    }
}
