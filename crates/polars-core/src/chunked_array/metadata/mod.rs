use std::fmt;

use bitflags::bitflags;
use polars_utils::IdxSize;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use self::collect::MetadataCollectable;
pub use self::env::MetadataEnv;
pub use self::guard::MetadataReadGuard;
pub use self::interior_mutable::IMMetadata;
pub use self::md_trait::MetadataTrait;
use super::PolarsDataType;
use crate::series::IsSorted;

#[macro_use]
mod env;
mod collect;
mod guard;
mod interior_mutable;
mod md_trait;

macro_rules! mdenv_may_bail {
    (get: $field:literal, $value:expr $(=> $default:expr)?) => {{
        if MetadataEnv::disabled() {
            return $($default)?;
        }
        if MetadataEnv::log() {
            mdlog!("Get: '{}' <- {:?}", $field, $value);
        }
        $value
    }};
    (set: $field:literal, $value:expr) => {
        if MetadataEnv::disabled() {
            return;
        }
        if MetadataEnv::log() {
            mdlog!("Set: '{}' <- {:?}", $field, $value);
        }
    };
    (init: $field:literal, $value:expr ; $default:expr) => {{
        if MetadataEnv::enabled() {
            if MetadataEnv::log() {
                mdlog!("Ini: '{}' <- {:?}", $field, $value);
            }
            $value
        } else {
            $default
        }
    }};
}

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

    /// Number of unique non-null values
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
        mdenv_may_bail!(set: "sorted", sorted);
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
        let sorted = if self.contains(MetadataFlags::SORTED_ASC) {
            IsSorted::Ascending
        } else if self.contains(MetadataFlags::SORTED_DSC) {
            IsSorted::Descending
        } else {
            IsSorted::Not
        };

        mdenv_may_bail!(get: "sorted", sorted => IsSorted::Not)
    }

    pub fn set_fast_explode_list(&mut self, fast_explode_list: bool) {
        mdenv_may_bail!(set: "fast_explode_list", fast_explode_list);
        self.set(Self::FAST_EXPLODE_LIST, fast_explode_list)
    }

    pub fn get_fast_explode_list(&self) -> bool {
        let value = self.contains(MetadataFlags::FAST_EXPLODE_LIST);
        mdenv_may_bail!(get: "fast_explode_list", value => false)
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

pub enum MetadataMerge<T: PolarsDataType> {
    Keep,
    Conflict,
    New(Metadata<T>),
}

impl<T: PolarsDataType> Metadata<T> {
    pub const DEFAULT: Metadata<T> = Self {
        flags: MetadataFlags::empty(),

        min_value: None,
        max_value: None,

        distinct_count: None,
    };

    // Builder Pattern Methods
    pub fn sorted(mut self, is_sorted: IsSorted) -> Self {
        self.flags.set_sorted_flag(is_sorted);
        self
    }
    pub fn fast_explode_list(mut self, fast_explode_list: bool) -> Self {
        self.flags.set_fast_explode_list(fast_explode_list);
        self
    }
    pub fn flags(mut self, flags: MetadataFlags) -> Self {
        self.set_flags(flags);
        self
    }
    pub fn min_value(mut self, min_value: T::OwnedPhysical) -> Self {
        self.set_min_value(Some(min_value));
        self
    }
    pub fn max_value(mut self, max_value: T::OwnedPhysical) -> Self {
        self.set_max_value(Some(max_value));
        self
    }
    pub fn distinct_count(mut self, distinct_count: IdxSize) -> Self {
        self.set_distinct_count(Some(distinct_count));
        self
    }
    pub fn sorted_opt(self, is_sorted: Option<IsSorted>) -> Self {
        if let Some(is_sorted) = is_sorted {
            self.sorted(is_sorted)
        } else {
            self
        }
    }
    pub fn fast_explode_list_opt(self, fast_explode_list: Option<bool>) -> Self {
        if let Some(fast_explode_list) = fast_explode_list {
            self.fast_explode_list(fast_explode_list)
        } else {
            self
        }
    }
    pub fn flags_opt(mut self, flags: Option<MetadataFlags>) -> Self {
        self.set_flags(flags.unwrap_or(MetadataFlags::empty()));
        self
    }
    pub fn min_value_opt(mut self, min_value: Option<T::OwnedPhysical>) -> Self {
        self.set_min_value(min_value);
        self
    }
    pub fn max_value_opt(mut self, max_value: Option<T::OwnedPhysical>) -> Self {
        self.set_max_value(max_value);
        self
    }
    pub fn distinct_count_opt(mut self, distinct_count: Option<IdxSize>) -> Self {
        self.set_distinct_count(distinct_count);
        self
    }

    /// Create a [`Metadata`] with only the properties set in `props`.
    pub fn filter_props_cast<O: PolarsDataType>(&self, props: MetadataProperties) -> Metadata<O> {
        if props.is_empty() {
            return Metadata::DEFAULT;
        }

        debug_assert!(!props.contains(P::MIN_VALUE));
        debug_assert!(!props.contains(P::MAX_VALUE));

        use {MetadataFlags as F, MetadataProperties as P};

        let sorted = if props.contains(P::SORTED) {
            self.flags & (F::SORTED_ASC | F::SORTED_DSC)
        } else {
            F::empty()
        };
        let fast_explode_list = if props.contains(P::FAST_EXPLODE_LIST) {
            self.flags & F::FAST_EXPLODE_LIST
        } else {
            F::empty()
        };

        Metadata {
            flags: sorted | fast_explode_list,
            min_value: None,
            max_value: None,
            distinct_count: self
                .distinct_count
                .as_ref()
                .cloned()
                .filter(|_| props.contains(P::DISTINCT_COUNT)),
        }
    }

    /// Create a [`Metadata`] with only the properties set in `props`.
    pub fn filter_props(&self, props: MetadataProperties) -> Self {
        if props.is_empty() {
            return Metadata::DEFAULT;
        }

        use {MetadataFlags as F, MetadataProperties as P};

        let sorted = if props.contains(P::SORTED) {
            self.flags & (F::SORTED_ASC | F::SORTED_DSC)
        } else {
            F::empty()
        };
        let fast_explode_list = if props.contains(P::FAST_EXPLODE_LIST) {
            self.flags & F::FAST_EXPLODE_LIST
        } else {
            F::empty()
        };

        let min_value = self
            .min_value
            .as_ref()
            .cloned()
            .filter(|_| props.contains(P::MIN_VALUE));
        let max_value = self
            .max_value
            .as_ref()
            .cloned()
            .filter(|_| props.contains(P::MAX_VALUE));
        let distinct_count = self
            .distinct_count
            .as_ref()
            .cloned()
            .filter(|_| props.contains(P::DISTINCT_COUNT));

        Self {
            flags: mdenv_may_bail!(init: "flags", sorted | fast_explode_list ; MetadataFlags::empty()),
            min_value: mdenv_may_bail!(init: "min_value", min_value ; None),
            max_value: mdenv_may_bail!(init: "max_value", max_value ; None),
            distinct_count: mdenv_may_bail!(init: "distinct_count", distinct_count ; None),
        }
    }

    /// Merge the maximum information from both [`Metadata`]s into one [`Metadata`].
    ///
    /// It returns
    /// - [`MetadataMerge::Keep`] if the `self` already contains all the information
    /// - [`MetadataMerge::New(md)`][MetadataMerge::New] if we have learned new information
    /// - [`MetadataMerge::Conflict`] if the two structures contain conflicting metadata
    pub fn merge(&self, other: Self) -> MetadataMerge<T> {
        if MetadataEnv::disabled() || other.is_empty() {
            return MetadataMerge::Keep;
        }

        let sorted_conflicts = matches!(
            (self.is_sorted(), other.is_sorted()),
            (IsSorted::Ascending, IsSorted::Descending)
                | (IsSorted::Descending, IsSorted::Ascending)
        );

        let is_conflict = sorted_conflicts
            || matches!((self.get_min_value(), other.get_min_value()), (Some(x), Some(y)) if x != y)
            || matches!((self.get_max_value(), other.get_max_value()), (Some(x), Some(y)) if x != y)
            || matches!((self.get_distinct_count(), other.get_distinct_count()), (Some(x), Some(y)) if x != y);

        if is_conflict {
            return MetadataMerge::Conflict;
        }

        let is_new = (!self.get_fast_explode_list() && other.get_fast_explode_list())
            || (self.is_sorted() == IsSorted::Not && other.is_sorted() != IsSorted::Not)
            || matches!(
                (self.get_min_value(), other.get_min_value()),
                (None, Some(_))
            )
            || matches!(
                (self.get_max_value(), other.get_max_value()),
                (None, Some(_))
            )
            || matches!(
                (self.get_distinct_count(), other.get_distinct_count()),
                (None, Some(_))
            );

        if !is_new {
            return MetadataMerge::Keep;
        }

        let min_value = self.min_value.as_ref().cloned().or(other.min_value);
        let max_value = self.max_value.as_ref().cloned().or(other.max_value);
        let distinct_count = self.distinct_count.or(other.distinct_count);

        MetadataMerge::New(Metadata {
            flags: mdenv_may_bail!(init: "flags", self.flags | other.flags ; MetadataFlags::empty()),
            min_value: mdenv_may_bail!(init: "min_value", min_value ; None),
            max_value: mdenv_may_bail!(init: "max_value", max_value ; None),
            distinct_count: mdenv_may_bail!(init: "distinct_count", distinct_count ; None),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
            && self.min_value.is_none()
            && self.max_value.is_none()
            && self.distinct_count.is_none()
    }

    pub fn is_sorted_ascending(&self) -> bool {
        self.flags.get_sorted_flag() == IsSorted::Ascending
    }

    pub fn set_sorted_ascending(&mut self, value: bool) {
        self.flags.set_sorted_flag(if value {
            IsSorted::Ascending
        } else {
            IsSorted::Not
        });
    }

    pub fn is_sorted_descending(&self) -> bool {
        self.flags.get_sorted_flag() == IsSorted::Descending
    }

    pub fn set_sorted_descending(&mut self, value: bool) {
        self.flags.set_sorted_flag(if value {
            IsSorted::Descending
        } else {
            IsSorted::Not
        });
    }

    pub fn get_fast_explode_list(&self) -> bool {
        self.flags.get_fast_explode_list()
    }

    pub fn set_fast_explode_list(&mut self, value: bool) {
        self.flags.set_fast_explode_list(value);
    }

    pub fn is_sorted_any(&self) -> bool {
        self.flags.get_sorted_flag() != IsSorted::Not
    }
    pub fn is_sorted(&self) -> IsSorted {
        self.flags.get_sorted_flag()
    }

    pub fn set_sorted_flag(&mut self, is_sorted: IsSorted) {
        self.flags.set_sorted_flag(is_sorted)
    }

    pub fn set_flags(&mut self, flags: MetadataFlags) {
        mdenv_may_bail!(set: "flags", flags);
        self.flags = flags;
    }
    pub fn set_min_value(&mut self, min_value: Option<T::OwnedPhysical>) {
        mdenv_may_bail!(set: "min_value", min_value);
        self.min_value = min_value;
    }
    pub fn set_max_value(&mut self, max_value: Option<T::OwnedPhysical>) {
        mdenv_may_bail!(set: "max_value", max_value);
        self.max_value = max_value;
    }
    pub fn set_distinct_count(&mut self, distinct_count: Option<IdxSize>) {
        mdenv_may_bail!(set: "distinct_count", distinct_count);
        self.distinct_count = distinct_count;
    }

    pub fn get_flags(&self) -> MetadataFlags {
        let flags = self.flags;
        mdenv_may_bail!(get: "flags", flags => MetadataFlags::empty())
    }
    pub fn get_min_value(&self) -> Option<&T::OwnedPhysical> {
        let min_value = self.min_value.as_ref();
        mdenv_may_bail!(get: "min_value", min_value => None)
    }
    pub fn get_max_value(&self) -> Option<&T::OwnedPhysical> {
        let max_value = self.max_value.as_ref();
        mdenv_may_bail!(get: "max_value", max_value => None)
    }
    pub fn get_distinct_count(&self) -> Option<IdxSize> {
        let distinct_count = self.distinct_count;
        mdenv_may_bail!(get: "distinct_count", distinct_count => None)
    }
}
