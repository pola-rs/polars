use std::hash::{BuildHasher, Hasher};
use std::sync::{Arc, LazyLock, Mutex, Weak};

use arrow::array::builder::StaticArrayBuilder;
use arrow::array::{Utf8ViewArray, Utf8ViewArrayBuilder};
use hashbrown::HashTable;
use hashbrown::hash_table::Entry;
use polars_error::{PolarsResult, polars_ensure};
use polars_utils::pl_str::PlSmallStr;
use uuid::Uuid;

use crate::prelude::*;

mod mapping;

pub use mapping::CategoricalMapping;

/// The physical datatype backing a categorical / enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CategoricalPhysical {
    U8,
    U16,
    U32,
}

impl CategoricalPhysical {
    pub fn dtype(&self) -> DataType {
        match self {
            CategoricalPhysical::U8 => DataType::UInt8,
            CategoricalPhysical::U16 => DataType::UInt16,
            CategoricalPhysical::U32 => DataType::UInt32,
        }
    }

    pub fn max_categories(&self) -> usize {
        // We might use T::MAX as an indicator, so the maximum number of categories is T::MAX
        // (giving T::MAX - 1 as the largest category).
        match self {
            CategoricalPhysical::U8 => u8::MAX as usize,
            CategoricalPhysical::U16 => u16::MAX as usize,
            CategoricalPhysical::U32 => u32::MAX as usize,
        }
    }
}

// Used to maintain a 1:1 mapping between Categories' UUID and the Categories objects themselves.
// This is important for serialization.
static CATEGORIES_REGISTRY: LazyLock<Mutex<PlHashMap<Uuid, Weak<Categories>>>> =
    LazyLock::new(|| Mutex::new(PlHashMap::new()));

// Used to make FrozenCategories unique based on their content. This allows comparison of datatypes
// in constant time by comparing pointers.
#[expect(clippy::type_complexity)]
static FROZEN_CATEGORIES_REGISTRY: LazyLock<Mutex<HashTable<(u64, Weak<FrozenCategories>)>>> =
    LazyLock::new(|| Mutex::new(HashTable::new()));

static FROZEN_CATEGORIES_HASHER: LazyLock<PlSeedableRandomStateQuality> =
    LazyLock::new(PlSeedableRandomStateQuality::random);

static GLOBAL_CATEGORIES: LazyLock<Arc<Categories>> = LazyLock::new(|| {
    let categories = Arc::new(Categories {
        name: PlSmallStr::from_static("__POLARS_GLOBAL_CATEGORIES"),
        physical: CategoricalPhysical::U32,
        uuid: Uuid::nil(),
        mapping: MaybeGcMapping::Gc(Mutex::new(Weak::new())),
    });
    CATEGORIES_REGISTRY
        .lock()
        .unwrap()
        .insert(Uuid::nil(), Arc::downgrade(&categories));
    categories
});

/// A (named) object which is used to indicate which categorical data types
/// have the same mapping. The underlying mapping is dynamic, and if gc is true
/// may be automatically cleared when the last reference to it goes away.
pub struct Categories {
    name: PlSmallStr,
    physical: CategoricalPhysical,
    uuid: Uuid,
    mapping: MaybeGcMapping,
}

enum MaybeGcMapping {
    Gc(Mutex<Weak<CategoricalMapping>>),
    Persistent(Arc<CategoricalMapping>),
}

impl Categories {
    /// Creates a new Categories object with the given name and physical type.
    ///
    /// If gc is true the underlying categories will automatically get cleaned
    /// up when the last CategoricalMapping reference goes away, otherwise they
    /// are persistent.
    pub fn new(name: PlSmallStr, physical: CategoricalPhysical, gc: bool) -> Arc<Self> {
        Self::new_with_registry(name, physical, gc, &mut CATEGORIES_REGISTRY.lock().unwrap())
    }

    /// Returns the Categories object with the given UUID. If the UUID is unknown a new one is created.
    pub fn from_uuid(
        name: PlSmallStr,
        physical: CategoricalPhysical,
        gc: bool,
        uuid: Uuid,
    ) -> Arc<Self> {
        if uuid.is_nil() {
            return Self::global();
        }

        let mut registry = CATEGORIES_REGISTRY.lock().unwrap();
        if let Some(cats_ref) = registry.get(&uuid) {
            if let Some(cats) = cats_ref.upgrade() {
                assert!(
                    cats.name == name,
                    "UUID already exists with a different name"
                );
                assert!(
                    cats.physical == physical,
                    "UUID already exists with a different physical type"
                );
                return cats;
            }
        }
        Self::new_with_registry(name, physical, gc, &mut registry)
    }

    /// Returns the global Categories.
    pub fn global() -> Arc<Self> {
        GLOBAL_CATEGORIES.clone()
    }

    fn new_with_registry(
        name: PlSmallStr,
        physical: CategoricalPhysical,
        gc: bool,
        registry: &mut PlHashMap<Uuid, Weak<Categories>>,
    ) -> Arc<Categories> {
        let uuid = Uuid::new_v4();

        let mapping = if gc {
            MaybeGcMapping::Gc(Mutex::new(Weak::new()))
        } else {
            MaybeGcMapping::Persistent(Arc::new(CategoricalMapping::new(physical.max_categories())))
        };

        let slf = Arc::new(Self {
            name,
            physical,
            uuid,
            mapping,
        });
        registry.insert(uuid, Arc::downgrade(&slf));
        slf
    }

    /// The name of this Categories object (not unique).
    pub fn name(&self) -> &PlSmallStr {
        &self.name
    }

    /// The mapping for this Categories object. If no mapping currently exists
    /// it creates a new empty mapping.
    pub fn mapping(&self) -> Arc<CategoricalMapping> {
        match &self.mapping {
            MaybeGcMapping::Gc(weak) => {
                let mut guard = weak.lock().unwrap();
                if let Some(arc) = guard.upgrade() {
                    return arc;
                }
                let arc = Arc::new(CategoricalMapping::new(self.physical.max_categories()));
                *guard = Arc::downgrade(&arc);
                arc
            },
            MaybeGcMapping::Persistent(arc) => arc.clone(),
        }
    }

    pub fn freeze(&self, physical: CategoricalPhysical) -> Arc<FrozenCategories> {
        let mapping = self.mapping();
        let n = mapping.num_cats_upper_bound();
        FrozenCategories::new(physical, (0..n).flat_map(|i| mapping.cat_to_str(i as u32))).unwrap()
    }
}

impl Drop for Categories {
    fn drop(&mut self) {
        CATEGORIES_REGISTRY.lock().unwrap().remove(&self.uuid);
    }
}

/// An ordered collection of unique strings with an associated pre-computed
/// mapping to go from string <-> index.
///
/// FrozenCategories are globally unique to facilitate constant-time comparison.
pub struct FrozenCategories {
    physical: CategoricalPhysical,
    combined_hash: u64,
    categories: Utf8ViewArray,
    mapping: Arc<CategoricalMapping>,
}

impl FrozenCategories {
    /// Creates a new FrozenCategories object (or returns a reference to an existing one
    /// in case these are already known). Returns an error if the categories are not unique.
    /// It is guaranteed that the nth string ends up with category n (0-indexed).
    pub fn new<'a, I: Iterator<Item = &'a str>>(
        physical: CategoricalPhysical,
        strings: I,
    ) -> PolarsResult<Arc<Self>> {
        let hasher = *FROZEN_CATEGORIES_HASHER;
        let mut mapping = CategoricalMapping::with_hasher(physical.max_categories(), hasher);
        let mut builder = Utf8ViewArrayBuilder::new(ArrowDataType::Utf8);
        builder.reserve(strings.size_hint().0);

        let hasher = PlFixedStateQuality::default();
        let mut combined_hasher = hasher.build_hasher();
        for s in strings {
            let hash = hasher.hash_one(s);
            combined_hasher.write_u64(hash);
            mapping.insert_cat_with_hash(s, hash)?;
            builder.push_value_ignore_validity(s);
            polars_ensure!(mapping.len() == builder.len(), ComputeError: "FrozenCategories must contain unique strings; found duplicate '{s}'");
        }

        let combined_hash = combined_hasher.finish();
        let categories = builder.freeze();

        let mut registry = FROZEN_CATEGORIES_REGISTRY.lock().unwrap();
        let mut last_compared = None; // We have to store the strong reference to avoid a race condition.
        match registry.entry(
            combined_hash,
            |(hash, weak)| {
                *hash == combined_hash && {
                    if let Some(frozen_cats) = weak.upgrade() {
                        let cmp = frozen_cats.categories == categories
                            && frozen_cats.physical == physical;
                        last_compared = Some(frozen_cats);
                        cmp
                    } else {
                        false
                    }
                }
            },
            |(hash, _weak)| *hash,
        ) {
            Entry::Occupied(_) => Ok(last_compared.unwrap()),
            Entry::Vacant(v) => {
                let slf = Arc::new(Self {
                    physical,
                    combined_hash,
                    categories,
                    mapping: Arc::new(mapping),
                });
                v.insert((combined_hash, Arc::downgrade(&slf)));
                Ok(slf)
            },
        }
    }

    /// The mapping for this FrozenCategories object.
    pub fn mapping(&self) -> &Arc<CategoricalMapping> {
        &self.mapping
    }
}

impl Drop for FrozenCategories {
    fn drop(&mut self) {
        let mut registry = FROZEN_CATEGORIES_REGISTRY.lock().unwrap();
        while let Ok(entry) =
            registry.find_entry(self.combined_hash, |(_, weak)| weak.strong_count() == 0)
        {
            entry.remove();
        }
    }
}
