use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::str::FromStr;
use std::sync::{Arc, LazyLock, Mutex, Weak};

use arrow::array::builder::StaticArrayBuilder;
use arrow::array::{Utf8ViewArray, Utf8ViewArrayBuilder};
use arrow::datatypes::ArrowDataType;
use hashbrown::HashTable;
use hashbrown::hash_table::Entry;
use polars_error::{PolarsResult, polars_bail, polars_ensure};
use polars_utils::aliases::*;
use polars_utils::pl_str::PlSmallStr;

mod catsize;
mod mapping;

pub use catsize::{CatNative, CatSize};
pub use mapping::CategoricalMapping;

/// The physical datatype backing a categorical / enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum CategoricalPhysical {
    U8,
    U16,
    U32,
}

impl CategoricalPhysical {
    pub fn max_categories(&self) -> usize {
        // We might use T::MAX as an indicator, so the maximum number of categories is T::MAX
        // (giving T::MAX - 1 as the largest category).
        match self {
            Self::U8 => u8::MAX as usize,
            Self::U16 => u16::MAX as usize,
            Self::U32 => u32::MAX as usize,
        }
    }

    pub fn smallest_physical(num_cats: usize) -> PolarsResult<Self> {
        if num_cats < u8::MAX as usize {
            Ok(Self::U8)
        } else if num_cats < u16::MAX as usize {
            Ok(Self::U16)
        } else if num_cats < u32::MAX as usize {
            Ok(Self::U32)
        } else {
            polars_bail!(ComputeError: "attempted to insert more categories than the maximum allowed")
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
        }
    }
}

impl FromStr for CategoricalPhysical {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "u8" => Ok(Self::U8),
            "u16" => Ok(Self::U16),
            "u32" => Ok(Self::U32),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CategoricalId {
    name: PlSmallStr,
    namespace: PlSmallStr,
    physical: CategoricalPhysical,
}

impl CategoricalId {
    fn global() -> Self {
        Self {
            name: PlSmallStr::from_static(""),
            namespace: PlSmallStr::from_static(""),
            physical: CategoricalPhysical::U32,
        }
    }
}

// Used to maintain a 1:1 mapping between Categories' ID and the Categories objects themselves.
// This is important for serialization.
static CATEGORIES_REGISTRY: LazyLock<Mutex<PlHashMap<CategoricalId, Weak<Categories>>>> =
    LazyLock::new(|| Mutex::new(PlHashMap::new()));

// Used to make FrozenCategories unique based on their content. This allows comparison of datatypes
// in constant time by comparing pointers.
#[expect(clippy::type_complexity)]
static FROZEN_CATEGORIES_REGISTRY: LazyLock<Mutex<HashTable<(u64, Weak<FrozenCategories>)>>> =
    LazyLock::new(|| Mutex::new(HashTable::new()));

static FROZEN_CATEGORIES_HASHER: LazyLock<PlSeedableRandomStateQuality> =
    LazyLock::new(PlSeedableRandomStateQuality::random);

static GLOBAL_CATEGORIES: LazyLock<Arc<Categories>> = LazyLock::new(|| {
    let mut registry = CATEGORIES_REGISTRY.lock().unwrap();
    let global_id = CategoricalId::global();
    if let Some(cats_ref) = registry.get(&global_id) {
        if let Some(cats) = cats_ref.upgrade() {
            return cats;
        }
    }
    let global = Arc::new(Categories {
        id: CategoricalId::global(),
        mapping: Mutex::new(Weak::new()),
    });
    registry.insert(global_id, Arc::downgrade(&global));
    global
});

/// A (named) object which is used to indicate which categorical data types have the same mapping.
pub struct Categories {
    id: CategoricalId,
    mapping: Mutex<Weak<CategoricalMapping>>,
}

impl Categories {
    /// Creates a new Categories object with the given name, namespace and physical type if none exists, otherwise
    /// get a reference to an existing object with the same name, namespace and physical type.
    pub fn new(
        name: PlSmallStr,
        namespace: PlSmallStr,
        physical: CategoricalPhysical,
    ) -> Arc<Self> {
        let id = CategoricalId {
            name,
            namespace,
            physical,
        };
        let mut registry = CATEGORIES_REGISTRY.lock().unwrap();
        if let Some(cats_ref) = registry.get(&id) {
            if let Some(cats) = cats_ref.upgrade() {
                return cats;
            }
        }
        let mapping = Mutex::new(Weak::new());
        let slf = Arc::new(Self {
            id: id.clone(),
            mapping,
        });
        registry.insert(id, Arc::downgrade(&slf));
        slf
    }

    /// Returns the global Categories.
    pub fn global() -> Arc<Self> {
        GLOBAL_CATEGORIES.clone()
    }

    /// Returns whether this refers to the global categories.
    pub fn is_global(self: &Arc<Self>) -> bool {
        Arc::ptr_eq(self, &*GLOBAL_CATEGORIES)
    }

    /// Generates a Categories with a random (UUID) name.
    pub fn random(namespace: PlSmallStr, physical: CategoricalPhysical) -> Arc<Self> {
        Self::new(uuid::Uuid::new_v4().to_string().into(), namespace, physical)
    }

    /// The name of this Categories object.
    pub fn name(&self) -> &PlSmallStr {
        &self.id.name
    }

    /// The namespace of this Categories object.
    pub fn namespace(&self) -> &PlSmallStr {
        &self.id.namespace
    }

    /// The physical dtype of the category ids.
    pub fn physical(&self) -> CategoricalPhysical {
        self.id.physical
    }

    /// A stable hash of this Categories object, not the contained categories.
    pub fn hash(&self) -> u64 {
        PlFixedStateQuality::default().hash_one(&self.id)
    }

    /// The mapping for this Categories object. If no mapping currently exists
    /// it creates a new empty mapping.
    pub fn mapping(&self) -> Arc<CategoricalMapping> {
        let mut guard = self.mapping.lock().unwrap();
        if let Some(arc) = guard.upgrade() {
            return arc;
        }
        let arc = Arc::new(CategoricalMapping::new(self.id.physical.max_categories()));
        *guard = Arc::downgrade(&arc);
        arc
    }

    pub fn freeze(&self) -> Arc<FrozenCategories> {
        let mapping = self.mapping();
        let n = mapping.num_cats_upper_bound();
        FrozenCategories::new((0..n).flat_map(|i| mapping.cat_to_str(i as CatSize))).unwrap()
    }
}

impl fmt::Debug for Categories {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Categories")
            .field("name", &self.id.name)
            .field("namespace", &self.id.namespace)
            .field("physical", &self.id.physical)
            .finish()
    }
}

impl Drop for Categories {
    fn drop(&mut self) {
        CATEGORIES_REGISTRY.lock().unwrap().remove(&self.id);
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
    pub fn new<'a, I: IntoIterator<Item = &'a str>>(strings: I) -> PolarsResult<Arc<Self>> {
        let strings = strings.into_iter();
        let hasher = *FROZEN_CATEGORIES_HASHER;
        let mut mapping = CategoricalMapping::with_hasher(usize::MAX, hasher);
        let mut builder = Utf8ViewArrayBuilder::new(ArrowDataType::Utf8);
        builder.reserve(strings.size_hint().0);

        let mut combined_hasher = PlFixedStateQuality::default().build_hasher();
        for s in strings {
            combined_hasher.write(s.as_bytes());
            mapping.insert_cat(s)?;
            builder.push_value_ignore_validity(s);
            polars_ensure!(mapping.len() == builder.len(), ComputeError: "FrozenCategories must contain unique strings; found duplicate '{s}'");
        }

        let combined_hash = combined_hasher.finish();
        let categories = builder.freeze();
        mapping.set_max_categories(categories.len()); // Don't allow any further inserts.

        let physical = CategoricalPhysical::smallest_physical(categories.len())?;
        let mut registry = FROZEN_CATEGORIES_REGISTRY.lock().unwrap();
        let mut last_compared = None; // We have to store the strong reference to avoid a race condition.
        match registry.entry(
            combined_hash,
            |(hash, weak)| {
                *hash == combined_hash && {
                    if let Some(frozen_cats) = weak.upgrade() {
                        let cmp = frozen_cats.categories == categories;
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

    /// The categories contained in this FrozenCategories object.
    pub fn categories(&self) -> &Utf8ViewArray {
        &self.categories
    }

    /// The physical dtype of the category ids.
    pub fn physical(&self) -> CategoricalPhysical {
        self.physical
    }

    /// The mapping for this FrozenCategories object.
    pub fn mapping(&self) -> &Arc<CategoricalMapping> {
        &self.mapping
    }

    /// A stable hash of the categories.
    pub fn hash(&self) -> u64 {
        self.combined_hash
    }
}

impl fmt::Debug for FrozenCategories {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrozenCategories")
            .field("physical", &self.physical)
            .field("categories", &self.categories)
            .finish()
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

pub fn ensure_same_categories(left: &Arc<Categories>, right: &Arc<Categories>) -> PolarsResult<()> {
    if Arc::ptr_eq(left, right) {
        return Ok(());
    }

    if left.name() != right.name() {
        polars_bail!(SchemaMismatch: "Categories name mismatch, left: '{}', right: '{}'.

Operations mixing different Categories are often not supported, you may have to cast.", left.name(), right.name())
    } else if left.namespace() != right.namespace() {
        polars_bail!(SchemaMismatch: "Categories have same name ('{}'), but have a mismatch in namespace, left: {}, right: {}.

Operations mixing different Categories are often not supported, you may have to cast.", left.name(), left.namespace(), right.namespace())
    } else if left.physical() != right.physical() {
        polars_bail!(SchemaMismatch: "Categories have same name and namespace ('{}', {}), but have a mismatch in dtype, left: {}, right: {}.

Operations mixing different Categories are often not supported, you may have to cast.", left.name(), left.namespace(), left.physical().as_str(), right.physical().as_str())
    } else {
        polars_bail!(SchemaMismatch: "Categories which should be equal have different backing objects.

This is a known problem when combining Polars with multiprocessing using fork().")
    }
}

pub fn ensure_same_frozen_categories(
    left: &Arc<FrozenCategories>,
    right: &Arc<FrozenCategories>,
) -> PolarsResult<()> {
    if Arc::ptr_eq(left, right) {
        return Ok(());
    }

    polars_bail!(SchemaMismatch: r#"Enum mismatch.

Operations mixing different Enums are often not supported, you may have to cast."#)
}
