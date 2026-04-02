use std::ops::Deref;
use std::sync::{Arc, LazyLock};

use polars_core::prelude::PlIndexMap;
use polars_utils::pl_str::PlSmallStr;

use crate::plans::Sorted;

static EMPTY_MAP: LazyLock<PlIndexMap<PlSmallStr, Sorted>> = LazyLock::new(Default::default);
static EMPTY: LazyLock<FramePartitioning> = LazyLock::new(FramePartitioning::new);

#[derive(Debug, Default, Clone)]
pub struct FramePartitioning {
    keys: Option<Arc<PlIndexMap<PlSmallStr, Sorted>>>,
}

impl FramePartitioning {
    pub const fn new() -> Self {
        Self { keys: None }
    }

    pub fn empty_static() -> &'static FramePartitioning {
        &*EMPTY
    }

    pub fn is_empty(&self) -> bool {
        self.keys.as_ref().is_none_or(|x| x.is_empty())
    }

    pub fn make_mut(&mut self) -> &mut PlIndexMap<PlSmallStr, Sorted> {
        Arc::make_mut(self.keys.get_or_insert_default())
    }

    pub fn truncate(&mut self, len: usize) {
        if len == 0 {
            *self = Self::default();
        }

        if self.len() > len {
            self.make_mut().truncate(len);
        }
    }
}

impl Deref for FramePartitioning {
    type Target = PlIndexMap<PlSmallStr, Sorted>;

    fn deref(&self) -> &Self::Target {
        self.keys.as_deref().unwrap_or(&*EMPTY_MAP)
    }
}

impl FromIterator<Sorted> for FramePartitioning {
    fn from_iter<T: IntoIterator<Item = Sorted>>(iter: T) -> Self {
        let map = PlIndexMap::from_iter(iter.into_iter().map(|s| (s.column.clone(), s)));

        Self {
            keys: (!map.is_empty()).then_some(map).map(Arc::new),
        }
    }
}
