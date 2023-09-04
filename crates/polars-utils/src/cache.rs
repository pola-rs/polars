use crate::aliases::PlHashMap;

pub struct CachedFunc<T, R, F> {
    func: F,
    cache: PlHashMap<T, R>,
}

impl<T, R, F> CachedFunc<T, R, F>
where
    F: FnMut(T) -> R,
    T: std::hash::Hash + Eq + Clone,
    R: Copy,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            cache: PlHashMap::with_capacity_and_hasher(0, Default::default()),
        }
    }

    pub fn eval(&mut self, x: T, use_cache: bool) -> R {
        if use_cache {
            *self
                .cache
                .entry(x)
                .or_insert_with_key(|xr| (self.func)(xr.clone()))
        } else {
            (self.func)(x)
        }
    }
}
