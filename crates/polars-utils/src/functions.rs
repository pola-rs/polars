use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::Arc;

// The ith portion of a range split in k (as equal as possible) parts.
#[inline(always)]
pub fn range_portion(i: usize, k: usize, r: Range<usize>) -> Range<usize> {
    // Each portion having size n / k leaves n % k elements unaccounted for.
    // Make the first n % k portions have 1 extra element.
    let n = r.len();
    let base_size = n / k;
    let num_one_larger = n % k;
    let num_before = base_size * i + i.min(num_one_larger);
    let our_size = base_size + (i < num_one_larger) as usize;
    r.start + num_before..r.start + num_before + our_size
}

// Faster than collecting from a flattened iterator.
pub fn flatten<T: Clone, R: AsRef<[T]>>(bufs: &[R], len: Option<usize>) -> Vec<T> {
    let len = len.unwrap_or_else(|| bufs.iter().map(|b| b.as_ref().len()).sum());

    let mut out = Vec::with_capacity(len);
    for b in bufs {
        out.extend_from_slice(b.as_ref());
    }
    out
}

pub fn arc_map<T: Clone, F: FnMut(T) -> T>(mut arc: Arc<T>, mut f: F) -> Arc<T> {
    unsafe {
        // Make the Arc unique (cloning if necessary).
        Arc::make_mut(&mut arc);

        // If f panics we must be able to drop the Arc without assuming it is initialized.
        let mut uninit_arc = Arc::from_raw(Arc::into_raw(arc).cast::<MaybeUninit<T>>());

        // Replace the value inside the arc.
        let ptr = Arc::get_mut(&mut uninit_arc).unwrap_unchecked() as *mut MaybeUninit<T>;
        *ptr = MaybeUninit::new(f(ptr.read().assume_init()));

        // Now the Arc is properly initialized again.
        Arc::from_raw(Arc::into_raw(uninit_arc).cast::<T>())
    }
}

pub fn try_arc_map<T: Clone, E, F: FnMut(T) -> Result<T, E>>(
    mut arc: Arc<T>,
    mut f: F,
) -> Result<Arc<T>, E> {
    unsafe {
        // Make the Arc unique (cloning if necessary).
        Arc::make_mut(&mut arc);

        // If f panics we must be able to drop the Arc without assuming it is initialized.
        let mut uninit_arc = Arc::from_raw(Arc::into_raw(arc).cast::<MaybeUninit<T>>());

        // Replace the value inside the arc.
        let ptr = Arc::get_mut(&mut uninit_arc).unwrap_unchecked() as *mut MaybeUninit<T>;
        *ptr = MaybeUninit::new(f(ptr.read().assume_init())?);

        // Now the Arc is properly initialized again.
        Ok(Arc::from_raw(Arc::into_raw(uninit_arc).cast::<T>()))
    }
}
