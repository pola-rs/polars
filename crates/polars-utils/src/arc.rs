use std::mem::MaybeUninit;
use std::sync::Arc;

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
