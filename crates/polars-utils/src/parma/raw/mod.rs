use std::alloc::Layout;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::without_provenance_mut;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, RwLock};

mod key;
mod probe;

pub use key::Key;
use probe::{Prober, TagGroup};

#[repr(C)]
struct AllocHeader<K: ?Sized, V> {
    num_entries: usize,
    num_deletions: AtomicUsize,

    // Must be decremented when starting an insertion, waiting for a new
    // allocation if the counter is zero.
    claim_start_semaphore: AtomicUsize,

    // Must be decremented when an insertion has claimed a slot in the entry table,
    // used to ensure all entries are up-to-date before starting the rehashing process.
    claim_done_barrier: AtomicUsize,

    marker: PhantomData<(Box<K>, V)>,
    align: [TagGroup; 0],
}

impl<K: ?Sized, V> AllocHeader<K, V> {
    fn layout(num_entries: usize) -> Layout {
        // Layout: AllocHeader [tags] [entries]
        assert!(num_entries.is_power_of_two() && num_entries >= size_of::<TagGroup>());
        let mut layout = Layout::new::<Self>();
        layout = layout
            .extend(Layout::array::<TagGroup>(num_entries / size_of::<TagGroup>()).unwrap())
            .unwrap()
            .0;
        layout = layout
            .extend(Layout::array::<AtomicPtr<EntryHeader<K, V>>>(num_entries).unwrap())
            .unwrap()
            .0;
        layout
    }

    #[inline(always)]
    unsafe fn tags(&self, alloc: *mut Self) -> &[TagGroup] {
        unsafe {
            let p = alloc.byte_add(size_of::<Self>());
            core::slice::from_raw_parts(p.cast(), self.num_entries / size_of::<TagGroup>())
        }
    }

    #[inline(always)]
    unsafe fn entries(&self, alloc: *mut Self) -> &[AtomicPtr<EntryHeader<K, V>>] {
        unsafe {
            let p = alloc.byte_add(size_of::<Self>() + self.num_entries);
            core::slice::from_raw_parts(p.cast(), self.num_entries)
        }
    }

    #[inline(always)]
    #[allow(clippy::mut_from_ref)] // Does not borrow from &self, but from alloc.
    unsafe fn tags_mut(&self, alloc: *mut Self) -> &mut [TagGroup] {
        unsafe {
            let p = alloc.byte_add(size_of::<Self>());
            core::slice::from_raw_parts_mut(p.cast(), self.num_entries / size_of::<TagGroup>())
        }
    }

    #[inline(always)]
    #[allow(clippy::mut_from_ref)] // Does not borrow from &self, but from alloc.
    unsafe fn entries_mut(&self, alloc: *mut Self) -> &mut [AtomicPtr<EntryHeader<K, V>>] {
        unsafe {
            let p = alloc.byte_add(size_of::<Self>() + self.num_entries);
            core::slice::from_raw_parts_mut(p.cast(), self.num_entries)
        }
    }

    fn new(num_entries: usize) -> *mut Self {
        let layout = Self::layout(num_entries);
        unsafe {
            let alloc = std::alloc::alloc(layout).cast::<Self>();
            if alloc.is_null() {
                std::alloc::handle_alloc_error(layout);
            }

            let max_load = probe::max_load(num_entries);
            alloc.write(Self {
                num_entries,
                num_deletions: AtomicUsize::new(0),
                claim_start_semaphore: AtomicUsize::new(max_load),
                claim_done_barrier: AtomicUsize::new(max_load),
                marker: PhantomData,
                align: [],
            });

            let tags_p = alloc.byte_add(size_of::<Self>()) as *mut u8;
            let tags: &mut [MaybeUninit<TagGroup>] =
                core::slice::from_raw_parts_mut(tags_p.cast(), num_entries / size_of::<TagGroup>());
            tags.fill_with(|| MaybeUninit::new(TagGroup::all_empty()));

            let entries_p = alloc.byte_add(size_of::<Self>() + num_entries);
            let entries: &mut [MaybeUninit<AtomicPtr<u8>>] =
                core::slice::from_raw_parts_mut(entries_p.cast(), num_entries);
            entries
                .fill_with(|| MaybeUninit::new(AtomicPtr::new(without_provenance_mut(UNCLAIMED))));

            alloc
        }
    }

    unsafe fn free(slf: *mut Self) {
        unsafe {
            if slf != &raw const EMPTY_ALLOC_LOC as _ {
                let layout = Self::layout((*slf).num_entries);
                std::alloc::dealloc(slf.cast(), layout);
            }
        }
    }

    // Returns true if you may proceed with the insert attempt, false if you
    // should wait for reallocation to occur.
    fn start_claim_attempt(&self) -> bool {
        self.claim_start_semaphore
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |attempts_left| {
                attempts_left.checked_sub(1)
            })
            .is_ok()
    }

    fn abort_claim_attempt(
        &self,
        alloc_lock: &Mutex<TableLockState<K, V>>,
        waiting_for_alloc: &Condvar,
    ) {
        let old = self.claim_start_semaphore.fetch_add(1, Ordering::Relaxed);
        if old == 0 {
            // We need to acquire the lock before notifying to prevent the race
            // condition [r:check -> w:notify -> r:wait].
            drop(alloc_lock.lock());
            waiting_for_alloc.notify_all();
        }
    }

    fn finish_claim_attempt(
        &self,
        alloc_lock: &Mutex<TableLockState<K, V>>,
        waiting_for_alloc: &Condvar,
    ) {
        let old = self.claim_done_barrier.fetch_sub(1, Ordering::Release);
        if old == 1 {
            // We need to hold the lock while notifying to prevent the race
            // condition [r:check -> w:notify -> r:wait].
            drop(alloc_lock.lock());
            waiting_for_alloc.notify_all();
        }
    }
}

// A pointer to an entry in the table must be in one of three states:
//     0               = unclaimed entry
//     p               = claimed entry
//     usize::MAX      = claimed entry, now deleted
// The only valid transitions are those which move down the above list.
const UNCLAIMED: usize = 0;
const DELETED: usize = usize::MAX;

// The state field inside an entry is determined by the bottom three bits.
// If the DELETE_BIT is set then the entry is considered to be deleted and the
// upper bits contain the next pointer in the freelist. For this reason entries
// have to be aligned to at least 8 bytes. Otherwise, the top bits contain
// the hash.
const INIT_BIT: usize = 0b001;
const WAIT_BIT: usize = 0b010;
const DELETE_BIT: usize = 0b100;

#[repr(C, align(8))]
struct EntryHeader<K: ?Sized, V> {
    state: AtomicPtr<EntryHeader<K, V>>,
    value: MaybeUninit<V>,
    marker: PhantomData<K>,
}

impl<K: Key + ?Sized, V> EntryHeader<K, V> {
    fn layout(key: &K) -> Layout {
        let key_layout = Layout::from_size_align(key.size(), K::align()).unwrap();
        Layout::new::<EntryHeader<K, V>>()
            .extend(key_layout)
            .unwrap()
            .0
    }

    #[inline(always)]
    fn state_ptr(entry: *mut Self) -> *mut AtomicPtr<EntryHeader<K, V>> {
        unsafe { &raw mut (*entry).state }
    }

    #[inline(always)]
    fn val_ptr(entry: *mut Self) -> *mut V {
        unsafe { (&raw mut (*entry).value).cast() }
    }

    #[inline(always)]
    unsafe fn key_ptr(entry: *mut Self) -> *mut u8 {
        unsafe {
            entry
                .byte_add(size_of::<EntryHeader<K, V>>().next_multiple_of(K::align()))
                .cast()
        }
    }

    fn new(hash: usize, key: &K) -> *mut Self {
        let layout = Self::layout(key);
        unsafe {
            let p = std::alloc::alloc(layout).cast::<Self>();
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let state = without_provenance_mut(hash & !(INIT_BIT | WAIT_BIT | DELETE_BIT));
            Self::state_ptr(p).write(AtomicPtr::new(state));
            key.init(Self::key_ptr(p));
            p
        }
    }

    unsafe fn free(entry: *mut Self) {
        unsafe {
            let key = K::get(Self::key_ptr(entry));
            let layout = Self::layout(key);
            std::alloc::dealloc(entry.cast(), layout);
        }
    }

    // Waits for this entry to be initialized. Returns true if successful, false
    // if the value was deleted.
    unsafe fn wait_for_init(
        entry: *mut Self,
        init_lock: &Mutex<()>,
        waiting_for_init: &Condvar,
    ) -> bool {
        unsafe {
            let state_loc = &*Self::state_ptr(entry);
            let mut state = state_loc.load(Ordering::Acquire);
            if state.addr() & (DELETE_BIT | INIT_BIT) != 0 {
                return state.addr() & DELETE_BIT == 0;
            }

            // First acquire the lock then try setting the wait bit.
            let mut guard = init_lock.lock().unwrap();
            if let Err(new_state) = state_loc.compare_exchange(
                state,
                state.map_addr(|p| p | WAIT_BIT),
                Ordering::Relaxed,
                Ordering::Acquire,
            ) {
                state = new_state;
            }

            // Wait until init is complete.
            loop {
                if state.addr() & (DELETE_BIT | INIT_BIT) != 0 {
                    return state.addr() & DELETE_BIT == 0;
                }

                guard = waiting_for_init.wait(guard).unwrap();
                state = state_loc.load(Ordering::Acquire);
            }
        }
    }
}

/// A concurrent hash table.
#[repr(align(128))] // To avoid false sharing.
pub struct RawTable<K: Key + ?Sized, V> {
    cur_alloc: AtomicPtr<AllocHeader<K, V>>,
    freelist_head: AtomicPtr<EntryHeader<K, V>>,
    alloc_lock: Mutex<TableLockState<K, V>>,
    waiting_for_alloc: Condvar,
    init_lock: Mutex<()>,
    waiting_for_init: Condvar,
    rehash_lock: RwLock<()>,
    marker: PhantomData<(Box<K>, V)>,
}

unsafe impl<K: Key + Send + ?Sized, V: Send> Send for RawTable<K, V> {}
unsafe impl<K: Key + Send + Sync + ?Sized, V: Send + Sync> Sync for RawTable<K, V> {}

struct TableLockState<K: ?Sized, V> {
    old_allocs: Vec<*mut AllocHeader<K, V>>,
}

impl<K: Key + ?Sized, V> RawTable<K, V> {
    /// Creates a new [`RawTable`].
    pub const fn new() -> Self {
        Self {
            cur_alloc: AtomicPtr::new(&raw const EMPTY_ALLOC_LOC as _),
            freelist_head: AtomicPtr::new(core::ptr::null_mut()),
            alloc_lock: Mutex::new(TableLockState {
                old_allocs: Vec::new(),
            }),
            waiting_for_alloc: Condvar::new(),
            init_lock: Mutex::new(()),
            waiting_for_init: Condvar::new(),
            rehash_lock: RwLock::new(()),
            marker: PhantomData,
        }
    }

    /// Creates a new [`RawTable`] that will not reallocate before `capacity` insertions are done.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }
        Self {
            cur_alloc: AtomicPtr::new(AllocHeader::new(probe::min_entries_for_load(capacity))),
            freelist_head: AtomicPtr::new(core::ptr::null_mut()),
            alloc_lock: Mutex::new(TableLockState {
                old_allocs: Vec::new(),
            }),
            waiting_for_alloc: Condvar::new(),
            init_lock: Mutex::new(()),
            waiting_for_init: Condvar::new(),
            rehash_lock: RwLock::new(()),
            marker: PhantomData,
        }
    }

    fn start_insert_attempt(&self) -> *mut AllocHeader<K, V> {
        unsafe {
            let alloc = self.cur_alloc.load(Ordering::Acquire);
            if (*alloc).start_claim_attempt() {
                return alloc;
            }

            let mut guard = self.alloc_lock.lock().unwrap();
            loop {
                let alloc = self.cur_alloc.load(Ordering::Acquire);
                let header = &*alloc;
                if header.start_claim_attempt() {
                    return alloc;
                }

                let barrier = header.claim_done_barrier.load(Ordering::Acquire);
                if barrier == 0 {
                    guard = self.realloc(guard);
                } else {
                    guard = self.waiting_for_alloc.wait(guard).unwrap();
                }
            }
        }
    }

    unsafe fn realloc<'a>(
        &'a self,
        mut alloc_guard: MutexGuard<'a, TableLockState<K, V>>,
    ) -> MutexGuard<'a, TableLockState<K, V>> {
        unsafe {
            let old_alloc = self.cur_alloc.load(Ordering::Relaxed);
            let old_header = &*old_alloc;
            let upper_bound_len = probe::max_load(old_header.num_entries)
                - old_header.num_deletions.load(Ordering::Acquire);

            let num_entries = (upper_bound_len * 2).next_power_of_two().max(32);
            let alloc = AllocHeader::<K, V>::new(num_entries);
            let header = &*alloc;

            // Rehash old entries. We must hold the rehash lock exclusively to prevent those operations
            // which may not occur during rehashing.
            let rehash_guard = self.rehash_lock.write();
            let mut entries_reinserted = 0;
            for entry in old_header.entries(old_alloc) {
                let entry_ptr = entry.load(Ordering::Relaxed);
                if entry_ptr.addr() != DELETED && entry_ptr.addr() != UNCLAIMED {
                    let state = (*EntryHeader::state_ptr(entry_ptr)).load(Ordering::Relaxed);
                    if state.addr() & DELETE_BIT == 0 {
                        Self::insert_uniq_entry_exclusive(alloc, state.addr(), entry_ptr);
                        entries_reinserted += 1;
                    }
                }
            }

            // Publish the new allocation.
            header
                .claim_start_semaphore
                .fetch_sub(entries_reinserted, Ordering::Relaxed);
            header
                .claim_done_barrier
                .fetch_sub(entries_reinserted, Ordering::Relaxed);
            alloc_guard.old_allocs.push(old_alloc);
            self.cur_alloc.store(alloc, Ordering::Release);
            drop(rehash_guard);
            self.waiting_for_alloc.notify_all();
            alloc_guard
        }
    }

    unsafe fn try_init_entry_val<E>(
        &self,
        hash: usize,
        header: &AllocHeader<K, V>,
        entry: &AtomicPtr<EntryHeader<K, V>>,
        new_entry_ptr: *mut EntryHeader<K, V>,
        val_f: impl FnOnce(&K) -> Result<V, E>,
    ) -> Result<*mut EntryHeader<K, V>, E> {
        unsafe {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let key_ptr = EntryHeader::key_ptr(new_entry_ptr);
                let key = K::get(key_ptr);
                EntryHeader::val_ptr(new_entry_ptr).write(val_f(key)?);
                Ok(())
            }));
            let state = &*EntryHeader::state_ptr(new_entry_ptr);
            let old_state;
            if !matches!(r, Ok(Ok(_))) {
                let old_head = self.freelist_head.swap(new_entry_ptr, Ordering::AcqRel);
                old_state = state.swap(old_head.map_addr(|a| a | DELETE_BIT), Ordering::Release);
                entry.store(without_provenance_mut(DELETED), Ordering::Relaxed);
                header.num_deletions.fetch_add(1, Ordering::Release);
            } else {
                old_state = state.swap(
                    without_provenance_mut((hash & !(WAIT_BIT | DELETE_BIT)) | INIT_BIT),
                    Ordering::Release,
                )
            };

            if old_state.addr() & WAIT_BIT != 0 {
                // We need to hold the lock while notifying to prevent the race
                // condition [r:check -> w:notify -> r:wait].
                drop(self.init_lock.lock());
                self.waiting_for_init.notify_all();
            }

            match r {
                Ok(Ok(())) => Ok(new_entry_ptr),
                Ok(Err(e)) => Err(e),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
    }

    fn find_impl(
        alloc: *mut AllocHeader<K, V>,
        hash: usize,
        mut eq: impl FnMut(&K) -> bool,
    ) -> Result<(*mut EntryHeader<K, V>, usize), Prober> {
        unsafe {
            let mut prober = Prober::new(hash);

            let header = &*alloc;
            let tags = header.tags(alloc);
            let entries = header.entries(alloc);
            let group_mask = TagGroup::idx_mask(header.num_entries);
            let mut needle = TagGroup::all_occupied(hash);
            loop {
                let group_idx = prober.get() & group_mask;
                let mut tag_group = TagGroup::load(tags.get_unchecked(group_idx));
                let mut matches = tag_group.matches(&mut needle);
                while matches.has_matches() {
                    let idx_in_group = matches.get();
                    let entry_idx = size_of::<TagGroup>() * group_idx + idx_in_group;
                    let entry_ptr = entries.get_unchecked(entry_idx).load(Ordering::Acquire);

                    // Matching tag but unclaimed, racy insert in process but definitely missing.
                    if entry_ptr.addr() == UNCLAIMED {
                        return Err(prober);
                    }

                    if entry_ptr.addr() != DELETED {
                        let state = (*EntryHeader::state_ptr(entry_ptr)).load(Ordering::Acquire);
                        if state.addr() & DELETE_BIT == 0
                            && eq(K::get(EntryHeader::<K, V>::key_ptr(entry_ptr)))
                        {
                            // Not deleted and a key hit, either a racy insert or a hit.
                            return if state.addr() & INIT_BIT != 0 {
                                Ok((entry_ptr, entry_idx))
                            } else {
                                Err(prober)
                            };
                        }
                    }
                    matches.advance();
                }

                if tag_group.empties().has_matches() {
                    return Err(prober);
                }

                prober.advance();
            }
        }
    }

    fn try_find_or_insert_impl<E>(
        &self,
        orig_probe_alloc: *mut AllocHeader<K, V>,
        mut prober: Prober,
        hash: usize,
        key: &K,
        val_f: impl FnOnce(&K) -> Result<V, E>,
        mut eq: impl FnMut(&K) -> bool,
    ) -> Result<*mut EntryHeader<K, V>, E> {
        unsafe {
            let new_entry_ptr = EntryHeader::<K, V>::new(hash, key);

            let alloc = self.start_insert_attempt();
            if alloc != orig_probe_alloc {
                prober = Prober::new(hash);
            }

            let header = &*alloc;
            let tags = header.tags(alloc);
            let entries = header.entries(alloc);
            let group_mask = TagGroup::idx_mask(header.num_entries);
            let mut needle = TagGroup::all_occupied(hash);

            'probe_loop: loop {
                let group_idx = prober.get() & group_mask;
                let mut tag_group = TagGroup::load(tags.get_unchecked(group_idx));
                let matches = tag_group.matches(&mut needle);
                let empties = tag_group.empties();
                let mut insert_locs = matches | empties;
                while insert_locs.has_matches() {
                    let idx_in_group = insert_locs.get();

                    // Insert a new tag if this insert location.
                    if empties.has_match_at(idx_in_group) {
                        if !tags.get_unchecked(group_idx).try_occupy(
                            &mut tag_group,
                            idx_in_group,
                            hash,
                        ) {
                            continue 'probe_loop;
                        }
                    }

                    let entry_idx = size_of::<TagGroup>() * group_idx + idx_in_group;
                    let entry = entries.get_unchecked(entry_idx);
                    let mut entry_ptr = entry.load(Ordering::Acquire);
                    if entry_ptr.addr() == UNCLAIMED {
                        // Try to claim this entry.
                        match entry.compare_exchange(
                            entry_ptr,
                            new_entry_ptr,
                            Ordering::Release,
                            Ordering::Acquire,
                        ) {
                            Ok(_) => {
                                header.finish_claim_attempt(
                                    &self.alloc_lock,
                                    &self.waiting_for_alloc,
                                );
                                return self.try_init_entry_val(
                                    hash,
                                    header,
                                    entry,
                                    new_entry_ptr,
                                    val_f,
                                );
                            },
                            Err(ev) => entry_ptr = ev,
                        }
                    }

                    // We couldn't claim the entry, see if our key is the same as
                    // whoever claimed this entry, assuming it's not deleted.
                    if entry_ptr.addr() != DELETED {
                        let entry_key = K::get(EntryHeader::key_ptr(entry_ptr));
                        if eq(entry_key) {
                            if EntryHeader::wait_for_init(
                                entry_ptr,
                                &self.init_lock,
                                &self.waiting_for_init,
                            ) {
                                EntryHeader::free(new_entry_ptr);
                                header
                                    .abort_claim_attempt(&self.alloc_lock, &self.waiting_for_alloc);
                                return Ok(entry_ptr);
                            }
                        }
                    }

                    insert_locs.advance();
                }

                prober.advance();
            }
        }
    }

    unsafe fn insert_uniq_entry_exclusive(
        alloc: *mut AllocHeader<K, V>,
        hash: usize,
        uniq_entry_ptr: *mut EntryHeader<K, V>,
    ) {
        unsafe {
            let header = &mut *alloc;
            let tags = header.tags_mut(alloc);
            let entries = header.entries_mut(alloc);
            let group_mask = TagGroup::idx_mask(header.num_entries);

            let mut prober = Prober::new(hash);
            loop {
                let group_idx = prober.get() & group_mask;
                let tag_group = tags.get_unchecked_mut(group_idx);
                let empties = tag_group.empties();
                if empties.has_matches() {
                    let idx_in_group = empties.get();
                    tag_group.occupy_mut(idx_in_group, hash);
                    let entry_idx = size_of::<TagGroup>() * group_idx + idx_in_group;
                    *entries.get_unchecked_mut(entry_idx).get_mut() = uniq_entry_ptr;
                    return;
                }

                prober.advance();
            }
        }
    }

    /// Free any resources which are no longer necessary.
    ///
    /// # Safety
    /// Until drop_guard gets called, there may not be any alive references
    /// returned by the [`RawTable`], or concurrent other operations.
    pub unsafe fn gc<F: FnOnce()>(&self, drop_guard: F) {
        let mut freelist_head = self
            .freelist_head
            .swap(core::ptr::null_mut(), Ordering::Acquire);
        let old_allocs = core::mem::take(&mut self.alloc_lock.lock().unwrap().old_allocs);
        drop_guard();

        unsafe {
            while !freelist_head.is_null() {
                let state = *(*EntryHeader::state_ptr(freelist_head)).get_mut();
                if state.addr() & INIT_BIT != 0 {
                    core::ptr::drop_in_place(EntryHeader::val_ptr(freelist_head));
                }
                K::drop_in_place(EntryHeader::key_ptr(freelist_head));
                EntryHeader::free(freelist_head);
                freelist_head = state.map_addr(|a| a & !(INIT_BIT | WAIT_BIT | DELETE_BIT));
            }

            for alloc in old_allocs {
                AllocHeader::free(alloc);
            }
        }
    }

    /// Finds the value corresponding to a key with the given hash and equality function.
    pub fn get(&self, hash: u64, eq: impl FnMut(&K) -> bool) -> Option<&V> {
        unsafe {
            let cur_alloc = self.cur_alloc.load(Ordering::Acquire);
            let entry_ptr = Self::find_impl(cur_alloc, hash as usize, eq).ok()?.0;
            Some(&*EntryHeader::val_ptr(entry_ptr))
        }
    }

    /// Finds the value corresponding to a key with the given hash and equality function, or insert
    /// a new one if the key does not exist.
    ///
    /// `val_f` is guaranteed to only be called when inserting a new key not currently found in the
    /// table, even if multiple concurrent inserts occur. The key reference passed to `val_f` lives
    /// as long as the new entry will.
    pub fn get_or_insert_with(
        &self,
        hash: u64,
        key: &K,
        eq: impl FnMut(&K) -> bool,
        val_f: impl FnOnce(&K) -> V,
    ) -> &V {
        unsafe {
            self.try_get_or_insert_with::<()>(hash, key, eq, |k| Ok(val_f(k)))
                .unwrap_unchecked()
        }
    }

    /// Finds the value corresponding to a key with the given hash and equality function, or insert
    /// a new one if the key does not exist.
    ///
    /// `val_f` is guaranteed to only be called when inserting a new key not currently found in the
    /// table, even if multiple concurrent inserts occur. The key reference passed to `val_f` lives
    /// as long as the new entry will.
    pub fn try_get_or_insert_with<E>(
        &self,
        hash: u64,
        key: &K,
        mut eq: impl FnMut(&K) -> bool,
        val_f: impl FnOnce(&K) -> Result<V, E>,
    ) -> Result<&V, E> {
        unsafe {
            let cur_alloc = self.cur_alloc.load(Ordering::Acquire);
            match Self::find_impl(cur_alloc, hash as usize, &mut eq) {
                Ok((entry_ptr, _)) => Ok(&*EntryHeader::val_ptr(entry_ptr)),
                Err(prober) => {
                    let entry_ptr = self.try_find_or_insert_impl(
                        cur_alloc,
                        prober,
                        hash as usize,
                        key,
                        val_f,
                        eq,
                    )?;
                    Ok(&*EntryHeader::val_ptr(entry_ptr))
                },
            }
        }
    }

    /// Finds and removes the value corresponding to a key with the given hash and equality function.
    ///
    /// Note that the value is not dropped until [`RawTable::gc`] is called or the [`RawTable`] is dropped.
    pub fn remove(&self, hash: u64, eq: impl FnMut(&K) -> bool) -> Option<&V> {
        unsafe {
            // TODO: perhaps deletions could be done during a rehash by synchronizing on state?
            let _rehash_guard = self.rehash_lock.read();
            let alloc = self.cur_alloc.load(Ordering::Acquire);
            let header = &*alloc;
            let (entry_ptr, entry_idx) = Self::find_impl(alloc, hash as usize, eq).ok()?;

            let state = &(*entry_ptr).state;
            let old_state = state
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |s| {
                    if s.addr() & DELETE_BIT != 0 {
                        return None;
                    }

                    Some(s.map_addr(|a| a | DELETE_BIT))
                })
                .ok()?;

            let group_idx = entry_idx / size_of::<TagGroup>();
            let idx_in_group = entry_idx % size_of::<TagGroup>();
            let old_head = self.freelist_head.swap(entry_ptr, Ordering::AcqRel);
            state.store(
                old_head.map_addr(|a| a | (old_state.addr() & INIT_BIT)),
                Ordering::Release,
            );
            header
                .entries(alloc)
                .get_unchecked(entry_idx)
                .store(without_provenance_mut(DELETED), Ordering::Relaxed);
            header
                .tags(alloc)
                .get_unchecked(group_idx)
                .delete(idx_in_group);
            header.num_deletions.fetch_add(1, Ordering::Release);
            Some(&*EntryHeader::val_ptr(entry_ptr))
        }
    }
}

impl<K: Key + ?Sized, V> Default for RawTable<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key + ?Sized, V> Drop for RawTable<K, V> {
    fn drop(&mut self) {
        unsafe {
            self.gc(|| {});

            let alloc = *self.cur_alloc.get_mut();
            let header = &*alloc;
            for entry in header.entries_mut(alloc) {
                let entry_ptr = *(*entry).get_mut();
                if entry_ptr.is_null() || entry_ptr.addr() == DELETED {
                    continue;
                }

                let state = (*EntryHeader::state_ptr(entry_ptr)).get_mut();
                if state.addr() & INIT_BIT != 0 {
                    core::ptr::drop_in_place(EntryHeader::val_ptr(entry_ptr));
                }
                K::drop_in_place(EntryHeader::key_ptr(entry_ptr));
                EntryHeader::free(entry_ptr);
            }
            AllocHeader::free(alloc);
        }
    }
}

#[repr(C)]
struct DummyAlloc {
    header: AllocHeader<(), ()>,
    tags: [TagGroup; 1],
    entries: [AtomicPtr<EntryHeader<(), ()>>; size_of::<TagGroup>()],
}

static EMPTY_ALLOC_LOC: DummyAlloc = DummyAlloc {
    header: AllocHeader {
        num_entries: size_of::<TagGroup>(),
        num_deletions: AtomicUsize::new(0),
        claim_start_semaphore: AtomicUsize::new(0),
        claim_done_barrier: AtomicUsize::new(0),
        marker: PhantomData,
        align: [],
    },
    tags: [TagGroup::all_empty()],
    entries: [
        AtomicPtr::new(without_provenance_mut(UNCLAIMED)),
        AtomicPtr::new(without_provenance_mut(UNCLAIMED)),
        AtomicPtr::new(without_provenance_mut(UNCLAIMED)),
        AtomicPtr::new(without_provenance_mut(UNCLAIMED)),
        AtomicPtr::new(without_provenance_mut(UNCLAIMED)),
        AtomicPtr::new(without_provenance_mut(UNCLAIMED)),
        AtomicPtr::new(without_provenance_mut(UNCLAIMED)),
        AtomicPtr::new(without_provenance_mut(UNCLAIMED)),
    ],
};
