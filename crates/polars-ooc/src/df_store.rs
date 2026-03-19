use std::cell::UnsafeCell;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use polars_core::prelude::DataFrame;
use tokio::sync::Notify;

fn yield_spin() {
    std::hint::spin_loop();
}

const INITIAL_SEG_SIZE: u32 = 1024;

const NUM_SEGMENTS: usize = 20;
const SENTINEL: u32 = u32::MAX;

// SlotMeta bit layout (AtomicU64):
// [63..60] 4 bits: SlotState
// [59..58] 2 bits: SpillState
// [57]     1 bit:  is_pinned
// [56..32] 25 bits: reserved (zero)
// [31..0]  32 bits: generation counter

const STATE_SHIFT: u32 = 60;
const STATE_MASK: u64 = 0xF << STATE_SHIFT;
const SPILL_SHIFT: u32 = 58;
const SPILL_MASK: u64 = 0x3 << SPILL_SHIFT;
const PIN_BIT: u64 = 1 << 57;
const GEN_MASK: u64 = 0xFFFF_FFFF;

#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free = 0,
    Reserved = 1,
    Occupied = 2,
    Locked = 3,
}

#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpillState {
    /// DataFrame is in RAM, ready to read.
    InMemory = 0,
    /// DataFrame is on disk (IPC file). RAM is freed.
    Spilled = 1,
    /// One thread is reading the IPC file back into RAM. Others must wait.
    Loading = 2,
    /// DataFrame has been taken out for writing to disk but the file
    /// is not written yet. Readers must wait.
    Spilling = 3,
}

/// Packed meta word for a slot. Wraps a `u64` with accessor methods.
#[derive(Clone, Copy)]
pub(crate) struct SlotMeta(u64);

impl SlotMeta {
    #[inline]
    fn new(state: SlotState, spill: SpillState, pinned: bool, generation: u32) -> Self {
        Self(
            ((state as u64) << STATE_SHIFT)
                | ((spill as u64) << SPILL_SHIFT)
                | if pinned { PIN_BIT } else { 0 }
                | generation as u64,
        )
    }

    #[inline]
    fn state(self) -> SlotState {
        match (self.0 & STATE_MASK) >> STATE_SHIFT {
            0 => SlotState::Free,
            1 => SlotState::Reserved,
            2 => SlotState::Occupied,
            3 => SlotState::Locked,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub(crate) fn spill(self) -> SpillState {
        match (self.0 & SPILL_MASK) >> SPILL_SHIFT {
            0 => SpillState::InMemory,
            1 => SpillState::Spilled,
            2 => SpillState::Loading,
            3 => SpillState::Spilling,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub(crate) fn is_pinned(self) -> bool {
        self.0 & PIN_BIT != 0
    }

    #[inline]
    pub(crate) fn generation(self) -> u32 {
        (self.0 & GEN_MASK) as u32
    }

    #[inline]
    fn with_state(self, state: SlotState) -> Self {
        Self((self.0 & !STATE_MASK) | ((state as u64) << STATE_SHIFT))
    }

    #[inline]
    fn with_spill(self, spill: SpillState) -> Self {
        Self((self.0 & !SPILL_MASK) | ((spill as u64) << SPILL_SHIFT))
    }
}

/// Result of [`DataFrameStore::try_load`].
pub(crate) enum LoadStatus {
    /// Entry is already in memory — no action needed.
    AlreadyLoaded,
    /// Caller transitioned `Spilled → Loading` — must read from disk
    /// and call [`DataFrameStore::finish_load`].
    Claimed,
    /// Another thread is already loading — register a waker and wait.
    Waiting,
}

#[inline]
fn pack_free(aba: u32, idx: u32) -> u64 {
    ((aba as u64) << 32) | idx as u64
}

#[inline]
fn unpack_free(raw: u64) -> (u32, u32) {
    let aba = (raw >> 32) as u32;
    let idx = raw as u32;
    (aba, idx)
}

/// Number of slots in segment `seg_idx`.
#[inline]
fn seg_size(seg_idx: usize) -> u32 {
    INITIAL_SEG_SIZE << seg_idx
}

struct Entry {
    df: Option<DataFrame>,
}

pub(crate) struct Slot {
    meta: AtomicU64,
    pub(crate) notify: Notify,
    next_free: AtomicU32,
    height: AtomicUsize,
    size_bytes: AtomicUsize,
    /// Monotonic counter incremented on each spill. Makes each spill
    /// filename unique (`spill_<index>_<gen>_<seq>.ipc`) so that async
    /// file deletions from a previous reload can never destroy a
    /// re-spilled file.
    spill_seq: AtomicU32,
    entry: UnsafeCell<Entry>,
}

impl Slot {
    fn new() -> Self {
        Self {
            meta: AtomicU64::new(SlotMeta::new(SlotState::Free, SpillState::InMemory, false, 0).0),
            notify: Notify::new(),
            next_free: AtomicU32::new(SENTINEL),
            height: AtomicUsize::new(0),
            size_bytes: AtomicUsize::new(0),
            spill_seq: AtomicU32::new(0),
            entry: UnsafeCell::new(Entry { df: None }),
        }
    }
}

/// Lock-free segmented slot store with Treiber stack free-list.
///
/// Each slot stores an optional [`DataFrame`] behind atomic state-machine
/// guards. Segments grow on demand and are never moved once allocated.
pub(crate) struct DataFrameStore {
    segments: [AtomicPtr<Slot>; NUM_SEGMENTS],
    // Potential optimization: shard into K independent free-lists (push by idx % K,
    // pop round-robin) to reduce CAS contention under high parallelism.
    free_head: AtomicU64,
    capacity: AtomicU32,
    len: AtomicU32,
}

unsafe impl Send for DataFrameStore {}
unsafe impl Sync for DataFrameStore {}

impl DataFrameStore {
    pub(crate) fn new() -> Self {
        let segments: [AtomicPtr<Slot>; NUM_SEGMENTS] =
            std::array::from_fn(|_| AtomicPtr::new(ptr::null_mut()));
        let store = Self {
            segments,
            free_head: AtomicU64::new(pack_free(0, SENTINEL)),
            capacity: AtomicU32::new(0),
            len: AtomicU32::new(0),
        };
        store.grow();
        store
    }

    /// O(1) index to slot reference.
    ///
    /// ```text
    /// adj    = index + INITIAL_SEG_SIZE
    /// seg    = floor(log2(adj)) - log2(INITIAL_SEG_SIZE)
    /// offset = adj - (INITIAL_SEG_SIZE << seg)
    /// ```
    pub(crate) fn locate(&self, index: u32) -> &Slot {
        let adj = index + INITIAL_SEG_SIZE;
        let seg = (u32::BITS - 1 - adj.leading_zeros()) - INITIAL_SEG_SIZE.trailing_zeros();
        let offset = adj - (INITIAL_SEG_SIZE << seg);
        let seg_ptr = self.segments[seg as usize].load(Ordering::Acquire);
        debug_assert!(!seg_ptr.is_null(), "segment {seg} not allocated");
        // SAFETY: seg_ptr points to a valid allocation of seg_size(seg) Slots.
        // offset < seg_size(seg) by construction.
        unsafe { &*seg_ptr.add(offset as usize) }
    }

    /// Allocate the next segment and push its slots onto the free-list.
    fn grow(&self) {
        let old_cap = self.capacity.load(Ordering::Acquire);
        let adj = old_cap + INITIAL_SEG_SIZE;
        let seg = (u32::BITS - 1 - adj.leading_zeros()) - INITIAL_SEG_SIZE.trailing_zeros();
        if seg as usize >= NUM_SEGMENTS {
            panic!("DataFrameStore: exceeded maximum segment count ({NUM_SEGMENTS})");
        }
        let size = seg_size(seg as usize);

        let ptr = alloc_segment(size);

        // CAS null → ptr. Only one thread wins.
        if self.segments[seg as usize]
            .compare_exchange(ptr::null_mut(), ptr, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            // Another thread allocated this segment — free ours.
            dealloc_segment(ptr, size);
            return;
        }

        // We won: update capacity and push all new slots onto the free-list.
        self.capacity.store(old_cap + size, Ordering::Release);
        self.push_range_to_freelist(old_cap, size);
    }

    /// Link a contiguous range of slot indices into the free-list.
    fn push_range_to_freelist(&self, first_index: u32, count: u32) {
        let last_index = first_index + count - 1;

        // Set up internal chain: slot[k].next_free = k+1.
        for i in first_index..last_index {
            self.locate(i).next_free.store(i + 1, Ordering::Relaxed);
        }

        // CAS loop to link the chain onto the existing free-list.
        loop {
            let old = self.free_head.load(Ordering::Acquire);
            let (aba, old_idx) = unpack_free(old);
            // Last slot links to old free-list head.
            self.locate(last_index)
                .next_free
                .store(old_idx, Ordering::Relaxed);
            let new = pack_free(aba.wrapping_add(1), first_index);
            if self
                .free_head
                .compare_exchange_weak(old, new, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Pop an index from the free-list. Grows a new segment if empty.
    fn pop_free(&self) -> u32 {
        loop {
            let old = self.free_head.load(Ordering::Acquire);
            let (aba, idx) = unpack_free(old);
            if idx == SENTINEL {
                self.grow();
                continue;
            }
            let next = self.locate(idx).next_free.load(Ordering::Relaxed);
            let new = pack_free(aba.wrapping_add(1), next);
            if self
                .free_head
                .compare_exchange_weak(old, new, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                // Mark as Reserved (no one else can see this slot).
                let slot = self.locate(idx);
                let old_meta = SlotMeta(slot.meta.load(Ordering::Relaxed));
                debug_assert_eq!(old_meta.state(), SlotState::Free);
                slot.meta.store(
                    SlotMeta::new(
                        SlotState::Reserved,
                        SpillState::InMemory,
                        false,
                        old_meta.generation(),
                    )
                    .0,
                    Ordering::Relaxed,
                );
                return idx;
            }
            yield_spin();
        }
    }

    /// Push an index back onto the free-list.
    fn push_free(&self, idx: u32) {
        loop {
            let old = self.free_head.load(Ordering::Acquire);
            let (aba, old_idx) = unpack_free(old);
            self.locate(idx).next_free.store(old_idx, Ordering::Relaxed);
            let new = pack_free(aba.wrapping_add(1), idx);
            if self
                .free_head
                .compare_exchange_weak(old, new, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
            yield_spin();
        }
    }

    /// Insert a [`DataFrame`] into the store. Returns `(index, generation)`.
    pub(crate) fn insert(&self, df: DataFrame, size_bytes: usize) -> (u32, u32) {
        let height = df.height();
        let idx = self.pop_free();
        let slot = self.locate(idx);

        // SAFETY: Reserved state — only we can access this slot.
        unsafe { (*slot.entry.get()).df = Some(df) };

        slot.height.store(height, Ordering::Relaxed);
        slot.size_bytes.store(size_bytes, Ordering::Relaxed);
        slot.spill_seq.store(0, Ordering::Relaxed);

        // Read old generation and bump it.
        let old_meta = SlotMeta(slot.meta.load(Ordering::Relaxed));
        let new_gen = old_meta.generation().wrapping_add(1);

        // Publish: Release-store Occupied with new generation.
        slot.meta.store(
            SlotMeta::new(SlotState::Occupied, SpillState::InMemory, false, new_gen).0,
            Ordering::Release,
        );

        self.len.fetch_add(1, Ordering::Relaxed);
        (idx, new_gen)
    }

    /// CAS Occupied(InMemory) → Locked. Shared logic for [`get`] and [`get_mut`].
    fn lock_occupied(&self, index: u32, generation: u32) -> Option<(&Slot, SlotMeta)> {
        let slot = self.locate(index);
        loop {
            let meta = SlotMeta(slot.meta.load(Ordering::Acquire));
            if meta.generation() != generation {
                return None;
            }
            match meta.state() {
                SlotState::Occupied => {
                    if meta.spill() != SpillState::InMemory {
                        return None;
                    }
                    let locked = meta.with_state(SlotState::Locked);
                    if slot
                        .meta
                        .compare_exchange_weak(
                            meta.0,
                            locked.0,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return Some((slot, meta));
                    }
                },
                SlotState::Locked => yield_spin(),
                SlotState::Free | SlotState::Reserved => return None,
            }
        }
    }

    /// Acquire a read guard. CAS Occupied → Locked.
    ///
    /// Returns `None` if the generation doesn't match, the slot is not
    /// Occupied, or the entry is not InMemory (spilled/loading).
    /// Callers must [`ensure_loaded`](crate::MemoryManager::ensure_loaded)
    /// before calling this.
    pub(crate) fn get(&self, index: u32, generation: u32) -> Option<SlotReadGuard<'_>> {
        let (slot, meta) = self.lock_occupied(index, generation)?;
        Some(SlotReadGuard {
            slot,
            occupied_meta: meta,
        })
    }

    /// Acquire a write guard. CAS Occupied → Locked.
    ///
    /// Same CAS as [`get`](Self::get) but the guard provides mutable access.
    /// Returns `None` if the entry is not InMemory.
    pub(crate) fn get_mut(&self, index: u32, generation: u32) -> Option<SlotWriteGuard<'_>> {
        let (slot, meta) = self.lock_occupied(index, generation)?;
        Some(SlotWriteGuard {
            slot,
            occupied_meta: meta,
        })
    }

    /// Take the [`DataFrame`] out of the slot. CAS Occupied → Free.
    ///
    /// Only succeeds if the entry is InMemory. Returns `None` if the
    /// generation doesn't match, the slot is not Occupied, or the entry
    /// is spilled/loading.
    pub(crate) fn take(&self, index: u32, generation: u32) -> Option<(DataFrame, usize)> {
        let slot = self.locate(index);
        loop {
            let meta = SlotMeta(slot.meta.load(Ordering::Acquire));
            if meta.generation() != generation {
                return None;
            }
            match meta.state() {
                SlotState::Occupied => {
                    if meta.spill() != SpillState::InMemory {
                        return None;
                    }
                    let freed =
                        SlotMeta::new(SlotState::Free, SpillState::InMemory, false, generation);
                    if slot
                        .meta
                        .compare_exchange_weak(meta.0, freed.0, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        let size = slot.size_bytes.load(Ordering::Relaxed);
                        // SAFETY: CAS succeeded — exclusive access to entry.
                        let df = unsafe { (*slot.entry.get()).df.take().unwrap() };
                        slot.height.store(0, Ordering::Relaxed);
                        slot.size_bytes.store(0, Ordering::Relaxed);
                        self.len.fetch_sub(1, Ordering::Relaxed);
                        self.push_free(index);
                        return Some((df, size));
                    }
                },
                SlotState::Locked => yield_spin(),
                SlotState::Free | SlotState::Reserved => return None,
            }
        }
    }

    /// Force-free a slot regardless of SpillState. For [`Token::drop`] cleanup.
    ///
    /// Returns `(old_spill_state, old_size_bytes, spill_seq)` so
    /// the caller can decide whether to delete a spill file and how much
    /// drift to subtract.
    /// Spin-yields on Locked, Loading, and Spilling states (all bounded).
    pub(crate) fn remove(&self, index: u32, generation: u32) -> Option<(SpillState, usize, u32)> {
        let slot = self.locate(index);
        loop {
            let meta = SlotMeta(slot.meta.load(Ordering::Acquire));
            if meta.generation() != generation {
                return None;
            }
            match meta.state() {
                SlotState::Occupied => {
                    let spill = meta.spill();
                    if matches!(spill, SpillState::Loading | SpillState::Spilling) {
                        yield_spin();
                        continue;
                    }
                    let freed =
                        SlotMeta::new(SlotState::Free, SpillState::InMemory, false, generation);
                    if slot
                        .meta
                        .compare_exchange_weak(meta.0, freed.0, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        let size = slot.size_bytes.load(Ordering::Relaxed);
                        let seq = slot.spill_seq.load(Ordering::Relaxed);
                        // SAFETY: CAS succeeded — exclusive access to entry.
                        unsafe { (*slot.entry.get()).df = None };
                        slot.height.store(0, Ordering::Relaxed);
                        slot.size_bytes.store(0, Ordering::Relaxed);
                        self.len.fetch_sub(1, Ordering::Relaxed);
                        self.push_free(index);
                        return Some((spill, size, seq));
                    }
                },
                SlotState::Locked => yield_spin(),
                SlotState::Free | SlotState::Reserved => return None,
            }
        }
    }

    /// Try to spill an InMemory entry to disk. CAS Occupied → Locked → Occupied(Spilling).
    ///
    /// Returns `(DataFrame, size_bytes, spill_seq)` if successful. The
    /// `spill_seq` is a monotonically increasing counter used to create
    /// unique filenames. The caller writes the DataFrame to disk and then
    /// calls [`finish_spill`](Self::finish_spill) to transition to `Spilled`.
    /// Returns `None` if the entry is not eligible (wrong generation, not
    /// Occupied, not InMemory, or pinned).
    pub(crate) fn try_spill(&self, index: u32, generation: u32) -> Option<(DataFrame, usize, u32)> {
        let slot = self.locate(index);
        loop {
            let meta = SlotMeta(slot.meta.load(Ordering::Acquire));
            if meta.generation() != generation {
                return None;
            }
            match meta.state() {
                SlotState::Occupied => {
                    if meta.spill() != SpillState::InMemory || meta.is_pinned() {
                        return None;
                    }
                    let locked = meta.with_state(SlotState::Locked);
                    if slot
                        .meta
                        .compare_exchange_weak(
                            meta.0,
                            locked.0,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        // SAFETY: Locked — exclusive access to entry.
                        let df = unsafe { (*slot.entry.get()).df.take() };
                        let size = slot.size_bytes.load(Ordering::Relaxed);
                        slot.size_bytes.store(0, Ordering::Relaxed);
                        // Increment spill_seq for a unique filename.
                        let seq = slot.spill_seq.load(Ordering::Relaxed);
                        slot.spill_seq.store(seq + 1, Ordering::Relaxed);
                        // Transition to (Occupied, Spilling). The caller
                        // must call finish_spill() after writing to disk.
                        slot.meta.store(
                            SlotMeta::new(
                                SlotState::Occupied,
                                SpillState::Spilling,
                                false,
                                generation,
                            )
                            .0,
                            Ordering::Release,
                        );
                        return Some((df.unwrap(), size, seq + 1));
                    }
                },
                SlotState::Locked => yield_spin(),
                _ => return None,
            }
        }
    }

    /// Complete a spill: transition Spilling → Spilled and wake waiters.
    ///
    /// Called by the thread that received `Some(...)` from
    /// [`try_spill`](Self::try_spill) after writing the file to disk.
    pub(crate) fn finish_spill(&self, index: u32, generation: u32) {
        let slot = self.locate(index);
        let meta = SlotMeta(slot.meta.load(Ordering::Relaxed));
        debug_assert_eq!(meta.spill(), SpillState::Spilling);
        slot.meta.store(
            SlotMeta::new(
                SlotState::Occupied,
                SpillState::Spilled,
                meta.is_pinned(),
                generation,
            )
            .0,
            Ordering::Release,
        );
        slot.notify.notify_waiters();
    }

    /// Try to claim load responsibility for a spilled entry.
    ///
    /// CAS (Occupied, Spilled) → (Occupied, Loading). Returns
    /// [`LoadStatus`] indicating what the caller should do.
    pub(crate) fn try_load(&self, index: u32, generation: u32) -> LoadStatus {
        let slot = self.locate(index);
        loop {
            let meta = SlotMeta(slot.meta.load(Ordering::Acquire));
            if meta.generation() != generation {
                return LoadStatus::AlreadyLoaded;
            }
            match meta.state() {
                SlotState::Occupied => match meta.spill() {
                    SpillState::InMemory => return LoadStatus::AlreadyLoaded,
                    SpillState::Loading | SpillState::Spilling => return LoadStatus::Waiting,
                    SpillState::Spilled => {
                        let loading = meta.with_spill(SpillState::Loading);
                        if slot
                            .meta
                            .compare_exchange_weak(
                                meta.0,
                                loading.0,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            )
                            .is_ok()
                        {
                            return LoadStatus::Claimed;
                        }
                        // CAS failed — retry (another thread may have claimed it).
                    },
                },
                SlotState::Locked => yield_spin(),
                _ => return LoadStatus::AlreadyLoaded,
            }
        }
    }

    /// Complete a load: put the DataFrame back, wake all parked waiters.
    ///
    /// Called by the single thread that received [`LoadStatus::Claimed`]
    /// from [`try_load`](Self::try_load).
    pub(crate) fn finish_load(&self, index: u32, generation: u32, df: DataFrame) {
        let slot = self.locate(index);
        let height = df.height();
        let size = df.estimated_size();

        // SAFETY: Loading state — only the loader can write to the entry.
        // get/get_mut/take all return None for non-InMemory entries.
        unsafe { (*slot.entry.get()).df = Some(df) };
        slot.height.store(height, Ordering::Relaxed);
        slot.size_bytes.store(size, Ordering::Relaxed);

        // Transition to InMemory. Release-store publishes the df write.
        let meta = SlotMeta(slot.meta.load(Ordering::Relaxed));
        slot.meta.store(
            SlotMeta::new(
                SlotState::Occupied,
                SpillState::InMemory,
                meta.is_pinned(),
                generation,
            )
            .0,
            Ordering::Release,
        );

        slot.notify.notify_waiters();
    }

    /// Row count of the stored DataFrame. Single atomic load (Relaxed).
    pub(crate) fn height(&self, index: u32) -> usize {
        self.locate(index).height.load(Ordering::Relaxed)
    }

    /// Estimated byte size of the stored DataFrame. Single atomic load.
    pub(crate) fn size_bytes(&self, index: u32) -> usize {
        self.locate(index).size_bytes.load(Ordering::Relaxed)
    }

    /// Current spill sequence number. Used by the loader to know which
    /// file to read (`spill_<index>_<gen>_<seq>.ipc`).
    pub(crate) fn spill_seq(&self, index: u32) -> u32 {
        self.locate(index).spill_seq.load(Ordering::Relaxed)
    }

    /// Set or clear the pin bit on a slot's meta.
    pub(crate) fn set_pinned(&self, index: u32, generation: u32, pinned: bool) {
        let slot = self.locate(index);
        loop {
            let meta = SlotMeta(slot.meta.load(Ordering::Acquire));
            if meta.generation() != generation {
                return;
            }
            let new = if pinned {
                meta.0 | PIN_BIT
            } else {
                meta.0 & !PIN_BIT
            };
            if meta.0 == new {
                return;
            }
            if slot
                .meta
                .compare_exchange_weak(meta.0, new, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Load the meta word for a slot.
    pub(crate) fn load_meta(&self, index: u32) -> SlotMeta {
        SlotMeta(self.locate(index).meta.load(Ordering::Acquire))
    }

    /// Number of occupied slots. Approximate because Relaxed ordering
    /// means concurrent inserts/removes may not be visible yet.
    /// Only used for diagnostics (Debug output).
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> u32 {
        self.len.load(Ordering::Relaxed)
    }
}

impl Drop for DataFrameStore {
    fn drop(&mut self) {
        for seg_idx in 0..NUM_SEGMENTS {
            // &mut self guarantees exclusive access — Relaxed is fine here.
            let ptr = self.segments[seg_idx].load(Ordering::Relaxed);
            if ptr.is_null() {
                continue;
            }
            dealloc_segment(ptr, seg_size(seg_idx));
        }
    }
}

fn alloc_segment(size: u32) -> *mut Slot {
    let slots: Vec<Slot> = (0..size).map(|_| Slot::new()).collect();
    Box::into_raw(slots.into_boxed_slice()) as *mut Slot
}

fn dealloc_segment(ptr: *mut Slot, size: u32) {
    // SAFETY: ptr came from Box::into_raw of a Box<[Slot]> with `size` elements.
    unsafe {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, size as usize));
    }
}

/// Read guard returned by [`DataFrameStore::get`]. Transitions Locked → Occupied on drop.
pub(crate) struct SlotReadGuard<'a> {
    slot: &'a Slot,
    occupied_meta: SlotMeta,
}

impl<'a> SlotReadGuard<'a> {
    pub(crate) fn with_df<R>(&self, f: impl FnOnce(&DataFrame) -> R) -> R {
        // SAFETY: Locked state guarantees exclusive access.
        let entry = unsafe { &*self.slot.entry.get() };
        f(entry.df.as_ref().unwrap())
    }
}

impl<'a> Drop for SlotReadGuard<'a> {
    fn drop(&mut self) {
        // Restore to Occupied. Release-store so subsequent Acquire sees
        // the same entry data.
        self.slot
            .meta
            .store(self.occupied_meta.0, Ordering::Release);
    }
}

/// Write guard returned by [`DataFrameStore::get_mut`]. Updates height/size_bytes
/// and transitions Locked → Occupied on drop.
pub(crate) struct SlotWriteGuard<'a> {
    slot: &'a Slot,
    occupied_meta: SlotMeta,
}

impl<'a> SlotWriteGuard<'a> {
    pub(crate) fn with_df_mut<R>(&mut self, f: impl FnOnce(&mut DataFrame) -> R) -> R {
        // SAFETY: Locked state guarantees exclusive access.
        let entry = unsafe { &mut *self.slot.entry.get() };
        f(entry.df.as_mut().unwrap())
    }
}

impl<'a> Drop for SlotWriteGuard<'a> {
    fn drop(&mut self) {
        // Update height and size_bytes from the (potentially mutated) entry.
        // SAFETY: Locked state guarantees exclusive access.
        let entry = unsafe { &*self.slot.entry.get() };
        if let Some(ref df) = entry.df {
            self.slot.height.store(df.height(), Ordering::Relaxed);
            self.slot
                .size_bytes
                .store(df.estimated_size(), Ordering::Relaxed);
        }
        // Restore to Occupied. Release-store to publish any mutations.
        self.slot
            .meta
            .store(self.occupied_meta.0, Ordering::Release);
    }
}
