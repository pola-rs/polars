//! Exporting Map rows to arrow drops the entries of null rows, which must cost memory per null
//! row, not per entry they span.
#![cfg(feature = "dtype-map")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use polars_arrow::array::{Array, BooleanArray, ListArray, MapArray, NullArray, StructArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::Offsets;
use polars_core::prelude::*;

/// Counts the bytes allocated, to measure the peak of an export.
struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn add(bytes: usize) {
    let current = CURRENT.fetch_add(bytes, Ordering::SeqCst) + bytes;
    PEAK.fetch_max(current, Ordering::SeqCst);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            add(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            add(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
            add(new_size);
        }
        new
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The most bytes `f` holds allocated at once, beyond those allocated before it.
fn peak_allocation<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let out = f();
    (out, PEAK.load(Ordering::SeqCst) - base)
}

fn map_dtype() -> DataType {
    DataType::Map(Box::new(DataType::Boolean), Box::new(DataType::Null))
}

/// A `Map(Boolean, Null)` Series of one chunk with `lens` entries per row; its zeroed keys are
/// allocated lazily, so the entries cost no memory until read.
fn map_series(lens: &[i64], validity: Bitmap) -> Series {
    let n = lens.iter().sum::<i64>() as usize;
    let entries = StructArray::new(
        map_dtype()
            .map_entries_dtype()
            .unwrap()
            .to_physical()
            .to_arrow(CompatLevel::newest()),
        n,
        vec![
            BooleanArray::new(ArrowDataType::Boolean, Bitmap::new_zeroed(n), None).boxed(),
            NullArray::new(ArrowDataType::Null, n).boxed(),
        ],
        None,
    );
    let offsets = Offsets::try_from_lengths(lens.iter().map(|&l| l as usize)).unwrap();
    let chunk = ListArray::<i64>::new(
        ListArray::<i64>::default_datatype(entries.dtype().clone()),
        offsets.into(),
        entries.boxed(),
        Some(validity),
    );
    unsafe {
        Series::from_chunks_and_dtype_unchecked("m".into(), vec![chunk.boxed()], &map_dtype())
    }
}

/// One test, as the allocations of tests running in parallel would mix.
#[test]
fn exporting_null_map_rows_allocates_per_row_not_per_entry() {
    const LIMIT: usize = 16 << 20;
    let half = 1i64 << 30;
    let cases = [
        // A null row too large for one arrow MAP, exported as a piece of its own.
        (vec![1, i32::MAX as i64 + 1], vec![true, false], vec![1, 0]),
        // A null row whose entries end the chunk, which keeps one run of entries.
        (vec![1, half], vec![true, false], vec![1, 0]),
        // A null row between live rows, which keeps two runs of entries.
        (vec![1, half, 1], vec![true, false, true], vec![1, 0, 1]),
    ];

    for (lens, validity, expected) in cases {
        let s = map_series(&lens, Bitmap::from(validity.as_slice()));

        let (arrays, peak) = peak_allocation(|| {
            s.try_to_arrow_split(0, CompatLevel::newest())
                .collect::<PolarsResult<Vec<_>>>()
                .unwrap()
        });

        assert!(peak < LIMIT, "{lens:?}: exporting allocated {peak} bytes");
        let entries: Vec<usize> = arrays
            .iter()
            .flat_map(|a| {
                let m: &MapArray = a.as_any().downcast_ref().unwrap();
                m.offsets().lengths().collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(entries, expected, "{lens:?}");
    }
}
