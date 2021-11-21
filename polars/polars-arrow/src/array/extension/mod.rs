use arrow::array::{Array, FixedSizeBinaryArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::buffer::{Buffer, MutableBuffer};
use arrow::datatypes::DataType;
use lazy_static::lazy_static;
use std::fmt::Debug;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{
    alloc::{dealloc, Layout},
    mem,
};

/// deallocate a vec, without calling T::drop
fn dealoc_vec_no_drop<T: Sized>(v: Vec<T>) {
    let size = mem::size_of::<T>() * v.capacity();
    let align = mem::align_of::<T>();
    let layout = unsafe { Layout::from_size_align_unchecked(size, align) };
    let ptr = v.as_ptr() as *const u8 as *mut u8;
    unsafe { dealloc(ptr, layout) }
    mem::forget(v);
}

lazy_static! {
    pub static ref PROCESS_ID: u128 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
}

pub struct PolarsExtension {
    array: FixedSizeBinaryArray,
}

impl Drop for PolarsExtension {
    fn drop(&mut self) {
        if let DataType::Extension(_, _, metadata) = self.array.data_type() {
            let metadata = metadata.as_ref().expect("should have metadata");
            let mut iter = metadata.split(';');

            let pid = iter.next().unwrap().parse::<u128>().unwrap();
            let ptr = iter.next().unwrap().parse::<usize>().unwrap();
            if pid == *PROCESS_ID {
                // implicitly drop by taking ownership
                let _et = unsafe {
                    Box::from_raw(ptr as *const ExtensionSentinel as *mut ExtensionSentinel)
                };
            }
        }
    }
}

/// Invariants
/// `ptr` must point to start a `T` allocation
/// `n_t_vals` must reprecent the correct number of `T` values in that allocation
unsafe fn create_drop<T: Sized>(mut ptr: *const u8, n_t_vals: usize) -> Box<dyn FnMut()> {
    Box::new(move || {
        let t_size = std::mem::size_of::<T>() as isize;
        for _ in 0..n_t_vals {
            let _ = std::ptr::read_unaligned(ptr as *const T);
            ptr = ptr.offset(t_size as isize)
        }
    })
}

struct ExtensionSentinel {
    drop_fn: Option<Box<dyn FnMut()>>,
}

impl Drop for ExtensionSentinel {
    fn drop(&mut self) {
        let mut drop_fn = self.drop_fn.take().unwrap();
        drop_fn()
    }
}

// https://stackoverflow.com/questions/28127165/how-to-convert-struct-to-u8d
// not entirely sure if padding bytes in T are intialized or not.
unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    std::slice::from_raw_parts((p as *const T) as *const u8, std::mem::size_of::<T>())
}

fn create_extension<T: Sized + Default>(vals: Vec<Option<T>>) -> PolarsExtension {
    let t_size = std::mem::size_of::<T>();
    let t_alignment = std::mem::align_of::<T>();
    let n_t_vals = vals.len();

    let mut buf = MutableBuffer::with_capacity(vals.len() * t_size);

    // when we transmute from &[u8] to T, T must be aligned correctly,
    // so we pad with bytes until the alignment matches
    let n_padding = (buf.as_ptr() as usize) % t_alignment;
    buf.extend_constant(n_padding, 0);

    let mut validity = MutableBitmap::with_capacity(vals.len());

    // transmute T as bytes and copy in buffer
    for opt_t in vals.iter() {
        match opt_t {
            Some(t) => {
                unsafe {
                    buf.extend_from_slice(any_as_u8_slice(t));
                    // Safety:
                    // validity size is pre allocated
                    validity.push_unchecked(true);
                }
            }
            None => {
                // use default, because we need to call drop on every value
                // we could not skip this if null, because the validity bitmaps
                // may be changed.
                // so if we extended with 0u8..0u8 we could be transmuting that into T
                // if bitmap was changed
                unsafe {
                    buf.extend_from_slice(any_as_u8_slice(&T::default()));
                    // Safety:
                    // validity size is pre allocated
                    validity.push_unchecked(false)
                };
            }
        }
    }
    dealoc_vec_no_drop(vals);

    let validity = if validity.null_count() > 0 {
        Some(validity.into())
    } else {
        None
    };

    // we slice the buffer because we want to ignore the padding bytes from here
    // they can be forgotten
    let buf: Buffer<u8> = buf.into();
    let len = buf.len() - n_padding;
    let buf = buf.slice(n_padding, len);

    // ptr to start of T, not to start of padding
    let ptr = buf.as_slice().as_ptr();

    // Safety:
    // ptr and t are correct
    let drop_fn = unsafe { create_drop::<T>(ptr, n_t_vals) };
    let et = Box::new(ExtensionSentinel {
        drop_fn: Some(drop_fn),
    });
    let et_ptr = &*et as *const ExtensionSentinel;
    std::mem::forget(et);

    let metadata = format!("{};{}", *PROCESS_ID, et_ptr as usize);

    let physical_type = DataType::FixedSizeBinary(t_size);
    let extension_type = DataType::Extension(
        "POLARS_EXTENSION_TYPE".into(),
        physical_type.into(),
        Some(metadata),
    );

    let array = FixedSizeBinaryArray::from_data(extension_type, buf, validity);

    PolarsExtension { array }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Clone, Debug, Default)]
    struct Foo {
        pub a: i32,
        pub b: u8,
        pub other_heap: String,
    }

    #[test]
    fn test_create_extension() {
        // Run this under MIRI.
        let foo = Foo {
            a: 1,
            b: 1,
            other_heap: "foo".into(),
        };
        let foo2 = Foo {
            a: 1,
            b: 1,
            other_heap: "bar".into(),
        };

        let vals = vec![None, Some(foo), Some(foo2), None];
        create_extension(vals);
    }
}
