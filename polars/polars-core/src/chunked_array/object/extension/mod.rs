pub(crate) mod drop;
pub(crate) mod extension;

use crate::{prelude::*, PROCESS_ID};
use arrow::array::{Array, FixedSizeBinaryArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::buffer::{Buffer, MutableBuffer};
use std::{
    alloc::{dealloc, Layout},
    mem,
};
use std::mem::ManuallyDrop;
use extension::PolarsExtension;

/// deallocate a vec, without calling T::drop
fn dealoc_vec_no_drop<T: Sized>(v: Vec<T>) {
    let size = mem::size_of::<T>() * v.capacity();
    let align = mem::align_of::<T>();
    let layout = unsafe { Layout::from_size_align_unchecked(size, align) };
    let ptr = v.as_ptr() as *const u8 as *mut u8;
    unsafe { dealloc(ptr, layout) }
    mem::forget(v);
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

/// Create an extension Array that can be sent to arrow and (once wrapped in `[PolarsExtension]` will
/// also call drop on `T`, when the array is dropped.
pub(crate) fn create_extension<
    I: IntoIterator<Item = Option<T>> + TrustedLen,
    T: Sized + Default,
>(
    iter: I,
) -> PolarsExtension {
    let t_size = std::mem::size_of::<T>();
    let t_alignment = std::mem::align_of::<T>();
    let n_t_vals = iter.size_hint().1.unwrap();

    let mut buf = MutableBuffer::with_capacity(n_t_vals * t_size);
    let mut validity = MutableBitmap::with_capacity(n_t_vals);

    // when we transmute from &[u8] to T, T must be aligned correctly,
    // so we pad with bytes until the alignment matches
    let n_padding = (buf.as_ptr() as usize) % t_alignment;
    buf.extend_constant(n_padding, 0);

    // transmute T as bytes and copy in buffer
    for opt_t in iter.into_iter() {
        match opt_t {
            Some(t) => {
                unsafe {
                    buf.extend_from_slice(any_as_u8_slice(&t));
                    // Safety: we allocated upfront
                    validity.push_unchecked(true)
                }
                mem::forget(t);
            }
            None => {
                unsafe {
                    buf.extend_from_slice(any_as_u8_slice(&T::default()));
                    // Safety: we allocated upfront
                    validity.push_unchecked(false)
                }
            }
        }
    }

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

    let physical_type = ArrowDataType::FixedSizeBinary(t_size);
    let extension_type = ArrowDataType::Extension(
        "POLARS_EXTENSION_TYPE".into(),
        physical_type.into(),
        Some(metadata),
    );
    let validity = if validity.null_count() > 0 {
        Some(validity.into())
    } else {
        None
    };

    let array = FixedSizeBinaryArray::from_data(extension_type, buf, validity);

    PolarsExtension::new(array)
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fmt::{Display, Formatter};

    #[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
    struct Foo {
        pub a: i32,
        pub b: u8,
        pub other_heap: String,
    }

    impl Display for Foo {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl PolarsObject for Foo {
        fn type_name() -> &'static str {
            "object"
        }
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

        let vals = vec![Some(foo), Some(foo2)];
        create_extension(vals.into_iter());
    }

    #[test]
    fn test_extension_to_list() {
        let foo1 = Foo {
            a: 1,
            b: 1,
            other_heap: "foo".into(),
        };
        let foo2 = Foo {
            a: 1,
            b: 1,
            other_heap: "bar".into(),
        };

        let values = &[Some(foo1), None, Some(foo2)];
        let ca = ObjectChunked::new_from_opt_slice("", values);

        let groups = vec![
            (0u32, vec![0u32, 1]),
            (2, vec![2])
        ];
        let out = ca.agg_list(&groups);

        dbg!(out);


    }
}
