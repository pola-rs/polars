pub(crate) mod drop;
mod list;
pub(crate) mod polars_extension;

use std::mem;

use arrow::array::{Array, FixedSizeBinaryArray};
use arrow::bitmap::MutableBitmap;
use arrow::buffer::Buffer;
use polars_extension::PolarsExtension;

use crate::prelude::*;
use crate::PROCESS_ID;

pub const EXTENSION_NAME: &str = "POLARS_EXTENSION_TYPE";

/// Invariants
/// `ptr` must point to start a `T` allocation
/// `n_t_vals` must represent the correct number of `T` values in that allocation
unsafe fn create_drop<T: Sized>(mut ptr: *const u8, n_t_vals: usize) -> Box<dyn FnMut()> {
    Box::new(move || {
        let t_size = std::mem::size_of::<T>() as isize;
        for _ in 0..n_t_vals {
            let _ = std::ptr::read_unaligned(ptr as *const T);
            ptr = ptr.offset(t_size)
        }
    })
}

#[allow(clippy::type_complexity)]
struct ExtensionSentinel {
    drop_fn: Option<Box<dyn FnMut()>>,
    // A function on the heap that take a `array: FixedSizeBinary` and a `name: &str`
    // and returns a `Series` of `ObjectChunked<T>`
    pub(crate) to_series_fn: Option<Box<dyn Fn(&FixedSizeBinaryArray, &str) -> Series>>,
}

impl Drop for ExtensionSentinel {
    fn drop(&mut self) {
        let mut drop_fn = self.drop_fn.take().unwrap();
        drop_fn()
    }
}

// https://stackoverflow.com/questions/28127165/how-to-convert-struct-to-u8d
// not entirely sure if padding bytes in T are initialized or not.
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
    let env = "POLARS_ALLOW_EXTENSION";
    std::env::var(env).unwrap_or_else(|_| {
        panic!("env var: {env} must be set to allow extension types to be created",)
    });
    let t_size = std::mem::size_of::<T>();
    let t_alignment = std::mem::align_of::<T>();
    let n_t_vals = iter.size_hint().1.unwrap();

    let mut buf = Vec::with_capacity(n_t_vals * t_size);
    let mut validity = MutableBitmap::with_capacity(n_t_vals);

    // when we transmute from &[u8] to T, T must be aligned correctly,
    // so we pad with bytes until the alignment matches
    let n_padding = (buf.as_ptr() as usize) % t_alignment;
    buf.extend(std::iter::repeat(0).take(n_padding));

    let mut null_count = 0 as IdxSize;
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
                null_count += 1;
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
    let buf = buf.sliced(n_padding, len);

    // ptr to start of T, not to start of padding
    let ptr = buf.as_slice().as_ptr();

    // Safety:
    // ptr and t are correct
    let drop_fn = unsafe { create_drop::<T>(ptr, n_t_vals) };
    let et = Box::new(ExtensionSentinel {
        drop_fn: Some(drop_fn),
        to_series_fn: None,
    });
    let et_ptr = &*et as *const ExtensionSentinel;
    std::mem::forget(et);

    let metadata = format!("{};{}", *PROCESS_ID, et_ptr as usize);

    let physical_type = ArrowDataType::FixedSizeBinary(t_size);
    let extension_type =
        ArrowDataType::Extension(EXTENSION_NAME.into(), physical_type.into(), Some(metadata));
    // first freeze, otherwise we compute null
    let validity = if null_count > 0 {
        Some(validity.into())
    } else {
        None
    };

    let array = FixedSizeBinaryArray::new(extension_type, buf, validity);

    // Safety:
    // we just heap allocated the ExtensionSentinel, so its alive.
    unsafe { PolarsExtension::new(array) }
}

#[cfg(test)]
mod test {
    use std::fmt::{Display, Formatter};

    use super::*;

    #[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
    struct Foo {
        pub a: i32,
        pub b: u8,
        pub other_heap: String,
    }

    impl Display for Foo {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self)
        }
    }

    impl PolarsObject for Foo {
        fn type_name() -> &'static str {
            "object"
        }
    }

    #[test]
    fn test_create_extension() {
        std::env::set_var("POLARS_ALLOW_EXTENSION", "1");
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
        std::env::set_var("POLARS_ALLOW_EXTENSION", "1");
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

        let values = &[Some(foo1), None, Some(foo2), None];
        let ca = ObjectChunked::new("", values);

        let groups = GroupsProxy::Idx(vec![(0, vec![0, 1]), (2, vec![2]), (3, vec![3])].into());
        let out = unsafe { ca.agg_list(&groups) };
        assert!(matches!(out.dtype(), DataType::List(_)));
        assert_eq!(out.len(), groups.len());
    }

    #[test]
    fn test_extension_to_list_explode() {
        std::env::set_var("POLARS_ALLOW_EXTENSION", "1");
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

        let values = &[Some(foo1.clone()), None, Some(foo2.clone()), None];
        let ca = ObjectChunked::new("", values);

        let groups = vec![(0, vec![0, 1]), (2, vec![2]), (3, vec![3])].into();
        let out = unsafe { ca.agg_list(&GroupsProxy::Idx(groups)) };
        let a = out.explode().unwrap();

        let ca_foo = a.as_any().downcast_ref::<ObjectChunked<Foo>>().unwrap();
        assert_eq!(ca_foo.get(0).unwrap(), &foo1);
        assert_eq!(ca_foo.get(1), None);
        assert_eq!(ca_foo.get(2).unwrap(), &foo2);
        assert_eq!(ca_foo.get(3), None);
    }
}
