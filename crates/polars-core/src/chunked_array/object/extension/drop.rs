use crate::chunked_array::object::extension::PolarsExtension;
use crate::prelude::*;

/// This will dereference a raw ptr when dropping the [`PolarsExtension`], make sure that it's valid.
pub(crate) unsafe fn drop_list(ca: &ListChunked) {
    let mut inner = ca.inner_dtype();
    let mut nested_count = 0;

    while let Some(a) = inner.inner_dtype() {
        nested_count += 1;
        inner = a;
    }

    if matches!(inner, DataType::Object(_, _)) {
        if nested_count != 0 {
            panic!("multiple nested objects not yet supported")
        }
        // if empty the memory is leaked somewhere
        assert!(!ca.chunks.is_empty());
        for lst_arr in &ca.chunks {
            if let ArrowDataType::LargeList(fld) = lst_arr.dtype() {
                let dtype = fld.dtype();

                assert!(matches!(dtype, ArrowDataType::Extension(_)));

                // recreate the polars extension so that the content is dropped
                let arr = lst_arr.as_any().downcast_ref::<LargeListArray>().unwrap();

                let values = arr.values();
                drop_object_array(values.as_ref())
            }
        }
    }
}

pub(crate) unsafe fn drop_object_array(values: &dyn Array) {
    let arr = values
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .unwrap();

    // If the buf is not shared with anyone but us we can deallocate.
    let buf = arr.values();
    if buf.storage_refcount() == 1 && !buf.is_empty() {
        PolarsExtension::new(arr.clone());
    };
}
