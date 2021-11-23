use crate::chunked_array::object::extension::PolarsExtension;
use crate::prelude::*;

/// This will dereference a raw ptr when dropping the PolarsExtension, make sure that it's valid.
pub(crate) unsafe fn drop_list(ca: &ListChunked) {
    let mut inner = ca.inner_dtype();
    let mut nested_count = 0;

    while let Some(a) = inner.inner_dtype() {
        nested_count += 1;
        inner = a.clone()
    }

    if matches!(inner, DataType::Object(_)) {
        if nested_count != 0 {
            panic!("multiple nested objects not yet supported")
        }
        // if empty the memory is leaked somewhere
        assert!(!ca.chunks.is_empty());
        for lst_arr in &ca.chunks {
            // This list can be cloned, so we check the ref count before we drop
            if let (ArrowDataType::LargeList(fld), 1) =
                (lst_arr.data_type(), Arc::strong_count(lst_arr))
            {
                let dtype = fld.data_type();

                assert!(matches!(dtype, ArrowDataType::Extension(_, _, _)));

                // recreate the polars extension so that the content is dropped
                let arr = lst_arr.as_any().downcast_ref::<LargeListArray>().unwrap();

                let values = arr.values();

                // The inner value also may be cloned, check the ref count
                if Arc::strong_count(values) == 1 {
                    let arr = values
                        .as_any()
                        .downcast_ref::<FixedSizeBinaryArray>()
                        .unwrap()
                        .clone();
                    PolarsExtension::new(arr);
                }
            }
        }
    }
}
