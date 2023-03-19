use polars_core::prelude::Utf8Chunked;

pub(super) fn to_lowercase<'a>(ca: &'a Utf8Chunked) -> Utf8Chunked {
    // amortize allocation
    let mut buf = String::new();
    let f = |s: &'a str| {
        buf.clear();
        buf.push_str(s);
        buf.make_ascii_lowercase();
        // extend lifetime
        // lifetime is bound to 'a
        let slice = buf.as_str();
        unsafe { std::mem::transmute::<&str, &'a str>(slice) }
    };
    ca.apply_mut(f)
}

pub(super) fn to_uppercase<'a>(ca: &'a Utf8Chunked) -> Utf8Chunked {
    // amortize allocation
    let mut buf = String::new();
    let f = |s: &'a str| {
        buf.clear();
        buf.push_str(s);
        buf.make_ascii_uppercase();
        // extend lifetime
        // lifetime is bound to 'a
        let slice = buf.as_str();
        unsafe { std::mem::transmute::<&str, &'a str>(slice) }
    };
    ca.apply_mut(f)
}
