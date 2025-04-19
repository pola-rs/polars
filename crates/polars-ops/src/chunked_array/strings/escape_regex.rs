use polars_core::prelude::{StringChunked, StringChunkedBuilder};

#[inline]
pub fn escape_regex_str(s: &str) -> String {
    regex_syntax::escape(s)
}

pub fn escape_regex(ca: &StringChunked) -> StringChunked {
    let mut buffer = String::new();
    let mut builder = StringChunkedBuilder::new(ca.name().clone(), ca.len());
    for opt_s in ca.iter() {
        if let Some(s) = opt_s {
            buffer.clear();
            regex_syntax::escape_into(s, &mut buffer);
            builder.append_value(&buffer);
        } else {
            builder.append_null();
        }
    }
    builder.finish()
}
