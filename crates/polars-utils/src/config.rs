pub(crate) fn verbose() -> bool {
    std::env::var("POLARS_VERBOSE").as_deref().unwrap_or("") == "1"
}
