pub(crate) fn verbose() -> bool {
    polars_config::config().verbose()
}
