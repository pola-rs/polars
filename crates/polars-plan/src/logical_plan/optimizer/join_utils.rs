pub(super) fn split_suffix<'a>(name: &'a str, suffix: &str) -> &'a str {
    let (original, _) = name.split_at(name.len() - suffix.len());
    original
}
