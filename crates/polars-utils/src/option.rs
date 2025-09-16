pub trait OptionTry<T>: Sized {
    fn try_map<U, E>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<Option<U>, E>;
}

impl<T> OptionTry<T> for Option<T> {
    fn try_map<U, E>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<Option<U>, E> {
        match self {
            None => Ok(None),
            Some(v) => f(v).map(Some),
        }
    }
}
