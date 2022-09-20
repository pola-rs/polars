use super::*;

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ListFunction {
    Concat,
}

impl Display for ListFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use self::*;
        match self {
            ListFunction::Concat => write!(f, "concat"),
        }
    }
}

#[cfg(feature = "is_in")]
pub(super) fn contains(args: &mut [Series]) -> PolarsResult<Series> {
    let list = &args[0];
    let is_in = &args[1];

    is_in.is_in(list).map(|mut ca| {
        ca.rename(list.name());
        ca.into_series()
    })
}

pub(super) fn concat(s: &mut [Series]) -> PolarsResult<Series> {
    let mut first = std::mem::take(&mut s[0]);
    let other = &s[1..];

    let first_ca = match first.list().ok() {
        Some(ca) => ca,
        None => {
            first = first.reshape(&[-1, 1]).unwrap();
            first.list().unwrap()
        }
    };
    first_ca.lst_concat(other).map(|ca| ca.into_series())
}
