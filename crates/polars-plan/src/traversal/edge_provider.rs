use polars_utils::array::array_split;
use polars_utils::collection::{Collection, CollectionWrap};

pub trait NodeEdgesProvider<Edge> {
    fn unpacker<'a>(&'a mut self) -> EdgesUnpacker<'a, Edge>
    where
        Edge: 'a;

    fn inputs<'a>(&'a mut self) -> CollectionWrap<Edge, &'a mut dyn Collection<Edge>>
    where
        Edge: 'a;

    fn outputs<'a>(&'a mut self) -> CollectionWrap<Edge, &'a mut dyn Collection<Edge>>
    where
        Edge: 'a;
}

pub struct EdgesUnpacker<'a, Edge> {
    /// `[..inputs, ..outputs]`
    combined: DynLenArray<&'a mut Edge>,
}

impl<'a, Edge> EdgesUnpacker<'a, Edge> {
    pub fn new(combined: DynLenArray<&'a mut Edge>) -> Self {
        Self { combined }
    }

    pub fn unpack<
        const NUM_INPUTS: usize,
        const NUM_OUTPUTS: usize,
        // Workaround for generic_const_exprs, have the caller pass in `NUM_INPUTS + NUM_OUTPUTS`
        const TOTAL_EDGES: usize,
    >(
        self,
    ) -> Option<([&'a mut Edge; NUM_INPUTS], [&'a mut Edge; NUM_OUTPUTS])> {
        const {
            assert!(NUM_INPUTS + NUM_OUTPUTS == TOTAL_EDGES);
        }

        let combined: [&mut Edge; TOTAL_EDGES] = self.combined.unpack()?;

        Some(array_split(combined))
    }
}

impl<'a, Edge> From<DynLenArray<&'a mut Edge>> for EdgesUnpacker<'a, Edge> {
    fn from(value: DynLenArray<&'a mut Edge>) -> Self {
        Self::new(value)
    }
}

/// TODO: Move to polars-utils
pub enum DynLenArray<T> {
    _0([T; 0]),
    _1([T; 1]),
    _2([T; 2]),
    _3([T; 3]),
    _4([T; 4]),
    _5([T; 5]),
    _6([T; 6]),
    _7([T; 7]),
    _8([T; 8]),
    _9([T; 9]),
}

impl<T> DynLenArray<T> {
    /// # Returns
    /// Returns None if the iterator produces more than 9 values.
    pub fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Option<Self> {
        use std::mem::MaybeUninit;

        let mut data: [MaybeUninit<T>; 9] = [const { MaybeUninit::uninit() }; _];

        let mut iter = iter.into_iter();
        let mut i: usize = 0;

        while let Some(value) = iter.next() {
            data.get_mut(i)?.write(value);
            i += 1;
        }

        const fn truncate<const IN_LEN: usize, const OUT_LEN: usize, T>(
            src: [MaybeUninit<T>; IN_LEN],
        ) -> [T; OUT_LEN] {
            use std::mem::{ManuallyDrop, transmute_copy};

            assert!(OUT_LEN <= IN_LEN);
            assert!(
                align_of::<ManuallyDrop<[MaybeUninit<T>; IN_LEN]>>() == align_of::<[T; OUT_LEN]>()
            );

            unsafe { transmute_copy(&ManuallyDrop::new(src)) }
        }

        use DynLenArray::*;

        Some(match i {
            0 => _0(truncate(data)),
            1 => _1(truncate(data)),
            2 => _2(truncate(data)),
            3 => _3(truncate(data)),
            4 => _4(truncate(data)),
            5 => _5(truncate(data)),
            6 => _6(truncate(data)),
            7 => _7(truncate(data)),
            8 => _8(truncate(data)),
            9 => _9(truncate(data)),
            _ => return None,
        })
    }

    pub fn unpack<const N: usize>(self) -> Option<[T; N]> {
        use DynLenArray::*;

        const fn unwrap<const N: usize, const M: usize, T>(src: [T; N]) -> [T; M] {
            use std::mem::{ManuallyDrop, transmute_copy};

            assert!(N == M);

            unsafe { transmute_copy(&ManuallyDrop::new(src)) }
        }

        Some(match (N, self) {
            (0, _0(v)) => unwrap(v),
            (1, _1(v)) => unwrap(v),
            (2, _2(v)) => unwrap(v),
            (3, _3(v)) => unwrap(v),
            (4, _4(v)) => unwrap(v),
            (5, _5(v)) => unwrap(v),
            (6, _6(v)) => unwrap(v),
            (7, _7(v)) => unwrap(v),
            (8, _8(v)) => unwrap(v),
            (9, _9(v)) => unwrap(v),
            _ => {
                const {
                    assert!(N <= 9);
                }

                return None;
            },
        })
    }

    pub fn map<U, F: FnMut(T) -> U>(self, f: F) -> DynLenArray<U> {
        use DynLenArray::*;

        match self {
            _0(v) => _0(v.map(f)),
            _1(v) => _1(v.map(f)),
            _2(v) => _2(v.map(f)),
            _3(v) => _3(v.map(f)),
            _4(v) => _4(v.map(f)),
            _5(v) => _5(v.map(f)),
            _6(v) => _6(v.map(f)),
            _7(v) => _7(v.map(f)),
            _8(v) => _8(v.map(f)),
            _9(v) => _9(v.map(f)),
        }
    }
}

#[macro_export]
macro_rules! apply_dyn_len_array {
    ($dyn_len_arr:expr, | $v:ident | $body:block) => {{
        use DynLenArray::*;

        match $dyn_len_arr {
            _0(v) => _0({
                let $v = v;
                $body
            }),
            _1(v) => _1({
                let $v = v;
                $body
            }),
            _2(v) => _2({
                let $v = v;
                $body
            }),
            _3(v) => _3({
                let $v = v;
                $body
            }),
            _4(v) => _4({
                let $v = v;
                $body
            }),
            _5(v) => _5({
                let $v = v;
                $body
            }),
            _6(v) => _6({
                let $v = v;
                $body
            }),
            _7(v) => _7({
                let $v = v;
                $body
            }),
            _8(v) => _8({
                let $v = v;
                $body
            }),
            _9(v) => _9({
                let $v = v;
                $body
            }),
        }
    }};
}
