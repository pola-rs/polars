use crate::{bitmap::Bitmap, offset::Offset};

use super::super::pages::{ListNested, Nested};
use super::rep::num_values;
use super::to_length;

trait DebugIter: Iterator<Item = (u32, usize)> + std::fmt::Debug {}

impl<A: Iterator<Item = (u32, usize)> + std::fmt::Debug> DebugIter for A {}

fn single_iter<'a>(
    validity: &'a Option<Bitmap>,
    is_optional: bool,
    length: usize,
) -> Box<dyn DebugIter + 'a> {
    match (is_optional, validity) {
        (false, _) => {
            Box::new(std::iter::repeat((0u32, 1usize)).take(length)) as Box<dyn DebugIter>
        }
        (true, None) => {
            Box::new(std::iter::repeat((1u32, 1usize)).take(length)) as Box<dyn DebugIter>
        }
        (true, Some(validity)) => {
            Box::new(validity.iter().map(|v| (v as u32, 1usize)).take(length)) as Box<dyn DebugIter>
        }
    }
}

fn single_list_iter<'a, O: Offset>(nested: &'a ListNested<O>) -> Box<dyn DebugIter + 'a> {
    match (nested.is_optional, &nested.validity) {
        (false, _) => Box::new(
            std::iter::repeat(0u32)
                .zip(to_length(&nested.offsets))
                .map(|(a, b)| (a + (b != 0) as u32, b)),
        ) as Box<dyn DebugIter>,
        (true, None) => Box::new(
            std::iter::repeat(1u32)
                .zip(to_length(&nested.offsets))
                .map(|(a, b)| (a + (b != 0) as u32, b)),
        ) as Box<dyn DebugIter>,
        (true, Some(validity)) => Box::new(
            validity
                .iter()
                .map(|x| (x as u32))
                .zip(to_length(&nested.offsets))
                .map(|(a, b)| (a + (b != 0) as u32, b)),
        ) as Box<dyn DebugIter>,
    }
}

fn iter<'a>(nested: &'a [Nested]) -> Vec<Box<dyn DebugIter + 'a>> {
    nested
        .iter()
        .map(|nested| match nested {
            Nested::Primitive(validity, is_optional, length) => {
                single_iter(validity, *is_optional, *length)
            }
            Nested::List(nested) => single_list_iter(nested),
            Nested::LargeList(nested) => single_list_iter(nested),
            Nested::Struct(validity, is_optional, length) => {
                single_iter(validity, *is_optional, *length)
            }
        })
        .collect()
}

/// Iterator adapter of parquet / dremel definition levels
#[derive(Debug)]
pub struct DefLevelsIter<'a> {
    // iterators of validities and lengths. E.g. [[[None,b,c], None], None] -> [[(true, 2), (false, 0)], [(true, 3), (false, 0)], [(false, 1), (true, 1), (true, 1)]]
    iter: Vec<Box<dyn DebugIter + 'a>>,
    // vector containing the remaining number of values of each iterator.
    // e.g. the iters [[2, 2], [3, 4, 1, 2]] after the first iteration will return [2, 3],
    // and remaining will be [2, 3].
    // on the second iteration, it will be `[2, 2]` (since iterations consume the last items)
    remaining: Vec<usize>, /* < remaining.len() == iter.len() */
    validity: Vec<u32>,
    // cache of the first `remaining` that is non-zero. Examples:
    // * `remaining = [2, 2] => current_level = 2`
    // * `remaining = [2, 0] => current_level = 1`
    // * `remaining = [0, 0] => current_level = 0`
    current_level: usize, /* < iter.len() */
    // the total definition level at any given point during the iteration
    total: u32, /* < iter.len() */
    // the total number of items that this iterator will return
    remaining_values: usize,
}

impl<'a> DefLevelsIter<'a> {
    pub fn new(nested: &'a [Nested]) -> Self {
        let remaining_values = num_values(nested);

        let iter = iter(nested);
        let remaining = vec![0; iter.len()];
        let validity = vec![0; iter.len()];

        Self {
            iter,
            remaining,
            validity,
            total: 0,
            current_level: 0,
            remaining_values,
        }
    }
}

impl<'a> Iterator for DefLevelsIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_values == 0 {
            return None;
        }

        if self.remaining.is_empty() {
            self.remaining_values -= 1;
            return Some(0);
        }

        let mut empty_contrib = 0u32;
        for ((iter, remaining), validity) in self
            .iter
            .iter_mut()
            .zip(self.remaining.iter_mut())
            .zip(self.validity.iter_mut())
            .skip(self.current_level)
        {
            let (is_valid, length): (u32, usize) = iter.next()?;
            *validity = is_valid;
            self.total += is_valid;

            *remaining = length;
            if length == 0 {
                *validity = 0;
                self.total -= is_valid;
                empty_contrib = is_valid;
                break;
            }
            self.current_level += 1;
        }

        // track
        if let Some(x) = self.remaining.get_mut(self.current_level.saturating_sub(1)) {
            *x = x.saturating_sub(1)
        }

        let r = Some(self.total + empty_contrib);

        for index in (1..self.current_level).rev() {
            if self.remaining[index] == 0 {
                self.current_level -= 1;
                self.remaining[index - 1] -= 1;
                self.total -= self.validity[index];
            }
        }
        if self.remaining[0] == 0 {
            self.current_level = self.current_level.saturating_sub(1);
            self.total -= self.validity[0];
        }
        self.remaining_values -= 1;
        r
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let length = self.remaining_values;
        (length, Some(length))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test(nested: Vec<Nested>, expected: Vec<u32>) {
        let mut iter = DefLevelsIter::new(&nested);
        assert_eq!(iter.size_hint().0, expected.len());
        let result = iter.by_ref().collect::<Vec<_>>();
        assert_eq!(result, expected);
        assert_eq!(iter.size_hint().0, 0);
    }

    #[test]
    fn struct_optional() {
        let b = [
            true, false, true, true, false, true, false, false, true, true,
        ];
        let nested = vec![
            Nested::Struct(None, true, 10),
            Nested::Primitive(Some(b.into()), true, 10),
        ];
        let expected = vec![2, 1, 2, 2, 1, 2, 1, 1, 2, 2];

        test(nested, expected)
    }

    #[test]
    fn nested_edge_simple() {
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 2].try_into().unwrap(),
                validity: None,
            }),
            Nested::Primitive(None, true, 2),
        ];
        let expected = vec![3, 3];

        test(nested, expected)
    }

    #[test]
    fn struct_optional_1() {
        let b = [
            true, false, true, true, false, true, false, false, true, true,
        ];
        let nested = vec![
            Nested::Struct(None, true, 10),
            Nested::Primitive(Some(b.into()), true, 10),
        ];
        let expected = vec![2, 1, 2, 2, 1, 2, 1, 1, 2, 2];

        test(nested, expected)
    }

    #[test]
    fn struct_optional_optional() {
        let nested = vec![
            Nested::Struct(None, true, 10),
            Nested::Primitive(None, true, 10),
        ];
        let expected = vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2];

        test(nested, expected)
    }

    #[test]
    fn l1_required_required() {
        let nested = vec![
            // [[0, 1], [], [2, 0, 3], [4, 5, 6], [], [7, 8, 9], [], [10]]
            Nested::List(ListNested {
                is_optional: false,
                offsets: vec![0, 2, 2, 5, 8, 8, 11, 11, 12].try_into().unwrap(),
                validity: None,
            }),
            Nested::Primitive(None, false, 12),
        ];
        let expected = vec![1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1];

        test(nested, expected)
    }

    #[test]
    fn l1_optional_optional() {
        // [[0, 1], None, [2, None, 3], [4, 5, 6], [], [7, 8, 9], None, [10]]

        let v0 = [true, false, true, true, true, true, false, true];
        let v1 = [
            true, true, //[0, 1]
            true, false, true, //[2, None, 3]
            true, true, true, //[4, 5, 6]
            true, true, true, //[7, 8, 9]
            true, //[10]
        ];
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 2, 2, 5, 8, 8, 11, 11, 12].try_into().unwrap(),
                validity: Some(v0.into()),
            }),
            Nested::Primitive(Some(v1.into()), true, 12),
        ];
        let expected = vec![3u32, 3, 0, 3, 2, 3, 3, 3, 3, 1, 3, 3, 3, 0, 3];

        test(nested, expected)
    }

    #[test]
    fn l2_required_required_required() {
        /*
        [
            [
                [1,2,3],
                [4,5,6,7],
            ],
            [
                [8],
                [9, 10]
            ]
        ]
        */
        let nested = vec![
            Nested::List(ListNested {
                is_optional: false,
                offsets: vec![0, 2, 4].try_into().unwrap(),
                validity: None,
            }),
            Nested::List(ListNested {
                is_optional: false,
                offsets: vec![0, 3, 7, 8, 10].try_into().unwrap(),
                validity: None,
            }),
            Nested::Primitive(None, false, 10),
        ];
        let expected = vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2];

        test(nested, expected)
    }

    #[test]
    fn l2_optional_required_required() {
        let a = [true, false, true, true];
        /*
        [
            [
                [1,2,3],
                [4,5,6,7],
            ],
            None,
            [
                [8],
                [],
                [9, 10]
            ]
        ]
        */
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 2, 2, 2, 5].try_into().unwrap(),
                validity: Some(a.into()),
            }),
            Nested::List(ListNested {
                is_optional: false,
                offsets: vec![0, 3, 7, 8, 8, 10].try_into().unwrap(),
                validity: None,
            }),
            Nested::Primitive(None, false, 10),
        ];
        let expected = vec![3, 3, 3, 3, 3, 3, 3, 0, 1, 3, 2, 3, 3];

        test(nested, expected)
    }

    #[test]
    fn l2_optional_optional_required() {
        let a = [true, false, true];
        let b = [true, true, true, true, false];
        /*
        [
            [
                [1,2,3],
                [4,5,6,7],
            ],
            None,
            [
                [8],
                [],
                None,
            ],
        ]
        */
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 2, 2, 5].try_into().unwrap(),
                validity: Some(a.into()),
            }),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 3, 7, 8, 8, 8].try_into().unwrap(),
                validity: Some(b.into()),
            }),
            Nested::Primitive(None, false, 8),
        ];
        let expected = vec![4, 4, 4, 4, 4, 4, 4, 0, 4, 3, 2];

        test(nested, expected)
    }

    #[test]
    fn l2_optional_optional_optional() {
        let a = [true, false, true];
        let b = [true, true, true, false];
        let c = [true, true, true, true, false, true, true, true];
        /*
        [
            [
                [1,2,3],
                [4,None,6,7],
            ],
            None,
            [
                [8],
                None,
            ],
        ]
        */
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 2, 2, 4].try_into().unwrap(),
                validity: Some(a.into()),
            }),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 3, 7, 8, 8].try_into().unwrap(),
                validity: Some(b.into()),
            }),
            Nested::Primitive(Some(c.into()), true, 8),
        ];
        let expected = vec![5, 5, 5, 5, 4, 5, 5, 0, 5, 2];

        test(nested, expected)
    }

    /*
        [{"a": "a"}, {"a": "b"}],
        None,
        [{"a": "b"}, None, {"a": "b"}],
        [{"a": None}, {"a": None}, {"a": None}],
        [],
        [{"a": "d"}, {"a": "d"}, {"a": "d"}],
        None,
        [{"a": "e"}],
    */
    #[test]
    fn nested_list_struct_nullable() {
        let a = [
            true, true, true, false, true, false, false, false, true, true, true, true,
        ];
        let b = [
            true, true, true, false, true, true, true, true, true, true, true, true,
        ];
        let c = [true, false, true, true, true, true, false, true];
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 2, 2, 5, 8, 8, 11, 11, 12].try_into().unwrap(),
                validity: Some(c.into()),
            }),
            Nested::Struct(Some(b.into()), true, 12),
            Nested::Primitive(Some(a.into()), true, 12),
        ];
        let expected = vec![4, 4, 0, 4, 2, 4, 3, 3, 3, 1, 4, 4, 4, 0, 4];

        test(nested, expected)
    }

    #[test]
    fn nested_list_struct_nullable1() {
        let c = [true, false];
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 1, 1].try_into().unwrap(),
                validity: Some(c.into()),
            }),
            Nested::Struct(None, true, 1),
            Nested::Primitive(None, true, 1),
        ];
        let expected = vec![4, 0];

        test(nested, expected)
    }

    #[test]
    fn nested_struct_list_nullable() {
        let a = [true, false, true, true, true, true, false, true];
        let b = [
            true, true, true, false, true, true, true, true, true, true, true, true,
        ];
        let nested = vec![
            Nested::Struct(None, true, 12),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 2, 2, 5, 8, 8, 11, 11, 12].try_into().unwrap(),
                validity: Some(a.into()),
            }),
            Nested::Primitive(Some(b.into()), true, 12),
        ];
        let expected = vec![4, 4, 1, 4, 3, 4, 4, 4, 4, 2, 4, 4, 4, 1, 4];

        test(nested, expected)
    }

    #[test]
    fn nested_struct_list_nullable1() {
        let a = [true, true, false];
        let nested = vec![
            Nested::Struct(None, true, 3),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 1, 1, 1].try_into().unwrap(),
                validity: Some(a.into()),
            }),
            Nested::Primitive(None, true, 1),
        ];
        let expected = vec![4, 2, 1];

        test(nested, expected)
    }

    #[test]
    fn nested_list_struct_list_nullable1() {
        /*
        [
            [{"a": ["b"]}, None],
        ]
        */

        let a = [true];
        let b = [true, false];
        let c = [true, false];
        let d = [true];
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 2].try_into().unwrap(),
                validity: Some(a.into()),
            }),
            Nested::Struct(Some(b.into()), true, 2),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 1, 1].try_into().unwrap(),
                validity: Some(c.into()),
            }),
            Nested::Primitive(Some(d.into()), true, 1),
        ];
        /*
                0 6
                1 6
                0 0
                0 6
                1 2
        */
        let expected = vec![6, 2];

        test(nested, expected)
    }

    #[test]
    fn nested_list_struct_list_nullable() {
        /*
            [
            [{"a": ["a"]}, {"a": ["b"]}],
            None,
            [{"a": ["b"]}, None, {"a": ["b"]}],
            [{"a": None}, {"a": None}, {"a": None}],
            [],
            [{"a": ["d"]}, {"a": [None]}, {"a": ["c", "d"]}],
            None,
            [{"a": []}],
        ]
            */
        let a = [true, false, true, true, true, true, false, true];
        let b = [
            true, true, true, false, true, true, true, true, true, true, true, true,
        ];
        let c = [
            true, true, true, false, true, false, false, false, true, true, true, true,
        ];
        let d = [true, true, true, true, true, false, true, true];
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 2, 2, 5, 8, 8, 11, 11, 12].try_into().unwrap(),
                validity: Some(a.into()),
            }),
            Nested::Struct(Some(b.into()), true, 12),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 1, 2, 3, 3, 4, 4, 4, 4, 5, 6, 8, 8]
                    .try_into()
                    .unwrap(),
                validity: Some(c.into()),
            }),
            Nested::Primitive(Some(d.into()), true, 8),
        ];
        let expected = vec![6, 6, 0, 6, 2, 6, 3, 3, 3, 1, 6, 5, 6, 6, 0, 4];

        test(nested, expected)
    }
}
