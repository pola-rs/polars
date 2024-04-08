use polars_utils::slice::GetSaferUnchecked;

use super::super::pages::Nested;
use super::to_length;

trait DebugIter: Iterator<Item = usize> + std::fmt::Debug {}

impl<A: Iterator<Item = usize> + std::fmt::Debug> DebugIter for A {}

fn iter<'a>(nested: &'a [Nested]) -> Vec<Box<dyn DebugIter + 'a>> {
    nested
        .iter()
        .filter_map(|nested| match nested {
            Nested::Primitive(_, _, _) => None,
            Nested::List(nested) => {
                Some(Box::new(to_length(&nested.offsets)) as Box<dyn DebugIter>)
            },
            Nested::LargeList(nested) => {
                Some(Box::new(to_length(&nested.offsets)) as Box<dyn DebugIter>)
            },
            Nested::FixedSizeList { width, len, .. } => {
                Some(Box::new(std::iter::repeat(*width).take(*len)) as Box<dyn DebugIter>)
            },
            Nested::Struct(_, _, _) => None,
        })
        .collect()
}

/// return number values of the nested
pub fn num_values(nested: &[Nested]) -> usize {
    let pr = match nested.last().unwrap() {
        Nested::Primitive(_, _, len) => *len,
        _ => unreachable!(),
    };

    iter(nested)
        .into_iter()
        .map(|lengths| {
            lengths
                .map(|length| if length == 0 { 1 } else { 0 })
                .sum::<usize>()
        })
        .sum::<usize>()
        + pr
}

/// Iterator adapter of parquet / dremel repetition levels
#[derive(Debug)]
pub struct RepLevelsIter<'a> {
    // iterators of lengths. E.g. [[[a,b,c], [d,e,f,g]], [[h], [i,j]]] -> [[2, 2], [3, 4, 1, 2]]
    iter: Vec<Box<dyn DebugIter + 'a>>,
    // vector containing the remaining number of values of each iterator.
    // e.g. the iters [[2, 2], [3, 4, 1, 2]] after the first iteration will return [2, 3],
    // and remaining will be [2, 3].
    // on the second iteration, it will be `[2, 2]` (since iterations consume the last items)
    remaining: Vec<usize>, /* < remaining.len() == iter.len() */
    // cache of the first `remaining` that is non-zero. Examples:
    // * `remaining = [2, 2] => current_level = 2`
    // * `remaining = [2, 0] => current_level = 1`
    // * `remaining = [0, 0] => current_level = 0`
    current_level: usize, /* < iter.len() */
    // the number to discount due to being the first element of the iterators.
    total: usize, /* < iter.len() */

    // the total number of items that this iterator will return
    remaining_values: usize,
}

impl<'a> RepLevelsIter<'a> {
    pub fn new(nested: &'a [Nested]) -> Self {
        let remaining_values = num_values(nested);

        let iter = iter(nested);
        let remaining = vec![0; iter.len()];

        Self {
            iter,
            remaining,
            total: 0,
            current_level: 0,
            remaining_values,
        }
    }
}

impl<'a> Iterator for RepLevelsIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_values == 0 {
            return None;
        }
        if self.remaining.is_empty() {
            self.remaining_values -= 1;
            return Some(0);
        }

        for (iter, remaining) in self
            .iter
            .iter_mut()
            .zip(self.remaining.iter_mut())
            .skip(self.current_level)
        {
            let length: usize = iter.next()?;
            *remaining = length;
            if length == 0 {
                break;
            }
            self.current_level += 1;
            self.total += 1;
        }

        // track
        if let Some(x) = self.remaining.get_mut(self.current_level.saturating_sub(1)) {
            *x = x.saturating_sub(1)
        }
        let r = Some((self.current_level - self.total) as u32);

        // update
        unsafe {
            for index in (1..self.current_level).rev() {
                if *self.remaining.get_unchecked_release(index) == 0 {
                    self.current_level -= 1;
                    *self.remaining.get_unchecked_release_mut(index - 1) -= 1;
                }
            }
            if *self.remaining.get_unchecked_release(0) == 0 {
                self.current_level = self.current_level.saturating_sub(1);
            }
        }
        self.total = 0;
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
    use super::super::super::pages::ListNested;
    use super::*;

    fn test(nested: Vec<Nested>, expected: Vec<u32>) {
        let mut iter = RepLevelsIter::new(&nested);
        assert_eq!(iter.size_hint().0, expected.len());
        assert_eq!(iter.by_ref().collect::<Vec<_>>(), expected);
        assert_eq!(iter.size_hint().0, 0);
    }

    #[test]
    fn struct_required() {
        let nested = vec![
            Nested::Struct(None, false, 10),
            Nested::Primitive(None, true, 10),
        ];
        let expected = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        test(nested, expected)
    }

    #[test]
    fn struct_optional() {
        let nested = vec![
            Nested::Struct(None, true, 10),
            Nested::Primitive(None, true, 10),
        ];
        let expected = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        test(nested, expected)
    }

    #[test]
    fn l1() {
        let nested = vec![
            Nested::List(ListNested {
                is_optional: false,
                offsets: vec![0, 2, 2, 5, 8, 8, 11, 11, 12].try_into().unwrap(),
                validity: None,
            }),
            Nested::Primitive(None, false, 12),
        ];
        let expected = vec![0u32, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0];

        test(nested, expected)
    }

    #[test]
    fn l2() {
        let nested = vec![
            Nested::List(ListNested {
                is_optional: false,
                offsets: vec![0, 2, 2, 4].try_into().unwrap(),
                validity: None,
            }),
            Nested::List(ListNested {
                is_optional: false,
                offsets: vec![0, 3, 7, 8, 10].try_into().unwrap(),
                validity: None,
            }),
            Nested::Primitive(None, false, 10),
        ];
        let expected = vec![0, 2, 2, 1, 2, 2, 2, 0, 0, 1, 2];

        test(nested, expected)
    }

    #[test]
    fn list_of_struct() {
        /*
        [
            [{"a": "b"}],[{"a": "c"}]
        ]
        */
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 1, 2].try_into().unwrap(),
                validity: None,
            }),
            Nested::Struct(None, true, 2),
            Nested::Primitive(None, true, 2),
        ];
        let expected = vec![0, 0];

        test(nested, expected)
    }

    #[test]
    fn list_struct_list() {
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 2, 3].try_into().unwrap(),
                validity: None,
            }),
            Nested::Struct(None, true, 3),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 3, 6, 7].try_into().unwrap(),
                validity: None,
            }),
            Nested::Primitive(None, true, 7),
        ];
        let expected = vec![0, 2, 2, 1, 2, 2, 0];

        test(nested, expected)
    }

    #[test]
    fn struct_list_optional() {
        /*
        {"f1": ["a", "b", None, "c"]}
        */
        let nested = vec![
            Nested::Struct(None, true, 1),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 4].try_into().unwrap(),
                validity: None,
            }),
            Nested::Primitive(None, true, 4),
        ];
        let expected = vec![0, 1, 1, 1];

        test(nested, expected)
    }

    #[test]
    fn l2_other() {
        let nested = vec![
            Nested::List(ListNested {
                is_optional: false,
                offsets: vec![0, 1, 1, 3, 5, 5, 8, 8, 9].try_into().unwrap(),
                validity: None,
            }),
            Nested::List(ListNested {
                is_optional: false,
                offsets: vec![0, 2, 4, 5, 7, 8, 9, 10, 11, 12].try_into().unwrap(),
                validity: None,
            }),
            Nested::Primitive(None, false, 12),
        ];
        let expected = vec![0, 2, 0, 0, 2, 1, 0, 2, 1, 0, 0, 1, 1, 0, 0];

        test(nested, expected)
    }

    #[test]
    fn list_struct_list_1() {
        /*
        [
            [{"a": ["a"]}, {"a": ["b"]}],
            [],
            [{"a": ["b"]}, None, {"a": ["b"]}],
            [{"a": []}, {"a": []}, {"a": []}],
            [],
            [{"a": ["d"]}, {"a": ["a"]}, {"a": ["c", "d"]}],
            [],
            [{"a": []}],
        ]
        // reps: [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 2, 0, 0]
        */
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 2, 2, 5, 8, 8, 11, 11, 12].try_into().unwrap(),
                validity: None,
            }),
            Nested::Struct(None, true, 12),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 1, 2, 3, 3, 4, 4, 4, 4, 5, 6, 8].try_into().unwrap(),
                validity: None,
            }),
            Nested::Primitive(None, true, 8),
        ];
        let expected = vec![0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 2, 0];

        test(nested, expected)
    }

    #[test]
    fn list_struct_list_2() {
        /*
        [
            [{"a": []}],
        ]
        // reps: [0]
        */
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 1].try_into().unwrap(),
                validity: None,
            }),
            Nested::Struct(None, true, 12),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 0].try_into().unwrap(),
                validity: None,
            }),
            Nested::Primitive(None, true, 0),
        ];
        let expected = vec![0];

        test(nested, expected)
    }

    #[test]
    fn list_struct_list_3() {
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 1, 1].try_into().unwrap(),
                validity: None,
            }),
            Nested::Struct(None, true, 12),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 0].try_into().unwrap(),
                validity: None,
            }),
            Nested::Primitive(None, true, 0),
        ];
        let expected = vec![0, 0];
        // [1, 0], [0]
        // pick last

        test(nested, expected)
    }
}
