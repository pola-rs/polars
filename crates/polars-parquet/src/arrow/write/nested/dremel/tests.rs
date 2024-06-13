use super::*;

mod def {
    use super::*;
    use crate::write::pages::{ListNested, PrimitiveNested, StructNested};

    fn test(nested: Vec<Nested>, expected: Vec<u16>) {
        let mut iter = BufferedDremelIter::new(&nested).map(|d| d.def);
        // assert_eq!(iter.size_hint().0, expected.len());
        let result = iter.by_ref().collect::<Vec<_>>();
        assert_eq!(result, expected);
        // assert_eq!(iter.size_hint().0, 0);
    }

    #[test]
    fn struct_dbl_optional() {
        let a = [true, true, true, false, true, true];
        let b = [true, false, true, false, false, true];
        let nested = vec![
            Nested::Struct(StructNested {
                is_optional: true,
                validity: Some(a.into()),
                length: 6,
            }),
            Nested::Primitive(PrimitiveNested {
                validity: Some(b.into()),
                is_optional: true,
                length: 6,
            }),
        ];
        let expected = vec![2, 1, 2, 0, 1, 2];

        test(nested, expected)
    }

    #[test]
    fn struct_optional() {
        let b = [
            true, false, true, true, false, true, false, false, true, true,
        ];
        let nested = vec![
            Nested::Struct(StructNested {
                is_optional: true,
                validity: None,
                length: 10,
            }),
            Nested::Primitive(PrimitiveNested {
                validity: Some(b.into()),
                is_optional: true,
                length: 10,
            }),
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
            Nested::Primitive(PrimitiveNested {
                validity: None,
                is_optional: true,
                length: 2,
            }),
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
            Nested::Struct(StructNested {
                validity: None,
                is_optional: true,
                length: 10,
            }),
            Nested::Primitive(PrimitiveNested {
                validity: Some(b.into()),
                is_optional: true,
                length: 10,
            }),
        ];
        let expected = vec![2, 1, 2, 2, 1, 2, 1, 1, 2, 2];

        test(nested, expected)
    }

    #[test]
    fn struct_optional_optional() {
        let nested = vec![
            Nested::Struct(StructNested {
                is_optional: true,
                validity: None,
                length: 10,
            }),
            Nested::Primitive(PrimitiveNested {
                validity: None,
                is_optional: true,
                length: 10,
            }),
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
            Nested::Primitive(PrimitiveNested {
                validity: None,
                is_optional: false,
                length: 12,
            }),
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
            Nested::Primitive(PrimitiveNested {
                validity: Some(v1.into()),
                is_optional: true,
                length: 12,
            }),
        ];
        let expected = vec![3, 3, 0, 3, 2, 3, 3, 3, 3, 1, 3, 3, 3, 0, 3];

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
            Nested::Primitive(PrimitiveNested {
                validity: None,
                is_optional: false,
                length: 10,
            }),
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
            Nested::Primitive(PrimitiveNested {
                validity: None,
                is_optional: false,
                length: 10,
            }),
        ];
        let expected = vec![3, 3, 3, 3, 3, 3, 3, 0, 1, 3, 2, 3, 3];

        test(nested, expected)
    }

    mod fixedlist {
        use super::*;

        #[test]
        fn fsl() {
            /* [ [ 1, 2 ], None, [ None, 3 ] ] */
            let a = [true, false, true];
            let b = [true, true, false, false, false, true];
            let nested = vec![
                Nested::fixed_size_list(Some(a.into()), true, 2, 3),
                Nested::primitive(Some(b.into()), true, 6),
            ];
            let expected = vec![3, 3, 0, 2, 3];

            test(nested, expected)
        }

        #[test]
        fn fsl_fsl() {
            // [
            //    [ [ 1, 2, 3 ], [ 4, 5, 6 ] ],
            //    None,
            //    [ None, [ 7, None, 9 ] ],
            // ]
            let a = [true, false, true];
            let b = [true, true, true, true, false, true];
            let c = [
                true, true, true, true, true, true, false, false, false, false, false, false,
                false, false, false, true, false, true,
            ];
            let nested = vec![
                Nested::fixed_size_list(Some(a.into()), true, 2, 3),
                Nested::fixed_size_list(Some(b.into()), true, 3, 6),
                Nested::primitive(Some(c.into()), true, 18),
            ];
            let expected = vec![5, 5, 5, 5, 5, 5, 0, 2, 5, 4, 5];

            test(nested, expected)
        }

        #[test]
        fn fsl_fsl_1() {
            // [
            //     [ [1, 5, 2], [42, 13, 37] ],
            //     None,
            //     [ None, [3, 1, 3] ]
            // ]
            let a = [true, false, true];
            let b = [true, true, false, false, false, true];
            let c = [
                true, true, true, true, true, true, false, false, false, false, false, false,
                false, false, false, true, true, true,
            ];
            let nested = vec![
                Nested::fixed_size_list(Some(a.into()), true, 2, 3),
                Nested::fixed_size_list(Some(b.into()), true, 3, 6),
                Nested::primitive(Some(c.into()), true, 18),
            ];
            let expected = vec![5, 5, 5, 5, 5, 5, 0, 2, 5, 5, 5];

            test(nested, expected)
        }
    }

    mod simple {
        use super::*;

        #[test]
        fn none() {
            /* [ None ] */
            let a = [false];
            let b = [];
            let nested = vec![
                Nested::List(ListNested {
                    is_optional: true,
                    offsets: vec![0, 0].try_into().unwrap(),
                    validity: Some(a.into()),
                }),
                Nested::List(ListNested {
                    is_optional: true,
                    offsets: vec![0].try_into().unwrap(),
                    validity: Some(b.into()),
                }),
                Nested::Primitive(PrimitiveNested {
                    validity: None,
                    is_optional: false,
                    length: 0,
                }),
            ];
            let expected = vec![0];

            test(nested, expected)
        }

        #[test]
        fn empty() {
            /* [ [ ] ] */
            let a = [true];
            let b = [];
            let nested = vec![
                Nested::List(ListNested {
                    is_optional: true,
                    offsets: vec![0, 0].try_into().unwrap(),
                    validity: Some(a.into()),
                }),
                Nested::List(ListNested {
                    is_optional: true,
                    offsets: vec![0].try_into().unwrap(),
                    validity: Some(b.into()),
                }),
                Nested::Primitive(PrimitiveNested {
                    validity: None,
                    is_optional: false,
                    length: 0,
                }),
            ];
            let expected = vec![1];

            test(nested, expected)
        }

        #[test]
        fn list_none() {
            /* [ [ None ] ] */
            let a = [true];
            let b = [false];
            let nested = vec![
                Nested::List(ListNested {
                    is_optional: true,
                    offsets: vec![0, 1].try_into().unwrap(),
                    validity: Some(a.into()),
                }),
                Nested::List(ListNested {
                    is_optional: true,
                    offsets: vec![0, 0].try_into().unwrap(),
                    validity: Some(b.into()),
                }),
                Nested::Primitive(PrimitiveNested {
                    validity: None,
                    is_optional: false,
                    length: 0,
                }),
            ];
            let expected = vec![2];

            test(nested, expected)
        }

        #[test]
        fn list_empty() {
            /* [ [ [] ] ] */
            let a = [true];
            let b = [true];
            let nested = vec![
                Nested::List(ListNested {
                    is_optional: true,
                    offsets: vec![0, 1].try_into().unwrap(),
                    validity: Some(a.into()),
                }),
                Nested::List(ListNested {
                    is_optional: true,
                    offsets: vec![0, 0].try_into().unwrap(),
                    validity: Some(b.into()),
                }),
                Nested::Primitive(PrimitiveNested {
                    validity: None,
                    is_optional: false,
                    length: 0,
                }),
            ];
            let expected = vec![3];

            test(nested, expected)
        }

        #[test]
        fn list_list_one() {
            /* [ [ [ 1 ] ] ] */
            let a = [true];
            let b = [true];
            let nested = vec![
                Nested::List(ListNested {
                    is_optional: true,
                    offsets: vec![0, 1].try_into().unwrap(),
                    validity: Some(a.into()),
                }),
                Nested::List(ListNested {
                    is_optional: true,
                    offsets: vec![0, 1].try_into().unwrap(),
                    validity: Some(b.into()),
                }),
                Nested::Primitive(PrimitiveNested {
                    validity: None,
                    is_optional: false,
                    length: 1,
                }),
            ];
            let expected = vec![4];

            test(nested, expected)
        }
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
            Nested::Primitive(PrimitiveNested {
                validity: None,
                is_optional: false,
                length: 8,
            }),
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
            Nested::Primitive(PrimitiveNested {
                validity: Some(c.into()),
                is_optional: true,
                length: 8,
            }),
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
            Nested::Struct(StructNested {
                validity: Some(b.into()),
                is_optional: true,
                length: 12,
            }),
            Nested::Primitive(PrimitiveNested {
                validity: Some(a.into()),
                is_optional: true,
                length: 12,
            }),
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
            Nested::Struct(StructNested {
                validity: None,
                is_optional: true,
                length: 1,
            }),
            Nested::Primitive(PrimitiveNested {
                validity: None,
                is_optional: true,
                length: 1,
            }),
        ];
        let expected = vec![4, 0];

        test(nested, expected)
    }

    #[test]
    fn nested_struct_list_nullable() {
        // [
        //     { "a": [] },
        //     { "a", [] },
        // ]
        let a = [true, false, true, true, true, true, false, true];
        let b = [
            true, true, true, false, true, true, true, true, true, true, true, true,
        ];
        let nested = vec![
            Nested::Struct(StructNested {
                validity: None,
                is_optional: true,
                length: 8,
            }),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 2, 2, 5, 8, 8, 11, 11, 12].try_into().unwrap(),
                validity: Some(a.into()),
            }),
            Nested::Primitive(PrimitiveNested {
                validity: Some(b.into()),
                is_optional: true,
                length: 12,
            }),
        ];
        let expected = vec![4, 4, 1, 4, 3, 4, 4, 4, 4, 2, 4, 4, 4, 1, 4];

        test(nested, expected)
    }

    #[test]
    fn nested_struct_list_nullable1() {
        let a = [true, true, false];
        let nested = vec![
            Nested::Struct(StructNested {
                validity: None,
                is_optional: true,
                length: 3,
            }),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 1, 1, 1].try_into().unwrap(),
                validity: Some(a.into()),
            }),
            Nested::Primitive(PrimitiveNested {
                validity: None,
                is_optional: true,
                length: 1,
            }),
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
            Nested::Struct(StructNested {
                validity: Some(b.into()),
                is_optional: true,
                length: 2,
            }),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 1, 1].try_into().unwrap(),
                validity: Some(c.into()),
            }),
            Nested::Primitive(PrimitiveNested {
                validity: Some(d.into()),
                is_optional: true,
                length: 1,
            }),
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
            Nested::Struct(StructNested {
                validity: Some(b.into()),
                is_optional: true,
                length: 12,
            }),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 1, 2, 3, 3, 4, 4, 4, 4, 5, 6, 8, 8]
                    .try_into()
                    .unwrap(),
                validity: Some(c.into()),
            }),
            Nested::Primitive(PrimitiveNested {
                validity: Some(d.into()),
                is_optional: true,
                length: 8,
            }),
        ];
        let expected = vec![6, 6, 0, 6, 2, 6, 3, 3, 3, 1, 6, 5, 6, 6, 0, 4];

        test(nested, expected)
    }
}

mod rep {
    use super::super::super::super::pages::ListNested;
    use super::*;

    fn test(nested: Vec<Nested>, expected: Vec<u16>) {
        let mut iter = BufferedDremelIter::new(&nested).map(|d| d.rep);
        // assert_eq!(iter.size_hint().0, expected.len());
        assert_eq!(iter.by_ref().collect::<Vec<_>>(), expected);
        // assert_eq!(iter.size_hint().0, 0);
    }

    #[test]
    fn struct_required() {
        let nested = vec![
            Nested::structure(None, false, 10),
            Nested::primitive(None, true, 10),
        ];
        let expected = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        test(nested, expected)
    }

    #[test]
    fn struct_optional() {
        let nested = vec![
            Nested::structure(None, true, 10),
            Nested::primitive(None, true, 10),
        ];
        let expected = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        test(nested, expected)
    }

    #[test]
    fn l1() {
        // [
        //    [ 1, 2 ],
        //    [],
        //    [ 3, 4, 5 ],
        //    [ 6, 7, 8 ],
        //    [],
        //    [ 9, 10, 11 ],
        //    [],
        //    [ 12 ],
        // ]
        let nested = vec![
            Nested::list(
                None,
                false,
                vec![0, 2, 2, 5, 8, 8, 11, 11, 12].try_into().unwrap(),
            ),
            Nested::primitive(None, false, 12),
        ];
        let expected = vec![0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0];

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
            Nested::primitive(None, false, 10),
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
            Nested::structure(None, true, 2),
            Nested::primitive(None, true, 2),
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
            Nested::structure(None, true, 3),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 3, 6, 7].try_into().unwrap(),
                validity: None,
            }),
            Nested::primitive(None, true, 7),
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
            Nested::structure(None, true, 1),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 4].try_into().unwrap(),
                validity: None,
            }),
            Nested::primitive(None, true, 4),
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
            Nested::primitive(None, false, 12),
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
        let a = [
            true, true, true, false, true, true, true, true, true, true, true, true,
        ];
        let nested = vec![
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 2, 2, 5, 8, 8, 11, 11, 12].try_into().unwrap(),
                validity: None,
            }),
            Nested::structure(Some(a.into()), true, 12),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 7, 8]
                    .try_into()
                    .unwrap(),
                validity: None,
            }),
            Nested::primitive(None, true, 8),
        ];
        let expected = vec![0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 2, 0, 0];

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
            Nested::structure(None, true, 12),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 0].try_into().unwrap(),
                validity: None,
            }),
            Nested::primitive(None, true, 0),
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
            Nested::structure(None, true, 12),
            Nested::List(ListNested {
                is_optional: true,
                offsets: vec![0, 0].try_into().unwrap(),
                validity: None,
            }),
            Nested::primitive(None, true, 0),
        ];
        let expected = vec![0, 0];
        // [1, 0], [0]
        // pick last

        test(nested, expected)
    }
}
