#[cfg(test)]
mod tests {
    use polars::prelude::*;
    use uint::Uint;
    use super::*;

    pub struct U256 {
        pub values: Vec<Uint<256>>,
        pub name: String,
    }

    impl U256 {
        pub fn new(values: Vec<Uint<256>>, name: String) -> Self {
            U256 { values, name }
        }

        pub fn sum(&self) -> Uint<256> {
            self.values.iter().fold(Uint::zero(), |acc, val| acc + val)
        }
    }

    #[test]
    fn test_u256_creation() {
        let values = vec![
            Uint::<256>::from(1),
            Uint::<256>::from(2),
            Uint::<256>::from(3),
        ];
        let u256 = U256::new(values, "test_u256".to_string());

        assert_eq!(u256.values.len(), 3);
        assert_eq!(u256.name, "test_u256");
    }

    #[test]
    fn test_u256_sum() {
        let values = vec![
            Uint::<256>::from(1),
            Uint::<256>::from(2),
            Uint::<256>::from(3),
        ];
        let u256 = U256::new(values, "test_u256".to_string());
        let sum = u256.sum();

        assert_eq!(sum, Uint::<256>::from(6));
    }

    #[test]
    fn test_u256_data_frame() {
        let values = vec![
            Uint::<256>::from(100),
            Uint::<256>::from(200),
            Uint::<256>::from(300),
        ];

        let df = df![
            "u256_column" => values
        ].unwrap();

        let sum = df["u256_column"].sum::<Uint<256>>().unwrap();
        assert_eq!(sum, Uint::<256>::from(600));
    }
}
