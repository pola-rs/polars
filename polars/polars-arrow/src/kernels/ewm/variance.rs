use num::Float;

// See: https://stats.stackexchange.com/a/111912/147321

pub fn ewm_std<T: Float>(x_vals: &[T], ewma_vals: &mut [T], alpha: T) {
    if !ewma_vals.is_empty() {
        let mut emwa_prev = ewma_vals[0];
        let mut emvar_prev = T::zero();
        ewma_vals[0] = T::zero();

        let one_sub_alpha = T::one() - alpha;
        let two = T::one() + T::one();

        let mut x_iter = x_vals.iter();
        x_iter.next();

        for (xi, ewma_i) in x_iter.zip(ewma_vals[1..].iter_mut()) {
            let delta_i = *xi - emwa_prev;
            emwa_prev = *ewma_i;

            emvar_prev = one_sub_alpha * (emvar_prev + alpha * delta_i.powf(two));
            *ewma_i = emvar_prev.sqrt();
        }
    }
}

pub fn ewm_var<T: Float>(x_vals: &[T], ewma_vals: &mut [T], alpha: T) {
    if !ewma_vals.is_empty() {
        let mut emwa_prev = ewma_vals[0];
        let mut emvar_prev = T::zero();
        ewma_vals[0] = T::zero();

        let one_sub_alpha = T::one() - alpha;
        let two = T::one() + T::one();

        let mut x_iter = x_vals.iter();
        x_iter.next();

        for (xi, ewma_i) in x_iter.zip(ewma_vals[1..].iter_mut()) {
            let delta_i = *xi - emwa_prev;
            emwa_prev = *ewma_i;

            emvar_prev = one_sub_alpha * (emvar_prev + alpha * delta_i.powf(two));
            *ewma_i = emvar_prev;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_emw_var() {
        let x = [1.0, 5.0, 7.0, 1.0, 2.0, 1.0, 4.0];
        let mut ewma = [
            1.0,
            3.6666666666666665,
            5.571428571428571,
            3.1333333333333333,
            2.5483870967741935,
            1.7619047619047619,
            2.8897637795275593,
        ];

        ewm_var(&x, &mut ewma, 0.5);
        let expected = [
            0.0,
            4.0,
            4.777777777777779,
            7.613378684807256,
            4.127800453514739,
            2.6632758771215737,
            2.583905512256932,
        ];
        assert_eq!(ewma, expected);
    }
}
