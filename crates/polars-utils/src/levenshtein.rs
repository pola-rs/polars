// Implementation from: https://github.com/wooorm/levenshtein-rs/blob/9c4730b1973d4e61e187f8fe6d5f299ad5c991fc/src/lib.rs
// (MIT licensed. Copyright (c) 2016 Titus Wormer) <tituswormer@gmail.com>
pub fn levenshtein(a: &str, b: &str) -> usize {
    let mut result = 0;

    if a == b {
        return result;
    }

    let length_a = a.chars().count();
    let length_b = b.chars().count();

    if length_a == 0 {
        return length_b;
    }

    if length_b == 0 {
        return length_a;
    }

    let mut cache: Vec<usize> = (1..).take(length_a).collect();

    for (index_b, code_b) in b.chars().enumerate() {
        result = index_b;
        let mut distance_a = index_b;

        for (index_a, code_a) in a.chars().enumerate() {
            let distance_b = if code_a == code_b {
                distance_a
            } else {
                distance_a + 1
            };

            distance_a = cache[index_a];

            result = if distance_a > result {
                if distance_b > result {
                    result + 1
                } else {
                    distance_b
                }
            } else if distance_b > distance_a {
                distance_a + 1
            } else {
                distance_b
            };

            cache[index_a] = result;
        }
    }

    result
}

/// Return the closest candidate to `name` whose edit distance is within
/// `max(1, name.len() / 3)`, or `None` if no candidate is close enough.
pub fn did_you_mean<'a>(name: &str, candidates: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    let threshold = (name.len() / 3).max(1);
    candidates
        .filter_map(|c| {
            let d = levenshtein(name, c);
            if d <= threshold { Some((c, d)) } else { None }
        })
        .min_by_key(|&(_, d)| d)
        .map(|(c, _)| c)
}
