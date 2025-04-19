use super::BooleanChunked;

impl BooleanChunked {
    pub fn num_trues(&self) -> usize {
        self.downcast_iter()
            .map(|arr| match arr.validity() {
                None => arr.values().set_bits(),
                Some(validity) => arr.values().num_intersections_with(validity),
            })
            .sum()
    }

    pub fn num_falses(&self) -> usize {
        self.downcast_iter()
            .map(|arr| match arr.validity() {
                None => arr.values().unset_bits(),
                Some(validity) => (!arr.values()).num_intersections_with(validity),
            })
            .sum()
    }
}
