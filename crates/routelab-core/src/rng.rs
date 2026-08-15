//! A small deterministic generator, so anything the library decides at random
//! is reproducible.
//!
//! Landmark selection, contraction order, and the tests' random instances all
//! want the same thing: given a seed, the same answer every run, so a failing
//! case can be rerun and a benchmark compared against itself. None of them want
//! statistical quality — nothing here is sampling a distribution — so xorshift
//! is enough and the crate takes no dependency for it.

/// A seeded xorshift64 generator.
#[derive(Debug, Clone)]
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        // Any nonzero state will do; xorshift stays stuck at zero. The multiply
        // spreads a small seed across the word, so seeds 0 and 1 do not produce
        // near-identical streams.
        Rng(seed.wrapping_mul(2_685_821_657_736_338_717).max(1))
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in `0..bound`. Modulo, with the bias that implies — the callers
    /// here are choosing nodes out of thousands, where it is unmeasurable.
    pub(crate) fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_stream() {
        let draw = |seed| {
            let mut rng = Rng::new(seed);
            (0..8).map(|_| rng.next()).collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
        assert_ne!(draw(7), draw(8));
    }

    #[test]
    fn a_zero_seed_still_generates() {
        // Left as bare state, xorshift would return zero forever.
        let mut rng = Rng::new(0);
        assert_ne!(rng.next(), 0);
        assert_ne!(rng.next(), 0);
    }

    #[test]
    fn below_stays_in_range() {
        let mut rng = Rng::new(3);
        assert!((0..500).all(|_| rng.below(10) < 10));
    }
}
