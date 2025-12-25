//! Token masking for constrained decoding.

/// A mask over the vocabulary indicating allowed tokens.
#[derive(Debug, Clone)]
pub struct TokenMask {
    /// Bit vector of allowed tokens.
    mask: Vec<u64>,
    /// Vocabulary size.
    vocab_size: usize,
    /// Number of allowed tokens.
    num_allowed: usize,
}

impl TokenMask {
    /// Create a mask that allows all tokens.
    pub fn allow_all(vocab_size: usize) -> Self {
        let num_words = vocab_size.div_ceil(64);
        let mut mask = vec![u64::MAX; num_words];

        // Clear bits beyond vocab_size
        let remainder = vocab_size % 64;
        if remainder > 0 {
            mask[num_words - 1] = (1u64 << remainder) - 1;
        }

        Self {
            mask,
            vocab_size,
            num_allowed: vocab_size,
        }
    }

    /// Create a mask that blocks all tokens.
    pub fn block_all(vocab_size: usize) -> Self {
        let num_words = vocab_size.div_ceil(64);
        Self {
            mask: vec![0; num_words],
            vocab_size,
            num_allowed: 0,
        }
    }

    /// Create a mask from allowed token IDs.
    pub fn from_allowed(vocab_size: usize, allowed: &[u32]) -> Self {
        let mut mask = Self::block_all(vocab_size);
        for &token in allowed {
            mask.allow(token);
        }
        mask
    }

    /// Check if a token is allowed.
    pub fn is_allowed(&self, token: u32) -> bool {
        if token as usize >= self.vocab_size {
            return false;
        }
        let word_idx = token as usize / 64;
        let bit_idx = token as usize % 64;
        (self.mask[word_idx] >> bit_idx) & 1 == 1
    }

    /// Allow a token.
    pub fn allow(&mut self, token: u32) {
        if token as usize >= self.vocab_size {
            return;
        }
        let word_idx = token as usize / 64;
        let bit_idx = token as usize % 64;
        if !self.is_allowed(token) {
            self.mask[word_idx] |= 1u64 << bit_idx;
            self.num_allowed += 1;
        }
    }

    /// Block a token.
    pub fn block(&mut self, token: u32) {
        if token as usize >= self.vocab_size {
            return;
        }
        let word_idx = token as usize / 64;
        let bit_idx = token as usize % 64;
        if self.is_allowed(token) {
            self.mask[word_idx] &= !(1u64 << bit_idx);
            self.num_allowed -= 1;
        }
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get number of allowed tokens.
    pub fn num_allowed(&self) -> usize {
        self.num_allowed
    }

    /// Apply mask to logits (set blocked tokens to -inf).
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        for (i, logit) in logits.iter_mut().enumerate() {
            if !self.is_allowed(i as u32) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// Iterate over allowed tokens.
    pub fn allowed_tokens(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.vocab_size as u32).filter(|&t| self.is_allowed(t))
    }

    /// Intersect with another mask.
    pub fn intersect(&mut self, other: &TokenMask) {
        debug_assert_eq!(self.vocab_size, other.vocab_size);
        self.num_allowed = 0;
        for (a, b) in self.mask.iter_mut().zip(other.mask.iter()) {
            *a &= *b;
            self.num_allowed += a.count_ones() as usize;
        }
    }

    /// Union with another mask.
    pub fn union(&mut self, other: &TokenMask) {
        debug_assert_eq!(self.vocab_size, other.vocab_size);
        self.num_allowed = 0;
        for (a, b) in self.mask.iter_mut().zip(other.mask.iter()) {
            *a |= *b;
            self.num_allowed += a.count_ones() as usize;
        }
    }

    /// Get the raw mask as bytes (for GPU transfer).
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.mask.as_ptr() as *const u8, self.mask.len() * 8) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allow_all() {
        let mask = TokenMask::allow_all(100);
        assert_eq!(mask.num_allowed(), 100);
        assert!(mask.is_allowed(0));
        assert!(mask.is_allowed(99));
        assert!(!mask.is_allowed(100));
    }

    #[test]
    fn test_block_all() {
        let mask = TokenMask::block_all(100);
        assert_eq!(mask.num_allowed(), 0);
        assert!(!mask.is_allowed(0));
    }

    #[test]
    fn test_allow_block() {
        let mut mask = TokenMask::block_all(100);
        mask.allow(42);
        assert!(mask.is_allowed(42));
        assert_eq!(mask.num_allowed(), 1);

        mask.block(42);
        assert!(!mask.is_allowed(42));
        assert_eq!(mask.num_allowed(), 0);
    }
}
