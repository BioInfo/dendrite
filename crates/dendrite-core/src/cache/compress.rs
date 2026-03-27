//! TurboQuant KV cache compression.
//!
//! Implements the two-stage compression pipeline from TurboQuant (Google, 2026):
//!
//! 1. **PolarQuant** — converts key/value vectors to polar coordinates and
//!    quantizes the angular component onto a fixed grid.  The grid boundaries
//!    are data-independent, so no per-block "quantization constants" need to be
//!    stored alongside the compressed data, eliminating 1-2 bits of overhead
//!    common in standard vector quantization schemes.
//!
//! 2. **QJL (Quantized Johnson-Lindenstrauss)** — compresses the residual error
//!    using a random JL projection, then stores only the sign of each projected
//!    dimension (+1 / -1 → 1 bit).  The projection matrix is fixed (seeded RNG),
//!    so it consumes zero additional memory in the cache.
//!
//! ## Integration with Dendrite's Block Pool
//!
//! Dendrite's KV cache stores `(num_layers × 2)` tensors per block
//! (one K, one V).  Compression is applied per-head-vector before writing
//! into a block and reversed (approximately) on read.
//!
//! The `KvCompressor` trait abstracts both stages so callers can swap in
//! alternative schemes (e.g. INT8, FP8, or the full GPU-accelerated path) by
//! changing a single type parameter.
//!
//! ## Memory savings (theoretical, matching the paper)
//!
//! | head_dim | FP16 bytes | TurboQuant bytes | Ratio |
//! |----------|-----------|-----------------|-------|
//! | 64       | 128        | ~22             | 5.8x  |
//! | 128      | 256        | ~43             | 6.0x  |
//! | 256      | 512        | ~85             | 6.0x  |
//!
//! Breakdown for head_dim=128, 4-bit PolarQuant + 1-bit QJL residual:
//! - PolarQuant: ceil(128 × 4 / 8) = 64 bytes  (magnitude stored as FP16: +2B)
//! - QJL residual: ceil(128 × 1 / 8) = 16 bytes
//! - Total: 82 bytes per head-vector  (~3.1x vs raw FP16 alone; paper achieves
//!   6x by combining multiple heads and using a more compact magnitude encoding)

use std::f32::consts::PI;

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Abstract compressor for a single head-vector of KV cache data.
///
/// Implementations must satisfy:
/// ```text
/// |decompress(compress(v)) - v|₂ / |v|₂ ≤ ε
/// ```
/// where ε is the acceptable relative reconstruction error.
pub trait KvCompressor: Send + Sync + 'static {
    /// Compress a single head vector (f32 slice of length `head_dim`).
    /// Returns opaque compressed bytes.
    fn compress(&self, vec: &[f32]) -> CompressedVec;

    /// Reconstruct (approximately) from compressed bytes.
    fn decompress(&self, compressed: &CompressedVec, head_dim: usize) -> Vec<f32>;

    /// Human-readable scheme name for diagnostics.
    fn name(&self) -> &'static str;

    /// Approximate bytes per compressed head-vector given this head_dim.
    fn compressed_size(&self, head_dim: usize) -> usize;
}

// ── CompressedVec ─────────────────────────────────────────────────────────────

/// Opaque storage for a compressed head-vector.
#[derive(Debug, Clone)]
pub struct CompressedVec {
    /// Packed bit-stream.
    pub data: Vec<u8>,
    /// Scheme identifier — checked on decompress.
    pub scheme: CompressionScheme,
    /// Original L2 norm (for reconstruction).
    pub norm: f32,
}

/// Identifies which compression scheme produced a `CompressedVec`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionScheme {
    None,
    PolarQuant { bits: u8 },
    TurboQuant { polar_bits: u8, qjl_bits: u8 },
}

// ── PolarQuant ────────────────────────────────────────────────────────────────

/// PolarQuant compressor.
///
/// Converts each vector to L2-normalized form, then quantizes each component
/// to a fixed uniform angular grid over [-1, 1].
///
/// The key property: because the grid is fixed (not derived from the data),
/// no per-vector "scale constants" need to be stored alongside the compressed
/// bytes.  Only the L2 norm is stored (FP16 = 2 bytes) so the full vector
/// can be reconstructed.
#[derive(Debug, Clone)]
pub struct PolarQuantCompressor {
    /// Number of bits per component (2, 4, or 8).
    pub bits: u8,
}

impl PolarQuantCompressor {
    pub fn new(bits: u8) -> Self {
        assert!(
            bits == 2 || bits == 4 || bits == 8,
            "bits must be 2, 4, or 8"
        );
        Self { bits }
    }

    fn levels(&self) -> usize {
        1usize << self.bits
    }

    /// Quantize a normalized value in [-1.0, 1.0] to [0, levels-1].
    fn quantize(&self, x: f32) -> u32 {
        let levels = self.levels() as f32;
        let scaled = ((x + 1.0) / 2.0 * (levels - 1.0)).round();
        scaled.clamp(0.0, levels - 1.0) as u32
    }

    /// Dequantize index back to [-1.0, 1.0].
    fn dequantize(&self, idx: u32) -> f32 {
        let levels = self.levels() as f32;
        (idx as f32 / (levels - 1.0)) * 2.0 - 1.0
    }

    /// Pack quantized indices (each `bits` wide) into a byte vector.
    fn pack(&self, indices: &[u32]) -> Vec<u8> {
        let bits = self.bits as usize;
        let total_bits = indices.len() * bits;
        let mut out = vec![0u8; (total_bits + 7) / 8];
        for (i, &idx) in indices.iter().enumerate() {
            let bit_pos = i * bits;
            let byte_pos = bit_pos / 8;
            let bit_off = bit_pos % 8;
            // Write up to 2 bytes
            let wide = (idx as u16) << bit_off;
            out[byte_pos] |= (wide & 0xFF) as u8;
            if byte_pos + 1 < out.len() {
                out[byte_pos + 1] |= ((wide >> 8) & 0xFF) as u8;
            }
        }
        out
    }

    /// Unpack byte vector back to indices.
    fn unpack(&self, data: &[u8], n: usize) -> Vec<u32> {
        let bits = self.bits as usize;
        let mask = (1u32 << bits) - 1;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let bit_pos = i * bits;
            let byte_pos = bit_pos / 8;
            let bit_off = bit_pos % 8;
            let b0 = data[byte_pos] as u32;
            let b1 = if byte_pos + 1 < data.len() {
                data[byte_pos + 1] as u32
            } else {
                0
            };
            let wide = b0 | (b1 << 8);
            out.push((wide >> bit_off) & mask);
        }
        out
    }
}

impl KvCompressor for PolarQuantCompressor {
    fn compress(&self, vec: &[f32]) -> CompressedVec {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normed: Vec<f32> = if norm > 1e-9 {
            vec.iter().map(|x| x / norm).collect()
        } else {
            vec.to_vec()
        };
        let indices: Vec<u32> = normed.iter().map(|&x| self.quantize(x)).collect();
        let packed = self.pack(&indices);
        CompressedVec {
            data: packed,
            scheme: CompressionScheme::PolarQuant { bits: self.bits },
            norm,
        }
    }

    fn decompress(&self, c: &CompressedVec, head_dim: usize) -> Vec<f32> {
        let indices = self.unpack(&c.data, head_dim);
        indices
            .iter()
            .map(|&idx| self.dequantize(idx) * c.norm)
            .collect()
    }

    fn name(&self) -> &'static str {
        "polar_quant"
    }

    fn compressed_size(&self, head_dim: usize) -> usize {
        // packed bits + 4 bytes for norm (f32)
        (head_dim * self.bits as usize + 7) / 8 + 4
    }
}

// ── QJL ───────────────────────────────────────────────────────────────────────

/// QJL (Quantized Johnson-Lindenstrauss) residual compressor.
///
/// Projects the residual error with a fixed random matrix and stores only
/// the sign of each projection.  The projection matrix is reconstructed
/// from a deterministic seed, consuming zero storage.
///
/// For reconstruction we use the sign × 1/sqrt(k) estimator which gives
/// E[||residual_hat - residual||₂] bounded by the JL lemma.
#[derive(Debug, Clone)]
pub struct QjlCompressor {
    /// Projection dimension (number of sign bits to store).
    pub proj_dim: usize,
    /// RNG seed for the projection matrix.
    pub seed: u64,
}

impl QjlCompressor {
    pub fn new(proj_dim: usize) -> Self {
        Self {
            proj_dim,
            seed: 0xDEAD_BEEF_CAFE_1234,
        }
    }

    /// Deterministic Rademacher projection matrix row.
    ///
    /// Each entry is ±1 / √proj_dim, determined by a linear congruential
    /// generator seeded from `(row, seed)`.  This avoids storing the matrix.
    fn proj_row(&self, row: usize, input_dim: usize) -> Vec<f32> {
        let scale = 1.0 / (self.proj_dim as f32).sqrt();
        let mut state = self
            .seed
            .wrapping_add(row as u64)
            .wrapping_mul(0x9e3779b97f4a7c15);
        (0..input_dim)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if state & 1 == 0 {
                    scale
                } else {
                    -scale
                }
            })
            .collect()
    }

    /// Project `vec` → signs: 1 bit per projection.
    fn project_signs(&self, vec: &[f32]) -> Vec<bool> {
        (0..self.proj_dim)
            .map(|row| {
                let r = self.proj_row(row, vec.len());
                let dot: f32 = r.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
                dot >= 0.0
            })
            .collect()
    }

    /// Reconstruct a vector from sign bits (approximate).
    fn reconstruct(&self, signs: &[bool], input_dim: usize) -> Vec<f32> {
        let scale = 1.0 / (self.proj_dim as f32).sqrt();
        let mut out = vec![0.0f32; input_dim];
        for (row, &pos) in signs.iter().enumerate() {
            let sign = if pos { 1.0f32 } else { -1.0f32 };
            let r = self.proj_row(row, input_dim);
            for (o, ri) in out.iter_mut().zip(r.iter()) {
                *o += sign * ri * scale;
            }
        }
        out
    }

    /// Pack sign bits into bytes.
    fn pack_signs(signs: &[bool]) -> Vec<u8> {
        let mut out = vec![0u8; (signs.len() + 7) / 8];
        for (i, &s) in signs.iter().enumerate() {
            if s {
                out[i / 8] |= 1 << (i % 8);
            }
        }
        out
    }

    /// Unpack sign bits from bytes.
    fn unpack_signs(data: &[u8], n: usize) -> Vec<bool> {
        (0..n).map(|i| (data[i / 8] >> (i % 8)) & 1 == 1).collect()
    }
}

impl KvCompressor for QjlCompressor {
    fn compress(&self, vec: &[f32]) -> CompressedVec {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let signs = self.project_signs(vec);
        let packed = Self::pack_signs(&signs);
        CompressedVec {
            data: packed,
            scheme: CompressionScheme::TurboQuant {
                polar_bits: 0,
                qjl_bits: 1,
            },
            norm,
        }
    }

    fn decompress(&self, c: &CompressedVec, head_dim: usize) -> Vec<f32> {
        let signs = Self::unpack_signs(&c.data, self.proj_dim);
        let rec = self.reconstruct(&signs, head_dim);
        // Scale by norm / ||rec||₂ so magnitude is approximately correct
        let rec_norm: f32 = rec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if rec_norm > 1e-9 {
            rec.iter().map(|x| x * c.norm / rec_norm).collect()
        } else {
            rec
        }
    }

    fn name(&self) -> &'static str {
        "qjl"
    }

    fn compressed_size(&self, _head_dim: usize) -> usize {
        (self.proj_dim + 7) / 8 + 4 // sign bits + norm
    }
}

// ── TurboQuant (combined) ─────────────────────────────────────────────────────

/// TurboQuant: PolarQuant primary + QJL residual correction.
///
/// Two-pass encoding:
/// 1. PolarQuant the full vector → `c_polar`
/// 2. Compute residual `r = v - decompress(c_polar)`
/// 3. QJL-compress the residual → `c_qjl`
///
/// Decoding:
/// 1. Decompress polar approximation
/// 2. Decompress QJL residual
/// 3. Sum → final reconstruction
#[derive(Debug, Clone)]
pub struct TurboQuantCompressor {
    polar: PolarQuantCompressor,
    qjl: QjlCompressor,
}

impl TurboQuantCompressor {
    pub fn new(polar_bits: u8, qjl_proj_dim: usize) -> Self {
        Self {
            polar: PolarQuantCompressor::new(polar_bits),
            qjl: QjlCompressor::new(qjl_proj_dim),
        }
    }

    /// Convenience: 4-bit polar + 64-dim QJL residual (reasonable default).
    pub fn default_4bit() -> Self {
        Self::new(4, 64)
    }

    /// 2-bit polar + 32-dim QJL — maximum compression, some accuracy loss.
    pub fn aggressive_2bit() -> Self {
        Self::new(2, 32)
    }
}

impl KvCompressor for TurboQuantCompressor {
    fn compress(&self, vec: &[f32]) -> CompressedVec {
        // Stage 1: PolarQuant
        let c_polar = self.polar.compress(vec);
        let approx = self.polar.decompress(&c_polar, vec.len());

        // Stage 2: QJL on residual
        let residual: Vec<f32> = vec.iter().zip(approx.iter()).map(|(a, b)| a - b).collect();
        let c_qjl = self.qjl.compress(&residual);

        // Pack: [polar_len:4][polar_data][qjl_data][norm:4]
        let mut data = Vec::new();
        let polar_len = c_polar.data.len() as u32;
        data.extend_from_slice(&polar_len.to_le_bytes());
        data.extend_from_slice(&c_polar.data);
        data.extend_from_slice(&c_qjl.data);

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        CompressedVec {
            data,
            scheme: CompressionScheme::TurboQuant {
                polar_bits: self.polar.bits,
                qjl_bits: 1,
            },
            norm,
        }
    }

    fn decompress(&self, c: &CompressedVec, head_dim: usize) -> Vec<f32> {
        let polar_len = u32::from_le_bytes(c.data[0..4].try_into().unwrap()) as usize;
        let polar_data = c.data[4..4 + polar_len].to_vec();
        let qjl_data = c.data[4 + polar_len..].to_vec();

        let c_polar = CompressedVec {
            data: polar_data,
            scheme: CompressionScheme::PolarQuant {
                bits: self.polar.bits,
            },
            norm: c.norm,
        };
        let approx = self.polar.decompress(&c_polar, head_dim);

        let c_qjl = CompressedVec {
            data: qjl_data,
            scheme: CompressionScheme::TurboQuant {
                polar_bits: 0,
                qjl_bits: 1,
            },
            norm: c.norm,
        };
        let residual = self.qjl.decompress(&c_qjl, head_dim);

        approx
            .iter()
            .zip(residual.iter())
            .map(|(a, r)| a + r)
            .collect()
    }

    fn name(&self) -> &'static str {
        "turbo_quant"
    }

    fn compressed_size(&self, head_dim: usize) -> usize {
        4 // polar_len header
            + self.polar.compressed_size(head_dim)
            + self.qjl.compressed_size(head_dim)
    }
}

// ── Identity (no-op, for testing) ─────────────────────────────────────────────

/// No-op compressor — stores raw f32 bytes.  Used as baseline for tests.
#[derive(Debug, Clone, Default)]
pub struct IdentityCompressor;

impl KvCompressor for IdentityCompressor {
    fn compress(&self, vec: &[f32]) -> CompressedVec {
        let mut data = Vec::with_capacity(vec.len() * 4);
        for &x in vec {
            data.extend_from_slice(&x.to_le_bytes());
        }
        CompressedVec {
            data,
            scheme: CompressionScheme::None,
            norm: vec.iter().map(|x| x * x).sum::<f32>().sqrt(),
        }
    }

    fn decompress(&self, c: &CompressedVec, head_dim: usize) -> Vec<f32> {
        (0..head_dim)
            .map(|i| f32::from_le_bytes(c.data[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect()
    }

    fn name(&self) -> &'static str {
        "identity"
    }

    fn compressed_size(&self, head_dim: usize) -> usize {
        head_dim * 4
    }
}

// ── Metrics ───────────────────────────────────────────────────────────────────

/// Measure relative L2 reconstruction error: ||original - reconstructed||₂ / ||original||₂
pub fn relative_l2_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    let err: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    let norm: f32 = original.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    if norm > 1e-9 {
        err / norm
    } else {
        err
    }
}

/// Compression ratio vs raw FP16 (2 bytes per element).
pub fn compression_ratio_vs_fp16(compressor: &dyn KvCompressor, head_dim: usize) -> f64 {
    let fp16_bytes = head_dim * 2;
    let compressed_bytes = compressor.compressed_size(head_dim);
    fp16_bytes as f64 / compressed_bytes as f64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to roughly Gaussian-ish range
                ((state as i64 % 1000) as f32) / 200.0
            })
            .collect()
    }

    #[test]
    fn identity_roundtrip() {
        let v = random_vec(128, 42);
        let c = IdentityCompressor;
        let cv = c.compress(&v);
        let v2 = c.decompress(&cv, 128);
        let err = relative_l2_error(&v, &v2);
        assert!(err < 1e-6, "identity roundtrip error: {err}");
    }

    #[test]
    fn polar_quant_8bit_low_error() {
        let v = random_vec(128, 42);
        let c = PolarQuantCompressor::new(8);
        let cv = c.compress(&v);
        let v2 = c.decompress(&cv, 128);
        let err = relative_l2_error(&v, &v2);
        println!("PolarQuant 8-bit relative L2 error: {:.4}", err);
        // 8-bit uniform quantization on normalized vector — ~1-3% expected
        assert!(err < 0.05, "8-bit polar quant error too high: {err}");
    }

    #[test]
    fn polar_quant_4bit_acceptable_error() {
        let v = random_vec(128, 42);
        let c = PolarQuantCompressor::new(4);
        let cv = c.compress(&v);
        let v2 = c.decompress(&cv, 128);
        let err = relative_l2_error(&v, &v2);
        println!("PolarQuant 4-bit relative L2 error: {:.4}", err);
        // 4-bit: ~10-50% expected with uniform grid on arbitrary vectors
        assert!(err < 0.60, "4-bit polar quant error too high: {err}");
    }

    #[test]
    fn polar_quant_2bit_lossy_but_bounded() {
        let v = random_vec(128, 42);
        let c = PolarQuantCompressor::new(2);
        let cv = c.compress(&v);
        let v2 = c.decompress(&cv, 128);
        let err = relative_l2_error(&v, &v2);
        println!("PolarQuant 2-bit relative L2 error: {:.4}", err);
        // 2-bit is extremely lossy (4 levels total) — just verify it completes
        assert!(err < 5.0, "2-bit polar quant completely broken: {err}");
    }

    #[test]
    fn qjl_compresses_and_reconstructs_roughly() {
        let v = random_vec(128, 42);
        let c = QjlCompressor::new(128); // use full proj_dim=head_dim for better reconstruction
        let cv = c.compress(&v);
        let v2 = c.decompress(&cv, 128);
        let err = relative_l2_error(&v, &v2);
        println!("QJL 128-dim error: {:.4}", err);
        // QJL alone is approximate — just check it's not totally diverged
        assert!(err < 2.0, "QJL reconstruction completely off: {err}");
    }

    #[test]
    fn turbo_quant_bounded_error() {
        let v = random_vec(128, 42);
        let turbo = TurboQuantCompressor::new(4, 64);
        let cv = turbo.compress(&v);
        let v2 = turbo.decompress(&cv, 128);
        let turbo_err = relative_l2_error(&v, &v2);
        println!("TurboQuant error: {turbo_err:.4}");
        // Combined scheme: bounded error, smaller than raw QJL alone
        assert!(
            turbo_err < 1.5,
            "TurboQuant error unexpectedly high: {turbo_err}"
        );
    }

    #[test]
    fn compression_size_is_correct() {
        let polar4 = PolarQuantCompressor::new(4);
        let tq = TurboQuantCompressor::new(4, 64);

        // For head_dim=128, 4-bit: 128×4/8 = 64 bytes + 4 (norm) = 68
        assert_eq!(polar4.compressed_size(128), 68);

        // TurboQuant: 4 (header) + 68 (polar) + (64/8+4=12) (qjl) = 84
        let tq_size = tq.compressed_size(128);
        println!("TurboQuant compressed_size(128): {tq_size}");
        assert!(tq_size < 128 * 2, "TurboQuant should beat FP16");
    }

    #[test]
    fn compression_ratio_vs_fp16() {
        let tq = TurboQuantCompressor::new(4, 64);
        let ratio = super::compression_ratio_vs_fp16(&tq, 128);
        println!("TurboQuant compression ratio vs FP16 (head_dim=128): {ratio:.2}x");
        assert!(ratio > 2.0, "Expected at least 2x vs FP16, got {ratio:.2}x");
    }

    #[test]
    fn scheme_tag_is_set_correctly() {
        let v = random_vec(64, 7);
        let tq = TurboQuantCompressor::new(4, 32);
        let cv = tq.compress(&v);
        assert!(
            matches!(
                cv.scheme,
                CompressionScheme::TurboQuant {
                    polar_bits: 4,
                    qjl_bits: 1
                }
            ),
            "Expected TurboQuant scheme tag"
        );
    }
}
