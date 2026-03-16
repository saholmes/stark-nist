//! Pluggable Fiat–Shamir transcript with backward-compatible API.
//!
//! Goldilocks-safe field embedding (64-bit).
//!
//! # Supported hash backends
//!
//!  * **Poseidon** — algebraic sponge, native field operations.
//!  * **SHA3-256** — 128-bit collision resistance.
//!  * **SHA3-384** — 192-bit collision resistance, wider internal state.
//!  * **SHA3-512** — 256-bit collision resistance, widest internal state.
//!  * **Blake3**  — 128-bit collision resistance, parallel-friendly.
//!
//! [`Transcript::challenge_bytes`] always returns **32 bytes** regardless of
//! the backend.  SHA3-384 and SHA3-512 produce wider native digests, but the
//! output is truncated to 32 bytes — this provides 256 bits of challenge
//! entropy, far exceeding what a 64-bit Goldilocks field requires.  The wider
//! internal hash state still benefits security: transcript state collisions
//! are harder to manufacture.
//!
//! # Backend behaviour note
//!
//! The SHA3 and Blake3 backends use a **fork-then-finalize** strategy for
//! [`challenge`](Transcript::challenge): they clone the running hash state,
//! finalize the clone, and leave the main state unchanged.  The Poseidon
//! backend instead **absorbs the label and squeezes**, permanently advancing
//! state.  Both approaches are correct when callers use unique labels and
//! interleave absorbs between challenges.  The practical difference is that
//! two consecutive `challenge(b"same_label")` calls *without* an intervening
//! absorb will return **identical** values on SHA3/Blake3 but **distinct**
//! values on Poseidon.

#![allow(dead_code)]
#![allow(unused_imports)]

use ark_ff::{BigInteger, PrimeField, Zero};
use ark_goldilocks::Goldilocks as F;
use std::sync::Once;

// ────────────────── Domain separation tags ──────────────────

pub mod ds {
    pub const TRANSCRIPT_INIT: &[u8] = b"FSv1-TRANSCRIPT-INIT";
    pub const ABSORB_BYTES: &[u8] = b"FSv1-ABSORB-BYTES";
    pub const CHALLENGE: &[u8] = b"FSv1-CHALLENGE";
}

// ────────────────── Helpers (Goldilocks-safe) ──────────────────

#[inline]
fn bytes_to_field_u64(bytes: &[u8]) -> F {
    let mut le = [0u8; 8];
    le[..bytes.len().min(8)].copy_from_slice(&bytes[..bytes.len().min(8)]);
    F::from(u64::from_le_bytes(le))
}

fn domain_tag_to_field(tag: &[u8]) -> F {
    bytes_to_field_u64(tag)
}

fn bytes_to_field_words(bytes: &[u8]) -> Vec<F> {
    bytes.chunks(8).map(bytes_to_field_u64).collect()
}

// ────────────────── Hash backend abstraction ──────────────────

pub trait HashBackend {
    fn name(&self) -> &'static str;
    fn absorb_bytes(&mut self, bytes: &[u8]);
    fn absorb_field(&mut self, x: F);

    /// Squeeze a single field-element challenge (64-bit for Goldilocks).
    fn challenge(&mut self, label: &[u8]) -> F;

    /// Squeeze a 32-byte challenge digest.
    ///
    /// The default implementation derives 32 bytes from four independent
    /// field-element squeezes.  Backends with native ≥256-bit output
    /// (SHA3, Blake3) override this to return the first 32 bytes of their
    /// native digest directly.
    fn challenge_bytes(&mut self, label: &[u8]) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0u8..4 {
            let mut sub_label = label.to_vec();
            sub_label.push(b'/');
            sub_label.push(i);
            let f = self.challenge(&sub_label);
            let le = f.into_bigint().to_bytes_le();
            let n = le.len().min(8);
            out[i as usize * 8..i as usize * 8 + n]
                .copy_from_slice(&le[..n]);
        }
        out
    }
}

// ────────────────── Poseidon backend ──────────────────

pub mod poseidon {
    pub use ::poseidon::*;
}

mod poseidon_backend {
    use super::*;
    use ::poseidon::{permute, PoseidonParams, RATE, T};

    pub struct PoseidonBackend {
        pub(crate) state: [F; T],
        pub(crate) pos: usize,
        pub(crate) params: PoseidonParams,
    }

    impl PoseidonBackend {
        pub fn new(params: PoseidonParams, init_label: &[u8]) -> Self {
            let mut s = Self {
                state: [F::zero(); T],
                pos: 0,
                params,
            };
            s.state[T - 1] = super::domain_tag_to_field(super::ds::TRANSCRIPT_INIT);
            s.absorb_bytes(init_label);
            s
        }

        fn absorb_field_internal(&mut self, x: F) {
            if self.pos == RATE {
                permute(&mut self.state, &self.params);
                self.pos = 0;
            }
            self.state[self.pos] += x;
            self.pos += 1;
        }

        fn squeeze(&mut self) -> F {
            permute(&mut self.state, &self.params);
            self.pos = 0;
            self.state[0]
        }
    }

    impl super::HashBackend for PoseidonBackend {
        fn name(&self) -> &'static str { "poseidon" }

        fn absorb_bytes(&mut self, bytes: &[u8]) {
            self.absorb_field_internal(super::domain_tag_to_field(super::ds::ABSORB_BYTES));
            for w in super::bytes_to_field_words(bytes) {
                self.absorb_field_internal(w);
            }
        }

        fn absorb_field(&mut self, x: F) {
            self.absorb_field_internal(x);
        }

        fn challenge(&mut self, label: &[u8]) -> F {
            self.absorb_field_internal(super::domain_tag_to_field(super::ds::CHALLENGE));
            self.absorb_bytes(label);
            self.squeeze()
        }

        // Poseidon uses the default `challenge_bytes` (4 × squeeze).
    }

    pub fn default_params() -> PoseidonParams {
        ::poseidon::params::generate_params_t17_x5(b"POSEIDON-T17-X5-TRANSCRIPT")
    }

    pub(crate) use PoseidonBackend as Backend;
}

// ────────────────── SHA3 backends (256 / 384 / 512) ──────────────────

mod sha3_backend {
    use super::*;
    use sha3::{Digest, Sha3_256, Sha3_384, Sha3_512};

    /// Generate a SHA3 Fiat–Shamir backend for a specific output width.
    ///
    /// Every generated backend uses the fork-then-finalize pattern: `challenge`
    /// and `challenge_bytes` clone the running hash state, finalize the clone,
    /// and leave the main state untouched.  State advances only through
    /// `absorb_bytes` / `absorb_field`.
    macro_rules! impl_sha3_backend {
        ($(#[$meta:meta])* $name:ident, $sha3:ty, $display:expr) => {
            $(#[$meta])*
            #[derive(Clone)]
            pub struct $name {
                h: $sha3,
            }

            impl $name {
                pub fn new(init_label: &[u8]) -> Self {
                    let mut h = <$sha3>::new();
                    Digest::update(&mut h, super::ds::TRANSCRIPT_INIT);
                    Digest::update(&mut h, init_label);
                    Self { h }
                }
            }

            impl super::HashBackend for $name {
                fn name(&self) -> &'static str { $display }

                fn absorb_bytes(&mut self, bytes: &[u8]) {
                    Digest::update(&mut self.h, super::ds::ABSORB_BYTES);
                    Digest::update(&mut self.h, bytes);
                }

                fn absorb_field(&mut self, x: F) {
                    let le = x.into_bigint().to_bytes_le();
                    self.absorb_bytes(&le[..8.min(le.len())]);
                }

                fn challenge(&mut self, label: &[u8]) -> F {
                    let mut h2 = self.h.clone();
                    Digest::update(&mut h2, super::ds::CHALLENGE);
                    Digest::update(&mut h2, label);
                    let out = h2.finalize();
                    super::bytes_to_field_u64(&out[..8])
                }

                /// Return 32 bytes derived from the native SHA3 digest.
                ///
                /// For SHA3-384 and SHA3-512 the wider native output is
                /// truncated to 32 bytes.  The internal state still benefits
                /// from the full collision-resistance of the chosen variant.
                fn challenge_bytes(&mut self, label: &[u8]) -> [u8; 32] {
                    let mut h2 = self.h.clone();
                    Digest::update(&mut h2, super::ds::CHALLENGE);
                    Digest::update(&mut h2, label);
                    let out = h2.finalize();
                    let mut result = [0u8; 32];
                    result.copy_from_slice(&out[..32]);
                    result
                }
            }
        };
    }

    impl_sha3_backend!(
        /// SHA3-256 Fiat–Shamir backend (128-bit collision resistance).
        Sha3_256Backend, Sha3_256, "sha3-256"
    );

    impl_sha3_backend!(
        /// SHA3-384 Fiat–Shamir backend (192-bit collision resistance).
        Sha3_384Backend, Sha3_384, "sha3-384"
    );

    impl_sha3_backend!(
        /// SHA3-512 Fiat–Shamir backend (256-bit collision resistance).
        Sha3_512Backend, Sha3_512, "sha3-512"
    );
}

// ────────────────── Blake3 backend ──────────────────

mod blake3_backend {
    use super::*;

    #[derive(Clone)]
    pub struct Blake3Backend {
        h: blake3::Hasher,
    }

    impl Blake3Backend {
        pub fn new(init_label: &[u8]) -> Self {
            let mut h = blake3::Hasher::new();
            h.update(super::ds::TRANSCRIPT_INIT);
            h.update(init_label);
            Self { h }
        }
    }

    impl HashBackend for Blake3Backend {
        fn name(&self) -> &'static str { "blake3" }

        fn absorb_bytes(&mut self, bytes: &[u8]) {
            self.h.update(super::ds::ABSORB_BYTES);
            self.h.update(bytes);
        }

        fn absorb_field(&mut self, x: F) {
            let le = x.into_bigint().to_bytes_le();
            self.absorb_bytes(&le[..8.min(le.len())]);
        }

        fn challenge(&mut self, label: &[u8]) -> F {
            let mut h2 = self.h.clone();
            h2.update(super::ds::CHALLENGE);
            h2.update(label);
            let out = h2.finalize();
            bytes_to_field_u64(out.as_bytes())
        }

        fn challenge_bytes(&mut self, label: &[u8]) -> [u8; 32] {
            let mut h2 = self.h.clone();
            h2.update(super::ds::CHALLENGE);
            h2.update(label);
            *h2.finalize().as_bytes()
        }
    }

    pub(crate) use Blake3Backend as Backend;
}

// ────────────────── Public types ──────────────────

/// Selector for the Fiat–Shamir hash backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsHash {
    Poseidon,
    Sha3_256,
    Sha3_384,
    Sha3_512,
    Blake3,
}

pub use poseidon_backend::default_params;

// ────────────────── Transcript API ──────────────────

/// Fiat–Shamir transcript with a dynamically selected hash backend.
pub struct Transcript {
    backend: Box<dyn HashBackend>,
}

impl Transcript {
    // ── Poseidon (backward-compatible default) ───────────────

    /// Create a Poseidon-backed transcript with explicit parameters.
    pub fn new(init_label: &[u8], params: poseidon::PoseidonParams) -> Self {
        emit_selected_backend_once();
        Self {
            backend: Box::new(poseidon_backend::Backend::new(params, init_label)),
        }
    }

    // ── SHA3 family ──────────────────────────────────────────

    /// SHA3-256 transcript.  Equivalent to [`new_sha3_256`](Self::new_sha3_256).
    pub fn new_sha3(init_label: &[u8]) -> Self {
        Self::new_sha3_256(init_label)
    }

    /// SHA3-256 transcript (128-bit collision resistance).
    pub fn new_sha3_256(init_label: &[u8]) -> Self {
        emit_selected_backend_once();
        Self {
            backend: Box::new(sha3_backend::Sha3_256Backend::new(init_label)),
        }
    }

    /// SHA3-384 transcript (192-bit collision resistance).
    pub fn new_sha3_384(init_label: &[u8]) -> Self {
        emit_selected_backend_once();
        Self {
            backend: Box::new(sha3_backend::Sha3_384Backend::new(init_label)),
        }
    }

    /// SHA3-512 transcript (256-bit collision resistance).
    pub fn new_sha3_512(init_label: &[u8]) -> Self {
        emit_selected_backend_once();
        Self {
            backend: Box::new(sha3_backend::Sha3_512Backend::new(init_label)),
        }
    }

    // ── Blake3 ───────────────────────────────────────────────

    /// Blake3 transcript (128-bit collision resistance).
    pub fn new_blake3(init_label: &[u8]) -> Self {
        emit_selected_backend_once();
        Self {
            backend: Box::new(blake3_backend::Backend::new(init_label)),
        }
    }

    // ── Generic constructors ─────────────────────────────────

    /// Select the backend at runtime.
    ///
    /// `params` is used only when `hash == FsHash::Poseidon`; it is ignored
    /// for all other variants.  Prefer [`from_hash`](Self::from_hash) when
    /// Poseidon parameters are not needed.
    pub fn with_backend(
        hash: FsHash,
        init_label: &[u8],
        params: poseidon::PoseidonParams,
    ) -> Self {
        emit_selected_backend_once();

        let backend: Box<dyn HashBackend> = match hash {
            FsHash::Poseidon => {
                Box::new(poseidon_backend::Backend::new(params, init_label))
            }
            FsHash::Sha3_256 => {
                Box::new(sha3_backend::Sha3_256Backend::new(init_label))
            }
            FsHash::Sha3_384 => {
                Box::new(sha3_backend::Sha3_384Backend::new(init_label))
            }
            FsHash::Sha3_512 => {
                Box::new(sha3_backend::Sha3_512Backend::new(init_label))
            }
            FsHash::Blake3 => {
                Box::new(blake3_backend::Backend::new(init_label))
            }
        };

        Self { backend }
    }

    /// Select the backend at runtime without requiring Poseidon parameters.
    ///
    /// When `hash == FsHash::Poseidon`, default parameters are used
    /// (equivalent to [`default_params`]).  For custom Poseidon parameters,
    /// use [`with_backend`](Self::with_backend) or [`new`](Self::new).
    pub fn from_hash(hash: FsHash, init_label: &[u8]) -> Self {
        match hash {
            FsHash::Poseidon => Self::new(init_label, default_params()),
            FsHash::Sha3_256 => Self::new_sha3_256(init_label),
            FsHash::Sha3_384 => Self::new_sha3_384(init_label),
            FsHash::Sha3_512 => Self::new_sha3_512(init_label),
            FsHash::Blake3  => Self::new_blake3(init_label),
        }
    }

    // ── Core operations ──────────────────────────────────────

    #[inline]
    pub fn absorb_bytes(&mut self, bytes: &[u8]) {
        self.backend.absorb_bytes(bytes)
    }

    #[inline]
    pub fn absorb_field(&mut self, x: F) {
        self.backend.absorb_field(x)
    }

    /// Squeeze a field-element challenge (64-bit for Goldilocks).
    #[inline]
    pub fn challenge(&mut self, label: &[u8]) -> F {
        self.backend.challenge(label)
    }

    /// Squeeze a 32-byte challenge digest.
    ///
    /// For SHA3-384 and SHA3-512, the wider native digest is truncated to
    /// 32 bytes.  The internal collision resistance of the running transcript
    /// state still matches the selected variant's full security level.
    #[inline]
    pub fn challenge_bytes(&mut self, label: &[u8]) -> [u8; 32] {
        self.backend.challenge_bytes(label)
    }
}

// ────────────────── Internal ──────────────────

static PRINT_SELECTED_FS_BACKEND: Once = Once::new();

fn emit_selected_backend_once() {
    PRINT_SELECTED_FS_BACKEND.call_once(|| {});
}

// ────────────────── Tests ──────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SHA3 variant isolation ─────────────────────────────

    #[test]
    fn sha3_variants_produce_distinct_challenges() {
        let mut t256 = Transcript::new_sha3_256(b"test");
        let mut t384 = Transcript::new_sha3_384(b"test");
        let mut t512 = Transcript::new_sha3_512(b"test");

        let c256 = t256.challenge(b"alpha");
        let c384 = t384.challenge(b"alpha");
        let c512 = t512.challenge(b"alpha");

        assert_ne!(c256, c384, "SHA3-256 vs SHA3-384 should differ");
        assert_ne!(c256, c512, "SHA3-256 vs SHA3-512 should differ");
        assert_ne!(c384, c512, "SHA3-384 vs SHA3-512 should differ");
    }

    #[test]
    fn sha3_variants_produce_distinct_challenge_bytes() {
        let mut t256 = Transcript::new_sha3_256(b"test");
        let mut t384 = Transcript::new_sha3_384(b"test");
        let mut t512 = Transcript::new_sha3_512(b"test");

        let b256 = t256.challenge_bytes(b"seed");
        let b384 = t384.challenge_bytes(b"seed");
        let b512 = t512.challenge_bytes(b"seed");

        assert_ne!(b256, b384);
        assert_ne!(b256, b512);
        assert_ne!(b384, b512);
    }

    // ── Determinism ────────────────────────────────────────

    #[test]
    fn challenge_is_deterministic() {
        for constructor in [
            Transcript::new_sha3_256 as fn(&[u8]) -> Transcript,
            Transcript::new_sha3_384,
            Transcript::new_sha3_512,
            Transcript::new_blake3,
        ] {
            let mut t1 = constructor(b"det-test");
            let mut t2 = constructor(b"det-test");

            t1.absorb_field(F::from(42u64));
            t2.absorb_field(F::from(42u64));

            assert_eq!(t1.challenge(b"r1"), t2.challenge(b"r1"));
            assert_eq!(
                t1.challenge_bytes(b"r2"),
                t2.challenge_bytes(b"r2")
            );
        }
    }

    // ── Absorb advances state ──────────────────────────────

    #[test]
    fn absorb_changes_subsequent_challenge() {
        let mut t1 = Transcript::new_sha3_384(b"test");
        let mut t2 = Transcript::new_sha3_384(b"test");

        t1.absorb_field(F::from(1u64));
        t2.absorb_field(F::from(2u64));

        assert_ne!(t1.challenge(b"c"), t2.challenge(b"c"));
    }

    // ── new_sha3 backward compatibility ────────────────────

    #[test]
    fn new_sha3_is_sha3_256() {
        let mut old = Transcript::new_sha3(b"compat");
        let mut explicit = Transcript::new_sha3_256(b"compat");

        old.absorb_field(F::from(7u64));
        explicit.absorb_field(F::from(7u64));

        assert_eq!(old.challenge(b"x"), explicit.challenge(b"x"));
    }

    // ── from_hash matches direct constructors ──────────────

    #[test]
    fn from_hash_matches_direct_constructors() {
        for (hash, ctor) in [
            (FsHash::Sha3_256, Transcript::new_sha3_256 as fn(&[u8]) -> Transcript),
            (FsHash::Sha3_384, Transcript::new_sha3_384),
            (FsHash::Sha3_512, Transcript::new_sha3_512),
            (FsHash::Blake3, Transcript::new_blake3),
        ] {
            let mut t_from = Transcript::from_hash(hash, b"match");
            let mut t_direct = ctor(b"match");

            t_from.absorb_field(F::from(99u64));
            t_direct.absorb_field(F::from(99u64));

            assert_eq!(
                t_from.challenge(b"q"),
                t_direct.challenge(b"q"),
                "from_hash({:?}) should match direct constructor",
                hash,
            );
        }
    }

    // ── Backend names ──────────────────────────────────────

    #[test]
    fn backend_names_are_correct() {
        use sha3_backend::*;
        use blake3_backend::Blake3Backend;

        assert_eq!(HashBackend::name(&Sha3_256Backend::new(b"")), "sha3-256");
        assert_eq!(HashBackend::name(&Sha3_384Backend::new(b"")), "sha3-384");
        assert_eq!(HashBackend::name(&Sha3_512Backend::new(b"")), "sha3-512");
        assert_eq!(HashBackend::name(&Blake3Backend::new(b"")), "blake3");
    }

    // ── Different labels produce different challenges ──────

    #[test]
    fn different_labels_yield_different_challenges() {
        let mut t = Transcript::new_sha3_512(b"label-test");
        t.absorb_field(F::from(1u64));

        let c1 = t.challenge(b"alpha");
        let c2 = t.challenge(b"beta");
        assert_ne!(c1, c2);
    }
}