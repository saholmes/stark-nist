//! Pluggable Fiat–Shamir transcript with backward-compatible API.
//! Goldilocks-safe field embedding (64-bit).
#![allow(dead_code)]
#![allow(unused_imports)]

use ark_ff::{BigInteger, PrimeField, Zero};
use ark_goldilocks::Goldilocks as F;
use std::sync::Once;

// ---------------- Domain separation tags ----------------

pub mod ds {
    pub const TRANSCRIPT_INIT: &[u8] = b"FSv1-TRANSCRIPT-INIT";
    pub const ABSORB_BYTES: &[u8] = b"FSv1-ABSORB-BYTES";
    pub const CHALLENGE: &[u8] = b"FSv1-CHALLENGE";
}

// ---------------- Helpers (Goldilocks-safe) ----------------

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

// ---------------- Hash backend abstraction ----------------

pub trait HashBackend {
    fn name(&self) -> &'static str;
    fn absorb_bytes(&mut self, bytes: &[u8]);
    fn absorb_field(&mut self, x: F);

    /// Squeeze a single field-element challenge (64-bit for Goldilocks).
    fn challenge(&mut self, label: &[u8]) -> F;

    /// ✅ Squeeze a full 32-byte challenge digest.
    ///
    /// The default implementation derives 32 bytes from four independent
    /// field-element squeezes.  Backends with native ≥256-bit output
    /// (SHA3, Blake3) override this to return the raw digest.
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

// ---------------- Poseidon backend ----------------

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

        // Poseidon uses the default challenge_bytes (4 × squeeze)
    }

    pub fn default_params() -> PoseidonParams {
        ::poseidon::params::generate_params_t17_x5(b"POSEIDON-T17-X5-TRANSCRIPT")
    }

    pub fn make(params: PoseidonParams, label: &[u8]) -> PoseidonBackend {
        PoseidonBackend::new(params, label)
    }

    pub(crate) use PoseidonBackend as Backend;
}

// ---------------- SHA3 backend ----------------

mod sha3_backend {
    use super::*;
    use sha3::{Digest, Sha3_256};

    #[derive(Clone)]
    pub struct Sha3Backend {
        h: Sha3_256,
    }

    impl Sha3Backend {
        pub fn new(init_label: &[u8]) -> Self {
            let mut h = Sha3_256::new();
            h.update(super::ds::TRANSCRIPT_INIT);
            h.update(init_label);
            Self { h }
        }
    }

    impl HashBackend for Sha3Backend {
        fn name(&self) -> &'static str { "sha3-256" }

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
            bytes_to_field_u64(&out[..8])
        }

        /// ✅ Override: return the full 32-byte SHA3-256 digest.
        fn challenge_bytes(&mut self, label: &[u8]) -> [u8; 32] {
            let mut h2 = self.h.clone();
            h2.update(super::ds::CHALLENGE);
            h2.update(label);
            h2.finalize().into()
        }
    }

    pub fn make(label: &[u8]) -> Sha3Backend {
        Sha3Backend::new(label)
    }

    pub(crate) use Sha3Backend as Backend;
}

// ---------------- Blake3 backend ----------------

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

        /// ✅ Override: return the full 32-byte Blake3 digest.
        fn challenge_bytes(&mut self, label: &[u8]) -> [u8; 32] {
            let mut h2 = self.h.clone();
            h2.update(super::ds::CHALLENGE);
            h2.update(label);
            *h2.finalize().as_bytes()
        }
    }

    pub fn make(label: &[u8]) -> Blake3Backend {
        Blake3Backend::new(label)
    }

    pub(crate) use Blake3Backend as Backend;
}

// ---------------- Public Transcript API ----------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsHash {
    Poseidon,
    Sha3_256,
    Blake3,
}

pub use poseidon_backend::default_params;

/// ✅ Backward-compatible Transcript wrapper
pub struct Transcript {
    backend: Box<dyn HashBackend>,
}

impl Transcript {
    /// Default = Poseidon (backward compat)
    pub fn new(init_label: &[u8], params: poseidon::PoseidonParams) -> Self {
        Self::with_backend(FsHash::Poseidon, init_label, params)
    }

    /// ✅ NEW: SHA3-256 transcript — no PoseidonParams required.
    pub fn new_sha3(init_label: &[u8]) -> Self {
        emit_selected_backend_once();
        Self {
            backend: Box::new(sha3_backend::Backend::new(init_label)),
        }
    }

    /// ✅ NEW: Blake3 transcript — no PoseidonParams required.
    pub fn new_blake3(init_label: &[u8]) -> Self {
        emit_selected_backend_once();
        Self {
            backend: Box::new(blake3_backend::Backend::new(init_label)),
        }
    }

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
                Box::new(sha3_backend::Backend::new(init_label))
            }
            FsHash::Blake3 => {
                Box::new(blake3_backend::Backend::new(init_label))
            }
        };

        Self { backend }
    }

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

    /// ✅ Squeeze a full 32-byte challenge digest.
    #[inline]
    pub fn challenge_bytes(&mut self, label: &[u8]) -> [u8; 32] {
        self.backend.challenge_bytes(label)
    }
}

// ---------------- Internal ----------------

static PRINT_SELECTED_FS_BACKEND: Once = Once::new();

fn emit_selected_backend_once() {
    PRINT_SELECTED_FS_BACKEND.call_once(|| {});
}