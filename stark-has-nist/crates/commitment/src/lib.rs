use ark_ff::{PrimeField, Zero};
use ark_goldilocks::Goldilocks as F;

use poseidon::{
    permute,
    params::generate_params_t17_x5,
    PoseidonParams,
    T,
};

use sha3::{Sha3_256, Sha3_384, Sha3_512, Digest};

// ────────────────────────────────────────────────────────────────────────
//  SHA3 variant selection
// ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaVariant {
    Sha3_256,
    Sha3_384,
    Sha3_512,
}

impl ShaVariant {
    pub fn output_len(self) -> usize {
        match self {
            ShaVariant::Sha3_256 => 32,
            ShaVariant::Sha3_384 => 48,
            ShaVariant::Sha3_512 => 64,
        }
    }
}

/// Hash a sequence of byte slices with the selected SHA3 variant.
fn sha3_hash(variant: ShaVariant, parts: &[&[u8]]) -> Vec<u8> {
    fn digest_parts<D: Digest>(parts: &[&[u8]]) -> Vec<u8> {
        let mut h = D::new();
        for p in parts {
            Digest::update(&mut h, p);
        }
        h.finalize().to_vec()
    }

    match variant {
        ShaVariant::Sha3_256 => digest_parts::<Sha3_256>(parts),
        ShaVariant::Sha3_384 => digest_parts::<Sha3_384>(parts),
        ShaVariant::Sha3_512 => digest_parts::<Sha3_512>(parts),
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Dual commitment
// ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DualCommitment {
    pub sha_commit: Vec<u8>,
    pub poseidon_root: F,
    pub trace_hash: Vec<u8>,
    pub sha_variant: ShaVariant,
}

// ────────────────────────────────────────────────────────────────────────
//  Merkle commitment
// ────────────────────────────────────────────────────────────────────────

pub struct MerkleCommitment {
    pub arity: usize,
    pub params: PoseidonParams,
    pub sha_variant: ShaVariant,
}

impl MerkleCommitment {
    pub fn with_default_params() -> Self {
        Self::new(ShaVariant::Sha3_256)
    }

    pub fn new(sha_variant: ShaVariant) -> Self {
        let seed = b"POSEIDON-T17-X5-SEED";
        let params = generate_params_t17_x5(seed);
        Self {
            arity: 16,
            params,
            sha_variant,
        }
    }

    // ────────────────────────────────────────────────────────────
    //  Field encoding
    // ────────────────────────────────────────────────────────────

    #[inline]
    fn field_to_bytes(x: &F) -> [u8; 8] {
        let limb0 = x.into_bigint().0[0];
        limb0.to_le_bytes()
    }

    fn encode_trace_flat(trace: &[Vec<F>]) -> Vec<u8> {
        let mut out = Vec::new();
        for row in trace {
            for x in row {
                out.extend_from_slice(&Self::field_to_bytes(x));
            }
        }
        out
    }

    // ────────────────────────────────────────────────────────────
    //  SHA3 helpers (dispatch through sha3_hash)
    // ────────────────────────────────────────────────────────────

    fn sha3_trace(&self, trace: &[Vec<F>]) -> Vec<u8> {
        let flat = Self::encode_trace_flat(trace);
        sha3_hash(self.sha_variant, &[b"TRACE_HASH_V1", &flat])
    }

    fn sha3_commit(&self, trace: &[Vec<F>], trace_hash: &[u8]) -> Vec<u8> {
        let flat = Self::encode_trace_flat(trace);
        sha3_hash(
            self.sha_variant,
            &[b"TRACE_BYTES_COMMIT_V1", trace_hash, &flat],
        )
    }

    // ────────────────────────────────────────────────────────────
    //  Poseidon sponge bound to trace_hash
    // ────────────────────────────────────────────────────────────

    fn poseidon_hash_with_ds(
        inputs: &[F],
        params: &PoseidonParams,
        trace_hash: &[u8],
    ) -> F {
        let mut state = [F::zero(); T];

        // Domain separation: first 8 bytes of trace_hash, regardless
        // of SHA3 output length.
        let mut ds_bytes = [0u8; 8];
        let copy_len = trace_hash.len().min(8);
        ds_bytes[..copy_len].copy_from_slice(&trace_hash[..copy_len]);
        state[T - 1] = F::from(u64::from_le_bytes(ds_bytes));

        for chunk in inputs.chunks(T - 1) {
            for (i, &x) in chunk.iter().enumerate() {
                state[i] += x;
            }
            permute(&mut state, params);
        }

        state[0]
    }

    // ────────────────────────────────────────────────────────────
    //  Poseidon Merkle commitment
    // ────────────────────────────────────────────────────────────

    pub fn commit(&self, trace: &[Vec<F>]) -> F {
        let trace_hash = self.sha3_trace(trace);
        self.commit_with_hash(trace, &trace_hash)
    }

    fn commit_with_hash(&self, trace: &[Vec<F>], trace_hash: &[u8]) -> F {
        // Hash each row directly from fields — no bytes round-trip.
        let mut level: Vec<F> = trace
            .iter()
            .map(|row| {
                Self::poseidon_hash_with_ds(row, &self.params, trace_hash)
            })
            .collect();

        while level.len() > 1 {
            let mut next = Vec::new();
            for chunk in level.chunks(self.arity) {
                let parent =
                    Self::poseidon_hash_with_ds(chunk, &self.params, trace_hash);
                next.push(parent);
            }
            level = next;
        }

        level[0]
    }

    // ────────────────────────────────────────────────────────────
    //  Dual commitment (SHA3 + Poseidon)
    // ────────────────────────────────────────────────────────────

    pub fn dual_commit(&self, trace: &[Vec<F>]) -> DualCommitment {
        let trace_hash = self.sha3_trace(trace);
        let sha_commit = self.sha3_commit(trace, &trace_hash);
        let poseidon_root = self.commit_with_hash(trace, &trace_hash);

        DualCommitment {
            sha_commit,
            poseidon_root,
            trace_hash,
            sha_variant: self.sha_variant,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merkle_commit_roundtrip() {
        let mc = MerkleCommitment::with_default_params();

        let trace = vec![
            vec![F::from(1u64), F::from(2u64), F::from(3u64)],
            vec![F::from(4u64), F::from(5u64), F::from(6u64)],
            vec![F::from(7u64), F::from(8u64), F::from(9u64)],
            vec![F::from(10u64), F::from(11u64), F::from(12u64)],
        ];

        let root1 = mc.commit(&trace);
        let root2 = mc.commit(&trace);

        assert_eq!(root1, root2);
    }

    #[test]
    fn dual_commit_deterministic() {
        let mc = MerkleCommitment::with_default_params();

        let trace = vec![
            vec![F::from(42u64)],
            vec![F::from(7u64)],
        ];

        let c1 = mc.dual_commit(&trace);
        let c2 = mc.dual_commit(&trace);

        assert_eq!(c1, c2);
    }

    #[test]
    fn poseidon_commit_binds_trace_hash() {
        let mc = MerkleCommitment::with_default_params();

        let t1 = vec![vec![F::from(1u64)]];
        let t2 = vec![vec![F::from(2u64)]];

        let c1 = mc.dual_commit(&t1);
        let c2 = mc.dual_commit(&t2);

        assert_ne!(c1.poseidon_root, c2.poseidon_root);
        assert_ne!(c1.trace_hash, c2.trace_hash);
    }

    #[test]
    fn sha3_384_produces_longer_output() {
        let mc = MerkleCommitment::new(ShaVariant::Sha3_384);

        let trace = vec![vec![F::from(1u64)]];
        let c = mc.dual_commit(&trace);

        assert_eq!(c.sha_commit.len(), 48);
        assert_eq!(c.trace_hash.len(), 48);
        assert_eq!(c.sha_variant, ShaVariant::Sha3_384);
    }

    #[test]
    fn sha3_512_produces_longer_output() {
        let mc = MerkleCommitment::new(ShaVariant::Sha3_512);

        let trace = vec![vec![F::from(1u64)]];
        let c = mc.dual_commit(&trace);

        assert_eq!(c.sha_commit.len(), 64);
        assert_eq!(c.trace_hash.len(), 64);
        assert_eq!(c.sha_variant, ShaVariant::Sha3_512);
    }

    #[test]
    fn different_variants_produce_different_commits() {
        let trace = vec![vec![F::from(99u64), F::from(100u64)]];

        let c256 = MerkleCommitment::new(ShaVariant::Sha3_256).dual_commit(&trace);
        let c384 = MerkleCommitment::new(ShaVariant::Sha3_384).dual_commit(&trace);
        let c512 = MerkleCommitment::new(ShaVariant::Sha3_512).dual_commit(&trace);

        // SHA commits differ (different hash functions)
        assert_ne!(c256.sha_commit, c384.sha_commit[..32]);
        assert_ne!(c384.sha_commit, c512.sha_commit[..48]);

        // Poseidon roots also differ because trace_hash differs,
        // which changes the domain separator fed into the sponge.
        assert_ne!(c256.poseidon_root, c384.poseidon_root);
        assert_ne!(c384.poseidon_root, c512.poseidon_root);
    }

    #[test]
    fn with_default_params_is_sha3_256() {
        let mc = MerkleCommitment::with_default_params();
        assert_eq!(mc.sha_variant, ShaVariant::Sha3_256);

        let trace = vec![vec![F::from(1u64)]];
        let c = mc.dual_commit(&trace);
        assert_eq!(c.sha_commit.len(), 32);
    }
}