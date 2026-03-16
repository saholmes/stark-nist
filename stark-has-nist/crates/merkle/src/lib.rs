//! SHA3 Merkle tree for STARK polynomial commitments.
//!
//! Multi-arity tree with configurable digest size (SHA3-256 / SHA3-384 /
//! SHA3-512) and domain-separated hashing at every node.
//!
//! Leaf hashes use a distinguished level sentinel (`LEAF_LEVEL_DS = u32::MAX`)
//! so that leaf and interior hashes are never domain-equivalent, providing
//! structural second-preimage resistance.

use ark_ff::{BigInteger, PrimeField};
use ark_goldilocks::Goldilocks;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha3::{Digest, Sha3_256, Sha3_384, Sha3_512};

use std::fmt;

// ═══════════════════════════════════════════════════════════════════════
//  Errors
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MerkleError {
    /// No leaves were pushed before finalization.
    EmptyTree,
    /// Attempted to open or query root before `finalize()`.
    NotFinalized,
    /// Leaf index exceeds the number of committed leaves.
    IndexOutOfBounds { index: usize, leaf_count: usize },
    /// A proof level contains fewer siblings than the arity requires.
    InsufficientSiblings { level: usize },
}

impl fmt::Display for MerkleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTree => write!(f, "cannot finalize an empty Merkle tree"),
            Self::NotFinalized => write!(f, "tree has not been finalized"),
            Self::IndexOutOfBounds { index, leaf_count } => {
                write!(f, "leaf index {index} out of bounds ({leaf_count} leaves)")
            }
            Self::InsufficientSiblings { level } => {
                write!(f, "proof has insufficient siblings at level {level}")
            }
        }
    }
}

impl std::error::Error for MerkleError {}

// ═══════════════════════════════════════════════════════════════════════
//  Field-element serialization
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerFr(pub Goldilocks);

impl From<Goldilocks> for SerFr {
    fn from(x: Goldilocks) -> Self {
        SerFr(x)
    }
}

impl From<SerFr> for Goldilocks {
    fn from(w: SerFr) -> Goldilocks {
        w.0
    }
}

impl Serialize for SerFr {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&field_to_bytes(&self.0))
    }
}

impl<'de> Deserialize<'de> for SerFr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        if bytes.len() != 8 {
            return Err(serde::de::Error::custom(format!(
                "expected exactly 8 bytes for Goldilocks element, got {}",
                bytes.len()
            )));
        }
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes);
        Ok(SerFr(bytes_to_field(&buf)))
    }
}

/// Canonical little-endian encoding of a Goldilocks field element.
pub fn field_to_bytes(field: &Goldilocks) -> [u8; 8] {
    let mut out = [0u8; 8];
    let bytes = field.into_bigint().to_bytes_le();
    let n = bytes.len().min(8);
    out[..n].copy_from_slice(&bytes[..n]);
    out
}

/// Decode 8 little-endian bytes into a Goldilocks field element.
pub fn bytes_to_field(bytes: &[u8; 8]) -> Goldilocks {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as u64) << (i * 8);
    }
    Goldilocks::from(v)
}

// ═══════════════════════════════════════════════════════════════════════
//  Domain separation
// ═══════════════════════════════════════════════════════════════════════

/// Sentinel level value for leaf hashes, ensuring leaves and interior nodes
/// are never domain-equivalent.
pub const LEAF_LEVEL_DS: u32 = u32::MAX;

/// Domain-separation label prepended to every hash invocation.
///
/// Layout (32 bytes, all little-endian):
///   `[0.. 8)` arity,
///   `[8..16)` level (`LEAF_LEVEL_DS` for leaves),
///   `[16..24)` position within the level,
///   `[24..32)` tree label (distinguishes independent trees).
#[derive(Clone, Copy, Debug)]
pub struct DsLabel {
    pub arity: usize,
    pub level: u32,
    pub position: u64,
    pub tree_label: u64,
}

impl DsLabel {
    pub fn to_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0..8].copy_from_slice(&(self.arity as u64).to_le_bytes());
        out[8..16].copy_from_slice(&(self.level as u64).to_le_bytes());
        out[16..24].copy_from_slice(&self.position.to_le_bytes());
        out[24..32].copy_from_slice(&self.tree_label.to_le_bytes());
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Configuration (hash-independent)
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for a multi-arity Merkle tree.
///
/// `layer_arities[i]` is the fan-in at level `i` (level 0 = leaf level).
/// If the tree grows deeper than `layer_arities.len()`, the last entry is
/// reused for all remaining levels.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleConfig {
    pub layer_arities: Vec<usize>,
    pub tree_label: u64,
}

impl MerkleConfig {
    pub fn new(layer_arities: Vec<usize>, tree_label: u64) -> Self {
        assert!(!layer_arities.is_empty(), "at least one arity is required");
        assert!(
            layer_arities.iter().all(|&a| a >= 2),
            "every arity must be >= 2"
        );
        Self { layer_arities, tree_label }
    }

    /// Convenience: binary (arity-2) tree.
    pub fn binary(tree_label: u64) -> Self {
        Self::new(vec![2], tree_label)
    }

    /// Returns the arity for a given level, clamping to the last entry.
    fn arity_at(&self, level: usize) -> usize {
        self.layer_arities[level.min(self.layer_arities.len() - 1)]
    }
}

/// Backward-compatible alias.
pub type MerkleChannelCfg = MerkleConfig;

// ═══════════════════════════════════════════════════════════════════════
//  MerkleHasher trait
// ═══════════════════════════════════════════════════════════════════════

/// Abstraction over the hash function used inside the Merkle tree.
///
/// Implement this trait to plug in a new hash family.  Three SHA3 variants
/// are provided out of the box: [`Sha3_256Hasher`], [`Sha3_384Hasher`],
/// and [`Sha3_512Hasher`].
pub trait MerkleHasher: Clone + Send + Sync + 'static {
    /// Digest length in bytes (e.g. 32, 48, 64).
    const DIGEST_SIZE: usize;

    /// The concrete fixed-size digest type.
    type Digest: Copy
        + PartialEq
        + Eq
        + fmt::Debug
        + Default
        + AsRef<[u8]>
        + Serialize
        + DeserializeOwned
        + Send
        + Sync;

    /// Hash field-element leaf data with domain separation.
    fn hash_leaf(ds: DsLabel, values: &[Goldilocks]) -> Self::Digest;

    /// Hash child digests into a parent digest with domain separation.
    fn hash_node(ds: DsLabel, children: &[Self::Digest]) -> Self::Digest;
}

// ═══════════════════════════════════════════════════════════════════════
//  SHA3 hasher implementations
// ═══════════════════════════════════════════════════════════════════════

macro_rules! impl_sha3_hasher {
    ($(#[$meta:meta])* $name:ident, $sha3:ty, $size:literal) => {
        $(#[$meta])*
        #[derive(Clone, Debug)]
        pub struct $name;

        impl MerkleHasher for $name {
            const DIGEST_SIZE: usize = $size;
            type Digest = [u8; $size];

            fn hash_leaf(ds: DsLabel, values: &[Goldilocks]) -> Self::Digest {
                let mut h = <$sha3>::new();
                Digest::update(&mut h, ds.to_bytes());
                for v in values {
                    Digest::update(&mut h, field_to_bytes(v));
                }
                let result = h.finalize();
                let mut out = [0u8; $size];
                out.copy_from_slice(&result[..]);
                out
            }

            fn hash_node(ds: DsLabel, children: &[Self::Digest]) -> Self::Digest {
                let mut h = <$sha3>::new();
                Digest::update(&mut h, ds.to_bytes());
                for c in children {
                    Digest::update(&mut h, c);
                }
                let result = h.finalize();
                let mut out = [0u8; $size];
                out.copy_from_slice(&result[..]);
                out
            }
        }
    };
}

impl_sha3_hasher!(
    /// SHA3-256 Merkle hasher — 32-byte digests (128-bit collision resistance).
    Sha3_256Hasher, Sha3_256, 32
);

impl_sha3_hasher!(
    /// SHA3-384 Merkle hasher — 48-byte digests (192-bit collision resistance).
    Sha3_384Hasher, Sha3_384, 48
);

impl_sha3_hasher!(
    /// SHA3-512 Merkle hasher — 64-byte digests (256-bit collision resistance).
    Sha3_512Hasher, Sha3_512, 64
);

// ═══════════════════════════════════════════════════════════════════════
//  Merkle opening (proof)
// ═══════════════════════════════════════════════════════════════════════

/// A Merkle authentication path, generic over the hash function.
///
/// `path[i]` contains the `arity − 1` sibling digests at level `i`,
/// ordered left-to-right with the opener's own position omitted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MerkleOpening<H: MerkleHasher> {
    /// Full leaf digest.
    pub leaf: H::Digest,
    /// Sibling digests at each level (leaves → root).
    pub path: Vec<Vec<H::Digest>>,
    /// Leaf index in the original ordering.
    pub index: usize,
}

impl<H: MerkleHasher> MerkleOpening<H> {
    /// Total wire size in bytes (for proof-size accounting).
    pub fn byte_size(&self) -> usize {
        H::DIGEST_SIZE
            + 8 // index as u64
            + self
                .path
                .iter()
                .map(|siblings| H::DIGEST_SIZE * siblings.len())
                .sum::<usize>()
    }

    /// Verify this opening against a root digest and configuration.
    pub fn verify(
        &self,
        cfg: &MerkleConfig,
        root: &H::Digest,
    ) -> Result<bool, MerkleError> {
        let mut cur = self.leaf;
        let mut idx = self.index;

        for (level, siblings) in self.path.iter().enumerate() {
            let arity = cfg.arity_at(level);
            let pos_in_group = idx % arity;

            let mut children: Vec<H::Digest> = Vec::with_capacity(arity);
            let mut sib_iter = siblings.iter();

            for i in 0..arity {
                if i == pos_in_group {
                    children.push(cur);
                } else {
                    match sib_iter.next() {
                        Some(&s) => children.push(s),
                        None => return Err(MerkleError::InsufficientSiblings { level }),
                    }
                }
            }

            let ds = DsLabel {
                arity,
                level: level as u32 + 1,
                position: (idx / arity) as u64,
                tree_label: cfg.tree_label,
            };

            cur = H::hash_node(ds, &children);
            idx /= arity;
        }

        Ok(cur == *root)
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Merkle tree
// ═══════════════════════════════════════════════════════════════════════

/// SHA3 Merkle tree with domain-separated, multi-arity hashing.
///
/// Generic over the hash function `H`: use [`Sha3_256Hasher`],
/// [`Sha3_384Hasher`], or [`Sha3_512Hasher`], or supply your own
/// implementation of [`MerkleHasher`].
///
/// # Lifecycle
///
/// 1. Create with [`MerkleTree::new`].
/// 2. Push leaves with [`push_leaf`](MerkleTree::push_leaf).
/// 3. Call [`finalize`](MerkleTree::finalize) (idempotent).
/// 4. Call [`open`](MerkleTree::open) to produce authentication paths.
/// 5. Ship `(root, MerkleOpening)` to the verifier.
pub struct MerkleTree<H: MerkleHasher> {
    cfg: MerkleConfig,
    /// `levels[0]` = leaf digests … `levels[last]` = single root digest.
    levels: Vec<Vec<H::Digest>>,
    /// Guard: set after interior nodes are built.
    finalized: bool,
}

impl<H: MerkleHasher> MerkleTree<H> {
    /// Create an empty tree with the given configuration.
    pub fn new(cfg: MerkleConfig) -> Self {
        Self {
            cfg,
            levels: vec![Vec::new()],
            finalized: false,
        }
    }

    /// Convenience: commit all leaf rows at once and finalize.
    pub fn from_field_rows(
        cfg: MerkleConfig,
        rows: &[impl AsRef<[Goldilocks]>],
    ) -> Result<Self, MerkleError> {
        if rows.is_empty() {
            return Err(MerkleError::EmptyTree);
        }
        let mut tree = Self::new(cfg);
        for row in rows {
            tree.push_leaf(row.as_ref());
        }
        tree.finalize()?;
        Ok(tree)
    }

    /// Borrow the tree's configuration.
    pub fn config(&self) -> &MerkleConfig {
        &self.cfg
    }

    /// Number of committed leaves.
    pub fn leaf_count(&self) -> usize {
        self.levels[0].len()
    }

    /// Number of levels including leaves and root.  Valid after finalization.
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Append a leaf composed of one or more field elements.
    ///
    /// # Panics
    ///
    /// Panics if the tree has already been finalized.
    pub fn push_leaf(&mut self, values: &[Goldilocks]) {
        assert!(!self.finalized, "cannot push leaves after finalization");

        let idx = self.levels[0].len();
        let ds = DsLabel {
            arity: self.cfg.arity_at(0),
            level: LEAF_LEVEL_DS,
            position: idx as u64,
            tree_label: self.cfg.tree_label,
        };

        self.levels[0].push(H::hash_leaf(ds, values));
    }

    /// Build interior nodes and return the root digest.
    ///
    /// Idempotent: a second call returns the cached root without rebuilding.
    ///
    /// Leaves whose count is not a multiple of the arity are padded by
    /// repeating the last digest (same rule at every level).
    pub fn finalize(&mut self) -> Result<H::Digest, MerkleError> {
        if self.finalized {
            return Ok(self.levels.last().unwrap()[0]);
        }

        if self.levels[0].is_empty() {
            return Err(MerkleError::EmptyTree);
        }

        let mut level = 0;
        while self.levels[level].len() > 1 {
            let arity = self.cfg.arity_at(level);
            let cur = &self.levels[level];

            let padded_len = ((cur.len() + arity - 1) / arity) * arity;
            let last = *cur.last().unwrap();
            let padded: Vec<H::Digest> = cur
                .iter()
                .copied()
                .chain(std::iter::repeat(last).take(padded_len - cur.len()))
                .collect();

            let parents: Vec<H::Digest> = padded
                .chunks_exact(arity)
                .enumerate()
                .map(|(i, chunk)| {
                    let ds = DsLabel {
                        arity,
                        level: level as u32 + 1,
                        position: i as u64,
                        tree_label: self.cfg.tree_label,
                    };
                    H::hash_node(ds, chunk)
                })
                .collect();

            self.levels.push(parents);
            level += 1;
        }

        self.finalized = true;
        Ok(self.levels.last().unwrap()[0])
    }

    /// Return the root digest.  Requires prior finalization.
    pub fn root(&self) -> Result<H::Digest, MerkleError> {
        if !self.finalized {
            return Err(MerkleError::NotFinalized);
        }
        Ok(self.levels.last().unwrap()[0])
    }

    /// Produce an opening proof for the leaf at `index`.
    pub fn open(&self, index: usize) -> Result<MerkleOpening<H>, MerkleError> {
        if !self.finalized {
            return Err(MerkleError::NotFinalized);
        }

        let leaf_count = self.levels[0].len();
        if index >= leaf_count {
            return Err(MerkleError::IndexOutOfBounds { index, leaf_count });
        }

        let mut idx = index;
        let mut path = Vec::with_capacity(self.levels.len() - 1);

        for level in 0..self.levels.len() - 1 {
            let nodes = &self.levels[level];
            let arity = self.cfg.arity_at(level);
            let group_start = (idx / arity) * arity;

            let mut siblings = Vec::with_capacity(arity - 1);
            for i in 0..arity {
                let pos = group_start + i;
                if pos == idx {
                    continue;
                }
                // Mirror the padding rule used in finalize().
                let digest = if pos < nodes.len() {
                    nodes[pos]
                } else {
                    *nodes.last().unwrap()
                };
                siblings.push(digest);
            }

            path.push(siblings);
            idx /= arity;
        }

        Ok(MerkleOpening {
            leaf: self.levels[0][index],
            path,
            index,
        })
    }

    /// Static verification helper (delegates to [`MerkleOpening::verify`]).
    ///
    /// Provided for callers that prefer calling verification on the tree type.
    pub fn verify_opening(
        cfg: &MerkleConfig,
        root: &H::Digest,
        opening: &MerkleOpening<H>,
    ) -> Result<bool, MerkleError> {
        opening.verify(cfg, root)
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Public helpers
// ═══════════════════════════════════════════════════════════════════════

/// Recompute the leaf digest for raw field values at a given index.
///
/// This lets the FRI verifier bind payload values to the Merkle leaf
/// without duplicating domain-separation logic.
pub fn compute_leaf_hash<H: MerkleHasher>(
    cfg: &MerkleConfig,
    index: usize,
    values: &[Goldilocks],
) -> H::Digest {
    let ds = DsLabel {
        arity: cfg.arity_at(0),
        level: LEAF_LEVEL_DS,
        position: index as u64,
        tree_label: cfg.tree_label,
    };
    H::hash_leaf(ds, values)
}

// ═══════════════════════════════════════════════════════════════════════
//  Convenience type aliases
// ═══════════════════════════════════════════════════════════════════════

pub type MerkleTree256 = MerkleTree<Sha3_256Hasher>;
pub type MerkleTree384 = MerkleTree<Sha3_384Hasher>;
pub type MerkleTree512 = MerkleTree<Sha3_512Hasher>;

pub type MerkleOpening256 = MerkleOpening<Sha3_256Hasher>;
pub type MerkleOpening384 = MerkleOpening<Sha3_384Hasher>;
pub type MerkleOpening512 = MerkleOpening<Sha3_512Hasher>;

/// Backward-compatible alias.  Prefer [`MerkleTree256`].
pub type MerkleTreeChannel = MerkleTree256;

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn gl(v: u64) -> Goldilocks {
        Goldilocks::from(v)
    }

    // ── construction & finalization ──────────────────────────────

    #[test]
    fn empty_tree_returns_error() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree256::new(cfg);
        assert_eq!(tree.finalize().unwrap_err(), MerkleError::EmptyTree);
    }

    fn single_leaf_inner<H: MerkleHasher>() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree::<H>::new(cfg.clone());
        tree.push_leaf(&[gl(42)]);
        let root = tree.finalize().unwrap();
        let expected = compute_leaf_hash::<H>(&cfg, 0, &[gl(42)]);
        assert_eq!(root, expected);
    }

    #[test]
    fn single_leaf_256() { single_leaf_inner::<Sha3_256Hasher>(); }
    #[test]
    fn single_leaf_384() { single_leaf_inner::<Sha3_384Hasher>(); }
    #[test]
    fn single_leaf_512() { single_leaf_inner::<Sha3_512Hasher>(); }

    #[test]
    fn finalize_is_idempotent() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree384::new(cfg);
        tree.push_leaf(&[gl(1)]);
        tree.push_leaf(&[gl(2)]);
        let r1 = tree.finalize().unwrap();
        let r2 = tree.finalize().unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    #[should_panic(expected = "cannot push leaves after finalization")]
    fn push_after_finalize_panics() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree256::new(cfg);
        tree.push_leaf(&[gl(1)]);
        tree.finalize().unwrap();
        tree.push_leaf(&[gl(2)]);
    }

    // ── open / verify roundtrips ────────────────────────────────

    fn binary_power_of_two_inner<H: MerkleHasher>() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree::<H>::new(cfg.clone());
        for i in 0..8u64 {
            tree.push_leaf(&[gl(i)]);
        }
        let root = tree.finalize().unwrap();
        for i in 0..8 {
            let proof = tree.open(i).unwrap();
            assert!(
                proof.verify(&cfg, &root).unwrap(),
                "proof for leaf {i} should verify"
            );
        }
    }

    #[test]
    fn binary_pow2_256() { binary_power_of_two_inner::<Sha3_256Hasher>(); }
    #[test]
    fn binary_pow2_384() { binary_power_of_two_inner::<Sha3_384Hasher>(); }
    #[test]
    fn binary_pow2_512() { binary_power_of_two_inner::<Sha3_512Hasher>(); }

    fn binary_non_power_of_two_inner<H: MerkleHasher>() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree::<H>::new(cfg.clone());
        for i in 0..5u64 {
            tree.push_leaf(&[gl(i)]);
        }
        let root = tree.finalize().unwrap();
        for i in 0..5 {
            let proof = tree.open(i).unwrap();
            assert!(
                proof.verify(&cfg, &root).unwrap(),
                "proof for leaf {i} should verify (5 leaves)"
            );
        }
    }

    #[test]
    fn binary_non_pow2_256() { binary_non_power_of_two_inner::<Sha3_256Hasher>(); }
    #[test]
    fn binary_non_pow2_384() { binary_non_power_of_two_inner::<Sha3_384Hasher>(); }
    #[test]
    fn binary_non_pow2_512() { binary_non_power_of_two_inner::<Sha3_512Hasher>(); }

    fn quaternary_inner<H: MerkleHasher>() {
        let cfg = MerkleConfig::new(vec![4], 1);
        let mut tree = MerkleTree::<H>::new(cfg.clone());
        for i in 0..16u64 {
            tree.push_leaf(&[gl(i)]);
        }
        let root = tree.finalize().unwrap();
        for i in 0..16 {
            let proof = tree.open(i).unwrap();
            assert!(
                proof.verify(&cfg, &root).unwrap(),
                "quaternary proof for leaf {i} should verify"
            );
        }
    }

    #[test]
    fn quaternary_256() { quaternary_inner::<Sha3_256Hasher>(); }
    #[test]
    fn quaternary_384() { quaternary_inner::<Sha3_384Hasher>(); }
    #[test]
    fn quaternary_512() { quaternary_inner::<Sha3_512Hasher>(); }

    fn mixed_arity_inner<H: MerkleHasher>() {
        let cfg = MerkleConfig::new(vec![4, 2], 99);
        let mut tree = MerkleTree::<H>::new(cfg.clone());
        for i in 0..16u64 {
            tree.push_leaf(&[gl(i), gl(i * 100)]);
        }
        let root = tree.finalize().unwrap();
        for i in 0..16 {
            let proof = tree.open(i).unwrap();
            assert!(
                proof.verify(&cfg, &root).unwrap(),
                "mixed-arity proof for leaf {i} should verify"
            );
        }
    }

    #[test]
    fn mixed_arity_256() { mixed_arity_inner::<Sha3_256Hasher>(); }
    #[test]
    fn mixed_arity_384() { mixed_arity_inner::<Sha3_384Hasher>(); }
    #[test]
    fn mixed_arity_512() { mixed_arity_inner::<Sha3_512Hasher>(); }

    // ── from_field_rows convenience ─────────────────────────────

    #[test]
    fn from_field_rows_roundtrip() {
        let cfg = MerkleConfig::binary(7);
        let rows: Vec<Vec<Goldilocks>> = (0..4u64).map(|i| vec![gl(i)]).collect();
        let tree = MerkleTree::<Sha3_512Hasher>::from_field_rows(cfg.clone(), &rows).unwrap();
        let root = tree.root().unwrap();
        for i in 0..4 {
            let proof = tree.open(i).unwrap();
            assert!(proof.verify(&cfg, &root).unwrap());
        }
    }

    // ── leaf hash consistency ───────────────────────────────────

    fn leaf_hash_matches_tree_inner<H: MerkleHasher>() {
        let cfg = MerkleConfig::binary(42);
        let values = vec![gl(100), gl(200)];
        let mut tree = MerkleTree::<H>::new(cfg.clone());
        tree.push_leaf(&values);
        tree.push_leaf(&[gl(0)]);
        tree.finalize().unwrap();

        let expected = compute_leaf_hash::<H>(&cfg, 0, &values);
        let proof = tree.open(0).unwrap();
        assert_eq!(proof.leaf, expected);
    }

    #[test]
    fn leaf_hash_matches_256() { leaf_hash_matches_tree_inner::<Sha3_256Hasher>(); }
    #[test]
    fn leaf_hash_matches_384() { leaf_hash_matches_tree_inner::<Sha3_384Hasher>(); }
    #[test]
    fn leaf_hash_matches_512() { leaf_hash_matches_tree_inner::<Sha3_512Hasher>(); }

    // ── error paths ─────────────────────────────────────────────

    #[test]
    fn open_before_finalize_errors() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree256::new(cfg);
        tree.push_leaf(&[gl(1)]);
        assert_eq!(tree.open(0).unwrap_err(), MerkleError::NotFinalized);
    }

    #[test]
    fn root_before_finalize_errors() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree256::new(cfg);
        tree.push_leaf(&[gl(1)]);
        assert_eq!(tree.root().unwrap_err(), MerkleError::NotFinalized);
    }

    #[test]
    fn index_out_of_bounds_errors() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree256::new(cfg);
        tree.push_leaf(&[gl(1)]);
        tree.push_leaf(&[gl(2)]);
        tree.finalize().unwrap();
        assert!(matches!(
            tree.open(99),
            Err(MerkleError::IndexOutOfBounds { .. })
        ));
    }

    // ── tamper detection (concrete type for direct byte access) ─

    #[test]
    fn tampered_sibling_fails_verification() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree256::new(cfg.clone());
        for i in 0..4u64 {
            tree.push_leaf(&[gl(i)]);
        }
        let root = tree.finalize().unwrap();
        let mut proof = tree.open(1).unwrap();

        proof.path[0][0][0] ^= 0xFF;
        assert!(!proof.verify(&cfg, &root).unwrap());
    }

    #[test]
    fn tampered_leaf_fails_verification() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree384::new(cfg.clone());
        for i in 0..4u64 {
            tree.push_leaf(&[gl(i)]);
        }
        let root = tree.finalize().unwrap();
        let mut proof = tree.open(0).unwrap();

        proof.leaf[0] ^= 0xFF;
        assert!(!proof.verify(&cfg, &root).unwrap());
    }

    #[test]
    fn wrong_root_fails_verification() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree512::new(cfg.clone());
        for i in 0..4u64 {
            tree.push_leaf(&[gl(i)]);
        }
        tree.finalize().unwrap();
        let proof = tree.open(0).unwrap();

        let fake_root = [0xABu8; 64];
        assert!(!proof.verify(&cfg, &fake_root).unwrap());
    }

    // ── domain separation ───────────────────────────────────────

    #[test]
    fn different_tree_labels_produce_different_roots() {
        let make = |label: u64| {
            let cfg = MerkleConfig::binary(label);
            let mut tree = MerkleTree256::new(cfg);
            tree.push_leaf(&[gl(1)]);
            tree.push_leaf(&[gl(2)]);
            tree.finalize().unwrap()
        };
        assert_ne!(make(0), make(1));
    }

    #[test]
    fn different_hashers_produce_different_roots() {
        let cfg = MerkleConfig::binary(0);

        let mut t256 = MerkleTree256::new(cfg.clone());
        let mut t384 = MerkleTree384::new(cfg.clone());
        let mut t512 = MerkleTree512::new(cfg.clone());

        for i in 0..4u64 {
            t256.push_leaf(&[gl(i)]);
            t384.push_leaf(&[gl(i)]);
            t512.push_leaf(&[gl(i)]);
        }

        let r256 = t256.finalize().unwrap();
        let r384 = t384.finalize().unwrap();
        let r512 = t512.finalize().unwrap();

        // Different hash functions on the same data must not collide
        // (comparing the leading 32 bytes is a sufficient sanity check).
        assert_ne!(r256.as_ref(), &r384.as_ref()[..32]);
        assert_ne!(r256.as_ref(), &r512.as_ref()[..32]);
        assert_ne!(&r384.as_ref()[..48], &r512.as_ref()[..48]);
    }

    // ── digest sizes ────────────────────────────────────────────

    #[test]
    fn digest_sizes_are_correct() {
        assert_eq!(Sha3_256Hasher::DIGEST_SIZE, 32);
        assert_eq!(Sha3_384Hasher::DIGEST_SIZE, 48);
        assert_eq!(Sha3_512Hasher::DIGEST_SIZE, 64);
    }

    // ── verify_opening static helper ────────────────────────────

    #[test]
    fn verify_opening_static_helper() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree256::new(cfg.clone());
        for i in 0..4u64 {
            tree.push_leaf(&[gl(i)]);
        }
        let root = tree.finalize().unwrap();
        let proof = tree.open(2).unwrap();
        assert!(MerkleTree256::verify_opening(&cfg, &root, &proof).unwrap());
    }

    // ── serde roundtrip ─────────────────────────────────────────

    fn serde_roundtrip_inner<H: MerkleHasher>() {
        let cfg = MerkleConfig::binary(0);
        let mut tree = MerkleTree::<H>::new(cfg.clone());
        for i in 0..4u64 {
            tree.push_leaf(&[gl(i)]);
        }
        let root = tree.finalize().unwrap();
        let proof = tree.open(2).unwrap();

        let encoded = bincode::serialize(&proof).unwrap();
        let decoded: MerkleOpening<H> = bincode::deserialize(&encoded).unwrap();

        assert_eq!(decoded.leaf, proof.leaf);
        assert_eq!(decoded.index, proof.index);
        assert_eq!(decoded.path, proof.path);
        assert!(decoded.verify(&cfg, &root).unwrap());
    }

    #[test]
    fn serde_roundtrip_256() { serde_roundtrip_inner::<Sha3_256Hasher>(); }
    #[test]
    fn serde_roundtrip_384() { serde_roundtrip_inner::<Sha3_384Hasher>(); }
    #[test]
    fn serde_roundtrip_512() { serde_roundtrip_inner::<Sha3_512Hasher>(); }

    #[test]
    fn field_serde_roundtrip() {
        let original = SerFr(gl(0xDEAD_BEEF_CAFE));
        let encoded = bincode::serialize(&original).unwrap();
        let decoded: SerFr = bincode::deserialize(&encoded).unwrap();
        assert_eq!(original, decoded);
    }
}