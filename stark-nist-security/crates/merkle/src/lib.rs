use ark_ff::{BigInteger, PrimeField};
use ark_goldilocks::Goldilocks as F;
use ark_goldilocks::Goldilocks;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use hash::SelectedHasher;
use hash::selected::HASH_BYTES;
use hash::sha3::Digest;

#[cfg(feature = "parallel")]
use rayon::prelude::*;


/// =======================
/// Serialization helpers
/// =======================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerFr(pub F);

impl From<F> for SerFr {
    fn from(x: F) -> Self {
        SerFr(x)
    }
}

impl From<SerFr> for F {
    fn from(w: SerFr) -> F {
        w.0
    }
}

impl Serialize for SerFr {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut out = [0u8; 8];
        let bytes = self.0.into_bigint().to_bytes_le();
        let n = bytes.len().min(8);
        out[..n].copy_from_slice(&bytes[..n]);
        serializer.serialize_bytes(&out)
    }
}

impl<'de> Deserialize<'de> for SerFr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        let mut v = 0u64;
        for (i, &b) in bytes.iter().take(8).enumerate() {
            v |= (b as u64) << (i * 8);
        }
        Ok(SerFr(Goldilocks::from(v)))
    }
}

pub fn field_to_bytes(field: &Goldilocks) -> [u8; 8] {
    let mut out = [0u8; 8];
    let bytes = field.into_bigint().to_bytes_le();
    let n = bytes.len().min(8);
    out[..n].copy_from_slice(&bytes[..n]);
    out
}

pub fn bytes_to_field(bytes: &[u8; 8]) -> Goldilocks {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as u64) << (i * 8);
    }
    Goldilocks::from(v)
}

/// =======================
/// Hash finalization helper
/// =======================

#[inline]
fn finalize_hash(h: SelectedHasher) -> [u8; HASH_BYTES] {
    let result = h.finalize();
    let slice = result.as_slice();
    assert!(
        slice.len() >= HASH_BYTES,
        "Hasher output ({} bytes) shorter than HASH_BYTES ({})",
        slice.len(),
        HASH_BYTES,
    );
    let mut out = [0u8; HASH_BYTES];
    out.copy_from_slice(&slice[..HASH_BYTES]);
    out
}

/// =======================
/// Domain separation
/// =======================

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

pub const LEAF_LEVEL_DS: u32 = u32::MAX;

/// =======================
/// Merkle config
/// =======================

#[derive(Clone)]
pub struct MerkleChannelCfg {
    pub layer_arities: Vec<usize>,
    pub tree_label: u64,
}

impl MerkleChannelCfg {
    pub fn new(layer_arities: Vec<usize>, tree_label: u64) -> Self {
        Self { layer_arities, tree_label }
    }
}

/// =======================
/// Merkle opening
/// =======================

#[derive(Clone, Debug)]
pub struct MerkleOpening {
    pub leaf: [u8; HASH_BYTES],
    pub path: Vec<Vec<[u8; HASH_BYTES]>>,
    pub index: usize,
}

/// =======================
/// Free helper: hash a single leaf (no &self needed)
/// =======================

fn compress_leaf_standalone(
    arity: usize,
    tree_label: u64,
    position: u64,
    values: &[F],
) -> [u8; HASH_BYTES] {
    let ds = DsLabel {
        arity,
        level: LEAF_LEVEL_DS,
        position,
        tree_label,
    };
    let mut h = SelectedHasher::new();
    Digest::update(&mut h, ds.to_bytes());
    for v in values {
        Digest::update(&mut h, field_to_bytes(v));
    }
    finalize_hash(h)
}

/// =======================
/// Merkle tree
/// =======================

pub struct MerkleTreeChannel {
    cfg: MerkleChannelCfg,
    levels: Vec<Vec<[u8; HASH_BYTES]>>,
}

impl MerkleTreeChannel {
    pub fn new(cfg: MerkleChannelCfg, _trace_hash: [u8; HASH_BYTES]) -> Self {
        Self {
            cfg,
            levels: Vec::new(),
        }
    }

    fn compress_leaf(&self, ds: DsLabel, children: &[F]) -> [u8; HASH_BYTES] {
        let mut h = SelectedHasher::new();
        Digest::update(&mut h, ds.to_bytes());
        for c in children {
            Digest::update(&mut h, field_to_bytes(c));
        }
        finalize_hash(h)
    }

    fn compress_node(&self, ds: DsLabel, children: &[[u8; HASH_BYTES]]) -> [u8; HASH_BYTES] {
        let mut h = SelectedHasher::new();
        Digest::update(&mut h, ds.to_bytes());
        for c in children {
            Digest::update(&mut h, c);
        }
        finalize_hash(h)
    }

    pub fn push_leaf(&mut self, values: &[F]) {
        if self.levels.is_empty() {
            self.levels.push(Vec::new());
        }

        let idx = self.levels[0].len();
        let ds = DsLabel {
            arity: self.cfg.layer_arities[0],
            level: LEAF_LEVEL_DS,
            position: idx as u64,
            tree_label: self.cfg.tree_label,
        };

        let leaf = self.compress_leaf(ds, values);
        self.levels[0].push(leaf);
    }

    /// Bulk-insert all leaves, using rayon when `parallel` is enabled.
    pub fn push_leaves_parallel(&mut self, all_values: &[Vec<F>]) {
        if self.levels.is_empty() {
            self.levels.push(Vec::new());
        }

        // Pull config out before the closure so we don't borrow &self inside par_iter
        let arity = self.cfg.layer_arities[0];
        let tree_label = self.cfg.tree_label;

        #[cfg(feature = "parallel")]
        let leaves: Vec<[u8; HASH_BYTES]> = all_values
            .par_iter()
            .enumerate()
            .map(|(idx, values)| {
                compress_leaf_standalone(arity, tree_label, idx as u64, values)
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let leaves: Vec<[u8; HASH_BYTES]> = all_values
            .iter()
            .enumerate()
            .map(|(idx, values)| {
                compress_leaf_standalone(arity, tree_label, idx as u64, values)
            })
            .collect();

        self.levels[0] = leaves;
    }

    pub fn finalize(&mut self) -> [u8; HASH_BYTES] {
        let mut level = 0;
        while self.levels[level].len() > 1 {
            let arity = self.cfg.layer_arities[level];
            let mut cur = self.levels[level].clone();

            if cur.len() % arity != 0 {
                let last = *cur.last().unwrap();
                cur.resize(cur.len() + (arity - cur.len() % arity), last);
            }

            let parents: Vec<[u8; HASH_BYTES]> = cur
                .chunks(arity)
                .enumerate()
                .map(|(i, c)| {
                    let ds = DsLabel {
                        arity,
                        level: level as u32 + 1,
                        position: i as u64,
                        tree_label: self.cfg.tree_label,
                    };
                    self.compress_node(ds, c)
                })
                .collect();

            self.levels.push(parents);
            level += 1;
        }
        self.levels.last().unwrap()[0]
    }

    pub fn open(&self, index: usize) -> MerkleOpening {
        let mut idx = index;
        let mut path = Vec::new();

        for level in 0..self.levels.len() - 1 {
            let nodes = &self.levels[level];
            let arity = self.cfg.layer_arities[level];
            let group_start = (idx / arity) * arity;

            let mut group = Vec::with_capacity(arity);
            for i in 0..arity {
                let pos = group_start + i;
                if pos < nodes.len() {
                    group.push(nodes[pos]);
                } else {
                    group.push(*nodes.last().unwrap());
                }
            }

            let siblings: Vec<[u8; HASH_BYTES]> = group
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| {
                    if group_start + i != idx {
                        Some(x)
                    } else {
                        None
                    }
                })
                .collect();

            path.push(siblings);
            idx /= arity;
        }

        MerkleOpening {
            leaf: self.levels[0][index],
            path,
            index,
        }
    }

    pub fn verify_opening(
        cfg: &MerkleChannelCfg,
        root: [u8; HASH_BYTES],
        opening: &MerkleOpening,
        _trace_hash: &[u8; HASH_BYTES],
    ) -> bool {
        let mut cur = opening.leaf;
        let mut idx = opening.index;

        for (level, siblings) in opening.path.iter().enumerate() {
            let arity = cfg.layer_arities[level];
            let pos = idx % arity;

            let mut children: Vec<[u8; HASH_BYTES]> = Vec::with_capacity(arity);
            let mut sibs = siblings.iter();

            for i in 0..arity {
                if i == pos {
                    children.push(cur);
                } else {
                    match sibs.next() {
                        Some(&x) => children.push(x),
                        None => return false,
                    }
                }
            }

            let ds = DsLabel {
                arity,
                level: level as u32 + 1,
                position: (idx / arity) as u64,
                tree_label: cfg.tree_label,
            };

            let mut h = SelectedHasher::new();
            Digest::update(&mut h, ds.to_bytes());
            for c in &children {
                Digest::update(&mut h, c);
            }
            cur = finalize_hash(h);

            idx /= arity;
        }

        cur == root
    }
} // ← impl MerkleTreeChannel ends here

/// =======================
/// Free function: compute a single leaf hash
/// =======================

pub fn compute_leaf_hash(cfg: &MerkleChannelCfg, index: usize, values: &[F]) -> [u8; HASH_BYTES] {
    compress_leaf_standalone(
        cfg.layer_arities[0],
        cfg.tree_label,
        index as u64,
        values,
    )
}