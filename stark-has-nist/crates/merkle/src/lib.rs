use ark_ff::{BigInteger, PrimeField};
use ark_goldilocks::Goldilocks as F;
use ark_goldilocks::Goldilocks;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use sha3::{Digest, Sha3_256};

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

/// Canonical, injective encoding of a Goldilocks field element to 8 bytes.
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
/// Merkle opening (full SHA3-256)
/// =======================

#[derive(Clone, Debug)]
pub struct MerkleOpening {
    /// ✅ Full 32-byte SHA3-256 leaf hash (was: F, i.e. 8 bytes)
    pub leaf: [u8; 32],
    /// ✅ Full 32-byte SHA3-256 sibling hashes at each level
    pub path: Vec<Vec<[u8; 32]>>,
    pub index: usize,
}

/// =======================
/// Merkle tree (full SHA3-256)
/// =======================

pub struct MerkleTreeChannel {
    cfg: MerkleChannelCfg,
    /// ✅ Every level stores full 32-byte SHA3-256 digests
    levels: Vec<Vec<[u8; 32]>>,
}

impl MerkleTreeChannel {
    pub fn new(cfg: MerkleChannelCfg, _trace_hash: [u8; 32]) -> Self {
        Self {
            cfg,
            levels: Vec::new(),
        }
    }

    /// ✅ Hash field-element leaves into a full 32-byte SHA3-256 digest.
    fn compress_leaf(&self, ds: DsLabel, children: &[F]) -> [u8; 32] {
        let mut h = Sha3_256::new();
        Digest::update(&mut h, ds.to_bytes());
        for c in children {
            Digest::update(&mut h, field_to_bytes(c));
        }
        h.finalize().into()
    }

    /// ✅ Hash child node digests into a full 32-byte SHA3-256 parent digest.
    fn compress_node(&self, ds: DsLabel, children: &[[u8; 32]]) -> [u8; 32] {
        let mut h = Sha3_256::new();
        Digest::update(&mut h, ds.to_bytes());
        for c in children {
            Digest::update(&mut h, c);
        }
        h.finalize().into()
    }

    /// ✅ Generic leaf: caller flattens values
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

    /// ✅ Returns full 32-byte SHA3-256 root (was: F)
    pub fn finalize(&mut self) -> [u8; 32] {
        let mut level = 0;
        while self.levels[level].len() > 1 {
            let arity = self.cfg.layer_arities[level];
            let mut cur = self.levels[level].clone();

            if cur.len() % arity != 0 {
                let last = *cur.last().unwrap();
                cur.resize(cur.len() + (arity - cur.len() % arity), last);
            }

            let parents: Vec<[u8; 32]> = cur
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

            let siblings: Vec<[u8; 32]> = group
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

    /// ✅ Verify a Merkle opening against a full 32-byte root.
    pub fn verify_opening(
        cfg: &MerkleChannelCfg,
        root: [u8; 32],
        opening: &MerkleOpening,
        _trace_hash: &[u8; 32],
    ) -> bool {
        let mut cur = opening.leaf;
        let mut idx = opening.index;

        for (level, siblings) in opening.path.iter().enumerate() {
            let arity = cfg.layer_arities[level];
            let pos = idx % arity;

            let mut children: Vec<[u8; 32]> = Vec::with_capacity(arity);
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

            let mut h = Sha3_256::new();
            Digest::update(&mut h, ds.to_bytes());
            for c in &children {
                Digest::update(&mut h, c);
            }
            cur = h.finalize().into();

            idx /= arity;
        }

        cur == root
    }
}

/// ✅ Public helper: recompute a leaf hash from raw field values.
///
/// This allows the FRI verifier to bind payload values to the Merkle leaf
/// without duplicating the domain-separation logic.
pub fn compute_leaf_hash(cfg: &MerkleChannelCfg, index: usize, values: &[F]) -> [u8; 32] {
    let ds = DsLabel {
        arity: cfg.layer_arities[0],
        level: LEAF_LEVEL_DS,
        position: index as u64,
        tree_label: cfg.tree_label,
    };
    let mut h = Sha3_256::new();
    Digest::update(&mut h, ds.to_bytes());
    for v in values {
        Digest::update(&mut h, field_to_bytes(v));
    }
    h.finalize().into()
}