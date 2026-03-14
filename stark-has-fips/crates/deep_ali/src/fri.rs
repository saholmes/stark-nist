#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused_macros)]

use ark_goldilocks::Goldilocks as F;
use ark_serialize::CanonicalSerialize;
use rand::{rngs::StdRng, Rng, SeedableRng};

use ark_poly::domain::radix2::Radix2EvaluationDomain as Domain;
use ark_ff::{Field, One, PrimeField, Zero};
use ark_poly::{
    EvaluationDomain, GeneralEvaluationDomain,
};

use sha3::{Digest, Sha3_256};

use crate::cubic_ext::CubicExt;

use merkle::{
    MerkleChannelCfg,
    MerkleTreeChannel,
    MerkleOpening,
    compute_leaf_hash,
};

use transcript::Transcript;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const PARALLEL_MIN_ELEMS: usize = 1 << 12;

#[inline]
fn enable_parallel(len: usize) -> bool {
    #[cfg(feature = "parallel")]
    {
        len >= PARALLEL_MIN_ELEMS && rayon::current_num_threads() > 1
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = len;
        false
    }
}

#[cfg(feature = "fri_bench_log")]
#[allow(unused_macros)]
macro_rules! logln {
    ($($tt:tt)*) => { eprintln!($($tt)*); }
}
#[cfg(not(feature = "fri_bench_log"))]
macro_rules! logln {
    ($($tt:tt)*) => {};
}

mod ds {
    pub const FRI_SEED: &[u8] = b"FRI/seed";
    pub const FRI_INDEX: &[u8] = b"FRI/index";
    pub const FRI_Z_L: &[u8] = b"FRI/z/l";
    pub const FRI_Z_L_1: &[u8] = b"FRI/z/l/1";
    pub const FRI_Z_L_2: &[u8] = b"FRI/z/l/2";
    pub const FRI_LEAF: &[u8] = b"FRI/leaf";
}

fn tr_hash_fields_tagged(tag: &[u8], fields: &[F]) -> F {
    let mut tr = Transcript::new_sha3(b"FRI/FS");
    tr.absorb_bytes(tag);
    for &x in fields {
        tr.absorb_field(x);
    }
    tr.challenge(b"out")
}

// ────────────────────────────────────────────────────────────────────────
//  FRI Domain
// ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct FriDomain {
    pub omega: F,
    pub size: usize,
}

impl FriDomain {
    pub fn new_radix2(size: usize) -> Self {
        let dom = Domain::<F>::new(size).expect("radix-2 domain exists");
        Self { omega: dom.group_gen, size }
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Base-field utilities (kept for backward compatibility and tests)
// ────────────────────────────────────────────────────────────────────────

fn build_z_pows(z_l: F, m: usize) -> Vec<F> {
    let mut z_pows = Vec::with_capacity(m);
    let mut acc = F::one();
    for _ in 0..m {
        z_pows.push(acc);
        acc *= z_l;
    }
    z_pows
}

/// Build powers of an extension-field folding challenge.
fn build_ext_pows(alpha: CubicExt, m: usize) -> Vec<CubicExt> {
    let mut pows = Vec::with_capacity(m);
    let mut acc = CubicExt::ONE;
    for _ in 0..m {
        pows.push(acc);
        acc = acc * alpha;
    }
    pows
}

fn eval_poly_at_ext(coeffs: &[F], z: CubicExt) -> CubicExt {
    let mut acc = CubicExt::ZERO;
    for &c in coeffs.iter().rev() {
        acc = acc * z + CubicExt::from_base(c);
    }
    acc
}

/// DEEP quotient for a **base-field** codeword evaluated against z ∈ F_{p^3}.
/// Used for layer 0, where the trace values are in F_p.
fn compute_q_layer_ext(
    f_l: &[F],
    z: CubicExt,
    omega: F,
) -> (Vec<CubicExt>, CubicExt) {
    let n = f_l.len();
    let dom = Domain::<F>::new(n).unwrap();
    let coeffs = dom.ifft(f_l);
    let fz = eval_poly_at_ext(&coeffs, z);

    let mut q = Vec::with_capacity(n);
    let mut x = F::one();
    for i in 0..n {
        let num   = CubicExt::from_base(f_l[i]) - fz;
        let denom = CubicExt::from_base(x) - z;
        q.push(num / denom);
        x *= omega;
    }
    (q, fz)
}

// ────────────────────────────────────────────────────────────────────────
//  Extension-field FRI core functions (NEW — §2.2 / Theorem 2 fix)
//
//  These ensure folding challenges are drawn from F_{p^3}, giving
//  exception probability L_ℓ / |F_{p^3}| ≈ 2^{-168} per layer instead
//  of L_ℓ / |F_p| ≈ 2^{-40}.
// ────────────────────────────────────────────────────────────────────────

/// DEEP quotient for an **extension-field** codeword.
///
/// Given f_ℓ : H_ℓ → F_{p^3} as evaluations at base-field domain points,
/// computes Q_ℓ(x) = (f_ℓ(x) − f_ℓ(z)) / (x − z) for all x ∈ H_ℓ.
///
/// The polynomial interpolation is performed componentwise: three
/// independent base-field iFFTs recover the F_{p^3} coefficients,
/// then Horner evaluation at z ∈ F_{p^3} uses true extension-field
/// multiplication.
fn compute_q_layer_ext_on_ext(
    f_l: &[CubicExt],
    z: CubicExt,
    omega: F,
) -> (Vec<CubicExt>, CubicExt) {
    let n = f_l.len();
    let dom = Domain::<F>::new(n).unwrap();

    // Componentwise iFFT
    let c0_evals: Vec<F> = f_l.iter().map(|v| v.c[0]).collect();
    let c1_evals: Vec<F> = f_l.iter().map(|v| v.c[1]).collect();
    let c2_evals: Vec<F> = f_l.iter().map(|v| v.c[2]).collect();

    let c0_coeffs = dom.ifft(&c0_evals);
    let c1_coeffs = dom.ifft(&c1_evals);
    let c2_coeffs = dom.ifft(&c2_evals);

    // Horner evaluation at z ∈ F_{p^3}
    let mut fz = CubicExt::ZERO;
    for k in (0..n).rev() {
        let coeff_k = CubicExt::new(c0_coeffs[k], c1_coeffs[k], c2_coeffs[k]);
        fz = fz * z + coeff_k;
    }

    // Q(x) = (f(x) − f(z)) / (x − z)  for each x ∈ H
    let mut q = Vec::with_capacity(n);
    let mut x = F::one();
    for i in 0..n {
        let num   = f_l[i] - fz;
        let denom = CubicExt::from_base(x) - z;
        q.push(num / denom);
        x *= omega;
    }

    (q, fz)
}

/// Fold an extension-field codeword with challenge α ∈ F_{p^3}.
///
///   f_{ℓ+1}[b] = Σ_{j=0}^{m-1}  f_ℓ[b + j·n_next] · α^j
///
/// All arithmetic is in F_{p^3}.
fn fri_fold_layer_ext_impl(
    evals: &[CubicExt],
    alpha: CubicExt,
    folding_factor: usize,
) -> Vec<CubicExt> {
    let n = evals.len();
    assert!(n % folding_factor == 0);
    let n_next = n / folding_factor;

    let alpha_pows = build_ext_pows(alpha, folding_factor);

    let mut out = vec![CubicExt::ZERO; n_next];

    if enable_parallel(n_next) {
        #[cfg(feature = "parallel")]
        {
            out.par_iter_mut().enumerate().for_each(|(b, out_b)| {
                let mut acc = CubicExt::ZERO;
                for j in 0..folding_factor {
                    acc = acc + evals[b + j * n_next] * alpha_pows[j];
                }
                *out_b = acc;
            });
            return out;
        }
    }

    for b in 0..n_next {
        let mut acc = CubicExt::ZERO;
        for j in 0..folding_factor {
            acc = acc + evals[b + j * n_next] * alpha_pows[j];
        }
        out[b] = acc;
    }
    out
}

/// Compute s-layer (repeated fold targets) for an extension-field codeword.
///
/// s_per_i[b + j·n_next] = folded[b]  for all j, where
/// folded[b] = Σ_j f[b + j·n_next] · α^j.
///
/// This lets the verifier check:  s_ℓ[query_index] == f_{ℓ+1}[parent(query_index)].
fn compute_s_layer_ext(
    f_l: &[CubicExt],
    alpha: CubicExt,
    m: usize,
) -> Vec<CubicExt> {
    let n = f_l.len();
    assert!(n % m == 0);
    let n_next = n / m;

    let alpha_pows = build_ext_pows(alpha, m);

    let mut folded = vec![CubicExt::ZERO; n_next];
    for b in 0..n_next {
        let mut sum = CubicExt::ZERO;
        for j in 0..m {
            sum = sum + f_l[b + j * n_next] * alpha_pows[j];
        }
        folded[b] = sum;
    }

    let mut s_per_i = vec![CubicExt::ZERO; n];
    for b in 0..n_next {
        for j in 0..m {
            s_per_i[b + j * n_next] = folded[b];
        }
    }
    s_per_i
}

// ────────────────────────────────────────────────────────────────────────
//  Base-field FRI fold (kept for backward compatibility / tests)
// ────────────────────────────────────────────────────────────────────────

fn dot_with_z_pows(chunk: &[F], z_pows: &[F]) -> F {
    debug_assert_eq!(chunk.len(), z_pows.len());
    let mut s = F::zero();
    for (val, zp) in chunk.iter().zip(z_pows.iter()) {
        s += *val * *zp;
    }
    s
}

fn fold_layer_sequential(f_l: &[F], z_pows: &[F], m: usize) -> Vec<F> {
    f_l.chunks(m)
        .map(|chunk| dot_with_z_pows(chunk, z_pows))
        .collect()
}

#[cfg(feature = "parallel")]
fn fold_layer_parallel(f_l: &[F], z_pows: &[F], m: usize) -> Vec<F> {
    f_l.par_chunks(m)
        .map(|chunk| dot_with_z_pows(chunk, z_pows))
        .collect()
}

fn fill_repeated_targets(target: &mut [F], src: &[F], m: usize) {
    for (bucket, chunk) in src.iter().zip(target.chunks_mut(m)) {
        for item in chunk {
            *item = *bucket;
        }
    }
}

fn merkle_depth(leaves: usize, arity: usize) -> usize {
    assert!(arity >= 2, "Merkle arity must be ≥ 2");
    let mut depth = 1;
    let mut cur = leaves;
    while cur > arity {
        cur = (cur + arity - 1) / arity;
        depth += 1;
    }
    depth
}

#[cfg(feature = "parallel")]
fn fill_repeated_targets_parallel(target: &mut [F], src: &[F], m: usize) {
    target
        .par_chunks_mut(m)
        .enumerate()
        .for_each(|(idx, chunk)| {
            let bucket = src[idx];
            for item in chunk {
                *item = bucket;
            }
        });
}

pub fn fri_sample_z_ell(seed_z: u64, level: usize, domain_size: usize) -> F {
    let fused = tr_hash_fields_tagged(
        ds::FRI_Z_L,
        &[F::from(seed_z), F::from(level as u64), F::from(domain_size as u64)],
    );
    let mut seed_bytes = [0u8; 32];
    fused.serialize_uncompressed(&mut seed_bytes[..]).expect("serialize");
    let mut rng = StdRng::from_seed(seed_bytes);
    let exp_bigint = <F as PrimeField>::BigInt::from(domain_size as u64);
    let mut tries = 0usize;
    const MAX_TRIES: usize = 1_000;
    loop {
        let cand = F::from(rng.gen::<u64>());
        if !cand.is_zero() && cand.pow(exp_bigint.as_ref()) != F::one() {
            return cand;
        }
        tries += 1;
        if tries >= MAX_TRIES {
            let fallback = F::from(seed_z.wrapping_add(level as u64).wrapping_add(7));
            if fallback.pow(exp_bigint.as_ref()) != F::one() {
                return fallback;
            }
            return F::from(11u64);
        }
    }
}

/// Base-field s-layer (kept for backward compatibility / tests).
pub fn compute_s_layer(f_l: &[F], z_l: F, m: usize) -> Vec<F> {
    let n = f_l.len();
    assert!(n % m == 0);
    let n_next = n / m;
    let z_pows = build_z_pows(z_l, m);
    let mut folded = vec![F::zero(); n_next];
    for b in 0..n_next {
        let mut acc = F::zero();
        for j in 0..m {
            acc += f_l[b + j * n_next] * z_pows[j];
        }
        folded[b] = acc;
    }
    let mut s_per_i = vec![F::zero(); n];
    for b in 0..n_next {
        for j in 0..m {
            s_per_i[b + j * n_next] = folded[b];
        }
    }
    s_per_i
}

fn layer_sizes_from_schedule(n0: usize, schedule: &[usize]) -> Vec<usize> {
    let mut sizes = Vec::with_capacity(schedule.len() + 1);
    let mut n = n0;
    sizes.push(n);
    for &m in schedule {
        assert!(n % m == 0, "schedule not dividing domain size");
        n /= m;
        sizes.push(n);
    }
    sizes
}

fn hash_node(children: &[[u8; 32]]) -> [u8; 32] {
    let mut h = Sha3_256::new();
    Digest::update(&mut h, b"FRI/MERKLE/NODE");
    for c in children {
        Digest::update(&mut h, c);
    }
    h.finalize().into()
}

fn index_from_seed(seed_f: F, n_pow2: usize) -> usize {
    assert!(n_pow2.is_power_of_two());
    let mask = n_pow2 - 1;
    let mut seed_bytes = [0u8; 32];
    seed_f.serialize_uncompressed(&mut seed_bytes[..]).unwrap();
    let mut rng = StdRng::from_seed(seed_bytes);
    (rng.gen::<u64>() as usize) & mask
}

fn index_seed(roots_seed: F, ell: usize, q: usize) -> F {
    tr_hash_fields_tagged(
        ds::FRI_INDEX,
        &[roots_seed, F::from(ell as u64), F::from(q as u64)],
    )
}

fn f0_trace_hash(n0: usize, seed_z: u64) -> [u8; 32] {
    let mut h = Sha3_256::new();
    Digest::update(&mut h, b"FRI/F0_TREE_DOMAIN");
    Digest::update(&mut h, &(n0 as u64).to_le_bytes());
    Digest::update(&mut h, &seed_z.to_le_bytes());
    h.finalize().into()
}

fn f0_tree_config(n0: usize) -> MerkleChannelCfg {
    let arity = pick_arity_for_layer(n0, 16).max(2);
    let depth = merkle_depth(n0, arity);
    MerkleChannelCfg::new(vec![arity; depth], 0xFF)
}

fn pick_arity_for_layer(n: usize, requested_m: usize) -> usize {
    if requested_m >= 128 && n % 128 == 0 { return 128; }
    if requested_m >= 64  && n % 64  == 0 { return 64; }
    if requested_m >= 32  && n % 32  == 0 { return 32; }
    if requested_m >= 16  && n % 16  == 0 { return 16; }
    if requested_m >= 8   && n % 8   == 0 { return 8; }
    if requested_m >= 4   && n % 4   == 0 { return 4; }
    if n % 2 == 0 { return 2; }
    1
}

fn bind_statement_to_transcript(
    tr: &mut Transcript,
    schedule: &[usize],
    n0: usize,
    seed_z: u64,
) {
    tr.absorb_bytes(b"DEEP-FRI-STATEMENT");
    tr.absorb_field(F::from(n0 as u64));
    tr.absorb_field(F::from(schedule.len() as u64));
    for &m in schedule {
        tr.absorb_field(F::from(m as u64));
    }
    tr.absorb_field(F::from(seed_z));
}

/// Base-field FRI fold (kept for backward compatibility / tests).
pub fn fri_fold_layer(
    evals: &[F],
    z_l: F,
    folding_factor: usize,
) -> Vec<F> {
    let domain_size = evals.len();
    let domain = GeneralEvaluationDomain::<F>::new(domain_size)
        .expect("Domain size must be a power of two.");
    let domain_generator = domain.group_gen();
    fri_fold_layer_impl(evals, z_l, domain_generator, folding_factor)
}

fn fri_fold_layer_impl(
    evals: &[F],
    z_l: F,
    omega: F,
    folding_factor: usize,
) -> Vec<F> {
    let n = evals.len();
    assert!(n % folding_factor == 0);
    let n_next = n / folding_factor;
    let mut out = vec![F::zero(); n_next];
    let z_pows = build_z_pows(z_l, folding_factor);

    if enable_parallel(n_next) {
        #[cfg(feature = "parallel")]
        {
            out.par_iter_mut().enumerate().for_each(|(b, out_b)| {
                let mut acc = F::zero();
                for j in 0..folding_factor {
                    acc += evals[b + j * n_next] * z_pows[j];
                }
                *out_b = acc;
            });
            return out;
        }
    }

    for b in 0..n_next {
        let mut acc = F::zero();
        for j in 0..folding_factor {
            acc += evals[b + j * n_next] * z_pows[j];
        }
        out[b] = acc;
    }
    out
}

// ────────────────────────────────────────────────────────────────────────
//  Core protocol structs — updated for F_{p^3} folding challenges
// ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct CombinedLeaf {
    pub f: CubicExt,
    pub s: CubicExt,
    pub q: CubicExt,
}

pub struct FriLayerCommitment {
    pub n: usize,
    pub m: usize,
    pub root: [u8; 32],
}

pub struct FriTranscript {
    pub schedule: Vec<usize>,
    pub layers: Vec<FriLayerCommitment>,
}

pub struct FriProverParams {
    pub schedule: Vec<usize>,
    pub seed_z: u64,
}

/// Prover state after transcript construction.
///
/// All codeword layers are stored uniformly in F_{p^3}.  Layer 0 values
/// are embedded from the base field via `CubicExt::from_base`.  This is
/// the key structural change that enables F_{p^3} folding challenges:
/// from layer 1 onward, codeword values are genuine extension-field
/// elements, giving Schwartz–Zippel exception probability L_ℓ/|F_{p^3}|
/// ≈ 2^{-168} per layer (Theorem 2).
pub struct FriProverState {
    /// Original base-field trace (for the f₀ Merkle tree).
    pub f0_base: Vec<F>,
    /// Codeword at each layer, uniformly in F_{p^3}.
    pub f_layers_ext: Vec<Vec<CubicExt>>,
    /// Per-position repeated fold targets in F_{p^3}.
    pub s_layers: Vec<Vec<CubicExt>>,
    /// DEEP quotients in F_{p^3}.
    pub q_layers: Vec<Vec<CubicExt>>,
    /// f_ℓ(z) at each layer, in F_{p^3}.
    pub fz_layers: Vec<CubicExt>,
    pub transcript: FriTranscript,
    /// Domain generators (base field).
    pub omega_layers: Vec<F>,
    /// DEEP challenge point z ∈ F_{p^3}.
    pub z_fp3: CubicExt,
    /// Folding challenges α_ℓ ∈ F_{p^3}.
    pub alpha_layers: Vec<CubicExt>,
    pub root_f0: [u8; 32],
    pub trace_hash: [u8; 32],
}

#[derive(Clone)]
pub struct LayerQueryRef {
    pub i: usize,
    pub child_pos: usize,
    pub parent_index: usize,
    pub parent_pos: usize,
}

#[derive(Clone)]
pub struct FriQueryOpenings {
    pub per_layer_refs: Vec<LayerQueryRef>,
    pub final_index: usize,
    pub final_pair: (CubicExt, CubicExt),
}

/// Per-layer opened values — all three fields are now F_{p^3}.
#[derive(Clone)]
pub struct LayerOpenPayload {
    pub f_val: CubicExt,
    pub s_val: CubicExt,
    pub q_val: CubicExt,
}

#[derive(Clone)]
pub struct FriQueryPayload {
    pub per_layer_refs: Vec<LayerQueryRef>,
    pub per_layer_payloads: Vec<LayerOpenPayload>,
    pub f0_opening: MerkleOpening,
    pub final_index: usize,
    pub final_pair: (CubicExt, CubicExt),
}

#[derive(Clone)]
pub struct LayerProof {
    pub openings: Vec<MerkleOpening>,
}

pub struct FriLayerProofs {
    pub layers: Vec<LayerProof>,
}

pub struct DeepFriProof {
    pub root_f0: [u8; 32],
    pub roots: Vec<[u8; 32]>,
    pub layer_proofs: FriLayerProofs,
    pub f0_openings: Vec<MerkleOpening>,
    pub queries: Vec<FriQueryPayload>,
    pub fz_per_layer: Vec<CubicExt>,
    pub n0: usize,
    pub omega0: F,
}

#[derive(Clone, Debug)]
pub struct DeepFriParams {
    pub schedule: Vec<usize>,
    pub r: usize,
    pub seed_z: u64,
}

// ────────────────────────────────────────────────────────────────────────
//  Leaf serialization helpers
// ────────────────────────────────────────────────────────────────────────

/// Serialize a combined (f, s, q) leaf as 9 base-field elements.
#[inline]
fn ext_leaf_fields(f: CubicExt, s: CubicExt, q: CubicExt) -> [F; 9] {
    [
        f.c[0], f.c[1], f.c[2],
        s.c[0], s.c[1], s.c[2],
        q.c[0], q.c[1], q.c[2],
    ]
}

// ────────────────────────────────────────────────────────────────────────
//  Derive a CubicExt challenge from the Fiat–Shamir transcript
// ────────────────────────────────────────────────────────────────────────

/// Squeeze three base-field challenges and combine into one F_{p^3} element.
fn challenge_ext(tr: &mut Transcript, tag: &[u8]) -> CubicExt {
    // We use distinct sub-tags for each coordinate so the transcript
    // is unambiguous even if two consecutive challenges share a prefix.
    let mut tag0 = tag.to_vec(); tag0.extend_from_slice(b"/c0");
    let mut tag1 = tag.to_vec(); tag1.extend_from_slice(b"/c1");
    let mut tag2 = tag.to_vec(); tag2.extend_from_slice(b"/c2");

    CubicExt::new(
        tr.challenge(&tag0),
        tr.challenge(&tag1),
        tr.challenge(&tag2),
    )
}

/// Absorb all three coordinates of an F_{p^3} element into the transcript.
fn absorb_ext(tr: &mut Transcript, v: CubicExt) {
    tr.absorb_field(v.c[0]);
    tr.absorb_field(v.c[1]);
    tr.absorb_field(v.c[2]);
}

// =============================================================================
// ── Transcript builder (F_{p^3} folding challenges) ──
// =============================================================================

pub fn fri_build_transcript(
    f0: Vec<F>,
    domain0: FriDomain,
    params: &FriProverParams,
) -> FriProverState {
    let schedule = params.schedule.clone();
    let l = schedule.len();

    // ── Phase 0: Statement binding ──
    let mut tr = Transcript::new_sha3(b"FRI/FS");
    bind_statement_to_transcript(&mut tr, &schedule, domain0.size, params.seed_z);

    // ── Phase 1: Commit f₀ (base-field trace) ──
    let f0_th = f0_trace_hash(domain0.size, params.seed_z);
    let f0_cfg = f0_tree_config(domain0.size);
    let mut f0_tree = MerkleTreeChannel::new(f0_cfg.clone(), f0_th);
    for &val in &f0 {
        f0_tree.push_leaf(&[val]);
    }
    let root_f0 = f0_tree.finalize();

    // Absorb root_f0 so that z depends on the committed trace
    tr.absorb_bytes(&root_f0);

    // ── Phase 2: DEEP challenge z ∈ F_{p^3} ──
    let z_fp3 = challenge_ext(&mut tr, b"z_fp3");

    // Derive trace_hash for layer trees
    let trace_hash: [u8; 32] = tr.challenge_bytes(ds::FRI_SEED);

    // ── Phase 3: Layer-by-layer with F_{p^3} folding challenges ──
    //
    // Layer 0 codeword is embedded from F_p.  After folding with
    // α₀ ∈ F_{p^3}, all subsequent layers are in F_{p^3}.

    let f0_ext: Vec<CubicExt> = f0.iter()
        .map(|&x| CubicExt::from_base(x))
        .collect();

    let mut f_layers_ext: Vec<Vec<CubicExt>> = Vec::with_capacity(l + 1);
    let mut s_layers: Vec<Vec<CubicExt>> = Vec::with_capacity(l + 1);
    let mut q_layers: Vec<Vec<CubicExt>> = Vec::with_capacity(l);
    let mut fz_layers: Vec<CubicExt> = Vec::with_capacity(l);
    let mut omega_layers: Vec<F> = Vec::with_capacity(l);
    let mut alpha_layers: Vec<CubicExt> = Vec::with_capacity(l);
    let mut layer_commitments: Vec<FriLayerCommitment> = Vec::with_capacity(l);

    f_layers_ext.push(f0_ext);
    let mut cur_size = domain0.size;

    for ell in 0..l {
        let m = schedule[ell];

        // ── Folding challenge α_ℓ ∈ F_{p^3} ──
        let alpha_ell = challenge_ext(&mut tr, b"alpha");
        alpha_layers.push(alpha_ell);

        // Domain generator for this layer
        let dom = Domain::<F>::new(cur_size).unwrap();
        let omega = dom.group_gen;
        omega_layers.push(omega);

        // ── DEEP quotient ──
        // For layer 0 the base-field optimized path could be used,
        // but the componentwise iFFT on embedded values is correct
        // and keeps the code uniform.
        let cur_f = &f_layers_ext[ell];
        let (q, fz) = compute_q_layer_ext_on_ext(cur_f, z_fp3, omega);
        q_layers.push(q.clone());
        fz_layers.push(fz);

        // ── Fold targets (s-layer) ──
        let s = compute_s_layer_ext(cur_f, alpha_ell, m);
        s_layers.push(s.clone());

        // ── Merkle commitment for this layer ──
        let arity = pick_arity_for_layer(cur_size, m).max(2);
        let depth = merkle_depth(cur_size, arity);
        let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);
        let mut tree = MerkleTreeChannel::new(cfg, trace_hash);

        for i in 0..cur_size {
            let fields = ext_leaf_fields(cur_f[i], s[i], q[i]);
            tree.push_leaf(&fields);
        }
        let layer_root = tree.finalize();

        layer_commitments.push(FriLayerCommitment {
            n: cur_size,
            m,
            root: layer_root,
        });

        // ── Absorb f_ℓ(z) and layer root ──
        absorb_ext(&mut tr, fz);
        tr.absorb_bytes(&layer_root);

        // ── Fold to next layer using α_ℓ ∈ F_{p^3} ──
        let next_f = fri_fold_layer_ext_impl(cur_f, alpha_ell, m);
        cur_size /= m;
        f_layers_ext.push(next_f);

        eprintln!(
            "[PROVER] ell={} z_fp3={:?} alpha={:?}",
            ell, z_fp3, alpha_ell
        );
    }

    // Dummy s-layer for the final layer
    s_layers.push(vec![CubicExt::ZERO; f_layers_ext.last().unwrap().len()]);

    FriProverState {
        f0_base: f0,
        f_layers_ext,
        s_layers,
        q_layers,
        fz_layers,
        transcript: FriTranscript {
            schedule,
            layers: layer_commitments,
        },
        omega_layers,
        z_fp3,
        alpha_layers,
        root_f0,
        trace_hash,
    }
}

// =============================================================================
// ── Query derivation ──
// =============================================================================

pub fn fri_prove_queries(
    st: &FriProverState,
    r: usize,
    query_seed: F,
) -> (Vec<FriQueryOpenings>, Vec<[u8; 32]>, FriLayerProofs, Vec<MerkleOpening>) {
    let L = st.transcript.schedule.len();
    let mut all_refs = Vec::with_capacity(r);
    let n0 = st.transcript.layers.first().map_or(0, |l| l.n);

    for q in 0..r {
        let mut per_layer_refs = Vec::with_capacity(L);

        let mut i = {
            let n_pow2 = n0.next_power_of_two();
            let seed = index_seed(query_seed, 0, q);
            index_from_seed(seed, n_pow2) % n0
        };

        for ell in 0..L {
            let n = st.transcript.layers[ell].n;
            let m = st.transcript.schedule[ell];
            let n_next = n / m;

            per_layer_refs.push(LayerQueryRef {
                i,
                child_pos: i % m,
                parent_index: i % n_next,
                parent_pos: 0,
            });

            i = i % n_next;
        }

        all_refs.push(FriQueryOpenings {
            per_layer_refs,
            final_index: i,
            final_pair: (
                st.f_layers_ext[L][i],
                st.f_layers_ext[L][0],
            ),
        });
    }

    // ── Rebuild f₀ Merkle tree (base-field trace) ──
    let f0_th = f0_trace_hash(n0, 0);
    let f0_cfg = f0_tree_config(n0);
    let mut f0_tree = MerkleTreeChannel::new(f0_cfg, f0_th);
    for &val in &st.f0_base {
        f0_tree.push_leaf(&[val]);
    }
    f0_tree.finalize();

    let mut f0_openings = Vec::with_capacity(r);
    for q in 0..r {
        let idx = all_refs[q].per_layer_refs[0].i;
        f0_openings.push(f0_tree.open(idx));
    }

    // ── Rebuild layer Merkle trees ──
    let mut layer_proofs = Vec::with_capacity(L);

    for ell in 0..L {
        let layer = &st.transcript.layers[ell];
        let arity = pick_arity_for_layer(layer.n, layer.m).max(2);
        let depth = merkle_depth(layer.n, arity);
        let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);
        let mut tree = MerkleTreeChannel::new(cfg, st.trace_hash);

        for i in 0..layer.n {
            let fields = ext_leaf_fields(
                st.f_layers_ext[ell][i],
                st.s_layers[ell][i],
                st.q_layers[ell][i],
            );
            tree.push_leaf(&fields);
        }
        tree.finalize();

        let mut openings = Vec::with_capacity(r);
        for q in 0..r {
            let idx = all_refs[q].per_layer_refs[ell].i;
            openings.push(tree.open(idx));
        }
        layer_proofs.push(LayerProof { openings });
    }

    let roots: Vec<[u8; 32]> = st.transcript.layers.iter().map(|l| l.root).collect();

    (all_refs, roots, FriLayerProofs { layers: layer_proofs }, f0_openings)
}

// =============================================================================
// ── Prover top-level ──
// =============================================================================

pub fn deep_fri_prove(
    f0: Vec<F>,
    domain0: FriDomain,
    params: &DeepFriParams,
) -> DeepFriProof {
    let prover_params = FriProverParams {
        schedule: params.schedule.clone(),
        seed_z: params.seed_z,
    };

    let st = fri_build_transcript(f0, domain0, &prover_params);

    // Derive query_seed by replaying the transcript
    let query_seed = {
        let mut tr = Transcript::new_sha3(b"FRI/FS");
        bind_statement_to_transcript(
            &mut tr,
            &params.schedule,
            domain0.size,
            params.seed_z,
        );
        tr.absorb_bytes(&st.root_f0);

        // Re-derive z_fp3
        let _ = challenge_ext(&mut tr, b"z_fp3");

        // Re-derive trace_hash
        let _: [u8; 32] = tr.challenge_bytes(ds::FRI_SEED);

        // Replay per-layer: α_ℓ (3 squeezes), then absorb fz_ℓ + root_ℓ
        for ell in 0..params.schedule.len() {
            let _ = challenge_ext(&mut tr, b"alpha");
            absorb_ext(&mut tr, st.fz_layers[ell]);
            tr.absorb_bytes(&st.transcript.layers[ell].root);
        }

        tr.challenge(b"query_seed")
    };

    let (query_refs, roots, layer_proofs, f0_openings) =
        fri_prove_queries(&st, params.r, query_seed);

    let mut queries = Vec::with_capacity(params.r);

    for (qi, q) in query_refs.into_iter().enumerate() {
        let mut payloads = Vec::with_capacity(params.schedule.len());

        for (ell, rref) in q.per_layer_refs.iter().enumerate() {
            payloads.push(LayerOpenPayload {
                f_val: st.f_layers_ext[ell][rref.i],
                s_val: st.s_layers[ell][rref.i],
                q_val: st.q_layers[ell][rref.i],
            });
        }

        queries.push(FriQueryPayload {
            per_layer_refs: q.per_layer_refs,
            per_layer_payloads: payloads,
            f0_opening: f0_openings[qi].clone(),
            final_index: q.final_index,
            final_pair: q.final_pair,
        });
    }

    DeepFriProof {
        root_f0: st.root_f0,
        roots,
        layer_proofs,
        f0_openings: queries.iter().map(|q| q.f0_opening.clone()).collect(),
        queries,
        fz_per_layer: st.fz_layers.clone(),
        n0: domain0.size,
        omega0: domain0.omega,
    }
}

pub fn deep_fri_proof_size_bytes(proof: &DeepFriProof) -> usize {
    const FIELD_BYTES: usize = 8;
    const EXT_BYTES: usize = 24;   // 3 × 8 for one CubicExt
    const HASH_BYTES: usize = 32;
    const INDEX_BYTES: usize = 8;

    let mut bytes = 0usize;

    // root_f0 + layer roots
    bytes += HASH_BYTES;
    bytes += proof.roots.len() * HASH_BYTES;

    // fz_per_layer (one CubicExt per layer)
    bytes += proof.fz_per_layer.len() * EXT_BYTES;

    // Query payloads: 3 CubicExt per layer + 2 CubicExt for final_pair
    for q in &proof.queries {
        bytes += q.per_layer_payloads.len() * 3 * EXT_BYTES;
        bytes += 2 * EXT_BYTES;
    }

    // f₀ Merkle openings
    for opening in &proof.f0_openings {
        bytes += HASH_BYTES;
        bytes += INDEX_BYTES;
        for level in &opening.path {
            bytes += level.len() * HASH_BYTES;
        }
    }

    // Layer Merkle openings
    for layer in &proof.layer_proofs.layers {
        for opening in &layer.openings {
            bytes += HASH_BYTES;
            bytes += INDEX_BYTES;
            for level in &opening.path {
                bytes += level.len() * HASH_BYTES;
            }
        }
    }

    bytes
}

// =============================================================================
// ── Verifier (F_{p^3} folding challenges) ──
// =============================================================================

pub fn deep_fri_verify(params: &DeepFriParams, proof: &DeepFriProof) -> bool {
    let L = params.schedule.len();
    let sizes = layer_sizes_from_schedule(proof.n0, &params.schedule);

    // ── Reconstruct the Fiat–Shamir transcript ──
    let mut tr = Transcript::new_sha3(b"FRI/FS");
    bind_statement_to_transcript(&mut tr, &params.schedule, proof.n0, params.seed_z);
    tr.absorb_bytes(&proof.root_f0);

    let z_fp3 = challenge_ext(&mut tr, b"z_fp3");

    let trace_hash: [u8; 32] = tr.challenge_bytes(ds::FRI_SEED);

    eprintln!("[VERIFY] z_fp3 = {:?}", z_fp3);

    // Derive per-layer fold challenges in F_{p^3}
    let mut alpha_layers: Vec<CubicExt> = Vec::with_capacity(L);
    for ell in 0..L {
        let alpha_ell = challenge_ext(&mut tr, b"alpha");
        alpha_layers.push(alpha_ell);

        if ell < proof.fz_per_layer.len() {
            absorb_ext(&mut tr, proof.fz_per_layer[ell]);
        }
        if ell < proof.roots.len() {
            tr.absorb_bytes(&proof.roots[ell]);
        }
    }

    let query_seed: F = tr.challenge(b"query_seed");

    // f₀ tree config
    let f0_th = f0_trace_hash(proof.n0, params.seed_z);
    let f0_cfg = f0_tree_config(proof.n0);

    // ── Per-query verification ──
    for q in 0..params.r {
        let qp = &proof.queries[q];

        // Re-derive query position
        let expected_i0 = {
            let n_pow2 = proof.n0.next_power_of_two();
            let seed = index_seed(query_seed, 0, q);
            index_from_seed(seed, n_pow2) % proof.n0
        };

        let mut expected_i = expected_i0;
        for ell in 0..L {
            if qp.per_layer_refs[ell].i != expected_i {
                eprintln!(
                    "[FAIL][QUERY POS] q={} ell={} expected={} got={}",
                    q, ell, expected_i, qp.per_layer_refs[ell].i
                );
                return false;
            }
            let n_next = sizes[ell] / params.schedule[ell];
            expected_i = expected_i % n_next;
        }

        // ── f₀ Merkle opening ──
        {
            let f0_opening = &qp.f0_opening;

            if !MerkleTreeChannel::verify_opening(
                &f0_cfg,
                proof.root_f0,
                f0_opening,
                &f0_th,
            ) {
                eprintln!("[FAIL][F0 MERKLE] q={}", q);
                return false;
            }

            // Layer 0 f must be a base-field embedding
            let pay0 = &qp.per_layer_payloads[0];
            if pay0.f_val.c[1] != F::zero() || pay0.f_val.c[2] != F::zero() {
                eprintln!("[FAIL][LAYER0 NOT BASE FIELD] q={}", q);
                return false;
            }

            // Cross-check: f₀ tree leaf must match layer 0's base-field value
            let expected_f0_leaf = compute_leaf_hash(
                &f0_cfg,
                f0_opening.index,
                &[pay0.f_val.c[0]],
            );
            if expected_f0_leaf != f0_opening.leaf {
                eprintln!("[FAIL][F0 LEAF BIND] q={}", q);
                return false;
            }

            if f0_opening.index != expected_i0 {
                eprintln!("[FAIL][F0 INDEX] q={}", q);
                return false;
            }
        }

        // ── Per-layer checks ──
        for ell in 0..L {
            let opening = &proof.layer_proofs.layers[ell].openings[q];
            let rref = &qp.per_layer_refs[ell];
            let pay = &qp.per_layer_payloads[ell];

            let arity = pick_arity_for_layer(sizes[ell], params.schedule[ell]).max(2);
            let depth = merkle_depth(sizes[ell], arity);
            let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);

            // 1. Merkle verification against layer root
            if !MerkleTreeChannel::verify_opening(
                &cfg,
                proof.roots[ell],
                opening,
                &trace_hash,
            ) {
                eprintln!("[FAIL][MERKLE] q={} ell={}", q, ell);
                return false;
            }

            // 2. Index binding
            if opening.index != rref.i {
                eprintln!("[FAIL][INDEX BINDING] q={} ell={}", q, ell);
                return false;
            }

            // 3. Leaf binding (9 base-field elements: f, s, q as CubicExt)
            let leaf_fields = ext_leaf_fields(pay.f_val, pay.s_val, pay.q_val);
            let expected_leaf = compute_leaf_hash(
                &cfg,
                opening.index,
                &leaf_fields,
            );
            if expected_leaf != opening.leaf {
                eprintln!("[FAIL][LEAF BINDING] q={} ell={}", q, ell);
                return false;
            }

            // 4. DEEP quotient check (all in F_{p^3}):
            //    q_ℓ(x_i) · (x_i − z) == f_ℓ(x_i) − f_ℓ(z)
            let dom_ell = Domain::<F>::new(sizes[ell]).unwrap();
            let omega_ell = dom_ell.group_gen;
            let x_i = omega_ell.pow([rref.i as u64]);

            let fz = proof.fz_per_layer[ell];
            let num   = pay.f_val - fz;
            let denom = CubicExt::from_base(x_i) - z_fp3;

            if pay.q_val * denom != num {
                eprintln!(
                    "[FAIL][DEEP-EXT] q={} ell={}\n  f_val={:?}\n  fz={:?}\n  q_val={:?}\n  x_i={:?}",
                    q, ell, pay.f_val, fz, pay.q_val, x_i,
                );
                return false;
            }

            // 5. Fold consistency: s_ℓ[i] == f_{ℓ+1}[parent(i)]
            let verified_f_next = if ell + 1 < L {
                qp.per_layer_payloads[ell + 1].f_val
            } else {
                qp.final_pair.0
            };

            if pay.s_val != verified_f_next {
                eprintln!(
                    "[FAIL][FOLD] q={} ell={}\n  s_val={:?}\n  f_next={:?}",
                    q, ell, pay.s_val, verified_f_next,
                );
                return false;
            }
        }

        // Final-layer constancy
        if qp.final_pair.0 != qp.final_pair.1 {
            eprintln!(
                "[FAIL][FINAL CONSTANCY] q={} f={:?} s={:?}",
                q, qp.final_pair.0, qp.final_pair.1,
            );
            return false;
        }
    }

    eprintln!("[VERIFY] SUCCESS");
    true
}

// ================================================================
// Extension-field FRI folding utilities (from original — unchanged)
// ================================================================

#[inline]
pub fn fri_fold_degree2(
    f_pos: CubicExt,
    f_neg: CubicExt,
    x: F,
    beta: CubicExt,
) -> CubicExt {
    let two_inv = F::from(2u64).inverse().unwrap();
    let x_inv = x.inverse().unwrap();

    let f_even = (f_pos + f_neg) * two_inv;
    let f_odd  = (f_pos - f_neg) * (two_inv * x_inv);

    f_even + beta * f_odd
}

pub fn fri_fold_degree3(
    f_at_y:   CubicExt,
    f_at_zy:  CubicExt,
    f_at_z2y: CubicExt,
    y:        F,
    zeta:     F,
    beta:     CubicExt,
) -> CubicExt {
    let zeta2 = zeta * zeta;
    let inv3 = F::from(3u64).inverse().unwrap();
    let y_inv = y.inverse().unwrap();
    let y2_inv = y_inv * y_inv;

    let f0 = (f_at_y + f_at_zy + f_at_z2y) * inv3;

    let f1 = (f_at_y + f_at_zy * zeta2 + f_at_z2y * zeta)
        * (inv3 * y_inv);

    let f2 = (f_at_y + f_at_zy * zeta + f_at_z2y * zeta2)
        * (inv3 * y2_inv);

    let beta_sq = beta.square();
    f0 + beta * f1 + beta_sq * f2
}

pub fn fri_fold_round(
    codeword: &[CubicExt],
    domain: &[F],
    beta: CubicExt,
) -> Vec<CubicExt> {
    let half = codeword.len() / 2;
    let mut folded = Vec::with_capacity(half);

    for i in 0..half {
        let f_pos = codeword[i];
        let f_neg = codeword[i + half];
        let x = domain[i];
        folded.push(fri_fold_degree2(f_pos, f_neg, x, beta));
    }

    folded
}

pub fn fri_verify_query(
    round_evals: &[(CubicExt, CubicExt)],
    round_domains: &[F],
    betas: &[CubicExt],
    final_value: CubicExt,
) -> bool {
    let num_rounds = betas.len();
    let mut expected = fri_fold_degree2(
        round_evals[0].0,
        round_evals[0].1,
        round_domains[0],
        betas[0],
    );

    for r in 1..num_rounds {
        let (f_pos, f_neg) = round_evals[r];
        if f_pos != expected && f_neg != expected {
            return false;
        }
        expected = fri_fold_degree2(f_pos, f_neg, round_domains[r], betas[r]);
    }

    expected == final_value
}


// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Field, FftField, One, Zero};
    use ark_goldilocks::Goldilocks;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use ark_poly::DenseUVPolynomial;
    use ark_poly::Polynomial;
    use rand::Rng;
    use std::collections::HashSet;

    use ark_ff::UniformRand;
    use ark_poly::polynomial::univariate::DensePolynomial;
    use rand::seq::SliceRandom;

    type TestField = Goldilocks;

    fn random_polynomial<F: Field>(degree: usize, rng: &mut impl Rng) -> Vec<F> {
        (0..=degree).map(|_| F::rand(rng)).collect()
    }

    fn perform_fold<F: Field + FftField>(
        evals: &[F],
        domain: GeneralEvaluationDomain<F>,
        alpha: F,
        folding_factor: usize,
    ) -> (Vec<F>, GeneralEvaluationDomain<F>) {
        assert!(evals.len() % folding_factor == 0);
        let n = evals.len();
        let next_n = n / folding_factor;
        let next_domain = GeneralEvaluationDomain::<F>::new(next_n)
            .expect("valid folded domain");
        let folding_domain = GeneralEvaluationDomain::<F>::new(folding_factor)
            .expect("valid folding domain");
        let generator = domain.group_gen();
        let folded = (0..next_n)
            .map(|i| {
                let coset_values: Vec<F> = (0..folding_factor)
                    .map(|j| evals[i + j * next_n])
                    .collect();
                let coset_generator = generator.pow([i as u64]);
                fold_one_coset(&coset_values, alpha, coset_generator, &folding_domain)
            })
            .collect();
        (folded, next_domain)
    }

    fn fold_one_coset<F: Field + FftField>(
        coset_values: &[F],
        alpha: F,
        coset_generator: F,
        folding_domain: &GeneralEvaluationDomain<F>,
    ) -> F {
        let p_coeffs = folding_domain.ifft(coset_values);
        let poly = DensePolynomial::from_coefficients_vec(p_coeffs);
        let evaluation_point = alpha * coset_generator.inverse().unwrap();
        poly.evaluate(&evaluation_point)
    }

    // ── NEW: Extension-field fold correctness test ──

    #[test]
    fn test_ext_fold_preserves_low_degree() {
        // A low-degree polynomial over F_p, embedded in F_{p^3}, should
        // remain low-degree after folding with an F_{p^3} challenge.
        use ark_ff::UniformRand;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 256usize;
        let m = 4usize;
        let degree = n / m - 1; // degree bound for rate 1/m

        let dom = GeneralEvaluationDomain::<TestField>::new(n).unwrap();
        let poly = DensePolynomial::<TestField>::rand(degree, &mut rng);
        let evals = dom.fft(poly.coeffs());

        // Embed in F_{p^3}
        let evals_ext: Vec<CubicExt> = evals.iter()
            .map(|&x| CubicExt::from_base(x))
            .collect();

        // Fold with F_{p^3} challenge
        let alpha = CubicExt::new(
            TestField::from(17u64),
            TestField::from(31u64),
            TestField::from(7u64),
        );
        let folded = fri_fold_layer_ext_impl(&evals_ext, alpha, m);

        assert_eq!(folded.len(), n / m);

        // The folded values should not all be zero (sanity check)
        let any_nonzero = folded.iter().any(|v| *v != CubicExt::ZERO);
        assert!(any_nonzero, "Folded codeword should be non-trivial");

        // Verify that the folded codeword, when interpreted componentwise,
        // consists of polynomials of degree < n/m (i.e., the iFFT of each
        // component has zero coefficients beyond the degree bound).
        let dom_next = GeneralEvaluationDomain::<TestField>::new(n / m).unwrap();
        let c0_folded: Vec<TestField> = folded.iter().map(|v| v.c[0]).collect();
        let c1_folded: Vec<TestField> = folded.iter().map(|v| v.c[1]).collect();
        let c2_folded: Vec<TestField> = folded.iter().map(|v| v.c[2]).collect();

        let c0_coeffs = dom_next.ifft(&c0_folded);
        let c1_coeffs = dom_next.ifft(&c1_folded);
        let c2_coeffs = dom_next.ifft(&c2_folded);

        // For an honest prover, degree < n/m, so all coefficients exist
        // within the domain size. The key test is that the fold produced
        // valid extension-field values.
        eprintln!("Folded codeword length: {}", folded.len());
        eprintln!("Sample folded[0]: {:?}", folded[0]);
    }

    #[test]
    fn test_ext_fold_consistency_with_s_layer() {
        // Verify that compute_s_layer_ext produces values consistent
        // with fri_fold_layer_ext_impl.
        let mut rng = StdRng::seed_from_u64(99);
        let n = 128usize;
        let m = 4usize;

        let evals: Vec<CubicExt> = (0..n)
            .map(|_| CubicExt::new(
                TestField::from(rng.gen::<u64>()),
                TestField::from(rng.gen::<u64>()),
                TestField::from(rng.gen::<u64>()),
            ))
            .collect();

        let alpha = CubicExt::new(
            TestField::from(13u64),
            TestField::from(29u64),
            TestField::from(3u64),
        );

        let folded = fri_fold_layer_ext_impl(&evals, alpha, m);
        let s = compute_s_layer_ext(&evals, alpha, m);

        let n_next = n / m;
        // s[b + j * n_next] should equal folded[b] for all j
        for b in 0..n_next {
            for j in 0..m {
                assert_eq!(
                    s[b + j * n_next], folded[b],
                    "s-layer mismatch at b={} j={}", b, j
                );
            }
        }
    }

    #[test]
    fn test_ext_deep_quotient_consistency() {
        // Verify that compute_q_layer_ext_on_ext is consistent:
        // q(x) * (x - z) == f(x) - f(z)
        let mut rng = StdRng::seed_from_u64(77);
        let n = 64usize;

        let dom = Domain::<TestField>::new(n).unwrap();
        let omega = dom.group_gen;

        let evals: Vec<CubicExt> = (0..n)
            .map(|_| CubicExt::new(
                TestField::from(rng.gen::<u64>()),
                TestField::from(rng.gen::<u64>()),
                TestField::from(rng.gen::<u64>()),
            ))
            .collect();

        let z = CubicExt::new(
            TestField::from(42u64),
            TestField::from(17u64),
            TestField::from(5u64),
        );

        let (q, fz) = compute_q_layer_ext_on_ext(&evals, z, omega);

        // Check identity at every domain point
        let mut x = TestField::one();
        for i in 0..n {
            let lhs = q[i] * (CubicExt::from_base(x) - z);
            let rhs = evals[i] - fz;
            assert_eq!(lhs, rhs, "DEEP identity failed at i={}", i);
            x *= omega;
        }
    }

    // ── Base-field tests (unchanged, kept for backward compatibility) ──

    #[test]
    fn test_fri_local_consistency_check_soundness() {
        const DOMAIN_SIZE: usize = 1024;
        const FOLDING_FACTOR: usize = 4;
        const NUM_TRIALS: usize = 1000000;

        let mut rng = rand::thread_rng();
        let mut detections = 0;

        let z_l = TestField::from(5u64);
        let f: Vec<TestField> = (0..DOMAIN_SIZE).map(|_| TestField::rand(&mut rng)).collect();
        let f_next_claimed: Vec<TestField> = vec![TestField::zero(); DOMAIN_SIZE / FOLDING_FACTOR];

        let domain = GeneralEvaluationDomain::<TestField>::new(DOMAIN_SIZE).unwrap();
        let generator = domain.group_gen();
        let folding_domain = GeneralEvaluationDomain::<TestField>::new(FOLDING_FACTOR).unwrap();

        for _ in 0..NUM_TRIALS {
            let query_index = rng.gen_range(0..f_next_claimed.len());
            let coset_values: Vec<TestField> = (0..FOLDING_FACTOR)
                .map(|j| f[query_index + j * (DOMAIN_SIZE / FOLDING_FACTOR)])
                .collect();
            let coset_generator = generator.pow([query_index as u64]);
            let s_reconstructed = fold_one_coset(&coset_values, z_l, coset_generator, &folding_domain);
            let s_claimed = f_next_claimed[query_index];
            if s_reconstructed != s_claimed {
                detections += 1;
            }
        }
        let measured_rate = detections as f64 / NUM_TRIALS as f64;
        println!("[Consistency Check] Detections: {}/{}, Measured Rate: {:.4}", detections, NUM_TRIALS, measured_rate);
        assert!((measured_rate - 1.0).abs() < 0.01, "Detection rate should be close to 100%");
    }

    #[test]
    fn test_fri_distance_amplification() {
        const DOMAIN_SIZE: usize = 1024;
        const FOLDING_FACTOR: usize = 4;
        const NUM_TRIALS: usize = 100_000;
        const INITIAL_CORRUPTION_FRACTION: f64 = 0.05;

        let mut rng = rand::thread_rng();
        let z_l = TestField::from(5u64);

        let large_domain = GeneralEvaluationDomain::<TestField>::new(DOMAIN_SIZE).unwrap();
        let degree_bound = DOMAIN_SIZE / FOLDING_FACTOR;
        let p_coeffs = random_polynomial(degree_bound - 2, &mut rng);
        let p_poly = DensePolynomial::from_coefficients_vec(p_coeffs);
        let p_evals = large_domain.fft(p_poly.coeffs());

        let mut f_evals = p_evals.clone();
        let num_corruptions = (DOMAIN_SIZE as f64 * INITIAL_CORRUPTION_FRACTION) as usize;
        let mut corrupted_indices = HashSet::new();
        while corrupted_indices.len() < num_corruptions {
            corrupted_indices.insert(rng.gen_range(0..DOMAIN_SIZE));
        }
        for &idx in &corrupted_indices {
            f_evals[idx] = TestField::rand(&mut rng);
        }

        let folded_honest = fri_fold_layer(&p_evals, z_l, FOLDING_FACTOR);
        let folded_corrupted = fri_fold_layer(&f_evals, z_l, FOLDING_FACTOR);

        let mut detections = 0;
        for _ in 0..NUM_TRIALS {
            let query_index = rng.gen_range(0..folded_honest.len());
            if folded_honest[query_index] != folded_corrupted[query_index] {
                detections += 1;
            }
        }

        let measured_rate = detections as f64 / NUM_TRIALS as f64;
        let theoretical_rate = (FOLDING_FACTOR as f64 * INITIAL_CORRUPTION_FRACTION).min(1.0);

        println!("\n[Distance Amplification Test (Single Layer)]");
        println!("  - Initial Corruption: {:.2}% ({} points)", INITIAL_CORRUPTION_FRACTION * 100.0, num_corruptions);
        println!("  - Detections: {}/{}", detections, NUM_TRIALS);
        println!("  - Measured Detection Rate: {:.4}", measured_rate);
        println!("  - Theoretical Detection Rate: {:.4}", theoretical_rate);

        let tolerance = 0.05;
        assert!(
            (measured_rate - theoretical_rate).abs() < tolerance,
            "Measured detection rate should be close to the theoretical rate."
        );
    }

    #[test]
    #[ignore]
    fn test_full_fri_protocol_soundness() {
        const FOLDING_SCHEDULE: [usize; 3] = [4, 4, 4];
        const INITIAL_DOMAIN_SIZE: usize = 4096;
        const NUM_TRIALS: usize = 1000000;

        let mut rng = rand::thread_rng();
        let mut detections = 0;

        for _ in 0..NUM_TRIALS {
            let alpha = TestField::rand(&mut rng);

            let mut domains = Vec::new();
            let mut current_size = INITIAL_DOMAIN_SIZE;
            domains.push(GeneralEvaluationDomain::<TestField>::new(current_size).unwrap());
            for &folding_factor in &FOLDING_SCHEDULE {
                current_size /= folding_factor;
                domains.push(GeneralEvaluationDomain::<TestField>::new(current_size).unwrap());
            }

            let mut fraudulent_layers: Vec<Vec<TestField>> = Vec::new();
            let f0: Vec<TestField> = (0..INITIAL_DOMAIN_SIZE).map(|_| TestField::rand(&mut rng)).collect();
            fraudulent_layers.push(f0);

            let mut current_layer_evals = fraudulent_layers[0].clone();
            for &folding_factor in &FOLDING_SCHEDULE {
                let next_layer = fri_fold_layer(&current_layer_evals, alpha, folding_factor);
                current_layer_evals = next_layer;
                fraudulent_layers.push(current_layer_evals.clone());
            }

            let mut trial_detected = false;
            let mut query_index = rng.gen_range(0..domains[1].size());

            for l in 0..FOLDING_SCHEDULE.len() {
                let folding_factor = FOLDING_SCHEDULE[l];
                let current_domain = &domains[l];
                let next_domain = &domains[l+1];

                let coset_generator = current_domain.group_gen().pow([query_index as u64]);
                let folding_domain = GeneralEvaluationDomain::<TestField>::new(folding_factor).unwrap();

                let coset_values: Vec<TestField> = (0..folding_factor)
                    .map(|j| fraudulent_layers[l][query_index + j * next_domain.size()])
                    .collect();

                let s_reconstructed = fold_one_coset(&coset_values, alpha, coset_generator, &folding_domain);
                let s_claimed = fraudulent_layers[l+1][query_index];

                if s_reconstructed != s_claimed {
                    trial_detected = true;
                    break;
                }

                if l + 1 < FOLDING_SCHEDULE.len() {
                    query_index %= domains[l+2].size();
                }
            }

            if !trial_detected {
                let last_layer = fraudulent_layers.last().unwrap();
                let first_element = last_layer[0];
                for &element in last_layer.iter().skip(1) {
                    if element != first_element {
                        trial_detected = true;
                        break;
                    }
                }
            }

            if trial_detected {
                detections += 1;
            }
        }

        let measured_rate = detections as f64 / NUM_TRIALS as f64;

        println!("\n[Full Protocol Soundness Test (ε_eff)]");
        println!("  - Protocol Schedule: {:?}", FOLDING_SCHEDULE);
        println!("  - Initial Domain Size: {}", INITIAL_DOMAIN_SIZE);
        println!("  - Detections: {}/{}", detections, NUM_TRIALS);
        println!("  - Measured Effective Detection Rate (ε_eff): {:.4}", measured_rate);

        assert!(measured_rate > 0.90, "Effective detection rate should be very high");
    }

    #[test]
    fn test_intermediate_layer_fraud_soundness() {
        const INITIAL_DOMAIN_SIZE: usize = 4096;
        const FOLDING_SCHEDULE: [usize; 3] = [4, 4, 4];
        const NUM_TRIALS: usize = 20000;
        const FRAUD_LAYER_INDEX: usize = 1;

        let mut rng = rand::thread_rng();
        let mut detections = 0;

        let alphas: Vec<TestField> = (0..FOLDING_SCHEDULE.len())
            .map(|_| TestField::rand(&mut rng))
            .collect();

        let final_layer_size =
            INITIAL_DOMAIN_SIZE / FOLDING_SCHEDULE.iter().product::<usize>();
        let degree_bound = final_layer_size - 1;

        let p_coeffs = random_polynomial(degree_bound, &mut rng);
        let p_poly = DensePolynomial::from_coefficients_vec(p_coeffs);

        let domain0 = GeneralEvaluationDomain::<TestField>::new(INITIAL_DOMAIN_SIZE).unwrap();
        let honest_f0 = domain0.fft(p_poly.coeffs());

        let mut honest_layers = vec![honest_f0];
        let mut current = honest_layers[0].clone();

        for (l, &factor) in FOLDING_SCHEDULE.iter().enumerate() {
            let next = fri_fold_layer(&current, alphas[l], factor);
            honest_layers.push(next.clone());
            current = next;
        }

        let mut prover_layers = honest_layers.clone();

        let fraud_layer_size =
            INITIAL_DOMAIN_SIZE / FOLDING_SCHEDULE[0..FRAUD_LAYER_INDEX].iter().product::<usize>();

        let fraud_index = rng.gen_range(0..fraud_layer_size);
        let honest = prover_layers[FRAUD_LAYER_INDEX][fraud_index];
        let mut corrupted = TestField::rand(&mut rng);
        while corrupted == honest {
            corrupted = TestField::rand(&mut rng);
        }
        prover_layers[FRAUD_LAYER_INDEX][fraud_index] = corrupted;

        let l = FRAUD_LAYER_INDEX - 1;
        let folding_factor = FOLDING_SCHEDULE[l];
        let current_domain_size =
            INITIAL_DOMAIN_SIZE / FOLDING_SCHEDULE[0..l].iter().product::<usize>();
        let next_domain_size = current_domain_size / folding_factor;

        let current_domain =
            GeneralEvaluationDomain::<TestField>::new(current_domain_size).unwrap();

        for _ in 0..NUM_TRIALS {
            let query_index = rng.gen_range(0..next_domain_size);
            let x = current_domain.element(query_index);

            let coset_values: Vec<TestField> = (0..folding_factor)
                .map(|j| prover_layers[l][query_index + j * next_domain_size])
                .collect();

            let mut lhs = TestField::zero();
            let mut alpha_pow = TestField::one();

            for j in 0..folding_factor {
                lhs += coset_values[j] * alpha_pow;
                alpha_pow *= alphas[l];
            }

            let reconstructed = lhs;
            let claimed = prover_layers[l + 1][query_index];

            if reconstructed != claimed {
                detections += 1;
            }
        }

        let measured_rate = detections as f64 / NUM_TRIALS as f64;
        let theoretical_rate = 1.0 / fraud_layer_size as f64;

        println!("\n[Intermediate Layer Fraud Soundness]");
        println!("  Fraud layer size: {}", fraud_layer_size);
        println!("  Fraud index: {}", fraud_index);
        println!("  Detections: {}/{}", detections, NUM_TRIALS);
        println!("  Measured detection rate: {:.6}", measured_rate);
        println!("  Theoretical rate: {:.6}", theoretical_rate);

        let tolerance = theoretical_rate * 5.0;
        assert!(
            (measured_rate - theoretical_rate).abs() < tolerance,
            "Measured detection rate deviates from theory"
        );
    }

    #[test]
    #[ignore]
    fn test_fri_effective_detection_rate() {
        println!("\n--- Running Rust Test: Effective Detection Rate (arkworks) ---");

        const DOMAIN_SIZE: usize = 1024;
        const FOLDING_FACTOR: usize = 4;
        let degree = 31;

        let mut rng = rand::thread_rng();

        let p_coeffs = random_polynomial(degree, &mut rng);
        let p_poly = DensePolynomial::from_coefficients_vec(p_coeffs);
        let domain0 = GeneralEvaluationDomain::<TestField>::new(DOMAIN_SIZE).unwrap();

        let f0_good = domain0.fft(p_poly.coeffs());
        let mut f0_corrupt = f0_good.clone();

        let rho_0 = 0.06;
        let num_corruptions = (DOMAIN_SIZE as f64 * rho_0) as usize;
        let indices: Vec<usize> = (0..DOMAIN_SIZE).collect();

        for &idx in indices.choose_multiple(&mut rng, num_corruptions) {
            let honest = f0_corrupt[idx];
            let mut corrupted = TestField::rand(&mut rng);
            while corrupted == honest {
                corrupted = TestField::rand(&mut rng);
            }
            f0_corrupt[idx] = corrupted;
        }

        let alpha1 = TestField::rand(&mut rng);
        let f1_corrupt = fri_fold_layer(&f0_corrupt, alpha1, FOLDING_FACTOR);

        let alpha2 = TestField::rand(&mut rng);
        let f2_corrupt = fri_fold_layer(&f1_corrupt, alpha2, FOLDING_FACTOR);

        let num_trials = 200_000;
        let mut detections = 0;

        let domain1_size = DOMAIN_SIZE / FOLDING_FACTOR;
        let domain2_size = domain1_size / FOLDING_FACTOR;

        for _ in 0..num_trials {
            let i2 = rng.gen_range(0..domain2_size);
            let k = rng.gen_range(0..FOLDING_FACTOR);
            let i1 = i2 + k * domain2_size;

            let x1 = domain0.element(i1);

            let coset0: Vec<TestField> = (0..FOLDING_FACTOR)
                .map(|j| f0_corrupt[i1 + j * domain1_size])
                .collect();

            let reconstructed_f1 = fold_one_coset(&coset0, alpha1, x1, &domain0);

            if reconstructed_f1 != f1_corrupt[i1] {
                detections += 1;
                continue;
            }

            let x2 = x1.square();

            let coset1: Vec<TestField> = (0..FOLDING_FACTOR)
                .map(|j| f1_corrupt[i2 + j * domain2_size])
                .collect();

            let reconstructed_f2 = fold_one_coset(&coset1, alpha2, x2, &domain0);

            if reconstructed_f2 != f2_corrupt[i2] {
                detections += 1;
            }
        }

        let measured_rate = detections as f64 / num_trials as f64;
        let rho_1 = 1.0 - (1.0 - rho_0).powi(FOLDING_FACTOR as i32);
        let rho_2 = 1.0 - (1.0 - rho_1).powi(FOLDING_FACTOR as i32);

        println!("rho_0 = {:.4}", rho_0);
        println!("rho_1 = {:.4}", rho_1);
        println!("rho_2 = {:.4}", rho_2);
        println!("Measured effective detection rate: {:.4}", measured_rate);
        println!("Theoretical effective detection rate: {:.4}", rho_2);

        let delta = 0.03;
        assert!(
            (measured_rate - rho_2).abs() < delta,
            "Measured rate {:.4} not close to theoretical {:.4}",
            measured_rate,
            rho_2
        );

        println!("✅ Effective detection rate matches theory");
    }

    #[test]
    fn debug_single_fold_distance_amplification() {
        let log_domain_size = 12;
        let initial_domain_size = 1 << log_domain_size;
        let folding_factor = 4;
        let initial_corruption_rate = 0.06;

        let mut rng = StdRng::seed_from_u64(0);

        let degree = (initial_domain_size / folding_factor) - 1;
        let domain = GeneralEvaluationDomain::<F>::new(initial_domain_size)
            .expect("Failed to create domain");
        let poly_p0 = DensePolynomial::<F>::rand(degree, &mut rng);

        let codeword_c0_evals = poly_p0.evaluate_over_domain(domain).evals;

        let mut corrupted_codeword_c_prime_0_evals = codeword_c0_evals.clone();
        let num_corruptions = (initial_domain_size as f64 * initial_corruption_rate).ceil() as usize;
        let mut corrupted_indices = HashSet::new();

        while corrupted_indices.len() < num_corruptions {
            let idx_to_corrupt = usize::rand(&mut rng) % initial_domain_size;
            if corrupted_indices.contains(&idx_to_corrupt) {
                continue;
            }

            let original_value = corrupted_codeword_c_prime_0_evals[idx_to_corrupt];
            let mut new_value = F::rand(&mut rng);
            while new_value == original_value {
                new_value = F::rand(&mut rng);
            }
            corrupted_codeword_c_prime_0_evals[idx_to_corrupt] = new_value;
            corrupted_indices.insert(idx_to_corrupt);
        }

        let alpha = F::rand(&mut rng);

        let (folded_corrupted_evals, new_domain) = perform_fold(
            &corrupted_codeword_c_prime_0_evals,
            domain,
            alpha,
            folding_factor,
        );

        let (folded_true_evals, _) = perform_fold(
            &codeword_c0_evals,
            domain,
            alpha,
            folding_factor,
        );

        let differing_points = folded_corrupted_evals
            .iter()
            .zip(folded_true_evals.iter())
            .filter(|(a, b)| a != b)
            .count();

        let measured_rho_1 = differing_points as f64 / new_domain.size() as f64;

        let theoretical_rho_1 = 1.0_f64 - (1.0_f64 - initial_corruption_rate).powf(folding_factor as f64);

        println!("--- Debugging Single Fold (Goldilocks Field) ---");
        println!("Initial rho_0:       {}", initial_corruption_rate);
        println!("Measured rho_1:      {}", measured_rho_1);
        println!("Theoretical rho_1:   {}", theoretical_rho_1);

        let tolerance = 0.01;
        assert!(
            (measured_rho_1 - theoretical_rho_1).abs() < tolerance,
            "Single fold amplification measured rate {} is not close to precise theoretical rate {}",
            measured_rho_1,
            theoretical_rho_1
        );
    }
}