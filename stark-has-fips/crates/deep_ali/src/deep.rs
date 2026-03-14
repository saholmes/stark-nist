// ================================================================
// deep.rs — DEEP quotient computation over the extension field
//
// The DEEP technique evaluates the trace polynomial at a random
// out-of-domain point  z ∈ GF(p³)  and reduces the proximity test
// to a quotient polynomial.
//
// CRITICAL CHANGE:  (x − z)  where  x ∈ Fp, z ∈ CubicExt
//   is an extension-field element, and dividing by it requires
//   a true extension-field inversion (not componentwise).
// ================================================================

use crate::cubic_ext::CubicExt;
use ark_goldilocks::Goldilocks as Fp;

/// Compute the DEEP quotient codeword.
///
/// Given:
///   - `trace_codeword[i]` = T(d_i) ∈ Fp   (trace poly evaluated on domain)
///   - `domain[i]` = d_i ∈ Fp               (evaluation domain)
///   - `z` ∈ CubicExt                       (out-of-domain challenge point)
///   - `claimed_eval` = T(z) ∈ CubicExt     (claimed evaluation)
///
/// Computes for each i:
///
///   q(d_i) = (T(d_i) − T(z)) / (d_i − z)   ∈ CubicExt
///
/// Note: T(d_i) ∈ Fp is embedded into CubicExt, and (d_i − z) is
/// a nonzero extension-field element (since d_i ∈ Fp but z ∉ Fp).
pub fn compute_deep_quotient(
    trace_codeword: &[Fp],
    domain: &[Fp],
    z: CubicExt,
    claimed_eval: CubicExt,
) -> Vec<CubicExt> {
    let n = trace_codeword.len();
    assert_eq!(domain.len(), n);

    // Precompute all (d_i − z) and batch-invert them.
    // Each (d_i − z) = (d_i − z₀) − z₁·α − z₂·α²  ∈ CubicExt.
    let mut denominators: Vec<CubicExt> = domain
        .iter()
        .map(|&d_i| CubicExt::from_base(d_i) - z)
        .collect();

    // Batch inversion:  1 extension-field inversion + O(n) extension muls.
    // This replaces n individual inversions.
    CubicExt::batch_inverse(&mut denominators);

    // Compute the quotient at each domain point.
    let mut quotient = Vec::with_capacity(n);
    for i in 0..n {
        let numerator = CubicExt::from_base(trace_codeword[i]) - claimed_eval;

        // *** CRITICAL OPERATION ***
        // Old code:  componentwise division  ← INSECURE
        // New code:  extension-field multiplication by precomputed inverse
        quotient.push(numerator * denominators[i]);     // ← EXTENSION FIELD MUL
    }

    quotient
}

/// Compute the DEEP quotient for multiple trace columns combined
/// with random verifier weights.
///
/// Given k trace columns T_j, the combined quotient is:
///
///   q(x) = Σ_j  λ_j · (T_j(x) − T_j(z)) / (x − z)
///
/// where λ_j ∈ CubicExt are random combination weights.
///
/// Since all columns share the same denominator (x − z), we
/// compute the combined numerator first then divide once.
pub fn compute_deep_quotient_multi(
    trace_codewords: &[Vec<Fp>],       // k columns, each length n
    domain: &[Fp],                      // length n
    z: CubicExt,                        // OOD point
    claimed_evals: &[CubicExt],         // T_j(z) for each column
    lambdas: &[CubicExt],              // random combination weights
) -> Vec<CubicExt> {
    let n = domain.len();
    let k = trace_codewords.len();
    assert_eq!(claimed_evals.len(), k);
    assert_eq!(lambdas.len(), k);

    // Batch-invert denominators
    let mut denom_inv: Vec<CubicExt> = domain
        .iter()
        .map(|&d_i| CubicExt::from_base(d_i) - z)
        .collect();
    CubicExt::batch_inverse(&mut denom_inv);

    // Combined quotient
    let mut quotient = vec![CubicExt::ZERO; n];
    for j in 0..k {
        for i in 0..n {
            let numer = CubicExt::from_base(trace_codewords[j][i]) - claimed_evals[j];
            // λ_j · (T_j(d_i) − T_j(z)) · (d_i − z)⁻¹
            quotient[i] += lambdas[j] * numer * denom_inv[i];  // ← EXTENSION FIELD
        }
    }

    quotient
}

/// Verify the DEEP evaluation claim for a single column.
///
/// Given a base-field polynomial T of degree < d, and the claim
/// T(z) = v  for z ∈ CubicExt, v ∈ CubicExt:
///
/// Evaluates T at z using Horner's method in the extension field.
pub fn verify_deep_claim(
    trace_coeffs: &[Fp],    // polynomial coefficients in Fp
    z: CubicExt,             // OOD point in GF(p³)
    claimed: CubicExt,       // claimed T(z)
) -> bool {
    let computed = CubicExt::eval_base_poly(trace_coeffs, z);
    computed == claimed
}
