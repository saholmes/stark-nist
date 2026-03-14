//! Cubic extension field  F_{p^3} = F_p[X] / (X^3 - W)
//!
//! p = Goldilocks prime = 2^64 - 2^32 + 1
//! W = 7  (a cubic non-residue mod p)
//!
//! Elements are represented as  c0 + c1·α + c2·α²  where  α³ = W.

extern crate alloc;
use alloc::vec::Vec;

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ── FIX 1: import the traits that provide ZERO, ONE, inverse(), pow() ──
use ark_ff::{Field, MontFp, PrimeField};

// Goldilocks base field  (Fp<MontBackend<GoldilocksConfig, 2>, 2>)
use ark_goldilocks::Goldilocks as Fp;

// ────────────────────────────────────────────────────────────────────────
//  Constants
// ────────────────────────────────────────────────────────────────────────

// ── FIX 2: Fp::new() expects BigInt<2>, not a bare integer.            ──
//    Use the MontFp! macro — it computes the Montgomery representation  ──
//    at compile time so the const is correct.                            ──
/// Cubic non-residue W = 7.  X^3 − 7 is irreducible over F_p.
pub const W: Fp = MontFp!("7");

// ── FIX 3: There is no Fp::ORDER in ark-ff 0.4.                       ──
//    The modulus lives at <Fp as PrimeField>::MODULUS (a BigInt<2>).     ──
//    Since p = 2^64 − 2^32 + 1, we have                                ──
//        (p − 1)/3 = 6_148_914_689_804_861_440                         ──
//    which fits in a single u64.  Encode as [u64; 2] for pow().         ──
const P_MINUS_1_OVER_3: [u64; 2] = [6_148_914_689_804_861_440u64, 0u64];

/// Runtime check that W is a cubic non-residue: W^{(p-1)/3} ≠ 1.
pub fn verify_cubic_non_residue() {
    // ── FIX 4: .exp() does not exist — the trait method is .pow() ──
    let result = W.pow(P_MINUS_1_OVER_3);
    assert!(
        result != Fp::ONE, // Fp::ONE now resolves via `use ark_ff::Field`
        "W = 7 is NOT a cubic non-residue — irreducibility check failed"
    );
}

/// Primitive cube root of unity  ω = W^{(p-1)/3}.
/// Satisfies ω ≠ 1 and ω³ = 1.
pub fn cube_root_of_unity() -> Fp {
    let omega = W.pow(P_MINUS_1_OVER_3); // FIX 4 again
    debug_assert_ne!(omega, Fp::ONE);
    debug_assert_eq!(omega * omega * omega, Fp::ONE);
    omega
}

// ────────────────────────────────────────────────────────────────────────
//  CubicExt
// ────────────────────────────────────────────────────────────────────────

/// Element of F_{p^3} = F_p[α]/(α³ − W),  stored as [c0, c1, c2].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CubicExt {
    pub c: [Fp; 3],
}

// ── FIX 5: Fp::ZERO / Fp::ONE come from the Field trait.              ──
//    In a `const` context, use MontFp!("0") / MontFp!("1") instead     ──
//    (they are const-evaluable; the trait consts may not be).            ──

impl CubicExt {
    /// Additive identity.
    pub const ZERO: Self = Self {
        c: [MontFp!("0"), MontFp!("0"), MontFp!("0")],
    };

    /// Multiplicative identity.
    pub const ONE: Self = Self {
        c: [MontFp!("1"), MontFp!("0"), MontFp!("0")],
    };

    /// The generator α of the extension.
    pub const ALPHA: Self = Self {
        c: [MontFp!("0"), MontFp!("1"), MontFp!("0")],
    };

    /// α²
    pub const ALPHA2: Self = Self {
        c: [MontFp!("0"), MontFp!("0"), MontFp!("1")],
    };

    /// Construct from three base-field coefficients.
    #[inline]
    pub fn new(c0: Fp, c1: Fp, c2: Fp) -> Self {
        Self { c: [c0, c1, c2] }
    }

    /// Embed a base-field element.
    #[inline]
    pub fn from_base(x: Fp) -> Self {
        Self {
            c: [x, Fp::ZERO, Fp::ZERO], // Field trait is in scope → OK
        }
    }

    /// Test for the additive identity.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.c[0] == Fp::ZERO && self.c[1] == Fp::ZERO && self.c[2] == Fp::ZERO
    }

    /// Test whether the element lies in the base field.
    #[inline]
    pub fn is_base_field(&self) -> bool {
        self.c[1] == Fp::ZERO && self.c[2] == Fp::ZERO
    }

    /// Project to base field if possible.
    pub fn to_base(&self) -> Option<Fp> {
        if self.is_base_field() {
            Some(self.c[0])
        } else {
            None
        }
    }

    // ────────────────────────────────────────────────────────────────
    //  Multiplication   (schoolbook, using α³ = W)
    //
    //   c0 = a0·b0 + W·(a1·b2 + a2·b1)
    //   c1 = a0·b1 + a1·b0 + W·a2·b2
    //   c2 = a0·b2 + a1·b1 + a2·b0
    // ────────────────────────────────────────────────────────────────

    #[inline]
    fn mul_impl(&self, rhs: &Self) -> Self {
        let (a0, a1, a2) = (self.c[0], self.c[1], self.c[2]);
        let (b0, b1, b2) = (rhs.c[0], rhs.c[1], rhs.c[2]);

        let c0 = a0 * b0 + W * (a1 * b2 + a2 * b1);
        let c1 = a0 * b1 + a1 * b0 + W * (a2 * b2);
        let c2 = a0 * b2 + a1 * b1 + a2 * b0;

        Self { c: [c0, c1, c2] }
    }

    /// Squaring (slightly fewer multiplications than generic mul).
    pub fn square(&self) -> Self {
        let (a0, a1, a2) = (self.c[0], self.c[1], self.c[2]);
        let two = a0 + a0; // 2·a0

        let c0 = a0 * a0 + W * (a1 * a2 + a1 * a2);
        let c1 = two * a1 + W * (a2 * a2);
        let c2 = two * a2 + a1 * a1;

        Self { c: [c0, c1, c2] }
    }

    // ────────────────────────────────────────────────────────────────
    //  Norm:  N(a) = det(M_a)  ∈  F_p
    //
    //   M_a = | a0    W·a2   W·a1 |
    //         | a1    a0     W·a2 |
    //         | a2    a1     a0   |
    //
    //   N = a0³ + W·a1³ + W²·a2³ − 3W·a0·a1·a2
    // ────────────────────────────────────────────────────────────────

    pub fn norm(&self) -> Fp {
        let (a0, a1, a2) = (self.c[0], self.c[1], self.c[2]);
        let w2 = W * W;
        let three_w: Fp = MontFp!("3") * W;

        a0 * a0 * a0
            + W * a1 * a1 * a1
            + w2 * a2 * a2 * a2
            - three_w * a0 * a1 * a2
    }

    // ────────────────────────────────────────────────────────────────
    //  Inverse via adjugate / norm
    //
    //   b0 = (a0² − W·a1·a2) / N
    //   b1 = (W·a2² − a0·a1) / N
    //   b2 = (a1² − a0·a2)   / N
    // ────────────────────────────────────────────────────────────────

    // ── FIX 6: .inverse() comes from the Field trait (now in scope). ──
    pub fn inverse(&self) -> Option<Self> {
        let n = self.norm();
        if n == Fp::ZERO {
            return None;
        }
        let n_inv = n.inverse()?; // single Fp inversion

        let (a0, a1, a2) = (self.c[0], self.c[1], self.c[2]);

        let d0 = (a0 * a0 - W * a1 * a2) * n_inv;
        let d1 = (W * a2 * a2 - a0 * a1) * n_inv;
        let d2 = (a1 * a1 - a0 * a2) * n_inv;

        Some(Self { c: [d0, d1, d2] })
    }

    /// Square-and-multiply exponentiation by a u64.
    pub fn pow_u64(&self, mut exp: u64) -> Self {
        if exp == 0 {
            return Self::ONE;
        }
        let mut base = *self;
        let mut result = Self::ONE;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base;
            }
            base = base.square();
            exp >>= 1;
        }
        result
    }

    // ────────────────────────────────────────────────────────────────
    //  Frobenius:  φ(c0 + c1·α + c2·α²) = c0 + c1·ωα + c2·ω²α²
    //              where ω = W^{(p-1)/3}
    // ────────────────────────────────────────────────────────────────

    pub fn frobenius(&self) -> Self {
        let omega = cube_root_of_unity();
        let omega2 = omega * omega;
        Self {
            c: [self.c[0], self.c[1] * omega, self.c[2] * omega2],
        }
    }

    // ────────────────────────────────────────────────────────────────
    //  Serialization (3 × 8 bytes, little-endian)
    // ────────────────────────────────────────────────────────────────

    // ── FIX 7: Fp has no .to_le_bytes().                              ──
    //    Use .into_bigint() → BigInt<2>, then read the first limb.     ──
    //    For Goldilocks all values < p < 2^64, so limb 0 suffices.     ──
    pub fn to_bytes_le(&self) -> [u8; 24] {
        let mut out = [0u8; 24];
        for (i, coeff) in self.c.iter().enumerate() {
            let bigint = coeff.into_bigint(); // Montgomery → standard
            let val: u64 = bigint.0[0]; // first limb (high limb is 0)
            out[i * 8..(i + 1) * 8].copy_from_slice(&val.to_le_bytes());
        }
        out
    }

    pub fn from_bytes_le(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 24 {
            return None;
        }
        Some(Self {
            c: [
                fp_from_8le(&bytes[0..8]),
                fp_from_8le(&bytes[8..16]),
                fp_from_8le(&bytes[16..24]),
            ],
        })
    }

    pub fn from_bytes_le_array(bytes: &[u8; 24]) -> Self {
        Self {
            c: [
                fp_from_8le(&bytes[0..8]),
                fp_from_8le(&bytes[8..16]),
                fp_from_8le(&bytes[16..24]),
            ],
        }
    }

    pub fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 24 {
            return None;
        }
        Some(Self {
            c: [
                fp_from_8le(&bytes[0..8]),
                fp_from_8le(&bytes[8..16]),
                fp_from_8le(&bytes[16..24]),
            ],
        })
    }

    /// Evaluate a polynomial with base-field (Fp) coefficients at an
    /// extension-field point `z`, using Horner's method.
    ///
    ///   coeffs = [c0, c1, c2, …, cn]  represents  c0 + c1·X + c2·X² + … + cn·X^n
    ///
    /// Returns the result in F_{p^3}.
    pub fn eval_base_poly(coeffs: &[Fp], z: CubicExt) -> CubicExt {
        if coeffs.is_empty() {
            return CubicExt::ZERO;
        }
        // Horner: start from the highest-degree coefficient
        let mut acc = CubicExt::from_base(coeffs[coeffs.len() - 1]);
        for i in (0..coeffs.len() - 1).rev() {
            acc = acc * z + CubicExt::from_base(coeffs[i]);
        }
        acc
    }

    /// In-place batch inversion using Montgomery's trick.
    ///
    /// Replaces each element of `vals` with its multiplicative inverse.
    /// Panics if any element is zero.
    pub fn batch_inverse(vals: &mut [CubicExt]) {
        let n = vals.len();
        if n == 0 {
            return;
        }

        // 1. Build prefix products:  prefix[i] = vals[0] * vals[1] * … * vals[i]
        let mut prefix = Vec::with_capacity(n);
        prefix.push(vals[0]);
        for i in 1..n {
            prefix.push(prefix[i - 1] * vals[i]);
        }

        // 2. Single inversion of the total product
        let mut inv = prefix[n - 1]
            .inverse()
            .expect("batch_inverse: zero element in input");

        // 3. Walk backwards, peeling off one factor at a time
        for i in (1..n).rev() {
            let original = vals[i];          // save aᵢ before overwriting
            vals[i] = prefix[i - 1] * inv;   // prefix[i-1] · (a₀…aᵢ)⁻¹ = aᵢ⁻¹  ✓
            inv = inv * original;             // now inv = (a₀…a_{i-1})⁻¹
        }
        vals[0] = inv; // inv is now a₀⁻¹
    }

}

// ────────────────────────────────────────────────────────────────────────
//  Operator impls
// ────────────────────────────────────────────────────────────────────────

impl Default for CubicExt {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Add for CubicExt {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            c: [
                self.c[0] + rhs.c[0],
                self.c[1] + rhs.c[1],
                self.c[2] + rhs.c[2],
            ],
        }
    }
}

impl AddAssign for CubicExt {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.c[0] += rhs.c[0];
        self.c[1] += rhs.c[1];
        self.c[2] += rhs.c[2];
    }
}

impl Sub for CubicExt {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            c: [
                self.c[0] - rhs.c[0],
                self.c[1] - rhs.c[1],
                self.c[2] - rhs.c[2],
            ],
        }
    }
}

impl SubAssign for CubicExt {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.c[0] -= rhs.c[0];
        self.c[1] -= rhs.c[1];
        self.c[2] -= rhs.c[2];
    }
}

impl Neg for CubicExt {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            c: [-self.c[0], -self.c[1], -self.c[2]],
        }
    }
}

impl Mul for CubicExt {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_impl(&rhs)
    }
}

impl MulAssign for CubicExt {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_impl(&rhs);
    }
}

impl Div for CubicExt {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse().expect("division by zero in CubicExt")
    }
}

impl DivAssign for CubicExt {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Scalar multiplication:  Fp × CubicExt
impl Mul<CubicExt> for Fp {
    type Output = CubicExt;
    #[inline]
    fn mul(self, rhs: CubicExt) -> CubicExt {
        CubicExt {
            c: [self * rhs.c[0], self * rhs.c[1], self * rhs.c[2]],
        }
    }
}

/// Scalar multiplication:  CubicExt × Fp
impl Mul<Fp> for CubicExt {
    type Output = CubicExt;
    #[inline]
    fn mul(self, rhs: Fp) -> CubicExt {
        CubicExt {
            c: [self.c[0] * rhs, self.c[1] * rhs, self.c[2] * rhs],
        }
    }
}

impl fmt::Display for CubicExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} + {}·α + {}·α²)", self.c[0], self.c[1], self.c[2])
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Sampling a random extension-field element (for challenges, etc.)
// ────────────────────────────────────────────────────────────────────────

impl CubicExt {
    /// Deterministic derivation from 24 bytes of hash output.
    /// Interprets each 8-byte chunk as a base-field element mod p.
    pub fn from_hash_bytes(bytes: &[u8; 24]) -> Self {
        Self::from_bytes_le_array(bytes)
    }
}

/// Reconstruct an Fp from 8 little-endian bytes.
/// (Fp uses BigInt<2>, so from_le_bytes_mod_order needs ≥16 bytes;
///  going through u64 avoids that.)
#[inline]
fn fp_from_8le(bytes: &[u8]) -> Fp {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    Fp::from(u64::from_le_bytes(buf))
}

// ────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_w_is_cubic_non_residue() {
        verify_cubic_non_residue();
    }

    #[test]
    fn test_cube_root_of_unity() {
        let omega = cube_root_of_unity();
        assert_ne!(omega, Fp::ONE);
        assert_eq!(omega * omega * omega, Fp::ONE);
        // ω² + ω + 1 = 0
        assert_eq!(omega * omega + omega + Fp::ONE, Fp::ZERO);
    }

    #[test]
    fn test_mul_identity() {
        let a = CubicExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        assert_eq!(a * CubicExt::ONE, a);
        assert_eq!(CubicExt::ONE * a, a);
    }

    #[test]
    fn test_mul_zero() {
        let a = CubicExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        assert_eq!(a * CubicExt::ZERO, CubicExt::ZERO);
    }

    #[test]
    fn test_add_sub_roundtrip() {
        let a = CubicExt::new(MontFp!("10"), MontFp!("20"), MontFp!("30"));
        let b = CubicExt::new(MontFp!("1"), MontFp!("2"), MontFp!("3"));
        assert_eq!((a + b) - b, a);
        assert_eq!((a - b) + b, a);
    }

    #[test]
    fn test_negation() {
        let a = CubicExt::new(MontFp!("10"), MontFp!("20"), MontFp!("30"));
        assert_eq!(a + (-a), CubicExt::ZERO);
    }

    #[test]
    fn test_inverse() {
        let a = CubicExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let a_inv = a.inverse().unwrap();
        let product = a * a_inv;
        assert_eq!(product, CubicExt::ONE);
    }

    #[test]
    fn test_inverse_of_base() {
        let a = CubicExt::from_base(MontFp!("42"));
        let a_inv = a.inverse().unwrap();
        assert_eq!(a * a_inv, CubicExt::ONE);
        assert!(a_inv.is_base_field());
    }

    #[test]
    fn test_zero_has_no_inverse() {
        assert!(CubicExt::ZERO.inverse().is_none());
    }

    #[test]
    fn test_norm_of_base_is_cube() {
        // N(a) = a³  when a ∈ F_p
        let a = MontFp!("5");
        let ext = CubicExt::from_base(a);
        assert_eq!(ext.norm(), a * a * a);
    }

    #[test]
    fn test_frobenius_cubed_is_identity() {
        let a = CubicExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let a_frob3 = a.frobenius().frobenius().frobenius();
        assert_eq!(a_frob3, a, "φ³ should be the identity on F_{{p^3}}");
    }

    #[test]
    fn test_frobenius_is_ring_homomorphism() {
        let a = CubicExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let b = CubicExt::new(MontFp!("7"), MontFp!("11"), MontFp!("13"));
        // φ(a·b) = φ(a)·φ(b)
        assert_eq!((a * b).frobenius(), a.frobenius() * b.frobenius());
        // φ(a+b) = φ(a)+φ(b)
        assert_eq!((a + b).frobenius(), a.frobenius() + b.frobenius());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let a = CubicExt::new(MontFp!("42"), MontFp!("99"), MontFp!("7"));
        let bytes = a.to_bytes_le();
        let b = CubicExt::from_bytes_le_array(&bytes);
        assert_eq!(a, b);
    }

    #[test]
    fn test_alpha_cubed_is_w() {
        let alpha3 = CubicExt::ALPHA * CubicExt::ALPHA * CubicExt::ALPHA;
        assert_eq!(alpha3, CubicExt::from_base(W));
    }

    #[test]
    fn test_mul_associative() {
        let a = CubicExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let b = CubicExt::new(MontFp!("7"), MontFp!("11"), MontFp!("13"));
        let c = CubicExt::new(MontFp!("17"), MontFp!("19"), MontFp!("23"));
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn test_mul_commutative() {
        let a = CubicExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let b = CubicExt::new(MontFp!("7"), MontFp!("11"), MontFp!("13"));
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn test_distributive() {
        let a = CubicExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let b = CubicExt::new(MontFp!("7"), MontFp!("11"), MontFp!("13"));
        let c = CubicExt::new(MontFp!("17"), MontFp!("19"), MontFp!("23"));
        assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn test_extension_field_size() {
        // |F_{p^3}| should have ~192-bit elements (3 × 64-bit limbs)
        // This is a structural test: serialization is 24 bytes = 192 bits
        let a = CubicExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        assert_eq!(a.to_bytes_le().len(), 24);
    }

    #[cfg(test)]
mod cubic_ext_sanity {
    use crate::cubic_ext::CubicExt;
    use crate::F;

    #[test]
    fn test_no_zero_divisors() {
        let a = CubicExt::new(F::from(1u64), F::from(0u64), F::from(0u64));
        let b = CubicExt::new(F::from(0u64), F::from(1u64), F::from(0u64));
        let product = a * b;

        assert!(
            product != CubicExt::new(F::from(0u64), F::from(0u64), F::from(0u64)),
            "CubicExt has zero divisors — this is componentwise, not a field!"
        );
        assert_eq!(product, b, "1 * α should equal α");
    }

    #[test]
    fn test_alpha_squared() {
        let alpha = CubicExt::new(F::from(0u64), F::from(1u64), F::from(0u64));
        let alpha_sq = alpha * alpha;

        let expected = CubicExt::new(F::from(0u64), F::from(0u64), F::from(1u64));
        assert_eq!(alpha_sq, expected, "α² should be (0, 0, 1)");
    }

    #[test]
    fn test_alpha_cubed_equals_w() {
        let alpha = CubicExt::new(F::from(0u64), F::from(1u64), F::from(0u64));
        let alpha_cubed = alpha * alpha * alpha;

        let expected = CubicExt::from_base(F::from(7u64));
        assert_eq!(
            alpha_cubed, expected,
            "α³ should equal W (the cubic non-residue). Got {:?}",
            alpha_cubed
        );
    }

    #[test]
    fn test_inverse_exists() {
        let a = CubicExt::new(
            F::from(3u64),
            F::from(5u64),
            F::from(11u64),
        );
        let a_inv = a.inverse().expect("nonzero element should have an inverse");
        let product = a * a_inv;

        let one = CubicExt::from_base(F::from(1u64));
        assert_eq!(product, one, "a * a⁻¹ should equal 1");
    }

    #[test]
    fn test_cross_terms() {
        let one_plus_alpha = CubicExt::new(
            F::from(1u64),
            F::from(1u64),
            F::from(0u64),
        );
        let sq = one_plus_alpha * one_plus_alpha;

        let expected = CubicExt::new(
            F::from(1u64),
            F::from(2u64),
            F::from(1u64),
        );
        assert_eq!(
            sq, expected,
            "(1 + α)² should be 1 + 2α + α², got {:?}",
            sq
        );
    }
}
}