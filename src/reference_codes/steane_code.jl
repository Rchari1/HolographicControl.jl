# The [[7,1,3]] Steane CSS code (Steane 1996). A 7-qubit code with 6
# stabilizer generators built from the classical [7,4,3] Hamming code:
# the parity-check matrix `H` of the Hamming code gives three X-type
# generators (one per row of `H`) and three Z-type generators (the same
# row pattern but with Z instead of X). The CSS construction guarantees
# all generators mutually commute because the Hamming code is self-dual
# (`H · H^T = 0 mod 2`).
#
# Distance 3 (corrects 1 general single-qubit error, or up to 2 known-
# location erasures), and unlike the [[5,1,3]] perfect code is NOT
# absolutely maximally entangled — some 3-erasure (= 4-qubit kept)
# patterns are correctable while others are not, depending on whether
# the kept region contains a logical operator support.

const STEANE_STABILIZER_STRINGS = (
    "IIIXXXX",  # g_X1: X on qubits 4,5,6,7
    "IXXIIXX",  # g_X2: X on qubits 2,3,6,7
    "XIXIXIX",  # g_X3: X on qubits 1,3,5,7
    "IIIZZZZ",  # g_Z1: Z on qubits 4,5,6,7
    "IZZIIZZ",  # g_Z2: Z on qubits 2,3,6,7
    "ZIZIZIZ",  # g_Z3: Z on qubits 1,3,5,7
)

const STEANE_LOGICAL_X = "XXXXXXX"
const STEANE_LOGICAL_Z = "ZZZZZZZ"

"""
    steane_stabilizers()

Return the six `128 × 128` stabilizer generators of the [[7,1,3]] Steane
code as a `Vector{Matrix{ComplexF64}}`:

    g_X1 = IIIXXXX,  g_X2 = IXXIIXX,  g_X3 = XIXIXIX,
    g_Z1 = IIIZZZZ,  g_Z2 = IZZIIZZ,  g_Z3 = ZIZIZIZ.

The X-type and Z-type generators share the same support pattern — the
rows of the classical [7,4,3] Hamming parity-check matrix — which is the
defining property of the Steane CSS construction. All Pauli strings
follow the package's big-endian convention.
"""
steane_stabilizers() = [pauli_string(s) for s in STEANE_STABILIZER_STRINGS]

"""
    steane_isometry() -> Matrix{ComplexF64}

Build the `128 × 2` encoding isometry `V` of the [[7,1,3]] Steane code
by projecting `|0000000⟩` onto the +1 eigenspace of the six stabilizer
generators via `P_stab = ∏_i (I + g_i)/2`, then defining
`|1⟩_L = X̄ |0⟩_L` with `X̄ = XXXXXXX`. Columns of `V` are `|0⟩_L`
and `|1⟩_L`.

Properties verified in `test/test_more_codes.jl`:
  * `V' V = I_2` (isometry)
  * All six stabilizers fix the code subspace (`g_i V = V`) to machine
    precision
  * Z̄ = ZZZZZZZ acts as logical Z on the basis: `V' Z̄ V = diag(+1, -1)`
  * Knill–Laflamme `C_{ab}` for the 22 single-qubit Pauli errors equals
    `I_22` to machine precision — the [[7,1,3]] is a non-degenerate
    distance-3 code, same Knill–Laflamme structure as the [[5,1,3]]
  * `petz_recovery_error ≈ 0` for ALL 1- and 2-qubit erasures
    (`d - 1 = 2` known-location erasures correctable for any pattern)
  * `petz_recovery_error` is bimodal `{0, 3/4}` over 3-qubit erasures
    (35 in total): some 4-qubit kept regions contain a full logical
    operator and so reconstruct the bulk perfectly, others don't and
    fully depolarize. This is the non-AME signature that distinguishes
    Steane from the [[5,1,3]] perfect code (where every kept region
    above the threshold recovers).
"""
function steane_isometry()
    n = 7
    d = 2^n
    Id = Matrix{ComplexF64}(I, d, d)

    # Stabilizer-group projector. P_stab is Hermitian, idempotent, with rank
    # 2^(n - n_stab) = 2 for this code (6 independent stabilizers).
    P_stab = Id
    for s in STEANE_STABILIZER_STRINGS
        P_stab = P_stab * (Id + pauli_string(s)) / 2
    end

    # Big-endian: |0000000⟩ → index 1. The seed lies in the +1 eigenspace
    # of the three Z-type generators (they have +1 eigenvalue on |0...0⟩);
    # the three X-type projectors mix in the 8 codewords formed by the
    # cosets of the Hamming dual code.
    seed = zeros(ComplexF64, d)
    seed[1] = 1

    psi0L = P_stab * seed
    n0 = norm(psi0L)
    n0 > 1e-12 || error("P_stab |0000000⟩ vanished — check stabilizer convention")
    psi0L ./= n0

    psi1L = pauli_string(STEANE_LOGICAL_X) * psi0L
    # X̄ = XXXXXXX commutes with all stabilizers (every g_i has an even
    # number of operator positions in common with X̄ for both X-type and
    # Z-type generators — for Z-type: 4 Z's vs 7 X's gives 4 anticomms,
    # even, so commute; for X-type: trivially commute). Re-project for
    # numerical hygiene.
    psi1L = P_stab * psi1L
    norm(psi1L) > 1e-12 || error("X̄|0⟩_L vanished — check logical operator")
    psi1L ./= norm(psi1L)

    return hcat(psi0L, psi1L)
end
