# The [[4,1,2]] quantum error-DETECTING code (Vaidman 1996, sometimes called
# the "4-qubit code"). Stabilized at the [[4,2,2]] level by two generators
# {XXXX, ZZZZ}; the [[4,1,2]] sub-encoding selects one logical qubit out of
# the two by freezing the other to |0⟩.
#
# Concretely, the [[4,2,2]] code has stabilizer group ⟨XXXX, ZZZZ⟩ and a
# logical Pauli basis
#
#     X̄_1 = XXII,  Z̄_1 = ZIZI,        # logical qubit 1
#     X̄_2 = IXIX,  Z̄_2 = IIZZ,        # logical qubit 2
#
# (a standard choice; X̄_a anticommutes with Z̄_a and commutes with everything
# else). To get the [[4,1,2]] sub-encoding we project onto the +1 eigenspace
# of the augmented stabilizer group ⟨XXXX, ZZZZ, IIZZ, ZIZI⟩, which fixes
# logical qubit 2 to |0⟩ AND fixes Z̄_1's value — leaving |0⟩_L unique up to
# phase. Then |1⟩_L = X̄_1 |0⟩_L flips Z̄_1, giving the orthogonal logical
# basis state.
#
# The resulting encoding is the well-known
#
#     |0⟩_L = (|0000⟩ + |1111⟩)/√2,
#     |1⟩_L = (|0011⟩ + |1100⟩)/√2.
#
# Distance 2: detects any single Pauli error (Knill–Laflamme C_{ab} matrix
# is diagonal but NOT proportional to identity — the constants pick up
# the structure that distinguishes single-qubit errors), corrects 1 known-
# location erasure (since d - 1 = 1), but fails on 2 or more erasures.

const FOUR_ONE_TWO_STABILIZER_STRINGS = ("XXXX", "ZZZZ")

# Auxiliary generator used to pin one of the two [[4,2,2]] logical qubits.
# Adding IIZZ to the stabilizer group reduces the code from k=2 (the
# [[4,2,2]] detection code) to k=1 (the [[4,1,2]] sub-encoding).
const FOUR_ONE_TWO_FROZEN_LOGICAL_Z = "IIZZ"

# A Z-type logical operator on the *retained* logical qubit, anticommuting
# with FOUR_ONE_TWO_LOGICAL_X and commuting with all three stabilizers
# above. Used to fully specify the |0⟩_L computational state inside the
# 2-dim code subspace (the projector ∏ (I + g_i)/2 alone is rank 2 and
# would only give a code-subspace projector; adding the +1 projector for
# this operator picks out |0⟩_L specifically).
const FOUR_ONE_TWO_LOGICAL_Z = "ZIZI"

# Logical X for the retained qubit. Anticommutes with FOUR_ONE_TWO_LOGICAL_Z
# (so |1⟩_L = X̄ |0⟩_L is orthogonal to |0⟩_L), commutes with all three
# stabilizers (so it preserves the code subspace).
const FOUR_ONE_TWO_LOGICAL_X = "XXII"

"""
    four_one_two_stabilizers()

Return the two `16 × 16` stabilizer generators of the [[4,1,2]] code as a
`Vector{Matrix{ComplexF64}}`:

    g1 = XXXX,  g2 = ZZZZ

under the package's big-endian convention. These are the stabilizers of
the parent [[4,2,2]] detection code; the [[4,1,2]] sub-encoding additionally
projects onto the +1 eigenspace of `IIZZ` and `ZIZI` to single out one of
the two logical qubits (see [`four_one_two_isometry`](@ref)).
"""
four_one_two_stabilizers() = [pauli_string(s) for s in FOUR_ONE_TWO_STABILIZER_STRINGS]

"""
    four_one_two_isometry() -> Matrix{ComplexF64}

Build the `16 × 2` encoding isometry `V` of the [[4,1,2]] error-detecting
code. Columns are the analytic logical basis states

    |0⟩_L = (|0000⟩ + |1111⟩)/√2,
    |1⟩_L = (|0011⟩ + |1100⟩)/√2,

obtained by projecting `|0000⟩` onto the +1 eigenspace of the augmented
generators `XXXX, ZZZZ, IIZZ, ZIZI` (the [[4,2,2]] stabilizers plus a
frozen-logical-qubit constraint plus a definite Z̄ value) and then
applying `X̄ = XXII` to get `|1⟩_L`.

Properties verified in `test/test_more_codes.jl`:
  * `V' V = I_2` (isometry)
  * Both [[4,2,2]] stabilizers fix the code subspace (`g_i V = V`)
  * Knill–Laflamme `C_{ab}` for the 13 single-qubit Pauli errors has
    `C_{aa} = 1` (every Pauli preserves codespace norm) and
    `C_{aI} = 0` for `a ≠ I` (single Pauli errors are orthogonal to
    the identity component on the codespace, hence DETECTABLE), but
    `‖C - I‖` is `O(1)` because some off-diagonal `C_{ab}` are nonzero
    and some residuals `‖M_{ab} - C_{ab} P‖` are `O(1)`. This is the
    textbook signature of a distance-2 detection-only code: errors are
    distinguishable from "no error" (=> detectable) but pairs of
    distinct single-qubit Paulis are not pairwise distinguishable
    (=> not correctable)
  * `petz_recovery_error` is ≈ 0 for every 1-qubit erasure (kept = 3
    qubits) — the distance-2 code corrects exactly `d - 1 = 1` known-
    location erasure
  * `petz_recovery_error = 1/2` for every 2-qubit erasure (kept = 2
    qubits) and `= 3/4` for every 3-qubit erasure (kept = 1 qubit, the
    fully-depolarizing limit `1 - 1/d_bulk² = 1 - 1/4`).
"""
function four_one_two_isometry()
    n = 4
    d = 2^n
    Id = Matrix{ComplexF64}(I, d, d)

    # Project onto +1 eigenspace of all four operators: the two [[4,2,2]]
    # stabilizers, the frozen-logical-Z (IIZZ), and the retained Z̄ = ZIZI.
    # All four mutually commute, so the four projectors commute and their
    # product is the rank-1 projector onto |0⟩_L (up to phase).
    P_stab = Id
    for s in FOUR_ONE_TWO_STABILIZER_STRINGS
        P_stab = P_stab * (Id + pauli_string(s)) / 2
    end
    P_stab = P_stab * (Id + pauli_string(FOUR_ONE_TWO_FROZEN_LOGICAL_Z)) / 2
    P_stab = P_stab * (Id + pauli_string(FOUR_ONE_TWO_LOGICAL_Z)) / 2

    # Big-endian: |0000⟩ → index 1. The seed lies in the +1 eigenspace of
    # every diagonal operator above, so only the (I+XXXX)/2 step does any
    # mixing — it produces (|0000⟩ + |1111⟩)/2.
    seed = zeros(ComplexF64, d)
    seed[1] = 1

    psi0L = P_stab * seed
    n0 = norm(psi0L)
    n0 > 1e-12 || error("P_stab |0000⟩ vanished — check stabilizer convention")
    psi0L ./= n0

    psi1L = pauli_string(FOUR_ONE_TWO_LOGICAL_X) * psi0L
    # X̄ commutes with all stabilizers we projected onto except ZIZI (the
    # retained logical Z, which X̄ ANTIcommutes with). So |1⟩_L is already
    # in the [[4,2,2]] codespace; re-project only onto the original
    # stabilizers and the frozen-logical-Z, NOT onto ZIZI (which would
    # zero out the -1 eigenstate we are trying to build).
    P_code = Id
    for s in FOUR_ONE_TWO_STABILIZER_STRINGS
        P_code = P_code * (Id + pauli_string(s)) / 2
    end
    P_code = P_code * (Id + pauli_string(FOUR_ONE_TWO_FROZEN_LOGICAL_Z)) / 2
    psi1L = P_code * psi1L
    norm(psi1L) > 1e-12 || error("X̄|0⟩_L vanished — check logical operator")
    psi1L ./= norm(psi1L)

    return hcat(psi0L, psi1L)
end
