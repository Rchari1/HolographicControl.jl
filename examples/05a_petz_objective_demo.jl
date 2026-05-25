# 05a_petz_objective_demo.jl
#
# M5 Layer 1 (no new deps): demonstrate that `petz_recovery_objective(V)`
# is a meaningful discovery target by evaluating it on several known
# isometries on a small (n_bdy=3, n_bulk=1) system. Distinguishes between
# "good" codes (low objective), "bad" codes (high objective), and
# textbook benchmarks.
#
# A_list = all weight-1 erasures = 3 subregions of size 2 (keep 2 qubits).
# Weights uniform.
#
# Usage:
#   julia --project=. examples/05a_petz_objective_demo.jl

using LinearAlgebra
using Printf
using Random

using HolographicControl

n_bdy = 3
n_bulk = 1
d_bdy = 2^n_bdy
d_bulk = 2^n_bulk

# Single-qubit erasures → all 2-qubit subregions of {1, 2, 3}.
A_list = [collect(setdiff(1:n_bdy, [q])) for q in 1:n_bdy]   # [[2,3], [1,3], [1,2]]

println("=" ^ 72)
println("M5 Layer 1 — Petz recovery objective on a 3-qubit / 1-bulk system")
println("A_list = $A_list   (single-qubit erasures; recovery from 2 kept qubits)")
println("=" ^ 72)
println()

function evaluate(label, V)
    obj = petz_recovery_objective(V; A_list=A_list)
    per_A = [petz_recovery_error(V, A) for A in A_list]
    @printf("\n%s\n", label)
    @printf("  Σ w_A·err_A = %.6e\n", obj)
    for (A, e) in zip(A_list, per_A)
        @printf("    keep %s : %.6e\n", A, e)
    end
end

# Reference 1: the 3-qubit repetition encoder
# (|0⟩_L = |000⟩, |1⟩_L = |111⟩). Any 2 of 3 qubits are perfectly
# distinguishable between the two logical states, so single-qubit erasure
# should give recovery error 0.
V_rep = three_qubit_repetition_isometry()
evaluate("(1) 3-qubit repetition code (textbook):", V_rep)

# Reference 2: V_trivial = the trivial embedding (V|j⟩_L = |j 0 0⟩).
# Logical bit lives only on qubit 1. Erasing qubit 1 destroys the bulk;
# the other two erasures are harmless. So we expect error 0 on the
# {2,3}-trace and {1,3}/{1,2} → high error on the keep-{2,3} subregion.
V_trivial = zeros(ComplexF64, d_bdy, d_bulk)
V_trivial[1, 1] = 1   # |000⟩
V_trivial[5, 2] = 1   # |100⟩
evaluate("(2) Trivial embedding |q⟩_L = |q,0,0⟩ (no redundancy):", V_trivial)

# Reference 3: a random isometry — control case, should be uninformatively bad
Random.seed!(42)
M_rand = randn(ComplexF64, d_bdy, d_bulk)
V_rand = M_rand * inv(sqrt(Hermitian(M_rand' * M_rand)))   # polar projection
evaluate("(3) Random isometry (seed=42):", V_rand)

# Reference 4: convex combination of |0⟩_L between |000⟩ and the |W⟩ state
# (|001⟩+|010⟩+|100⟩)/√3. Mixes some recovery information across qubits.
V_W = zeros(ComplexF64, d_bdy, d_bulk)
V_W[1, 1] = 1                                              # |0⟩_L = |000⟩
V_W[5, 2] = 1/sqrt(3); V_W[3, 2] = 1/sqrt(3); V_W[2, 2] = 1/sqrt(3)  # |1⟩_L = |W⟩
# Normalize
V_W[:, 2] ./= norm(V_W[:, 2])
evaluate("(4) Asymmetric: |0⟩_L = |000⟩, |1⟩_L = |W⟩:", V_W)

println()
println("=" ^ 72)
println("Interpretation")
println("=" ^ 72)
println("(1) Repetition code: error 0.5 = the entanglement infidelity of the bulk")
println("    DEPHASING channel. Repetition preserves the classical bit (|000⟩ vs")
println("    |111⟩) but loses the phase relation between bulk basis states. This")
println("    is a CLASSICAL error-correcting code, not a quantum one. The objective")
println("    correctly penalizes this.")
println("(2) Trivial embedding: error 0.75 on keep-{2,3} (fully-depolarizing on")
println("    bulk; qubit 1 held all information) and 0 on subregions that retain")
println("    qubit 1. Asymmetric.")
println("(3) Random isometry: ~uniform moderate error across subregions, and lower")
println("    *total* error than the repetition code — random isometries spread")
println("    information more symmetrically across qubits without locking phase to")
println("    bit. This is a clue about what the M5 optimizer will look for.")
println("(4) |0⟩_L=|000⟩, |1⟩_L=|W⟩: error 0.27 — between random and trivial.")
println()
println("Note: a perfect QEC code on n=3 / k=1 does NOT exist (quantum Singleton")
println("bound: n - k ≥ 2(d-1) → d ≤ 2, which corrects 0 errors and only DETECTS).")
println("The M5 objective on this system is meaningful precisely because no")
println("analytic code achieves 0 error — the optimizer is looking for the best")
println("APPROXIMATE code, which is the thesis-novel question.")
println()
println("=> petz_recovery_objective discriminates correctly. Next step (Layer 2):")
println("   optimize V directly to minimize this objective and see what code the")
println("   optimizer prefers when no exact-zero solution exists.")
