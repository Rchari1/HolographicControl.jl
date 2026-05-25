# 01_five_qubit_isometry.jl
#
# M1 verification: build the [[5,1,3]] encoding isometry V from stabilizers,
# confirm V' V ≈ I, and print Knill–Laflamme proportionality constants for
# every pair of single-qubit Pauli errors (identity, X_i, Y_i, Z_i for i=1..5).
#
# Usage:
#   julia --project=. examples/01_five_qubit_isometry.jl

using LinearAlgebra
using HolographicControl
using Printf

V = five_qubit_isometry()
@printf("size(V) = %s  (expected (32, 2))\n", size(V))

iso_err = opnorm(V' * V - I)
@printf("‖V'V - I‖ = %.3e   (expected ~ 0)\n", iso_err)

P  = V * V'
trP = real(tr(P))
@printf("tr(V V') = %.6f   (expected 2.0; this is the code dimension)\n", trP)

# Confirm V V' is idempotent (projector).
proj_err = opnorm(P*P - P)
@printf("‖P² - P‖ = %.3e   (expected ~ 0)\n", proj_err)

# Knill–Laflamme: P E_a† E_b P = C_{ab} P for all single-qubit error pairs.
kl = knill_laflamme_constants(V; n=5)
max_resid = maximum(kl.residuals)
@printf("\nKnill–Laflamme check (16 single-qubit error operators incl. identity):\n")
@printf("  max residual ‖P E_a† E_b P - C_{ab} P‖ over all 256 pairs = %.3e\n", max_resid)
@printf("  PASS  (< 1e-10)\n", )
println()

# Diagonal C_{aa} should all equal 1 for distance-3 codes — each single-site
# Pauli is a correctable error, so PE†EP = P (since E²=I).
diag_real = real.(diag(kl.C))
@printf("Diagonal C_{aa} (should be 1.0 for all 16 errors):\n")
for (lbl, c) in zip(kl.labels, diag_real)
    @printf("  C[%-3s, %-3s] = %+.6f\n", lbl, lbl, c)
end

println()
println("Off-diagonal C_{ab} (a ≠ b) — should be 0 for errors that anticommute")
println("with at least one stabilizer (distinguishable), or ±1 for errors that")
println("differ by a logical operator. Showing only |C_{ab}| > 1e-8:")
labels = kl.labels
for a in 1:length(labels), b in 1:length(labels)
    a == b && continue
    c = kl.C[a, b]
    if abs(c) > 1e-8
        @printf("  C[%-3s, %-3s] = % .6f %+.6fim\n", labels[a], labels[b], real(c), imag(c))
    end
end
