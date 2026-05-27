# 06_page_curve_demo.jl
#
# Black holes as evaporating codes — the Page curve.
#
# In the QEC interpretation of black-hole evaporation, the interior is a bulk
# logical subsystem encoded into n boundary physical qubits. As the hole
# evaporates, qubits leave it and join the radiation R. The radiation entropy
# S(R) traces the PAGE CURVE: it rises while R is small, peaks at the Page
# time, then falls back to zero as the interior becomes reconstructable from R.
#
# This demo computes the Page curve (analytically, no Piccolo optimization, so
# it runs in well under a minute) for three codes and contrasts them:
#   (1) the [[5,1,3]] perfect code  — an AME(5,2) state, the canonical case
#   (2) the 3-qubit repetition code — a CLASSICAL code, for contrast
#   (3) a random 32×2 isometry      — a generic encoder, for contrast
#
# Usage:
#   julia --project=. examples/06_page_curve_demo.jl

using LinearAlgebra
using Printf
using Random

using HolographicControl

function show_page_curve(label, V)
    pc = page_curve(V)
    pt = page_time(V)
    @printf("\n%s\n", label)
    println("  |R|   S(R) [bits]   reconstructable?")
    for (k, s, r) in zip(pc.k, pc.S_R, pc.reconstructable)
        marker = (k == pt) ? "   <-- page time (bulk first recoverable)" : ""
        @printf("  %2d    %8.4f       %-5s%s\n", k, s, r ? "yes" : "no", marker)
    end
    @printf("  page_time = %d,  peak S(R) = %.4f bits\n", pt, maximum(pc.S_R))
    return pc, pt
end

println("=" ^ 72)
println("Black holes as evaporating codes — the Page curve")
println("S(R) = entanglement entropy of the radiation R (a pure code state),")
println("averaged over all size-|R| radiation regions. base = 2 (bits).")
println("=" ^ 72)

# (1) [[5,1,3]] perfect code — AME(5,2).
pc5, pt5 = show_page_curve("(1) [[5,1,3]] perfect code (AME(5,2)):", five_qubit_isometry())

# (2) 3-qubit repetition code — classical.
pcr, ptr = show_page_curve("(2) 3-qubit repetition code (classical):", three_qubit_repetition_isometry())

# (3) Random isometry — generic encoder.
Random.seed!(2026)
Vr = random_isometry(32, 2)
pcrand, ptrand = show_page_curve("(3) Random 32x2 isometry (seed=2026):", Vr)

println()
println("=" ^ 72)
println("Interpretation (what we actually observe)")
println("=" ^ 72)
println("""
(1) [[5,1,3]]: S(R) = 0, 1, 2, 2, 1, 0 — the CANONICAL Page curve. It rises
    linearly at 1 bit per escaped qubit (S(R) = |R| for an AME state) up to the
    half system, then falls symmetrically. The peak (2 bits) is the maximum
    entropy of a 2-qubit subsystem. page_time = 3: ANY 3 of the 5 boundary
    qubits reconstruct the interior, but no 2 do. This is the textbook
    symmetric tent — S(R=k) == S(R=5-k) because the code state is pure.

(2) repetition: S(R) = 0, 1, 1, 0 — symmetric, but with a FLAT TOP capped at
    1 bit instead of rising to the 2-bit AME peak. The representative codeword
    is the GHZ state (|000>+|111>)/sqrt(2); every reduced state has rank 2, so
    S is pinned at 1 bit. This is the classical/quantum contrast: the classical
    code stores only a bit, so its radiation never carries more than 1 bit of
    entropy. Its page_time is 3 — the SAME as the perfect code — but for a
    different reason: any 2 qubits pin the classical bit yet leave a DEPHASING
    channel on the quantum bulk (Petz error 0.5), so the full quantum interior
    is only recoverable from all 3 qubits. The honest contrast is the peak
    height (1 bit vs 2 bits) and the manner of recovery, NOT an earlier
    page_time.

(3) random isometry: also a symmetric tent (purity is exact, not statistical),
    with a peak near 1.6 bits — below the AME maximum because a single Haar-
    random codeword is highly but not perfectly entangled. It carries no
    protected stabilizer structure, so no proper sub-region recovers the bulk;
    its page_time here is 5 (only the full system purifies the bulk qubit).

Bottom line: the symmetric tent is the universal signature of evaporating a
PURE code state. The perfect code achieves the maximal (AME) 2-bit tent and a
clean distance-3 page_time with genuine 3-qubit subsystem recovery; the
classical code is capped at 1 bit and only recovers the quantum bulk from the
whole system; the random encoder scrambles strongly (~1.6-bit peak) but lacks
structured protection (page_time = n).
""")
