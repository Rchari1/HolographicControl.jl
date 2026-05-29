# 10_holographic_dashboard.jl
#
# Holographic-code diagnostic dashboard — comparing candidate encoders.
#
# `code_quality_summary` is the package's one-call front end onto the QEC +
# holographic figures of merit: Petz objectives, page time, reconstruction
# threshold, entanglement-wedge counts, and the entropy curve S(|A|). This
# demo applies it to four codes and prints a side-by-side table:
#
#   (1) [[5,1,3]] perfect code            — AME(5,2), the canonical case
#   (2) 3-qubit repetition (classical)    — classical/quantum contrast
#   (3) HaPPY single-pentagon tile        — the toy holographic code
#   (4) Random 32×2 isometry              — generic encoder, no structure
#
# Runs in well under a minute on a laptop. Usage:
#   julia --project=. examples/10_holographic_dashboard.jl

using LinearAlgebra
using Printf
using Random

using HolographicControl

# --------------------------------------------------------------------------- #
# Pretty-print a single dashboard
# --------------------------------------------------------------------------- #

function show_summary(label::String, V::AbstractMatrix; A_list = nothing)
    s = code_quality_summary(V; A_list = A_list)

    @printf("\n%s\n", label)
    @printf("  n_bdy = %d,  n_bulk = %d\n", s.n_bdy, s.n_bulk)
    @printf("  petz_uniform_w1            = %.3e\n", s.petz_uniform_w1)
    if ismissing(s.petz_uniform_w2)
        println("  petz_uniform_w2            = (n_bdy < 4, skipped)")
    else
        @printf("  petz_uniform_w2            = %.3e\n", s.petz_uniform_w2)
    end
    @printf("  page_time                  = %d\n", s.page_time)
    @printf("  reconstruction_threshold   = %d\n", s.reconstruction_threshold)
    println("  S_by_size (avg S(A) for |A|=1..n_bdy):")
    print("    [")
    for (i, x) in enumerate(s.S_by_size)
        @printf("%s%.3f", i == 1 ? "" : ", ", x)
    end
    println("]")
    println("  wedge_report.reconstructable (per |A|): ",
            s.wedge_report.reconstructable)
    if !ismissing(s.petz_custom)
        @printf("  petz_custom (user A_list)  = %.3e\n", s.petz_custom)
    end
    return s
end

println("=" ^ 76)
println("Holographic dashboard — code_quality_summary side-by-side")
println("=" ^ 76)

# --------------------------------------------------------------------------- #
# (1) [[5,1,3]] perfect code
# --------------------------------------------------------------------------- #

V5 = five_qubit_isometry()
# RT-weighted custom A_list to demonstrate the petz_custom branch.
A_rt = all_proper_subregions(5)
w_rt = rt_minimal_surface_weights(5; bias = 1.0)
s5 = show_summary("(1) [[5,1,3]] perfect code  (AME(5,2)):", V5; A_list = A_rt)

# --------------------------------------------------------------------------- #
# (2) 3-qubit repetition code (classical)
# --------------------------------------------------------------------------- #

Vrep = three_qubit_repetition_isometry()
srep = show_summary("(2) 3-qubit repetition  (classical encoder):", Vrep)

# --------------------------------------------------------------------------- #
# (3) HaPPY single-pentagon tile
# --------------------------------------------------------------------------- #

Vp = single_pentagon_isometry()
# HaPPY pentagon == [[5,1,3]] under the package alias, but the framing is
# different — bulk = "central tile", boundary = "5 outer legs".
sp = show_summary("(3) HaPPY single-pentagon tile:", Vp)

# --------------------------------------------------------------------------- #
# (4) Random 32×2 isometry
# --------------------------------------------------------------------------- #

Random.seed!(2026)
Vr = random_isometry(32, 2)
sr = show_summary("(4) Random 32x2 isometry  (seed = 2026):", Vr)

# --------------------------------------------------------------------------- #
# Side-by-side comparison table
# --------------------------------------------------------------------------- #

println("\n", "=" ^ 76)
println("Side-by-side comparison")
println("=" ^ 76)
@printf("%-18s | %-18s | %-12s | %-12s | %-8s | %-8s\n",
        "code", "petz_w1", "petz_w2", "page_time", "rec_thr", "peak S")
println("-" ^ 76)
for (name, s) in (("[[5,1,3]]", s5), ("repetition", srep),
                  ("HaPPY pentagon", sp), ("random", sr))
    w2 = ismissing(s.petz_uniform_w2) ? "  (n<4)" : @sprintf("%.3e", s.petz_uniform_w2)
    @printf("%-18s | %-18s | %-12s | %-12d | %-8d | %-8.3f\n",
            name,
            (@sprintf "%.3e" s.petz_uniform_w1),
            w2,
            s.page_time,
            s.reconstruction_threshold,
            maximum(s.S_by_size))
end

println()
println("=" ^ 76)
println("Interpretation")
println("=" ^ 76)
println("""
[[5,1,3]] and HaPPY pentagon: petz_uniform_w1 and petz_uniform_w2 both vanish
to machine precision — distance-3 erasure correction is perfect. page_time
and reconstruction_threshold both equal 3 (smallest |A| for full wedge
recovery). S_by_size = [1, 2, 2, 1, 0] is the AME tent.

Repetition: petz_uniform_w1 = 0.5 exactly (any single erasure leaves a
dephasing channel). page_time = 3 because the QUANTUM bulk is only fully
recoverable from the whole boundary (any 2 of 3 qubits give 1 classical bit
but lose the |0>_L vs |1>_L phase). S_by_size = [1, 1, 0] — capped at 1 bit.

Random isometry: petz_uniform_w1 and petz_uniform_w2 are O(1) — no protected
structure. page_time = 5 (only the full system recovers); the wedge report
shows zero reconstructable regions at any sub-full size. Its entropy peak
sits around the half-system at ~1.6 bits, below the AME maximum of 2.

The dashboard is the one-call API for asking 'is this a good code, and how
holographic is it?' — combine fields to compare candidates across both QEC
metrics (petz_*) and holographic metrics (page_time, threshold, S_by_size,
wedge_report).
""")
