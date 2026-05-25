# 04g_synthesize_five_qubit_xy.jl
#
# M4 WORKING VERSION — synthesize the [[5,1,3]] perfect-code encoder via
# Piccolo, escaping the 0.91 stagnation basin documented in
# examples/04_synthesize_five_qubit.jl by switching the drift Hamiltonian
# from nearest-neighbor Heisenberg (XX+YY+ZZ) to XY (XX+YY only).
#
# Rationale: the structural sweep (examples/04f_structural_sweep.jl) found
# that dropping the ZZ term from the drift moves Phase 1 fidelity from
# 0.94 to 0.98 (seed=2, all else equal). The follow-up XY-drift seed sweep
# (examples/04h_xy_seed_sweep.jl) then found that seed=3 reaches Phase 1
# fidelity 0.9954 — over the 0.99 bar from L-BFGS alone, where Phase 2 will
# polish toward machine precision. The XY drift's basin structure (range
# 0.013 across 5 seeds, vs 0.036 under Heisenberg) is richer than
# Heisenberg's, so seed selection actually buys us something here.
#
# Two departures from HANDOFF §M4, under explicit user authorization:
#   1. Drift Hamiltonian: Heisenberg → XX+YY (drop the ZZ term)
#   2. Random seed: tried multiple, settled on seed=3
#
# System:
#   - 5 qubits (n_bdy=5, n_bulk=1)
#   - Drift: nearest-neighbor XX+YY, J = 1.0
#   - Drives: single-site X, Y on each qubit (10 drives, |u| ≤ 1.0)
#   - Cold start, seed=3 (winning seed at XY drift)
#
# Usage:
#   julia --project=. examples/04g_synthesize_five_qubit_xy.jl

using LinearAlgebra
using Printf
using Random

using HolographicControl
using Piccolo: solve!

V_target = five_qubit_isometry()
H_drift  = nn_xx_yy_drift(5; J=1.0)
H_drives = single_qubit_xy_drives(5)
drive_bounds = fill(1.0, length(H_drives))

println("=" ^ 72)
println("M4 — synthesize [[5,1,3]] encoder via Piccolo  (XY drift variant)")
println("Drift: nearest-neighbor XX+YY, J=1.0")
println("Drives: single-site X, Y on each of 5 qubits  (10 drives, bound = 1.0)")
println("Seed: 3  (winner of the XY-drift seed sweep at Phase 1)")
println("=" ^ 72)

qcp = isometry_synthesis_problem(
    V_target, H_drift, H_drives, drive_bounds;
    T = 25, duration = 10.0, Q = 100.0, R = 1e-2, ddu_bound = 1.0, seed = 3,
)

t_start = time()

println("\n--- Phase 1: L-BFGS exploration (max_iter=200, eval_hessian=false) ---")
t_p1 = time()
solve!(qcp; max_iter=200, eval_hessian=false)
t_p1 = time() - t_p1
fid_p1 = subspace_fidelity(V_target, synthesized_isometry(qcp))
@printf("After Phase 1: NLP subspace fidelity = %.6f  (time: %.1f s)\n", fid_p1, t_p1)

println("\n--- Phase 2: exact-Hessian refinement (max_iter=30) ---")
t_p2 = time()
solve!(qcp; max_iter=30)
t_p2 = time() - t_p2
V_opt = synthesized_isometry(qcp)
fid_p2 = subspace_fidelity(V_target, V_opt)
@printf("After Phase 2: NLP subspace fidelity = %.6f  (time: %.1f s)\n", fid_p2, t_p2)

V_rolled = rolled_out_isometry(qcp)
fid_rolled = subspace_fidelity(V_target, V_rolled)

println()
println("=" ^ 72)
println("Part (a) — subspace fidelity vs analytic [[5,1,3]]")
println("=" ^ 72)
@printf("NLP fidelity          = %.6f\n", fid_p2)
@printf("Rollout fidelity      = %.6f   (Tsit5, :constant interp, auto-detected)\n", fid_rolled)
@printf("Gap NLP - rollout     = %+.3e   (≪1 ⇒ dynamics constraints satisfied)\n",
        fid_p2 - fid_rolled)

# HANDOFF §M4 part (b): Petz recovery error on each single-qubit erasure
# must match the analytic [[5,1,3]] value (≈ 0) within 1e-3.
println()
println("=" ^ 72)
println("Part (b) — Petz recovery error on every single-qubit erasure")
println("=" ^ 72)
@printf("%-15s | %-18s | %-18s | %-12s\n", "erase qubit", "analytic V (M1)", "synthesized V_opt", "|Δ|")
println("-" ^ 72)
erasure_rows = Tuple{Int, Vector{Int}, Float64, Float64, Float64}[]
for q in 1:5
    A = collect(setdiff(1:5, [q]))
    err_analytic = petz_recovery_error(V_target, A)
    err_opt      = petz_recovery_error(V_rolled,  A)
    Δ = abs(err_opt - err_analytic)
    push!(erasure_rows, (q, A, err_analytic, err_opt, Δ))
end
for (q, A, err_analytic, err_opt, Δ) in erasure_rows
    flag = Δ < 1e-3 ? "" : "  ← exceeds 1e-3"
    @printf("erase q=%-2d (A=%s) | %.6e | %.6e | %.6e%s\n",
            q, A, err_analytic, err_opt, Δ, flag)
end
all_ok = all(row[5] < 1e-3 for row in erasure_rows)

t_elapsed = time() - t_start
println()
println("=" ^ 72)
@printf("Wall time: %.1f s   (Phase 1: %.1f s, Phase 2: %.1f s)\n",
        t_elapsed, t_p1, t_p2)

passes_a = fid_rolled > 0.99
passes_b = all_ok
if passes_a && passes_b
    println("PASS — M4 (XY variant) complete: (a) fidelity > 0.99 AND (b) erasure recovery within 1e-3")
else
    println("FAIL:  (a) $(passes_a ? "PASS" : "FAIL")   (b) $(passes_b ? "PASS" : "FAIL")")
end
