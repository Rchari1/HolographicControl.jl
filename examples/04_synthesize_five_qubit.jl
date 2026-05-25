# 04_synthesize_five_qubit.jl
#
# M4 — synthesize the [[5,1,3]] perfect code via Piccolo.
#
# STATUS: STAGNATION CASE STUDY. Cold-start L-BFGS + exact-Hessian Phase 2
# converges KKT-exact (inf_du ~ 1e-9, inf_pr ~ 1e-15) but lands in a strong
# local minimum at subspace fidelity ≈ 0.912 — below the success criterion
# of 0.99. Increasing T from 25 → 50 (more control resolution) did NOT escape
# the basin: both T values found the same attractor under seed=0. Likely
# escapes (untested in this branch):
#   - multistart over multiple seeds + take best (amico:multistart skill)
#   - free_phase=true on the SmoothPulseProblem (if [[5,1,3]] target is
#     equivalent up to per-qubit Z phases under the Heisenberg drift)
#   - alternative drift (e.g., XY + single-site Z controls)
#   - warm-start from M1's analytic V (rather than cold-start)
#
# This script is preserved as the diagnostic that exposed the stagnation and
# as a foundation for the eventual escape attempt.
#
# Target / system (HANDOFF §M4):
#   - 5 qubits (n_bdy = 5, n_bulk = 1)
#   - Drift: nearest-neighbor Heisenberg, J = 1.0
#   - Drives: single-site X, Y on each qubit (10 drives, |u|≤1)
#
# Success criteria (currently failing):
#   (a) subspace fidelity > 0.99   — got 0.912
#   (b) Petz erasure recovery error within 1e-3 of analytic ≈ 0
#
# Usage:
#   julia --project=. examples/04_synthesize_five_qubit.jl
# Wall time: ~2 hours on a laptop (Phase 2 exact Hessian dominates).

using LinearAlgebra
using Printf
using Random

using HolographicControl
using Piccolo: solve!, get_trajectory, get_system

Random.seed!(0)

V_target = five_qubit_isometry()
H_drift  = nn_heisenberg_drift(5; J=1.0)
H_drives = single_qubit_xy_drives(5)         # 10 drives: X_1,Y_1,...,X_5,Y_5
drive_bounds = fill(1.0, length(H_drives))

println("=" ^ 72)
println("M4 — synthesize [[5,1,3]] perfect-code encoding isometry via Piccolo")
println("Target: 32×2 isometry from five_qubit_isometry()")
println("Drift: nearest-neighbor Heisenberg on 5 qubits (J = 1.0)")
println("Drives: single-site X,Y on each of 5 qubits  (10 drives, bound = 1.0)")
println("=" ^ 72)

qcp = isometry_synthesis_problem(
    V_target, H_drift, H_drives, drive_bounds;
    T = 50,                       # bumped from 25 → 50 for more control resolution
    duration = 10.0,
    Q = 100.0,
    R = 1e-2,
    ddu_bound = 1.0,
    seed = 0,
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

# Physical verification: roll the optimized piecewise-constant controls
# through a fresh Tsit5 ODE solver (interpolation=:constant baked into
# `rolled_out_isometry`).
V_rolled = rolled_out_isometry(qcp)
fid_rolled = subspace_fidelity(V_target, V_rolled)

# HANDOFF §M4 part (a)
println()
println("=" ^ 72)
println("Part (a) — subspace fidelity vs analytic [[5,1,3]]")
println("=" ^ 72)
@printf("NLP fidelity          = %.6f\n", fid_p2)
@printf("Rollout fidelity      = %.6f   (Tsit5, :constant interp)\n", fid_rolled)
@printf("Gap NLP - rollout     = %+.3e   (≪1 ⇒ dynamics constraints satisfied)\n",
        fid_p2 - fid_rolled)

# HANDOFF §M4 part (b): Petz recovery error on each single-qubit erasure must
# match the analytic [[5,1,3]] value (≈ 0) within 1e-3.
println()
println("=" ^ 72)
println("Part (b) — Petz recovery error on every single-qubit erasure")
println("=" ^ 72)
println("Each row compares analytic vs synthesized; both should be ≈ 0.")
println()
@printf("%-15s | %-18s | %-18s | %-12s\n", "erase qubit", "analytic V (M1)", "synthesized V_opt", "|Δ|")
println("-" ^ 72)
# Collect first (avoids soft-scope ambiguity on `all_ok` inside the for-loop)
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

# Final wall-time report
t_elapsed = time() - t_start
println()
println("=" ^ 72)
@printf("Wall time: %.1f s   (Phase 1: %.1f s, Phase 2: %.1f s)\n",
        t_elapsed, t_p1, t_p2)

passes_a = fid_rolled > 0.99
passes_b = all_ok
if passes_a && passes_b
    println("PASS — M4 complete: (a) fidelity > 0.99 AND (b) erasure recovery within 1e-3")
else
    println("FAIL:  (a) $(passes_a ? "PASS" : "FAIL")   (b) $(passes_b ? "PASS" : "FAIL")")
end
