# 05e_piccolo_m5_n3.jl
#
# M5 Layer 3 — the actual thesis contribution end-to-end.
#
# Run Piccolo with the Petz aggregate recovery error as the optimization
# objective (no fixed V_target) on the (n_bdy=3, n_bulk=1) system. The
# optimizer searches over physical control pulses on a real Hamiltonian
# and finds the lowest-Petz-error code reachable from random init.
#
# Compare:
#   * Standalone M5 optimum (no Piccolo, polar projection on Stiefel):
#     0.067 — the manifold-only result from examples/05b.
#   * M3 Piccolo synthesis targeting the 3-qubit repetition code:
#     Petz objective 0.500 (dephasing channel, classical code).
#   * THIS — Piccolo with the Petz objective directly on the same
#     physical system as M3.
#
# Strategy: cold start + two-phase solve (L-BFGS exploration → exact
# Hessian refinement). Exact Hessian works because the smooth Petz path
# (Denman–Beavers inverse sqrt, no eigen) is ForwardDiff-traceable
# through both gradient and Hessian.
#
# Usage:
#   julia --project=. examples/05e_piccolo_m5_n3.jl
# Wall: ~10-30 min depending on convergence.

using LinearAlgebra
using Printf
using Random

using HolographicControl
using Piccolo: solve!

# --- physical system: 3Q chain, XY drift, single-site X,Y controls
n_bdy = 3
n_bulk = 1
H_drift  = nn_xx_yy_drift(n_bdy; J=1.0)
H_drives = single_qubit_xy_drives(n_bdy)
drive_bounds = fill(1.0, length(H_drives))

# Petz objective: uniform weight over all weight-1 erasures.
A_list = uniform_erasure_subregions(n_bdy, 1)

println("=" ^ 72)
println("M5 Layer 3 — Piccolo + Petz objective on (n_bdy=$n_bdy, n_bulk=$n_bulk)")
println("Physical system: XY drift, J=1.0, |u|≤1.0 single-site X,Y controls")
println("Erasure pattern: uniform over $(length(A_list)) weight-1 subregions")
println("=" ^ 72)

qcp = petz_isometry_synthesis_problem(
    H_drift, H_drives, drive_bounds;
    n_bdy = n_bdy, n_bulk = n_bulk,
    A_list = A_list,
    T = 25, duration = 10.0,
    Q_petz = 100.0,
    ε_petz = 1e-6,
    seed = 0,
)

t_start = time()

# --- Phase 1: L-BFGS exploration
println("\n--- Phase 1: L-BFGS (max_iter=200, eval_hessian=false) ---")
t_p1 = time()
solve!(qcp; max_iter=200, eval_hessian=false)
t_p1 = time() - t_p1

V_p1_nlp = synthesized_isometry(qcp)
V_p1_rolled = rolled_out_isometry(qcp)
obj_p1_nlp = try
    petz_recovery_objective(V_p1_nlp; A_list=A_list)
catch
    NaN
end
obj_p1_rolled = petz_recovery_objective(V_p1_rolled; A_list=A_list)
@printf("Phase 1: NLP obj = %.6e   rollout obj = %.6e   (wall %.1f s)\n",
        obj_p1_nlp, obj_p1_rolled, t_p1)
@printf("         (NLP and rollout agree only when dynamics constraints are tight)\n")

# --- Phase 2: exact Hessian refinement
println("\n--- Phase 2: exact Hessian (max_iter=30) ---")
t_p2 = time()
solve!(qcp; max_iter=30)
t_p2 = time() - t_p2

V_p2_nlp = synthesized_isometry(qcp)
V_p2_rolled = rolled_out_isometry(qcp)
obj_p2_nlp = try
    petz_recovery_objective(V_p2_nlp; A_list=A_list)
catch
    NaN
end
obj_p2_rolled = petz_recovery_objective(V_p2_rolled; A_list=A_list)
@printf("Phase 2: NLP obj = %.6e   rollout obj = %.6e   (wall %.1f s)\n",
        obj_p2_nlp, obj_p2_rolled, t_p2)

t_elapsed = time() - t_start

# --- Final result
println()
println("=" ^ 72)
println("Final M5 Layer 3 result on (n_bdy=$n_bdy, n_bulk=$n_bulk)")
println("=" ^ 72)
@printf("Petz objective (rollout V_opt):    %.6f\n", obj_p2_rolled)
println()
println("Per-erasure breakdown:")
for A in A_list
    @printf("  keep %s : %.6f\n", A, petz_recovery_error(V_p2_rolled, A))
end

println()
println("Reference points:")
println("  random isometry                  : ~0.165")
println("  3Q repetition (M3 Piccolo target): 0.500")
@printf("  standalone M5 optimum (05b)      : 0.067\n")
@printf("  THIS (Piccolo + Petz obj)        : %.6f\n", obj_p2_rolled)
println()

if obj_p2_rolled < 0.1
    println("PASS — Piccolo found a code with Petz objective < 0.1 via PHYSICAL")
    println("       Hamiltonian dynamics, on par with the standalone manifold optimum.")
elseif obj_p2_rolled < 0.2
    println("PARTIAL — Petz objective below 0.2 (beats random), but not yet at the")
    println("          standalone optimum. May need warm-start or more iterations.")
else
    println("WEAK — optimizer didn't escape into a low-Petz region from cold start.")
    println("       Try: warm-start from M3's converged pulse, or longer Phase 1.")
end

@printf("\nWall time: %.1f s   (Phase 1: %.1f s, Phase 2: %.1f s)\n",
        t_elapsed, t_p1, t_p2)
