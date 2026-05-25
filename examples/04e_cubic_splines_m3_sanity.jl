# 04e_cubic_splines_m3_sanity.jl
#
# Before throwing cubic splines at M4 (where they may or may not escape the
# basin), sanity-check the cubic-spline path on M3 (3-qubit repetition).
# We already know M3 is reachable; this confirms the new code path produces
# fidelity > 0.99 on a known-solvable problem before we burn hours on M4.
#
# Usage:
#   julia --project=. examples/04e_cubic_splines_m3_sanity.jl
# Wall: ~5-10 min.

using LinearAlgebra
using Printf
using Random

using HolographicControl
using Piccolo: solve!

V_target = three_qubit_repetition_isometry()
H_drift  = nn_xx_yy_drift(3; J=1.0)
H_drives = single_qubit_xy_drives(3)
drive_bounds = fill(1.0, length(H_drives))

println("=" ^ 72)
println("M3 sanity check via cubic splines — should reach fidelity > 0.99")
println("System: 3Q XY drift + single-site X,Y drives,  N_knots=15, duration=10")
println("=" ^ 72)

qcp = isometry_synthesis_problem_cubic(
    V_target, H_drift, H_drives, drive_bounds;
    N_knots = 15,            # fewer than M3's T=100 ZOH; cubic is more expressive per knot
    duration = 10.0,
    Q = 100.0,
    R = 1e-2,
    du_bound = 10.0,
    seed = 0,
)

t0 = time()
println("\n--- Phase 1: L-BFGS (max_iter=200, eval_hessian=false) ---")
solve!(qcp; max_iter=200, eval_hessian=false)
t_p1 = time() - t0
V_p1 = synthesized_isometry(qcp)
fid_p1 = subspace_fidelity(V_target, V_p1)
@printf("Phase 1 NLP fidelity = %.6f  (wall %.1f s)\n", fid_p1, t_p1)

t1 = time()
println("\n--- Phase 2: exact Hessian (max_iter=30) ---")
solve!(qcp; max_iter=30)
t_p2 = time() - t1
V_p2 = synthesized_isometry(qcp)
fid_p2 = subspace_fidelity(V_target, V_p2)
@printf("Phase 2 NLP fidelity = %.6f  (wall %.1f s)\n", fid_p2, t_p2)

V_rolled = rolled_out_isometry(qcp)
fid_rolled = subspace_fidelity(V_target, V_rolled)
@printf("\nRollout fidelity (interpolation auto-detected = :cubic): %.6f\n", fid_rolled)
@printf("Gap NLP - rollout = %+.3e\n", fid_p2 - fid_rolled)

t_total = time() - t0
@printf("\nTotal wall: %.1f s\n", t_total)
if fid_rolled > 0.99
    println("PASS — cubic-spline path works on M3.")
else
    println("FAIL — cubic-spline path is BROKEN. Fix before attempting M4.")
end
