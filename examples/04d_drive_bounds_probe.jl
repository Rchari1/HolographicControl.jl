# 04d_drive_bounds_probe.jl
#
# Structure probe revealed H_struct (states themselves wrong, not just phase).
# This script tests whether the structural basin is caused by drive
# *amplitude* being too small relative to the J=1 drift.
#
# Heisenberg drift with J=1 and drive bound |u| ≤ 1 means drives and drift
# are equally strong. The [[5,1,3]] is highly entangled and may require more
# control authority than this to be reached via L-BFGS. HANDOFF §7 cautions
# against increasing drive bounds, but the user has authorized autonomy for
# the M4 escape, and we document the change here.
#
# Test: seed=2 (best Phase 1 seed at default bounds), Phase 1 only, with
# drive_bounds escalated to 2.0 and 4.0.

using LinearAlgebra
using Printf

using HolographicControl
using Piccolo: solve!

V_target = five_qubit_isometry()
H_drift  = nn_heisenberg_drift(5; J=1.0)
H_drives = single_qubit_xy_drives(5)
const SEED = 2

println("=" ^ 72)
println("M4 escape — Stage 1c: drive-bounds probe (seed=2, Phase 1 only)")
println("=" ^ 72)
println("Heisenberg J=1.0, T=25, duration=10.")
println()

results = NamedTuple[]
for bound in (1.0, 2.0, 4.0)
    println("\n--- drive bound |u| ≤ $bound ---")
    drive_bounds = fill(bound, length(H_drives))
    qcp = isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 25, duration = 10.0, seed = SEED,
    )
    t0 = time()
    solve!(qcp; max_iter=200, eval_hessian=false)
    elapsed = time() - t0
    V_opt = rolled_out_isometry(qcp)
    fid = subspace_fidelity(V_target, V_opt)
    push!(results, (bound=bound, fid=fid, wall=elapsed))
    @printf("|u| ≤ %.1f → Phase 1 rollout fidelity = %.6f  (wall %.1f s)\n",
            bound, fid, elapsed)
end

println()
println("=" ^ 72)
println("Summary")
println("=" ^ 72)
@printf("%-12s | %-12s | %-10s\n", "drive bound", "fidelity", "wall (s)")
println("-" ^ 38)
for r in results
    @printf("%-12.1f | %-12.6f | %-10.1f\n", r.bound, r.fid, r.wall)
end
println()

best = argmax(r -> r.fid, results)
@printf("Best: bound=%.1f, fid=%.6f\n", best.bound, best.fid)
if best.fid > 0.97
    println("=> Significant improvement with larger bounds. Drive authority was the bottleneck.")
elseif best.fid > maximum(r.fid for r in results[1:1]) + 0.02
    println("=> Modest improvement with larger bounds. Worth combining with Phase 2.")
else
    println("=> No meaningful improvement. Bounds are NOT the structural bottleneck.")
    println("   Next: cubic splines.")
end
