# 04f_structural_sweep.jl
#
# After Stages 1a/1b/1c showed seeds, free_phase, and drive bounds aren't
# the issue, and cubic splines proved broken under BilinearIntegrator, the
# remaining cheap structural change is the (drift, control-set) pair
# itself. The HANDOFF prescribed Heisenberg + single-site X,Y but other
# combinations are physically interpretable and may have different basin
# geometry.
#
# Sweep five configurations under seed=2 (the best M4 Phase 1 seed) with
# Phase 1 only. Pick the winner, then commit Phase 2 budget to it.
#
# Usage:
#   julia --project=. examples/04f_structural_sweep.jl
# Wall: ~30-45 min (5 configs × ~5-8 min each).

using LinearAlgebra
using Printf
using Random

using HolographicControl
using Piccolo: solve!

V_target = five_qubit_isometry()
const SEED = 2

configs = [
    (name = "Heisenberg + X,Y         (baseline, known 0.94)",
        drift = nn_heisenberg_drift(5; J=1.0),
        drives = single_qubit_xy_drives(5)),
    (name = "XY-only drift + X,Y       (drop ZZ from drift)",
        drift = nn_xx_yy_drift(5; J=1.0),
        drives = single_qubit_xy_drives(5)),
    (name = "Heisenberg + X,Y,Z       (add Z controls)",
        drift = nn_heisenberg_drift(5; J=1.0),
        drives = single_qubit_xyz_drives(5)),
    (name = "XY-only drift + X,Y,Z    (drop ZZ, add Z controls)",
        drift = nn_xx_yy_drift(5; J=1.0),
        drives = single_qubit_xyz_drives(5)),
    (name = "ZZ-only drift + X,Y       (Ising + transverse drives)",
        drift = nn_zz_drift(5; J=1.0),
        drives = single_qubit_xy_drives(5)),
]

println("=" ^ 78)
println("M4 escape — Stage 2: structural sweep (Phase 1 only, seed=$SEED)")
println("=" ^ 78)
println("Each config: same V_target, same T=25, duration=10, |u|≤1.0, 200 L-BFGS iters")
println()

results = NamedTuple[]
for (idx, c) in enumerate(configs)
    println("\n[$idx/$(length(configs))] $(c.name)")
    println("    ($(length(c.drives)) drives)")
    drive_bounds = fill(1.0, length(c.drives))
    qcp = isometry_synthesis_problem(
        V_target, c.drift, c.drives, drive_bounds;
        T = 25, duration = 10.0, seed = SEED,
    )
    t0 = time()
    solve!(qcp; max_iter=200, eval_hessian=false)
    elapsed = time() - t0
    V_opt = rolled_out_isometry(qcp)
    fid = subspace_fidelity(V_target, V_opt)
    push!(results, (name=c.name, n_drives=length(c.drives), fid=fid, wall=elapsed))
    @printf("    → fidelity %.6f  (wall %.1f s)\n", fid, elapsed)
end

println()
println("=" ^ 78)
println("Summary")
println("=" ^ 78)
@printf("%-50s | %-8s | %-12s | %-9s\n", "config", "n_drives", "fidelity", "wall (s)")
println("-" ^ 90)
for r in results
    @printf("%-50s | %-8d | %-12.6f | %-9.1f\n", r.name, r.n_drives, r.fid, r.wall)
end
println()

best = argmax(r -> r.fid, results)
@printf("Best: %s  (fid %.6f)\n", best.name, best.fid)
if best.fid > 0.97
    println("=> Promising! Commit Phase 2 budget on this configuration.")
elseif best.fid > 0.95
    println("=> Marginal improvement. Phase 2 may close the gap; worth trying.")
else
    println("=> No structural change in this sweep gets us close. Need warm-start or")
    println("   a fundamentally different approach (gate decomposition).")
end
