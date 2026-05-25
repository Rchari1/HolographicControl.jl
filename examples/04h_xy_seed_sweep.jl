# 04h_xy_seed_sweep.jl
#
# At the WINNING configuration (XY drift + X,Y controls), Phase 1 with
# seed=2 reaches 0.983 and Phase 2 polishes to 0.987 — close to the 0.99
# bar but not over. The structural sweep tested only seed=2. Other seeds
# at this drift may land in deeper basins (the earlier seed diagnostic
# was on Heisenberg drift, which has stronger SO(3) symmetry and tighter
# clustering).
#
# Sweep Phase 1 on seeds 0, 1, 3, 4 at the XY-drift configuration. If any
# seed reaches > 0.985 in Phase 1, it's worth Phase-2 budget.
#
# Usage:
#   julia --project=. examples/04h_xy_seed_sweep.jl
# Wall: ~15-20 min.

using LinearAlgebra
using Printf
using Random

using HolographicControl
using Piccolo: solve!

V_target = five_qubit_isometry()
H_drift  = nn_xx_yy_drift(5; J=1.0)
H_drives = single_qubit_xy_drives(5)
drive_bounds = fill(1.0, length(H_drives))

const SEEDS = [0, 1, 3, 4]  # seed=2 known: Phase 1 0.983, Phase 2 0.987

println("=" ^ 72)
println("M4 XY-drift seed sweep — Phase 1 only on seeds $(SEEDS)")
println("(seed=2 known: Phase 1 0.983 / Phase 2 0.987)")
println("=" ^ 72)

results = NamedTuple[]
for seed in SEEDS
    println("\n--- seed = $seed ---")
    qcp = isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 25, duration = 10.0, seed = seed,
    )
    t0 = time()
    solve!(qcp; max_iter=200, eval_hessian=false)
    elapsed = time() - t0
    V_opt = rolled_out_isometry(qcp)
    fid = subspace_fidelity(V_target, V_opt)
    push!(results, (seed=seed, fid=fid, wall=elapsed))
    @printf("seed=%d: Phase 1 fidelity = %.6f  (wall %.1f s)\n", seed, fid, elapsed)
end

println()
println("=" ^ 72)
println("Summary (XY drift + single-site X,Y, T=25, duration=10, |u|≤1.0)")
println("=" ^ 72)
@printf("%-6s | %-12s | %-10s\n", "seed", "Phase 1 fid", "wall (s)")
println("-" ^ 36)
@printf("%-6d | %-12.6f | %-10.1f    (known)\n", 2, 0.982732, 224.7)
for r in results
    @printf("%-6d | %-12.6f | %-10.1f\n", r.seed, r.fid, r.wall)
end

best = argmax(r -> r.fid, results)
@printf("\nBest of new seeds: seed=%d at fidelity %.6f\n", best.seed, best.fid)
if best.fid > 0.985
    println("=> Worth Phase 2 budget on this seed.")
elseif best.fid > 0.983
    println("=> Marginal improvement on seed=2. Maybe not worth Phase 2.")
else
    println("=> No improvement on seed=2. Accept 0.987 as the M4 best and move to M5.")
end
