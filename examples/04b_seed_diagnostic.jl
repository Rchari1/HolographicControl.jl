# 04b_seed_diagnostic.jl
#
# Stage 1 of the M4 escape plan: probe the optimization landscape by running
# Phase 1 (L-BFGS, no exact Hessian) on several different random seeds.
#
# Purpose: distinguish two hypotheses for why M4 stagnates at fidelity 0.91:
#
#   H1 (seed-dependent basin): seed=0 happened to land in a 0.91 basin, but
#       other seeds land in deeper basins. → multistart will fix M4.
#
#   H2 (structural basin):     all seeds converge to ~0.91 because the
#       0.91 attractor is symmetric/dominant in this landscape. → multistart
#       won't help; need to change the problem (free_phase, cubic splines,
#       different drift, warm-start).
#
# We run Phase 1 only (eval_hessian=false, ~5 min per seed at T=25). We
# don't pay for Phase 2 because we only need a coarse fidelity reading per
# seed, not a KKT-feasible solution.
#
# Usage:
#   julia --project=. examples/04b_seed_diagnostic.jl
#
# Wall time: ~25 min (4 seeds × ~5 min Phase 1 each, plus load).

using LinearAlgebra
using Printf
using Random

using HolographicControl
using Piccolo: solve!

const SEEDS = [1, 2, 3, 4]  # seed 0 is already known to land at 0.912

V_target = five_qubit_isometry()
H_drift  = nn_heisenberg_drift(5; J=1.0)
H_drives = single_qubit_xy_drives(5)
drive_bounds = fill(1.0, length(H_drives))

println("=" ^ 72)
println("M4 escape — Stage 1 seed diagnostic")
println("Target: five_qubit_isometry(),  Drift: nn-Heisenberg,  T=25, duration=10")
println("Running Phase 1 (L-BFGS, 200 iter, no exact Hessian) per seed.")
println("Known: seed=0 → fidelity 0.912 (after both phases).")
println("=" ^ 72)

results = Dict{Int, NamedTuple}()
total_start = time()

for seed in SEEDS
    println("\n--- seed = $seed ---")
    qcp = isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 25,
        duration = 10.0,
        Q = 100.0,
        R = 1e-2,
        ddu_bound = 1.0,
        seed = seed,
    )
    t0 = time()
    solve!(qcp; max_iter=200, eval_hessian=false)
    elapsed = time() - t0
    V_phase1 = synthesized_isometry(qcp)
    fid = subspace_fidelity(V_target, V_phase1)
    results[seed] = (fidelity=fid, wall=elapsed)
    @printf("seed=%d: Phase 1 fidelity = %.6f  (wall %.1f s)\n", seed, fid, elapsed)
end

total_elapsed = time() - total_start

# Summary table — the key output of this diagnostic
println()
println("=" ^ 72)
println("Summary")
println("=" ^ 72)
@printf("%-6s | %-12s | %-10s\n", "seed", "Phase 1 fid", "wall (s)")
println("-" ^ 36)
@printf("%-6d | %-12.6f | %-10.1f    (known: from prior run, Phase 2 took it to 0.912)\n", 0, 0.911148, 315.0)
for seed in SEEDS
    r = results[seed]
    @printf("%-6d | %-12.6f | %-10.1f\n", seed, r.fidelity, r.wall)
end
println()
@printf("Total wall: %.1f s = %.1f min\n", total_elapsed, total_elapsed / 60)

# Decision criterion
println()
all_fids = vcat([0.911148], [results[s].fidelity for s in SEEDS])
fid_range = maximum(all_fids) - minimum(all_fids)
best_fid = maximum(all_fids)
@printf("Fidelity range across all 5 seeds: %.6f  (%.6f → %.6f)\n",
        fid_range, minimum(all_fids), maximum(all_fids))
println()
if best_fid > 0.95
    println("=> H1 supported (at least one seed > 0.95):")
    println("   the 0.91 basin is NOT a structural attractor; multistart will likely succeed.")
elseif fid_range > 0.05
    println("=> Partial H1 support (seeds spread > 0.05):")
    println("   landscape has multiple basins; multistart with more seeds may eventually win.")
else
    println("=> H2 supported (all seeds within 0.05 of each other):")
    println("   the 0.91 basin is structural. Multistart unlikely to help.")
    println("   Recommended next: cubic splines or alternative drift/controls.")
end
