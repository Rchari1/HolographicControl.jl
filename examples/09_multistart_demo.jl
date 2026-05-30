# 09_multistart_demo.jl
#
# Demonstration of `multistart_synthesis`: the seed-sweep utility extracted
# from examples/04h_xy_seed_sweep.jl. The pattern was previously inline; it
# now lives in `src/optimization.jl` and is exported as a first-class
# package function.
#
# Target: 3-qubit repetition encoder. Cheap (~minutes) demo of the seed
# sweep — Phase-1-only L-BFGS across 4 seeds, prints the winner.
#
# Usage:
#   julia --project=. examples/09_multistart_demo.jl
# Wall: ~3-5 min (4 seeds × 100 iters of Phase-1 on a 3Q system).

using LinearAlgebra
using Printf

using HolographicControl

V_target = three_qubit_repetition_isometry()
H_drift  = nn_xx_yy_drift(3; J = 1.0)
H_drives = single_qubit_xy_drives(3)
drive_bounds = fill(1.0, length(H_drives))

println("=" ^ 72)
println("multistart_synthesis demo — 3-qubit repetition encoder")
println("Sweep seeds 0..3 with Phase-1-only L-BFGS (max_iter=100)")
println("=" ^ 72)

builder = seed -> isometry_synthesis_problem(
    V_target, H_drift, H_drives, drive_bounds;
    T = 25, duration = 10.0, seed = seed,
)

t0 = time()
(; results, winner) = multistart_synthesis(
    builder;
    seeds = 0:3,
    max_iter = 100,
    score = :rollout_fidelity,
    V_target = V_target,
    verbose = false,
)
total_wall = time() - t0

println()
@printf("%-6s | %-15s | %-10s\n", "seed", "rollout fid", "wall (s)")
println("-" ^ 40)
for r in results
    @printf("%-6d | %-15.6f | %-10.1f\n", r.seed, r.fidelity_to_target, r.wall)
end

println()
@printf("WINNER: seed=%d   rollout fidelity = %.6f\n",
        winner.seed, winner.fidelity_to_target)
@printf("Total sweep wall time: %.1f s\n", total_wall)

println()
println("(See examples/04h_xy_seed_sweep.jl for the original M4 inline sweep.)")
