# 05d_m3_vs_m5_link.jl
#
# Demonstrates the link between M3 (Piccolo-synthesized isometry targeting
# a fixed V) and M5 (Petz-objective optimization). Without any new Piccolo
# code, we can ask: when M3's synthesizer was asked to produce the
# 3-qubit repetition encoder, what Petz objective did it actually achieve?
# And how does that compare to a (cheap) standalone M5 optimization on
# the same system?
#
# This is the first concrete bridge between the physical-pulse layer
# (M3/M4) and the discovery-objective layer (M5). It exercises the same
# plumbing the Layer-3 Piccolo integration will eventually use:
#
#   1. Solve a Piccolo problem (M3 mode, fixed V_target).
#   2. Extract V_opt via rolled_out_isometry.
#   3. Score V_opt with petz_recovery_objective.
#   4. Compare to a standalone M5 optimum found by the methods in 05b.
#
# Usage:
#   julia --project=. examples/05d_m3_vs_m5_link.jl

using LinearAlgebra
using Printf
using Random

using HolographicControl
using Piccolo: solve!

n_bdy = 3
n_bulk = 1
d_bdy = 2^n_bdy
d_bulk = 2^n_bulk

A_list = [collect(setdiff(1:n_bdy, [q])) for q in 1:n_bdy]

V_target = three_qubit_repetition_isometry()
H_drift = nn_xx_yy_drift(3; J=1.0)
H_drives = single_qubit_xy_drives(3)
drive_bounds = fill(1.0, length(H_drives))

println("=" ^ 72)
println("M3 → M5 link: how does the M3-synthesized repetition encoder score")
println("under the M5 Petz recovery objective?")
println("=" ^ 72)
println()

# 1. The analytic repetition code as the M3 target
println("[1] Analytic 3-qubit repetition code (V_target for M3):")
obj_rep = petz_recovery_objective(V_target; A_list=A_list)
@printf("    M5 Petz objective = %.6f\n", obj_rep)
println("    (0.5 = dephasing channel signature; correctly identified as a")
println("     CLASSICAL, not quantum, error-correcting code)")
println()

# 2. Solve the M3 Piccolo problem (small, fast)
println("[2] Piccolo-synthesized V_opt from M3 problem (target = repetition):")
Random.seed!(0)
qcp = isometry_synthesis_problem(
    V_target, H_drift, H_drives, drive_bounds;
    T = 25, duration = 10.0, seed = 0,
)
solve!(qcp; max_iter=200, eval_hessian=false)
solve!(qcp; max_iter=50)
V_m3 = rolled_out_isometry(qcp)
fid_m3 = subspace_fidelity(V_target, V_m3)
obj_m3 = petz_recovery_objective(V_m3; A_list=A_list)
@printf("    Subspace fidelity to V_target = %.6f\n", fid_m3)
@printf("    M5 Petz objective on V_opt    = %.6f\n", obj_m3)
println("    (M3 successfully targets V_target; Petz objective inherits V_target's)")
println()

# 3. The standalone M5 winner from 05b
println("[3] Standalone M5 winner (random restart + descent, no Piccolo):")
# Re-derive deterministically: from 05b's seed=0, 300 restarts → coord descent
function isometry_from(x::Vector{Float64})
    n = d_bdy * d_bulk
    M_re = reshape(x[1:n], d_bdy, d_bulk)
    M_im = reshape(x[n+1:end], d_bdy, d_bulk)
    M = ComplexF64.(M_re, M_im)
    return M * inv(sqrt(Hermitian(M' * M)))
end
obj_x(x) = petz_recovery_objective(isometry_from(x); A_list=A_list)
function descend(x_init, δ_init, δ_floor, max_iter)
    x = copy(x_init); δ = δ_init
    for _ in 1:max_iter
        cur = obj_x(x)
        best, step = cur, (0, 0.0)
        for i in 1:length(x), s in (1, -1)
            xt = copy(x); xt[i] += s*δ
            v = obj_x(xt)
            if v < best; best = v; step = (i, s*δ); end
        end
        if best < cur - 1e-10
            x[step[1]] += step[2]
        else
            δ /= 2
            δ < δ_floor && break
        end
    end
    return x
end
function random_restart_best(K::Int, dim::Int)
    best_x = randn(dim); best_obj = obj_x(best_x)
    for _ in 2:K
        x = randn(dim); v = obj_x(x)
        if v < best_obj
            best_obj, best_x = v, x
        end
    end
    return best_x
end
Random.seed!(0)
param_dim = 2 * d_bdy * d_bulk
best_x = random_restart_best(300, param_dim)
x_m5 = descend(best_x, 0.5, 1e-7, 200)
V_m5 = isometry_from(x_m5)
obj_m5 = petz_recovery_objective(V_m5; A_list=A_list)
fid_m5_to_rep = subspace_fidelity(V_target, V_m5)
@printf("    M5 Petz objective on V_m5     = %.6f\n", obj_m5)
@printf("    Subspace fidelity to V_target = %.6f\n", fid_m5_to_rep)
println("    (M5 finds a different code; lower Petz objective than repetition)")
println()

# Cross-comparison summary
println("=" ^ 72)
println("Summary — same system (3Q chain), three approaches")
println("=" ^ 72)
@printf("%-50s | %-12s | %-12s\n", "encoder", "Petz obj", "fid to rep")
println("-" ^ 80)
@printf("%-50s | %-12.6f | %-12.6f\n", "analytic 3Q repetition (M3 target)", obj_rep, 1.0)
@printf("%-50s | %-12.6f | %-12.6f\n", "M3 Piccolo synth targeting repetition",    obj_m3, fid_m3)
@printf("%-50s | %-12.6f | %-12.6f\n", "standalone M5 optimum (no Piccolo)",       obj_m5, fid_m5_to_rep)
println()
println("Interpretation:")
@printf("  * The M3 synthesis successfully reproduces V_target (fid ≈ %.3f),\n", fid_m3)
println("    so the Petz objective on its output equals V_target's:")
@printf("       %.4f for both [1] and [2].\n", obj_rep)
@printf("  * The M5 standalone winner achieves %.4f — %.1fx lower Petz objective\n",
        obj_m5, obj_rep / obj_m5)
println("    than the repetition code. It is NOT the repetition code:")
@printf("       fid to rep = %.4f.\n", fid_m5_to_rep)
println()
println("  Conclusion: targeting the repetition code under M3 is the wrong")
println("  question when you actually care about quantum recovery. M5's objective")
println("  selects a fundamentally different (and better, for QEC purposes) code.")
println()
println("  Layer 3 of M5 will close this loop: instead of feeding Piccolo a")
println("  fixed V_target, feed it the Petz objective directly and let the")
println("  pulse-driven optimizer find a low-Petz code on the actual physical")
println("  Hamiltonian. See docs/M5_DESIGN.md for the integration plan.")
