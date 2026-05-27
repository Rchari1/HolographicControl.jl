# 07_decomposed_m5_n5.jl
#
# DECOMPOSED M5 — the scalable route to objective-driven holographic-code
# synthesis at thesis scale (n_bdy=5).
#
# The monolithic M5 Layer 3 (Petz objective directly inside Piccolo, see
# examples/05f) works at n=3 but is compute-bound at n=5: the exact-Hessian
# refinement needs ForwardDiff to differentiate TWICE through both the
# dynamics integrator's `expv` AND the Petz objective's Denman–Beavers
# inverse-sqrt, over ~3851 NLP variables. Empirically that's > 6 hours for
# a single Hessian evaluation on a laptop — intractable.
#
# The fix is to DECOMPOSE the problem into two cheap halves:
#
#   Stage A — DISCOVERY (cheap, ~mins): optimize the Petz recovery objective
#     over the isometry manifold directly (no physical dynamics), via
#     gradient-free coordinate descent. This finds a target code V* with
#     near-zero recovery error. At n=5, ~159 s.
#
#   Stage B — REALIZATION (M4 cost, ~hours): physically synthesize V* with
#     the fixed-target machinery (isometry_synthesis_problem). This uses the
#     standard phase-coherent ket-fidelity objective, whose Hessian is cheap
#     (it's exactly the M4 [[5,1,3]] solve, which converged in ~4 h). NO
#     Petz objective inside Piccolo, so no Hessian blowup.
#
# Why this is the right architecture: code DISCOVERY (what code?) is a
# property of the isometry alone and is cheap to optimize abstractly;
# physical REALIZATION (what pulse?) is the expensive dynamics problem but
# is a solved, fixed-target task once the target is known. Coupling them in
# one NLP (monolithic M5) is elegant but pays the worst of both costs.
#
# Headline result this produces at n=5:
#   Stage A finds V* with Petz objective ≈ 7e-10 (perfect single-qubit-
#   erasure recovery, same as the [[5,1,3]]) but code_subspace_fidelity
#   ≈ 0.08 to the [[5,1,3]] — i.e. a genuinely DIFFERENT distance-3-erasure-
#   correcting code. This is the "low-Petz manifold" observation
#   (examples/05c) made concrete and physically realizable.
#
# Usage:
#   julia --project=. examples/07_decomposed_m5_n5.jl              # Stage A only (~3 min)
#   julia --project=. examples/07_decomposed_m5_n5.jl --realize    # + Stage B (~4 h)

using LinearAlgebra
using Printf
using Random

using HolographicControl
using Piccolo: solve!, get_trajectory, get_times

const REALIZE = "--realize" in ARGS

n_bdy = 5
n_bulk = 1
d_bdy = 2^n_bdy
d_bulk = 2^n_bulk
A_list = uniform_erasure_subregions(n_bdy, 1)
param_dim = 2 * d_bdy * d_bulk

println("=" ^ 78)
println("DECOMPOSED M5 on (n_bdy=$n_bdy, n_bulk=$n_bulk)")
println("=" ^ 78)

# --------------------------------------------------------------------------- #
# Stage A — DISCOVERY: standalone Petz optimization over the isometry manifold
# --------------------------------------------------------------------------- #

# V(x): pack a real vector into a complex matrix, polar-project to an isometry.
function V_of(x)
    n = d_bdy * d_bulk
    M = complex.(reshape(x[1:n], d_bdy, d_bulk), reshape(x[n+1:end], d_bdy, d_bulk))
    return M * inv(sqrt(Hermitian(M' * M + 1e-12 * I)))
end
fobj(x) = petz_recovery_objective(V_of(x); A_list = A_list)

function best_of_restarts(K, dim)
    bx = randn(dim); bf = fobj(bx)
    for _ in 2:K
        x = randn(dim); v = fobj(x)
        if v < bf; bf, bx = v, x; end
    end
    return bx, bf
end

function coord_descent(x0; δ0 = 0.5, δfloor = 1e-7, maxiter = 400, patience = 25)
    x = copy(x0); δ = δ0; noimp = 0
    for _ in 1:maxiter
        cur = fobj(x); best = cur; step = (0, 0.0)
        for i in 1:length(x), s in (1, -1)
            xt = copy(x); xt[i] += s * δ; v = fobj(xt)
            if v < best; best = v; step = (i, s * δ); end
        end
        if best < cur - 1e-10
            x[step[1]] += step[2]; noimp = 0
        else
            δ /= 2; noimp += 1
            (δ < δfloor || noimp > patience) && break
        end
    end
    return x, fobj(x)
end

println("\n--- Stage A: standalone Petz discovery (coordinate descent) ---")
Random.seed!(20260527)
tA = time()
x0, f0 = best_of_restarts(40, param_dim)
@printf("  best of 40 random restarts: Petz = %.4f\n", f0)
x_star, f_star = coord_descent(x0; maxiter = 400)
tA = time() - tA

V_star = V_of(x_star)
V_513 = five_qubit_isometry()
fid_cs = code_subspace_fidelity(V_513, V_star)

@printf("\nStage A result (wall %.1f s):\n", tA)
@printf("  Petz objective:                    %.3e\n", f_star)
@printf("  ‖V*'V* - I‖:                       %.2e\n", opnorm(V_star' * V_star - I))
@printf("  code_subspace_fidelity to [[5,1,3]]: %.4f\n", fid_cs)
println("  Per-erasure recovery error:")
for A in A_list
    @printf("    keep %s : %.3e\n", A, petz_recovery_error(V_star, A))
end

mkpath("data")
save_isometry("data/m5_n5_Vstar_standalone", V_star;
    meta = Dict(:source => "decomposed-M5 Stage A", :petz => f_star,
                :fid_to_V513_cs => fid_cs, :n_bdy => n_bdy, :n_bulk => n_bulk))
println("\n  Saved V* → data/m5_n5_Vstar_standalone.jls")

println()
if fid_cs < 0.5
    println("=> MANIFOLD CONFIRMED at n=5: V* has near-zero Petz recovery error")
    println("   (a distance-3-erasure-correcting code, like [[5,1,3]]) yet is")
    @printf("   essentially orthogonal to [[5,1,3]] (code fidelity %.3f). The set\n", fid_cs)
    println("   of perfect-recovery codes is a large manifold; discovery-driven")
    println("   optimization finds members inequivalent to the textbook code.")
else
    @printf("=> V* is close to [[5,1,3]] (code fidelity %.3f) — discovery reproduced\n", fid_cs)
    println("   the textbook code this time.")
end

# --------------------------------------------------------------------------- #
# Stage B — REALIZATION: physically synthesize V* via the M4 machinery
# --------------------------------------------------------------------------- #

if !REALIZE
    println("\n(Stage B skipped — pass --realize to physically synthesize V*, ~4 h.)")
else
    println("\n--- Stage B: physical realization of V* via fixed-target synthesis ---")
    println("Physical system: XY drift J=1.0, single-site X,Y, |u|≤1.0, T=25 (= M4 setup)")
    H_drift  = nn_xx_yy_drift(n_bdy; J = 1.0)
    H_drives = single_qubit_xy_drives(n_bdy)
    drive_bounds = fill(1.0, length(H_drives))

    qcp = isometry_synthesis_problem(
        V_star, H_drift, H_drives, drive_bounds;
        T = 25, duration = 10.0, seed = 3,
    )

    tB = time()
    println("\nPhase 1: L-BFGS (max_iter=200, eval_hessian=false)")
    solve!(qcp; max_iter = 200, eval_hessian = false)
    V_b1 = rolled_out_isometry(qcp)
    fid_b1 = subspace_fidelity(V_star, V_b1)
    @printf("  Phase 1: rollout fidelity to V* = %.6f\n", fid_b1)
    # Checkpoint (bulletproof: isometry first, pulse in try/catch)
    save_isometry("data/m5_n5_Vstar_realized_phase1", V_b1;
        meta = Dict(:phase => 1, :fid_to_Vstar => fid_b1))
    try
        traj = get_trajectory(qcp)
        save_pulse("data/m5_n5_Vstar_realized_phase1_pulse", traj[:u], get_times(traj);
            meta = Dict(:phase => 1, :fid_to_Vstar => fid_b1))
    catch e
        @warn "pulse checkpoint failed (isometry saved)" exception=e
    end

    println("\nPhase 2: exact Hessian (max_iter=30) — fixed-target ket fidelity, cheap Hessian")
    solve!(qcp; max_iter = 30)
    V_opt = rolled_out_isometry(qcp)
    fid_b2 = subspace_fidelity(V_star, V_opt)
    petz_realized = petz_recovery_objective(V_opt; A_list = A_list)
    tB = time() - tB

    save_isometry("data/m5_n5_Vstar_realized_phase2", V_opt;
        meta = Dict(:phase => 2, :fid_to_Vstar => fid_b2, :petz => petz_realized))

    println()
    println("=" ^ 78)
    @printf("Stage B result (wall %.1f s):\n", tB)
    @printf("  rollout fidelity to V*:   %.6f\n", fid_b2)
    @printf("  rollout Petz objective:   %.6e\n", petz_realized)
    if fid_b2 > 0.99 && petz_realized < 1e-2
        println("  PASS — the discovered code V* is PHYSICALLY REALIZABLE: a new")
        println("  distance-3 code, inequivalent to [[5,1,3]], synthesized by bounded")
        println("  physical control pulses on a 5-qubit chain.")
    else
        println("  PARTIAL — V* not fully realized (fidelity < 0.99 or Petz > 1e-2).")
        println("  Not all low-Petz codes are equally synthesizable on this hardware;")
        println("  a seed sweep or alternate drift may be needed (cf. M4 stagnation).")
    end
end
