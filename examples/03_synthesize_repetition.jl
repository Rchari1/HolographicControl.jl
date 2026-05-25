# 03_synthesize_repetition.jl
#
# M3: first Piccolo isometry synthesis. Target: the 3-qubit repetition-code
# encoder
#     V|0⟩_L = |000⟩,   V|1⟩_L = |111⟩.
# System: 3-qubit chain with nearest-neighbor XX + YY drift and single-site
# X, Y controls on each qubit (6 controls total). HANDOFF §M3 success
# criterion: subspace fidelity > 0.99 in < 1 minute on a laptop.
#
# Usage:
#   julia --project=. examples/03_synthesize_repetition.jl

using LinearAlgebra
using Printf
using Random

using HolographicControl
using Piccolo: solve!, get_trajectory, ket_rollout, iso_to_ket, get_system

Random.seed!(0)

V_target = three_qubit_repetition_isometry()
H_drift  = nn_xx_yy_drift(3; J=1.0)
H_drives = single_qubit_xy_drives(3)
drive_bounds = fill(1.0, length(H_drives))

println("=" ^ 70)
println("M3 — synthesize 3-qubit repetition code via Piccolo")
println("V_target = 8×2 isometry; |0⟩_L = |000⟩, |1⟩_L = |111⟩")
println("Drift: nearest-neighbor XX+YY on 3 qubits  (J = 1.0)")
println("Drives: single-site X, Y on each of 3 qubits  (6 drives, bound = 1.0)")
println("=" ^ 70)

qcp = isometry_synthesis_problem(
    V_target, H_drift, H_drives, drive_bounds;
    T = 25,                       # reduced from 50 → quarters the Hessian cost
    duration = 10.0,              # Δt₀ ≈ 0.42 (Padé still accurate for J=1)
    Q = 100.0,
    R = 1e-2,
    ddu_bound = 1.0,
    seed = 0,
)

# Cold-start strategy per amico:setup Rule 1: L-BFGS to get into a basin, then
# exact Hessian to tighten feasibility. L-BFGS alone oscillates because it
# lacks the curvature info to balance objective vs constraint-violation; exact
# Hessian gives super-linear convergence on the dynamics equality constraints.
# Trade-off: exact Hessian iter cost scales O(n²) via ForwardDiff through `expv`;
# T=25 keeps this manageable (~tens of seconds per iter at this problem size).
t_start = time()

println("\n--- Phase 1: L-BFGS exploration (max_iter=200, eval_hessian=false) ---")
t_p1 = time()
solve!(qcp; max_iter=200, eval_hessian=false)
t_p1 = time() - t_p1
V_p1 = synthesized_isometry(qcp)
fid_p1_nlp = subspace_fidelity(V_target, V_p1)
@printf("After Phase 1: NLP subspace fidelity = %.6f  (time: %.1f s)\n", fid_p1_nlp, t_p1)

println("\n--- Phase 2: exact-Hessian refinement (max_iter=30) ---")
t_p2 = time()
solve!(qcp; max_iter=30)
t_p2 = time() - t_p2
V_p2 = synthesized_isometry(qcp)
fid_p2_nlp = subspace_fidelity(V_target, V_p2)
@printf("After Phase 2: NLP subspace fidelity = %.6f  (time: %.1f s)\n", fid_p2_nlp, t_p2)

t_elapsed = time() - t_start

# Use Phase-2 result for the final reporting and rollout verification.
V_opt = V_p2
fid_nlp = fid_p2_nlp

V_opt = synthesized_isometry(qcp)
fid_nlp = subspace_fidelity(V_target, V_opt)

println()
println("=" ^ 70)
println("As reported by the NLP trajectory")
println("=" ^ 70)
@printf("Subspace fidelity  |tr(V'V_opt)/d_bulk|²        = %.6f\n", fid_nlp)
println("Per-column |⟨V[:,j]|V_opt[:,j]⟩|² and norms:")
for j in axes(V_target, 2)
    overlap = abs2(dot(V_target[:, j], V_opt[:, j]))
    norm_sq = abs2(norm(V_opt[:, j]))
    @printf("    column %d:  |⟨·|·⟩|² = %.6f    ‖V_opt[:,%d]‖² = %.6f\n",
            j, overlap, j, norm_sq)
end

# Independent verification: take the optimized controls and roll them out with
# a high-accuracy adaptive ODE solver (Tsit5, abstol=reltol=1e-8). This bypasses
# the Padé dynamics constraints used during NLP and tells us what the actual
# physical trajectory does. If the L-BFGS solution is "the right answer for the
# wrong reason" the rollout fidelity will diverge from the NLP fidelity.
println()
println("=" ^ 70)
println("Independent rollout verification (Tsit5, tol=1e-8)")
println("=" ^ 70)

traj = get_trajectory(qcp)
sys  = get_system(qcp)
d_bulk = size(V_target, 2)
d_bdy  = size(V_target, 1)

# Roll out with BOTH interpolation modes. ZeroOrderPulse is piecewise-constant,
# so :constant should match the NLP's Bilinear/Padé physics. :linear interpolates
# between knots — represents a different pulse, useful as a sanity contrast.
function rollout_isometry(interp::Symbol)
    V_r = zeros(ComplexF64, d_bdy, d_bulk)
    for j in 1:d_bulk
        ψ̃_traj_j = ket_rollout(traj, sys; state_name=Symbol("ψ̃$j"), interpolation=interp)
        V_r[:, j] = iso_to_ket(ψ̃_traj_j[:, end])
    end
    return V_r
end

V_rolled_const  = rollout_isometry(:constant)
V_rolled_linear = rollout_isometry(:linear)

fid_const  = subspace_fidelity(V_target, V_rolled_const)
fid_linear = subspace_fidelity(V_target, V_rolled_linear)

@printf(":constant interpolation (matches ZeroOrderPulse physics):\n")
@printf("    subspace fidelity = %.6f\n", fid_const)
for j in axes(V_target, 2)
    overlap = abs2(dot(V_target[:, j], V_rolled_const[:, j]))
    norm_sq = abs2(norm(V_rolled_const[:, j]))
    @printf("    column %d: |⟨·|·⟩|² = %.6f  ‖·‖² = %.6f\n", j, overlap, norm_sq)
end
@printf("\n:linear interpolation (smooths knot jumps; for contrast only):\n")
@printf("    subspace fidelity = %.6f\n", fid_linear)
for j in axes(V_target, 2)
    overlap = abs2(dot(V_target[:, j], V_rolled_linear[:, j]))
    @printf("    column %d: |⟨·|·⟩|² = %.6f\n", j, overlap)
end

# Inspect the optimized controls to see if they're spiky (which would explain
# why interpolation choice matters so much).
u_opt = traj[:u]
println("\nOptimized control statistics (size $(size(u_opt))):")
for d in axes(u_opt, 1)
    drive_max = maximum(abs, u_opt[d, :])
    drive_jumps = maximum(abs, diff(u_opt[d, :]))
    @printf("    drive %d: max|u|=%.3f   max|Δu/step|=%.3f\n", d, drive_max, drive_jumps)
end

println()
println("=" ^ 70)
@printf("Wall time: %.1f s\n", t_elapsed)
fid_real = fid_const  # the physically-honest one
println(fid_real > 0.99 ? "PASS  (rollout :constant > 0.99)" : "FAIL  (rollout :constant ≤ 0.99)")
@printf("Gap   NLP - rollout_const = %+.3e\n", fid_nlp - fid_const)
@printf("Gap   const - linear      = %+.3e\n", fid_const - fid_linear)
