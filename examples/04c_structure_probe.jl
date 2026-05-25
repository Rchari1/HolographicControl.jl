# 04c_structure_probe.jl
#
# After Stage 1 confirmed all seeds cluster around 0.91 (H2 structural basin),
# probe WHAT KIND of mismatch the 0.91 V_opt has with V_target. Two
# hypotheses:
#
#   H_phase: V_opt's columns are individually close to V_target's columns
#       (per-column |⟨V|V_opt⟩|² ≈ 1) but with a relative phase mismatch
#       between them. → subspace fidelity is suppressed by the phase but the
#       per-column states are right. Fix: free_phase=true absorbs the phases.
#
#   H_struct: V_opt's columns are individually wrong (per-column ≈ 0.91 each).
#       The optimizer isn't getting close to the right STATES, not just the
#       right phases. Fix needs deeper changes (cubic splines, drive bounds,
#       drift, warm-start).
#
# Test: re-run seed=2 (the best Phase 1 seed) and inspect per-column data.

using LinearAlgebra
using Printf
using Random

using HolographicControl
using Piccolo: solve!, get_trajectory, get_system

V_target = five_qubit_isometry()
H_drift  = nn_heisenberg_drift(5; J=1.0)
H_drives = single_qubit_xy_drives(5)
drive_bounds = fill(1.0, length(H_drives))

println("=" ^ 72)
println("M4 escape — Stage 1b: structure probe on seed=2 (Phase 1 only)")
println("=" ^ 72)

qcp = isometry_synthesis_problem(
    V_target, H_drift, H_drives, drive_bounds;
    T = 25, duration = 10.0, seed = 2,
)

t0 = time()
solve!(qcp; max_iter=200, eval_hessian=false)
elapsed = time() - t0

V_opt = rolled_out_isometry(qcp)            # use the physical rollout (unit norms)
fid_subspace = subspace_fidelity(V_target, V_opt)

@printf("\nWall time: %.1f s\n", elapsed)
@printf("Subspace fidelity = %.6f\n\n", fid_subspace)

# Per-column overlaps and norms
println("Per-column diagnostics (V_opt from rolled-out, unit norms enforced):")
@printf("%-8s | %-14s | %-14s | %-12s\n",
        "column", "|⟨V|V_opt⟩|²", "⟨V|V_opt⟩ phase", "‖V_opt[:,j]‖²")
println("-" ^ 56)
for j in axes(V_target, 2)
    inner = dot(V_target[:, j], V_opt[:, j])
    overlap = abs2(inner)
    phase = angle(inner)
    norm_sq = abs2(norm(V_opt[:, j]))
    @printf("%-8d | %-14.6f | %+.4f rad     | %.6f\n",
            j, overlap, phase, norm_sq)
end

# Diagnostic logic
println()
println("=" ^ 72)
println("Diagnosis")
println("=" ^ 72)
overlaps = [abs2(dot(V_target[:, j], V_opt[:, j])) for j in axes(V_target, 2)]
phases   = [angle(dot(V_target[:, j], V_opt[:, j])) for j in axes(V_target, 2)]
mean_overlap = sum(overlaps) / length(overlaps)
phase_diff = phases[2] - phases[1]

@printf("Mean per-column overlap:        %.6f\n", mean_overlap)
@printf("Relative phase Δφ = φ_2 - φ_1:  %+.4f rad (%+.2f°)\n",
        phase_diff, rad2deg(phase_diff))
@printf("Subspace fidelity:              %.6f\n", fid_subspace)
println()

# If per-column overlaps are high (>0.97) but subspace fidelity is much
# lower, it's a phase issue. If per-column ≈ subspace, it's structural.
if mean_overlap > 0.97
    println("=> H_phase: per-column overlaps are high but subspace fidelity is")
    println("   suppressed. The optimizer is mostly hitting the right STATES but")
    println("   with a relative phase mismatch. free_phase=true should help.")
elseif mean_overlap < fid_subspace + 0.01
    println("=> H_struct: per-column overlaps are at the same level as subspace")
    println("   fidelity. The states themselves are wrong, not just the phases.")
    println("   Need deeper changes: cubic splines, drive bounds, drift, warm-start.")
else
    println("=> Mixed: per-column overlaps moderately exceed subspace fidelity.")
    println("   Both phase mismatch and structural deficits contribute. Worth")
    println("   trying free_phase first (cheap) and cubic splines if needed.")
end
