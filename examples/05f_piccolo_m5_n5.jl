# 05f_piccolo_m5_n5.jl
#
# M5 Layer 3 at thesis scale: Piccolo with Petz objective on (n_bdy=5,
# n_bulk=1) — the same physical system as M4 (5Q chain, XY drift,
# single-site X,Y controls, |u|≤1.0).
#
# UNLIKE M4 this run has NO fixed V_target. The optimizer searches over
# physical control pulses on a real Hamiltonian to minimize the Petz
# aggregate over weight-1 erasures.
#
# THE NOVEL TEST. At this system size a perfect QEC code DOES exist (the
# [[5,1,3]] from M1 achieves Petz objective ≈ 0). Two outcomes are
# scientifically interesting:
#
#   1. M5 finds a code with Petz ≈ 0 that IS the [[5,1,3]] up to a logical
#      basis rotation (high `code_subspace_fidelity` to V_513). Confirms
#      M5 can reproduce M4's result without being told the target.
#
#   2. M5 finds a code with Petz ≈ 0 that is NOT the [[5,1,3]] (low
#      `code_subspace_fidelity` to V_513). Confirms the MANIFOLD
#      observation from the standalone 05c: under physical Hamiltonian
#      constraints there are MULTIPLE inequivalent codes with the same
#      recovery properties, and discovery-driven optimization finds them.
#
# Either way — the M3 contrast holds:
#   M4 (Piccolo fixed target [[5,1,3]]):    Petz objective ≤ 1e-3 (M4 success)
#   M5 (Piccolo with Petz objective):       Petz objective ≤ 1e-3 (this run, hopefully)
#
# Cost estimate based on n=3 Petz solve + M4 XY-drift scaling: ~5-6
# hours wall. Phase 1 L-BFGS ~5 min; Phase 2 exact Hessian dominates.
#
# Usage:
#   julia --project=. examples/05f_piccolo_m5_n5.jl

using LinearAlgebra
using Printf
using Random

using HolographicControl
using Piccolo: solve!, get_trajectory, get_times

# --- physical system: same as M4
n_bdy = 5
n_bulk = 1
H_drift  = nn_xx_yy_drift(n_bdy; J=1.0)
H_drives = single_qubit_xy_drives(n_bdy)
drive_bounds = fill(1.0, length(H_drives))

# Petz objective: uniform over all 5 weight-1 erasures (the smallest
# meaningful target — the same A_list as the M4 verification used)
A_list = uniform_erasure_subregions(n_bdy, 1)

println("=" ^ 78)
println("M5 Layer 3 — Piccolo + Petz objective on (n_bdy=$n_bdy, n_bulk=$n_bulk)")
println("Physical system: identical to M4 (XY drift, J=1.0, single-site X,Y, |u|≤1.0)")
println("Erasure pattern: uniform over $(length(A_list)) weight-1 subregions")
println("=" ^ 78)

# At n=5, seed selection mattered for M4 (the XY-drift seed sweep showed
# seed=3 had the best Phase 1 basin). Try the same seed here so we're
# comparing apples to apples; if cold-start fails we'll come back with
# multistart over seeds.
SEED = 3

println("\nUsing seed=$SEED (best M4 XY-drift seed from the M4 seed sweep)")
qcp = petz_isometry_synthesis_problem(
    H_drift, H_drives, drive_bounds;
    n_bdy = n_bdy, n_bulk = n_bulk,
    A_list = A_list,
    T = 25, duration = 10.0,
    Q_petz = 100.0,
    ε_petz = 1e-6,
    seed = SEED,
)

t_start = time()

# --- Phase 1: L-BFGS exploration
println("\n--- Phase 1: L-BFGS (max_iter=200, eval_hessian=false) ---")
t_p1 = time()
solve!(qcp; max_iter=200, eval_hessian=false)
t_p1 = time() - t_p1
V_p1_rolled = rolled_out_isometry(qcp)
obj_p1 = petz_recovery_objective(V_p1_rolled; A_list=A_list)
@printf("Phase 1: rollout Petz obj = %.6e   (wall %.1f s)\n", obj_p1, t_p1)

# --- CHECKPOINT after Phase 1 so a crash mid-Phase-2 doesn't lose the work.
#     The isometry is the cheap, always-savable artifact (analysis only needs
#     V). The pulse save is wrapped defensively so an extraction hiccup can't
#     abort the run. (Earlier bug: get_trajectory wasn't imported into Main,
#     which killed the whole run after Phase 1 — now imported at the top.)
mkpath("data")
save_isometry("data/m5_n5_phase1_isometry", V_p1_rolled;
    meta = Dict(:phase => 1, :rollout_petz => obj_p1, :seed => SEED))
try
    traj = get_trajectory(qcp)
    save_pulse("data/m5_n5_phase1_pulse", traj[:u], get_times(traj);
        meta = Dict(:phase => 1, :rollout_petz => obj_p1, :seed => SEED))
    println("Checkpoint saved: data/m5_n5_phase1_{isometry,pulse}.jls")
catch e
    @warn "Phase-1 pulse checkpoint failed (isometry was still saved)" exception=e
end

# --- Phase 2: exact-Hessian refinement (kept short to bound wall time —
#     each n=5 exact-Hessian iter is ~15-30 min, so 12 iters caps it at a
#     few hours; the Phase-1 checkpoint above is the safety net)
println("\n--- Phase 2: exact Hessian (max_iter=12) ---")
t_p2 = time()
solve!(qcp; max_iter=12)
t_p2 = time() - t_p2

V_opt_nlp = synthesized_isometry(qcp)
V_opt = rolled_out_isometry(qcp)
obj_nlp = try
    petz_recovery_objective(V_opt_nlp; A_list=A_list)
catch
    NaN
end
obj_rolled = petz_recovery_objective(V_opt; A_list=A_list)
@printf("Phase 2: NLP obj = %.6e   rollout obj = %.6e   (wall %.1f s)\n",
        obj_nlp, obj_rolled, t_p2)
@printf("NLP-rollout gap: %+.3e\n", obj_nlp - obj_rolled)

# --- Save the Phase-2 result IMMEDIATELY (before any further analysis that
#     could error), so the converged isometry is never lost.
save_isometry("data/m5_n5_phase2_isometry", V_opt;
    meta = Dict(:phase => 2, :rollout_petz => obj_rolled, :seed => SEED))
println("Phase-2 isometry saved: data/m5_n5_phase2_isometry.jls")

# --- The novel-test question: is V_opt the same code as [[5,1,3]]?
V_513 = five_qubit_isometry()
fid_pc  = subspace_fidelity(V_513, V_opt)        # phase-coherent (logical-basis sensitive)
fid_cs  = code_subspace_fidelity(V_513, V_opt)   # basis-invariant (only the SUBSPACE matters)
@printf("\nIs the discovered code equivalent to the analytic [[5,1,3]]?\n")
@printf("  Phase-coherent fidelity     subspace_fidelity(V_513, V_opt)      = %.6f\n", fid_pc)
@printf("  Basis-invariant fidelity    code_subspace_fidelity(V_513, V_opt) = %.6f\n", fid_cs)

t_elapsed = time() - t_start
println()
println("=" ^ 78)
println("Final result")
println("=" ^ 78)
@printf("Petz objective (rollout V_opt): %.6e\n", obj_rolled)
println()
println("Per-erasure breakdown:")
for A in A_list
    @printf("  keep %s : %.6e\n", A, petz_recovery_error(V_opt, A))
end
println()

# --- Reference points
println("Reference points (all on the same physical system):")
@printf("  M4 fixed-target [[5,1,3]] synthesis: Petz ≤ 1e-3  (rollout fid 0.999)\n")
@printf("  THIS (Piccolo with Petz obj):        Petz = %.4e\n", obj_rolled)
println()

# --- Diagnose the manifold question
if obj_rolled < 1e-3
    if fid_cs > 0.99
        println("=> M5 converged to the [[5,1,3]] code subspace (code_subspace_fidelity > 0.99).")
        println("   Discovery-objective optimization REPRODUCES fixed-target synthesis when")
        println("   the perfect code is reachable. Confirms M5 doesn't degrade compared to M4.")
    else
        println("=> M5 found a DIFFERENT low-Petz code on the [[5,1,3]] manifold")
        println("   (code_subspace_fidelity = $(round(fid_cs, digits=3)) to V_513 but Petz ≈ 0).")
        println("   This is the MANIFOLD OBSERVATION under physical Hamiltonian constraints —")
        println("   the thesis-novel result: discovery-driven QOC finds inequivalent codes")
        println("   with equivalent recovery properties.")
    end
elseif obj_rolled < 0.1
    println("=> Partial success: Petz objective dropped well below random (~0.16) but didn't")
    println("   hit the theoretical floor. Likely needs more iterations or warm-start from")
    println("   M4's [[5,1,3]] pulse.")
else
    println("=> Cold-start stagnated. The 5-qubit landscape is rough (per the LOG.md M4")
    println("   diagnostics); will need multistart or warm-start to escape.")
end

@printf("\nWall time: %.1f s   (Phase 1: %.1f s, Phase 2: %.1f s)\n",
        t_elapsed, t_p1, t_p2)

# --- Save the result so future analysis can reload without re-running
mkpath("data")
save_isometry("data/m5_n5_petz_xy_seed$SEED", V_opt;
    meta = Dict(
        :n_bdy            => n_bdy,
        :n_bulk           => n_bulk,
        :drift            => "nn_xx_yy_drift(5)",
        :drives           => "single_qubit_xy_drives(5)",
        :drive_bound      => 1.0,
        :T                => 25,
        :duration         => 10.0,
        :seed             => SEED,
        :A_list           => A_list,
        :Q_petz           => 100.0,
        :ε_petz           => 1e-6,
        :phase1_iters     => 200,
        :phase2_iters     => 15,
        :petz_objective   => obj_rolled,
        :fid_to_V513_pc   => fid_pc,
        :fid_to_V513_cs   => fid_cs,
        :wall_time_s      => t_elapsed,
        :script           => "examples/05f_piccolo_m5_n5.jl",
        :date             => "2026-05-26",
    ))
println("\nSaved isometry to data/m5_n5_petz_xy_seed$SEED.jls")
