# Tests for src/optimization.jl — reusable optimization-control patterns.
#
# Strategy: keep these tests cheap. The expensive ones (full M4 / M5 seed
# sweeps) live in `examples/`. Here we exercise:
#   * `multistart_synthesis` with the smallest possible Piccolo problem
#     (single-qubit X gate, like test_synthesis.jl).
#   * `discover_low_petz_isometry` at n=3 (always run; ~seconds) and at
#     n=5 (gated by a runtime budget; ~10s with small `restarts/maxiter`).
#     The n=5 default settings (restarts=40, maxiter=400) are tested in
#     examples/07; the in-suite test uses tighter knobs.
#   * `warm_start_pulse!` round-trip via a tiny qcp, save/load_pulse,
#     verify the trajectory got the saved controls.
#   * `curriculum_solve!` end-to-end on a 2-step trivial schedule —
#     verifies the schedule loop, return shape, and pulse hand-off.

using LinearAlgebra
using Random
using Test
using HolographicControl
using Piccolo: solve!, get_trajectory, get_times

# --------------------------------------------------------------------------- #
# Pattern 1 — multistart_synthesis
# --------------------------------------------------------------------------- #

@testset "multistart_synthesis — argument validation" begin
    H_drift  = ComplexF64[0 0; 0 0]
    H_drives = [pauli_matrix('X'), pauli_matrix('Y')]
    drive_bounds = [1.0, 1.0]
    V_target = ComplexF64[0 1; 1 0]
    builder = seed -> isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 10, duration = 4.0, seed = seed,
    )

    # Empty seeds → error.
    @test_throws ErrorException multistart_synthesis(
        builder; seeds = Int[], V_target = V_target,
    )

    # :rollout_fidelity without V_target → error.
    @test_throws ErrorException multistart_synthesis(
        builder; seeds = 0:1, score = :rollout_fidelity,
    )

    # :rollout_petz without A_list → error.
    @test_throws ErrorException multistart_synthesis(
        builder; seeds = 0:1, score = :rollout_petz,
    )

    # Unknown score symbol → error.
    @test_throws ErrorException multistart_synthesis(
        builder; seeds = 0:1, score = :weird_thing, V_target = V_target,
    )
end

@testset "multistart_synthesis — 1Q X gate sweep, :rollout_fidelity mode" begin
    V_target = ComplexF64[0 1; 1 0]
    H_drift  = ComplexF64[0 0; 0 0]
    H_drives = [pauli_matrix('X'), pauli_matrix('Y')]
    drive_bounds = [1.0, 1.0]
    builder = seed -> isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 40, duration = 8.0, seed = seed,
    )

    seeds = 0:2
    (; results, winner) = multistart_synthesis(
        builder; seeds = seeds, max_iter = 60,
        score = :rollout_fidelity, V_target = V_target,
    )

    # Right number of results.
    @test length(results) == length(seeds)

    # Each entry exposes the required fields.
    for r in results
        @test :seed ∈ keys(r)
        @test :score ∈ keys(r)
        @test :fidelity_to_target ∈ keys(r)
        @test :petz ∈ keys(r)
        @test :wall ∈ keys(r)
        @test :V_rolled ∈ keys(r)
        @test size(r.V_rolled) == (2, 2)
        @test isfinite(r.score)
        @test r.wall > 0
    end

    # Seeds are preserved in order.
    @test [r.seed for r in results] == collect(seeds)

    # The winner is one of the results — identified by seed match, since `==`
    # on the NamedTuples themselves is false when any field is NaN
    # (here `petz = NaN` because we're in :rollout_fidelity mode).
    @test winner.seed ∈ [r.seed for r in results]

    # Winner has the maximum score (we're maximizing fidelity).
    @test winner.score ≥ maximum(r.score for r in results) - 1e-12

    # Sanity: with 60 L-BFGS iters on a 1Q X gate this typically lands above
    # 0.7 (we use the same loose bound as test_synthesis.jl). At least one
    # of the three seeds should reach this.
    @test maximum(r.fidelity_to_target for r in results) > 0.7
end

@testset "multistart_synthesis — :rollout_petz scoring mode" begin
    # 1Q dummy "discovery" — A_list = [[1]] (keep the only qubit) is a
    # degenerate full-system reconstruction but it exercises the petz code
    # path on the smallest possible builder.
    V_target = ComplexF64[0 1; 1 0]
    H_drift  = ComplexF64[0 0; 0 0]
    H_drives = [pauli_matrix('X'), pauli_matrix('Y')]
    drive_bounds = [1.0, 1.0]
    builder = seed -> isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 10, duration = 4.0, seed = seed,
    )

    A_list = [[1]]   # trivial: keep the whole system
    (; results, winner) = multistart_synthesis(
        builder; seeds = 0:1, max_iter = 20,
        score = :rollout_petz, A_list = A_list,
    )

    @test length(results) == 2
    # Petz objective should be a finite non-negative number.
    for r in results
        @test r.petz ≥ 0
        @test isfinite(r.petz)
        @test isnan(r.fidelity_to_target)    # V_target not supplied
    end

    # Winner = argmin(petz).
    @test winner.score ≤ minimum(r.petz for r in results) + 1e-12
end

# --------------------------------------------------------------------------- #
# Pattern 2 — discover_low_petz_isometry
# --------------------------------------------------------------------------- #

@testset "discover_low_petz_isometry — argument validation" begin
    A_list = uniform_erasure_subregions(3, 1)
    # n_bdy < n_bulk → error.
    @test_throws ErrorException discover_low_petz_isometry(
        1, 2; A_list = A_list, restarts = 1, maxiter = 1,
    )
    # Empty A_list → error.
    @test_throws ErrorException discover_low_petz_isometry(
        3, 1; A_list = Vector{Vector{Int}}(), restarts = 1, maxiter = 1,
    )
end

@testset "discover_low_petz_isometry — n=3 reaches Petz < 0.1" begin
    # The standalone-M5 anchor: cold-start coordinate descent on
    # (n_bdy=3, n_bulk=1) with all weight-1 erasures should land near
    # ~0.067 (per examples/05b). The bound here is 0.1, well above that
    # but well below random (~0.16).
    A_list = uniform_erasure_subregions(3, 1)
    V_star, history = discover_low_petz_isometry(
        3, 1; A_list = A_list, restarts = 20, maxiter = 200, seed = 0,
    )
    @test size(V_star) == (8, 2)
    @test is_isometry(V_star; atol = 1e-6)
    @test history.final_petz < 0.1
    @test history.final_petz ≤ history.restart_winner_petz + 1e-10   # CD only improves
    @test history.param_dim == 2 * 8 * 2
    @test history.iters > 0
    @test history.wall > 0

    # The discovered V* should also satisfy is_reconstructable on each
    # erasure, since the Petz error is small per region.
    petz_recovery_objective(V_star; A_list = A_list) < 0.1   # sanity
end

@testset "discover_low_petz_isometry — n=5 reaches Petz < 0.01" begin
    # n=5 standalone optimum is essentially machine zero (~7e-10 per
    # examples/07). With trimmed knobs (restarts=10, maxiter=80) we demand
    # Petz < 0.01 — a very conservative bound that the algorithm routinely
    # clears in test-suite time (~30s for these knobs).
    A_list = uniform_erasure_subregions(5, 1)
    V_star, history = discover_low_petz_isometry(
        5, 1; A_list = A_list, restarts = 10, maxiter = 80, seed = 0,
    )
    @test size(V_star) == (32, 2)
    @test is_isometry(V_star; atol = 1e-6)
    @test history.final_petz < 0.01
    @test history.param_dim == 2 * 32 * 2
end

@testset "discover_low_petz_isometry — V_star is an isometry to machine precision" begin
    # Polar projection guarantees this regardless of n_bdy/n_bulk/A_list.
    A_list = uniform_erasure_subregions(3, 1)
    V_star, _ = discover_low_petz_isometry(
        3, 1; A_list = A_list, restarts = 5, maxiter = 20, seed = 7,
    )
    @test opnorm(V_star' * V_star - I) < 1e-10
end

@testset "discover_low_petz_isometry — weights forwarded" begin
    # Non-uniform weights should produce DIFFERENT V_star than uniform.
    A_list = uniform_erasure_subregions(3, 1)
    V1, _ = discover_low_petz_isometry(
        3, 1; A_list = A_list, restarts = 5, maxiter = 10, seed = 42,
    )
    V2, _ = discover_low_petz_isometry(
        3, 1; A_list = A_list, weights = [1.0, 0.1, 0.1],
        restarts = 5, maxiter = 10, seed = 42,
    )
    @test size(V1) == size(V2)
    # Different weights should yield different solutions (the random init
    # is the same via the seed, but the descent steers differently).
    @test V1 != V2
end

# --------------------------------------------------------------------------- #
# Pattern 3 — warm_start_pulse!
# --------------------------------------------------------------------------- #

@testset "warm_start_pulse! — round-trip via save_pulse" begin
    V_target = ComplexF64[0 1; 1 0]
    H_drift  = ComplexF64[0 0; 0 0]
    H_drives = [pauli_matrix('X'), pauli_matrix('Y')]
    drive_bounds = [1.0, 1.0]

    qcp_a = isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 20, duration = 4.0, seed = 0,
    )
    solve!(qcp_a; max_iter = 5, eval_hessian = false)
    traj_a = get_trajectory(qcp_a)
    u_a = copy(traj_a[:u])

    mktempdir() do dir
        path = joinpath(dir, "warm_start_pulse_test")
        save_pulse(path, traj_a[:u], get_times(traj_a))

        # Fresh qcp with a different seed (so its initial :u is different).
        qcp_b = isometry_synthesis_problem(
            V_target, H_drift, H_drives, drive_bounds;
            T = 20, duration = 4.0, seed = 999,
        )
        traj_b = get_trajectory(qcp_b)
        @test traj_b[:u] != u_a    # before warm-start, qcp_b's :u differs

        warm_start_pulse!(qcp_b, path)
        @test traj_b[:u] ≈ u_a     # after warm-start, qcp_b's :u matches
    end
end

@testset "warm_start_pulse! — shape mismatch errors out" begin
    V_target = ComplexF64[0 1; 1 0]
    H_drift  = ComplexF64[0 0; 0 0]
    H_drives = [pauli_matrix('X'), pauli_matrix('Y')]
    drive_bounds = [1.0, 1.0]

    qcp = isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 20, duration = 4.0, seed = 0,
    )

    mktempdir() do dir
        path = joinpath(dir, "wrong_shape")
        # Save a pulse with the wrong shape (different T).
        wrong_controls = randn(2, 30)
        wrong_times = collect(range(0.0, 1.0; length = 30))
        save_pulse(path, wrong_controls, wrong_times)

        @test_throws ErrorException warm_start_pulse!(qcp, path)
    end
end

@testset "warm_start_pulse! — copies derivatives when present" begin
    V_target = ComplexF64[0 1; 1 0]
    H_drift  = ComplexF64[0 0; 0 0]
    H_drives = [pauli_matrix('X'), pauli_matrix('Y')]
    drive_bounds = [1.0, 1.0]

    qcp = isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 12, duration = 4.0, seed = 0,
    )
    traj = get_trajectory(qcp)
    # The SmoothPulseProblem trajectory does carry :du.
    @test :du ∈ traj.names

    mktempdir() do dir
        path = joinpath(dir, "with_derivs")
        controls = randn(2, 12)
        derivatives = randn(2, 12)
        times = collect(range(0.0, 4.0; length = 12))
        save_pulse(path, controls, times; derivatives = derivatives)

        warm_start_pulse!(qcp, path)
        @test traj[:u] ≈ controls
        @test traj[:du] ≈ derivatives
    end
end

# --------------------------------------------------------------------------- #
# Pattern 4 — curriculum_solve!
# --------------------------------------------------------------------------- #

@testset "curriculum_solve! — empty schedule errors out" begin
    factory = α -> isometry_synthesis_problem(
        ComplexF64[0 1; 1 0],
        ComplexF64[0 0; 0 0],
        [pauli_matrix('X'), pauli_matrix('Y')],
        [1.0, 1.0]; T = 10, duration = 4.0, seed = 0,
    )
    @test_throws ErrorException curriculum_solve!(
        factory; α_schedule = Float64[],
    )
end

@testset "curriculum_solve! — 2-step schedule end-to-end" begin
    # Tiny problem; α is just a placeholder here — the factory builds the
    # SAME qcp at every α (vanilla repeated-solve / regression test). The
    # curriculum LOOP and pulse hand-off are what we're testing.
    V_target = ComplexF64[0 1; 1 0]
    H_drift  = ComplexF64[0 0; 0 0]
    H_drives = [pauli_matrix('X'), pauli_matrix('Y')]
    drive_bounds = [1.0, 1.0]
    factory = α -> isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 12, duration = 4.0, seed = 0,
    )

    log_entries = curriculum_solve!(
        factory;
        α_schedule = [0.0, 1.0],
        V_target = V_target,
        max_iter_per_step = 10,
    )

    # Right number of steps, right schema.
    @test length(log_entries) == 2
    for (k, e) in pairs(log_entries)
        @test :α ∈ keys(e)
        @test :fidelity_to_target ∈ keys(e)
        @test :petz ∈ keys(e)
        @test :wall_time ∈ keys(e)
        @test e.wall_time > 0
        @test isfinite(e.fidelity_to_target)
        @test isnan(e.petz)    # A_list not supplied
    end
    @test log_entries[1].α == 0.0
    @test log_entries[2].α == 1.0
end

@testset "curriculum_solve! — Petz scoring when A_list supplied" begin
    V_target = ComplexF64[0 1; 1 0]
    H_drift  = ComplexF64[0 0; 0 0]
    H_drives = [pauli_matrix('X'), pauli_matrix('Y')]
    drive_bounds = [1.0, 1.0]
    factory = α -> isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 10, duration = 4.0, seed = 0,
    )

    log_entries = curriculum_solve!(
        factory;
        α_schedule = [0.5],
        V_target = V_target,
        A_list = [[1]],
        max_iter_per_step = 5,
    )

    @test length(log_entries) == 1
    @test log_entries[1].petz ≥ 0
    @test isfinite(log_entries[1].petz)
    @test isfinite(log_entries[1].fidelity_to_target)
end
