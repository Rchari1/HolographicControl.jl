# Test the Piccolo problem builders for structural correctness (no actual
# Piccolo solves — see test_synthesis.jl for a small end-to-end test).
# Verifies that the builders accept valid input, reject invalid input, and
# produce QuantumControlProblem objects with the expected internal shape.

using LinearAlgebra
using Test
using HolographicControl
using Piccolo: QuantumControlProblem, get_trajectory

@testset "isometry_synthesis_problem — input validation" begin
    H_drift = nn_xx_yy_drift(3)
    H_drives = single_qubit_xy_drives(3)
    drive_bounds = fill(1.0, length(H_drives))

    # Non-power-of-2 boundary dim
    V_bad_bdy = randn(ComplexF64, 6, 2)
    @test_throws ErrorException isometry_synthesis_problem(
        V_bad_bdy, H_drift, H_drives, drive_bounds)

    # Non-power-of-2 bulk dim
    V_bad_bulk = randn(ComplexF64, 8, 3)
    @test_throws ErrorException isometry_synthesis_problem(
        V_bad_bulk, H_drift, H_drives, drive_bounds)

    # Not an isometry
    V_not_iso = ComplexF64[1 1; 0 0; 0 0; 0 0; 0 0; 0 0; 0 0; 0 0]
    @test_throws ErrorException isometry_synthesis_problem(
        V_not_iso, H_drift, H_drives, drive_bounds)

    # drive_bounds / H_drives length mismatch
    V = three_qubit_repetition_isometry()
    @test_throws ErrorException isometry_synthesis_problem(
        V, H_drift, H_drives, fill(1.0, length(H_drives) + 1))
end

@testset "isometry_synthesis_problem — builds for 3Q repetition" begin
    V_target = three_qubit_repetition_isometry()
    H_drift  = nn_xx_yy_drift(3)
    H_drives = single_qubit_xy_drives(3)
    drive_bounds = fill(1.0, length(H_drives))

    qcp = isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 20, duration = 5.0, seed = 0,
    )

    @test qcp isa QuantumControlProblem

    # The trajectory should expose ψ̃1 and ψ̃2 (the two bulk-basis kets)
    traj = get_trajectory(qcp)
    @test :ψ̃1 ∈ traj.names
    @test :ψ̃2 ∈ traj.names
    @test :u ∈ traj.names

    # qcp.qtraj is the MultiKetTrajectory; should carry the right initials/goals
    @test length(qcp.qtraj.initials) == 2
    @test length(qcp.qtraj.goals) == 2
end

@testset "isometry_synthesis_problem — builds for [[5,1,3]]" begin
    V_target = five_qubit_isometry()
    H_drift  = nn_xx_yy_drift(5)
    H_drives = single_qubit_xy_drives(5)
    drive_bounds = fill(1.0, length(H_drives))

    qcp = isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 25, duration = 10.0, seed = 0,
    )

    @test qcp isa QuantumControlProblem
    @test length(qcp.qtraj.initials) == 2
    # Each input state should live in 32-dim Hilbert space
    @test length(qcp.qtraj.initials[1]) == 32
end

# ============================================================================ #
# petz_isometry_synthesis_problem — M5 Layer 3 problem builder
# ============================================================================ #

@testset "petz_isometry_synthesis_problem — input validation" begin
    H_drift = nn_xx_yy_drift(3)
    H_drives = single_qubit_xy_drives(3)
    drive_bounds = fill(1.0, length(H_drives))
    A_list = uniform_erasure_subregions(3, 1)

    # H_drift shape mismatch
    @test_throws ErrorException petz_isometry_synthesis_problem(
        randn(ComplexF64, 4, 4), H_drives, drive_bounds;
        n_bdy = 3, n_bulk = 1, A_list = A_list)

    # length mismatch on drive_bounds
    @test_throws ErrorException petz_isometry_synthesis_problem(
        H_drift, H_drives, fill(1.0, length(H_drives) + 1);
        n_bdy = 3, n_bulk = 1, A_list = A_list)

    # empty A_list
    @test_throws ErrorException petz_isometry_synthesis_problem(
        H_drift, H_drives, drive_bounds;
        n_bdy = 3, n_bulk = 1, A_list = Vector{Vector{Int}}())

    # weights length mismatch
    @test_throws ErrorException petz_isometry_synthesis_problem(
        H_drift, H_drives, drive_bounds;
        n_bdy = 3, n_bulk = 1, A_list = A_list, weights = [1.0])
end

@testset "petz_isometry_synthesis_problem — builds for 3Q system" begin
    H_drift = nn_xx_yy_drift(3)
    H_drives = single_qubit_xy_drives(3)
    drive_bounds = fill(1.0, length(H_drives))
    A_list = uniform_erasure_subregions(3, 1)

    qcp = petz_isometry_synthesis_problem(
        H_drift, H_drives, drive_bounds;
        n_bdy = 3, n_bulk = 1,
        A_list = A_list,
        T = 20, duration = 5.0,
        Q_petz = 100.0, ε_petz = 1e-6,
        seed = 0,
    )

    @test qcp isa QuantumControlProblem

    # The trajectory should expose ψ̃1 (and only one because n_bulk=1)
    traj = get_trajectory(qcp)
    @test :ψ̃1 ∈ traj.names
    @test :u ∈ traj.names

    # qcp.qtraj should be a MultiKetTrajectory with d_bulk = 2^1 = 2
    @test length(qcp.qtraj.initials) == 2
    @test length(qcp.qtraj.goals) == 2
    @test length(qcp.qtraj.initials[1]) == 8     # 2^n_bdy
end
