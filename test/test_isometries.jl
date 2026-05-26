using LinearAlgebra
using Random
using Test
using HolographicControl

@testset "is_isometry / check_isometry" begin
    # Built isometries pass
    @test is_isometry(five_qubit_isometry())
    @test is_isometry(three_qubit_repetition_isometry())

    # Identity column-stacks are isometries
    @test is_isometry(Matrix{ComplexF64}(I, 8, 2))

    # Non-isometry: columns not orthonormal
    bad = ComplexF64[1 1; 0 0; 0 0]
    @test !is_isometry(bad)
    @test_throws ErrorException check_isometry(bad)

    # check_isometry returns nothing on success
    @test check_isometry(five_qubit_isometry()) === nothing
end

@testset "polar_isometry projects to Stiefel manifold" begin
    Random.seed!(7)
    for (d_bdy, d_bulk) in ((4, 2), (8, 1), (8, 2), (16, 4))
        M = randn(ComplexF64, d_bdy, d_bulk)
        V = polar_isometry(M)
        @test size(V) == (d_bdy, d_bulk)
        @test is_isometry(V; atol=1e-10)
    end

    # Already-isometric input is preserved (up to numerics)
    V0 = random_isometry(8, 2; rng = MersenneTwister(11))
    V = polar_isometry(V0)
    @test V ≈ V0 atol=1e-9
end

@testset "random_isometry" begin
    # Shape + isometry property
    for _ in 1:5
        V = random_isometry(8, 2; rng = MersenneTwister(rand(1:10000)))
        @test size(V) == (8, 2)
        @test is_isometry(V)
    end

    # d_bulk > d_bdy is an error
    @test_throws ErrorException random_isometry(2, 4)

    # Invariant: tr(V V') = d_bulk for every isometry V (since V'V = I_{d_bulk}
    # implies tr(V V') = tr(V' V) = d_bulk by cyclic permutation).
    rng = MersenneTwister(123)
    Vs = [random_isometry(8, 2; rng) for _ in 1:30]
    @test all(V -> real(tr(V * V')) ≈ 2.0, Vs)
end

@testset "encoding_input_states" begin
    # 3-qubit boundary, 1-qubit bulk: 2 input states
    V_target = three_qubit_repetition_isometry()
    inputs = encoding_input_states(V_target)
    @test length(inputs) == 2
    # |0⟩_bulk ⊗ |00⟩_anc = |000⟩ → big-endian index 1
    @test inputs[1] == ComplexF64[1, 0, 0, 0, 0, 0, 0, 0]
    # |1⟩_bulk ⊗ |00⟩_anc = |100⟩ → big-endian index 5
    @test inputs[2] == ComplexF64[0, 0, 0, 0, 1, 0, 0, 0]

    # 5-qubit boundary, 1-qubit bulk: 2 input states, big-endian
    V5 = five_qubit_isometry()
    inputs5 = encoding_input_states(V5)
    @test length(inputs5) == 2
    @test length(inputs5[1]) == 32
    # |0⟩_bulk → |00000⟩ = index 1
    @test argmax(abs.(inputs5[1])) == 1
    # |1⟩_bulk → |10000⟩ = index 17 (under big-endian, q1=1 contributes 2^4)
    @test argmax(abs.(inputs5[2])) == 17

    # Non-power-of-2 dimensions are rejected
    @test_throws ErrorException encoding_input_states(randn(ComplexF64, 6, 2))
    @test_throws ErrorException encoding_input_states(randn(ComplexF64, 8, 3))
end

@testset "subspace_fidelity — basic identities" begin
    V = five_qubit_isometry()

    # Fidelity to self is 1
    @test subspace_fidelity(V, V) ≈ 1.0

    # Global phase doesn't matter
    @test subspace_fidelity(V, exp(im * 0.7) * V) ≈ 1.0

    # Logical X (swap columns) drops fidelity
    V_swap = hcat(V[:, 2], V[:, 1])
    @test subspace_fidelity(V, V_swap) < 1e-10

    # Shape mismatch errors
    @test_throws DimensionMismatch subspace_fidelity(V, V[:, 1:1])
end

@testset "code_subspace_fidelity — basis-invariant" begin
    V = five_qubit_isometry()

    # Self: 1
    @test code_subspace_fidelity(V, V) ≈ 1.0

    # Logical unitary on V leaves the SUBSPACE unchanged → code fidelity 1
    U_log = ComplexF64[1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)]   # Hadamard
    V_rot = V * U_log
    @test code_subspace_fidelity(V, V_rot) ≈ 1.0 atol=1e-12
    # ...but subspace_fidelity drops because the column-by-column overlap changes
    @test subspace_fidelity(V, V_rot) < 1.0

    # Orthogonal subspaces give 0
    n = 32
    e1 = zeros(ComplexF64, n); e1[1] = 1
    e2 = zeros(ComplexF64, n); e2[2] = 1
    e3 = zeros(ComplexF64, n); e3[3] = 1
    e4 = zeros(ComplexF64, n); e4[4] = 1
    V_a = hcat(e1, e2)
    V_b = hcat(e3, e4)
    @test code_subspace_fidelity(V_a, V_b) ≈ 0.0 atol=1e-12

    # Shape mismatch errors
    @test_throws DimensionMismatch code_subspace_fidelity(V, V[:, 1:1])
end

@testset "code_subspace_fidelity ≥ subspace_fidelity" begin
    # For any pair of equal-shape isometries, the basis-invariant code
    # fidelity is an upper bound on the column-phase-coherent one.
    Random.seed!(31)
    for _ in 1:5
        V1 = random_isometry(8, 2)
        V2 = random_isometry(8, 2)
        @test code_subspace_fidelity(V1, V2) ≥ subspace_fidelity(V1, V2) - 1e-12
    end
end
