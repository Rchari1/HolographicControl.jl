using LinearAlgebra
using Test
using HolographicControl

# Small Pauli helpers needed only inside this test file (matching the
# big-endian convention in src/HolographicControl.jl).
const σ0 = ComplexF64[1 0; 0 1]
const σ1 = ComplexF64[0 1; 1 0]    # X
const σ3 = ComplexF64[1 0; 0 -1]   # Z

ket0 = ComplexF64[1.0, 0.0]
ket1 = ComplexF64[0.0, 1.0]
ket_plus  = (ket0 + ket1) / sqrt(2)

@testset "partial_trace — analytic 2-qubit cases" begin
    # |00⟩⟨00|, trace out qubit 2 → |0⟩⟨0| on qubit 1
    ρ = kron(ket0, ket0) * kron(ket0, ket0)'
    @test partial_trace(ρ, [1], 2) ≈ ket0 * ket0' atol=1e-12
    @test partial_trace(ρ, [2], 2) ≈ ket0 * ket0' atol=1e-12

    # Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2: tracing either qubit gives I/2
    bell = (kron(ket0, ket0) + kron(ket1, ket1)) / sqrt(2)
    ρ_bell = bell * bell'
    @test partial_trace(ρ_bell, [1], 2) ≈ I(2) / 2 atol=1e-12
    @test partial_trace(ρ_bell, [2], 2) ≈ I(2) / 2 atol=1e-12

    # Product state |0⟩ ⊗ |+⟩: each qubit's reduced state is the pure marginal
    ρ_prod = kron(ket0 * ket0', ket_plus * ket_plus')
    @test partial_trace(ρ_prod, [1], 2) ≈ ket0 * ket0' atol=1e-12
    @test partial_trace(ρ_prod, [2], 2) ≈ ket_plus * ket_plus' atol=1e-12

    # tr(ρ) is preserved under any partial trace
    ρ_rand = randn(ComplexF64, 4, 4); ρ_rand = ρ_rand + ρ_rand'
    @test tr(partial_trace(ρ_rand, [1], 2)) ≈ tr(ρ_rand) atol=1e-12
    @test tr(partial_trace(ρ_rand, [2], 2)) ≈ tr(ρ_rand) atol=1e-12
end

@testset "embed_operator is HS-adjoint of partial_trace" begin
    # ⟨X, Tr_Ā(Y)⟩ = ⟨embed_operator(X, A, n), Y⟩  for random X, Y
    n = 3
    for A in ([1], [2], [3], [1,2], [1,3], [2,3])
        d_A = 2^length(A)
        d_full = 2^n
        X = randn(ComplexF64, d_A, d_A); X = X + X'
        Y = randn(ComplexF64, d_full, d_full); Y = Y + Y'
        lhs = tr(X' * partial_trace(Y, A, n))
        rhs = tr(embed_operator(X, A, n)' * Y)
        @test lhs ≈ rhs atol=1e-12
    end

    # embed_operator(I_A, A, n) = I_full
    @test embed_operator(I(4), [1,2], 3) ≈ I(8) atol=1e-12
    @test embed_operator(I(4), [2,3], 3) ≈ I(8) atol=1e-12
    @test embed_operator(I(4), [1,3], 3) ≈ I(8) atol=1e-12
end

@testset "matrix_(inv_)sqrt — eigenvalue cutoff" begin
    # Diagonal positive matrix: sqrt is elementwise
    D = Diagonal([4.0, 1.0, 0.25])
    @test matrix_sqrt(Matrix(D)) ≈ Diagonal([2.0, 1.0, 0.5]) atol=1e-12
    @test matrix_inv_sqrt(Matrix(D)) ≈ Diagonal([0.5, 1.0, 2.0]) atol=1e-12

    # Rank-deficient: eigenvalue below cutoff → pseudoinverse drops it
    D2 = Diagonal([4.0, 1e-14, 1.0])
    inv_half = matrix_inv_sqrt(Matrix(D2); cutoff=1e-10)
    @test inv_half ≈ Diagonal([0.5, 0.0, 1.0]) atol=1e-10

    # Round-trip on a Hermitian PSD: (M^{1/2})^2 ≈ M
    A = randn(ComplexF64, 4, 4)
    M = A * A' + 0.1 * I  # positive definite
    Mh = matrix_sqrt(M)
    @test Mh * Mh ≈ M atol=1e-10
end

@testset "Petz recovery on [[5,1,3]]" begin
    V = five_qubit_isometry()
    d_bulk = size(V, 2)

    # Distance-3 code corrects up to d-1 = 2 erasures (location known).
    # The [[5,1,3]] saturates this bound (it is the n=5 AME state).

    # 0 erasures (trivial)
    @test petz_recovery_error(V, [1,2,3,4,5]) < 1e-10

    # 1 erasure: every 4-qubit subregion is correctable
    for q in 1:5
        A = collect(setdiff(1:5, [q]))
        @test petz_recovery_error(V, A) < 1e-10
    end

    # 2 erasures: every 3-qubit subregion is correctable (AME property)
    for a in 1:5, b in (a+1):5
        A = collect(setdiff(1:5, [a, b]))
        @test petz_recovery_error(V, A) < 1e-10
    end

    # 3 erasures: every 2-qubit subregion fails. The recovered channel is
    # fully depolarizing on the bulk, so error = 1 - 1/d_bulk².
    depolarizing_err = 1 - 1 / d_bulk^2
    for a in 1:5, b in (a+1):5
        A = [a, b]
        @test petz_recovery_error(V, A) ≈ depolarizing_err atol=1e-10
    end
end
