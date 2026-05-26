using LinearAlgebra
using Test
using HolographicControl

@testset "pauli_matrix" begin
    @test pauli_matrix('I') == ComplexF64[1 0; 0 1]
    @test pauli_matrix('X') == ComplexF64[0 1; 1 0]
    @test pauli_matrix('Y') == ComplexF64[0 -im; im 0]
    @test pauli_matrix('Z') == ComplexF64[1 0; 0 -1]
    @test_throws ErrorException pauli_matrix('Q')
end

@testset "pauli_matrix algebraic identities" begin
    X = pauli_matrix('X')
    Y = pauli_matrix('Y')
    Z = pauli_matrix('Z')

    # P_i² = I
    @test X * X ≈ I(2)
    @test Y * Y ≈ I(2)
    @test Z * Z ≈ I(2)

    # Anticommutators: {P_i, P_j} = 0 for i ≠ j
    @test X * Y + Y * X ≈ zeros(2, 2) atol=1e-15
    @test Y * Z + Z * Y ≈ zeros(2, 2) atol=1e-15
    @test Z * X + X * Z ≈ zeros(2, 2) atol=1e-15

    # XY = iZ, YZ = iX, ZX = iY
    @test X * Y ≈ im * Z
    @test Y * Z ≈ im * X
    @test Z * X ≈ im * Y
end

@testset "pauli_string — single-qubit cases" begin
    @test pauli_string("I") == pauli_matrix('I')
    @test pauli_string("X") == pauli_matrix('X')
    @test pauli_string("Y") == pauli_matrix('Y')
    @test pauli_string("Z") == pauli_matrix('Z')
end

@testset "pauli_string — multi-qubit big-endian" begin
    X = pauli_matrix('X')
    I2 = pauli_matrix('I')
    # "XI" means X on qubit 1, I on qubit 2 → kron(X, I) under big-endian
    @test pauli_string("XI") == kron(X, I2)
    @test pauli_string("IX") == kron(I2, X)

    # 3-qubit cases
    Z = pauli_matrix('Z')
    @test pauli_string("XZI") == kron(X, kron(Z, I2))
    @test pauli_string("IZX") == kron(I2, kron(Z, X))
end

@testset "pauli_string — Hermitian and unitary" begin
    for s in ("X", "XY", "XZZXI", "ZXIXZ", "YYYY")
        P = pauli_string(s)
        @test P ≈ P'         # Hermitian
        @test P * P ≈ I      # unitary and squares to I (single Paulis)
    end
end

@testset "pauli_string — empty string errors" begin
    @test_throws ErrorException pauli_string("")
end

@testset "single_qubit_pauli_errors" begin
    # n=1: identity + 3 single-site Paulis
    e1 = single_qubit_pauli_errors(1)
    @test length(e1) == 4
    @test first(e1)[1] == "I"
    labels1 = [t[1] for t in e1]
    @test "X1" ∈ labels1
    @test "Y1" ∈ labels1
    @test "Z1" ∈ labels1

    # n=5: identity + 15 single-site Paulis (3 × 5)
    e5 = single_qubit_pauli_errors(5)
    @test length(e5) == 16
    @test first(e5)[1] == "I"
    @test first(e5)[2] == I(32)

    # Each error is a 32×32 matrix
    for (lbl, op) in e5
        @test size(op) == (32, 32)
        @test op ≈ op'      # Hermitian
    end

    # Site-major ordering: X1, Y1, Z1, X2, Y2, Z2, ...
    @test e5[2][1] == "X1"
    @test e5[3][1] == "Y1"
    @test e5[4][1] == "Z1"
    @test e5[5][1] == "X2"
    @test e5[end][1] == "Z5"
end

@testset "single_qubit_pauli_errors invariants" begin
    n = 3
    errors = single_qubit_pauli_errors(n)
    d = 2^n
    @test length(errors) == 3n + 1

    # Identity is exactly I_d
    @test errors[1][2] == I(d)

    # Each single-site Pauli E satisfies E² = I, tr(E) = 0
    for (lbl, E) in errors[2:end]
        @test E * E ≈ I(d) atol=1e-12
        @test abs(tr(E)) < 1e-12
    end
end
