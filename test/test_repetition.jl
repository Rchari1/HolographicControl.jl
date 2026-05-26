using LinearAlgebra
using Test
using HolographicControl

@testset "three_qubit_repetition_isometry — shape and isometry" begin
    V = three_qubit_repetition_isometry()
    @test size(V) == (8, 2)
    @test is_isometry(V)
end

@testset "three_qubit_repetition_isometry — |0⟩_L=|000⟩, |1⟩_L=|111⟩" begin
    V = three_qubit_repetition_isometry()
    # Column 1 = |000⟩ → big-endian index 1
    @test V[:, 1] == ComplexF64[1, 0, 0, 0, 0, 0, 0, 0]
    # Column 2 = |111⟩ → big-endian index 8
    @test V[:, 2] == ComplexF64[0, 0, 0, 0, 0, 0, 0, 1]
end

@testset "repetition encoder is the M3 target" begin
    # The M3 synthesis aims to produce exactly this matrix; we verify here
    # that the analytic target is well-formed.
    V = three_qubit_repetition_isometry()
    @test eltype(V) == ComplexF64
    @test V' * V == ComplexF64[1 0; 0 1]
end
