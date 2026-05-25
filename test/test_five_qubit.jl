using LinearAlgebra
using Test
using HolographicControl

@testset "[[5,1,3]] reference code" begin
    V = five_qubit_isometry()

    @testset "shape and isometry" begin
        @test size(V) == (32, 2)
        @test V' * V ≈ I(2) atol=1e-12
    end

    @testset "code projector" begin
        P = V * V'
        @test real(tr(P)) ≈ 2 atol=1e-12
        @test P ≈ P' atol=1e-12          # Hermitian
        @test P * P ≈ P  atol=1e-12      # idempotent
    end

    @testset "Knill–Laflamme exact" begin
        kl = knill_laflamme_constants(V; n=5)
        # The [[5,1,3]] code is non-degenerate distance-3: C must be I_16.
        @test size(kl.C) == (16, 16)
        @test maximum(kl.residuals) < 1e-10
        @test kl.C ≈ I(16) atol=1e-10
    end

    @testset "stabilizer eigenvalues" begin
        # Each generator g_i fixes the code subspace: g_i V = V (eigenvalue +1).
        for g in five_qubit_stabilizers()
            @test g * V ≈ V atol=1e-12
        end
    end
end
