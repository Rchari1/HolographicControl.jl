using LinearAlgebra
using Test
using HolographicControl

@testset "single_pentagon_isometry aliases five_qubit_isometry" begin
    V_pentagon = single_pentagon_isometry()
    V_513 = five_qubit_isometry()
    @test V_pentagon == V_513
end

@testset "single pentagon is the [[5,1,3]] (verifies HANDOFF §2.3 claim)" begin
    V = single_pentagon_isometry()
    @test size(V) == (32, 2)
    @test is_isometry(V)
    # All four [[5,1,3]] stabilizers must fix the pentagon code subspace
    for g in five_qubit_stabilizers()
        @test g * V ≈ V atol=1e-12
    end
end

@testset "two_pentagon_isometry — documented stub" begin
    # Multi-pentagon HaPPY is left as future work (see docstring + LOG.md).
    # The stub raises an informative error rather than silently returning
    # something wrong.
    @test_throws ErrorException two_pentagon_isometry()
end
