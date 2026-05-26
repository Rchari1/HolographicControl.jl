using LinearAlgebra
using Random
using Test
using HolographicControl

@testset "petz_recovery_objective — vanishes on [[5,1,3]]" begin
    V = five_qubit_isometry()
    A_list = uniform_erasure_subregions(5, 1)
    @test petz_recovery_objective(V; A_list=A_list) < 1e-10
end

@testset "petz_recovery_objective — uniform weight default" begin
    V = five_qubit_isometry()
    A_list = uniform_erasure_subregions(5, 1)
    # Each single-site erasure gives error ~ 0; the average is also ~0
    obj = petz_recovery_objective(V; A_list=A_list)
    @test obj ≥ 0
    @test obj < 1e-10
end

@testset "petz_recovery_objective — weight normalization" begin
    # Random isometry on a small system; verify weights normalize to sum 1
    Random.seed!(11)
    V = polar_isometry(randn(ComplexF64, 8, 2))
    A_list = uniform_erasure_subregions(3, 1)
    w1 = ones(length(A_list))
    w2 = 2 * ones(length(A_list))   # different scale, same shape
    @test petz_recovery_objective(V; A_list=A_list, weights=w1) ≈
          petz_recovery_objective(V; A_list=A_list, weights=w2) atol=1e-12
end

@testset "petz_recovery_objective — input validation" begin
    V = five_qubit_isometry()
    A_list = uniform_erasure_subregions(5, 1)

    # empty A_list
    @test_throws ErrorException petz_recovery_objective(V; A_list=Vector{Vector{Int}}())

    # weight length mismatch
    @test_throws ErrorException petz_recovery_objective(V; A_list=A_list, weights=[1.0])

    # negative weight
    @test_throws ErrorException petz_recovery_objective(
        V; A_list=A_list, weights=[-1.0, 1.0, 1.0, 1.0, 1.0])

    # zero-sum weights
    @test_throws ErrorException petz_recovery_objective(
        V; A_list=A_list, weights=zeros(length(A_list)))
end

@testset "uniform_erasure_subregions — counts" begin
    # C(n, n-w) total subregions per weight class
    for n in (3, 5, 7), w in 0:n
        subs = uniform_erasure_subregions(n, w)
        @test length(subs) == binomial(n, n - w)
    end
end

@testset "uniform_erasure_subregions — content" begin
    # weight=1, n=3: keep 2 of 3 qubits → the set {[1,2], [1,3], [2,3]}.
    # (Order is lex; we test as a Set so the test is order-agnostic.)
    subs = uniform_erasure_subregions(3, 1)
    @test length(subs) == 3
    @test Set(subs) == Set([[1, 2], [1, 3], [2, 3]])

    # weight=2, n=4: keep 2 of 4 → 6 subregions
    subs = uniform_erasure_subregions(4, 2)
    @test length(subs) == 6
    @test all(length(A) == 2 for A in subs)

    # weight=0: keep everyone (only one subregion = all qubits)
    @test uniform_erasure_subregions(5, 0) == [collect(1:5)]
end

@testset "uniform_erasure_subregions — input validation" begin
    @test_throws ErrorException uniform_erasure_subregions(3, -1)
    @test_throws ErrorException uniform_erasure_subregions(3, 4)
end

@testset "erasure_subregions_up_to" begin
    # n=3, up to weight=2: all weight-1 (3) + weight-2 (3) = 6 subregions
    subs = erasure_subregions_up_to(3, 2)
    @test length(subs) == 6
    # First 3 are weight-1 (size 2), next 3 are weight-2 (size 1)
    @test all(length(A) == 2 for A in subs[1:3])
    @test all(length(A) == 1 for A in subs[4:6])
end

@testset "Petz objective distinguishes good vs bad codes" begin
    # On n=3 qubits with single-qubit erasure:
    A_list = uniform_erasure_subregions(3, 1)

    # The repetition code's Petz objective = 0.5 (dephasing channel)
    V_rep = three_qubit_repetition_isometry()
    obj_rep = petz_recovery_objective(V_rep; A_list=A_list)
    @test isapprox(obj_rep, 0.5; atol=1e-10)

    # The trivial embedding |q, 0, 0⟩ gives error = 0.25 (depolarizing on
    # the keep-{2,3} subregion only)
    V_triv = zeros(ComplexF64, 8, 2)
    V_triv[1, 1] = 1   # |000⟩
    V_triv[5, 2] = 1   # |100⟩
    obj_triv = petz_recovery_objective(V_triv; A_list=A_list)
    @test isapprox(obj_triv, 0.25; atol=1e-10)
end
