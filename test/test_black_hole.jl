using LinearAlgebra
using Random
using Test
using HolographicControl

# Black-hole evaporation diagnostics. Every claim here is cross-checked
# against an analytic value or directly against `petz_recovery_error`.

@testset "page_time — reconstruction threshold" begin
    V5 = five_qubit_isometry()

    # [[5,1,3]] AME(5,2): any 3 qubits recover the bulk, no 2 do → page_time 3.
    @test page_time(V5) == 3

    # Cross-check against petz_recovery_error directly.
    # Keeping 3 qubits → recoverable (every 3-subset works for the AME code).
    for R in ([1,2,3], [3,4,5], [1,3,5], [2,4,5])
        @test petz_recovery_error(V5, R) < 1e-8
    end
    # Keeping 2 qubits → NOT recoverable (depolarizing on the bulk).
    for R in ([1,2], [4,5], [1,5])
        @test petz_recovery_error(V5, R) > 1e-2
    end

    # Repetition code: CLASSICAL. Although any 2 qubits pin the classical
    # logical *bit*, recovering the full quantum bulk (which carries the phase
    # |0>_L vs |1>_L) requires all 3 qubits — any 2-qubit region gives a
    # dephasing channel with Petz error 0.5, NOT recoverable. So its page_time
    # is 3 (only the trivial full-system recovery), equal to the [[5,1,3]].
    Vrep = three_qubit_repetition_isometry()
    @test petz_recovery_error(Vrep, [1, 2]) ≈ 0.5 atol=1e-10
    @test petz_recovery_error(Vrep, [1, 2, 3]) < 1e-8
    @test page_time(Vrep) == 3
    @test page_time(Vrep) == page_time(V5)
end

@testset "page_curve — [[5,1,3]] canonical symmetric tent" begin
    V5 = five_qubit_isometry()
    pc = page_curve(V5)

    @test pc.k == collect(0:5)
    @test pc.base == 2
    @test length(pc.S_R) == 6
    @test length(pc.reconstructable) == 6

    # S(R) at |R| = 0 is exactly 0 (no radiation).
    @test pc.S_R[1] ≈ 0.0 atol=1e-10
    # S(R) at |R| = 5 (whole system) is 0 — pure code state.
    @test pc.S_R[6] ≈ 0.0 atol=1e-10

    # AME(5,2): S(R) = |R| bits up to the half system, then falls.
    @test pc.S_R[2] ≈ 1.0 atol=1e-10   # |R|=1
    @test pc.S_R[3] ≈ 2.0 atol=1e-10   # |R|=2
    @test pc.S_R[4] ≈ 2.0 atol=1e-10   # |R|=3 (mirror of 2)
    @test pc.S_R[5] ≈ 1.0 atol=1e-10   # |R|=4 (mirror of 1)

    # Maximum sits at the half system (|R| = 2 or 3).
    @test maximum(pc.S_R) ≈ 2.0 atol=1e-10
    @test argmax(pc.S_R) in (3, 4)

    # SYMMETRIC tent for a pure code state: S(R=k) == S(R=5-k).
    for k in 0:5
        @test pc.S_R[k + 1] ≈ pc.S_R[(5 - k) + 1] atol=1e-10
    end

    # Curve rises then falls (strictly up to the peak, strictly down after).
    @test pc.S_R[1] < pc.S_R[2] < pc.S_R[3]
    @test pc.S_R[4] > pc.S_R[5] > pc.S_R[6]

    # Reconstructability flips on exactly at the page time and stays on.
    pt = page_time(V5)
    @test pt == 3
    @test all(.!pc.reconstructable[1:pt])      # not recoverable below page time
    @test all(pc.reconstructable[(pt + 1):end]) # recoverable from there on
end

@testset "page_curve — repetition code contrast" begin
    Vrep = three_qubit_repetition_isometry()
    pc = page_curve(Vrep)

    @test pc.k == collect(0:3)
    # Endpoints zero (pure GHZ-like codeword), symmetric.
    @test pc.S_R[1] ≈ 0.0 atol=1e-10
    @test pc.S_R[4] ≈ 0.0 atol=1e-10
    @test pc.S_R[2] ≈ pc.S_R[3] atol=1e-10        # symmetry S(k)=S(3-k)

    # Classical code: the representative codeword is the GHZ state
    # (|000⟩+|111⟩)/√2, whose every reduced state has rank 2 ⇒ S = 1 bit,
    # capped — it does NOT rise to 1.5 bits the way an AME would. This is the
    # qualitative classical/quantum contrast: a flat-top plateau at 1 bit.
    @test pc.S_R[2] ≈ 1.0 atol=1e-10
    @test pc.S_R[3] ≈ 1.0 atol=1e-10
    @test maximum(pc.S_R) ≈ 1.0 atol=1e-10

    # Contrast with [[5,1,3]]: the qualitative difference is in the PEAK
    # entropy — the perfect code's AME peak (2 bits) exceeds the classical
    # code's capped peak (1 bit). Both share page_time 3, but the [[5,1,3]]
    # reaches it via genuine sub-system recovery (any 3 qubits) while the
    # repetition code only ever recovers the quantum bulk from the full system.
    pc5 = page_curve(five_qubit_isometry())
    @test maximum(pc5.S_R) > maximum(pc.S_R)
    @test maximum(pc.S_R) ≈ 1.0 atol=1e-10
    @test maximum(pc5.S_R) ≈ 2.0 atol=1e-10
    # The [[5,1,3]] has a strict 3-qubit (proper-subset) recovery; the
    # repetition code's only recovering region is the full 3-qubit system.
    @test petz_recovery_error(five_qubit_isometry(), [1, 2, 3]) < 1e-8
    @test petz_recovery_error(Vrep, [1, 2]) > 1e-2  # no proper-subset recovery
end

@testset "page_curve — random isometry contrast" begin
    Random.seed!(7)
    Vr = random_isometry(32, 2)
    @test is_isometry(Vr)
    pc = page_curve(Vr)

    @test pc.k == collect(0:5)
    # Pure code state ⇒ endpoints 0 and the curve is symmetric S(k)=S(5-k),
    # even for a random (Haar-ish) isometry — purity is exact, not statistical.
    @test pc.S_R[1] ≈ 0.0 atol=1e-10
    @test pc.S_R[6] ≈ 0.0 atol=1e-10
    for k in 0:5
        @test pc.S_R[k + 1] ≈ pc.S_R[(5 - k) + 1] atol=1e-10
    end
    # Entropies are bounded by the number of bits in the smaller side.
    for k in 0:5
        @test pc.S_R[k + 1] ≤ min(k, 5 - k) + 1e-9
    end
    # A generic isometry typically recovers the (one) bulk qubit from a region
    # well before the full system — its page_time is finite and ≤ n.
    @test 0 ≤ page_time(Vr) ≤ 5
end

@testset "hayden_preskill_recoverable agrees with petz check" begin
    V5 = five_qubit_isometry()
    tol = 1e-8
    test_regions = [[1,2,3], [3,4,5], [1,3,5], [1,2], [4,5], [1,2,3,4], [1,2,3,4,5]]
    for R in test_regions
        @test hayden_preskill_recoverable(V5, R; tol=tol) ==
              (petz_recovery_error(V5, R) < tol)
    end

    # Spot-checks against the known AME structure.
    @test hayden_preskill_recoverable(V5, [1, 2, 3])    # 3 qubits → yes
    @test !hayden_preskill_recoverable(V5, [1, 2])       # 2 qubits → no

    # Repetition code: 2 qubits do NOT recover the quantum bulk (dephasing);
    # only the full 3-qubit system does.
    Vrep = three_qubit_repetition_isometry()
    @test !hayden_preskill_recoverable(Vrep, [1, 2])
    @test hayden_preskill_recoverable(Vrep, [1, 2, 3])
end

@testset "holographic_weighted_subregions — structure" begin
    n = 5
    A_list, w = holographic_weighted_subregions(n, 1; bias=1.0)

    # Length: every non-empty subregion of 1:n → 2^n - 1.
    @test length(A_list) == 2^n - 1
    @test length(w) == length(A_list)

    # Weights normalized and strictly positive.
    @test all(>(0), w)
    @test sum(w) ≈ 1.0 atol=1e-12

    # Larger regions get larger raw (and hence normalized) weight: for bias>0
    # the per-region weight is monotone in |A|.
    bias = 2.0
    A2, w2 = holographic_weighted_subregions(n, 1; bias=bias)
    sizes = [length(A) for A in A2]
    # Two regions, the bigger one has the bigger weight.
    for i in eachindex(A2), j in eachindex(A2)
        if sizes[i] < sizes[j]
            @test w2[i] < w2[j]
        elseif sizes[i] == sizes[j]
            @test w2[i] ≈ w2[j] atol=1e-12
        end
    end

    # bias=0 ⇒ flat weighting over all subregions.
    _, w0 = holographic_weighted_subregions(n, 1; bias=0.0)
    @test all(x -> x ≈ first(w0), w0)

    # The minimum-size floor (`weight`) excludes smaller regions.
    A3, _ = holographic_weighted_subregions(n, 3; bias=1.0)
    @test all(A -> length(A) ≥ 3, A3)
    @test minimum(length, A3) == 3
    @test maximum(length, A3) == n

    # Error handling.
    @test_throws ErrorException holographic_weighted_subregions(0, 1)
    @test_throws ErrorException holographic_weighted_subregions(3, 0)
    @test_throws ErrorException holographic_weighted_subregions(3, 4)
    @test_throws ErrorException holographic_weighted_subregions(3, 1; bias=-1.0)
end

@testset "holographic_weighted_subregions feeds petz_recovery_objective" begin
    V5 = five_qubit_isometry()

    # Bias toward regions of size ≥ 3 — all recoverable for the AME code, so
    # the holographically-weighted objective is ≈ 0.
    A_list, weights = holographic_weighted_subregions(5, 3; bias=1.0)
    obj = petz_recovery_objective(V5; A_list=A_list, weights=weights)
    @test obj ≥ 0
    @test obj < 1e-8

    # Sanity: the same objective on a random isometry is O(1) and larger.
    Random.seed!(11)
    Vr = random_isometry(32, 2)
    obj_rand = petz_recovery_objective(Vr; A_list=A_list, weights=weights)
    @test obj_rand > obj
end

@testset "page_curve consistency with reconstructable flag" begin
    # The `reconstructable` field must agree with page_time for several codes.
    for V in (five_qubit_isometry(), three_qubit_repetition_isometry())
        pc = page_curve(V)
        pt = page_time(V)
        # First true index (0-based k) equals the page time.
        first_true = findfirst(pc.reconstructable) - 1
        @test first_true == pt
        # Monotone: once recoverable, stays recoverable as more radiation
        # accumulates (a region of size k+1 contains a recoverable size-k one).
        flips = pc.reconstructable
        @test all(flips[findfirst(flips):end])
    end
end
