using LinearAlgebra
using Random
using Test
using HolographicControl

# Analytic checks for the holographic-extensions layer (`src/holography.jl`).
# Every claim is cross-checked against an exact AdS/CFT-dictionary value
# (AME(5,2) saturation, pure-state purity, channel-level recovery), or
# against the underlying Petz/entanglement layer to guarantee consistency.

const V5     = five_qubit_isometry()
const VREP   = three_qubit_repetition_isometry()

# Helper: all k-qubit kept-subregions of 1:n.
all_regions(n, k) = uniform_erasure_subregions(n, n - k)

# --------------------------------------------------------------------------- #
# all_proper_subregions
# --------------------------------------------------------------------------- #

@testset "all_proper_subregions — non-empty proper subsets" begin
    # Definition: excludes BOTH the empty set and the full set, so 2^n - 2 entries.
    for n in 2:6
        L = all_proper_subregions(n)
        @test length(L) == 2^n - 2
        # Every entry is non-empty and strictly smaller than the full system.
        @test all(A -> 1 ≤ length(A) ≤ n - 1, L)
        # Every entry is a sorted subset of 1:n with unique elements.
        @test all(A -> A == sort(A), L)
        @test all(A -> allunique(A), L)
        @test all(A -> issubset(A, 1:n), L)
        # Uniqueness of the list.
        @test length(unique(L)) == length(L)
    end

    # n=3 exact contents (size-stratified order).
    L3 = all_proper_subregions(3)
    @test L3 == [[1], [2], [3], [1, 2], [1, 3], [2, 3]]

    # Size-stratified: sizes are non-decreasing along the list.
    L5 = all_proper_subregions(5)
    @test issorted([length(A) for A in L5])

    # Errors for n < 2 (no proper subregion exists).
    @test_throws ErrorException all_proper_subregions(1)
end

# --------------------------------------------------------------------------- #
# rt_minimal_surface_weights
# --------------------------------------------------------------------------- #

@testset "rt_minimal_surface_weights — bias=1: w_A ∝ |A|, sum=1" begin
    for n in 3:5
        w = rt_minimal_surface_weights(n; bias = 1.0)
        A_list = all_proper_subregions(n)
        @test length(w) == length(A_list)
        @test sum(w) ≈ 1.0 atol = 1e-12
        @test all(>=(0), w)
        # ratio between two regions of sizes k1, k2 is k1/k2.
        for i in eachindex(A_list), j in eachindex(A_list)
            k_i, k_j = length(A_list[i]), length(A_list[j])
            @test w[i] / w[j] ≈ k_i / k_j atol = 1e-12
        end
    end

    # n=3 exact weights: 3 singletons (|A|=1) + 3 pairs (|A|=2), total raw = 9.
    w3 = rt_minimal_surface_weights(3; bias = 1.0)
    @test w3 ≈ [1, 1, 1, 2, 2, 2] ./ 9 atol = 1e-12
end

@testset "rt_minimal_surface_weights — bias=2: w_A ∝ |A|²" begin
    # n=3 exact weights: 3·1² + 3·2² = 15 raw.
    w3 = rt_minimal_surface_weights(3; bias = 2.0)
    @test w3 ≈ [1, 1, 1, 4, 4, 4] ./ 15 atol = 1e-12

    # ratio between |A|=2 and |A|=1 should be 4 (=2²/1²).
    @test w3[4] / w3[1] ≈ 4.0 atol = 1e-12

    # n=4 ratio check between |A|=3 and |A|=1 should be 9.
    A4 = all_proper_subregions(4)
    w4 = rt_minimal_surface_weights(4; bias = 2.0)
    idx_1 = findfirst(A -> length(A) == 1, A4)
    idx_3 = findfirst(A -> length(A) == 3, A4)
    @test w4[idx_3] / w4[idx_1] ≈ 9.0 atol = 1e-12
end

@testset "rt_minimal_surface_weights — bias=0 recovers flat weighting" begin
    for n in 2:5
        w = rt_minimal_surface_weights(n; bias = 0.0)
        target = 1 / (2^n - 2)
        @test all(x -> isapprox(x, target; atol = 1e-12), w)
    end
end

@testset "rt_minimal_surface_weights — input validation" begin
    @test_throws ErrorException rt_minimal_surface_weights(1)
    @test_throws ErrorException rt_minimal_surface_weights(3; bias = -1.0)
end

@testset "rt_minimal_surface_weights — works with petz_recovery_objective" begin
    # On [[5,1,3]] every proper subregion of size ≥ 3 is recoverable, but the
    # 1- and 2-qubit ones contribute non-zero error. The objective is just a
    # weighted sum of those, so it equals zero only if we exclude the bad sizes.
    A_list = all_proper_subregions(5)
    w      = rt_minimal_surface_weights(5; bias = 1.0)
    J = petz_recovery_objective(V5; A_list = A_list, weights = w)
    @test J > 0                # at least one size-1 / size-2 region contributes
    @test J < 1.0              # bounded above (each contribution ≤ 1)
end

# --------------------------------------------------------------------------- #
# bulk_operator_reconstructable
# --------------------------------------------------------------------------- #

@testset "bulk_operator_reconstructable — [[5,1,3]] logical Z̄ on 3-qubit regions" begin
    Z_L = ComplexF64[1 0; 0 -1]   # logical Z basis matrix on the 2-dim bulk
    # Every 3-qubit region reconstructs the bulk algebra (AME(5,2)), so in
    # particular Z̄. We hit the full binomial(5,3) = 10 regions.
    for A in all_regions(5, 3)
        @test bulk_operator_reconstructable(V5, Z_L, A) == true
    end
    # 4- and 5-qubit regions trivially also reconstruct.
    for A in all_regions(5, 4)
        @test bulk_operator_reconstructable(V5, Z_L, A) == true
    end
    @test bulk_operator_reconstructable(V5, Z_L, [1, 2, 3, 4, 5]) == true
end

@testset "bulk_operator_reconstructable — [[5,1,3]] logical Z̄ NOT on small regions" begin
    Z_L = ComplexF64[1 0; 0 -1]
    # All 1- and 2-qubit regions FAIL — the bulk algebra is not in their wedge.
    for A in all_regions(5, 1)
        @test bulk_operator_reconstructable(V5, Z_L, A) == false
    end
    for A in all_regions(5, 2)
        @test bulk_operator_reconstructable(V5, Z_L, A) == false
    end
end

@testset "bulk_operator_reconstructable — identity always reconstructs" begin
    I2 = ComplexF64[1 0; 0 1]
    # The identity is in every algebra. Always reconstructable; the recovered
    # operator is V*I*V' = V V', which is preserved by the round-trip.
    for A in vcat(all_regions(5, 1), all_regions(5, 2), all_regions(5, 3))
        @test bulk_operator_reconstructable(V5, I2, A) == true
    end
end

@testset "bulk_operator_reconstructable — agrees with is_reconstructable" begin
    # If the WHOLE algebra is reconstructable from A, then every bulk operator
    # is. We test both X̄ and Z̄ on all regions of [[5,1,3]] and confirm the
    # per-operator answer matches `is_reconstructable`.
    X_L = ComplexF64[0 1; 1 0]
    Z_L = ComplexF64[1 0; 0 -1]
    for k in 1:5, A in all_regions(5, k)
        algebra_ok = is_reconstructable(V5, A)
        @test bulk_operator_reconstructable(V5, X_L, A) == algebra_ok
        @test bulk_operator_reconstructable(V5, Z_L, A) == algebra_ok
    end
end

@testset "bulk_operator_reconstructable — dimension-mismatch error" begin
    bad = ComplexF64[1 0 0; 0 1 0; 0 0 1]    # 3×3, but d_bulk = 2
    @test_throws DimensionMismatch bulk_operator_reconstructable(V5, bad, [1, 2, 3])
end

# --------------------------------------------------------------------------- #
# mutual_information_bulk_boundary
# --------------------------------------------------------------------------- #

@testset "mutual_information_bulk_boundary — [[5,1,3]] saturation" begin
    # When A's wedge contains the bulk, I(R:A) saturates at 2 log d_bulk.
    # For d_bulk = 2, that is 2 bits.
    for k in 3:5
        for A in all_regions(5, k)
            @test mutual_information_bulk_boundary(V5, A) ≈ 2.0 atol = 1e-9
        end
    end
end

@testset "mutual_information_bulk_boundary — [[5,1,3]] no reconstruction = 0" begin
    # 1- and 2-qubit regions of an AME state are maximally mixed and
    # uncorrelated with the bulk reference → I(R:A) = 0.
    for A in all_regions(5, 1)
        @test mutual_information_bulk_boundary(V5, A) ≈ 0.0 atol = 1e-9
    end
    for A in all_regions(5, 2)
        @test mutual_information_bulk_boundary(V5, A) ≈ 0.0 atol = 1e-9
    end
end

@testset "mutual_information_bulk_boundary — non-negativity and bounds" begin
    # I(R:A) ≥ 0 (subadditivity) and ≤ 2 log d_bulk (Hayden bound).
    Random.seed!(42)
    Vr = random_isometry(2^5, 2)
    for A in (([1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]))
        I = mutual_information_bulk_boundary(Vr, A)
        @test I ≥ -1e-9
        @test I ≤ 2.0 + 1e-9
    end
end

@testset "mutual_information_bulk_boundary — repetition (classical) contrast" begin
    # The 3-qubit repetition encodes a single logical qubit but the radiation
    # marginals look like a classical bit, so I(R:A) is capped at 1 bit (the
    # classical information) for sub-full regions; the full system gives 2.
    @test mutual_information_bulk_boundary(VREP, [1, 2, 3]) ≈ 2.0 atol = 1e-9
    # Single qubits and pairs: classical correlation only, < 2 bits.
    @test mutual_information_bulk_boundary(VREP, [1]) < 2.0 - 1e-3
    @test mutual_information_bulk_boundary(VREP, [1, 2]) < 2.0 - 1e-3
end

@testset "mutual_information_bulk_boundary — base / units" begin
    # base = 2 → 2 bits; base = ℯ → 2 ln 2 nats on saturation.
    A = [1, 2, 3]
    @test mutual_information_bulk_boundary(V5, A; base = 2)  ≈ 2.0      atol = 1e-9
    @test mutual_information_bulk_boundary(V5, A; base = ℯ) ≈ 2 * log(2) atol = 1e-9
end

# --------------------------------------------------------------------------- #
# subregion_complement_purity
# --------------------------------------------------------------------------- #

@testset "subregion_complement_purity — [[5,1,3]] S(A) = S(Ā) exactly" begin
    for A in ([1], [2], [1, 2], [3, 4], [1, 2, 3], [1, 3, 5])
        r = subregion_complement_purity(V5, A)
        @test r.A == collect(A)
        @test r.Ac == collect(setdiff(1:5, A))
        @test r.S_A ≈ r.S_Ac atol = 1e-10
        @test abs(r.difference) < 1e-10
    end
end

@testset "subregion_complement_purity — full system has S=0" begin
    # |A| = n_bdy: complement is empty, S(Ā) = 0; S(A) is the entropy of the
    # whole pure state = 0. So difference = 0.
    r = subregion_complement_purity(V5, [1, 2, 3, 4, 5])
    @test r.S_A ≈ 0.0 atol = 1e-10
    @test r.S_Ac ≈ 0.0 atol = 1e-10
end

@testset "subregion_complement_purity — repetition GHZ purity" begin
    # |+>_L of the 3-qubit repetition is GHZ; S([1]) = S([2,3]) = 1 bit.
    r = subregion_complement_purity(VREP, [1])
    @test r.S_A ≈ 1.0 atol = 1e-9
    @test r.S_Ac ≈ 1.0 atol = 1e-9
    @test abs(r.difference) < 1e-9
end

@testset "subregion_complement_purity — custom pure bulk_state" begin
    # |0>_L is a pure logical input; for the AME code its marginals are still flat.
    r = subregion_complement_purity(V5, [1, 2]; bulk_state = ComplexF64[1, 0])
    @test r.S_A ≈ r.S_Ac atol = 1e-9
end

# --------------------------------------------------------------------------- #
# code_quality_summary
# --------------------------------------------------------------------------- #

@testset "code_quality_summary — [[5,1,3]] field structure" begin
    s = code_quality_summary(V5)
    # Field structure
    @test s.n_bdy == 5
    @test s.n_bulk == 1
    @test s.page_time == 3
    @test s.reconstruction_threshold == 3
    # Petz on uniform weight-1 erasures is ≈ 0 — [[5,1,3]] corrects single
    # erasures perfectly.
    @test s.petz_uniform_w1 < 1e-8
    # Petz on uniform weight-2 erasures is ALSO ≈ 0 — distance 3 means *any*
    # 2-qubit erasure is correctable (any 3-qubit keep region recovers).
    @test !ismissing(s.petz_uniform_w2)
    @test s.petz_uniform_w2 < 1e-8
    # Entropy curve matches the AME tent: S(A) = min(k, 5-k).
    @test s.S_by_size ≈ [1.0, 2.0, 2.0, 1.0, 0.0] atol = 1e-9
    # Wedge report fields
    @test s.wedge_report.threshold == 3
    @test s.wedge_report.reconstructable == [0, 0, 10, 5, 1]
end

@testset "code_quality_summary — petz_custom branch" begin
    # When A_list is supplied, petz_custom is the uniform-weight Petz objective.
    A_list = all_proper_subregions(5)
    s = code_quality_summary(V5; A_list = A_list)
    @test !ismissing(s.petz_custom)
    # On [[5,1,3]] only the |A| = 1, 2 entries contribute, so the custom
    # objective is bounded.
    @test 0 ≤ s.petz_custom ≤ 1.0
    # If no A_list is given, petz_custom is missing.
    s_nil = code_quality_summary(V5)
    @test ismissing(s_nil.petz_custom)
end

@testset "code_quality_summary — repetition (n=3, no weight-2 entry)" begin
    s = code_quality_summary(VREP)
    @test s.n_bdy == 3
    @test s.n_bulk == 1
    # Weight-2 entry is missing because n_bdy < 4.
    @test ismissing(s.petz_uniform_w2)
    # Single-qubit erasures: each kept 2-qubit region gives a dephasing
    # channel of Petz error 0.5, averaged → 0.5.
    @test s.petz_uniform_w1 ≈ 0.5 atol = 1e-9
    # GHZ entropy curve: S(A) = min(k, 1) up to k = n-1, then 0.
    @test s.S_by_size ≈ [1.0, 1.0, 0.0] atol = 1e-9
    # Page time = 3, threshold = 3 (quantum recovery only at full system).
    @test s.page_time == 3
    @test s.reconstruction_threshold == 3
end

@testset "code_quality_summary — random isometry is finite and well-typed" begin
    Random.seed!(2025)
    Vr = random_isometry(2^5, 2)
    s = code_quality_summary(Vr)
    @test s.n_bdy == 5
    @test s.n_bulk == 1
    @test isfinite(s.petz_uniform_w1) && s.petz_uniform_w1 > 0
    @test !ismissing(s.petz_uniform_w2) && isfinite(s.petz_uniform_w2)
    @test 0 ≤ s.page_time ≤ 5
    @test 1 ≤ s.reconstruction_threshold ≤ 5
    @test length(s.S_by_size) == 5
    @test all(isfinite, s.S_by_size)
end

# --------------------------------------------------------------------------- #
# Cross-consistency between modules
# --------------------------------------------------------------------------- #

@testset "cross-check: I(R:A) saturation ↔ is_reconstructable" begin
    # For an AME / perfect code, A reconstructs ⇔ I(R:A) = 2 log d_bulk.
    for k in 1:5, A in all_regions(5, k)
        recon = is_reconstructable(V5, A)
        I_RA = mutual_information_bulk_boundary(V5, A)
        if recon
            @test I_RA ≈ 2.0 atol = 1e-9
        else
            @test I_RA < 2.0 - 1e-3
        end
    end
end

@testset "cross-check: reconstruction_threshold == page_time on V5" begin
    s = code_quality_summary(V5)
    # For an isometry these two are the same number — they both ask "smallest
    # k such that SOME k-qubit region reconstructs the bulk."
    @test s.page_time == s.reconstruction_threshold
end
