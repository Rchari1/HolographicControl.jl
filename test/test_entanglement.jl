using LinearAlgebra
using Test
using HolographicControl

# Analytic checks for the entanglement-structure / subregion-duality layer.
# The [[5,1,3]] code is the AME(5,2) state, so its pure-code-state marginals
# have a known, flat entanglement spectrum: S(A) = min(|A|, 5-|A|) bits, and
# every Rényi entropy equals the von Neumann entropy.

const V5 = five_qubit_isometry()
const VREP = three_qubit_repetition_isometry()

# All k-qubit subregions of 1:n.
all_regions(n, k) = uniform_erasure_subregions(n, n - k)

@testset "entanglement_entropy — AME(5,2) sharp values S(A)=min(|A|,5-|A|)" begin
    expected = Dict(1 => 1.0, 2 => 2.0, 3 => 2.0, 4 => 1.0, 5 => 0.0)
    for k in 1:5
        for A in all_regions(5, k)
            @test entanglement_entropy(V5, A) ≈ expected[k] atol = 1e-9
        end
    end
end

@testset "entanglement_entropy — base and units" begin
    # base=2 gives bits; base=ℯ gives nats. S([1]) = 1 bit = ln(2) nats.
    @test entanglement_entropy(V5, [1]; base = 2) ≈ 1.0 atol = 1e-9
    @test entanglement_entropy(V5, [1]; base = ℯ) ≈ log(2) atol = 1e-9
    # |A|=2 → 2 bits → 2 ln 2 nats
    @test entanglement_entropy(V5, [1, 2]; base = ℯ) ≈ 2 * log(2) atol = 1e-9
end

@testset "entanglement_entropy — purity: S(A) = S(complement)" begin
    # For a pure code state the entropy of a region equals that of its complement.
    for A in ([1], [2], [1, 2], [3, 4], [1, 2, 3], [1, 3, 5])
        Ac = collect(setdiff(1:5, A))
        @test entanglement_entropy(V5, A) ≈ entanglement_entropy(V5, Ac) atol = 1e-9
    end
    # Whole-system entropy of a pure state is 0.
    @test entanglement_entropy(V5, [1, 2, 3, 4, 5]) ≈ 0.0 atol = 1e-9
end

@testset "entanglement_entropy — :mixed vs :logical convention" begin
    # The maximally-mixed code state carries an extra log2(d_bulk)=1 bit of
    # bulk entropy when the region's wedge contains the bulk. For |A| with a
    # reconstructable wedge (k≥3) the mixed marginal entropy exceeds the pure one.
    @test entanglement_entropy(V5, [1, 2, 3]; code_state = :mixed) ≈ 3.0 atol = 1e-9
    @test entanglement_entropy(V5, [1, 2, 3]; code_state = :logical) ≈ 2.0 atol = 1e-9
    # Small regions (k < threshold) are maximally mixed either way.
    @test entanglement_entropy(V5, [1]; code_state = :mixed) ≈ 1.0 atol = 1e-9
    # Invalid symbol errors.
    @test_throws ErrorException entanglement_entropy(V5, [1]; code_state = :bogus)
end

@testset "entanglement_entropy — custom bulk_state" begin
    # |0>_L is a pure logical basis state; for AME its marginals are still flat.
    @test entanglement_entropy(V5, [1, 2]; bulk_state = ComplexF64[1, 0]) ≈ 2.0 atol = 1e-9
    @test entanglement_entropy(V5, [1]; bulk_state = ComplexF64[0, 1]) ≈ 1.0 atol = 1e-9
    @test_throws ErrorException entanglement_entropy(V5, [1]; bulk_state = ComplexF64[1, 0, 0])
end

@testset "renyi_entropy — flat spectrum: S_α = S for all α (AME)" begin
    expected = Dict(1 => 1.0, 2 => 2.0, 3 => 2.0, 4 => 1.0)
    for α in (0.0, 0.5, 2.0, 3.0, Inf)
        for k in 1:4
            for A in all_regions(5, k)
                @test renyi_entropy(V5, A, α) ≈ expected[k] atol = 1e-9
            end
        end
    end
end

@testset "renyi_entropy — α→1 limit equals von Neumann" begin
    for A in ([1], [1, 2], [1, 2, 3])
        @test renyi_entropy(V5, A, 1.0) ≈ entanglement_entropy(V5, A) atol = 1e-12
        # Continuity: α just off 1 should be very close.
        @test renyi_entropy(V5, A, 1.0000001) ≈ entanglement_entropy(V5, A) atol = 1e-5
    end
    @test_throws ErrorException renyi_entropy(V5, [1], -0.5)
end

@testset "mutual_information — non-negativity and AME monogamy" begin
    # AME(5,2): any two single boundary qubits are uncorrelated → I=0.
    for a in 1:5, b in (a+1):5
        I = mutual_information(V5, [a], [b])
        @test I ≈ 0.0 atol = 1e-9
        @test I ≥ -1e-9
    end
    # I(A:B) ≥ 0 (subadditivity) on assorted disjoint regions.
    for (A, B) in (([1], [2, 3]), ([1, 2], [3]), ([1], [2, 3, 4]), ([1, 2], [3, 4]))
        @test mutual_information(V5, A, B) ≥ -1e-9
    end
    # I(AB:C) = S(AB)+S(C)-S(ABC) = 2 + 1 - 1 = 2 bits for A=[1,2],B... use [1,2] vs [3]
    @test mutual_information(V5, [1, 2], [3]) ≈ 1.0 atol = 1e-9
    # disjointness is enforced
    @test_throws ErrorException mutual_information(V5, [1, 2], [2, 3])
end

@testset "subadditivity and strong subadditivity (random subregions)" begin
    # Subadditivity: S(A∪B) ≤ S(A) + S(B).
    for (A, B) in (([1], [2]), ([1, 2], [3]), ([1], [3, 4]), ([2, 3], [4, 5]))
        AB = sort(vcat(A, B))
        @test entanglement_entropy(V5, AB) ≤
              entanglement_entropy(V5, A) + entanglement_entropy(V5, B) + 1e-9
    end
    # Strong subadditivity: S(AB) + S(BC) ≥ S(B) + S(ABC).
    triples = (([1], [2], [3]), ([1], [2, 3], [4]), ([1, 2], [3], [4, 5]), ([1], [2], [3, 4, 5]))
    for (A, B, C) in triples
        AB = sort(vcat(A, B))
        BC = sort(vcat(B, C))
        ABC = sort(vcat(A, B, C))
        lhs = entanglement_entropy(V5, AB) + entanglement_entropy(V5, BC)
        rhs = entanglement_entropy(V5, B) + entanglement_entropy(V5, ABC)
        @test lhs ≥ rhs - 1e-9
    end
end

@testset "is_reconstructable — [[5,1,3]] erasure-correction structure" begin
    # k=1,2: never reconstructable. k=3,4,5: always reconstructable.
    for k in 1:5
        recon = k ≥ 3
        for A in all_regions(5, k)
            @test is_reconstructable(V5, A) == recon
            # Cross-check directly against the Petz machinery.
            err = petz_recovery_error(V5, A)
            @test (err < 1e-8) == recon
        end
    end
end

@testset "reconstruction_threshold" begin
    @test reconstruction_threshold(V5) == 3
    @test reconstruction_threshold(VREP) == 3
end

@testset "entanglement_wedge_report — [[5,1,3]] subregion duality" begin
    r = entanglement_wedge_report(V5)
    @test r.n_bdy == 5
    @test r.n_bulk == 1
    @test r.sizes == [1, 2, 3, 4, 5]
    @test r.total == [5, 10, 10, 5, 1]          # binomial(5, k)
    @test r.reconstructable == [0, 0, 10, 5, 1]  # all k≥3 regions reconstruct
    @test r.threshold == 3
end

@testset "repetition-code contrast (classical / GHZ logical state)" begin
    # |+>_L = (|000> + |111>)/√2 is a GHZ state. Single- and double-qubit
    # marginals are maximally mixed (rank 2) → S = 1 bit; full system → 0.
    @test entanglement_entropy(VREP, [1]) ≈ 1.0 atol = 1e-9
    @test entanglement_entropy(VREP, [2]) ≈ 1.0 atol = 1e-9
    @test entanglement_entropy(VREP, [1, 2]) ≈ 1.0 atol = 1e-9
    @test entanglement_entropy(VREP, [1, 2, 3]) ≈ 0.0 atol = 1e-9

    # Sharp CONTRAST with AME: two boundary qubits are CLASSICALLY correlated,
    # so I([1]:[2]) = 1 bit (GHZ), whereas the AME gives exactly 0.
    @test mutual_information(VREP, [1], [2]) ≈ 1.0 atol = 1e-9
    @test mutual_information(V5, [1], [2]) ≈ 0.0 atol = 1e-9

    # GHZ marginals also have a flat (2-level) spectrum, so Rényi = VN here.
    for α in (0.5, 2.0, 3.0)
        @test renyi_entropy(VREP, [1], α) ≈ 1.0 atol = 1e-9
    end

    # Wedge structure: as a quantum erasure code the repetition code is trivial
    # — erasing ANY single qubit of the GHZ logical leaves the bulk unrecoverable,
    # so only the full 3-qubit boundary reconstructs.
    rr = entanglement_wedge_report(VREP)
    @test rr.total == [3, 3, 1]
    @test rr.reconstructable == [0, 0, 1]
    @test rr.threshold == 3
end

@testset "random isometry — sanity (no analytic value)" begin
    # Entropies stay within [0, |A|] bits and MI stays non-negative for a
    # generic encoding; this guards against sign/normalization regressions.
    V = random_isometry(2^4, 2)   # 4 boundary qubits, 1 logical qubit
    for A in ([1], [1, 2], [2, 3])
        S = entanglement_entropy(V, A)
        @test 0 - 1e-9 ≤ S ≤ length(A) + 1e-9
    end
    @test mutual_information(V, [1], [2]) ≥ -1e-9
end
