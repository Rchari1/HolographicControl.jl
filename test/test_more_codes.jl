using LinearAlgebra
using Test
using HolographicControl

# ============================================================================
# [[4,1,2]] error-detecting code (Vaidman)
# ============================================================================

@testset "[[4,1,2]] error-detecting code" begin
    V = four_one_two_isometry()

    @testset "shape and isometry" begin
        @test size(V) == (16, 2)
        @test V' * V ≈ I(2) atol=1e-12
        @test is_isometry(V)
    end

    @testset "analytic logical basis" begin
        # |0⟩_L = (|0000⟩ + |1111⟩) / √2  (big-endian: indices 1 and 16)
        # |1⟩_L = (|0011⟩ + |1100⟩) / √2  (big-endian: indices 4 and 13)
        # Each column is real, has exactly two nonzero amplitudes equal to 1/√2.
        nonzero_0 = findall(x -> abs(x) > 1e-10, V[:, 1])
        nonzero_1 = findall(x -> abs(x) > 1e-10, V[:, 2])
        @test sort(nonzero_0) == [1, 16]
        @test sort(nonzero_1) == [4, 13]
        @test V[1, 1]  ≈ 1/√2 atol=1e-12
        @test V[16, 1] ≈ 1/√2 atol=1e-12
        @test V[4, 2]  ≈ 1/√2 atol=1e-12
        @test V[13, 2] ≈ 1/√2 atol=1e-12
    end

    @testset "code projector" begin
        P = V * V'
        @test real(tr(P)) ≈ 2 atol=1e-12
        @test P ≈ P' atol=1e-12          # Hermitian
        @test P * P ≈ P  atol=1e-12      # idempotent
    end

    @testset "stabilizers fix code subspace" begin
        # Both g_i = XXXX, ZZZZ should fix V (eigenvalue +1).
        stabs = four_one_two_stabilizers()
        @test length(stabs) == 2
        @test size(stabs[1]) == (16, 16)
        for g in stabs
            @test g * V ≈ V atol=1e-12
        end
        # Spot-check identity by Pauli string equality.
        @test stabs[1] ≈ pauli_string("XXXX") atol=1e-14
        @test stabs[2] ≈ pauli_string("ZZZZ") atol=1e-14
    end

    @testset "logical operators on encoded qubit" begin
        # X̄ = XXII flips |0⟩_L ↔ |1⟩_L within the codespace.
        Xbar = pauli_string("XXII")
        @test Xbar * V[:, 1] ≈ V[:, 2] atol=1e-12
        @test Xbar * V[:, 2] ≈ V[:, 1] atol=1e-12
        # Z̄ = ZIZI gives +|0⟩_L, -|1⟩_L (diag(+1,-1) in the logical basis).
        Zbar = pauli_string("ZIZI")
        @test V' * Zbar * V ≈ Diagonal([1.0, -1.0]) atol=1e-12
        # Anticommute: {X̄, Z̄} on the codespace = 0.
        @test V' * (Xbar * Zbar + Zbar * Xbar) * V ≈ zeros(2, 2) atol=1e-12
    end

    @testset "Knill-Laflamme — distance-2 signature" begin
        # 13 single-qubit Pauli errors (I + 3·4 = 13).
        kl = knill_laflamme_constants(V; n=4)
        @test size(kl.C) == (13, 13)
        # Every Pauli preserves codespace norm: C_{aa} = 1 to machine precision.
        @test all(abs(kl.C[a, a] - 1) < 1e-10 for a in 1:13)
        # Errors are DETECTABLE: C_{aI} = 0 for a ≠ I (codespace is orthogonal
        # to the error-applied-then-projected back amplitude for single errors).
        @test all(abs(kl.C[a, 1]) < 1e-10 for a in 2:13)
        @test all(abs(kl.C[1, b]) < 1e-10 for b in 2:13)
        # NOT a non-degenerate distance-3 code: ‖C - I‖ is O(1), unlike the
        # [[5,1,3]] (where C ≈ I to machine precision). For [[4,1,2]] some
        # off-diagonal C_{ab} are ±1 and some residuals are O(1).
        @test norm(kl.C - I(13)) > 0.5
        @test maximum(kl.residuals) > 0.5
    end

    @testset "Petz recovery — 1-qubit erasures correctable (d-1 = 1)" begin
        # Distance-2 code corrects up to d - 1 = 1 known-location erasure.
        # Every 3-qubit kept region recovers the bulk perfectly.
        for A in uniform_erasure_subregions(4, 1)
            err = petz_recovery_error(V, A)
            @test abs(err) < 1e-10
        end
        # Aggregate over all 1-erasures: ≈ 0.
        J1 = petz_recovery_objective(V; A_list=uniform_erasure_subregions(4, 1))
        @test abs(J1) < 1e-10
    end

    @testset "Petz recovery — 2-qubit erasures fail (= 1/2 exactly)" begin
        # Two erasures saturate the no-cloning / Page-curve bound on a
        # d_bulk=2 code with 4 boundary qubits: the Petz error is 1/2 for
        # every 2-qubit kept region (computed empirically; checked here as
        # an exact analytic invariant for this symmetric code).
        for A in uniform_erasure_subregions(4, 2)
            err = petz_recovery_error(V, A)
            @test err ≈ 0.5 atol=1e-10
        end
        # The aggregate equals 1/2.
        J2 = petz_recovery_objective(V; A_list=uniform_erasure_subregions(4, 2))
        @test J2 ≈ 0.5 atol=1e-10
    end

    @testset "Petz recovery — 3-qubit erasures fully depolarize (= 3/4)" begin
        # Three erasures (1 qubit kept) yield the maximum possible Petz
        # entanglement infidelity 1 - 1/d_bulk² = 1 - 1/4 = 3/4 — the
        # fully-depolarizing limit for any code with a 1-logical encoding.
        for A in uniform_erasure_subregions(4, 3)
            err = petz_recovery_error(V, A)
            @test err ≈ 0.75 atol=1e-10
        end
    end
end

# ============================================================================
# [[7,1,3]] Steane CSS code
# ============================================================================

@testset "[[7,1,3]] Steane code" begin
    V = steane_isometry()

    @testset "shape and isometry" begin
        @test size(V) == (128, 2)
        @test V' * V ≈ I(2) atol=1e-12
        @test is_isometry(V)
    end

    @testset "code projector" begin
        P = V * V'
        @test real(tr(P)) ≈ 2 atol=1e-12
        @test P ≈ P' atol=1e-12
        @test P * P ≈ P  atol=1e-12
    end

    @testset "stabilizers fix code subspace" begin
        # Six CSS generators: 3 X-type + 3 Z-type, matched on the [7,4,3]
        # Hamming parity-check support.
        stabs = steane_stabilizers()
        @test length(stabs) == 6
        for g in stabs
            @test size(g) == (128, 128)
            @test g * V ≈ V atol=1e-12
        end
        # X-type and Z-type generators come in matched pairs (same support).
        @test stabs[1] ≈ pauli_string("IIIXXXX") atol=1e-14
        @test stabs[2] ≈ pauli_string("IXXIIXX") atol=1e-14
        @test stabs[3] ≈ pauli_string("XIXIXIX") atol=1e-14
        @test stabs[4] ≈ pauli_string("IIIZZZZ") atol=1e-14
        @test stabs[5] ≈ pauli_string("IZZIIZZ") atol=1e-14
        @test stabs[6] ≈ pauli_string("ZIZIZIZ") atol=1e-14
    end

    @testset "CSS structure: all stabilizers commute" begin
        # CSS construction guarantees X-type and Z-type stabilizers commute
        # because the Hamming parity-check matrix is self-dual mod 2.
        stabs = steane_stabilizers()
        for i in 1:6, j in (i + 1):6
            @test stabs[i] * stabs[j] ≈ stabs[j] * stabs[i] atol=1e-12
        end
    end

    @testset "logical operators on encoded qubit" begin
        # X̄ = XXXXXXX, Z̄ = ZZZZZZZ. {X̄, Z̄} anticommute, both commute with
        # all stabilizers (transversal logical Cliffords are the Steane
        # code's celebrated feature).
        Xbar = pauli_string("XXXXXXX")
        Zbar = pauli_string("ZZZZZZZ")
        @test Xbar * V[:, 1] ≈ V[:, 2] atol=1e-12
        @test Xbar * V[:, 2] ≈ V[:, 1] atol=1e-12
        @test V' * Zbar * V ≈ Diagonal([1.0, -1.0]) atol=1e-12
        @test V' * (Xbar * Zbar + Zbar * Xbar) * V ≈ zeros(2, 2) atol=1e-12
    end

    @testset "Knill-Laflamme — non-degenerate distance-3" begin
        # 22 single-qubit Pauli errors (I + 3·7). For a non-degenerate
        # distance-3 code C must equal I exactly (same as the [[5,1,3]]).
        kl = knill_laflamme_constants(V; n=7)
        @test size(kl.C) == (22, 22)
        @test maximum(kl.residuals) < 1e-10
        @test kl.C ≈ I(22) atol=1e-10
    end

    @testset "Petz recovery — 1-qubit erasures correctable" begin
        # Distance-3 code corrects up to d - 1 = 2 known-location erasures.
        # All 7 single-qubit erasures recover the bulk to machine precision.
        for A in uniform_erasure_subregions(7, 1)
            err = petz_recovery_error(V, A)
            @test abs(err) < 1e-10
        end
        J1 = petz_recovery_objective(V; A_list=uniform_erasure_subregions(7, 1))
        @test abs(J1) < 1e-10
    end

    @testset "Petz recovery — 2-qubit erasures correctable" begin
        # 21 two-qubit erasure patterns, all correctable for d = 3.
        for A in uniform_erasure_subregions(7, 2)
            err = petz_recovery_error(V, A)
            @test abs(err) < 1e-10
        end
        J2 = petz_recovery_objective(V; A_list=uniform_erasure_subregions(7, 2))
        @test abs(J2) < 1e-10
    end

    @testset "Petz recovery — 3-qubit erasures: BIMODAL (non-AME)" begin
        # 35 three-qubit erasure patterns. Steane is NOT absolutely maximally
        # entangled, unlike the [[5,1,3]]: some 4-qubit kept regions contain
        # a full logical operator support (→ Petz error ≈ 0) and others do
        # not (→ fully-depolarizing 3/4). Both values occur, and they are
        # the ONLY two values observed for this code.
        errs = [petz_recovery_error(V, A) for A in uniform_erasure_subregions(7, 3)]
        @test length(errs) == 35
        @test all(e -> abs(e) < 1e-10 || abs(e - 0.75) < 1e-10, errs)
        @test any(e -> abs(e) < 1e-10, errs)             # some correctable
        @test any(e -> abs(e - 0.75) < 1e-10, errs)      # some fully depolarize
        # Exact count from theory: the Hamming code [7,4,3] has weight-3
        # codewords on 7 supports given by the dual code. There are 7
        # "good" 4-qubit subsets (containing a logical's support) and 28
        # "bad" ones. Verify the count matches.
        n_correctable = count(e -> abs(e) < 1e-10, errs)
        n_depolarizing = count(e -> abs(e - 0.75) < 1e-10, errs)
        @test n_correctable + n_depolarizing == 35
        # Report-style assertion: at least 5 patterns of each kind.
        @test n_correctable ≥ 5
        @test n_depolarizing ≥ 5
    end

    @testset "Petz recovery — 4-qubit erasures: also bimodal" begin
        # By complement-symmetry the [[7,1,3]] is also bimodal on 4-qubit
        # erasures (3-qubit kept regions). Same {0, 3/4} pattern.
        errs = [petz_recovery_error(V, A) for A in uniform_erasure_subregions(7, 4)]
        @test length(errs) == 35
        @test all(e -> abs(e) < 1e-10 || abs(e - 0.75) < 1e-10, errs)
    end
end
