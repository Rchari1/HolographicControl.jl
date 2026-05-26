# End-to-end Piccolo synthesis test on a small problem.
#
# Goal: verify the whole stack (problem builder → solve → extract →
# rolled-out verification → Petz scoring) works on a tiny synthesis task
# that runs in a few seconds. NOT a fidelity-quality test — for that, see
# examples/03_synthesize_repetition.jl and examples/04g.

using LinearAlgebra
using Test
using HolographicControl
using Piccolo: solve!

@testset "End-to-end pipeline: build → solve → extract → roll out" begin
    # Single-qubit X gate as an "isometry" (V = X) — the smallest possible
    # synthesis target. Avoids the multi-qubit Piccolo cost; this test
    # finishes in seconds.
    V_target = ComplexF64[0 1; 1 0]   # the X gate as a 2×2 isometry on n_bdy=1
    H_drift  = ComplexF64[0 0; 0 0]   # no drift
    H_drives = [pauli_matrix('X'), pauli_matrix('Y')]
    drive_bounds = [1.0, 1.0]

    qcp = isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T = 40, duration = 8.0, seed = 0,
    )

    # Cold-start L-BFGS only — fast, gets us most of the way
    solve!(qcp; max_iter = 80, eval_hessian = false)

    V_nlp = synthesized_isometry(qcp)
    V_rolled = rolled_out_isometry(qcp)

    # Both extractors return the right shape
    @test size(V_nlp) == (2, 2)
    @test size(V_rolled) == (2, 2)

    # Rollout columns must have unit norm (the ODE preserves it)
    for j in axes(V_rolled, 2)
        @test isapprox(norm(V_rolled[:, j]), 1.0; atol = 1e-6)
    end

    # Pipeline produces a meaningful improvement over random — fidelity
    # should be well above 0.5. We don't demand a tight bound here because
    # 80 L-BFGS iters at this problem size won't fully converge.
    fid_nlp = subspace_fidelity(V_target, V_nlp)
    fid_rol = subspace_fidelity(V_target, V_rolled)
    @test fid_nlp > 0.7
    @test fid_rol > 0.7
end
