# 05b_petz_optimization_demo.jl
#
# M5 Layer 2 (no new deps): optimize an isometry V directly against the
# Petz recovery objective on the (n_bdy=3, n_bulk=1) system.
#
# This is a standalone demonstration BEFORE the eventual Piccolo
# integration: it bypasses the physical-Hamiltonian layer and just searches
# the isometry manifold abstractly. The point is to show that
#   * the M5 objective is a meaningful discovery target,
#   * a generic optimizer can find a code that beats every known/manual
#     baseline from 05a, and
#   * the codes found are interpretable (we report their per-erasure
#     errors and shape) so they can be cross-checked against the
#     literature.
#
# Method: random restart + local coordinate descent.
#   1. Generate K random complex 8×2 matrices, polar-project each to an
#      isometry, evaluate the Petz objective. Take the best.
#   2. From the best, do greedy coordinate descent: perturb each of the
#      32 real parameters by ±δ, accept the move that lowers the
#      objective most, halve δ on no-improvement. Repeat until δ is tiny.
#
# Usage:
#   julia --project=. examples/05b_petz_optimization_demo.jl
# Wall: ~1-2 min.

using LinearAlgebra
using Printf
using Random

using HolographicControl

n_bdy = 3
n_bulk = 1
d_bdy = 2^n_bdy
d_bulk = 2^n_bulk

A_list = [collect(setdiff(1:n_bdy, [q])) for q in 1:n_bdy]   # weight-1 erasures

# Parameterization: real vector x ∈ R^{2·d_bdy·d_bulk} → complex matrix M
# → isometry V = M (M' M)^{-1/2}. The polar projection always lands on the
# Stiefel manifold of isometries; we optimize unconstrained over the real
# vector and let the projection handle the constraint.
param_dim = 2 * d_bdy * d_bulk    # = 32

function isometry_from(x::Vector{Float64})
    n = d_bdy * d_bulk
    M_re = reshape(x[1:n], d_bdy, d_bulk)
    M_im = reshape(x[n+1:end], d_bdy, d_bulk)
    M = ComplexF64.(M_re, M_im)
    G = Hermitian(M' * M)
    return M * inv(sqrt(G))
end

function obj(x::Vector{Float64})
    V = isometry_from(x)
    return petz_recovery_objective(V; A_list=A_list)
end

println("=" ^ 72)
println("M5 Layer 2 — optimize V directly to minimize Petz recovery objective")
println("System: n_bdy=$n_bdy, n_bulk=$n_bulk, A_list=$A_list")
println("=" ^ 72)

function random_restart_best(K::Int, dim::Int; seed::Int=0)
    Random.seed!(seed)
    best_x = randn(dim)
    best_obj = obj(best_x)
    for _ in 2:K
        x = randn(dim)
        v = obj(x)
        if v < best_obj
            best_obj, best_x = v, x
        end
    end
    return best_x, best_obj
end

function coordinate_descent(x_init::Vector{Float64};
                            δ_init::Float64=0.5,
                            δ_floor::Float64=1e-7,
                            max_iter::Int=200,
                            patience::Int=30)
    x = copy(x_init)
    δ = δ_init
    iter = 0
    no_improve = 0
    while δ > δ_floor && iter < max_iter
        iter += 1
        cur_obj = obj(x)
        best_step_obj = cur_obj
        best_step = (0, 0.0)
        for i in 1:length(x), sign in (1, -1)
            x_trial = copy(x)
            x_trial[i] += sign * δ
            v = obj(x_trial)
            if v < best_step_obj
                best_step_obj = v
                best_step = (i, sign * δ)
            end
        end
        if best_step_obj < cur_obj - 1e-10
            x[best_step[1]] += best_step[2]
            no_improve = 0
            if iter % 10 == 0
                @printf("  iter %3d  obj = %.6f  δ = %.2e\n", iter, best_step_obj, δ)
            end
        else
            δ /= 2
            no_improve += 1
            if no_improve > patience
                break
            end
        end
    end
    return x, iter, δ
end

const K = 300
@printf("\nStep 1: %d random restarts\n", K)
best_x, best_obj = random_restart_best(K, param_dim; seed=0)
@printf("  Best of %d random isometries: objective = %.6f\n", K, best_obj)

@printf("\nStep 2: greedy coordinate descent from the best random restart\n")
x_final, iter, δ_final = coordinate_descent(best_x)
@printf("  After %d iters: objective = %.6f  (δ = %.2e)\n", iter, obj(x_final), δ_final)
x = x_final

# Final result
V_opt = isometry_from(x)
final_obj = petz_recovery_objective(V_opt; A_list=A_list)
println()
println("=" ^ 72)
println("Result")
println("=" ^ 72)
@printf("Final objective = %.6e\n", final_obj)
println("Per-erasure errors on V_opt:")
for A in A_list
    @printf("  keep %s : %.6e\n", A, petz_recovery_error(V_opt, A))
end

# Compare against Layer 1 baselines
println("\nFor reference (from 05a_petz_objective_demo.jl):")
println("  3-qubit repetition         : 0.500000")
println("  trivial embedding |q,0,0⟩  : 0.250000")
println("  random isometry (seed=42)  : 0.165135")
println("  |0⟩_L=|000⟩, |1⟩_L=|W⟩    : 0.271447")
println()
@printf("V_opt beats best baseline by factor %.2fx\n", 0.165135 / final_obj)

# Inspect what V_opt looks like — print column norms, overlaps
println("\nV_opt structure (column 1, then column 2):")
for j in 1:d_bulk
    @printf("  column %d (|%d⟩_L):  norm = %.6f\n", j, j-1, norm(V_opt[:, j]))
    # Largest 3 components by magnitude
    sorted = sortperm(abs.(V_opt[:, j]), rev=true)[1:3]
    for idx in sorted
        comp = V_opt[idx, j]
        @printf("     basis state |%s⟩ (idx %d) :  %+.4f %+.4fi  (|·|=%.4f)\n",
                bitstring(UInt8(idx-1))[6:8], idx, real(comp), imag(comp), abs(comp))
    end
end

# Column-1 / column-2 overlap (should be 0 for true isometry)
@printf("\n⟨V[:,1] | V[:,2]⟩ = %.6e + %.6ei   (should be ≈ 0)\n",
        real(dot(V_opt[:, 1], V_opt[:, 2])),
        imag(dot(V_opt[:, 1], V_opt[:, 2])))
@printf("‖V'V - I‖ = %.6e   (should be ≈ 0)\n", opnorm(V_opt' * V_opt - I))
