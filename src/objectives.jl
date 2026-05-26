# Code-quality objectives — scalar functionals of an isometry `V` that
# summarize how good `V` is as a quantum error-correcting / approximate-QEC
# encoder. These are the *targets* for objective-driven synthesis
# (HANDOFF §M5: optimize over isometries directly).
#
# All objectives here take an isometry `V` and return a non-negative real
# number where smaller is better (0 = ideal). They are pure functions of
# `V` — no Piccolo dependency.

# --------------------------------------------------------------------------- #
# Petz recovery aggregate
# --------------------------------------------------------------------------- #

"""
    petz_recovery_objective(V; A_list, weights=nothing, cutoff=1e-10) -> Float64

Sum the Petz recovery error of `V` over the boundary subregions in
`A_list`, optionally weighted:

    J(V) = Σ_A w_A · petz_recovery_error(V, A).

This is the M5 discovery objective: replace a fixed target isometry with
this functional and let the optimizer find any `V` that minimizes total
recovery error across the chosen erasure patterns. Useful weighting
choices:
  * uniform over all weight-`w` erasures — drives the code toward
    distance ≥ `w + 1` if achievable
  * biased toward larger subregions — pushes for higher distance
  * single specific `A` — recovers that subregion's wedge only

Each entry of `A_list` is a list of kept-qubit indices under the
package's big-endian convention. `weights` is normalized internally
(`Σ w = 1`) for scale invariance unless you pass a specific weighting.

For the analytic [[5,1,3]] target and `A_list = all single-qubit erasures`,
this returns essentially 0 (≤ 1e-15 in practice — see
`examples/05c_petz_optimization_n5.jl`). For a random isometry it returns
`O(1)`.

See also [`uniform_erasure_subregions`](@ref) for a convenience constructor.
"""
function petz_recovery_objective(
    V::AbstractMatrix;
    A_list::AbstractVector{<:AbstractVector{Int}},
    weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
    cutoff::Float64 = 1e-10,
)
    isempty(A_list) && error("A_list must contain at least one subregion")
    w = if isnothing(weights)
        fill(1.0 / length(A_list), length(A_list))
    else
        length(weights) == length(A_list) ||
            error("weights length $(length(weights)) ≠ A_list length $(length(A_list))")
        any(<(0), weights) && error("weights must be non-negative")
        s = sum(weights)
        s > 0 || error("weights must sum to a positive value")
        weights ./ s
    end

    total = 0.0
    @inbounds for (i, A) in pairs(A_list)
        total += w[i] * petz_recovery_error(V, A; cutoff)
    end
    return total
end

# --------------------------------------------------------------------------- #
# Convenience constructors for A_list
# --------------------------------------------------------------------------- #

"""
    uniform_erasure_subregions(n_qubits, weight) -> Vector{Vector{Int}}

All `binomial(n, n-weight)` subregions of `1:n_qubits` obtained by erasing
exactly `weight` qubits. Useful as the `A_list` argument to
[`petz_recovery_objective`](@ref) when the goal is "correct any `weight`-
qubit erasure."

```julia-repl
julia> uniform_erasure_subregions(3, 1)   # erase 1 qubit, keep 2
3-element Vector{Vector{Int64}}:
 [2, 3]
 [1, 3]
 [1, 2]
```
"""
function uniform_erasure_subregions(n_qubits::Int, weight::Int)
    0 ≤ weight ≤ n_qubits ||
        error("weight $weight must be between 0 and n_qubits ($n_qubits)")
    if weight == 0
        return [collect(1:n_qubits)]
    end
    keep_size = n_qubits - weight
    subs = Vector{Vector{Int}}()
    # Enumerate combinations of size `keep_size` from `1:n_qubits`
    function recur!(start::Int, chosen::Vector{Int})
        if length(chosen) == keep_size
            push!(subs, copy(chosen))
            return
        end
        remaining = keep_size - length(chosen)
        for i in start:(n_qubits - remaining + 1)
            push!(chosen, i)
            recur!(i + 1, chosen)
            pop!(chosen)
        end
    end
    recur!(1, Int[])
    return subs
end

"""
    erasure_subregions_up_to(n_qubits, max_weight) -> Vector{Vector{Int}}

All subregions of `1:n_qubits` obtained by erasing 1 through `max_weight`
qubits (inclusive). Includes all weight-1, weight-2, ..., weight-`max_weight`
erasures.
"""
function erasure_subregions_up_to(n_qubits::Int, max_weight::Int)
    out = Vector{Vector{Int}}()
    for w in 1:max_weight
        append!(out, uniform_erasure_subregions(n_qubits, w))
    end
    return out
end
